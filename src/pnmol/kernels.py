from abc import ABC, abstractmethod
from functools import cached_property, partial

import jax
import jax.numpy as jnp


class Kernel(ABC):
    """Covariance kernel interface."""

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, X, Y):

        # Single element of the Gram matrix:
        # X.shape=(d,), Y.shape=(d,) -> K.shape = ()
        if X.ndim == Y.ndim <= 1:
            return self.pairwise(X, Y)

        # Diagonal of the Gram matrix:
        # X.shape=(N,d), Y.shape=(N,d) -> K.shape = (N,)
        if X.shape == Y.shape:
            return self._evaluate_inner(X, Y)

        # Full Gram matrix:
        # X.shape=[N,d), Y.shape=(d,K) -> K.shape = (N,K)
        return self._evaluate_outer(X, Y)

    @abstractmethod
    def pairwise(self, x, y):
        raise NotImplementedError

    @cached_property
    def _evaluate_inner(self):
        return jax.jit(jax.vmap(self.pairwise, (0, 0), 0))

    @cached_property
    def _evaluate_outer(self):
        _pairwise_row = jax.jit(jax.vmap(self.pairwise, (0, None), 0))
        return jax.jit(jax.vmap(_pairwise_row, (None, 1), 1))

    def __str__(self):
        return f"{self.__class__.__name__}()"

    def __add__(self, other):
        @jax.jit
        def pairwise_new(x, y):
            return self.pairwise(x, y) + other.pairwise(x, y)

        return Lambda(fun=pairwise_new)


class Lambda(Kernel):
    def __init__(self, fun):
        self._lambda_fun = jax.jit(fun)

    @partial(jax.jit, static_argnums=(0,))
    def pairwise(self, x, y):
        return self._lambda_fun(x, y)


class _RadialKernel(Kernel):
    r"""Radial kernels.

    k(x,y) = output_scale * \varphi(\|x-y\|*input_scale)
    """

    def __init__(
        self,
        *,
        output_scale=1.0,
        input_scale=1.0,
    ):
        self._output_scale = output_scale
        self._input_scale = input_scale

    @property
    def output_scale(self):
        return self._output_scale

    @property
    def output_scale_squared(self):
        return self.output_scale ** 2

    @property
    def input_scale(self):
        return self._input_scale

    @property
    def input_scale_squared(self):
        return self.input_scale ** 2

    @abstractmethod
    def pairwise(self, X, Y):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=0)
    def _distance_squared_l2(self, X, Y):
        return (X - Y).dot(X - Y)


class SquareExponential(_RadialKernel):
    @partial(jax.jit, static_argnums=0)
    def pairwise(self, x, y):
        dist_squared = self._distance_squared_l2(x, y) * self.input_scale_squared
        return self.output_scale_squared * jnp.exp(-dist_squared / 2.0)


class Matern52(_RadialKernel):

    # Careful! Matern52 is not differentiable at x=y!
    # Therefore, it is likely unusable for PNMOL...
    @partial(jax.jit, static_argnums=(0,))
    def pairwise(self, x, y):
        dist_unscaled = self._distance_squared_l2(x, y)
        dist_scaled = jnp.sqrt(5.0 * dist_unscaled * self.input_scale_squared)
        A = 1 + dist_scaled + dist_scaled ** 2.0 / 3.0
        B = jnp.exp(-dist_scaled)
        return self.output_scale_squared * A * B


class Polynomial(Kernel):
    """k(x,y) = (x.T @ y + c)^d"""

    def __init__(self, *, order=2, const=1.0):
        self._order = order
        self._const = const

    @property
    def order(self):
        return self._order

    @property
    def const(self):
        return self._const

    @partial(jax.jit, static_argnums=(0,))
    def pairwise(self, x, y):
        return (x.dot(y) + self.const) ** self.order


class WhiteNoise(Kernel):
    def __init__(self, output_scale=1.0):
        self._output_scale = output_scale

    @property
    def output_scale(self):
        return self._output_scale

    @partial(jax.jit, static_argnums=(0,))
    def pairwise(self, x, y):
        return self.output_scale ** 2 * jnp.all(x == y)
