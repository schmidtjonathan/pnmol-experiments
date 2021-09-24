import operator
from abc import ABC, abstractmethod
from functools import partial

import jax
import jax.numpy as jnp


def pairwise(func, a, b):
    return jax.vmap(lambda x1: jax.vmap(lambda y1: func(x1, y1))(b))(a)


class Kernel(ABC):
    @abstractmethod
    def _evaluate(
        self,
        X: jnp.ndarray,
        X_: jnp.ndarray,
        return_gradient: bool,
    ):
        pass

    @property
    @abstractmethod
    def num_params(self):
        pass

    @property
    @abstractmethod
    def theta(self):
        pass

    @theta.setter
    @abstractmethod
    def theta(self, params):
        pass

    @partial(jax.jit, static_argnums=(0,))
    def __call__(self, X: jnp.ndarray, X_: jnp.ndarray):
        return self._evaluate(jnp.atleast_2d(X), jnp.atleast_2d(X_))


class LambdaKernel(Kernel):
    def __init__(self, fun):
        self._fun = jax.jit(fun)

    @property
    def num_params(self):
        return None

    @property
    def theta(self):
        return None

    @theta.setter
    def theta(self, params):
        raise NotImplementedError

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, X: jnp.ndarray, X_: jnp.ndarray):
        return jnp.array(
            [[self._fun(jnp.atleast_1d(x), jnp.atleast_1d(y)) for y in X_] for x in X]
        )

    def __str__(self):
        return f"LambdaKernel"


class SquareExponentialKernel(Kernel):
    def __init__(
        self,
        scale: float,
        lengthscale: float,
    ):
        self._scale = scale
        self._lengthscale = lengthscale
        self._theta = jnp.array([self._scale, self._lengthscale])

    @property
    def num_params(self):
        return 2

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, params):
        if not isinstance(params, jnp.ndarray) or params.size != self.num_params:
            raise ValueError()
        self._scale = params[0]
        self._lengthscale = params[1]
        self._theta = params

    @partial(jax.jit, static_argnums=(0,))
    def _evaluate(self, X: jnp.ndarray, X_: jnp.ndarray):
        if X.shape[-1] == X_.shape[-1] == 1:
            squared_distances = (
                jnp.reshape(X, (-1, 1)) - jnp.reshape(X_, (1, -1))
            ) ** 2
        else:
            diff = pairwise(operator.sub, X, X_)
            squared_distances = jnp.linalg.norm(diff, axis=-1) ** 2

        log_K = -squared_distances / (2.0 * self._lengthscale ** 2)
        K = jnp.exp(log_K)

        return self._scale ** 2 * K

    def __str__(self):
        return f"SquareExp(scale={self._scale}, lengthscale={self._lengthscale})"


class WienerKernel(Kernel):
    def __init__(
        self,
        scale: float,
    ):
        self._scale = scale
        self._theta = jnp.array([self._scale])

    @property
    def num_params(self):
        return 1

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, params):
        if not isinstance(params, jnp.ndarray) or params.size != self.num_params:
            raise ValueError()
        self._scale = params[0]
        self._theta = params

    def _evaluate(
        self,
        X: jnp.ndarray,
        X_: jnp.ndarray,
        return_gradient: bool,
    ):

        K = jnp.minimum(jnp.squeeze(X)[:, jnp.newaxis], jnp.squeeze(X_)[jnp.newaxis, :])
        if return_gradient:
            gradients = jnp.zeros_like(K, shape=list(K.shape)[:2] + [1])
            # d k / d scale
            gradients[..., 0] = 2.0 * self._scale * K
            return self._scale ** 2 * K, gradients

        return self._scale ** 2 * K

    def __str__(self):
        return f"Wiener(scale={self._scale})"


class OnceIntegratedBrownianMotionKernel(Kernel):
    def __init__(
        self,
        scale: float,
    ):
        self._scale = scale
        self._theta = jnp.array([self._scale])

    @property
    def num_params(self):
        return 1

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, params):
        if not isinstance(params, jnp.ndarray) or params.size != self.num_params:
            raise ValueError()
        self._scale = params[0]
        self._theta = params

    def _evaluate(self, X: jnp.ndarray, X_: jnp.ndarray):

        abs_distances = jnp.abs(X[:, jnp.newaxis] - X_[jnp.newaxis, :])

        mins = jnp.minimum(
            jnp.squeeze(X)[:, jnp.newaxis], jnp.squeeze(X_)[jnp.newaxis, :]
        )

        kern = (mins ** 3 / 3.0) + abs_distances * 0.5 * mins ** 2

        return self._scale ** 2 * kern

    def __str__(self):
        return f"IBM(scale={self._scale})"


class RationalQuadraticKernel(Kernel):
    def __init__(
        self,
        scale: float,
        lengthscale: float,
        scale_mixture: float,
    ):
        self._scale = scale
        self._lengthscale = lengthscale
        self._alpha = scale_mixture
        self._theta = jnp.array([self._scale, self._lengthscale, self._alpha])

    @property
    def num_params(self):
        return 3

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, params):
        if not isinstance(params, jnp.ndarray) or params.size != self.num_params:
            raise ValueError()
        self._scale = params[0]
        self._lengthscale = params[1]
        self._scale_mixture = params[2]
        self._theta = params

    def _evaluate(self, X: jnp.ndarray, X_: jnp.ndarray):
        sq_distances = (X[:, jnp.newaxis] - X_[jnp.newaxis, :]) ** 2

        K = 1.0 + (sq_distances / (2.0 * self._alpha * self._lengthscale ** 2))
        K_to_the_minus_alpha = K ** (-self._alpha)

        return self._scale ** 2 * K_to_the_minus_alpha

    def __str__(self):
        return f"RationalQuadratic(scale={self._scale}, lengthscale={self._lengthscale}, alpha={self._alpha})"


class OrnsteinUhlenbeckKernel(Kernel):
    def __init__(
        self,
        scale: float,
        lengthscale: float,
    ):
        self._scale = scale
        self._lengthscale = lengthscale
        self._theta = jnp.array([self._scale, self._lengthscale])

    @property
    def num_params(self):
        return 2

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, params):
        if not isinstance(params, jnp.ndarray) or params.size != self.num_params:
            raise ValueError()
        self._scale = params[0]
        self._lengthscale = params[1]
        self._theta = params

    def _evaluate(self, X: jnp.ndarray, X_: jnp.ndarray):
        abs_distances = jnp.abs(X[:, jnp.newaxis] - X_[jnp.newaxis, :])
        log_K = -abs_distances / self._lengthscale
        K = jnp.exp(log_K)

        return self._scale ** 2 * K

    def __str__(self):
        return f"OU(scale={self._scale}, lengthscale={self._lengthscale})"


class Matern3_2Kernel(Kernel):
    def __init__(
        self,
        scale: float,
        lengthscale: float,
    ):
        self._scale = scale
        self._lengthscale = lengthscale
        self._theta = jnp.array([self._scale, self._lengthscale])

    @property
    def num_params(self):
        return 2

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, params):
        if not isinstance(params, jnp.ndarray) or params.size != self.num_params:
            raise ValueError()
        self._scale = params[0]
        self._lengthscale = params[1]
        self._theta = params

    def _evaluate(self, X: jnp.ndarray, X_: jnp.ndarray):
        abs_distances = jnp.abs(X[:, jnp.newaxis] - X_[jnp.newaxis, :])

        s = jnp.sqrt(3.0) * abs_distances / self._lengthscale
        K_1 = 1.0 + s
        K_2 = jnp.exp(-s)
        K = K_1 * K_2

        return self._scale ** 2 * K

    def __str__(self):
        return f"Matern3/2(scale={self._scale}, lengthscale={self._lengthscale})"


class Matern5_2Kernel(Kernel):
    def __init__(
        self,
        scale: float,
        lengthscale: float,
    ):
        self._scale = scale
        self._lengthscale = lengthscale
        self._theta = jnp.array([self._scale, self._lengthscale])

    @property
    def num_params(self):
        return 2

    @property
    def theta(self):
        return self._theta

    @theta.setter
    def theta(self, params):
        if not isinstance(params, jnp.ndarray) or params.size != self.num_params:
            raise ValueError()
        self._scale = params[0]
        self._lengthscale = params[1]
        self._theta = params

    def _evaluate(self, X: jnp.ndarray, X_: jnp.ndarray):
        abs_distances = jnp.abs(X[:, jnp.newaxis] - X_[jnp.newaxis, :])

        s = jnp.sqrt(5.0) * abs_distances / self._lengthscale
        K_1 = 1.0 + s + (s ** 2 / 3.0)
        K_2 = jnp.exp(-s)
        K = K_1 * K_2

        return self._scale ** 2 * K

    def __str__(self):
        return f"Matern5/2(scale={self._scale}, lengthscale={self._lengthscale})"
