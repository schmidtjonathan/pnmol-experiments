r"""Differential operators acting on functions.

    Examples
    --------
    >>> import jax.numpy as jnp

    Make a test function

    >>> fun = lambda x: jnp.linalg.norm(x)**2
    >>> t0, x0 = 2., jnp.ones(2)

    The identity and arbitrary powers are differential operators

    >>> I = identity()
    >>> print(jnp.round(I(fun)(t0, x0), 2))
    2.0
    >>> P = power(3)
    >>> print(jnp.round(P(fun)(t0, x0), 2))
    8.0

    Laplace operators are differential operators

    >>> DD = laplace()
    >>> print(jnp.round(DD(fun)(t0, x0), 2))
    4.0

    >>> c = lambda x: 5.0*jnp.eye(x.shape[0])
    >>> DD2 = laplace(c)
    >>> print(jnp.round(DD2(fun)(t0, x0), 2))
    20.0

    Addition and multiplication is supported

    >>> op = I + P * DD2
    >>> print(jnp.round(op(fun)(t0, x0), 2))
    162.0

    Composition is simple to achieve

    >>> op2 = P.compose_with(DD2)
    >>> print(jnp.round(op2(fun)(t0, x0), 2))
    8000.0

    Construct the Cahn-Hilliard spatial differential operator

    .. math:: D f = \nabla^2(f^3 - f - \gamma \nabla^2 f)

    for :math:`\gamma=2`.

    >>> cahn_hilliard = laplace().compose_with(power(3) - identity() - scalar_mult(2.).compose_with(laplace()))
    >>> print(jnp.round(cahn_hilliard(fun)(t0, x0), 2))
    140.0

    Construct the Kardar-Parisi-Zhang spatial differential operator

    .. math:: D f = \nu \nabla^2 f + \lambda (\nabla f)^2 + \eta

    for :math:`\nu=2`, :math:`\lambda=3`, and :math:`\eta=4`.

    >>> kpz = scalar_mult(2.) * laplace() + scalar_mult(3.)*(gradient()@gradient()) + constant(4.)
    >>> print(jnp.round(kpz(fun)(t0, x0), 2))
    68.0


    The latter examples are inspired by
    ``https://py-pde.readthedocs.io/en/latest/packages/pde.pdes.html#``
"""


import typing

import jax
import jax.numpy as jnp


class DifferentialOperator:
    r"""Differential operator interfaces.

    Callables that map functions to functions and support basic arithmetic.

    Parameters
    ----------

    """

    def __init__(
        self, differentiate: typing.Callable[[typing.Callable], typing.Callable]
    ):
        self._differentiate = differentiate

    def __repr__(self):
        return "<DifferentialOperator object>"

    def __call__(self, fun: typing.Callable, argnums=0) -> typing.Callable:
        return self._differentiate(fun, argnums=argnums)

    def __add__(self, other: "DifferentialOperator") -> "DifferentialOperator":
        def sum_of_operators(fun, argnums=0):
            def evaluate_derivatives(*args):
                return self(fun, argnums=argnums)(*args) + other(fun, argnums=argnums)(
                    *args
                )

            return evaluate_derivatives

        return DifferentialOperator(differentiate=sum_of_operators)

    def __sub__(self, other: "DifferentialOperator") -> "DifferentialOperator":
        def diff_of_operators(fun, argnums=0):
            def evaluate_derivatives(*args):
                return self(fun, argnums=argnums)(*args) - other(fun, argnums=argnums)(
                    *args
                )

            return evaluate_derivatives

        return DifferentialOperator(differentiate=diff_of_operators)

    def __mul__(self, other: "DifferentialOperator") -> "DifferentialOperator":
        def prod_of_operators(fun, argnums=0):
            def evaluate_derivatives(*args):
                return self(fun, argnums=argnums)(*args) * other(fun, argnums=argnums)(
                    *args
                )

            return evaluate_derivatives

        return DifferentialOperator(differentiate=prod_of_operators)

    def __matmul__(self, other: "DifferentialOperator") -> "DifferentialOperator":
        def prod_of_operators(fun, argnums=0):
            def evaluate_derivatives(*args):
                A = self(fun, argnums=argnums)(*args)
                if A.ndim < 1:
                    A = A.reshape(-1, 1)
                B = other(fun, argnums=argnums)(*args)
                if B.ndim < 1:
                    B = B.reshape(1, -1)

                return A @ B

            return evaluate_derivatives

        return DifferentialOperator(differentiate=prod_of_operators)

    def compose_with(self, other: "DifferentialOperator") -> "DifferentialOperator":
        """Compose a differential operator with another differential operator."""

        def comp_of_operators(fun, argnums=0):
            def evaluate_derivatives(*args):

                return self(other(fun, argnums=argnums))(*args)

            return evaluate_derivatives

        return DifferentialOperator(differentiate=comp_of_operators)


def divergence():
    """Divergence of a fun, argnums=argnumsction as the trace of the Jacobian."""

    def my_div(fun, argnums=0):
        jac = jax.jacrev(fun, argnums=argnums)
        return lambda *args: jnp.trace(jac(*args))

    return DifferentialOperator(my_div)


def gradient():
    """Gradient of a function."""

    def my_grad(fun, argnums=0):
        _assure_scalar_fn = lambda *args, **kwargs: fun(*args, **kwargs).squeeze()
        return jax.grad(_assure_scalar_fn, argnums=argnums)

    return DifferentialOperator(my_grad)


def gradient_by_dimension(output_coordinate=0):
    """Gradient of a vector-valued function w.r.t. a single output-dimension."""

    def my_grad(fun, argnums=0):
        jac = jax.jacrev(fun, argnums=argnums)
        return lambda *args: jac(*args)[output_coordinate]

    return DifferentialOperator(my_grad)


def laplace():
    """Laplace operator of a function with an optional coefficient field."""

    def my_laplace(fun, argnums=0):

        div = divergence()
        grad = gradient()

        def inner(*args):
            return grad(fun, argnums=argnums)(*args)

        return div(inner)

    return DifferentialOperator(my_laplace)


def identity():
    """Identity operator as a differential operator."""

    def my_identity(fun, argnums=0):
        return fun

    return DifferentialOperator(my_identity)


def power(order):
    """Power of an operator as a differential operator."""

    def my_power(fun, argnums=0, p=order):
        def pw(*args):
            return fun(*args) ** p

        return pw

    return DifferentialOperator(my_power)


def scalar_mult(scalar):
    """Scalar multiplication as a differential operator."""

    def my_scalar_mult(fun, argnums=0, scalar=scalar):
        def pw(*args):
            return scalar * fun(*args)

        return pw

    return DifferentialOperator(my_scalar_mult)


def constant(scalar):
    """Constant functions as a differential operator."""

    def my_constant(fun, argnums=0, scalar=scalar):
        def pw(*args):
            return scalar

        return pw

    return DifferentialOperator(my_constant)
