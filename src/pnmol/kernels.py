import jax
import jax.numpy as jnp

import pnmol


def _mesh_as_array(m):
    if isinstance(m, pnmol.mesh.Mesh):
        return m.points
    return jnp.asarray(m)


class Kernel:
    """
    Jax-based Kernel functions.

    Callables with two inputs plus some further information about partial derivatives.
    Callable input arguments are jitted automatically.
    """

    def __init__(self, fun, params=(), description=None):
        self._jitted_fun = jax.jit(fun)
        self._params = params
        self._description = (
            str(description) if description is not None else str(type(self).__name__)
        )

    def __repr__(self):
        return f"<{self._description} object>"

    def __call__(self, x, y, as_matrix=False):
        """
        Differentiable implementation of the kernel.
        Not vectorised (for now; for clarity reasons).
        """
        x = _mesh_as_array(x)
        y = _mesh_as_array(y)
        if as_matrix is True:
            x = x.reshape((-1, 1)) if x.ndim < 2 else x
            y = y.reshape((-1, 1)) if y.ndim < 2 else y
            return self._matrix(x_set=x, y_set=y)
        if as_matrix is False:
            errormsg = (
                f"Input shapes {x.shape} and {y.shape} not compatible with this kernel."
            )
            if x.ndim != y.ndim:
                raise TypeError(errormsg + " Did you mean as_matrix=True?")
            if x.ndim == y.ndim == 2:
                if x.shape[1] != y.shape[1]:
                    raise TypeError(errormsg)
                raise ValueError(errormsg + " Did you mean as_matrix=True?")

        return self._jitted_fun(x, y)

    def _matrix(self, x_set, y_set):
        """Build kernel matrix. Not vectorised at the moment for clarity reasons."""
        x_set = jnp.asarray(x_set)
        y_set = jnp.asarray(y_set)
        if not x_set.ndim == y_set.ndim == 2:  # should always pass!
            errormsg = f"Input shapes {x_set.shape} and {y_set.shape} not compatible with this kernel."
            raise TypeError(errormsg)

        return jnp.array([[self.__call__(x, y) for y in y_set] for x in x_set])

    def matrix_sqrt(self, x_set, y_set):
        """Square-root of the kernel matrix."""
        raise NotImplementedError


class PolynomialKernel(Kernel):
    """
    Polynomial kernels.


    Parameters
    coef : array_like
        Coefficients of the polynomial. If the polynomial is 1d, it has shape (order,).
        If the polynomial is 2d, it has shape (order_x, order_y), where coef[i, j] is
        the coefficient in front of x_1^i x_2^j.
        The order follows numpy.polyval, i.e. we lead with the highest order polynomial.
    """

    def __init__(self, coef):
        self.coef = jnp.asarray(coef)
        super().__init__(fun=self._evaluate)

    def _evaluate(self, x, y):
        """
        Differentiable implementation of the polynomial kernel.

        Not vectorised.
        """
        inner_prod = jnp.dot(x, y)
        return jnp.polyval(p=self.coef, x=inner_prod)

    def _matrix(self, x_set, y_set):
        ax, ay = self.matrix_sqrt(x_set=x_set, y_set=y_set)
        return jnp.dot(ax, ay.T)

    def matrix_sqrt(self, x_set, y_set):  # this is too unstable
        """Square-root of the kernel matrix.

        Matrix :math:`A=A(x)` such that :math:`K(x, y)=A(x)A(y)^\\intercal`
        """
        if x_set.ndim != y_set.ndim:
            raise ValueError("Point sets do not align.")
        if x_set.ndim == y_set.ndim == 1:
            x_set = x_set.reshape(-1)
            y_set = y_set.reshape(-1)
            return self._matrix_sqrt_1d(x_set=x_set, y_set=y_set)
        if x_set.ndim == y_set.ndim == 2 and x_set.shape[1] == y_set.shape[1] == 2:
            return self._matrix_sqrt_2d(x_set=x_set, y_set=y_set)
        raise NotImplementedError

    def _matrix_sqrt_1d(self, x_set, y_set):
        assert x_set.ndim == y_set.ndim == 1
        coefs = jnp.sqrt(jnp.flip(self.coef[None, :]))
        ax = coefs * x_set[:, None] ** jnp.arange(self.order)
        ay = coefs * y_set[:, None] ** jnp.arange(self.order)
        return ax, ay

    def _matrix_sqrt_2d(self, x_set, y_set):

        assert x_set.ndim == y_set.ndim == 2 and x_set.shape[1] == y_set.shape[1] == 2

        if self.coef.ndim != 2:
            raise ValueError("Coefficients for 2d polynomial need to be a matrix.")

        coefs = jnp.sqrt((self.coef))
        print(coefs ** 2)
        # coefs = jnp.sqrt(jnp.flip(self.coef))

        def single_row(pt1, pt2, d1=self.coef.shape[0], d2=self.coef.shape[1], c=coefs):
            """An entire row of the pseudo-Vandermonde matrix."""
            return (
                c * jnp.outer((pt1 ** jnp.arange(d1)), pt2 ** jnp.arange(d2))
            ).flatten()

        ax = jnp.array([single_row(x[0], x[1]) for x in x_set])
        ay = jnp.array([single_row(y[0], y[1]) for y in y_set])
        return ax, ay

    @property
    def order(self):
        if self.coef.ndim == 1:
            return len(self.coef)
        if self.coef.ndim == 2:
            return jnp.amax(self.coef.shape[0], self.coef.shape[1])


class GaussianKernel(Kernel):
    """Gaussian kernel."""

    def __init__(self, lengthscale):
        self.lengthscale = lengthscale
        super().__init__(
            fun=self._evaluate,
            description=f"GaussianKernel(lengthscale={self.lengthscale})",
        )

    def _evaluate(self, x, y):
        """Inputs are arrays of ndim=0 or ndim=1. Outputs are scalars"""
        x, y = jnp.asarray(x), jnp.asarray(y)
        return jnp.exp(-0.5 / self.lengthscale * jnp.dot(x - y, x - y))

    def _matrix(self, x_set, y_set):
        """Inputs are arrays of ndim=1 or ndim=2. Outputs are matrices"""
        x_set, y_set = jnp.asarray(x_set), jnp.asarray(y_set)
        if not x_set.ndim == y_set.ndim == 2:
            errormsg = f"Input shapes {x_set.shape} and {y_set.shape} not compatible with this kernel."
            raise TypeError(errormsg)
        x_broadcasted = jnp.expand_dims(x_set, axis=1)
        y_broadcasted = jnp.expand_dims(y_set, axis=0)
        diff = x_broadcasted - y_broadcasted  # x[:, None] - y[None, :]
        dist = jnp.linalg.norm(diff, axis=-1)
        return jnp.exp(-0.5 / self.lengthscale * dist ** 2)
