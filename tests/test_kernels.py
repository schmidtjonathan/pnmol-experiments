import jax
import jax.numpy as jnp
import pytest

from pnmol import differential_operator, kernels

K1a = kernels.SquareExponential()
K1b = kernels.SquareExponential(input_scale=0.1, output_scale=2.0)
K2 = kernels.Lambda(fun=lambda x, y: (x - y).dot(x - y))
K3a = kernels.Matern52()
K3b = kernels.Matern52(input_scale=0.1, output_scale=2.0)
K4a = kernels.Polynomial()
K4b = kernels.Polynomial(order=4, const=1.2345)
K5a = kernels.WhiteNoise()
K5b = kernels.WhiteNoise(output_scale=1.234)

# Decorator to use for the tests.
ALL_KERNELS = pytest.mark.parametrize(
    "kernel", [K1a, K1b, K2, K3a, K3b, K4a, K4b, K5a, K5b]
)
ALL_KERNELS_BUT_MATERN = pytest.mark.parametrize(
    "kernel", [K1a, K1b, K2, K4a, K4b, K5a, K5b]
)


@pytest.fixture
def num_data_x():
    return 1


@pytest.fixture
def num_data_y():
    return 2


@pytest.fixture
def dim():
    return 2


@pytest.fixture
def X(num_data_x, dim):
    return 1.0 * jnp.arange(num_data_x * dim).reshape((num_data_x, dim))


@pytest.fixture
def Y(num_data_y, dim):
    return 0.5 * jnp.arange(2, 2 + num_data_y * dim).reshape((num_data_y, dim))


class TestEvaluateShape:
    @staticmethod
    @ALL_KERNELS
    def test_outer(kernel, X, Y):
        Kxy = kernel(X, Y.T)
        assert Kxy.shape == (X.shape[0], Y.shape[0])

    @staticmethod
    @ALL_KERNELS
    def test_inner(kernel, X):
        Kxy = kernel(X, X)
        assert Kxy.shape == (X.shape[0],)

    @staticmethod
    @ALL_KERNELS
    def test_single_element(kernel, X, Y):
        Kxy = kernel(X[0], Y[0])
        assert Kxy.shape == ()

    @staticmethod
    @ALL_KERNELS
    def test_cov(kernel, X):
        Kxy = kernel(X, X.T)
        assert Kxy.shape == (X.shape[0], X.shape[0])

    @staticmethod
    @ALL_KERNELS
    def test_scalar(kernel, X):
        x, y = jnp.array(2.0), jnp.array(3.0)
        Kxy = kernel(x, y)
        assert Kxy.shape == ()


def test_white_noise_diagonal(X, Y):
    k = kernels.WhiteNoise()
    Kxy = k(X, Y.T)
    assert jnp.allclose(Kxy, jnp.diag(jnp.diag(Kxy)))
