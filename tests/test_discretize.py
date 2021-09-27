"""Tests for discretisation."""

import jax.numpy as jnp
import pytest

from pnmol import differential_operator, discretize, kernels, mesh


@pytest.fixture
def bbox():
    return jnp.array([0.0, 1.0])


@pytest.fixture
def dx():
    return 0.1


@pytest.fixture
def grid_1d(bbox, dx):
    return mesh.RectangularMesh.from_bounding_boxes_1d(bbox, dx)


@pytest.fixture
def diffop():
    return differential_operator.laplace()


@pytest.fixture
def k():
    return kernels.Polynomial(const=1.0)


@pytest.fixture
def L_k(k, diffop):
    return kernels.Lambda(diffop(k.pairwise, argnums=0))


@pytest.fixture
def LL_k(L_k, diffop):
    return kernels.Lambda(diffop(L_k.pairwise, argnums=1))


class TestFDCoefficients:
    """Check coefficients for polynomial kernels against known FD coefficients.

    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """

    class TestCentral:
        @staticmethod
        @pytest.fixture
        def fd_coeff(grid_1d, k, L_k, LL_k, dx):
            x0 = grid_1d[1]
            return discretize.fd_coeff(
                x=x0,
                grid=grid_1d,
                stencil_size=3,
                k=k,
                L_k=L_k,
                LL_k=LL_k,
                cov_damping=0.0,
            )

        @staticmethod
        def test_weights(fd_coeff, dx):
            weights_normalized = fd_coeff[0] * dx ** 2
            assert jnp.allclose(weights_normalized, jnp.array([-2.0, 1.0, 1.0]))

        @staticmethod
        def test_uncertainty_zero(fd_coeff):
            uncertainty = fd_coeff[1]
            assert jnp.allclose(uncertainty, jnp.array(0.0))

    class TestForward:
        @staticmethod
        @pytest.fixture
        def fd_coeff(grid_1d, k, L_k, LL_k, dx):
            x0 = grid_1d[0]
            return discretize.fd_coeff(
                x=x0,
                grid=grid_1d,
                stencil_size=3,
                k=k,
                L_k=L_k,
                LL_k=LL_k,
                cov_damping=0.0,
            )

        @staticmethod
        def test_weights(fd_coeff, dx):
            weights_normalized = fd_coeff[0] * dx ** 2
            weights_expected = jnp.array([1.0, -2.0, 1.0])
            assert jnp.allclose(weights_normalized, weights_expected)

        @staticmethod
        def test_uncertainty_zero(fd_coeff):
            uncertainty = fd_coeff[1]
            assert jnp.allclose(uncertainty, jnp.array(0.0))

    class TestBackward:
        @staticmethod
        @pytest.fixture
        def fd_coeff(grid_1d, k, L_k, LL_k, dx):
            x0 = grid_1d[-1]
            return discretize.fd_coeff(
                x=x0,
                grid=grid_1d,
                stencil_size=3,
                k=k,
                L_k=L_k,
                LL_k=LL_k,
                cov_damping=0.0,
            )

        @staticmethod
        def test_weights(fd_coeff, dx):
            weights_computed = fd_coeff[0]
            weights_normalized = weights_computed * dx ** 2
            weights_expected = jnp.array([1.0, -2.0, 1.0])
            assert jnp.allclose(weights_normalized, weights_expected)

        @staticmethod
        def test_uncertainty_zero(fd_coeff):
            uncertainty = fd_coeff[1]
            assert jnp.allclose(uncertainty, jnp.array(0.0))


class TestDiscretise:
    @staticmethod
    @pytest.fixture
    def discretised(diffop, grid_1d, k):
        return discretize.discretize(
            diffop=diffop, mesh=grid_1d, kernel=k, stencil_size=3, cov_damping=0.0
        )

    @staticmethod
    def test_L_shape(discretised, grid_1d):
        L, _ = discretised
        n = grid_1d.shape[0]
        assert L.shape == (n, n)

    @staticmethod
    def test_E_sqrtm_shape(discretised, grid_1d):
        _, E_sqrtm = discretised
        n = grid_1d.shape[0]
        assert E_sqrtm.shape == (n, n)
