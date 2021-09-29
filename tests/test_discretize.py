"""Tests for discretisation."""

import jax.numpy as jnp
import pytest

from pnmol import diffops, discretize, kernels, mesh


@pytest.fixture
def bbox():
    return jnp.array([0.0, 1.0])


@pytest.fixture
def dx():
    return 0.1


@pytest.fixture
def mesh_spatial_1d(bbox, dx):
    return mesh.RectangularMesh.from_bounding_boxes_1d(bbox, dx)


@pytest.fixture
def diffop():
    return diffops.laplace()


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
        def fd_coeff(mesh_spatial_1d, k, L_k, LL_k, dx):
            x0 = mesh_spatial_1d[1]
            return discretize.fd_coeff(
                x=x0,
                neighbors=mesh_spatial_1d[((1, 0, 2),)],
                k=k,
                L_k=L_k,
                LL_k=LL_k,
                nugget_gram_matrix=0.0,
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
        def fd_coeff(mesh_spatial_1d, k, L_k, LL_k, dx):
            x0 = mesh_spatial_1d[0]
            return discretize.fd_coeff(
                x=x0,
                neighbors=mesh_spatial_1d[((0, 1, 2),)],
                k=k,
                L_k=L_k,
                LL_k=LL_k,
                nugget_gram_matrix=0.0,
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
        def fd_coeff(mesh_spatial_1d, k, L_k, LL_k, dx):
            x0 = mesh_spatial_1d[-1]
            return discretize.fd_coeff(
                x=x0,
                neighbors=mesh_spatial_1d[((-1, -2, -3),)],
                k=k,
                L_k=L_k,
                LL_k=LL_k,
                nugget_gram_matrix=0.0,
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
    def discretised(diffop, mesh_spatial_1d, k):
        return discretize.discretize(
            diffop=diffop,
            mesh_spatial=mesh_spatial_1d,
            kernel=k,
            stencil_size=3,
            nugget_gram_matrix=0.0,
        )

    @staticmethod
    def test_L_shape(discretised, mesh_spatial_1d):
        L, _ = discretised
        n = mesh_spatial_1d.shape[0]
        assert L.shape == (n, n)

    @staticmethod
    def test_E_sqrtm_shape(discretised, mesh_spatial_1d):
        _, E_sqrtm = discretised
        n = mesh_spatial_1d.shape[0]
        assert E_sqrtm.shape == (n, n)
