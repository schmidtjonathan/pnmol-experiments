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
    return mesh.RectangularMesh.from_bbox_1d(bbox, dx)


@pytest.fixture
def diffop():
    return diffops.laplace()


class TestFDCoefficients:
    """Check coefficients for polynomial kernels against known FD coefficients.

    https://en.wikipedia.org/wiki/Finite_difference_coefficient
    """

    @staticmethod
    @pytest.fixture
    def k():
        return kernels.Polynomial(const=1.0)

    @staticmethod
    @pytest.fixture
    def L_k(k, diffop):
        return kernels.Lambda(diffop(k.pairwise, argnums=0))

    @staticmethod
    @pytest.fixture
    def LL_k(L_k, diffop):
        return kernels.Lambda(diffop(L_k.pairwise, argnums=1))

    @staticmethod
    @pytest.fixture
    def fd_coeff(mesh_spatial_1d, k, L_k, LL_k, dx):
        x0 = mesh_spatial_1d[1]
        return discretize.fd_coefficients(
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


class TestFDProbabilistic:
    @staticmethod
    @pytest.fixture
    def fd_approximation(diffop, mesh_spatial_1d):
        return discretize.fd_probabilistic(
            diffop=diffop,
            mesh_spatial=mesh_spatial_1d,
            stencil_size=3,
            nugget_gram_matrix=0.0,
        )

    @staticmethod
    def test_L_shape(fd_approximation, mesh_spatial_1d):
        L, _ = fd_approximation
        n = mesh_spatial_1d.shape[0]
        assert L.shape == (n, n)

    @staticmethod
    def test_E_sqrtm_shape(fd_approximation, mesh_spatial_1d):
        _, E_sqrtm = fd_approximation
        n = mesh_spatial_1d.shape[0]
        assert E_sqrtm.shape == (n, n)


class TestCollocationGlobal:
    @staticmethod
    @pytest.fixture
    def collocation_global(diffop, mesh_spatial_1d):
        return discretize.collocation_global(
            diffop=diffop,
            mesh_spatial=mesh_spatial_1d,
            nugget_gram_matrix=1e-10,
        )

    @staticmethod
    def test_L_shape(collocation_global, mesh_spatial_1d):
        L, _ = collocation_global
        n = mesh_spatial_1d.shape[0]
        assert L.shape == (n, n)

    @staticmethod
    def test_E_sqrtm_shape(collocation_global, mesh_spatial_1d):
        _, E_sqrtm = collocation_global
        n = mesh_spatial_1d.shape[0]
        assert E_sqrtm.shape == (n, n)


class TestNeumann:
    @staticmethod
    @pytest.fixture
    def fd_probabilistic_neumann(mesh_spatial_1d):
        return discretize.fd_probabilistic_neumann_1d(mesh_spatial_1d)

    @staticmethod
    def test_shape_L(fd_probabilistic_neumann, mesh_spatial_1d):
        L, _ = fd_probabilistic_neumann
        n = mesh_spatial_1d.shape[0]
        assert L.shape == (2, n)

    @staticmethod
    def test_shape_E_sqrtm(fd_probabilistic_neumann):
        _, E_sqrtm = fd_probabilistic_neumann
        assert E_sqrtm.shape == (2, 2)
