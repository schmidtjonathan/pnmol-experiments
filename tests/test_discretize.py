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
    return kernels.Polynomial()


@pytest.fixture
def L_k(k, diffop):
    return kernels.Lambda(diffop(k.pairwise, argnums=0))


@pytest.fixture
def LL_k(L_k, diffop):
    return kernels.Lambda(diffop(L_k.pairwise, argnums=1))


@pytest.fixture
def x0(grid_1d):
    return grid_1d[0]


class TestFDCoefficients:
    @staticmethod
    @pytest.fixture
    def fd_coeff(x0, grid_1d, k, L_k, LL_k, dx):
        return discretize.fd_coeff(
            x=x0, grid=grid_1d, stencil_size=3, k=k, L_k=L_k, LL_k=LL_k, cov_damping=0.0
        )

    @staticmethod
    def test_polynomial_weights_1_2_1(fd_coeff, dx):
        weights_normalized = fd_coeff[0] * dx ** 2
        assert jnp.allclose(weights_normalized, jnp.array([1.0, -2.0, 1.0]))

    @staticmethod
    def test_polynomial_uncertainty_zero(fd_coeff):

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
