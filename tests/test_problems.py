import jax.numpy as jnp
import pytest
import tornadox

import pnmol

# Make the second dirichlet a neumann once the basic tests are established.
problems_1d_all = pytest.mark.parametrize(
    "prob1d",
    [
        pnmol.problems.heat_1d_discretized(bcond="dirichlet"),
        pnmol.problems.heat_1d_discretized(bcond="neumann"),
    ],
    ids=["dirichlet", "neumann"],
)


class TestProb1dDiscretized:
    @staticmethod
    @pytest.fixture
    def num_grid_points(prob1d):
        return prob1d.mesh_spatial.shape[0]

    @staticmethod
    @pytest.fixture
    def num_boundary_points(prob1d):
        return prob1d.mesh_spatial.boundary[0].shape[0]

    @staticmethod
    @problems_1d_all
    def test_type(prob1d):
        assert isinstance(prob1d, pnmol.problems.PDE)

    # IVP Functionality

    @staticmethod
    @problems_1d_all
    def test_t0(prob1d):
        assert jnp.isscalar(prob1d.t0)

    @staticmethod
    @problems_1d_all
    def test_tmax(prob1d):
        assert jnp.isscalar(prob1d.t0)

    @staticmethod
    @problems_1d_all
    def test_y0(prob1d, num_grid_points):
        assert prob1d.y0.shape == (num_grid_points,)

    # Discretisations

    @staticmethod
    @problems_1d_all
    def test_L(prob1d, num_grid_points):
        L = prob1d.L
        assert L.shape == (num_grid_points, num_grid_points)

    @staticmethod
    @problems_1d_all
    def test_E_sqrtm(prob1d, num_grid_points):
        E_sqrtm = prob1d.E_sqrtm
        assert E_sqrtm.shape == (num_grid_points, num_grid_points)

    @staticmethod
    @problems_1d_all
    def test_B(prob1d, num_grid_points, num_boundary_points):
        B = prob1d.B
        assert B.shape == (num_boundary_points, num_grid_points)

    @staticmethod
    @problems_1d_all
    def test_R_sqrtm(prob1d, num_boundary_points):
        R_sqrtm = prob1d.R_sqrtm
        assert R_sqrtm.shape == (num_boundary_points, num_boundary_points)

    @staticmethod
    @problems_1d_all
    def test_to_ivp(prob1d):
        ivp = prob1d.to_tornadox_ivp()
        assert isinstance(ivp, tornadox.ivp.InitialValueProblem)

        f0 = ivp.f(ivp.t0, ivp.y0)
        assert f0.shape == ivp.y0.shape

        df0 = ivp.df(ivp.t0, ivp.y0)
        assert df0.shape == (ivp.y0.shape[0], ivp.y0.shape[0])
