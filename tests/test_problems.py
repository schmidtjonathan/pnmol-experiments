import jax.numpy as jnp
import pytest
import tornadox

import pnmol

# Make the second dirichlet a neumann once the basic tests are established.
problems_1d_all = pytest.mark.parametrize(
    "prob1d",
    [
        pnmol.problems.heat_1d_discretized(dx=0.1, bcond="dirichlet"),
        pnmol.problems.heat_1d_discretized(dx=0.1, bcond="neumann"),
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


def test_to_ivp():
    """Does the transformation work correctly?"""

    bcond = "neumann"
    dx = 0.2
    diffusion_rate = 0.01

    heat = pnmol.problems.heat_1d_discretized(
        dx=dx,
        bcond=bcond,
        kernel=pnmol.kernels.Polynomial(),
        diffusion_rate=diffusion_rate,
    )

    heat_as_ivp = heat.to_tornadox_ivp()

    # The IVP stuff is copied correctly
    assert jnp.allclose(heat.y0[1:-1], heat_as_ivp.y0)
    assert jnp.allclose(heat.t0, heat_as_ivp.t0)
    assert jnp.allclose(heat.tmax, heat_as_ivp.tmax)

    # The Jacobian is constant and as expected (depending on the BCs)
    dfy0a = heat_as_ivp.df(heat_as_ivp.t0, heat_as_ivp.y0)
    dfy0b = heat_as_ivp.df(heat_as_ivp.t0, heat_as_ivp.y0 + 1.0)
    assert jnp.allclose(dfy0a, dfy0b)
    if bcond == "neumann":
        assert jnp.allclose(
            dfy0a[0, :2] * dx ** 2 / diffusion_rate, jnp.array([-1.0, 1.0])
        )
    if bcond == "dirichlet":
        assert jnp.allclose(dfy0a, heat.L[1:-1, 1:-1])

    bcond = "dirichlet"
    heat = pnmol.problems.heat_1d_discretized(
        dx=dx,
        bcond=bcond,
        kernel=pnmol.kernels.Polynomial(),
        diffusion_rate=diffusion_rate,
    )

    heat_as_ivp = heat.to_tornadox_ivp()

    # The IVP stuff is copied correctly
    assert jnp.allclose(heat.y0[1:-1], heat_as_ivp.y0)
    assert jnp.allclose(heat.t0, heat_as_ivp.t0)
    assert jnp.allclose(heat.tmax, heat_as_ivp.tmax)

    # The Jacobian is constant and as expected (depending on the BCs)
    dfy0a = heat_as_ivp.df(heat_as_ivp.t0, heat_as_ivp.y0)
    dfy0b = heat_as_ivp.df(heat_as_ivp.t0, heat_as_ivp.y0 + 1.0)
    assert jnp.allclose(dfy0a, dfy0b)
    if bcond == "neumann":
        assert jnp.allclose(
            dfy0a[0, :2] * dx ** 2 / diffusion_rate, jnp.array([-1.0, 1.0])
        )
    if bcond == "dirichlet":
        assert jnp.allclose(dfy0a, heat.L[1:-1, 1:-1])

    # The vector field is indeed linear
    fy0 = heat_as_ivp.f(heat_as_ivp.t0, heat_as_ivp.y0)
    assert jnp.allclose(dfy0a @ heat_as_ivp.y0, fy0)
