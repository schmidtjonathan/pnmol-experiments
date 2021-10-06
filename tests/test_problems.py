import jax
import jax.numpy as jnp
import pytest
import tornadox

import pnmol
from pnmol.pde import examples, problems

# Make the second dirichlet a neumann once the basic tests are established.
problems_1d_all = pytest.mark.parametrize(
    "prob1d",
    [
        examples.heat_1d_discretized(dx=0.1, bcond="dirichlet"),
        examples.heat_1d_discretized(dx=0.1, bcond="neumann"),
        examples.sir_1d_discretized(),
        examples.spruce_budworm_1d_discretized(bcond="dirichlet"),
        examples.spruce_budworm_1d_discretized(bcond="neumann"),
    ],
    ids=[
        "heat-dirichlet",
        "heat-neumann",
        "sir",
        "spruce-budworm-dirichlet",
        "spruce-budworm-neumann",
    ],
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
        assert isinstance(prob1d, problems.PDE)

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
        assert prob1d.y0.ndim == 1

    # Discretisations

    @staticmethod
    @problems_1d_all
    def test_L(prob1d, num_grid_points):
        L = prob1d.L
        assert L.shape[0] == L.shape[1]

    @staticmethod
    @problems_1d_all
    def test_E_sqrtm(prob1d, num_grid_points):
        E_sqrtm = prob1d.E_sqrtm
        assert E_sqrtm.shape[0] == E_sqrtm.shape[1]

    @staticmethod
    @problems_1d_all
    def test_B(prob1d, num_grid_points, num_boundary_points):
        B = prob1d.B
        assert B.shape[1] * num_boundary_points == B.shape[0] * num_grid_points

    @staticmethod
    @problems_1d_all
    def test_R_sqrtm(prob1d, num_boundary_points):
        R_sqrtm = prob1d.R_sqrtm
        assert R_sqrtm.shape[0] == R_sqrtm.shape[1]

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

    heat = examples.heat_1d_discretized(
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
    heat = examples.heat_1d_discretized(
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


def test_pde_system():
    pde1 = examples.heat_1d(bcond="neumann")
    pde2 = examples.heat_1d(bcond="neumann")
    diffop = (pde1.diffop, pde2.diffop)
    diffop_scale = (pde1.diffop_scale, pde2.diffop_scale)

    pde = problems.SystemLinearPDENeumann(
        diffop=diffop, diffop_scale=diffop_scale, bbox=pde1.bbox
    )

    assert pde.L is None
    assert pde.E_sqrtm is None

    mesh = pnmol.mesh.RectangularMesh.from_bbox_1d([0.0, 1.0], step=0.1)
    pde1.discretize(
        mesh_spatial=mesh, kernel=pnmol.kernels.SquareExponential(), stencil_size=3
    )
    pde2.discretize(
        mesh_spatial=mesh, kernel=pnmol.kernels.SquareExponential(), stencil_size=3
    )

    pde.discretize_system(
        mesh_spatial=mesh, kernel=pnmol.kernels.SquareExponential(), stencil_size=3
    )

    L_expected = jax.scipy.linalg.block_diag(pde1.L, pde2.L)
    E_sqrtm_expected = jax.scipy.linalg.block_diag(pde1.E_sqrtm, pde2.E_sqrtm)
    assert jnp.allclose(pde.L, L_expected)
    assert jnp.allclose(pde.E_sqrtm, E_sqrtm_expected)

    B_expected = jax.scipy.linalg.block_diag(pde1.B, pde2.B)
    R_sqrtm_expected = jax.scipy.linalg.block_diag(pde1.R_sqrtm, pde2.R_sqrtm)
    assert jnp.allclose(pde.B, B_expected)
    assert jnp.allclose(pde.R_sqrtm, R_sqrtm_expected)
