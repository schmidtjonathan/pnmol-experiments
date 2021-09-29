import jax.numpy as jnp
import pytest
import tornadox

import pnmol

# Make the second dirichlet a neumann once the basic tests are established.
problems_1d_all = pytest.mark.parametrize(
    "prob1d",
    [
        pnmol.problems.heat_1d(bcond="dirichlet"),
        pnmol.problems.heat_1d(bcond="neumann"),
    ],
    ids=["dirichlet", "neumann"],
)


@pytest.fixture
def kernel():
    return pnmol.kernels.SquareExponential()


@pytest.fixture
def mesh_spatial(prob1d):
    return pnmol.mesh.RectangularMesh.from_bounding_boxes_1d(prob1d.bbox, step=0.1)


class TestProb1dDiscretized:
    @staticmethod
    @pytest.fixture
    def prob1d_discretized(prob1d, mesh_spatial, kernel):
        prob1d.discretize(
            mesh_spatial=mesh_spatial,
            kernel=kernel,
            stencil_size=3,
            nugget_gram_matrix=1e-10,
            progressbar=False,
        )
        return prob1d

    @staticmethod
    @pytest.fixture
    def num_grid_points(prob1d_discretized):
        return prob1d_discretized.mesh_spatial.shape[0]

    @staticmethod
    @pytest.fixture
    def num_boundary_points(prob1d_discretized):
        return prob1d_discretized.mesh_spatial.boundary[0].shape[0]

    @staticmethod
    @problems_1d_all
    def test_type(prob1d_discretized):
        assert isinstance(prob1d_discretized, pnmol.problems.PDE)

    # IVP Functionality

    @staticmethod
    @problems_1d_all
    def test_t0(prob1d_discretized):
        assert jnp.isscalar(prob1d_discretized.t0)

    @staticmethod
    @problems_1d_all
    def test_tmax(prob1d_discretized):
        assert jnp.isscalar(prob1d_discretized.t0)

    @staticmethod
    @problems_1d_all
    def test_y0(prob1d_discretized, num_grid_points):
        assert prob1d_discretized.y0.shape == (num_grid_points,)

    # Discretisations

    @staticmethod
    @problems_1d_all
    def test_L(prob1d_discretized, num_grid_points):
        L = prob1d_discretized.L
        assert L.shape == (num_grid_points, num_grid_points)

    @staticmethod
    @problems_1d_all
    def test_E_sqrtm(prob1d_discretized, num_grid_points):
        E_sqrtm = prob1d_discretized.E_sqrtm
        assert E_sqrtm.shape == (num_grid_points, num_grid_points)

    @staticmethod
    @problems_1d_all
    def test_B(prob1d_discretized, num_grid_points, num_boundary_points):
        B = prob1d_discretized.B
        assert B.shape == (num_boundary_points, num_grid_points)

    @staticmethod
    @problems_1d_all
    def test_R_sqrtm(prob1d_discretized, num_boundary_points):
        R_sqrtm = prob1d_discretized.R_sqrtm
        assert R_sqrtm.shape == (num_boundary_points, num_boundary_points)

    @staticmethod
    @problems_1d_all
    def test_to_ivp(prob1d_discretized):
        ivp = prob1d_discretized.to_tornadox_ivp()
        assert isinstance(ivp, tornadox.ivp.InitialValueProblem)
