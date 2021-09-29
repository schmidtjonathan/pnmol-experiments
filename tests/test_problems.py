import jax.numpy as jnp
import pytest

import pnmol

# Make the second dirichlet a neumann once the basic tests are established.
problems_1d_all = pytest.mark.parametrize(
    "prob1d",
    [
        pnmol.problems.heat_1d(bcond="dirichlet"),
        pnmol.problems.heat_1d(bcond="dirichlet"),
    ],
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
    def N(prob1d_discretized):
        return prob1d_discretized.mesh_spatial.shape[0]

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
    def test_y0(prob1d_discretized, N):
        assert prob1d_discretized.y0_array.shape == (N,)

    # Discretisations
    @staticmethod
    @problems_1d_all
    def test_L(prob1d_discretized, N):
        L = prob1d_discretized.L
        assert L.shape == (N, N)

    @staticmethod
    @problems_1d_all
    def test_E_sqrtm(prob1d_discretized, N):
        E_sqrtm = prob1d_discretized.E_sqrtm
        assert E_sqrtm.shape == (N, N)
