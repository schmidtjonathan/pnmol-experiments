import jax.numpy as jnp
import pytest

import pnmol


@pytest.fixture
def prob1d():
    return pnmol.problems.heat_1d()


@pytest.fixture
def mesh_spatial(prob1d):
    return pnmol.mesh.RectangularMesh.from_bounding_boxes_1d(prob1d.bbox, step=0.1)


@pytest.fixture
def kernel():
    return pnmol.kernels.SquareExponential()


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
    def test_type(prob1d_discretized):
        assert isinstance(prob1d_discretized, pnmol.problems.LinearPDE)

    @staticmethod
    def test_t0(prob1d_discretized):
        assert jnp.isscalar(prob1d_discretized.t0)

    @staticmethod
    def test_tmax(prob1d_discretized):
        assert jnp.isscalar(prob1d_discretized.t0)

    @staticmethod
    def test_y0(prob1d_discretized):
        N = prob1d_discretized.mesh_spatial.shape[0]
        assert prob1d_discretized.y0_array.shape == (N,)
