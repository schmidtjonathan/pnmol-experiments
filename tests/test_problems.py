import pytest

import pnmol


@pytest.fixture
def heat_1d():
    return pnmol.pde.heat_1d()


def test_heat_1d_type(heat_1d):
    assert isinstance(heat_1d, pnmol.pde.PDE)
    assert isinstance(heat_1d, pnmol.pde.LinearPDE)


@pytest.fixture
def mesh_spatial(heat_1d):
    return pnmol.mesh.RectangularMesh.from_bounding_boxes_1d(heat_1d.bbox, step=0.1)


@pytest.fixture
def kernel():
    return pnmol.kernels.SquareExponential()


@pytest.fixture
def heat_1d_discretized(heat_1d, mesh_spatial, kernel):
    heat_1d.discretize(
        mesh_spatial=mesh_spatial,
        kernel=kernel,
        stencil_size=3,
        nugget_gram_matrix=1e-10,
        progressbar=False,
    )
    return heat_1d


def test_heat_1d_discretized_type(heat_1d_discretized):
    assert isinstance(heat_1d_discretized, pnmol.pde.PDE)
    assert isinstance(heat_1d_discretized, pnmol.pde.LinearPDE)
