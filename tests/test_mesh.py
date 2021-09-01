import unittest

import jax.numpy as jnp

from pnmol import mesh

SET = jnp.array(
    [
        [0.0, 0.0, -1.0],
        [0.1, 0.2, 0.3],
        [0.6, 0.21, 0.672],
        [1.0, 1.0, 2.0],
        [0.0, 1.0, 0.0],
    ]
)

SET_BOUNDARY = jnp.array([True, False, False, True, True])  # pt on boundary
SET_INTERIOR = jnp.invert(SET_BOUNDARY)  # pt not on boundary (i.e. in interior)
TRUE_BOXES = jnp.array([[0.0, 1.0], [0.0, 1.0], [-1.0, 2.0]])


class ReadBoundingBoxesTestCase(unittest.TestCase):
    def test_read_bounding_boxes(self):
        read_boxes = mesh.read_bounding_boxes(SET)
        discrepancy = jnp.linalg.norm(TRUE_BOXES - read_boxes)
        self.assertLess(discrepancy, 1e-14)


class TestRectangularMesh(unittest.TestCase):
    def setUp(self):
        self.mesh = mesh.RectangularMesh(points=SET)

    def test_boundary(self):
        boundary_pts, indices = self.mesh.boundary
        discrepancy = jnp.linalg.norm(SET[SET_BOUNDARY] - boundary_pts)
        self.assertLess(discrepancy, 1e-14)
        self.assertEqual(jnp.all(indices == SET_BOUNDARY), True)

    def test_interior(self):
        interior_pts, indices = self.mesh.interior
        discrepancy = jnp.linalg.norm(SET[SET_INTERIOR] - interior_pts)
        self.assertLess(discrepancy, 1e-14)
        self.assertEqual(jnp.all(indices == SET_INTERIOR), True)

    def test_neighbours(self):
        neighbours, _ = self.mesh.neighbours(SET[0], num=2)
        error_n1 = jnp.linalg.norm(neighbours[0] - SET[0])
        error_n2 = jnp.linalg.norm(neighbours[1] - SET[1])
        self.assertLess(error_n1, 1e-14)
        self.assertLess(error_n2, 1e-14)

    def test_neighbours_bounding_boxes(self):
        """
        Bounding boxes of the neighbours() result must be the same
        as those of the original set. This was a bug once.
        """
        neighbours, _ = self.mesh.neighbours(SET[0], num=2)
        original_bbox = self.mesh.bounding_boxes
        new_bbox = neighbours.bounding_boxes
        discrepancy = jnp.linalg.norm(original_bbox - new_bbox)
        self.assertLess(discrepancy, 1e-14)

    def test_len(self):
        self.assertEqual(len(self.mesh), len(SET))

    def test_getitem_index(self):
        self.assertEqual(self.mesh[2, 0], SET[2, 0])

    def test_getitem_slice(self):
        difference = jnp.linalg.norm(self.mesh[1:3] - SET[1:3])
        self.assertLess(difference, 1e-14)

    def test_getitem_ellipsis(self):
        difference = jnp.linalg.norm(self.mesh[:, 0] - SET[:, 0])
        self.assertLess(difference, 1e-14)

    def test_from_bounding_boxes_1d(self):
        grid = mesh.RectangularMesh.from_bounding_boxes_1d(
            bounding_boxes=TRUE_BOXES[0], step=0.1
        )
        self.assertEqual(isinstance(grid, mesh.RectangularMesh), True)
        self.assertEqual(grid.ndim, 2)
        self.assertEqual(grid.dimension, 1)

    def test_from_bounding_boxes_2d(self):
        grid = mesh.RectangularMesh.from_bounding_boxes_2d(
            bounding_boxes=TRUE_BOXES[jnp.array([0, 1])], steps=[0.1, 0.2]
        )
        self.assertEqual(isinstance(grid, mesh.RectangularMesh), True)
        self.assertEqual(grid.ndim, 2)
        self.assertEqual(grid.dimension, 2)


if __name__ == "__main__":
    unittest.main()
