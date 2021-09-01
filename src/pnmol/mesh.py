"""Mesh containers."""

import abc

import jax.numpy as jnp
import numpy as np
import scipy.spatial


class Mesh(abc.ABC):
    """Scattered points."""

    def __init__(self, points):
        self.points = points
        self._tree = scipy.spatial.KDTree(data=self.points)

    @abc.abstractmethod
    def neighbours(self, point, num):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def boundary(self):
        raise NotImplementedError

    @property
    @abc.abstractmethod
    def interior(self):
        raise NotImplementedError

    def sort(self):
        try:
            interior_pts, _ = self.interior
            boundary_pts, _ = self.boundary
            self.points = jnp.vstack((interior_pts, boundary_pts))
            self._tree = scipy.spatial.KDTree(data=self.points)
        except NotImplementedError:
            raise NotImplementedError(
                "sort() is only available if boundary and interior are available."
            )

    def __len__(self):
        return len(self.points)

    def __getitem__(self, key):
        return self.points.__getitem__(key)

    def __repr__(self):
        return f"{type(self).__name__}({repr(self.points)})"

    @property
    def shape(self):
        return self.points.shape

    @property
    def ndim(self):
        """Dimension of the mesh AS AN ARRAY."""
        return self.points.ndim

    @property
    def dimension(self):
        """Spatial dimension of the mesh."""
        return self.points.shape[-1]

    @property
    def fill_distance(self):
        return jnp.amin(scipy.spatial.distance_matrix(self.points, self.points))


class RectangularMesh(Mesh):
    """Rectangular mesh."""

    def __init__(self, points, bounding_boxes=None):
        if bounding_boxes is not None:
            self.bounding_boxes = bounding_boxes
        else:
            self.bounding_boxes = read_bounding_boxes(points)
        super().__init__(points)

    @staticmethod
    def from_bounding_boxes_1d(bounding_boxes, step=None, num=None):

        bounding_boxes = jnp.asarray(bounding_boxes)

        if int(step is None) + int(num is None) != 1:
            raise ValueError("Provide exactly one of step or num.")

        if step is not None:
            num = int((bounding_boxes[1] - bounding_boxes[0]) / step) + 1

        X = jnp.linspace(
            start=bounding_boxes[0], stop=bounding_boxes[1], num=num, endpoint=True
        )

        return RectangularMesh(X.reshape(-1, 1))

    @staticmethod
    def from_bounding_boxes_2d(bounding_boxes, steps=None, nums=None):

        bounding_boxes = jnp.asarray(bounding_boxes)

        if int(steps is None) + int(nums is None) != 1:
            raise ValueError("Provide exactly one of step or num.")

        if steps is not None:
            step_y, step_x = steps
            num_y = int((bounding_boxes[0, 1] - bounding_boxes[0, 0]) / step_y) + 1
            num_x = int((bounding_boxes[1, 1] - bounding_boxes[1, 0]) / step_x) + 1

        if nums is not None:
            num_y, num_x = nums

        Y = jnp.linspace(
            start=bounding_boxes[0, 0],
            stop=bounding_boxes[0, 1],
            num=num_y,
            endpoint=True,
        )
        X = jnp.linspace(
            start=bounding_boxes[1, 0],
            stop=bounding_boxes[1, 1],
            num=num_x,
            endpoint=True,
        )
        X_mesh, Y_mesh = jnp.meshgrid(X, Y)

        return RectangularMesh(jnp.array(list(zip(X_mesh.flatten(), Y_mesh.flatten()))))

    def neighbours(self, point, num):
        if num <= 0:
            raise ValueError("num >= 1 required!")
        elif num == 1:
            return RectangularMesh(points=point[None, :])
        distances, indices = self._tree.query(x=point, k=num)
        neighbours = self.points[indices]
        return RectangularMesh(neighbours, bounding_boxes=self.bounding_boxes), indices

    @property
    def boundary(self):
        is_boundary = jnp.logical_or(
            self.points[:, 0] == self.bounding_boxes[0, 0],
            self.points[:, 0] == self.bounding_boxes[0, 1],
        )

        for d in range(1, len(self.bounding_boxes)):
            this_dim = jnp.logical_or(
                self.points[:, d] == self.bounding_boxes[d, 0],
                self.points[:, d] == self.bounding_boxes[d, 1],
            )
            is_boundary = jnp.logical_or(is_boundary, this_dim)
        return self.points[is_boundary], is_boundary

    @property
    def interior(self):
        is_boundary = jnp.logical_and(
            self.points[:, 0] != self.bounding_boxes[0, 0],
            self.points[:, 0] != self.bounding_boxes[0, 1],
        )

        for d in range(1, len(self.bounding_boxes)):
            this_dim = jnp.logical_and(
                self.points[:, d] != self.bounding_boxes[d, 0],
                self.points[:, d] != self.bounding_boxes[d, 1],
            )
            is_boundary = jnp.logical_and(is_boundary, this_dim)
        return self.points[is_boundary], is_boundary


def read_bounding_boxes(points):
    points = jnp.asarray(points)
    bboxes = np.nan * np.ones((points.shape[-1], 2))
    for d in range(points.shape[-1]):
        bboxes[d, 0] = np.amin(points[:, d])
        bboxes[d, 1] = np.amax(points[:, d])
    return jnp.array(bboxes)
