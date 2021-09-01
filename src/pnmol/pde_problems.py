from collections import namedtuple

import jax
import jax.numpy as jnp
import scipy.stats

from pnmol import differential_operator, discretize, kernels, mesh


class DiscretizedPDE(
    namedtuple(
        "_DiscretizedPDE",
        "f spatial_grid t0 tmax y0 df",
        defaults=(None, None),
    )
):
    """Initial value problems."""

    @property
    def dimension(self):
        if jnp.isscalar(self.y0):
            return 1
        return self.y0.shape[0]

    @property
    def t_span(self):
        return self.t0, self.tmax


def heat_1d(bbox=None, dx=0.02, stencil_size=3, t0=0.0, tmax=20.0, y0=None):
    # Bounding box for spatial discretization grid
    if bbox is None:
        bbox = [0.0, 1.0]
    bbox = jnp.asarray(bbox)
    assert bbox.ndim == 1

    # Create spatial discretization grid
    grid = mesh.RectangularMesh.from_bounding_boxes_1d(bounding_boxes=bbox, step=dx)

    # Spatial initial condition at t=0
    if y0 is None:
        y0 = jnp.array(
            scipy.stats.norm(0.5 * (bbox[1] - bbox[0]), 0.05).pdf(
                grid.points.reshape(-1)
            )
        )
        y0 = y0 / y0.max()

    # PNMOL discretization
    lengthscale = dx * int(stencil_size / 2)
    gauss_kernel = kernels.GaussianKernel(lengthscale)
    laplace = differential_operator.laplace()
    L, E = discretize.discretize(
        diffop=laplace, mesh=grid, kernel=gauss_kernel, stencil_size=stencil_size
    )

    @jax.jit
    def f(_, x):
        return L @ x

    @jax.jit
    def df(_, x):
        return L

    return (
        DiscretizedPDE(f=f, spatial_grid=grid, t0=t0, tmax=tmax, y0=y0, df=df),
        L,
        E,
    )


def burgers_1d(
    bbox=None, dx=0.02, stencil_size=3, t0=0.0, tmax=20.0, y0=None, diffusion_param=0.01
):
    """Burgers' Equation in 1D.

    According to the first equation in https://en.wikipedia.org/wiki/Burgers'_equation
    """
    # Bounding box for spatial discretization grid
    if bbox is None:
        bbox = [0.0, 1.0]
    bbox = jnp.asarray(bbox)
    assert bbox.ndim == 1

    # Create spatial discretization grid
    grid = mesh.RectangularMesh.from_bounding_boxes_1d(bounding_boxes=bbox, step=dx)

    # Spatial initial condition at t=0
    if y0 is None:
        y0 = jnp.array(scipy.stats.norm(0.5, 0.02).pdf(grid.points.reshape(-1)))
        y0 = y0 / y0.max()

    # PNMOL discretization
    lengthscale = dx * int(stencil_size / 2)
    gauss_kernel = kernels.GaussianKernel(lengthscale)
    laplace = differential_operator.laplace()
    grad = differential_operator.gradient()
    L_laplace, E_laplace = discretize.discretize(
        diffop=laplace, mesh=grid, kernel=gauss_kernel, stencil_size=stencil_size
    )
    L_grad, E_grad = discretize.discretize(
        diffop=grad, mesh=grid, kernel=gauss_kernel, stencil_size=stencil_size
    )

    @jax.jit
    def f(_, x):
        return diffusion_param * L_laplace @ x - x * L_grad @ x

    @jax.jit
    def df(_, x):
        return jax.jacfwd(f, argnums=1)(_, x)

    return DiscretizedPDE(f=f, spatial_grid=grid, t0=t0, tmax=tmax, y0=y0, df=df)

    # PNMOL discretization
    lengthscale = dx * int(stencil_size / 2)
    gauss_kernel = kernels.GaussianKernel(lengthscale)
    laplace = differential_operator.laplace()
    L, E = discretize.discretize(
        diffop=laplace, mesh=grid, kernel=gauss_kernel, stencil_size=stencil_size
    )

    @jax.jit
    def f(_, x):
        return x * (L @ x)

    @jax.jit
    def df(_, x):
        return jax.jacfwd(f, argnums=1)(_, x)

    return (
        DiscretizedPDE(f=f, spatial_grid=grid, t0=t0, tmax=tmax, y0=y0, df=df),
        L,
        E,
    )
