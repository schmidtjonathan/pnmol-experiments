from collections import namedtuple

import jax
import jax.numpy as jnp
import numpy as np
import scipy.stats
import tornadox

from pnmol import differential_operator, discretize, kernels, mesh


class DiscretizedPDE(
    namedtuple(
        "_DiscretizedPDE",
        "f spatial_grid t0 tmax y0 df df_diagonal L E",
        defaults=(None, None, None),
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

    def to_tornadox_ivp_1d(self):
        @jax.jit
        def new_f(t, x):

            # Pad x into zeros (dirichlet cond.)
            padded_x = jnp.pad(x, pad_width=1, mode="constant", constant_values=0.0)

            # Evaluate self.f
            new_x = self.f(t, padded_x)

            # Return the interior again
            return new_x[1:-1]

        new_df = jax.jit(jax.jacfwd(new_f, argnums=1))
        new_df_diagonal = None

        return tornadox.ivp.InitialValueProblem(
            f=new_f,
            t0=self.t0,
            tmax=self.tmax,
            y0=self.y0[1:-1],
            df=new_df,
            df_diagonal=new_df_diagonal,
        )


def heat_1d(
    bbox=None, dx=0.05, stencil_size=3, t0=0.0, tmax=20.0, y0=None, diffusion_rate=0.1
):
    # Bounding box for spatial discretization grid
    if bbox is None:
        bbox = [0.0, 1.0]
    bbox = jnp.asarray(bbox)
    assert bbox.ndim == 1

    # Create spatial discretization grid
    grid = mesh.RectangularMesh.from_bounding_boxes_1d(bounding_boxes=bbox, step=dx)

    # Spatial initial condition at t=0
    x = grid.points.reshape((-1,))
    if y0 is None:
        y0 = gaussian_bell_1d(x) * sin_bell_1d(x)

    # PNMOL discretization
    square_exp_kernel = kernels.SquareExponentialKernel(scale=1.0, lengthscale=1.0)
    laplace = differential_operator.laplace()
    L, E = discretize.discretize(
        diffop=laplace, mesh=grid, kernel=square_exp_kernel, stencil_size=stencil_size
    )

    @jax.jit
    def f(_, x):
        return diffusion_rate * L @ x

    @jax.jit
    def df(_, x):
        return diffusion_rate * L

    @jax.jit
    def df_diagonal(_, x):
        return diffusion_rate * jnp.diagonal(L)

    return DiscretizedPDE(
        f=f,
        spatial_grid=grid,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df,
        df_diagonal=df_diagonal,
        L=L,
        E=E,
    )


def wave_1d(bbox=None, dx=0.01, stencil_size=3, t0=0.0, tmax=20.0, y0=None):

    # Bounding box for spatial discretization grid
    if bbox is None:
        bbox = [0.0, 1.0]
    bbox = jnp.asarray(bbox)
    assert bbox.ndim == 1

    # Create spatial discretization grid
    grid = mesh.RectangularMesh.from_bounding_boxes_1d(bounding_boxes=bbox, step=dx)

    # Spatial initial condition at t=0
    if y0 is None:
        mid = (bbox[1] - bbox[0]) * 0.5
        y0 = jnp.pad(
            jnp.exp(-100.0 * (grid.points.reshape(-1)[1:-1] - mid) ** 2),
            pad_width=1,
            mode="constant",
            constant_values=0.0,
        )
        y0 = jnp.concatenate((y0, jnp.zeros_like(y0)))

    # PNMOL discretization
    gauss_kernel = kernels.SquareExponentialKernel(scale=1.0, lengthscale=1.0)
    laplace = differential_operator.laplace()
    L, E = discretize.discretize(
        diffop=laplace, mesh=grid, kernel=gauss_kernel, stencil_size=stencil_size
    )

    I_d = jnp.eye(len(grid))
    zeros = jnp.zeros((len(grid), len(grid)))
    L_extended = jnp.block([[zeros, I_d], [L, zeros]])
    E_extended = jnp.block([[zeros, zeros], [zeros, E]])

    @jax.jit
    def f(_, x):
        return L_extended @ x

    @jax.jit
    def df(_, x):
        return L_extended

    @jax.jit
    def df_diagonal(_, x):
        return jnp.diagonal(L_extended)

    return DiscretizedPDE(
        f=f,
        spatial_grid=grid,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df,
        df_diagonal=df_diagonal,
        L=L_extended,
        E=E_extended,
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
        mid_point = 0.5 * (bbox[1] - bbox[0])
        y0 = jnp.exp(-100 * (grid.points.reshape(-1) - mid_point) ** 2)
        y0 = y0 / y0.max()

    # PNMOL discretization
    lengthscale = dx * int(stencil_size / 2)
    gauss_kernel = kernels.SquareExponentialKernel(
        scale=lengthscale, lengthscale=lengthscale
    )
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

    return DiscretizedPDE(
        f=f,
        spatial_grid=grid,
        t0=t0,
        tmax=tmax,
        y0=y0,
        df=df,
        L=(L_laplace, L_grad),
        E=(E_laplace, E_grad),
    )


def burgers_2d(
    bbox=None,
    dx=0.02,
    stencil_size=3,
    t0=0.0,
    tmax=20.0,
    u0=None,
    v0=None,
    diffusion_param=0.01,
):
    """Burgers' equation in 2D.

    See e.g. https://nbviewer.jupyter.org/github/barbagroup/CFDPython/blob/master/lessons/10_Step_8.ipynb
    """
    # Bounding box for spatial discretization grid
    if bbox is None:
        bbox = [[0.0, 1.0], [0.0, 1.0]]
    bbox = jnp.asarray(bbox)
    assert bbox.ndim == 2

    num_y = int((bbox[0, 1] - bbox[0, 0]) / dx) + 1
    num_x = int((bbox[1, 1] - bbox[1, 0]) / dx) + 1

    # Create spatial discretization grid
    grid = mesh.RectangularMesh.from_bounding_boxes_2d(
        bounding_boxes=bbox, nums=[num_y, num_x]
    )

    # grid_x = mesh.RectangularMesh.from_bounding_boxes_1d(
    #     bounding_boxes=bbox[1], num=num_x
    # )
    # grid_y = mesh.RectangularMesh.from_bounding_boxes_1d(
    #     bounding_boxes=bbox[0], num=num_y
    # )

    # Spatial initial condition at t=0
    if u0 is None:
        u0 = jnp.array(
            scipy.stats.multivariate_normal(np.array([0.5, 0.5]), 0.03 * np.eye(2)).pdf(
                grid.points.reshape(num_y, num_x, 2)
            )
        )
        u0 = (u0 / u0.max()).reshape(-1)
    if v0 is None:
        v0 = jnp.array(
            scipy.stats.multivariate_normal(np.array([0.5, 0.5]), 0.06 * np.eye(2)).pdf(
                grid.points.reshape(num_y, num_x, 2)
            )
        )
        v0 = (v0 / v0.max()).reshape(-1)

    y0 = jnp.concatenate((u0, v0))

    # PNMOL discretization
    lengthscale = dx * int(stencil_size / 2)
    gauss_kernel = kernels.SquareExponentialKernel(1.0, lengthscale)
    laplace = differential_operator.laplace()
    grad_y = differential_operator.gradient_by_dimension(output_coordinate=0)
    grad_x = differential_operator.gradient_by_dimension(output_coordinate=1)
    L_laplace, E_laplace = discretize.discretize(
        diffop=laplace, mesh=grid, kernel=gauss_kernel, stencil_size=stencil_size
    )
    L_grad_x, E_grad_x = discretize.discretize(
        diffop=grad_x, mesh=grid, kernel=gauss_kernel, stencil_size=stencil_size
    )
    L_grad_y, E_grad_y = discretize.discretize(
        diffop=grad_y, mesh=grid, kernel=gauss_kernel, stencil_size=stencil_size
    )

    print(L_laplace.shape, L_grad_x.shape, L_grad_y.shape)
    print(grid.points.shape)
    print(y0.shape)

    @jax.jit
    def f(_, x):
        u, v = jnp.split(x, 2)
        u_new = diffusion_param * L_laplace @ u - u * L_grad_y @ u - v * L_grad_x @ u
        v_new = diffusion_param * L_laplace @ v - u * L_grad_y @ v - v * L_grad_x @ v
        return jnp.concatenate((u_new, v_new))

    @jax.jit
    def df(_, x):
        return jax.jacfwd(f, argnums=1)(_, x)

    return DiscretizedPDE(
        f=f, spatial_grid=grid, t0=t0, tmax=tmax, y0=y0, df=df, L=L_laplace, E=E_laplace
    )


# A bunch of initial condition defaults
# They all adhere to Dirichlet conditions.


def gaussian_bell_1d_centered(x, bbox):
    midpoint = bbox[1] - bbox[0]
    return jnp.exp(-((x - midpoint) ** 2))


def gaussian_bell_1d(x):
    return jnp.exp(-(x ** 2))


def sin_bell_1d(x):
    return jnp.sin(jnp.pi * x)
