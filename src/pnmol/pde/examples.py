"""Example PDE problem recipes."""

import functools

import jax.numpy as jnp
import jax.scipy.linalg
import tornadox

from pnmol import diffops, discretize, kernels, mesh
from pnmol.pde import problems


def heat_1d_discretized(
    bbox=None,
    dx=0.05,
    stencil_size=3,
    t0=0.0,
    tmax=5.0,
    y0_fun=None,
    diffusion_rate=0.05,
    nugget_gram_matrix_fd=0.0,
    kernel=None,
    bcond="dirichlet",
):
    heat = heat_1d(
        bbox=bbox,
        t0=t0,
        tmax=tmax,
        y0_fun=y0_fun,
        diffusion_rate=diffusion_rate,
        bcond=bcond,
    )
    mesh_spatial = mesh.RectangularMesh.from_bbox_1d(heat.bbox, step=dx)

    if kernel is None:
        kernel = kernels.SquareExponential()

    heat.discretize(
        mesh_spatial=mesh_spatial,
        kernel=kernel,
        stencil_size=stencil_size,
        nugget_gram_matrix=nugget_gram_matrix_fd,
    )
    return heat


def heat_1d(
    bbox=None, t0=0.0, tmax=5.0, y0_fun=None, diffusion_rate=0.05, bcond="dirichlet"
):
    laplace = diffops.laplace()

    if bbox is None:
        bbox = [0.0, 1.0]
    bbox = jnp.asarray(bbox)

    if y0_fun is None:
        bell_centered = functools.partial(gaussian_bell_1d_centered, bbox=bbox)
        y0_fun = lambda x: bell_centered(x) * sin_bell_1d(x)

    if bcond == "dirichlet":
        return problems.LinearEvolutionDirichlet(
            diffop=laplace,
            diffop_scale=diffusion_rate,
            bbox=bbox,
            t0=t0,
            tmax=tmax,
            y0_fun=y0_fun,
        )
    elif bcond == "neumann":
        return problems.LinearEvolutionNeumann(
            diffop=laplace,
            diffop_scale=diffusion_rate,
            bbox=bbox,
            t0=t0,
            tmax=tmax,
            y0_fun=y0_fun,
        )
    raise ValueError


def sir_1d_discretized(
    bbox=None,
    dx=0.05,
    t0=0.0,
    tmax=50.0,
    beta=0.3,
    gamma=0.07,
    N=1000.0,
    diffusion_rate_S=0.1,
    diffusion_rate_I=0.1,
    diffusion_rate_R=0.1,
    kernel=None,
    nugget_gram_matrix_fd=0.0,
    stencil_size=3,
):
    sir = sir_1d(
        bbox=bbox,
        t0=t0,
        tmax=tmax,
        diffusion_rate_S=diffusion_rate_S,
        diffusion_rate_I=diffusion_rate_I,
        diffusion_rate_R=diffusion_rate_R,
        beta=beta,
        gamma=gamma,
        N=N,
    )
    mesh_spatial = mesh.RectangularMesh.from_bbox_1d(sir.bbox, step=dx)

    if kernel is None:
        kernel = kernels.SquareExponential()

    sir.discretize_system(
        mesh_spatial=mesh_spatial,
        kernel=kernel,
        stencil_size=stencil_size,
        nugget_gram_matrix=nugget_gram_matrix_fd,
    )
    return sir


def sir_1d(
    bbox=None,
    t0=0.0,
    tmax=50.0,
    diffusion_rate_S=0.1,
    diffusion_rate_I=0.1,
    diffusion_rate_R=0.1,
    beta=0.3,
    gamma=0.07,
    N=1000.0,
):

    if bbox is None:
        bbox = [0.0, 1.0]
    bbox = jnp.asarray(bbox)

    def y0_fun(x):
        init_infectious = 800.0 * gaussian_bell_1d_centered(x, bbox, width=0.5) + 1.0
        s0 = N * jnp.ones_like(init_infectious) - init_infectious
        i0 = init_infectious
        r0 = jnp.zeros_like(init_infectious)
        return jnp.concatenate((s0, i0, r0))

    @jax.jit
    def _sir_rhs(s, i, r):
        spatial_N = s + i + r
        new_s = -beta * s * i / spatial_N
        new_i = beta * s * i / spatial_N - gamma * i
        new_r = gamma * i
        return new_s, new_i, new_r

    @jax.jit
    def f(t, x):
        s, i, r = jnp.split(x, 3)
        new_s, new_i, new_r = _sir_rhs(s, i, r)
        return jnp.concatenate((new_s, new_i, new_r))

    df = jax.jit(jax.jacfwd(f, argnums=1))

    laplace = diffops.laplace()
    return problems.SystemSemiLinearEvolutionNeumann(
        diffop=(laplace, laplace, laplace),
        diffop_scale=(diffusion_rate_S, diffusion_rate_I, diffusion_rate_R),
        bbox=bbox,
        t0=t0,
        tmax=tmax,
        y0_fun=y0_fun,
        f=f,
        df=df,
        df_diagonal=None,
    )


def spruce_budworm_1d_discretized(
    bbox=None,
    t0=0.0,
    tmax=10.0,
    diffusion_rate=1.0,
    y0_fun=None,
    dx=0.1,
    kernel=None,
    nugget_gram_matrix_fd=0.0,
    stencil_size=3,
    bcond="dirichlet",
    growth_rate=1.0,
):
    spruce = spruce_budworm_1d(
        bbox=bbox,
        t0=t0,
        tmax=tmax,
        diffusion_rate=diffusion_rate,
        y0_fun=y0_fun,
        bcond=bcond,
    )
    mesh_spatial = mesh.RectangularMesh.from_bbox_1d(spruce.bbox, step=dx)

    if kernel is None:
        kernel = kernels.SquareExponential()

    spruce.discretize(
        mesh_spatial=mesh_spatial,
        kernel=kernel,
        stencil_size=stencil_size,
        nugget_gram_matrix=nugget_gram_matrix_fd,
    )
    return spruce


def spruce_budworm_1d(
    bbox=None,
    t0=0.0,
    tmax=10.0,
    diffusion_rate=1.0,
    y0_fun=None,
    bcond="dirichlet",
    growth_rate=1.0,
):
    """Explained in https://www-m6.ma.tum.de/~kuttler/script_reaktdiff.pdf (ctrl+f for "spruce")"""
    if bbox is None:
        bbox = [0.0, 1.0]
    bbox = jnp.asarray(bbox)

    if y0_fun is None:
        y0_fun = functools.partial(gaussian_bell_1d_centered, bbox=bbox, width=1.0)

    def f_spruce_general(_, x, c):
        return c * x * (1.0 - x)

    f_spruce = jax.jit(functools.partial(f_spruce_general, c=growth_rate))
    df_spruce = jax.jit(jax.jacfwd(f_spruce, argnums=1))

    if bcond == "dirichlet":
        return problems.SemiLinearEvolutionDirichlet(
            t0=t0,
            tmax=tmax,
            y0_fun=y0_fun,
            bbox=bbox,
            diffop=diffops.laplace(),
            diffop_scale=diffusion_rate,
            f=f_spruce,
            df=df_spruce,
            df_diagonal=None,
        )
    elif bcond == "neumann":
        return problems.SemiLinearEvolutionNeumann(
            t0=t0,
            tmax=tmax,
            y0_fun=y0_fun,
            bbox=bbox,
            diffop=diffops.laplace(),
            diffop_scale=diffusion_rate,
            f=f_spruce,
            df=df_spruce,
            df_diagonal=None,
        )
    raise ValueError


# A bunch of initial condition defaults. They all adhere to Dirichlet conditions.


def gaussian_bell_1d_centered(x, bbox, width=1.0):
    midpoint = 0.5 * (bbox[1] + bbox[0])
    return jnp.exp(-((x - midpoint) ** 2) / width ** 2)


def gaussian_bell_1d(x):
    return jnp.exp(-(x ** 2))


def sin_bell_1d(x):
    return jnp.sin(jnp.pi * x)
