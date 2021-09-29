from collections import namedtuple

import jax
import jax.numpy as jnp
import tornadox

from pnmol import differential_operator, discretize, kernels, mesh


class PDEProblemMixin:
    """Properties and functionalities for general PDE problems."""

    @property
    def dimension(self):
        if jnp.isscalar(self.y0):
            return 1
        return self.y0.shape[0]

    @property
    def t_span(self):
        return self.t0, self.tmax


_LinearPDEBaseClass = namedtuple(
    "_LinearPDEBaseClass", "spatial_grid t0 tmax y0 L E_sqrtm"
)

_NonLinearPDEBaseClass = namedtuple(
    "_SemiLinearPDEProblem", "spatial_grid t0 tmax y0 f df L E_sqrtm"
)


class LinearPDEProblem(
    _LinearPDEBaseClass,
    PDEProblemMixin,
):
    def to_tornadox_ivp_1d(self):
        @jax.jit
        def new_f(t, x):

            # Pad x into zeros (dirichlet cond.)
            padded_x = jnp.pad(x, pad_width=1, mode="constant", constant_values=0.0)

            # Evaluate self.f
            new_x = self.L @ padded_x

            # Return the interior again
            return new_x[1:-1]

        new_df = jax.jit(jax.jacfwd(new_f, argnums=1))

        return tornadox.ivp.InitialValueProblem(
            f=new_f,
            t0=self.t0,
            tmax=self.tmax,
            y0=self.y0[1:-1],
            df=new_df,
            df_diagonal=None,
        )


class SemiLinearPDEProblem(
    _NonLinearPDEBaseClass,
    PDEProblemMixin,
):
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

        return tornadox.ivp.InitialValueProblem(
            f=new_f,
            t0=self.t0,
            tmax=self.tmax,
            y0=self.y0[1:-1],
            df=new_df,
            df_diagonal=None,
        )


def heat_1d(
    bbox=None,
    dx=0.05,
    stencil_size=3,
    t0=0.0,
    tmax=20.0,
    y0=None,
    diffusion_rate=0.1,
    cov_damping_fd=0.0,
    kernel=None,
    progressbar=False,
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

    # Default kernels
    if kernel is None:
        kernel = kernels.SquareExponential()

    # PNMOL discretization
    laplace = differential_operator.laplace()
    L, E_sqrtm = discretize.discretize(
        diffop=laplace,
        mesh=grid,
        kernel=kernel,
        stencil_size=stencil_size,
        cov_damping=cov_damping_fd,
        progressbar=progressbar,
    )
    scaled_laplace = diffusion_rate * L
    scaled_sqrt_error = diffusion_rate * E_sqrtm

    return LinearPDEProblem(
        spatial_grid=grid,
        t0=t0,
        tmax=tmax,
        y0=y0,
        L=scaled_laplace,
        E_sqrtm=scaled_sqrt_error,
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
