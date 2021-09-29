"""PDE Problems and some example implementations."""

import functools

import jax.numpy as jnp

from pnmol import diffops, discretize

# PDE Base class and some problem-type-specific implementations


class PDE:
    """PDE base class."""

    def __init__(self, *, diffop, diffop_scale, bbox, **kwargs):
        self.diffop = diffop
        self.diffop_scale = diffop_scale
        self.bbox = bbox

        # The following fields store an optional discretization.
        # They are filled by discretize(), provided by the
        # DiscretizationMixIn below.
        self.L = None
        self.E_sqrtm = None
        self.mesh_spatial = None
        super().__init__(**kwargs)

    def __repr__(self):
        return f"{self.__class__.__name__}(is_discretized={self.is_discretized})"

    @property
    def is_discretized(self):
        return self.L is not None

    @property
    def dimension(self):
        return self.bbox.ndim


class LinearPDE(PDE):
    """Linear PDE problem. Requires mixing with some boundary condition."""

    def to_tornadox_ivp(self):
        """Transform PDE into an IVP. Requires prior discretisation."""

        def f_new(_, x):
            x_padded = self.bc_pad(x)
            x_new = self.L @ x_padded
            return self.bc_remove_pad(x_new)

        df_new = jax.jacfwd(f_new)
        y0_new = self.bc_remove_pad(self.y0_array)
        return f_new, df_new, y0_new, self.t0, self.tmax


class NonLinearMixIn:
    def __init__(self, *, f, df, df_diagonal, **kwargs):
        self.f = f
        self.df = df
        self.df_diagonal = df_diagonal
        super().__init__(**kwargs)


class SemiLinearPDE(PDE, NonLinearMixIn):
    """Linear PDE problem. Requires mixing with some boundary condition."""

    def to_tornadox_ivp(self):
        """Transform PDE into an IVP. Requires prior discretisation."""

        def f_new(_, x):
            x_padded = self.bc_pad(x)
            x_new = self.L @ x_padded + self.f(x_padded)
            return self.bc_remove_pad(x_new)

        df_new = jax.jacfwd(f_new)
        y0_new = self.bc_remove_pad(self.y0_array)
        return f_new, df_new, y0_new, self.t0, self.tmax


# Add boundary conditions through a MixIn


class NeumannMixIn:
    """Neumann condition functionality for PDE problems."""

    def __init__(self, **kwargs):
        self.N = None
        self.W_sqrtm = None
        super().__init__(**kwargs)

    def bc_pad(self, x):
        raise NotImplementedError

    def bc_remove_pad(self, x):
        raise NotImplementedError


class DirichletMixIn:
    """Dirichlet condition functionality for PDE problems."""

    def bc_pad(self, x):
        raise NotImplementedError

    def bc_remove_pad(self, x):
        raise NotImplementedError


# Make the PDE time-dependent and add initial values


class IVPMixIn:
    """Initial value problem functionality for PDE problems.

    Turns a purely spatial PDE into an evolution equation.
    """

    def __init__(self, *, t0, tmax, y0_fun):
        self.t0 = t0
        self.tmax = tmax
        self.y0_fun = y0_fun

        # Holds the discretised initial condition.
        self.y0_array = None


# Add discretisation functionality


class DiscretizationMixIn:
    """Discretisation functionality for PDE problems."""

    def discretize(self, *, mesh_spatial, **kwargs):
        L, E_sqrtm = discretize.discretize(
            self.diffop, mesh_spatial=mesh_spatial, **kwargs
        )

        self.L = self.diffop_scale * L
        self.E_sqrtm = self.diffop_scale * E_sqrtm
        self.mesh_spatial = mesh_spatial

        if isinstance(self, NeumannMixIn):
            raise NotImplementedError
            # self.N = "discretized"
            # self.W_sqrtm = "discretized"

        if isinstance(self, IVPMixIn):
            self.y0_array = self.y0_fun(mesh_spatial.points)


# Mix and match a range of PDE problems.


class LinearEvolutionDirichlet(
    LinearPDE, IVPMixIn, DirichletMixIn, DiscretizationMixIn
):
    pass


class SemiLinearEvolutionDirichlet(
    SemiLinearPDE, IVPMixIn, DirichletMixIn, DiscretizationMixIn
):
    pass


# Some precomputed recipes for PDE examples.


def heat_1d(
    bbox=None, t0=0.0, tmax=20.0, y0_fun=None, diffusion_rate=0.1, bcond="dirichlet"
):
    laplace = diffops.laplace()

    if bbox is None:
        bbox = [0.0, 1.0]
    bbox = jnp.asarray(bbox)

    if y0_fun is None:
        y0_fun = functools.partial(gaussian_bell_1d_centered, bbox=bbox)

    if bcond == "dirichlet":
        return LinearEvolutionDirichlet(
            diffop=laplace,
            diffop_scale=diffusion_rate,
            bbox=bbox,
            t0=t0,
            tmax=tmax,
            y0_fun=y0_fun,
        )
    return LinearEvolutionNeumann(
        diffop=laplace,
        diffop_scale=diffusion_rate,
        bbox=bbox,
        t0=t0,
        tmax=tmax,
        y0_fun=y0_fun,
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
    laplace = diffops.laplace()
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
        mesh_spatial=grid,
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
