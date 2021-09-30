"""PDE Problems and some example implementations.

The central object here is the :class:`PDE`, which holds a differential operator,
a bounding box, and fields that store the discretised operator later on.

This can be combined with a number of additional options:
    * :class:`IVPMixIn`: adds the required fields for making the problem time-dependent.
      (currently in :class:`PDE` per default, but can be extracted in the future)
    * :class:`DiscretizationMixIn:` adds a .discretize() functionality
      (currently in :class:`PDE` per default, but can be extracted in the future)
    * :class:`NonLinearMixIn`: which adds non-linearities (f, df, etc.)

They are optionally combined with Dirichlet or Neumann boundary conditions
(optional, because (a) PNMOL does not _need_ boundary conditions to work on paper,
and because (b) some problems dont have boundary conditions (e.g. PDEs on the sphere)
    * :class:`DirichletMixIn`: adds Dirichlet boundary conditions.
    * :class:`NeumannMixIn`: adds Neumann boundary conditions.

These building blocks allow combination in various ways, some predefined recipes are
    * :class:`LinearEvolutionDirichlet`: used for e.g. the heat equation with Dirichlet conditions.
    * :class:`LinearEvolutionNeumann`: used for e.g. the heat equation with Neumann conditions.
"""

import functools

import jax.numpy as jnp
import jax.scipy.linalg
import tornadox

from pnmol import diffops, discretize, kernels, mesh

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
        self.y0 = None

    @property
    def t_span(self):
        return self.t0, self.tmax


# Add discretisation functionality


class DiscretizationMixIn:
    """Discretisation functionality for PDE problems."""

    def discretize(self, *, mesh_spatial, kernel, **kwargs):
        L, E_sqrtm = discretize.discretize(
            self.diffop, mesh_spatial=mesh_spatial, kernel=kernel, **kwargs
        )

        self.L = self.diffop_scale * L
        self.E_sqrtm = self.diffop_scale * E_sqrtm
        self.mesh_spatial = mesh_spatial

        if isinstance(self, NeumannMixIn):
            if self.dimension > 1:
                raise NotImplementedError
            diffop = diffops.gradient()  # 1d

            # The below is entirely harakiri, but it somehow works.
            k = kernel
            Lk = kernels.Lambda(diffop(k.pairwise, argnums=0))
            LLk = kernels.Lambda(diffop(Lk.pairwise, argnums=1))
            x_left = mesh_spatial[0]
            neighbors_left = mesh_spatial[((0, 1),)]
            weights_left, uncertainty_left = discretize.fd_coeff(
                x=x_left,
                neighbors=neighbors_left,
                k=k,
                L_k=Lk,
                LL_k=LLk,
            )
            x_right = mesh_spatial[-1]
            neighbors_right = mesh_spatial[((-1, -2),)]
            weights_right, uncertainty_right = discretize.fd_coeff(
                x=x_right,
                neighbors=neighbors_right,
                k=k,
                L_k=Lk,
                LL_k=LLk,
                nugget_gram_matrix=1e-10,
            )
            # -1 and -2 are swapped, which reflects their locations in 'neighbors'
            B = jnp.eye(len(mesh_spatial))[((0, 1, -1, -2),)]
            self.B = jax.scipy.linalg.block_diag(-weights_left, weights_right) @ B
            self.R_sqrtm = jnp.diag(jnp.array([uncertainty_left, uncertainty_right]))

        elif isinstance(self, DirichletMixIn):
            self.B = mesh_spatial.boundary_projection_matrix
            self.R_sqrtm = jnp.zeros((self.B.shape[0], self.B.shape[0]))

        if isinstance(self, IVPMixIn):

            # Enforce a scalar initial value
            self.y0 = self.y0_fun(mesh_spatial.points)[:, 0]


# PDE Base class and some problem-type-specific implementations


class PDE(DiscretizationMixIn, IVPMixIn):
    """PDE base class.

    The PDE class is central to all the options below.
    It is extended by LinearPDE, and SemiLinearPDE.
    The additional functionalities IVPMixIn, DirichletMixIn/NeumannMixIn,
    and DiscretizationMixIn rely on the attributes provided herein.
    """

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

        df_new = jax.jacfwd(f_new, argnums=1)
        y0_new = self.bc_remove_pad(self.y0)
        return tornadox.ivp.InitialValueProblem(
            f=f_new, df=df_new, y0=y0_new, t0=self.t0, tmax=self.tmax, df_diagonal=None
        )


class NonLinearMixIn:
    def __init__(self, *, f, df, df_diagonal, **kwargs):
        self.f = f
        self.df = df
        self.df_diagonal = df_diagonal
        super().__init__(**kwargs)


class SemiLinearPDE(PDE, NonLinearMixIn):
    """Semi-Linear PDE problem. Requires mixing with some boundary condition."""

    def to_tornadox_ivp(self):
        """Transform PDE into an IVP. Requires prior discretisation."""

        def f_new(_, x):
            x_padded = self.bc_pad(x)
            x_new = self.L @ x_padded + self.f(x_padded)
            return self.bc_remove_pad(x_new)

        df_new = jax.jacfwd(f_new, argnums=1)
        y0_new = self.bc_remove_pad(self.y0)
        return tornadox.ivp.InitialValueProblem(
            f=f_new, df=df_new, y0=y0_new, t0=self.t0, tmax=self.tmax, df_diagonal=None
        )


# Add boundary conditions through a MixIn


class _BoundaryCondition:
    def __init__(self, **kwargs):
        self.B = None
        self.R_sqrtm = None
        super().__init__(**kwargs)

    def bc_pad(self, x):
        raise NotImplementedError

    def bc_remove_pad(self, x):
        raise NotImplementedError


class NeumannMixIn(_BoundaryCondition):
    """Neumann condition functionality for PDE problems."""

    def bc_pad(self, x):
        if self.dimension > 1:
            raise NotImplementedError
        return jnp.pad(x, pad_width=1, mode="edge")

    def bc_remove_pad(self, x):
        if self.dimension > 1:
            raise NotImplementedError
        return x[1:-1]


class DirichletMixIn(_BoundaryCondition):
    """Dirichlet condition functionality for PDE problems."""

    def bc_pad(self, x):
        if self.dimension > 1:
            raise NotImplementedError

        return jnp.pad(x, pad_width=1, mode="constant", constant_values=1.0)

    def bc_remove_pad(self, x):
        if self.dimension > 1:
            raise NotImplementedError

        return x[1:-1]


# Mix and match a range of PDE problems.


class LinearEvolutionDirichlet(LinearPDE, DirichletMixIn):
    pass


class LinearEvolutionNeumann(LinearPDE, NeumannMixIn):
    pass


# Some precomputed recipes for PDE examples.


def heat_1d_discretized(
    bbox=None,
    dx=0.05,
    stencil_size=3,
    t0=0.0,
    tmax=20.0,
    y0_fun=None,
    diffusion_rate=0.1,
    cov_damping_fd=0.0,
    kernel=None,
    progressbar=False,
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
    mesh_spatial = mesh.RectangularMesh.from_bounding_boxes_1d(heat.bbox, step=dx)

    if kernel is None:
        kernel = kernels.SquareExponential()

    heat.discretize(
        mesh_spatial=mesh_spatial,
        kernel=kernel,
        stencil_size=stencil_size,
        nugget_gram_matrix=cov_damping_fd,
        progressbar=progressbar,
    )
    return heat


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


# A bunch of initial condition defaults
# They all adhere to Dirichlet conditions.


def gaussian_bell_1d_centered(x, bbox):
    midpoint = bbox[1] - bbox[0]
    return jnp.exp(-((x - midpoint) ** 2))


def gaussian_bell_1d(x):
    return jnp.exp(-(x ** 2))


def sin_bell_1d(x):
    return jnp.sin(jnp.pi * x)
