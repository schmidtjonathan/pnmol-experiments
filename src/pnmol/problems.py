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

# List of all the classes in the present module:
# (Might help clarity)
__all__ = [
    # MixIns:
    "PDE",
    "DiscretizationMixIn",
    "SystemDiscretizationMixIn",
    "IVPMixIn",
    "IVPConversionLinearMixIn",
    "IVPConversionSemiLinearMixIn",
    "DirichletMixIn",
    "NeumannMixIn",
    "NonLinearMixIn",
    # PDE Classes:
    "LinearEvolutionDirichlet",  # e.g. heat equation
    "LinearEvolutionNeumann",  # e.g. heat equation
    "LinearPDESystemDirichlet",  # for testing
    "LinearPDESystemNeumann",  # for testing
    "SemiLinearEvolutionSystemDirichlet",  # SIR
    # Full recipes:
    "heat_1d_discretized",
    "heat_1d",
    "sir_1d_discretized",
    "sir_1d",
]

# PDE Base class and some problem-type-specific implementations


class PDE:
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


# Add discretisation functionality


class DiscretizationMixIn:
    """Add functionality for discretization of PDEs to the PDE class."""

    def discretize(self, *, mesh_spatial, kernel, stencil_size, nugget_gram_matrix=0.0):
        L, E_sqrtm = discretize.fd_probabilistic(
            self.diffop,
            mesh_spatial=mesh_spatial,
            kernel=kernel,
            stencil_size=stencil_size,
            nugget_gram_matrix=nugget_gram_matrix,
        )

        self.L = self.diffop_scale * L
        self.E_sqrtm = self.diffop_scale * E_sqrtm
        self.mesh_spatial = mesh_spatial

        if isinstance(self, NeumannMixIn):
            if self.dimension > 1:
                raise NotImplementedError
            self.B, self.R_sqrtm = discretize.fd_probabilistic_neumann_1d(
                mesh_spatial=mesh_spatial,
                kernel=kernel,
                stencil_size=2,
                nugget_gram_matrix=nugget_gram_matrix,
            )

        elif isinstance(self, DirichletMixIn):

            self.B = mesh_spatial.boundary_projection_matrix
            self.R_sqrtm = jnp.zeros((self.B.shape[0], self.B.shape[0]))

        if isinstance(self, IVPMixIn):

            # Enforce a scalar initial value by slicing the zeroth dimension
            self.y0 = self.y0_fun(mesh_spatial.points)[:, 0]


class SystemDiscretizationMixIn:
    """Add functionality for discretization of systems of PDEs to the PDE class."""

    # Overwrite the discretization functionality
    def discretize_system(
        self, *, mesh_spatial, kernel, stencil_size, nugget_gram_matrix=0.0
    ):

        fd = functools.partial(
            discretize.fd_probabilistic,
            mesh_spatial=mesh_spatial,
            kernel=kernel,
            stencil_size=stencil_size,
            nugget_gram_matrix=nugget_gram_matrix,
        )

        # Compute the FD approximation for each differential operator
        fd_output = tuple(map(fd, self.diffop))
        fd_output_scaled = tuple(
            map(lambda s, x: (s * x[0], s * x[1]), self.diffop_scale, fd_output)
        )

        # Read out the coefficients and append to a list
        L_list_scaled, E_sqrtm_list_scaled = [], []
        for (l, e) in fd_output_scaled:
            L_list_scaled.append(l)
            E_sqrtm_list_scaled.append(e)

        # Turn the list of coefficients into a block diagonal matrix
        self.L = jax.scipy.linalg.block_diag(*L_list_scaled)
        self.E_sqrtm = jax.scipy.linalg.block_diag(*E_sqrtm_list_scaled)
        self.mesh_spatial = mesh_spatial

        if isinstance(self, NeumannMixIn):
            if self.dimension > 1:
                raise NotImplementedError
            B, R_sqrtm = discretize.fd_probabilistic_neumann_1d(
                mesh_spatial=mesh_spatial,
                kernel=kernel,
                stencil_size=2,
                nugget_gram_matrix=nugget_gram_matrix,
            )
        elif isinstance(self, DirichletMixIn):
            B = mesh_spatial.boundary_projection_matrix
            R_sqrtm = jnp.zeros((self.B.shape[0], self.B.shape[0]))

        n = len(self.diffop)
        self.B = jax.scipy.linalg.block_diag(*([B] * n))
        self.R_sqrtm = jax.scipy.linalg.block_diag(*([R_sqrtm] * n))

        if isinstance(self, IVPMixIn):

            # Enforce a scalar initial value by slicing the zeroth dimension
            self.y0 = self.y0_fun(mesh_spatial.points).squeeze()


# Make the PDE time-dependent and add initial values


class IVPMixIn:
    """Initial value problem functionality for PDE problems.

    Turns a purely spatial PDE into an evolution equation.
    """

    def __init__(self, *, t0, tmax, y0_fun, **kwargs):
        self.t0 = t0
        self.tmax = tmax
        self.y0_fun = y0_fun

        # Holds the discretised initial condition.
        self.y0 = None

        super().__init__(**kwargs)

    @property
    def t_span(self):
        return self.t0, self.tmax


# Implement to_ivp() conversions for some simple problems.


class _IVPConversionMixIn:
    """Interface for IVP-conversion mixins."""

    def to_tornadox_ivp(self):
        raise NotImplementedError

    def _check_ivp_conversion_conditions(self):
        if not isinstance(self, _BoundaryConditionMixIn):
            raise Exception(
                "Conversion to an IVP requires boundary condition functionality."
            )
        if not isinstance(self, IVPMixIn):
            raise Exception("Conversion to an IVP requires IVP functionality.")
        if self.L is None:
            raise AttributeError("Conversion to an IVP requires prior discretization.")


class IVPConversionLinearMixIn(_IVPConversionMixIn):
    """Add functionality for conversion of linear PDEs to IVPs to the PDE class."""

    def to_tornadox_ivp(self):

        self._check_ivp_conversion_conditions()

        def f_new(_, x):
            assert x.ndim == 1
            x_padded = self.bc_pad(x)
            x_new = self.L @ x_padded
            return self.bc_remove_pad(x_new)

        df_new = jax.jacfwd(f_new, argnums=1)
        y0_new = self.bc_remove_pad(self.y0)
        return tornadox.ivp.InitialValueProblem(
            f=f_new, df=df_new, y0=y0_new, t0=self.t0, tmax=self.tmax, df_diagonal=None
        )


class IVPConversionSemiLinearMixIn(_IVPConversionMixIn):
    """Add functionality for conversion of semilinear PDEs to IVPs to the PDE class."""

    def to_tornadox_ivp(self):
        self._check_ivp_conversion_conditions()

        def f_new(t, x):
            x_padded = self.bc_pad(x)
            x_new = self.L @ x_padded + self.f(t, x_padded)
            return self.bc_remove_pad(x_new)

        df_new = jax.jacfwd(f_new, argnums=1)
        y0_new = self.bc_remove_pad(self.y0)
        return tornadox.ivp.InitialValueProblem(
            f=f_new, df=df_new, y0=y0_new, t0=self.t0, tmax=self.tmax, df_diagonal=None
        )


# Add boundary conditions


class _BoundaryConditionMixIn:
    def __init__(self, **kwargs):
        self.B = None
        self.R_sqrtm = None
        super().__init__(**kwargs)

    def bc_pad(self, x):
        raise NotImplementedError

    def bc_remove_pad(self, x):
        raise NotImplementedError


class NeumannMixIn(_BoundaryConditionMixIn):
    """Neumann condition functionality for PDE problems."""

    def bc_pad(self, x):
        if self.dimension > 1:
            raise NotImplementedError
        return jnp.pad(x, pad_width=1, mode="edge")

    def bc_remove_pad(self, x):
        if self.dimension > 1:
            raise NotImplementedError
        return x[1:-1]


class DirichletMixIn(_BoundaryConditionMixIn):
    """Dirichlet condition functionality for PDE problems."""

    def bc_pad(self, x):
        if self.dimension > 1:
            raise NotImplementedError

        return jnp.pad(x, pad_width=1, mode="constant", constant_values=0.0)

    def bc_remove_pad(self, x):
        if self.dimension > 1:
            raise NotImplementedError

        return x[1:-1]


# Add nonlinearities


class NonLinearMixIn:
    def __init__(self, *, f, df, df_diagonal, **kwargs):
        self.f = f
        self.df = df
        self.df_diagonal = df_diagonal
        super().__init__(**kwargs)


# Mix and match a range of PDE problems.


class LinearEvolutionDirichlet(
    IVPMixIn, IVPConversionLinearMixIn, DiscretizationMixIn, DirichletMixIn, PDE
):
    pass


class LinearEvolutionNeumann(
    IVPMixIn, IVPConversionLinearMixIn, DiscretizationMixIn, NeumannMixIn, PDE
):
    pass


# For testing purposes
class LinearPDESystemNeumann(SystemDiscretizationMixIn, NeumannMixIn, PDE):
    pass


class SemiLinearEvolutionSystemNeumann(
    IVPMixIn,
    NonLinearMixIn,
    IVPConversionSemiLinearMixIn,
    SystemDiscretizationMixIn,
    NeumannMixIn,
    PDE,
):
    pass


# Some precomputed recipes for PDE examples.


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
        return LinearEvolutionDirichlet(
            diffop=laplace,
            diffop_scale=diffusion_rate,
            bbox=bbox,
            t0=t0,
            tmax=tmax,
            y0_fun=y0_fun,
        )
    elif bcond == "neumann":
        return LinearEvolutionNeumann(
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
    return SemiLinearEvolutionSystemNeumann(
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


# A bunch of initial condition defaults. They all adhere to Dirichlet conditions.


def gaussian_bell_1d_centered(x, bbox, width=1.0):
    midpoint = 0.5 * (bbox[1] + bbox[0])
    return jnp.exp(-((x - midpoint) ** 2) / width ** 2)


def gaussian_bell_1d(x):
    return jnp.exp(-(x ** 2))


def sin_bell_1d(x):
    return jnp.sin(jnp.pi * x)
