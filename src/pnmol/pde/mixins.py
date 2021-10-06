"""Extend PDE problem functionality.

Includes time-dependency, boundary conditions, discretisation, and more.
"""
import functools

import jax.numpy as jnp
import jax.scipy.linalg
import tornadox

from pnmol import diffops, discretize, kernels, mesh

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

        if isinstance(self, _BoundaryConditionMixInInterface):
            if isinstance(self, (NeumannMixIn, SystemNeumannMixIn)):
                if self.dimension > 1:
                    raise NotImplementedError
                B, R_sqrtm = discretize.fd_probabilistic_neumann_1d(
                    mesh_spatial=mesh_spatial,
                    kernel=kernel,
                    stencil_size=2,
                    nugget_gram_matrix=nugget_gram_matrix,
                )
            elif isinstance(self, (DirichletMixIn, SystemDirichletMixIn)):
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


class _IVPConversionMixInInterface:
    """Interface for IVP-conversion mixins."""

    def to_tornadox_ivp(self):
        raise NotImplementedError

    def _check_ivp_conversion_conditions(self):
        """Conversion to an IVP relies on a few assumptions. Check them here."""
        if not isinstance(self, _BoundaryConditionMixInInterface):
            raise Exception(
                "Conversion to an IVP requires boundary condition functionality."
            )
        if not isinstance(self, IVPMixIn):
            raise Exception("Conversion to an IVP requires IVP functionality.")
        if self.L is None:
            raise AttributeError("Conversion to an IVP requires prior discretization.")
        if self.dimension > 1:
            raise NotImplementedError(
                "Conversion to an IVP in more than one spatial dimension is not supported."
            )


class IVPConversionLinearMixIn(_IVPConversionMixInInterface):
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


class IVPConversionSemiLinearMixIn(_IVPConversionMixInInterface):
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


class _BoundaryConditionMixInInterface:
    def __init__(self, **kwargs):
        self.B = None
        self.R_sqrtm = None
        super().__init__(**kwargs)

    def bc_pad(self, x):
        raise NotImplementedError

    def bc_remove_pad(self, x):
        raise NotImplementedError


class _SystemBoundaryConditionMixinInterface(_BoundaryConditionMixInInterface):
    def __init__(self, *, bc, **kwargs):
        self.bc = bc
        super().__init__(**kwargs)

    def bc_pad(self, x):
        n = len(self.diffop)
        x_reshaped = x.reshape((n, -1))
        x_split_padded = jnp.apply_along_axis(self.bc.bc_pad, -1, x_reshaped)
        return x_split_padded.reshape((-1,))

    def bc_remove_pad(self, x):
        n = len(self.diffop)
        x_reshaped = x.reshape((n, -1))
        x_reshaped_no_pad = jnp.apply_along_axis(self.bc.bc_remove_pad, -1, x_reshaped)
        return x_reshaped_no_pad.reshape((-1,))


class SystemNeumannMixIn(_SystemBoundaryConditionMixinInterface):
    def __init__(self, **kwargs):
        super().__init__(bc=NeumannMixIn(), **kwargs)


class SystemDirichletMixIn(_SystemBoundaryConditionMixinInterface):
    def __init__(self, **kwargs):
        super().__init__(bc=DirichletMixIn(), **kwargs)


class NeumannMixIn(_BoundaryConditionMixInInterface):
    """Neumann condition functionality for PDE problems."""

    def bc_pad(self, x):
        return jnp.pad(x, pad_width=1, mode="edge")

    def bc_remove_pad(self, x):
        return x[1:-1]


class DirichletMixIn(_BoundaryConditionMixInInterface):
    """Dirichlet condition functionality for PDE problems."""

    def __init__(self, **kwargs):
        self.neumann = NeumannMixIn()
        super().__init__(**kwargs)

    def bc_pad(self, x):
        return jnp.pad(x, pad_width=1, mode="constant", constant_values=0.0)

    def bc_remove_pad(self, x):
        return x[1:-1]


# Add nonlinearities


class NonLinearMixIn:
    def __init__(self, *, f, df, df_diagonal, **kwargs):
        self.f = f
        self.df = df
        self.df_diagonal = df_diagonal
        super().__init__(**kwargs)
