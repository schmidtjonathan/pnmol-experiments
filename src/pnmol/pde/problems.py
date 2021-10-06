import functools

import jax.numpy as jnp
import jax.scipy.linalg
import tornadox

from pnmol import diffops, discretize, kernels, mesh
from pnmol.pde import mixins


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


class LinearEvolutionDirichlet(
    mixins.IVPMixIn,
    mixins.IVPConversionLinearMixIn,
    mixins.DiscretizationMixIn,
    mixins.DirichletMixIn,
    PDE,
):
    """Linear, time-dependent evolution equations with Dirichlet boundary conditions."""

    pass


class LinearEvolutionNeumann(
    mixins.IVPMixIn,
    mixins.IVPConversionLinearMixIn,
    mixins.DiscretizationMixIn,
    mixins.NeumannMixIn,
    PDE,
):
    """Linear, time-dependent evolution equations with Neumann boundary conditions."""

    pass


# For testing purposes
class SystemLinearPDENeumann(
    mixins.SystemDiscretizationMixIn, mixins.NeumannMixIn, PDE
):
    """Systems of linear PDEs with Neumann boundary conditions."""


class SystemSemiLinearEvolutionNeumann(
    mixins.IVPMixIn,
    mixins.NonLinearMixIn,
    mixins.IVPConversionSemiLinearMixIn,
    mixins.SystemDiscretizationMixIn,
    mixins.SystemNeumannMixIn,
    PDE,
):
    """Systems of semilinear, time-dependent PDEs with Neumann boundary conditions."""

    pass


class SemiLinearEvolutionNeumann(
    mixins.IVPMixIn,
    mixins.NonLinearMixIn,
    mixins.IVPConversionSemiLinearMixIn,
    mixins.DiscretizationMixIn,
    mixins.NeumannMixIn,
    PDE,
):
    pass


class SemiLinearEvolutionDirichlet(
    mixins.IVPMixIn,
    mixins.NonLinearMixIn,
    mixins.IVPConversionSemiLinearMixIn,
    mixins.DiscretizationMixIn,
    mixins.DirichletMixIn,
    PDE,
):
    pass
