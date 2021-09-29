"""

"""

from . import (
    base,
    differential_operator,
    discretize,
    iwp,
    kalman,
    kernels,
    lfsolver,
    mesh,
    ode,
    odefilter,
    pde_problems,
    solver,
    sqrt,
    stacked_ssm,
)

__version__ = "0.0.1"


# for all modules:
from jax.config import config

config.update("jax_enable_x64", True)
