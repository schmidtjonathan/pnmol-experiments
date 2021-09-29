"""

"""

from . import (
    differential_operator,
    discretize,
    init,
    iwp,
    kalman,
    kernels,
    lfsolver,
    mesh,
    ode,
    odefilter,
    pde_problems,
    rv,
    solver,
    sqrt,
    stacked_ssm,
)

__version__ = "0.0.1"


# for all modules:
from jax.config import config

config.update("jax_enable_x64", True)
