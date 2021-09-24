"""

"""

from . import (
    differential_operator,
    discretize,
    init,
    iwp,
    kernels,
    mesh,
    odefilter,
    pde_problems,
    rv,
    sqrt,
    step,
)

__version__ = "0.0.1"


# for all modules:
from jax.config import config

config.update("jax_enable_x64", True)
