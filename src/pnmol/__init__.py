"""

"""

from . import (
    differential_operator,
    discretize,
    kernels,
    mesh,
    pde_problems,
    solver,
    sqrt,
)

__version__ = "0.0.1"


# for all modules:
from jax.config import config

config.update("jax_enable_x64", True)
