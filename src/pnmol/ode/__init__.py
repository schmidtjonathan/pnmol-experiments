"""ODE solver implementations (step-size selection, etc.)"""

# for all modules:
from jax.config import config

from . import init, step

config.update("jax_enable_x64", True)
