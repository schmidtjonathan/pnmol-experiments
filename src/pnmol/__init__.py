__version__ = "0.0.1"


# for all modules:
from jax.config import config

config.update("jax_enable_x64", True)
