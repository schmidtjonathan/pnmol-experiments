# for all modules:
from jax.config import config

from . import examples, mixins, problems

config.update("jax_enable_x64", True)
