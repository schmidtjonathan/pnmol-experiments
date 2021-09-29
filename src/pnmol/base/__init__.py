"""Auxiliary implementations for PNMOL (random variables, kalman filters, etc.)."""

# for all modules:
from jax.config import config

from . import iwp, kalman, rv, sqrt, stacked_ssm

config.update("jax_enable_x64", True)
