"""Random variables."""


from collections import namedtuple

import jax.numpy as jnp


class MultivariateNormal(namedtuple("_MultivariateNormal", "mean cov_sqrtm")):
    """Multivariate normal distributions."""

    @property
    def cov(self):
        return self.cov_sqrtm @ self.cov_sqrtm.T
