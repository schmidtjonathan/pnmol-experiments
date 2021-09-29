"""Tests for random variables."""


import jax
import jax.numpy as jnp
import pytest

import pnmol.base.rv

# Common fixtures


@pytest.fixture
def dimension():
    return 3


@pytest.fixture
def batch_size():
    return 5


@pytest.fixture
def mean(dimension):
    return jnp.arange(dimension)


@pytest.fixture
def cov_sqrtm(dimension):
    return jnp.arange(dimension ** 2).reshape((dimension, dimension))


class TestMultivariateNormal:
    @staticmethod
    @pytest.fixture
    def multivariate_normal(mean, cov_sqrtm):
        return pnmol.base.rv.MultivariateNormal(mean=mean, cov_sqrtm=cov_sqrtm)

    @staticmethod
    def test_type(multivariate_normal):
        assert isinstance(multivariate_normal, pnmol.base.rv.MultivariateNormal)

    @staticmethod
    def test_cov(multivariate_normal):
        SC = multivariate_normal.cov_sqrtm
        C = multivariate_normal.cov
        assert jnp.allclose(C, SC @ SC.T)

    @staticmethod
    def test_jittable(multivariate_normal):
        def fun(rv):
            m, sc = rv
            return pnmol.base.rv.MultivariateNormal(2 * m, 2 * sc)

        fun_jitted = jax.jit(fun)
        out = fun_jitted(multivariate_normal)
        assert type(out) == type(multivariate_normal)
