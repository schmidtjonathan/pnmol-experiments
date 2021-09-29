import jax.numpy as jnp
import pytest
from scipy.integrate import solve_ivp

import pnmol

pde_problems = pytest.mark.parametrize("pde", [pnmol.pde_problems.heat_1d])


@pde_problems
def test_problem(pde):

    discretized_pde = pde(t0=0.0, tmax=10.0)
    L = discretized_pde.L
    E_sqrtm = discretized_pde.E_sqrtm
    assert L.shape == E_sqrtm.shape
    # Assert that E is diagonal
    assert jnp.count_nonzero(E_sqrtm - jnp.diag(jnp.diagonal(E_sqrtm))) == 0
