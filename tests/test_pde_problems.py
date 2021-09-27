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
    sol = solve_ivp(
        discretized_pde.f,
        t_span=discretized_pde.t_span,
        y0=discretized_pde.y0,
        method="Radau",
    )

    assert len(sol.t) == len(sol.y.T)
