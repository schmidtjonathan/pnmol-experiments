import jax.numpy as jnp
import pytest
import tornadox

import pnmol


@pytest.fixture
def num_derivatives():
    return 3


@pytest.fixture
def steps():
    return tornadox.step.AdaptiveSteps(abstol=1e-3, reltol=1e-3)


@pytest.fixture
def solver(steps, num_derivatives):
    return pnmol.solver.LatentForceEK0(num_derivatives=num_derivatives, steprule=steps)


pde_problems = pytest.mark.parametrize("pde", [pnmol.pde_problems.heat_1d])


@pde_problems
def test_solver(solver, pde):

    discretized_pde = pde(t0=0.0, tmax=1.0)
    L, E = discretized_pde.L, discretized_pde.E
    assert L.shape == E.shape
    # Assert that E is diagonal
    assert jnp.count_nonzero(E - jnp.diag(jnp.diagonal(E))) == 0
    sol = solver.solve(ivp=discretized_pde)

    assert len(sol.t) == len(sol.mean) == len(sol.cov)
