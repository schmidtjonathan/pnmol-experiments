"""Test the PDE filter implementations."""
import jax
import jax.numpy as jnp
import pytest
import tornadox

import pnmol
import pnmol.ode
import pnmol.pde


@pytest.mark.parametrize(
    "solver", [pnmol.white.LinearWhiteNoiseEK1, pnmol.latent.LinearLatentForceEK1]
)
@pytest.mark.parametrize("bcond", ["neumann", "dirichlet"])
def test_solve_linear(solver, bcond):
    """The Heat equation is solved without creating NaNs."""
    dt = 0.1
    nu = 2
    steprule = pnmol.ode.step.Constant(dt)
    heat = pnmol.pde.examples.heat_1d_discretized(
        tmax=1.0,
        dx=0.2,
        stencil_size=3,
        diffusion_rate=0.05,
        kernel=pnmol.kernels.Polynomial(),
        nugget_gram_matrix_fd=0.0,
        bcond=bcond,
    )
    # Solve the discretised PDE
    solver = solver(num_derivatives=nu, steprule=steprule)
    out = solver.solve(heat)
    assert not jnp.any(jnp.isnan(out.mean))
    assert not jnp.any(jnp.isnan(out.cov_sqrtm))


@pytest.mark.parametrize(
    "solver",
    [pnmol.white.SemiLinearWhiteNoiseEK1, pnmol.latent.SemiLinearLatentForceEK1],
)
@pytest.mark.parametrize("bcond", ["neumann", "dirichlet"])
def test_solve_semilinear(solver, bcond):
    """The Heat equation is solved without creating NaNs."""
    dt = 0.1
    nu = 2
    steprule = pnmol.ode.step.Constant(dt)
    heat = pnmol.pde.examples.spruce_budworm_1d_discretized(
        tmax=1.0,
        dx=0.2,
        stencil_size=3,
        diffusion_rate=0.05,
        kernel=pnmol.kernels.Polynomial(),
        nugget_gram_matrix_fd=0.0,
        bcond=bcond,
    )
    # Solve the discretised PDE
    solver = solver(num_derivatives=nu, steprule=steprule)
    out = solver.solve(heat)
    assert not jnp.any(jnp.isnan(out.mean))
    assert not jnp.any(jnp.isnan(out.cov_sqrtm))
