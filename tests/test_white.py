import jax.numpy as jnp
import pytest
import tornadox

import pnmol
import pnmol.ode


@pytest.mark.parametrize("solver", [pnmol.white.LinearWhiteNoiseEK1])
def test_solve(solver):
    """The Heat equation is solved without creating NaNs."""
    dt = 0.1
    nu = 2
    steprule = pnmol.ode.step.Constant(dt)
    heat = pnmol.problems.heat_1d_discretized(
        tmax=1.0,
        dx=0.2,
        stencil_size=3,
        diffusion_rate=0.05,
        kernel=pnmol.kernels.Polynomial(),
        cov_damping_fd=0.0,
    )
    # Solve the discretised PDE
    solver = solver(num_derivatives=nu, steprule=steprule)
    out = solver.solve(heat)
    assert not jnp.any(jnp.isnan(out.mean))
    assert not jnp.any(jnp.isnan(out.cov_sqrtm))
