"""Test the PDE filter implementations."""
import jax
import jax.numpy as jnp
import pytest
import tornadox
from pytest_cases import parametrize_with_cases

import pnmol
import pnmol.odetools
import pnmol.pde

# Common fixtures used for all cases


@pytest.fixture
def num_derivatives():
    return 2


@pytest.fixture
def steprule():
    return pnmol.odetools.step.Constant(dt=0.1)


@pytest.fixture
def kernel():
    return pnmol.kernels.SquareExponential()


# Linear PDE test problem


@pytest.fixture
def pde_linear(bcond, kernel):
    heat = pnmol.pde.examples.heat_1d_discretized(
        tmax=1.0,
        dx=0.2,
        stencil_size=3,
        diffusion_rate=0.05,
        kernel=kernel,
        nugget_gram_matrix_fd=0.0,
        bcond=bcond,
    )
    return heat


def case_linear_latent(pde_linear, steprule, num_derivatives, kernel):
    solver = pnmol.latent.LinearLatentForceEK1(
        num_derivatives=num_derivatives,
        steprule=steprule,
        spatial_kernel=kernel + pnmol.kernels.WhiteNoise(),
    )
    return solver, pde_linear


def case_linear_white(pde_linear, steprule, num_derivatives, kernel):
    solver = pnmol.white.LinearWhiteNoiseEK1(
        num_derivatives=num_derivatives,
        steprule=steprule,
        spatial_kernel=kernel + pnmol.kernels.WhiteNoise(),
    )

    return solver, pde_linear


# Semilinear PDEs


@pytest.fixture
def pde_semilinear(bcond, kernel):
    return pnmol.pde.examples.spruce_budworm_1d_discretized(
        tmax=1.0,
        dx=0.2,
        stencil_size=3,
        diffusion_rate=0.05,
        kernel=kernel,
        nugget_gram_matrix_fd=0.0,
        bcond=bcond,
    )


def case_semilinear_latent(pde_semilinear, steprule, num_derivatives, kernel):
    solver = pnmol.latent.SemiLinearLatentForceEK1(
        num_derivatives=num_derivatives,
        steprule=steprule,
        spatial_kernel=kernel + pnmol.kernels.WhiteNoise(),
    )
    return solver, pde_semilinear


def case_semilinear_white(pde_semilinear, steprule, num_derivatives, kernel):
    solver = pnmol.white.SemiLinearWhiteNoiseEK1(
        num_derivatives=num_derivatives,
        steprule=steprule,
        spatial_kernel=kernel + pnmol.kernels.WhiteNoise(),
    )
    return solver, pde_semilinear


# Systems of PDEs
@pytest.fixture
def pde_semilinear_system(bcond, kernel):
    return pnmol.pde.examples.sir_1d_discretized(
        kernel=kernel,
        nugget_gram_matrix_fd=0.0,
    )


# Currently, the SIR test fails because of concatenation issues in the solvers.
# Therefore, the case below is deactivated.
# To activate it, remove the "not_yet_a".
def not_yet_a_case_semilinear_system_latent(
    pde_semilinear_system, steprule, num_derivatives, kernel
):
    kernel = pnmol.kernels.duplicate(kernel + pnmol.kernels.WhiteNoise(), num=3)
    solver = pnmol.latent.SemiLinearLatentForceEK1(
        num_derivatives=num_derivatives,
        steprule=steprule,
        spatial_kernel=kernel,
    )
    return solver, pde_semilinear_system


# Currently, the SIR test fails because of concatenation issues in the solvers.
# Therefore, the case below is deactivated.
# To activate it, remove the "not_yet_a".
def not_yet_a_case_semilinear_system_white(
    pde_semilinear_system, steprule, num_derivatives, kernel
):
    kernel = pnmol.kernels.duplicate(kernel + pnmol.kernels.WhiteNoise(), num=3)
    solver = pnmol.white.SemiLinearWhiteNoiseEK1(
        num_derivatives=num_derivatives,
        steprule=steprule,
        spatial_kernel=kernel,
    )
    return solver, pde_semilinear_system


@pytest.mark.parametrize("bcond", ["dirichlet", "neumann"])
@parametrize_with_cases("solver,problem", cases=".")
def test_solve_no_nan(solver, problem):
    solution = solver.solve(problem)
    assert not jnp.any(jnp.isnan(solution.mean))
    assert not jnp.any(jnp.isnan(solution.cov_sqrtm))
