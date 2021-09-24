"""Tests for ODEFilter interfaces."""

from collections import namedtuple

import jax
import jax.numpy as jnp
import pytest

import pnmol


class EulerState(namedtuple("_EulerState", "t y error_estimate reference_state")):
    pass


class EulerAsODEFilter(pnmol.odefilter.ODEFilter):
    def initialize(self, ivp):
        y = pnmol.rv.MultivariateNormal(
            ivp.y0, cov_sqrtm=jnp.zeros((ivp.y0.shape[0], ivp.y0.shape[0]))
        )
        return EulerState(
            y=y, t=ivp.t0, error_estimate=jnp.nan * ivp.y0, reference_state=ivp.y0
        )

    def attempt_step(self, state, dt, ivp):
        y = state.y.mean + dt * ivp.f(state.t, state.y.mean)
        t = state.t + dt
        y = pnmol.rv.MultivariateNormal(
            y, cov_sqrtm=jnp.zeros((y.shape[0], y.shape[0]))
        )
        new_state = EulerState(
            y=y, t=t, error_estimate=jnp.nan * ivp.y0, reference_state=y.mean
        )
        return new_state, {}


@pytest.fixture
def ivp():
    ivp = pnmol.pde_problems.heat_1d(t0=0.0, tmax=1.5)
    return ivp


@pytest.fixture
def steps():
    return pnmol.step.ConstantSteps(dt=0.1)


@pytest.fixture
def solver(steps):
    solver_order = 1
    solver = EulerAsODEFilter(
        steprule=steps,
        num_derivatives=solver_order,
    )
    return solver


def test_simulate_final_point(ivp, solver):
    sol, _ = solver.simulate_final_state(ivp)
    assert isinstance(sol, EulerState)


def test_solve(ivp, solver):
    sol = solver.solve(ivp)
    assert isinstance(sol, pnmol.odefilter.ODESolution)


@pytest.fixture
def locations():
    return jnp.array([1.234])


def test_solve_stop_at(ivp, solver, locations):
    sol = solver.solve(ivp, stop_at=locations)
    assert jnp.isin(locations[0], jnp.array(sol.t))


def test_odefilter_state_jittable(ivp):
    def fun(state):
        t, y, err, ref = state
        return pnmol.odefilter.ODEFilterState(t, y, err, ref)

    fun_jitted = jax.jit(fun)
    x = jnp.zeros(3)
    state = pnmol.odefilter.ODEFilterState(
        t=0, y=x, error_estimate=x, reference_state=x
    )
    out = fun_jitted(state)
    assert type(out) == type(state)
