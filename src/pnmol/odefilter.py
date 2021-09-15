"""ODE solver interface."""

import dataclasses
from abc import ABC, abstractmethod
from collections import namedtuple
from functools import partial
from typing import Dict, Iterable, Union

import jax
import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from pnmol import init, step

ODEFilterState = namedtuple("_ODEFilterState", "t y error_estimate reference_state")


@dataclasses.dataclass(frozen=False)
class ODESolution:
    t: jnp.ndarray
    mean: jnp.ndarray
    cov_sqrtm: jnp.ndarray
    info: Dict


class ODEFilter(ABC):
    """Interface for filtering-based ODE solvers in ProbNum."""

    def __init__(self, *, steprule=None, num_derivatives=4, initialization=None):

        # Step-size selection
        self.steprule = steprule or step.AdaptiveSteps()

        # Number of derivatives
        self.num_derivatives = num_derivatives

        # IWP(nu) prior -- will be assembled in initialize()
        self.iwp = None

        # Initialization strategy
        self.init = initialization or init.TaylorMode()

    def __repr__(self):
        return f"{self.__class__.__name__}(num_derivatives={self.num_derivatives}, steprule={self.steprule}, initialization={self.init})"

    def solve(self, *args, **kwargs):
        solution_generator = self.solution_generator(*args, **kwargs)
        means = []
        cov_sqrtms = []
        times = []
        info = dict()

        for state, info in solution_generator:
            times.append(state.t)
            means.append(state.y.mean)
            cov_sqrtms.append(state.y.cov_sqrtm)

        return ODESolution(
            t=jnp.stack(times),
            mean=jnp.stack(means),
            cov_sqrtm=jnp.stack(cov_sqrtms),
            info=info,
        )

    def simulate_final_state(self, *args, **kwargs):
        solution_generator = self.solution_generator(*args, **kwargs)
        state, info = None, None
        for state, info in solution_generator:
            pass
        return state, info

    def solution_generator(self, discretized_pde, stop_at=None, progressbar=False):
        """Generate ODE solver steps."""

        time_stopper = self._process_event_inputs(stop_at_locations=stop_at)
        state = self.initialize(discretized_pde)
        info = dict(
            num_f_evaluations=0,
            num_df_evaluations=0,
            num_df_diagonal_evaluations=0,
            num_steps=0,
            num_attempted_steps=0,
        )
        yield state, info

        dt = self.steprule.first_dt(discretized_pde)

        progressbar_steps = 100
        progressbar_update_threshold = progressbar_update_increment = (
            discretized_pde.tmax / progressbar_steps
        )
        pbar = tqdm(total=progressbar_steps) if progressbar else None
        while state.t < discretized_pde.tmax:

            if pbar is not None:
                while state.t + dt >= progressbar_update_threshold:
                    pbar.update()
                    progressbar_update_threshold += progressbar_update_increment
                pbar.set_description(f"t={state.t:.4f}, dt={dt:.2E}")

            if time_stopper is not None:
                dt = time_stopper.adjust_dt_to_time_stops(state.t, dt)

            state, dt, step_info = self.perform_full_step(state, dt, discretized_pde)

            # if dt < self.steprule.min_step:
            #     raise ValueError(
            #         f"Step-size ({dt}) smaller than minimum step-size ({self.steprule.min_step})"
            #     )
            # if dt > self.steprule.max_step:
            #     raise ValueError(
            #         f"Step-size ({dt}) larger than maximum step-size ({self.steprule.max_step})"
            #     )

            info["num_steps"] += 1
            info["num_f_evaluations"] += step_info["num_f_evaluations"]
            info["num_df_evaluations"] += step_info["num_df_evaluations"]
            info["num_df_diagonal_evaluations"] += step_info[
                "num_df_diagonal_evaluations"
            ]
            info["num_attempted_steps"] += step_info["num_attempted_steps"]
            yield state, info
        if pbar is not None:
            pbar.update()
            pbar.close()

    @staticmethod
    def _process_event_inputs(stop_at_locations):
        """Process callbacks and time-stamps into a format suitable for solve()."""

        if stop_at_locations is not None:
            time_stopper = _TimeStopper(stop_at_locations)
        else:
            time_stopper = None
        return time_stopper

    def perform_full_step(self, state, initial_dt, discretized_pde):
        """Perform a full ODE solver step.

        This includes the acceptance/rejection decision as governed by error estimation
        and steprule.
        """
        dt = initial_dt
        step_is_sufficiently_small = False
        proposed_state = None
        step_info = dict(
            num_f_evaluations=0,
            num_df_evaluations=0,
            num_df_diagonal_evaluations=0,
            num_attempted_steps=0,
        )
        while not step_is_sufficiently_small:

            proposed_state, attempt_step_info = self.attempt_step(
                state, dt, discretized_pde
            )

            # Gather some stats
            step_info["num_attempted_steps"] += 1
            if "num_f_evaluations" in attempt_step_info:
                nfevals = attempt_step_info["num_f_evaluations"]
                step_info["num_f_evaluations"] += nfevals
            if "num_df_evaluations" in attempt_step_info:
                ndfevals = attempt_step_info["num_df_evaluations"]
                step_info["num_df_evaluations"] += ndfevals
            if "num_df_diagonal_evaluations" in attempt_step_info:
                ndfevals_diag = attempt_step_info["num_df_diagonal_evaluations"]
                step_info["num_df_diagonal_evaluations"] += ndfevals_diag

            # Acceptance/Rejection due to the step-rule
            internal_norm = self.steprule.scale_error_estimate(
                unscaled_error_estimate=dt * proposed_state.error_estimate
                if proposed_state.error_estimate is not None
                else None,
                reference_state=proposed_state.reference_state,
            )
            step_is_sufficiently_small = self.steprule.is_accepted(internal_norm)
            suggested_dt = self.steprule.suggest(
                dt, internal_norm, local_convergence_rate=self.num_derivatives + 1
            )
            # Get a new step-size for the next step
            if step_is_sufficiently_small:
                dt = min(suggested_dt, discretized_pde.tmax - proposed_state.t)
            else:
                dt = min(suggested_dt, discretized_pde.tmax - state.t)

            assert dt >= 0, f"Invalid step size: dt={dt}"

        return proposed_state, dt, step_info

    @abstractmethod
    def initialize(self, discretized_pde):
        raise NotImplementedError

    @abstractmethod
    def attempt_step(self, state, dt, discretized_pde):
        raise NotImplementedError


class _TimeStopper:
    """Make the ODE solver stop at specified time-points."""

    def __init__(self, locations: Iterable):
        self._locations = iter(locations)
        self._next_location = next(self._locations)

    def adjust_dt_to_time_stops(self, t, dt):
        """Check whether the next time-point is supposed to be stopped at."""

        if t >= self._next_location:
            try:
                self._next_location = next(self._locations)
            except StopIteration:
                self._next_location = np.inf

        if t + dt > self._next_location:
            dt = self._next_location - t
        return dt