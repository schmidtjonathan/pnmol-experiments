"""ODE solver interface."""

import dataclasses
from abc import ABC, abstractmethod
from collections import namedtuple
from typing import Dict, Iterable, Union

import jax.numpy as jnp
import numpy as np
from tqdm import tqdm

from pnmol import kernels
from pnmol.ode import init, step


class PDEFilterState(
    namedtuple("_", "t y error_estimate reference_state diffusion_squared_local")
):
    """PDE filter state."""

    pass


@dataclasses.dataclass(frozen=False)
class PDESolution:
    t: jnp.ndarray
    mean: jnp.ndarray
    cov_sqrtm: jnp.ndarray
    info: Dict
    diffusion_squared_calibrated: float


class PDEFilter(ABC):
    """Interface for filtering-based ODE solvers in ProbNum."""

    def __init__(
        self,
        *,
        steprule=None,
        num_derivatives=2,
        initialization=None,
        spatial_kernel=None,
    ):

        # Step-size selection
        self.steprule = steprule or step.Adaptive()

        # Number of derivatives
        self.num_derivatives = num_derivatives

        # IWP(nu) prior -- will be assembled in initialize()
        self.iwp = None

        # Initialization strategy
        self.init = initialization or init.RungeKutta()

        # Spatial covariance kernel
        self.spatial_kernel = (
            spatial_kernel or kernels.Matern52() + kernels.WhiteNoise()
        )

    def __repr__(self):
        return f"{self.__class__.__name__}(num_derivatives={self.num_derivatives}, steprule={self.steprule}, initialization={self.init})"

    def solve(self, *args, **kwargs):
        solution_generator = self.solution_generator(*args, **kwargs)
        means = []
        cov_sqrtms = []
        times = []
        info = dict()

        diffusion_squared_list = []

        for state, info in solution_generator:
            times.append(state.t)
            means.append(state.y.mean)
            cov_sqrtms.append(state.y.cov_sqrtm)
            diffusion_squared_list.append(state.diffusion_squared_local)

        diffusion_squared_calibrated = jnp.mean(jnp.array(diffusion_squared_list))

        return PDESolution(
            t=jnp.stack(times),
            mean=jnp.stack(means),
            cov_sqrtm=jnp.stack(cov_sqrtms),
            info=info,
            diffusion_squared_calibrated=diffusion_squared_calibrated,
        )

    def simulate_final_state(self, *args, **kwargs):
        solution_generator = self.solution_generator(*args, **kwargs)
        state, info = None, None
        diffusion_squared_list = []
        for state, info in solution_generator:
            diffusion_squared_list.append(state.diffusion_squared_local)
            pass
        diffusion_squared_calibrated = jnp.mean(jnp.array(diffusion_squared_list))
        cov_sqrtm_new = state.cov_sqrtm * jnp.sqrt(diffusion_squared_calibrated)
        return state._replace(cov_sqrtm=cov_sqrtm_new), info

    def solution_generator(self, pde, /, *, stop_at=None, progressbar=False):
        """Generate ODE solver steps."""

        time_stopper = self._process_event_inputs(stop_at_locations=stop_at)
        state = self.initialize(pde)
        info = dict(
            num_f_evaluations=0,
            num_df_evaluations=0,
            num_df_diagonal_evaluations=0,
            num_steps=0,
            num_attempted_steps=0,
        )
        yield state, info

        dt = self.steprule.first_dt(pde)

        progressbar_steps = 100
        progressbar_update_threshold = progressbar_update_increment = (
            pde.tmax / progressbar_steps
        )
        pbar = tqdm(total=progressbar_steps) if progressbar else None

        while state.t < pde.tmax:

            if pbar is not None:
                while state.t + dt >= progressbar_update_threshold:
                    pbar.update()
                    progressbar_update_threshold += progressbar_update_increment
                pbar.set_description(f"t={state.t:.4f}, dt={dt:.2E}")

            if time_stopper is not None:
                dt = time_stopper.adjust_dt_to_time_stops(state.t, dt)

            state, dt, step_info = self.perform_full_step(state, dt, pde)

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

    def perform_full_step(self, state, initial_dt, pde):
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

            proposed_state, attempt_step_info = self.attempt_step(state, dt, pde)

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
                dt = min(suggested_dt, pde.tmax - proposed_state.t)
            else:
                dt = min(suggested_dt, pde.tmax - state.t)

            assert dt >= 0, f"Invalid step size: dt={dt}"

        return proposed_state, dt, step_info

    @abstractmethod
    def initialize(self, pde):
        raise NotImplementedError

    @abstractmethod
    def attempt_step(self, state, dt, pde):
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
