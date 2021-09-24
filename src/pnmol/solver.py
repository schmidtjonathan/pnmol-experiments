from functools import partial

import jax
import jax.numpy as jnp
from tornadox import iwp, odefilter, rv

from pnmol import sqrt


class MyODEFilter(odefilter.ODEFilter):
    def perform_full_step(
        self, state, initial_dt, f, spatial_grid, t0, tmax, y0, df, df_diagonal, L, E
    ):
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
                state, dt, f, spatial_grid, t0, tmax, y0, df, df_diagonal, L, E
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
                dt = min(suggested_dt, tmax - proposed_state.t)
            else:
                dt = min(suggested_dt, tmax - state.t)

            assert dt >= 0, f"Invalid step size: dt={dt}"

        return proposed_state, dt, step_info


class MeasurementCovarianceEK0(MyODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None

    def initialize(self, f, spatial_grid, t0, tmax, y0, df, df_diagonal, L, E):

        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=y0.shape[0],
        )

        self.P0 = self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)

        extended_dy0, cov_sqrtm = self.init(
            f=f,
            df=df,
            y0=y0,
            t0=t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        y = rv.MultivariateNormal(
            mean=extended_dy0, cov_sqrtm=jnp.kron(jnp.eye(y0.shape[0]), cov_sqrtm)
        )
        return odefilter.ODEFilterState(
            t=t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(
        self, state, dt, f, spatial_grid, t0, tmax, y0, df, df_diagonal, L, E
    ):
        P, Pinv = self.iwp.nordsieck_preconditioner(dt=dt)
        A, Ql = self.iwp.preconditioned_discretize
        n, d = self.num_derivatives + 1, y0.shape[0]
        B = spatial_grid.boundary_projection_matrix

        # [Setup]
        # Pull states into preconditioned state
        m, Cl = Pinv @ state.y.mean.reshape((-1,), order="F"), Pinv @ state.y.cov_sqrtm

        # [Predict]
        mp = self.predict_mean(A, m)

        # Measure / calibrate
        z, H = self.evaluate_ode(
            f=f, p0=self.E0 @ P, p1=self.E1 @ P, m_pred=mp, t=state.t + dt, B=B
        )
        E_with_bc = jax.scipy.linalg.block_diag(E, jnp.zeros((2, 2)))

        sigma, error = self.estimate_error(ql=Ql, z=z, h=H, E=E_with_bc)

        Clp = sqrt.propagate_cholesky_factor(A @ Cl, sigma * Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp, E=E_with_bc)
        m_new = mp - K @ z

        # Push back to non-preconditioned state
        Cl_new = P @ Cl_new
        m_new = P @ m_new

        m_new = m_new.reshape((n, d), order="F")
        y_new = jnp.abs(m_new[0])

        new_state = odefilter.ODEFilterState(
            t=state.t + dt,
            error_estimate=error,
            reference_state=y_new,
            y=rv.MultivariateNormal(m_new, Cl_new),
        )
        info_dict = dict(num_f_evaluations=1)
        return new_state, info_dict

    @staticmethod
    @jax.jit
    def predict_mean(A, m):
        return A @ m

    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def evaluate_ode(f, p0, p1, m_pred, t, B):
        m_at = p0 @ m_pred
        fx = f(t, m_at)

        H = jnp.vstack((p1, B @ p0))
        shift = jnp.hstack((fx, jnp.zeros(B.shape[0])))
        z = H @ m_pred - shift
        return z, H

    @staticmethod
    @jax.jit
    def estimate_error(ql, z, h, E):
        S = h @ ql @ ql.T @ h.T + jnp.abs(E)
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error = jnp.sqrt(jnp.diag(S)) * sigma
        return sigma, error


class MeasurementCovarianceEK1(MyODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None

    def initialize(self, f, spatial_grid, t0, tmax, y0, df, df_diagonal, L, E):

        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=y0.shape[0],
        )

        self.P0 = self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)

        extended_dy0, cov_sqrtm = self.init(
            f=f,
            df=df,
            y0=y0,
            t0=t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        y = rv.MultivariateNormal(
            mean=extended_dy0, cov_sqrtm=jnp.kron(jnp.eye(y0.shape[0]), cov_sqrtm)
        )
        return odefilter.ODEFilterState(
            t=t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(
        self, state, dt, f, spatial_grid, t0, tmax, y0, df, df_diagonal, L, E
    ):
        P, Pinv = self.iwp.nordsieck_preconditioner(dt=dt)
        A, Ql = self.iwp.preconditioned_discretize
        n, d = self.num_derivatives + 1, y0.shape[0]
        B = spatial_grid.boundary_projection_matrix

        # [Setup]
        # Pull states into preconditioned state
        m, Cl = Pinv @ state.y.mean.reshape((-1,), order="F"), Pinv @ state.y.cov_sqrtm

        # [Predict]
        mp = self.predict_mean(A, m)

        # Measure / calibrate
        z, H = self.evaluate_ode(
            f=f,
            df=df,
            p0=self.E0 @ P,
            p1=self.E1 @ P,
            m_pred=mp,
            t=state.t + dt,
            B=B,
        )
        E_with_bc = jax.scipy.linalg.block_diag(E, jnp.zeros((2, 2)))

        sigma, error = self.estimate_error(ql=Ql, z=z, h=H, E=E_with_bc)

        Clp = sqrt.propagate_cholesky_factor(A @ Cl, sigma * Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp, E=E_with_bc)
        m_new = mp - K @ z

        # Push back to non-preconditioned state
        Cl_new = P @ Cl_new
        m_new = P @ m_new

        m_new = m_new.reshape((n, d), order="F")
        y_new = jnp.abs(m_new[0])

        new_state = odefilter.ODEFilterState(
            t=state.t + dt,
            error_estimate=error,
            reference_state=y_new,
            y=rv.MultivariateNormal(m_new, Cl_new),
        )
        info_dict = dict(num_f_evaluations=1)
        return new_state, info_dict

    @staticmethod
    @jax.jit
    def predict_mean(A, m):
        return A @ m

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def evaluate_ode(f, df, p0, p1, m_pred, t, B):
        m_at = p0 @ m_pred
        fx = f(t, m_at)
        Jx = df(t, m_at)
        b = Jx @ m_at - fx

        H_ode = p1 - Jx @ p0
        H = jnp.vstack((H_ode, B @ p0))

        shift = jnp.hstack((b, jnp.zeros(B.shape[0])))

        z = H @ m_pred + shift
        return z, H

    @staticmethod
    @jax.jit
    def estimate_error(ql, z, h, E):
        S = h @ ql @ ql.T @ h.T + jnp.abs(E)
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error = jnp.sqrt(jnp.diag(S)) * sigma
        return sigma, error
