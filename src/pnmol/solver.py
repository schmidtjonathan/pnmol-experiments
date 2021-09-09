from functools import partial

import jax
import jax.numpy as jnp
from tornadox import iwp, odefilter, rv, sqrt


class MeasurementCovarianceEK0(odefilter.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None

    def initialize(self, ivp):

        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )

        self.P0 = self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)

        extended_dy0, cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        y = rv.MultivariateNormal(
            mean=extended_dy0, cov_sqrtm=jnp.kron(jnp.eye(ivp.dimension), cov_sqrtm)
        )
        return odefilter.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt, verbose=False):
        # [Setup]
        m, Cl = state.y.mean.reshape((-1,), order="F"), state.y.cov_sqrtm
        A, Ql = self.iwp.non_preconditioned_discretize(dt)
        n, d = self.num_derivatives + 1, state.ivp.dimension

        # [Predict]
        mp = self.predict_mean(A, m)

        # Measure / calibrate
        z, H = self.evaluate_ode(
            f=state.ivp.f, e0=self.E0, e1=self.E1, m_at=mp, t=state.t, dt=dt
        )

        sigma, error = self.estimate_error(ql=Ql, z=z, h=H)

        Clp = sqrt.propagate_cholesky_factor(A @ Cl, sigma * Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp, E=state.ivp.E)
        m_new = mp - K @ z

        m_new = m_new.reshape((n, d), order="F")
        y_new = jnp.abs(m_new[0])

        new_state = odefilter.ODEFilterState(
            ivp=state.ivp,
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
    def evaluate_ode(f, e0, e1, m_at, t, dt):
        z = e1 @ m_at - f(t + dt, e0 @ m_at)
        H = e1
        return z, H

    @staticmethod
    @jax.jit
    def estimate_error(ql, z, h):
        S = h @ ql @ ql.T @ h.T  # + state.ivp.E
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error = jnp.sqrt(jnp.diag(S)) * sigma
        return sigma, error
