from functools import partial

import jax
import jax.numpy as jnp
from tornadox import iwp, odefilter, rv, sqrt


class MeasurementCovarianceEK0(odefilter.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.P1 = None

    def initialize(self, ivp):

        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )
        self.P0 = self.iwp.projection_matrix(0)
        self.P1 = self.iwp.projection_matrix(1)

        extended_dy0, cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.iwp.num_derivatives,
        )
        mean = extended_dy0  # .reshape((-1,), order="F")
        y = rv.MultivariateNormal(mean, jnp.kron(jnp.eye(ivp.dimension), cov_sqrtm))
        return odefilter.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt):
        # Extract system matrices
        P, Pinv = self.iwp.nordsieck_preconditioner(dt=dt)
        A, Q = self.iwp.preconditioned_discretize
        t = state.t + dt
        n, d = self.num_derivatives + 1, state.ivp.dimension

        # Pull states into preconditioned state
        m, C = Pinv @ state.y.mean.reshape((-1,), order="F"), Pinv @ state.y.cov

        cov, error_estimate, new_mean = self.attempt_unit_step(A, P, C, Q, m, state, t)

        # Push back to non-preconditioned state
        cov = P @ cov
        new_mean = P @ new_mean
        new_mean = new_mean.reshape((n, d), order="F")
        new_rv = rv.MultivariateNormal(new_mean, jnp.linalg.cholesky(cov))

        y1 = jnp.abs(state.y.mean[0])
        y2 = jnp.abs(new_mean[0])
        reference_state = jnp.maximum(y1, y2)

        # Return new state
        new_state = odefilter.ODEFilterState(
            ivp=state.ivp,
            t=t,
            y=new_rv,
            error_estimate=error_estimate,
            reference_state=reference_state,
        )
        info_dict = dict(num_f_evaluations=1, num_df_evaluations=1)
        return new_state, info_dict

    def attempt_unit_step(self, A, P, C, Q, m, state, t):
        m_pred = self.predict_mean(m=m, phi=A)
        H, z = self.evaluate_ode(
            t=t,
            f=state.ivp.f,
            df=state.ivp.df,
            p=P,
            m_pred=m_pred,
            e0=self.P0,
            e1=self.P1,
        )
        error_estimate, sigma = self.estimate_error(h=H, q=Q, z=z, e=state.ivp.E)
        C_pred = self.predict_cov(c=C, phi=A, q=sigma * Q)
        cov, Kgain = self.update(H, C_pred, E=state.ivp.E)
        new_mean = m_pred - Kgain @ z
        return cov, error_estimate, new_mean

    # Low level functions

    @staticmethod
    @jax.jit
    def update(H, C_pred, E):
        crosscov = C_pred @ H.T
        innov = H @ crosscov + E
        gain = jax.scipy.linalg.solve(innov.T, crosscov.T).T
        new_cov = C_pred - gain @ innov @ gain.T
        return new_cov, gain

    @staticmethod
    @jax.jit
    def predict_mean(m, phi):
        return phi @ m

    @staticmethod
    @jax.jit
    def predict_cov(c, phi, q):
        return phi @ c @ phi.T + q

    @staticmethod
    @partial(jax.jit, static_argnums=(1, 2))
    def evaluate_ode(t, f, df, p, m_pred, e0, e1):
        P0 = e0 @ p
        P1 = e1 @ p
        m_at = P0 @ m_pred
        fx = f(t, m_at)
        Jx = df(t, m_at)
        H = P1 - Jx @ P0
        b = -fx + Jx @ m_at
        z = H @ m_pred + b
        return H, z

    @staticmethod
    def estimate_error(h, q, z, e):
        S = h @ q @ h.T
        print(S)
        S = S + e
        print(e)
        print(S)
        print("---")
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error_estimate = jnp.sqrt(jnp.diag(S)) * sigma

        return error_estimate, sigma
