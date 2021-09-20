from functools import partial

import jax
import jax.numpy as jnp
from tornadox import iwp, odefilter, rv, sqrt

from pnmol import stacked_ssm


class LatentForceEK0(odefilter.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None

    def initialize(self, ivp):

        self.state_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )
        self.lf_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=0, wiener_process_dimension=ivp.dimension
        )
        self.ssm = stacked_ssm.StackedSSM(processes=[self.state_iwp, self.lf_iwp])

        self.P0 = self.E0 = self.ssm.projection_matrix(0)
        self.E1 = self.ssm.projection_matrix(1)

        extended_dy0, state_cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.state_iwp.num_derivatives,
        )
        mean = jnp.concatenate((extended_dy0, jnp.zeros_like(extended_dy0[0:1, :])))

        cov_sqrtm = jax.scipy.linalg.block_diag(
            jnp.kron(jnp.eye(ivp.dimension), state_cov_sqrtm), jnp.sqrt(jnp.abs(ivp.E))
        )
        y = rv.MultivariateNormal(mean=mean, cov_sqrtm=cov_sqrtm)

        assert mean.size == cov_sqrtm.shape[0] == cov_sqrtm.shape[1]

        return odefilter.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt, verbose=False):
        P, Pinv = self.ssm.nordsieck_preconditioner(dt=dt)
        A, Ql = self.ssm.preconditioned_discretize
        n, d = self.num_derivatives + 1, state.ivp.dimension

        # [Setup]
        # Pull states into preconditioned state
        state_mean, eps_mean = jnp.split(state.y.mean, (n,), axis=0)
        flattened_mean = jnp.concatenate(
            (state_mean.reshape((-1,), order="F"), eps_mean.reshape((-1,), order="F"))
        )
        m, Cl = Pinv @ flattened_mean, Pinv @ state.y.cov_sqrtm

        # [Predict]
        mp = self.predict_mean(A, m)

        # Measure / calibrate
        z, H = self.evaluate_ode(
            f=state.ivp.f, p0=self.E0 @ P, p1=self.E1 @ P, m_pred=mp, t=state.t + dt
        )

        sigma, error = self.estimate_error(ql=Ql, z=z, h=H)

        Clp = sqrt.propagate_cholesky_factor(A @ Cl, sigma * Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp)
        m_new = mp - K @ z

        # Push back to non-preconditioned state
        Cl_new = P @ Cl_new
        m_new = P @ m_new

        state_m_new, eps_m_new = jnp.split(m_new, (n * d,))
        state_m_new = state_m_new.reshape((n, d), order="F")
        eps_m_new = eps_m_new.reshape(
            (1, state.ivp.dimension), order="F"
        )  # TODO: works only if num_derivatives = 0 for epsilon process

        m_new = jnp.concatenate((state_m_new, eps_m_new), axis=0)
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
    def evaluate_ode(f, p0, p1, m_pred, t):
        m_at = p0 @ m_pred
        state_at, eps_at = jnp.split(m_at, 2)
        fx = f(t, state_at)
        H = jnp.split(p1, 2, axis=0)[0]
        z = H @ m_pred - fx - eps_at
        return z, H

    @staticmethod
    @jax.jit
    def estimate_error(ql, z, h):
        S = h @ ql @ ql.T @ h.T
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]  # TODO <--- correct?
        sigma = jnp.sqrt(sigma_squared)
        error = jnp.sqrt(jnp.diag(S)) * sigma
        return sigma, error
