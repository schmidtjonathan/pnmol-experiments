from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp
from tornadox import iwp, odefilter, rv, sqrt

from pnmol import stacked_ssm


class StackedMultivariateNormal(
    namedtuple("_StackedMultivariateNormal", "mean cov_sqrtm")
):
    @property
    def cov(self):
        return self.cov_sqrtm @ self.cov_sqrtm.T


class LatentForceEK1(odefilter.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None

        self.num_derivatives_eps = 2  # TODO provide interface

    def initialize(self, ivp):

        self.state_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )
        self.lf_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives_eps,
            wiener_process_dimension=ivp.dimension,
            scale_process_noise=0.1,
        )
        self.ssm = stacked_ssm.StackedSSM(processes=[self.state_iwp, self.lf_iwp])

        self.P0 = self.E0 = self.ssm.projection_matrix(0)
        self.E1 = self.ssm.projection_matrix(1)

        extended_dy0, cov_sqrtm_state = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.state_iwp.num_derivatives,
        )
        mean = [extended_dy0, jnp.zeros((self.num_derivatives_eps + 1, ivp.dimension))]

        cov_sqrtm_state = jnp.kron(jnp.eye(ivp.dimension), cov_sqrtm_state)

        # Initialize the covariance of the Error process
        # with E (state) and zeros (derivatives)
        left_cov_factor = jnp.zeros(
            (self.num_derivatives_eps + 1, self.num_derivatives_eps + 1)
        )
        left_cov_factor = jax.ops.index_update(left_cov_factor, jnp.array([0, 0]), 1.0)
        cov_sqrtm_eps = jnp.kron(left_cov_factor, jnp.sqrt(jnp.abs(ivp.E)))

        cov_sqrtm = jax.scipy.linalg.block_diag(
            cov_sqrtm_state,
            cov_sqrtm_eps,
        )

        y = StackedMultivariateNormal(mean=mean, cov_sqrtm=cov_sqrtm)

        assert sum(m.size for m in mean) == cov_sqrtm.shape[0] == cov_sqrtm.shape[1]

        return odefilter.ODEFilterState(
            ivp=ivp,
            t=ivp.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt):
        A, Ql = self.ssm.non_preconditioned_discretize(dt)
        n, d = self.num_derivatives + 1, state.ivp.dimension
        n_eps = self.num_derivatives_eps + 1

        # [Predict]
        mp = self.predict_mean(A, state.y.mean)

        # Measure / calibrate
        z, H = self.evaluate_ode(
            f=state.ivp.f,
            df=state.ivp.df,
            p0=self.E0,
            p1=self.E1,
            m_pred=mp,
            t=state.t + dt,
        )

        sigma, error = self.estimate_error(ql=Ql, z=z, h=H)

        Cl = state.y.cov_sqrtm
        block_diag_A = jax.scipy.linalg.block_diag(*A)
        Clp = sqrt.propagate_cholesky_factor(block_diag_A @ Cl, sigma * Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp)
        flattened_mp = jnp.concatenate(mp)
        flattened_m_new = flattened_mp - K @ z

        state_m_new, eps_m_new = jnp.split(flattened_m_new, (n * d,))
        state_m_new = state_m_new.reshape((n, d), order="F")
        eps_m_new = eps_m_new.reshape((n_eps, state.ivp.dimension), order="F")

        m_new = [state_m_new, eps_m_new]
        y_new = jnp.abs(m_new[0])

        new_state = odefilter.ODEFilterState(
            ivp=state.ivp,
            t=state.t + dt,
            error_estimate=error,
            reference_state=y_new,
            y=StackedMultivariateNormal(m_new, Cl_new),
        )
        info_dict = dict(num_f_evaluations=1)
        return new_state, info_dict

    @staticmethod
    @jax.jit
    def predict_mean(A, m):
        return [A[i] @ m[i].reshape((-1,), order="F") for i in range(len(A))]

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def evaluate_ode(f, df, p0, p1, m_pred, t):
        state_at = p0[0] @ m_pred[0]
        eps_at = p0[1] @ m_pred[1]
        fx = f(t, state_at)
        Jx = df(t, state_at)
        H_state = p1[0] - Jx @ p0[0]
        H_eps = -p0[1]

        H = jnp.concatenate((H_state, H_eps), -1)
        b = Jx @ state_at - fx - eps_at
        z = H @ jnp.concatenate(m_pred) + b
        return z, H

    @staticmethod
    @jax.jit
    def estimate_error(ql, z, h):
        S = h @ ql @ ql.T @ h.T
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]  # TODO <--- correct?
        sigma = jnp.sqrt(sigma_squared)
        error = jnp.sqrt(jnp.diag(S)) * sigma
        return sigma, error
