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
        self.W = kwargs["W"]

        self.num_derivatives_eps = self.num_derivatives

    def initialize(self, ivp):

        self.state_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
        )
        self.lf_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives_eps,
            wiener_process_dimension=ivp.dimension,
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

        cov_sqrtm_eps = jnp.kron(
            jnp.sqrt(ivp.E), 1.0 / 10000.0 * jnp.eye(cov_sqrtm_state.shape[0])
        )
        cov_sqrtm_state = jnp.kron(jnp.eye(ivp.dimension), cov_sqrtm_state)

        # print(cov_sqrtm_eps.shape, cov_sqrtm_state.shape)

        # Initialize the covariance of the Error process
        # with E (state) and zeros (derivatives)
        # left_cov_factor = jnp.zeros(
        #     (self.num_derivatives_eps + 1, self.num_derivatives_eps + 1)
        # )
        # left_cov_factor = jax.ops.index_update(left_cov_factor, jnp.array([0, 0]), 1.0)
        # cov_sqrtm_eps = jnp.kron(left_cov_factor, jnp.sqrt(jnp.abs(ivp.E)))

        cov_sqrtm = jax.scipy.linalg.block_diag(
            cov_sqrtm_state,
            cov_sqrtm_eps,
        )

        y = StackedMultivariateNormal(mean=mean, cov_sqrtm=cov_sqrtm)

        return odefilter.ODEFilterState(
            t=ivp.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt, f, t0, tmax, y0, df, df_diagonal):
        A, Ql = self.ssm.non_preconditioned_discretize(dt)
        n, d = self.num_derivatives + 1, self.state_iwp.wiener_process_dimension
        n_eps = self.num_derivatives_eps + 1

        # [Predict]
        mp = self.predict_mean(A, state.y.mean)

        # Measure / calibrate
        z, H = self.evaluate_ode(
            f=f, df=df, p0=self.E0, p1=self.E1, m_pred=mp, t=state.t + dt, W=self.W
        )

        # meascov_sqrtm = jnp.sqrt(1e-1) * jnp.eye(d)
        # sigma, error = self.estimate_error(
        #     ql=Ql, z=z, h=H  # , meascov=meascov_sqrtm @ meascov_sqrtm.T
        # )

        Cl = state.y.cov_sqrtm
        block_diag_A = jax.scipy.linalg.block_diag(*A)
        Clp = sqrt.propagate_cholesky_factor(block_diag_A @ Cl, Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp)  # , meascov_sqrtm=meascov_sqrtm)
        flattened_mp = jnp.concatenate(mp)
        flattened_m_new = flattened_mp - K @ z

        state_m_new, eps_m_new = jnp.split(flattened_m_new, (n * d,))
        state_m_new = state_m_new.reshape((n, d), order="F")
        eps_m_new = eps_m_new.reshape((n_eps, d), order="F")

        new_mean = [state_m_new, eps_m_new]

        y1 = jnp.abs(state.y.mean[0][0])
        y2 = jnp.abs(state_m_new[0])
        reference_state = jnp.maximum(y1, y2)

        new_state = odefilter.ODEFilterState(
            t=state.t + dt,
            error_estimate=None,
            reference_state=reference_state,
            y=StackedMultivariateNormal(new_mean, Cl_new),
        )
        info_dict = dict(num_f_evaluations=1)
        return new_state, info_dict

    @staticmethod
    @jax.jit
    def predict_mean(A, m):
        return [A[i] @ m[i].reshape((-1,), order="F") for i in range(len(A))]

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def evaluate_ode(f, df, p0, p1, m_pred, t, W):
        state_at = p0[0] @ m_pred[0]
        eps_at = p0[1] @ m_pred[1]
        fx = f(t, state_at)
        Jx = df(t, state_at)

        print("W", W.shape)
        print("E0", p0[0].shape)

        H_state = p1[0] - Jx @ p0[0]
        H_eps = -p0[1]
        H_boundaries = W @ p0[0]
        H_zeros = jnp.zeros_like(H_boundaries)
        H = jnp.block([[H_state, H_eps], [H_boundaries, H_zeros]])

        zeros_bc = jnp.zeros((W.shape[0],))

        b = jnp.concatenate([Jx @ state_at - fx, zeros_bc])
        print(H.shape, b.shape)
        z = H @ jnp.concatenate(m_pred) + b
        print(z.shape)
        return z, H

    @staticmethod
    @jax.jit
    def estimate_error(ql, z, h, meascov=None):
        S = h @ ql @ ql.T @ h.T
        if meascov is not None:
            S = S + meascov
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]  # TODO <--- correct?
        sigma = jnp.sqrt(sigma_squared)
        error = jnp.sqrt(jnp.diag(S)) * sigma
        return sigma, error
