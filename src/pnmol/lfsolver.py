from collections import namedtuple
from functools import partial

import jax
import jax.numpy as jnp

from pnmol import iwp, odefilter, rv, sqrt, stacked_ssm


class LatentForceEK1(odefilter.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None

    def initialize(self, ivp):

        self.state_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
            wp_diffusion_sqrtm=ivp.Kxx_sqrtm,
        )
        self.lf_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=ivp.dimension,
            wp_diffusion_sqrtm=ivp.E_sqrtm,
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
        mean = jnp.concatenate([extended_dy0, jnp.zeros_like(extended_dy0)], -1)

        cov_sqrtm_state = jnp.kron(ivp.Kxx_sqrtm, cov_sqrtm_state)
        cov_sqrtm_eps = jnp.kron(ivp.E_sqrtm, 1e-10 * jnp.eye(self.num_derivatives + 1))

        cov_sqrtm = jax.scipy.linalg.block_diag(
            cov_sqrtm_state,
            cov_sqrtm_eps,
        )

        y = rv.MultivariateNormal(mean=mean, cov_sqrtm=cov_sqrtm)

        return odefilter.ODEFilterState(
            t=ivp.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt, discretized_pde):
        A, Ql = self.ssm.non_preconditioned_discretize(dt)
        n, d = self.num_derivatives + 1, self.state_iwp.wiener_process_dimension

        # [Predict]
        glued_batched_mean = state.y.mean
        batched_state_mean, batched_eps_mean = jnp.split(glued_batched_mean, 2, axis=-1)
        assert batched_state_mean.shape == batched_eps_mean.shape
        flat_state_mean = batched_state_mean.reshape((-1,), order="F")
        flat_eps_mean = batched_eps_mean.reshape((-1,), order="F")
        glued_flat_mean = jnp.concatenate((flat_state_mean, flat_eps_mean))
        mp = self.predict_mean(A, glued_flat_mean)

        # Measure / calibrate
        z, H = self.evaluate_ode(
            f=discretized_pde.f,
            df=discretized_pde.df,
            p0=self.E0,
            p1=self.E1,
            m_pred=mp,
            t=state.t + dt,
            B=discretized_pde.spatial_grid.boundary_projection_matrix,
        )

        Cl = state.y.cov_sqrtm
        Clp = sqrt.propagate_cholesky_factor(A @ Cl, Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(
            H, Clp, meascov_sqrtm=jnp.zeros((H.shape[0], H.shape[0]))
        )
        flat_m_new = mp - K @ z

        flat_state_m_new, flat_eps_m_new = jnp.split(flat_m_new, 2)
        batched_state_m_new = flat_state_m_new.reshape((n, d), order="F")
        batched_eps_m_new = flat_eps_m_new.reshape((n, d), order="F")

        glued_new_mean = jnp.concatenate([batched_state_m_new, batched_eps_m_new], -1)

        new_state = odefilter.ODEFilterState(
            t=state.t + dt,
            error_estimate=None,
            reference_state=None,
            y=rv.MultivariateNormal(glued_new_mean, Cl_new),
        )
        info_dict = dict(num_f_evaluations=1)
        return new_state, info_dict

    @staticmethod
    @jax.jit
    def predict_mean(A, m):
        return A @ m

    @staticmethod
    @partial(jax.jit, static_argnums=1)
    def extract_blocks_from_block_diag(block_diag_mat, num_blocks):
        """ATTENTION: ASSUMES EQUAL-SIZED SQUARE BLOCKS!"""
        block_rows = jnp.split(block_diag_mat, num_blocks, axis=0)
        return [
            jnp.split(
                block_rows[i],
                num_blocks,
                axis=1,
            )[i]
            for i in range(num_blocks)
        ]

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1))
    def evaluate_ode(f, df, p0, p1, m_pred, t, B):

        m_at = p0 @ m_pred  # Project to first derivatives
        state_at, eps_at = jnp.split(m_at, 2)  # Split up into ODE state and error

        fx = f(t, state_at)  # Evaluate vector field
        Jx = df(t, state_at)  # Evaluate Jacobian of the vector field

        E0_state, E0_eps = LatentForceEK1.extract_blocks_from_block_diag(
            p0, num_blocks=2
        )
        E1_state, E1_eps = LatentForceEK1.extract_blocks_from_block_diag(
            p1, num_blocks=2
        )

        H_state = E1_state - Jx @ E0_state
        H_eps = -E0_eps
        H_boundaries = B @ E0_state
        H_zeros = jnp.zeros_like(H_boundaries)
        H = jnp.block([[H_state, H_eps], [H_boundaries, H_zeros]])

        zeros_bc = jnp.zeros((B.shape[0],))

        b = jnp.concatenate([Jx @ state_at - fx, zeros_bc])
        z = H @ m_pred + b
        return z, H
