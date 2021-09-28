import abc
from functools import partial

import jax
import jax.numpy as jnp

from pnmol import iwp, odefilter, rv, sqrt, stacked_ssm


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


class LatentForceEK1Base(odefilter.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None

    def initialize(self, discretized_pde):

        X = discretized_pde.spatial_grid.points
        diffusion_state_sqrtm = jnp.linalg.cholesky(self.spatial_kernel(X, X.T))

        self.state_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=discretized_pde.dimension,
            wp_diffusion_sqrtm=discretized_pde.Kxx_sqrtm,
        )
        self.lf_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=discretized_pde.dimension,
            wp_diffusion_sqrtm=discretized_pde.E_sqrtm,
        )
        self.ssm = stacked_ssm.StackedSSM(processes=[self.state_iwp, self.lf_iwp])

        self.P0 = self.E0 = self.state_iwp.projection_matrix(0)
        self.E1 = self.state_iwp.projection_matrix(1)

        extended_dy0, cov_sqrtm_state = self.init(
            f=discretized_pde.f,
            df=discretized_pde.df,
            y0=discretized_pde.y0,
            t0=discretized_pde.t0,
            num_derivatives=self.state_iwp.num_derivatives,
            wp_diffusion_sqrtm=diffusion_state_sqrtm,
        )
        mean = jnp.concatenate([extended_dy0, jnp.zeros_like(extended_dy0)], -1)

        cov_sqrtm_state = jnp.kron(discretized_pde.Kxx_sqrtm, cov_sqrtm_state)
        cov_sqrtm_eps = jnp.kron(
            discretized_pde.E_sqrtm, 1e-10 * jnp.eye(self.num_derivatives + 1)
        )

        cov_sqrtm = jax.scipy.linalg.block_diag(
            cov_sqrtm_state,
            cov_sqrtm_eps,
        )

        y = rv.MultivariateNormal(mean=mean, cov_sqrtm=cov_sqrtm)

        return odefilter.ODEFilterState(
            t=discretized_pde.t0,
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
            discretized_pde=discretized_pde,
            p0=self.E0,
            p1=self.E1,
            m_pred=mp,
            t=state.t + dt,
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

    @abc.abstractstaticmethod
    def evaluate_ode(*args, **kwargs):
        pass


class LinearLatentForceEK1(LatentForceEK1Base):
    @staticmethod
    @partial(jax.jit, static_argnums=(0,))
    def evaluate_ode(discretized_pde, p0, p1, m_pred, t):
        L = discretized_pde.L
        B = discretized_pde.spatial_grid.boundary_projection_matrix

        E0_state = E0_eps = p0
        E1_state = p1
        E0_stacked = jax.scipy.linalg.block_diag(E0_state, E0_eps)

        m_at = E0_stacked @ m_pred  # Project to first derivatives
        state_at, eps_at = jnp.split(m_at, 2)  # Split up into ODE state and error

        fx = L @ state_at  # Evaluate vector field
        Jx = L  # Evaluate Jacobian of the vector field

        H_state = E1_state - Jx @ E0_state
        H_eps = -E0_eps
        H_boundaries = B @ E0_state
        H_zeros = jnp.zeros_like(H_boundaries)
        H = jnp.block([[H_state, H_eps], [H_boundaries, H_zeros]])

        zeros_bc = jnp.zeros((B.shape[0],))

        b = jnp.concatenate([Jx @ state_at - fx, zeros_bc])
        z = H @ m_pred + b
        return z, H
