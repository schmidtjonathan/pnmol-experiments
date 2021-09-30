import abc
from functools import partial

import jax
import jax.numpy as jnp

from pnmol import pdefilter
from pnmol.base import iwp, rv, sqrt, stacked_ssm


class _LatentForceEK1Base(pdefilter.PDEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None

    def initialize(self, pde):

        X = pde.mesh_spatial.points
        diffusion_state_sqrtm = jnp.linalg.cholesky(self.spatial_kernel(X, X.T))

        self.state_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=X.shape[0],
            wp_diffusion_sqrtm=diffusion_state_sqrtm,
        )
        self.lf_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=X.shape[0],
            wp_diffusion_sqrtm=pde.E_sqrtm,
        )
        self.ssm = stacked_ssm.StackedSSM(processes=[self.state_iwp, self.lf_iwp])

        self.P0 = self.E0 = self.state_iwp.projection_matrix(0)
        self.E1 = self.state_iwp.projection_matrix(1)

        # This is kind of wrong still... RK init should get the proper diffusion.
        ivp = pde.to_tornadox_ivp()
        extended_dy0, cov_sqrtm_state = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.state_iwp.num_derivatives,
            wp_diffusion_sqrtm=diffusion_state_sqrtm,
        )
        dy0_padded = jnp.pad(
            extended_dy0, pad_width=1, mode="constant", constant_values=0.0
        )
        dy0_full = dy0_padded[1:-1]
        mean = jnp.concatenate([dy0_full, jnp.zeros_like(dy0_full)], -1)

        cov_sqrtm_state_ = jnp.kron(diffusion_state_sqrtm, cov_sqrtm_state)
        cov_sqrtm_eps = jnp.kron(pde.E_sqrtm, cov_sqrtm_state)

        cov_sqrtm = jax.scipy.linalg.block_diag(
            cov_sqrtm_state_,
            cov_sqrtm_eps,
        )

        y = rv.MultivariateNormal(mean=mean, cov_sqrtm=cov_sqrtm)

        return pdefilter.PDEFilterState(
            t=pde.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
            diffusion_squared_local=0.0,
        )

    def attempt_step(self, state, dt, pde):
        A, Ql = self.ssm.non_preconditioned_discretize(dt)
        n, d = self.num_derivatives + 1, self.state_iwp.wiener_process_dimension

        # [Predict]
        glued_batched_mean = state.y.mean
        batched_state_mean, batched_eps_mean = jnp.split(glued_batched_mean, 2, axis=-1)

        flat_state_mean = batched_state_mean.reshape((-1,), order="F")
        flat_eps_mean = batched_eps_mean.reshape((-1,), order="F")
        glued_flat_mean = jnp.concatenate((flat_state_mean, flat_eps_mean))
        mp = self.predict_mean(A, glued_flat_mean)

        # Measure / calibrate
        z, H = self.evaluate_ode(
            pde=pde,
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

        residual_white = jax.scipy.linalg.solve_triangular(Sl.T, z, lower=False)
        diffusion_squared_local = (
            residual_white @ residual_white / residual_white.shape[0]
        )

        flat_state_m_new, flat_eps_m_new = jnp.split(flat_m_new, 2)
        batched_state_m_new = flat_state_m_new.reshape((n, d), order="F")
        batched_eps_m_new = flat_eps_m_new.reshape((n, d), order="F")

        glued_new_mean = jnp.concatenate([batched_state_m_new, batched_eps_m_new], -1)

        new_state = pdefilter.PDEFilterState(
            t=state.t + dt,
            error_estimate=None,
            reference_state=None,
            y=rv.MultivariateNormal(glued_new_mean, Cl_new),
            diffusion_squared_local=diffusion_squared_local,
        )
        info_dict = dict(num_f_evaluations=1)
        return new_state, info_dict

    @staticmethod
    @jax.jit
    def predict_mean(A, m):
        return A @ m

    @abc.abstractstaticmethod
    def evaluate_ode(*args, **kwargs):
        raise NotImplementedError


class LinearLatentForceEK1(_LatentForceEK1Base):
    @staticmethod
    def evaluate_ode(pde, p0, p1, m_pred, t):
        L = pde.L

        E0_state = E0_eps = p0
        E1_state = p1
        E0_stacked = jax.scipy.linalg.block_diag(E0_state, E0_eps)

        m_at = E0_stacked @ m_pred  # Project to first derivatives
        state_at, eps_at = jnp.split(m_at, 2)  # Split up into ODE state and error

        fx = L @ state_at  # Evaluate vector field
        Jx = L  # Evaluate Jacobian of the vector field

        H_state = E1_state - Jx @ E0_state
        H_eps = -E0_eps
        H_boundaries = pde.B @ E0_state
        H_zeros = jnp.zeros_like(H_boundaries)
        H = jnp.block([[H_state, H_eps], [H_boundaries, H_zeros]])

        zeros_bc = jnp.zeros((pde.B.shape[0],))

        b = jnp.concatenate([Jx @ state_at - fx, zeros_bc])
        z = H @ m_pred + b
        return z, H
