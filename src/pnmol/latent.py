import abc
from functools import partial

import jax
import jax.numpy as jnp

from pnmol import pdefilter
from pnmol.base import iwp, rv, sqrt, stacked_ssm


class _LatentForceEK1Base(pdefilter.PDEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # Stacked state space model
        self.ssm = None
        self.state_iwp = None
        self.lf_iwp = None

    def initialize(self, pde):
        # The initialization of latent PDE filters is a bit complicated...
        # It starts with assembling the state space components.
        # Then, m0 and C0 are chosen as a standard Normal RV.
        # Then, the state RV is updated on the PDE initial condition.
        # This requires a tiny nugget, because we will continue conditioning the same RV.
        # Then, the state RV and the latent RV are stacked together.
        # The ODE is evaluated, and the stack of state and latent force
        # is updated on an EK1-linearised PDE measurement (which includes the BCs).
        # Altogether, this would be equivalent to initialising (y, \dot y0) accurately,
        # and the rest with standard-normals.
        # The specialty herein is that the BCs are used faithfully.
        #
        # To faithfully compare with e.g. tornadox,
        # use the tornadox.Stack(use_df=False) initialization for the ODE filters.
        #
        # One thing that should not be discarded is that due to the updating nature here,
        # the global diffusion MLE is affected (y0 and the PDE measurement are data).

        # [Initialize state-space model]
        (
            self.state_iwp,
            self.lf_iwp,
            self.E0,
            self.E1,
            diffusion_state_sqrtm,
        ) = self.initialize_iwp_latent(pde=pde)
        self.ssm = stacked_ssm.StackedSSM(processes=[self.state_iwp, self.lf_iwp])

        # [Initialize random variables]

        # Shorthand access to the shapes of the initial conditions
        n, d = self.num_derivatives + 1, pde.L.shape[0]

        # Starting point for the initial conditions
        # Dont make it a non-zero mean without updating the code below!
        m0_state_raw = jnp.zeros((n, d))
        m0_latent_raw = jnp.zeros((n, d))
        m0_state_raw_flat = m0_state_raw.reshape((-1,), order="F")
        m0_latent_raw_flat = m0_latent_raw.reshape((-1,), order="F")
        c0 = self.diffuse_prior_scale * jnp.eye(n)  # shorthand
        C0_sqrtm_state_raw = jnp.kron(diffusion_state_sqrtm, c0)
        C0_sqrtm_latent_raw = jnp.kron(pde.E_sqrtm, c0)

        # Update state on initial condition
        # There is a clash with the certain initial conditions (via y0)
        # and the assumed-to-be-certain boundary conditions (below).
        # Until this is made up for (can it even?), we add a nugget on the diagonal
        # of the observation covariance matrices (i.e. assume a larg(ish) meascov).
        # Both get the same nugget. This fixes most of the issue.
        z_y0, H_y0 = pde.y0, self.E0
        matrix_nugget = 1e-10 * jnp.eye(d)
        C0_sqrtm_state_y0, kgain_y0, S_sqrtm_y0 = sqrt.update_sqrt(
            transition_matrix=H_y0,
            cov_cholesky=C0_sqrtm_state_raw,
            meascov_sqrtm=matrix_nugget,
        )
        m0_state_flat_y0 = kgain_y0 @ z_y0  # prior mean was zero

        # Stack m0 and e0 together
        m0_stack = jnp.concatenate((m0_state_flat_y0, m0_latent_raw_flat))
        C0_sqrtm_block = jax.scipy.linalg.block_diag(
            C0_sqrtm_state_y0, C0_sqrtm_latent_raw
        )

        # Evaluate ODE at the initial condition
        p_empty = jnp.eye(n * d)
        z_pde, H_pde = self.evaluate_ode(
            pde=pde,
            p0=self.E0,
            p1=self.E1,
            m_pred=m0_stack,
            t=pde.t0,
            p_state=p_empty,
            p_eps=p_empty,
        )

        # Update the stack of state and latent force on the PDE measurement.
        matrix_nugget = 1e-10 * jnp.eye(d + pde.B.shape[0])
        C0_sqrtm_state_latent, kgain, S_pde = sqrt.update_sqrt(
            transition_matrix=H_pde,
            cov_cholesky=C0_sqrtm_block,
            meascov_sqrtm=matrix_nugget,
        )
        m0_state_latent = m0_stack - kgain @ z_pde

        # Reshape carefully
        m0_state, m0_latent = jnp.split(m0_state_latent, 2)
        m0_state_reshaped = m0_state.reshape((n, d), order="F")
        m0_latent_reshaped = m0_latent.reshape((n, d), order="F")
        m0_state_latent_reshaped = jnp.concatenate(
            (m0_state_reshaped, m0_latent_reshaped), axis=1
        )

        y = rv.MultivariateNormal(
            mean=m0_state_latent_reshaped, cov_sqrtm=C0_sqrtm_state_latent
        )

        # Dont forget that the initial data affects the quasi-MLE for the diffusion!
        S_y0, S_pde = S_sqrtm_y0 @ S_sqrtm_y0.T, S_pde @ S_pde.T
        diffusion_squared_local_y0 = z_y0 @ jnp.linalg.solve(S_y0, z_y0) / z_y0.shape[0]
        diffusion_squared_local_pde = (
            z_pde @ jnp.linalg.solve(S_pde, z_pde) / z_pde.shape[0]
        )

        return pdefilter.PDEFilterState(
            t=pde.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
            diffusion_squared_local=[
                # diffusion_squared_local_y0,
                # diffusion_squared_local_pde,
            ],
        )

    def initialize_iwp_latent(self, pde):

        X = pde.mesh_spatial.points
        diffusion_state_sqrtm = jnp.linalg.cholesky(self.spatial_kernel(X, X.T))
        prior_state = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=pde.y0.shape[0],
            wp_diffusion_sqrtm=diffusion_state_sqrtm,
        )
        prior_latent = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=pde.y0.shape[0],
            wp_diffusion_sqrtm=pde.E_sqrtm,
        )
        E0 = prior_latent.projection_matrix(0)
        E1 = prior_latent.projection_matrix(1)

        return prior_state, prior_latent, E0, E1, diffusion_state_sqrtm

    def attempt_step(self, state, dt, pde):

        P, Pinv = self.ssm.nordsieck_preconditioner(dt=dt)
        P_state, Pinv_state = self.state_iwp.nordsieck_preconditioner(dt=dt)
        P_eps, Pinv_eps = self.lf_iwp.nordsieck_preconditioner(dt=dt)

        A, Ql = self.ssm.preconditioned_discretize
        n, d = self.num_derivatives + 1, self.state_iwp.wiener_process_dimension

        # [Predict]
        glued_batched_mean = state.y.mean  # (nu + 1, 2 * d)
        batched_state_mean, batched_eps_mean = jnp.split(
            glued_batched_mean, 2, axis=-1
        )  # [(nu + 1, 2 * d), (nu + 1, 2 * d)]

        flat_state_mean = batched_state_mean.reshape(
            (-1,), order="F"
        )  # (d * (nu + 1), )
        flat_eps_mean = batched_eps_mean.reshape((-1,), order="F")  # (d * (nu + 1), )
        glued_flat_mean = jnp.concatenate(
            (flat_state_mean, flat_eps_mean)
        )  # (2 * d * (nu + 1),)

        # Pull states into preconditioned space
        glued_flat_mean, Cl = Pinv @ glued_flat_mean, Pinv @ state.y.cov_sqrtm

        mp = self.predict_mean(A, glued_flat_mean)

        # Measure / calibrate
        z, H = self.evaluate_ode(
            pde=pde,
            p0=self.E0,
            p1=self.E1,
            m_pred=mp,
            t=state.t + dt,
            p_state=P_state,
            p_eps=P_eps,
        )

        Clp = sqrt.propagate_cholesky_factor(A @ Cl, Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt_no_meascov(H, Clp)
        flat_m_new = mp - K @ z

        # Back into non-preconditioned space
        flat_m_new, Cl_new = P @ flat_m_new, P @ Cl_new

        # Calibrate local diffusion
        residual_white = jax.scipy.linalg.solve_triangular(Sl.T, z, lower=False)
        diffusion_squared_local = (
            residual_white @ residual_white / residual_white.shape[0]
        )

        flat_state_m_new, flat_eps_m_new = jnp.split(flat_m_new, 2)
        batched_state_m_new = flat_state_m_new.reshape((n, d), order="F")
        batched_eps_m_new = flat_eps_m_new.reshape((n, d), order="F")

        glued_new_mean = jnp.concatenate(
            [batched_state_m_new, batched_eps_m_new], axis=-1
        )

        new_state = pdefilter.PDEFilterState(
            t=state.t + dt,
            error_estimate=None,
            reference_state=None,
            y=rv.MultivariateNormal(glued_new_mean, Cl_new),
            diffusion_squared_local=diffusion_squared_local,
        )
        info_dict = dict(num_f_evaluations=1, num_df_evaluations=1)
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
    def evaluate_ode(pde, p0, p1, m_pred, t, p_state, p_eps):
        L = pde.L

        E0_state = p0 @ p_state
        E0_eps = p0 @ p_eps
        E1_state = p1 @ p_state
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


class SemiLinearLatentForceEK1(_LatentForceEK1Base):
    @staticmethod
    def evaluate_ode(pde, p0, p1, m_pred, t, p_state, p_eps):
        L = pde.L

        E0_state = p0 @ p_state
        E0_eps = p0 @ p_eps
        E1_state = p1 @ p_state
        E0_stacked = jax.scipy.linalg.block_diag(E0_state, E0_eps)

        m_at = E0_stacked @ m_pred  # Project to first derivatives
        state_at, eps_at = jnp.split(m_at, 2)  # Split up into ODE state and error

        fx = pde.f(t, state_at)  # Evaluate vector field
        Jx = pde.df(t, state_at)  # Evaluate Jacobian of the vector field

        H_state = E1_state - Jx @ E0_state - L @ E0_state
        H_eps = -E0_eps
        H_boundaries = pde.B @ E0_state
        H_zeros = jnp.zeros_like(H_boundaries)
        H = jnp.block([[H_state, H_eps], [H_boundaries, H_zeros]])

        zeros_bc = jnp.zeros((pde.B.shape[0],))

        b = jnp.concatenate([Jx @ state_at - fx, zeros_bc])
        z = H @ m_pred + b
        return z, H
