import abc
from functools import partial

import jax
import jax.numpy as jnp

from pnmol import pdefilter
from pnmol.base import iwp, rv, sqrt


class _WhiteNoiseEK1Base(pdefilter.PDEFilter):
    def initialize(self, pde):

        self.iwp, self.E0, self.E1, diffusion_state_sqrtm = self.initialize_iwp(pde=pde)

        # Shorthand access to the shapes of the initial conditions
        n, d = self.num_derivatives + 1, pde.L.shape[0]

        # Starting point for the initial conditions
        # Dont make it a non-zero mean without updating the code below!
        m0_raw = jnp.zeros((n, d))
        m0_raw_flat = m0_raw.reshape((-1,), order="F")
        c0 = self.diffuse_prior_scale * jnp.eye(n)  # shorthand
        C0_sqrtm_raw = jnp.kron(diffusion_state_sqrtm, c0)

        # Update state on initial condition
        # There is a clash with the certain initial conditions (via y0)
        # and the assumed-to-be-certain boundary conditions (below).
        # Until this is made up for (can it even?), we add a nugget on the diagonal
        # of the observation covariance matrices (i.e. assume a larg(ish) meascov).
        # Both get the same nugget. This fixes most of the issue.
        z_y0, H_y0 = pde.y0, self.E0
        matrix_nugget = 1e-6 * jnp.eye(d)
        C0_sqrtm_y0, kgain_y0, S_sqrtm_y0 = sqrt.update_sqrt(
            transition_matrix=H_y0,
            cov_cholesky=C0_sqrtm_raw,
            meascov_sqrtm=matrix_nugget,
        )
        m0_flat_y0 = kgain_y0 @ z_y0  # prior mean was zero

        # Evaluate ODE at the initial condition
        z_pde, H_pde, E_sqrtm_pde = self.evaluate_ode(
            pde=pde,
            p0=self.E0,
            p1=self.E1,
            m_pred=m0_flat_y0,
            t=pde.t0,
        )

        # Update the stack of state and latent force on the PDE measurement.
        matrix_nugget = 1e-5 * jnp.eye(d + pde.B.shape[0])
        C0_sqrtm, kgain, S_pde = sqrt.update_sqrt(
            transition_matrix=H_pde,
            cov_cholesky=C0_sqrtm_y0,
            meascov_sqrtm=E_sqrtm_pde + matrix_nugget,
        )
        # residual_pde = H_pde @ m0_flat_y0 - z_pde
        m0 = m0_flat_y0 - kgain @ z_pde

        # Reshape and initialise the RV
        m0_reshaped = m0.reshape((n, d), order="F")
        y = rv.MultivariateNormal(mean=m0_reshaped, cov_sqrtm=C0_sqrtm)

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

    def initialize_iwp(self, pde):

        X = pde.mesh_spatial.points
        diffusion_state_sqrtm = jnp.linalg.cholesky(self.spatial_kernel(X, X.T))
        prior = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=pde.y0.shape[0],
            wp_diffusion_sqrtm=diffusion_state_sqrtm,
        )
        E0 = prior.projection_matrix(0)
        E1 = prior.projection_matrix(1)

        return prior, E0, E1, diffusion_state_sqrtm

    def attempt_step(self, state, dt, pde):
        P, Pinv = self.iwp.nordsieck_preconditioner(dt=dt)
        A, Ql = self.iwp.preconditioned_discretize
        # print(A.shape, Ql.shape)
        n, d = self.num_derivatives + 1, pde.y0.shape[0]

        # [Setup]
        # Pull states into preconditioned state
        m, Cl = Pinv @ state.y.mean.reshape((-1,), order="F"), Pinv @ state.y.cov_sqrtm

        # [Predict]
        mp = self.predict_mean(A, m)

        # Measure / calibrate
        z, H, E_with_bc_sqrtm = self.evaluate_ode(
            pde=pde,
            p0=self.E0 @ P,
            p1=self.E1 @ P,
            m_pred=mp,
            t=state.t + dt,
        )
        _, error = self.estimate_error(ql=Ql, z=z, h=H, E_sqrtm=E_with_bc_sqrtm)
        Clp = sqrt.propagate_cholesky_factor(A @ Cl, Ql)
        error = error[: -pde.B.shape[0]]

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp, meascov_sqrtm=E_with_bc_sqrtm)
        m_new = mp - K @ z

        residual_white = jax.scipy.linalg.solve_triangular(Sl.T, z, lower=False)
        diffusion_squared_local = (
            residual_white @ residual_white / residual_white.shape[0]
        )
        error = dt * error

        # Push back to non-preconditioned state
        Cl_new = P @ Cl_new
        m_new = P @ m_new

        m_new = m_new.reshape((n, d), order="F")
        y_new = jnp.abs(m_new[0])

        new_state = pdefilter.PDEFilterState(
            t=state.t + dt,
            error_estimate=error,
            reference_state=y_new,
            y=rv.MultivariateNormal(m_new, Cl_new),
            diffusion_squared_local=diffusion_squared_local,
        )
        info_dict = dict(num_f_evaluations=1, num_df_evaluations=1)
        return new_state, info_dict

    @staticmethod
    @jax.jit
    def predict_mean(A, m):
        return A @ m

    @staticmethod
    def estimate_error(ql, z, h, E_sqrtm):
        q = ql @ ql.T
        s_no_e = h @ q @ h.T
        e = E_sqrtm @ E_sqrtm.T
        S = s_no_e + e
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error = jnp.sqrt(jnp.diag(S)) * sigma
        return sigma, error

    @abc.abstractstaticmethod
    def evaluate_ode(*args, **kwargs):
        raise NotImplementedError


class LinearWhiteNoiseEK1(_WhiteNoiseEK1Base):
    @staticmethod
    def evaluate_ode(pde, p0, p1, m_pred, t):
        L = pde.L

        m_at = p0 @ m_pred
        fx = L @ m_at
        Jx = L
        b = Jx @ m_at - fx

        H_ode = p1 - Jx @ p0
        H = jnp.vstack((H_ode, pde.B @ p0))
        shift = jnp.hstack((b, jnp.zeros(pde.B.shape[0])))
        z = H @ m_pred + shift

        E_with_bc_sqrtm = jax.scipy.linalg.block_diag(pde.E_sqrtm, pde.R_sqrtm)

        return z, H, E_with_bc_sqrtm


class SemiLinearWhiteNoiseEK1(_WhiteNoiseEK1Base):
    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def evaluate_ode(pde, p0, p1, m_pred, t):
        B = pde.B
        L = pde.L

        m_at = p0 @ m_pred
        fx = pde.f(t, m_at)
        Jx = pde.df(t, m_at)
        b = Jx @ m_at - fx

        H_ode = p1 - Jx @ p0 - L @ p0
        H = jnp.vstack((H_ode, B @ p0))
        shift = jnp.hstack((b, jnp.zeros(B.shape[0])))
        z = H @ m_pred + shift

        E_with_bc_sqrtm = jax.scipy.linalg.block_diag(pde.E_sqrtm, pde.R_sqrtm)
        return z, H, E_with_bc_sqrtm
