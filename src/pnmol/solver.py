import abc
from functools import partial

import jax
import jax.numpy as jnp

from pnmol import iwp, odefilter, rv, sqrt


class _MeasurementCovarianceEK1Base(odefilter.ODEFilter):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.P0 = None
        self.E0 = None
        self.E1 = None

    def initialize(self, discretized_pde):

        X = discretized_pde.spatial_grid.points
        diffusion_state_sqrtm = jnp.linalg.cholesky(self.spatial_kernel(X, X.T))

        self.iwp = iwp.IntegratedWienerTransition(
            num_derivatives=self.num_derivatives,
            wiener_process_dimension=discretized_pde.y0.shape[0],
            wp_diffusion_sqrtm=diffusion_state_sqrtm,
        )
        self.P0 = self.E0 = self.iwp.projection_matrix(0)
        self.E1 = self.iwp.projection_matrix(1)

        # This is kind of wrong still... RK init should get the proper diffusion.
        ivp = discretized_pde.to_tornadox_ivp_1d()
        extended_dy0, cov_sqrtm = self.init(
            f=ivp.f,
            df=ivp.df,
            y0=ivp.y0,
            t0=ivp.t0,
            num_derivatives=self.iwp.num_derivatives,
            wp_diffusion_sqrtm=diffusion_state_sqrtm,
        )
        dy0_padded = jnp.pad(
            extended_dy0, pad_width=1, mode="constant", constant_values=0.0
        )
        dy0_full = dy0_padded[1:-1]

        y = rv.MultivariateNormal(
            mean=dy0_full,
            cov_sqrtm=jnp.kron(diffusion_state_sqrtm, cov_sqrtm),
        )
        return odefilter.ODEFilterState(
            t=discretized_pde.t0,
            y=y,
            error_estimate=None,
            reference_state=None,
        )

    def attempt_step(self, state, dt, discretized_pde):
        P, Pinv = self.iwp.nordsieck_preconditioner(dt=dt)
        A, Ql = self.iwp.preconditioned_discretize
        n, d = self.num_derivatives + 1, discretized_pde.y0.shape[0]

        # [Setup]
        # Pull states into preconditioned state
        m, Cl = Pinv @ state.y.mean.reshape((-1,), order="F"), Pinv @ state.y.cov_sqrtm

        # [Predict]
        mp = self.predict_mean(A, m)

        # Measure / calibrate
        z, H = self.evaluate_ode(
            discretized_pde=discretized_pde,
            p0=self.E0 @ P,
            p1=self.E1 @ P,
            m_pred=mp,
            t=state.t + dt,
        )
        E_with_bc_sqrtm = jax.scipy.linalg.block_diag(
            discretized_pde.E_sqrtm, jnp.zeros((2, 2))
        )
        sigma, error = self.estimate_error(ql=Ql, z=z, h=H, E_sqrtm=E_with_bc_sqrtm)
        Clp = sqrt.propagate_cholesky_factor(A @ Cl, sigma * Ql)

        # [Update]
        Cl_new, K, Sl = sqrt.update_sqrt(H, Clp, meascov_sqrtm=sigma * E_with_bc_sqrtm)
        m_new = mp - K @ z

        # Push back to non-preconditioned state
        Cl_new = P @ Cl_new
        m_new = P @ m_new

        m_new = m_new.reshape((n, d), order="F")
        y_new = jnp.abs(m_new[0])

        new_state = odefilter.ODEFilterState(
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
    @jax.jit
    def estimate_error(ql, z, h, E_sqrtm):
        S = h @ ql @ ql.T @ h.T + E_sqrtm @ E_sqrtm.T
        sigma_squared = z @ jnp.linalg.solve(S, z) / z.shape[0]
        sigma = jnp.sqrt(sigma_squared)
        error = jnp.sqrt(jnp.diag(S)) * sigma
        return sigma, error

    @abc.abstractstaticmethod
    def evaluate_ode(*args, **kwargs):
        pass


class LinearMeasurementCovarianceEK1(_MeasurementCovarianceEK1Base):
    @staticmethod
    # @partial(jax.jit, static_argnums=(0,))
    def evaluate_ode(discretized_pde, p0, p1, m_pred, t):
        B = discretized_pde.spatial_grid.boundary_projection_matrix
        L = discretized_pde.L

        m_at = p0 @ m_pred
        fx = L @ m_at
        Jx = L
        b = Jx @ m_at - fx

        H_ode = p1 - Jx @ p0
        H = jnp.vstack((H_ode, B @ p0))
        shift = jnp.hstack((b, jnp.zeros(B.shape[0])))
        z = H @ m_pred + shift
        return z, H
