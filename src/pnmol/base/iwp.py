from collections import namedtuple
from functools import cached_property, partial

import jax
import jax.numpy as jnp
import scipy.linalg
import scipy.special


class IntegratedWienerTransition(
    namedtuple("_IWP", "wiener_process_dimension num_derivatives wp_diffusion_sqrtm")
):
    @cached_property
    def preconditioned_discretize_1d(self):
        """Preconditioned system matrices for one dimension.

        Returns
        -------
        state_trans_mat: jax array
            1D preconditioned state transition matrix.
        proc_noise_cov_cholesky: jax array
            1D preconditioned cholesky factor of the process noise covariance matrix.
        """
        A_1d = jnp.flip(
            jnp.array(
                scipy.linalg.pascal(self.num_derivatives + 1, kind="lower", exact=False)
            )
        )
        Q_1d = jnp.flip(jnp.array(scipy.linalg.hilbert(self.num_derivatives + 1)))
        return A_1d, jnp.linalg.cholesky(Q_1d)

    @cached_property
    def preconditioned_discretize(self):
        """Preconditioned system matrices.

        Returns
        -------
        state_trans_mat: jax array
            Preconditioned state transition matrix.
        proc_noise_cov_cholesky: jax array
            Preconditioned cholesky factor of the process noise covariance matrix.
        """
        A_1d, L_Q1d = self.preconditioned_discretize_1d
        id_factor = jnp.eye(self.wiener_process_dimension)
        A = jnp.kron(
            id_factor,
            A_1d,
        )
        L_Q = jnp.kron(
            self.wp_diffusion_sqrtm,
            L_Q1d,
        )
        return A, L_Q

    def nordsieck_preconditioner_1d_raw(self, dt):
        powers = jnp.arange(self.num_derivatives, -1, -1)
        scales = jnp.array(scipy.special.factorial(powers))
        powers = powers + 0.5

        scaling_vector = (jnp.abs(dt) ** powers) / scales
        scaling_vector_inv = (jnp.abs(dt) ** (-powers)) * scales
        return scaling_vector, scaling_vector_inv

    def nordsieck_preconditioner_1d(self, dt):
        """Create matrix for 1-D Nordsieck preconditioner and its inverse.

        Returns
        -------
        nordsieck_precond: jax array
            Nordsieck preconditioning matrix
        nordsieck_precond_inv: jax array
            Inverse Nordsieck preconditioning matrix
        """
        scaling_vector, scaling_vector_inv = self.nordsieck_preconditioner_1d_raw(dt)
        nordsieck_precond_1d = jnp.diag(scaling_vector)
        nordsieck_procond_inv_1d = jnp.diag(scaling_vector_inv)
        return nordsieck_precond_1d, nordsieck_procond_inv_1d

    def nordsieck_preconditioner(self, dt):
        """Create matrix for Nordsieck preconditioner and its inverse.

        Returns
        -------
        nordsieck_precond: jax array
            Nordsieck preconditioning matrix
        nordsieck_precond_inv: jax array
            Inverse Nordsieck preconditioning matrix
        """
        (
            nordsieck_precond_1d,
            nordsieck_procond_inv_1d,
        ) = self.nordsieck_preconditioner_1d(dt)
        id_factor = jnp.eye(self.wiener_process_dimension)
        return (
            jnp.kron(id_factor, nordsieck_precond_1d),
            jnp.kron(id_factor, nordsieck_procond_inv_1d),
        )

    def non_preconditioned_discretize(self, dt):
        """Non-preconditioned system matrices. Mainly for testing and debugging.

        Returns
        -------
        state_trans_mat: jax array
            Non-preconditioned state transition matrix.
        proc_noise_cov_cholesky: jax array
            Non-preconditioned cholesky factor of the process noise covariance matrix.
        """

        nordsieck_precond, nordsieck_precond_inv = self.nordsieck_preconditioner(dt)
        (
            precond_state_trans_mat,
            precond_proc_noise_chol,
        ) = self.preconditioned_discretize

        state_trans_mat = (
            nordsieck_precond @ precond_state_trans_mat @ nordsieck_precond_inv
        )

        proc_noise_cov_cholesky = nordsieck_precond @ precond_proc_noise_chol

        return (state_trans_mat, proc_noise_cov_cholesky)

    # No jit because weird internal jax tracing stuff
    def projection_matrix(self, derivative_to_project_onto):
        """Creates a projection matrix kron(I_d, e_p)"""
        I_d = jnp.eye(self.wiener_process_dimension)
        return jnp.kron(I_d, self.projection_matrix_1d(derivative_to_project_onto))

    # No jit because weird internal jax tracing stuff
    def projection_matrix_1d(self, derivative_to_project_onto):
        """Creates a projection matrix e_p"""
        return jnp.eye(1, self.num_derivatives + 1, derivative_to_project_onto)

    @property
    def state_dimension(self):
        return self.wiener_process_dimension * (self.num_derivatives + 1)
