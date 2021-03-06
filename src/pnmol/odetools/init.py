import abc
from functools import partial

import jax
import jax.numpy as jnp
import scipy.integrate
from jax.experimental.jet import jet

from pnmol.base import iwp, kalman, sqrt


class InitializationRoutine(abc.ABC):
    @abc.abstractmethod
    def __call__(self, f, df, y0, t0, num_derivatives, wp_diffusion_sqrtm):
        raise NotImplementedError


class TaylorMode(InitializationRoutine):

    # Adapter to make it work with ODEFilters
    def __call__(self, f, df, y0, t0, num_derivatives, wp_diffusion_sqrtm=None):
        m0 = TaylorMode.taylor_mode(
            fun=f, y0=y0, t0=t0, num_derivatives=num_derivatives
        )
        return m0, jnp.zeros((num_derivatives + 1, num_derivatives + 1))

    def __repr__(self):
        return f"{self.__class__.__name__}()"

    @staticmethod
    def taylor_mode(fun, y0, t0, num_derivatives):
        """Initialize a probabilistic ODE solver with Taylor-mode automatic differentiation."""

        extended_state = jnp.concatenate((jnp.ravel(y0), jnp.array([t0])))
        evaluate_ode_for_extended_state = partial(
            TaylorMode._evaluate_ode_for_extended_state, fun=fun, y0=y0
        )

        # Corner case 1: num_derivatives == 0
        derivs = [y0]
        if num_derivatives == 0:
            return jnp.stack(derivs)

        # Corner case 2: num_derivatives == 1
        initial_series = (jnp.ones_like(extended_state),)
        (
            initial_taylor_coefficient,
            taylor_coefficients,
        ) = TaylorMode.augment_taylor_coefficients(
            evaluate_ode_for_extended_state, extended_state, initial_series
        )
        derivs.append(initial_taylor_coefficient[:-1])
        if num_derivatives == 1:
            return jnp.stack(derivs)

        # Order > 1
        for _ in range(1, num_derivatives):
            _, taylor_coefficients = TaylorMode.augment_taylor_coefficients(
                evaluate_ode_for_extended_state, extended_state, taylor_coefficients
            )
            derivs.append(taylor_coefficients[-2][:-1])
        return jnp.stack(derivs)

    @staticmethod
    def augment_taylor_coefficients(fun, x, taylor_coefficients):
        (init_coeff, [*remaining_taylor_coefficents]) = jet(
            fun=fun,
            primals=(x,),
            series=(taylor_coefficients,),
        )
        taylor_coefficients = (
            init_coeff,
            *remaining_taylor_coefficents,
        )

        return init_coeff, taylor_coefficients

    @staticmethod
    def _evaluate_ode_for_extended_state(extended_state, fun, y0):
        r"""Evaluate the ODE for an extended state (x(t), t).

        More precisely, compute the derivative of the stacked state (x(t), t) according to the ODE.
        This function implements a rewriting of non-autonomous as autonomous ODEs.
        This means that

        .. math:: \dot x(t) = f(t, x(t))

        becomes

        .. math:: \dot z(t) = \dot (x(t), t) = (f(x(t), t), 1).

        Only considering autonomous ODEs makes the jet-implementation
        (and automatic differentiation in general) easier.
        """
        x, t = jnp.reshape(extended_state[:-1], y0.shape), extended_state[-1]
        dx = fun(t, x)
        dx_ravelled = jnp.ravel(dx)
        stacked_ode_eval = jnp.concatenate((dx_ravelled, jnp.array([1.0])))
        return stacked_ode_eval


# RK initialisation


class RungeKutta(InitializationRoutine):
    def __init__(self, dt=0.01, method="RK45", use_df=True):
        self.dt = dt
        self.method = method
        self.stack_initvals = Stack(use_df=use_df)

    def __repr__(self):
        return f"{self.__class__.__name__}(dt={self.dt}, method={self.method})"

    def __call__(self, f, df, y0, t0, num_derivatives, wp_diffusion_sqrtm):
        num_steps = num_derivatives + 1
        ts, ys = self.rk_data(
            f=f, t0=t0, dt=self.dt, num_steps=num_steps, y0=y0, method=self.method
        )
        m, sc = self.stack_initvals(
            f=f, df=df, y0=y0, t0=t0, num_derivatives=num_derivatives
        )
        return RungeKutta.rk_init_improve(
            m=m, sc=sc, t0=t0, ts=ts, ys=ys, wp_diffusion_sqrtm=wp_diffusion_sqrtm
        )

    @staticmethod
    def rk_data(f, t0, dt, num_steps, y0, method):

        # Force fixed steps via t_eval
        t_eval = jnp.arange(t0, t0 + num_steps * dt, dt)

        # Compute the data with atol=rtol=1e12 (we want fixed steps!)
        sol = scipy.integrate.solve_ivp(
            fun=f,
            t_span=(min(t_eval), max(t_eval)),
            y0=y0,
            atol=1e12,
            rtol=1e12,
            t_eval=t_eval,
            method=method,
        )
        return sol.t, sol.y.T

    # Jitting this is possibly, but debatable -- it is not very fast due to the for-loop logic underneath.
    # I think for now we leave it to a "user" -> us :)
    @staticmethod
    def rk_init_improve(m, sc, t0, ts, ys, wp_diffusion_sqrtm):
        """Improve an initial mean estimate by fitting it to a number of RK steps."""

        d = m.shape[1]
        num_derivatives = m.shape[0] - 1

        # Prior
        prior_iwp = iwp.IntegratedWienerTransition(
            num_derivatives=num_derivatives,
            wiener_process_dimension=d // 2,
            wp_diffusion_sqrtm=wp_diffusion_sqrtm,
        )
        phi_1d, sq_1d = prior_iwp.preconditioned_discretize_1d

        # Store  -- mean and cov are needed, the other ones should be taken from future steps!
        filter_res = [(m, sc, None, None, None, None, None, None)]
        t_loc = t0

        # Ignore the first (t,y) pair because this information is already contained in the initial value
        # with certainty, thus it would lead to clashes.
        for t, y in zip(ts[1:], ys[1:]):

            # Fetch preconditioner
            dt = t - t_loc
            p_1d_raw, p_inv_1d_raw = prior_iwp.nordsieck_preconditioner_1d_raw(dt)

            # Make the next step but return ALL the intermediate quantities
            # (they are needed for efficient smoothing)
            (m, sc, m_pred, sc_pred, sgain, x,) = RungeKutta._forward_filter_step(
                y, sc, m, sq_1d, p_1d_raw, p_inv_1d_raw, phi_1d
            )

            # Store parameters;
            # (m, sc) are in "normal" coordinates, the others are already preconditioned!
            filter_res.append(
                (m, sc, sgain, m_pred, sc_pred, x, p_1d_raw, p_inv_1d_raw)
            )
            t_loc = t

        # Smoothing pass. Make heavy use of the filter output.
        final_out = filter_res[-1]
        m_fut, sc_fut, sgain_fut, m_pred, _, x, p_1d_raw, p_inv_1d_raw = final_out

        for filter_output in reversed(filter_res[:-1]):

            # Push means and covariances into the preconditioned space
            m_, sc_ = filter_output[0], filter_output[1]
            m, sc = p_inv_1d_raw[:, None] * m_, p_inv_1d_raw[:, None] * sc_
            m_fut_, sc_fut_ = (
                p_inv_1d_raw[:, None] * m_fut,
                p_inv_1d_raw[:, None] * sc_fut,
            )

            # Make smoothing step
            m_fut__, sc_fut__ = kalman.smoother_step_sqrt(
                m=m,
                sc=sc,
                m_fut=m_fut_,
                sc_fut=sc_fut_,
                sgain=sgain_fut,
                sq=sq_1d,
                mp=m_pred,
                x=x,
            )

            # Pull means and covariances back into old coordinates
            # Only for the result of the smoothing step.
            # The other means and covariances are not used anymore.
            m_fut, sc_fut = p_1d_raw[:, None] * m_fut__, p_1d_raw[:, None] * sc_fut__

            # Read out the new parameters
            # They are already preconditioned. m_fut, sc_fut are not,
            # but will be pushed into the correct coordinates in the next iteration.
            _, _, sgain_fut, m_pred, _, x, p_1d_raw, p_inv_1d_raw = filter_output

        return m_fut, sc_fut

    @staticmethod
    @jax.jit
    def _forward_filter_step(y, sc, m, sq_1d, p_1d_raw, p_inv_1d_raw, phi_1d):

        # Apply preconditioner
        m = p_inv_1d_raw[:, None] * m
        sc = p_inv_1d_raw[:, None] * sc

        # Predict from t_loc to t
        m_pred = phi_1d @ m
        x = phi_1d @ sc
        sc_pred = sqrt.propagate_cholesky_factor(x, sq_1d)

        # Compute the gainzz
        cross = (x @ sc.T).T
        sgain = jax.scipy.linalg.cho_solve((sc_pred, True), cross.T).T

        # Measure (H := "slicing" \circ "remove preconditioner")
        sc_pred_np = p_1d_raw[:, None] * sc_pred
        h_sc_pred = sc_pred_np[0, :]
        s = h_sc_pred @ h_sc_pred.T
        cross = sc_pred @ h_sc_pred.T
        kgain = cross / s
        z = (p_1d_raw[:, None] * m_pred)[0]

        # Update (with a good sprinkle of broadcasting)
        m_loc = m_pred - kgain[:, None] * (z - y)[None, :]
        sc_loc = sc_pred - kgain[:, None] * h_sc_pred[None, :]

        # Undo preconditioning
        m = p_1d_raw[:, None] * m_loc
        sc = p_1d_raw[:, None] * sc_loc

        return m, sc, m_pred, sc_pred, sgain, x


class Stack(InitializationRoutine):
    def __init__(self, use_df=True):
        self.use_df = use_df

    def __call__(self, f, df, y0, t0, num_derivatives, wp_diffusion_sqrtm=None):
        if self.use_df:
            return Stack.initial_state_jac(
                f=f, df=df, y0=y0, t0=t0, num_derivatives=num_derivatives
            )
        return Stack.initial_state_no_jac(
            f=f, y0=y0, t0=t0, num_derivatives=num_derivatives
        )

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 1, 4))
    def initial_state_jac(f, df, y0, t0, num_derivatives):
        d = y0.shape[0]
        n = num_derivatives + 1

        fy = f(t0, y0)
        dfy = df(t0, y0)
        m = jnp.stack([y0, fy, dfy @ fy] + [jnp.zeros(d)] * (n - 3))
        sc = jnp.diag(jnp.array([0.0, 0.0, 0.0] + [1e3] * (n - 3)))
        return m, sc

    @staticmethod
    @partial(jax.jit, static_argnums=(0, 3))
    def initial_state_no_jac(f, y0, t0, num_derivatives):
        d = y0.shape[0]
        n = num_derivatives + 1

        fy = f(t0, y0)
        m = jnp.stack([y0, fy] + [jnp.zeros(d)] * (n - 2))
        sc = jnp.diag(jnp.array([0.0, 0.0] + [1e3] * (n - 2)))
        return m, sc
