"""Code to generate figure 1."""


import jax
import jax.numpy as jnp
import plotting
import scipy.integrate
import tornadox

import pnmol


def solve_pde_pnmol_white(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.odetools.step.Constant(dt)
    ek1 = pnmol.white.LinearWhiteNoiseEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )
    sol = ek1.solve(pde, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    means, stds = read_mean_and_std(sol, E0)
    gamma = jnp.sqrt(sol.diffusion_squared_calibrated)
    print(gamma)
    return means, gamma * stds, sol.t, pde.mesh_spatial.points


def solve_pde_pnmol_latent(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.odetools.step.Constant(dt)
    ek1 = pnmol.latent.LinearLatentForceEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )
    sol = ek1.solve(pde, progressbar=progressbar)
    E0 = ek1.state_iwp.projection_matrix(0)
    means, stds = read_mean_and_std_latent(sol, E0)
    gamma = jnp.sqrt(sol.diffusion_squared_calibrated)
    print(gamma)
    return means, gamma * stds, sol.t, pde.mesh_spatial.points


def solve_pde_tornadox(pde, *, dt, nu, progressbar):
    steprule = tornadox.step.ConstantSteps(dt)
    ivp = pde.to_tornadox_ivp()
    ek1 = tornadox.ek1.ReferenceEK1(
        num_derivatives=nu,
        steprule=steprule,
        initialization=tornadox.init.Stack(use_df=False),
    )
    sol = ek1.solve(ivp, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    means, stds = read_mean_and_std(sol, E0)

    means = jnp.pad(means, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...]
    stds = jnp.pad(stds, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...]
    return means, stds, sol.t, pde.mesh_spatial.points


def solve_pde_reference(pde, *, dt, high_res_factor_dx, high_res_factor_dt):
    t_eval = jnp.arange(pde.t0, pde.tmax, step=dt)
    ivp = pde.to_tornadox_ivp()
    sol = scipy.integrate.solve_ivp(ivp.f, ivp.t_span, ivp.y0, t_eval=t_eval)

    means = sol.y.T
    stds = 0.0 * sol.y.T
    ts = t_eval[::high_res_factor_dt]

    means = jnp.pad(means, pad_width=1, mode="constant", constant_values=0.0)[
        1:-1, ...
    ][::high_res_factor_dt, ::high_res_factor_dx]
    stds = jnp.pad(stds, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...][
        ::high_res_factor_dt, ::high_res_factor_dx
    ]

    return means, stds, ts, pde.mesh_spatial.points[::high_res_factor_dx]


def read_mean_and_std(sol, E0):
    means = sol.mean[:, 0]
    cov = sol.cov_sqrtm @ jnp.transpose(sol.cov_sqrtm, axes=(0, 2, 1))
    stds = jnp.sqrt(jnp.diagonal(cov, axis1=1, axis2=2) @ E0.T)
    return means, stds


def read_mean_and_std_latent(sol, E0):
    means = jnp.split(sol.mean, 2, axis=-1)[0]
    means = means[:, 0, :]
    cov = sol.cov_sqrtm @ jnp.transpose(sol.cov_sqrtm, axes=(0, 2, 1))
    vars = jnp.diagonal(cov, axis1=1, axis2=2)
    stds = jnp.sqrt(jnp.split(vars, 2, axis=-1)[0] @ E0.T)
    return means, stds


def save_result(result, /, *, prefix, path="experiments/results/figure1/"):
    means, stds, ts, xs = result
    path_means = path + prefix + "_means.npy"
    path_stds = path + prefix + "_stds.npy"
    path_ts = path + prefix + "_ts.npy"
    path_xs = path + prefix + "_xs.npy"
    jnp.save(path_means, means)
    jnp.save(path_stds, stds)
    jnp.save(path_ts, ts)
    jnp.save(path_xs, xs)


# Hyperparameters (method)
DT = 0.005
DX = 0.2
HIGH_RES_FACTOR_DX = 8
HIGH_RES_FACTOR_DT = 8
NUM_DERIVATIVES = 2
NUGGET_COV_FD = 0.0
STENCIL_SIZE = 3
PROGRESSBAR = True

INPUT_SCALE = 1.0
KERNEL = pnmol.kernels.Matern52(input_scale=INPUT_SCALE)

# Hyperparameters (problem)
T0, TMAX = 0.0, 3.0
DIFFUSION_RATE = 0.035


# PDE problems
with jax.disable_jit():
    PDE_PNMOL = pnmol.pde.examples.heat_1d_discretized(
        t0=T0,
        tmax=TMAX,
        dx=DX,
        stencil_size_interior=STENCIL_SIZE,
        stencil_size_boundary=STENCIL_SIZE + 1,
        diffusion_rate=DIFFUSION_RATE,
        kernel=KERNEL,
        nugget_gram_matrix_fd=NUGGET_COV_FD,
        bcond="dirichlet",
    )
    PDE_TORNADOX = pnmol.pde.examples.heat_1d_discretized(
        t0=T0,
        tmax=TMAX,
        dx=DX,
        stencil_size_interior=STENCIL_SIZE,
        stencil_size_boundary=STENCIL_SIZE + 1,
        diffusion_rate=DIFFUSION_RATE,
        kernel=KERNEL,
        nugget_gram_matrix_fd=NUGGET_COV_FD,
        bcond="dirichlet",
    )
    PDE_REFERENCE = pnmol.pde.examples.heat_1d_discretized(
        t0=T0,
        tmax=TMAX,
        dx=DX / HIGH_RES_FACTOR_DX,
        stencil_size_interior=STENCIL_SIZE,
        stencil_size_boundary=STENCIL_SIZE + 1,
        diffusion_rate=DIFFUSION_RATE,
        kernel=KERNEL,
        nugget_gram_matrix_fd=NUGGET_COV_FD,
        bcond="dirichlet",
    )

# Solve the PDE with the different methods
KERNEL_NUGGET = pnmol.kernels.WhiteNoise(output_scale=1e-7)
KERNEL_DIFFUSION_PNMOL = KERNEL  # + KERNEL_NUGGET

RESULT_PNMOL_WHITE = solve_pde_pnmol_white(
    PDE_PNMOL,
    dt=DT,
    nu=NUM_DERIVATIVES,
    progressbar=PROGRESSBAR,
    kernel=KERNEL_DIFFUSION_PNMOL,
)
RESULT_PNMOL_LATENT = solve_pde_pnmol_latent(
    PDE_PNMOL,
    dt=DT,
    nu=NUM_DERIVATIVES,
    progressbar=PROGRESSBAR,
    kernel=KERNEL_DIFFUSION_PNMOL,
)
RESULT_TORNADOX = solve_pde_tornadox(
    PDE_TORNADOX, dt=DT, nu=NUM_DERIVATIVES, progressbar=PROGRESSBAR
)
RESULT_REFERENCE = solve_pde_reference(
    PDE_REFERENCE,
    dt=DT / HIGH_RES_FACTOR_DT,
    high_res_factor_dt=HIGH_RES_FACTOR_DT,
    high_res_factor_dx=HIGH_RES_FACTOR_DX,
)
save_result(RESULT_PNMOL_WHITE, prefix="pnmol_white")
save_result(RESULT_PNMOL_LATENT, prefix="pnmol_latent")
save_result(RESULT_TORNADOX, prefix="tornadox")
save_result(RESULT_REFERENCE, prefix="reference")


plotting.figure_1()
