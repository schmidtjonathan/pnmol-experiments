"""Code to generate figure 1."""

import itertools
import pathlib

import jax.numpy as jnp
import plotting
import scipy.integrate
import tornadox
import tqdm

import pnmol


def solve_pde_pnmol_white(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.odetools.step.Constant(dt)
    ek1 = pnmol.white.LinearWhiteNoiseEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )
    final_state, info = ek1.simulate_final_state(pde, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    mean, std = read_mean_and_std(final_state, E0)

    return mean, std, final_state.t, pde.mesh_spatial.points, info


def solve_pde_pnmol_latent(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.odetools.step.Constant(dt)
    ek1 = pnmol.latent.LinearLatentForceEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )
    final_state, info = ek1.simulate_final_state(pde, progressbar=progressbar)
    E0 = ek1.state_iwp.projection_matrix(0)
    mean, std = read_mean_and_std_latent(final_state, E0)

    return mean, std, final_state.t, pde.mesh_spatial.points, info


def solve_pde_tornadox(pde, *, dt, nu, progressbar):
    steprule = tornadox.step.ConstantSteps(dt)
    ivp = pde.to_tornadox_ivp()
    ek1 = tornadox.ek1.ReferenceEK1(
        num_derivatives=nu, steprule=steprule, initialization=tornadox.init.RungeKutta()
    )
    final_state, info = ek1.simulate_final_state(ivp, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    mean, std = read_mean_and_std(final_state, E0)

    mean = jnp.pad(mean, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...]
    std = jnp.pad(std, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...]
    return mean, std, final_state.t, pde.mesh_spatial.points, info


def read_mean_and_std(final_state, E0):
    # print("White")
    # print(final_state.y.mean.shape, final_state.y.cov_sqrtm.shape)
    mean = final_state.y.mean[0, :]
    cov = final_state.y.cov_sqrtm @ final_state.y.cov_sqrtm.T
    std = E0 @ jnp.sqrt(jnp.diagonal(cov))
    return mean, std


def read_mean_and_std_latent(final_state, E0):
    # print("Latent")
    # print(final_state.y.mean.shape, final_state.y.cov_sqrtm.shape)
    mean, _ = jnp.split(final_state.y.mean[0, :], 2)
    cov = final_state.y.cov_sqrtm @ final_state.y.cov_sqrtm.T
    varis = jnp.diagonal(cov)
    std = E0 @ jnp.sqrt(jnp.split(varis, 2)[0])
    return mean, std


def save_result(result, /, *, prefix, path="experiments/results"):
    path = pathlib.Path(path) / "figure_main"
    if not path.is_dir():
        path.mkdir(parents=True)

    means, stds, ts, xs, info = result
    n_steps = info["num_steps"]
    path_means = path / (prefix + "_means.npy")
    path_stds = path / (prefix + "_stds.npy")
    path_ts = path / (prefix + "_ts.npy")
    path_xs = path / (prefix + "_xs.npy")
    path_nsteps = path / (prefix + "_numsteps.npy")
    jnp.save(path_means, means)
    jnp.save(path_stds, stds)
    jnp.save(path_ts, ts)
    jnp.save(path_xs, xs)
    jnp.save(path_nsteps, n_steps)


# Ranges
DTs = [0.1, 0.2, 0.5]  # [0.01, 0.05, 0.1, 0.2, 0.5]
DXs = [0.1, 0.2, 0.5]  # [0.01, 0.05, 0.1, 0.2, 0.5]

# Hyperparameters (method)

HIGH_RES_FACTOR_DX = 10
HIGH_RES_FACTOR_DT = 10
NUM_DERIVATIVES = 2
NUGGET_COV_FD = 0.0
STENCIL_SIZE = 3
PROGRESSBAR = True

# Hyperparameters (problem)
T0, TMAX = 0.0, 4.0
DIFFUSION_RATE = 0.035


for exp_i, (dt, dx) in enumerate(itertools.product(DTs, DXs), start=1):

    print(f"\n======| Experiment {exp_i}: dt={dt}, dx={dx} \n")

    # PDE problems
    PDE_PNMOL = pnmol.pde.examples.heat_1d_discretized(
        t0=T0,
        tmax=TMAX,
        dx=dx,
        stencil_size=STENCIL_SIZE,
        diffusion_rate=DIFFUSION_RATE,
        kernel=pnmol.kernels.SquareExponential(),
        nugget_gram_matrix_fd=NUGGET_COV_FD,
        bcond="dirichlet",
    )
    PDE_TORNADOX = pnmol.pde.examples.heat_1d_discretized(
        t0=T0,
        tmax=TMAX,
        dx=dx,
        stencil_size=STENCIL_SIZE,
        diffusion_rate=DIFFUSION_RATE,
        kernel=pnmol.kernels.SquareExponential(),
        nugget_gram_matrix_fd=NUGGET_COV_FD,
        bcond="dirichlet",
    )

    # Solve the PDE with the different methods
    KERNEL_NUGGET = pnmol.kernels.WhiteNoise(output_scale=1e-3)
    KERNEL_DIFFUSION_PNMOL = pnmol.kernels.Matern52() + KERNEL_NUGGET
    RESULT_PNMOL_WHITE = solve_pde_pnmol_white(
        PDE_PNMOL,
        dt=dt,
        nu=NUM_DERIVATIVES,
        progressbar=PROGRESSBAR,
        kernel=KERNEL_DIFFUSION_PNMOL,
    )
    # RESULT_PNMOL_LATENT = solve_pde_pnmol_latent(
    #     PDE_PNMOL,
    #     dt=dt,
    #     nu=NUM_DERIVATIVES,
    #     progressbar=PROGRESSBAR,
    #     kernel=KERNEL_DIFFUSION_PNMOL,
    # )
    RESULT_TORNADOX = solve_pde_tornadox(
        PDE_TORNADOX, dt=dt, nu=NUM_DERIVATIVES, progressbar=PROGRESSBAR
    )

    save_result(RESULT_PNMOL_WHITE, prefix="pnmol_white")
    # save_result(RESULT_PNMOL_LATENT, prefix="pnmol_latent")
    save_result(RESULT_TORNADOX, prefix="tornadox")
