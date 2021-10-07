"""Code to generate figure 1."""

import itertools
import pathlib

import jax.numpy as jnp
import plotting
import scipy.integrate
import tornadox
import tqdm

import pnmol


def solve_pde_reference(pde, *, dt, high_res_factor_dx, high_res_factor_dt):
    t_eval = jnp.array([pde.tmax])
    ivp = pde.to_tornadox_ivp()
    sol = scipy.integrate.solve_ivp(ivp.f, ivp.t_span, ivp.y0, t_eval=t_eval)

    means = sol.y.T
    assert means.shape == (1, ivp.y0.size)
    stds = 0.0 * sol.y.T

    means = jnp.pad(means, pad_width=1, mode="constant", constant_values=0.0)[
        1:-1, ...
    ][::high_res_factor_dt, ::high_res_factor_dx]
    stds = jnp.pad(stds, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...][
        ::high_res_factor_dt, ::high_res_factor_dx
    ]

    return means, stds, t_eval[0], pde.mesh_spatial.points[::high_res_factor_dx], -1


def solve_pde_pnmol_white(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.odetools.step.Constant(dt)
    ek1 = pnmol.white.LinearWhiteNoiseEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )
    final_state, info = ek1.simulate_final_state(pde, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    mean, std = read_mean_and_std(final_state, E0)

    return mean, std, final_state.t, pde.mesh_spatial.points, info["num_steps"]


# def solve_pde_pnmol_latent(pde, *, dt, nu, progressbar, kernel):
#     steprule = pnmol.odetools.step.Constant(dt)
#     ek1 = pnmol.latent.LinearLatentForceEK1(
#         num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
#     )
#     final_state, info = ek1.simulate_final_state(pde, progressbar=progressbar)
#     E0 = ek1.state_iwp.projection_matrix(0)
#     mean, std = read_mean_and_std_latent(final_state, E0)

#     return mean, std, final_state.t, pde.mesh_spatial.points, info["num_steps"]


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
    return mean, std, final_state.t, pde.mesh_spatial.points, info["num_steps"]


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

    means = {k: v[0] for (k, v) in result.items()}
    stds = {k: v[1] for (k, v) in result.items()}
    ts = {k: v[2] for (k, v) in result.items()}
    xs = {k: v[3] for (k, v) in result.items()}
    n_steps = {k: v[4] for (k, v) in result.items()}

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


RESULT_PNMOL_WHITE, RESULT_TORNADOX, RESULT_REFERENCE = {}, {}, {}


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
    PDE_REFERENCE = pnmol.pde.examples.heat_1d_discretized(
        t0=T0,
        tmax=TMAX,
        dx=dx / HIGH_RES_FACTOR_DX,
        stencil_size=STENCIL_SIZE,
        diffusion_rate=DIFFUSION_RATE,
        kernel=pnmol.kernels.SquareExponential(),
        nugget_gram_matrix_fd=NUGGET_COV_FD,
        bcond="dirichlet",
    )

    # Solve the PDE with the different methods
    KERNEL_NUGGET = pnmol.kernels.WhiteNoise(output_scale=1e-3)
    KERNEL_DIFFUSION_PNMOL = pnmol.kernels.Matern52() + KERNEL_NUGGET
    res_white = solve_pde_pnmol_white(
        PDE_PNMOL,
        dt=dt,
        nu=NUM_DERIVATIVES,
        progressbar=PROGRESSBAR,
        kernel=KERNEL_DIFFUSION_PNMOL,
    )

    res_tornadox = solve_pde_tornadox(
        PDE_TORNADOX, dt=dt, nu=NUM_DERIVATIVES, progressbar=PROGRESSBAR
    )
    res_reference = solve_pde_reference(
        PDE_REFERENCE,
        dt=dt / HIGH_RES_FACTOR_DT,
        high_res_factor_dt=HIGH_RES_FACTOR_DT,
        high_res_factor_dx=HIGH_RES_FACTOR_DX,
    )

    print(res_white[2], res_tornadox[2], res_reference[2])

    RESULT_PNMOL_WHITE[(dt, dx)] = res_white
    RESULT_TORNADOX[(dt, dx)] = res_tornadox
    RESULT_REFERENCE[(dt, dx)] = res_reference


save_result(RESULT_PNMOL_WHITE, prefix="pnmol_white")
save_result(RESULT_TORNADOX, prefix="tornadox")
save_result(RESULT_REFERENCE, prefix="reference")
