"""Code to generate figure 1."""

import itertools
import pathlib
import time

import jax.numpy as jnp
import numpy
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

    return means, stds, -1


def solve_pde_pnmol_white(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.odetools.step.Constant(dt)
    ek1 = pnmol.white.LinearWhiteNoiseEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )

    start = time.time()
    final_state, _ = ek1.simulate_final_state(pde, progressbar=progressbar)
    elapsed_time = time.time() - start

    E0 = ek1.iwp.projection_matrix(0)
    mean, std = read_mean_and_std(final_state, E0)

    return mean, std, elapsed_time


def solve_pde_tornadox(pde, *, dt, nu, progressbar):
    steprule = tornadox.step.ConstantSteps(dt)
    ivp = pde.to_tornadox_ivp()
    ek1 = tornadox.ek1.ReferenceEK1(
        num_derivatives=nu, steprule=steprule, initialization=tornadox.init.RungeKutta()
    )

    start = time.time()
    final_state, info = ek1.simulate_final_state(ivp, progressbar=progressbar)
    elapsed_time = time.time() - start

    E0 = ek1.iwp.projection_matrix(0)
    mean, std = read_mean_and_std(final_state, E0)

    mean = jnp.pad(mean, pad_width=1, mode="constant", constant_values=0.0)
    std = jnp.pad(std, pad_width=1, mode="constant", constant_values=0.0)

    return mean, std, elapsed_time


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
    path = pathlib.Path(path) / "figure3"
    if not path.is_dir():
        path.mkdir(parents=True)

    path_error = path / (prefix + "_error.npy")
    path_std = path / (prefix + "_std.npy")
    path_runtime = path / (prefix + "_runtime.npy")

    jnp.save(path_error, result["error"])
    jnp.save(path_std, result["std"])
    jnp.save(path_runtime, result["runtime"])


def save_dtdx(path="experiments/results"):
    path = pathlib.Path(path) / "figure3"
    if not path.is_dir():
        path.mkdir(parents=True)

    dtdxsavepath = path / "dtdx.npy"
    dtdx_array = jnp.stack((DTs, DXs))
    jnp.save(dtdxsavepath, dtdx_array)
    print(f"Saved dtdx of shape {dtdx_array.shape} to {dtdxsavepath}")


# Ranges
DTs = jnp.logspace(
    numpy.log10(0.001), numpy.log10(0.01), num=15, endpoint=True, base=10
)  # jnp.logspace(-2, 0, num=25)
DXs = jnp.logspace(
    numpy.log10(0.1), numpy.log10(0.25), num=15, endpoint=True, base=10
)  # jnp.logspace(-2, 0, num=25)


# Hyperparameters (method)

HIGH_RES_FACTOR_DX = 5
HIGH_RES_FACTOR_DT = 5
NUM_DERIVATIVES = 2
NUGGET_COV_FD = 0.0
STENCIL_SIZE = 3
PROGRESSBAR = True

# Hyperparameters (problem)
T0, TMAX = 0.0, 4.0
DIFFUSION_RATE = 0.035


RESULT_WHITE, RESULT_TORNADOX = [
    {
        "error": numpy.zeros((len(DXs), len(DTs))),
        "std": numpy.zeros((len(DXs), len(DTs))),
        "runtime": numpy.zeros((len(DXs), len(DTs))),
    }
    for _ in range(2)
]

i_exp = 0
num_exp_total = len(DXs) * len(DTs)


save_dtdx()

for i_dx, dx in enumerate(DXs):
    for i_dt, dt in enumerate(DTs):
        i_exp = i_exp + 1

        print(
            f"\n======| Experiment {i_exp} of {num_exp_total} +++ dt={dt}, dx={dx} \n"
        )

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

        mean_white, std_white, elapsed_time_white = solve_pde_pnmol_white(
            PDE_PNMOL,
            dt=dt,
            nu=NUM_DERIVATIVES,
            progressbar=PROGRESSBAR,
            kernel=KERNEL_DIFFUSION_PNMOL,
        )

        mean_tornadox, std_tornadox, elapsed_time_tornadox = solve_pde_tornadox(
            PDE_TORNADOX, dt=dt, nu=NUM_DERIVATIVES, progressbar=PROGRESSBAR
        )
        mean_reference, std_reference, elapsed_time_reference = solve_pde_reference(
            PDE_REFERENCE,
            dt=dt / HIGH_RES_FACTOR_DT,
            high_res_factor_dt=HIGH_RES_FACTOR_DT,
            high_res_factor_dx=HIGH_RES_FACTOR_DX,
        )

        error_white = jnp.mean(jnp.abs(mean_white - mean_reference))
        error_tornadox = jnp.mean(jnp.abs(mean_tornadox - mean_reference))

        mean_std_white = jnp.mean(std_white)
        mean_std_tornadox = jnp.mean(std_tornadox)

        RESULT_WHITE["error"][i_dx, i_dt] = error_white
        RESULT_WHITE["std"][i_dx, i_dt] = mean_std_white
        RESULT_WHITE["runtime"][i_dx, i_dt] = elapsed_time_white

        RESULT_TORNADOX["error"][i_dx, i_dt] = error_tornadox
        RESULT_TORNADOX["std"][i_dx, i_dt] = mean_std_tornadox
        RESULT_TORNADOX["runtime"][i_dx, i_dt] = elapsed_time_tornadox


save_result(RESULT_WHITE, prefix="pnmol_white")
save_result(RESULT_TORNADOX, prefix="tornadox")
