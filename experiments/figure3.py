"""Code to generate figure 1."""

import itertools
import pathlib
import time

import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy
import plotting
import scipy.integrate
import tornadox
import tqdm

import pnmol


def solve_pde_reference(pde, *, high_res_factor_dx):
    t_eval = jnp.array([pde.tmax])
    ivp = pde.to_tornadox_ivp()
    sol = scipy.integrate.solve_ivp(
        ivp.f, ivp.t_span, ivp.y0, t_eval=t_eval, atol=1e-10, rtol=1e-10
    )

    mean = sol.y.T
    std = 0.0 * sol.y.T
    assert mean.shape == (1, ivp.y0.size) == std.shape
    mean, std = mean.squeeze(), std.squeeze()  # (highres * dx,)

    means = [
        m[high_res_factor_dx - 1 :: high_res_factor_dx] for m in jnp.split(mean, 3)
    ]  # (dx, )
    stds = [s[high_res_factor_dx - 1 :: high_res_factor_dx] for s in jnp.split(std, 3)]
    mean = jnp.concatenate(means)
    std = jnp.concatenate(stds)

    return mean, std, -1


def solve_pde_pnmol_white(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.odetools.step.Constant(dt)
    ek1 = pnmol.white.SemiLinearWhiteNoiseEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )

    start = time.time()
    final_state, _ = ek1.simulate_final_state(pde, progressbar=progressbar)
    elapsed_time = time.time() - start

    E0 = ek1.iwp.projection_matrix(0)
    mean, std = read_mean_and_std(final_state, E0)

    means = [m[1:-1] for m in jnp.split(mean, 3)]  # (dx, )
    stds = [s[1:-1] for s in jnp.split(std, 3)]
    mean = jnp.concatenate(means)
    std = jnp.concatenate(stds)

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
    path_dt = path / (prefix + "_dt.npy")
    path_dx = path / (prefix + "_dx.npy")

    jnp.save(path_error, result["error"])
    jnp.save(path_std, result["std"])
    jnp.save(path_runtime, result["runtime"])
    jnp.save(path_dt, result["dt"])
    jnp.save(path_dx, result["dx"])


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
    # numpy.log10(0.001), numpy.log10(0.01), num=10, endpoint=True, base=10
    numpy.log10(0.05),
    numpy.log10(0.5),
    num=10,
    endpoint=True,
    base=10,
)
DXs = jnp.logspace(numpy.log10(0.1), numpy.log10(0.25), num=10, endpoint=True, base=10)


# Hyperparameters (method)

HIGH_RES_FACTOR_DX = 1
NUM_DERIVATIVES = 2
NUGGET_COV_FD = 0.0
STENCIL_SIZE = 3
PROGRESSBAR = True

# Hyperparameters (problem)
T0, TMAX = 0.0, 5.0
DIFFUSION_RATE = 0.035


RESULT_WHITE, RESULT_TORNADOX = [
    {
        "error": numpy.zeros((len(DXs), len(DTs))),
        "std": numpy.zeros((len(DXs), len(DTs))),
        "runtime": numpy.zeros((len(DXs), len(DTs))),
        "dx": numpy.zeros((len(DXs), len(DTs))),
        "dt": numpy.zeros((len(DXs), len(DTs))),
    }
    for _ in range(2)
]

i_exp = 0
num_exp_total = len(DXs) * len(DTs)


save_dtdx()

# Solve the PDE with the different methods
KERNEL_DIFFUSION_PNMOL = pnmol.kernels.duplicate(
    pnmol.kernels.Matern52() + pnmol.kernels.WhiteNoise(), num=3
)

for i_dx, dx in enumerate(DXs):
    # PDE problems
    PDE_PNMOL = pnmol.pde.examples.sir_1d_discretized(
        t0=T0,
        tmax=TMAX,
        dx=dx,
        stencil_size_interior=STENCIL_SIZE,
        stencil_size_boundary=STENCIL_SIZE,
        diffusion_rate_S=DIFFUSION_RATE,
        diffusion_rate_I=DIFFUSION_RATE,
        diffusion_rate_R=DIFFUSION_RATE,
        nugget_gram_matrix_fd=NUGGET_COV_FD,
    )
    PDE_REFERENCE = pnmol.pde.examples.sir_1d_discretized(
        t0=T0,
        tmax=TMAX,
        dx=dx / HIGH_RES_FACTOR_DX,
        stencil_size_interior=STENCIL_SIZE,
        stencil_size_boundary=STENCIL_SIZE,
        diffusion_rate_S=DIFFUSION_RATE,
        diffusion_rate_I=DIFFUSION_RATE,
        diffusion_rate_R=DIFFUSION_RATE,
        nugget_gram_matrix_fd=NUGGET_COV_FD,
    )
    for i_dt, dt in enumerate(DTs):
        i_exp = i_exp + 1

        dim = PDE_PNMOL.y0.size
        print(
            f"\n======| Experiment {i_exp} of {num_exp_total} +++ dt={dt}, dx={dx} (state dimension: {dim} = 3 * {dim//3}) \n"
        )

        mean_white, std_white, elapsed_time_white = solve_pde_pnmol_white(
            PDE_PNMOL,
            dt=dt,
            nu=NUM_DERIVATIVES,
            progressbar=PROGRESSBAR,
            kernel=KERNEL_DIFFUSION_PNMOL,
        )

        mean_tornadox, std_tornadox, elapsed_time_tornadox = solve_pde_tornadox(
            PDE_PNMOL, dt=dt, nu=NUM_DERIVATIVES, progressbar=PROGRESSBAR
        )
        mean_reference, std_reference, elapsed_time_reference = solve_pde_reference(
            PDE_REFERENCE,
            high_res_factor_dx=HIGH_RES_FACTOR_DX,
        )

        error_white = jnp.linalg.norm(
            (mean_white - mean_reference) / mean_reference
        ) / jnp.sqrt(mean_reference.size)
        error_tornadox = jnp.linalg.norm(
            (mean_tornadox - mean_reference) / mean_reference
        ) / jnp.sqrt(mean_reference.size)

        mean_std_white = jnp.mean(std_white)
        mean_std_tornadox = jnp.mean(std_tornadox)

        RESULT_WHITE["error"][i_dx, i_dt] = error_white
        RESULT_WHITE["std"][i_dx, i_dt] = mean_std_white
        RESULT_WHITE["runtime"][i_dx, i_dt] = elapsed_time_white
        RESULT_WHITE["dt"][i_dx, i_dt] = dt
        RESULT_WHITE["dx"][i_dx, i_dt] = dx

        RESULT_TORNADOX["error"][i_dx, i_dt] = error_tornadox
        RESULT_TORNADOX["std"][i_dx, i_dt] = mean_std_tornadox
        RESULT_TORNADOX["runtime"][i_dx, i_dt] = elapsed_time_tornadox
        RESULT_TORNADOX["dt"][i_dx, i_dt] = dt
        RESULT_TORNADOX["dx"][i_dx, i_dt] = dx


save_result(RESULT_WHITE, prefix="pnmol_white")
save_result(RESULT_TORNADOX, prefix="tornadox")
