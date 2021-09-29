"""Code to generate figure 1."""

import pathlib

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import plotting
import tornadox

import pnmol


def solve_pde_pnmol_white(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.ode.step.Constant(dt)
    ek1 = pnmol.white.LinearWhiteNoiseEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )
    sol = ek1.solve(pde, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    return *read_mean_and_std(sol, E0), sol.t, pde.spatial_grid.points


def solve_pde_pnmol_latent(pde, *, dt, nu, progressbar, kernel):
    steprule = pnmol.ode.step.Constant(dt)
    ek1 = pnmol.latent.LinearLatentForceEK1(
        num_derivatives=nu, steprule=steprule, spatial_kernel=kernel
    )
    sol = ek1.solve(pde, progressbar=progressbar)
    E0 = ek1.state_iwp.projection_matrix(0)
    return *read_mean_and_std_latent(sol, E0), sol.t, pde.spatial_grid.points


def solve_pde_tornadox(pde, *, dt, nu, progressbar):
    steprule = tornadox.step.ConstantSteps(dt)
    ivp = pde.to_tornadox_ivp_1d()
    ek1 = tornadox.ek1.ReferenceEK1(
        num_derivatives=nu, steprule=steprule, initialization=tornadox.init.RungeKutta()
    )
    sol = ek1.solve(ivp, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    means, stds = read_mean_and_std(sol, E0)

    means = jnp.pad(means, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...]
    stds = jnp.pad(stds, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...]
    return means, stds, sol.t, pde.spatial_grid.points


def solve_pde_reference(
    pde, *, dt, nu, progressbar, high_res_factor_dx, high_res_factor_dt
):
    steprule = tornadox.step.ConstantSteps(dt)
    ivp = pde.to_tornadox_ivp_1d()
    ek1 = tornadox.ek1.ReferenceEK1(
        num_derivatives=nu, steprule=steprule, initialization=tornadox.init.RungeKutta()
    )
    sol = ek1.solve(ivp, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    means, stds = read_mean_and_std(sol, E0)

    means = jnp.pad(means, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...]
    stds = jnp.pad(stds, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...]
    return (
        means[::high_res_factor_dt, ::high_res_factor_dx],
        stds[::high_res_factor_dt, ::high_res_factor_dx],
        sol.t[::high_res_factor_dt],
        pde.spatial_grid.points,
    )


def read_mean_and_std(sol, E0):
    means = sol.mean[:, 0]
    cov = sol.cov_sqrtm @ jnp.transpose(sol.cov_sqrtm, axes=(0, 2, 1))
    stds = jnp.sqrt(jnp.diagonal(cov, axis1=1, axis2=2) @ E0.T)
    return means, stds


def read_mean_and_std_latent(sol, E0):
    d = E0.shape[0]
    n = E0.shape[1] // d
    means = jnp.split(sol.mean, 2, axis=-1)[0][:, 0]
    cov = sol.cov_sqrtm @ jnp.transpose(sol.cov_sqrtm, axes=(0, 2, 1))
    vars = jnp.diagonal(cov, axis1=1, axis2=2)
    stds = jnp.sqrt(
        jnp.split(vars, 2, axis=-1)[0].reshape((cov.shape[0], n, -1), order="F")[
            :, 0, :
        ]
    )
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
DT = 0.01
DX = 0.2
HIGH_RES_FACTOR_DX = 12
HIGH_RES_FACTOR_DT = 2
NUM_DERIVATIVES = 2
NUGGET_COV_FD = 0.0
STENCIL_SIZE = 3
PROGRESSBAR = True

# Hyperparameters (problem)
T0, TMAX = 0.0, 5.0
DIFFUSION_RATE = 0.05


# PDE problems
PDE_PNMOL = pnmol.problems.heat_1d(
    t0=T0,
    tmax=TMAX,
    dx=DX,
    stencil_size=STENCIL_SIZE,
    diffusion_rate=DIFFUSION_RATE,
    kernel=pnmol.kernels.SquareExponential(),
    cov_damping_fd=NUGGET_COV_FD,
)
PDE_TORNADOX = pnmol.problems.heat_1d(
    t0=T0,
    tmax=TMAX,
    dx=DX,
    stencil_size=STENCIL_SIZE,
    diffusion_rate=DIFFUSION_RATE,
    kernel=pnmol.kernels.Polynomial(),
    cov_damping_fd=NUGGET_COV_FD,
)
PDE_REFERENCE = pnmol.problems.heat_1d(
    t0=T0,
    tmax=TMAX,
    dx=DX / HIGH_RES_FACTOR_DX,
    stencil_size=STENCIL_SIZE,
    diffusion_rate=DIFFUSION_RATE,
    kernel=pnmol.kernels.Polynomial(),
    cov_damping_fd=NUGGET_COV_FD,
)

# Solve the PDE with the different methods
KERNEL_DIFFUSION_PNMOL = pnmol.kernels.Matern52() + pnmol.kernels.WhiteNoise(
    output_scale=1e-7
)
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
    nu=NUM_DERIVATIVES,
    progressbar=PROGRESSBAR,
    high_res_factor_dt=HIGH_RES_FACTOR_DT,
    high_res_factor_dx=HIGH_RES_FACTOR_DX,
)
save_result(RESULT_PNMOL_WHITE, prefix="pnmol_white")
save_result(RESULT_PNMOL_LATENT, prefix="pnmol_latent")
save_result(RESULT_TORNADOX, prefix="tornadox")
save_result(RESULT_REFERENCE, prefix="reference")


plotting.figure_1()
#
# assert False
#
# # Plot mean and std
#
#
# def plot_contour(ax, *args, **kwargs):
#     """Contour lines with fill color and sharp edges."""
#     ax.contour(*args, **kwargs)
#     return ax.contourf(*args, **kwargs)
#
#
# def infer_vrange_std(res):
#     (_, stds), _ = res
#     return jnp.amin(stds), jnp.amax(stds)
#
#
# # Need this later
# xgrid = discretized_pde_pnmol.spatial_grid.points.squeeze()
# xgrid2 = discretized_pde_high_res.spatial_grid.points.squeeze()
#
# # Create 2x2 grid
# fig, axes = plt.subplots(
#     ncols=3,
#     nrows=2,
#     dpi=400,
#     figsize=(5, 3),
#     sharex=True,
#     sharey=True,
#     constrained_layout=True,
# )
#
# # Plot the means and stds
# #
# # Reversed so the "last" colormap is the first row (PNMOL),
# # which is set to be the figures colorbar below.
# vmin_std, vmax_std = infer_vrange_std(res_pnmol)
#
# (means_ref, _), _ = res_reference
# for i, (row_axes, res) in enumerate(
#     zip(reversed(axes), reversed([res_pnmol, res_tornadox]))
# ):
#     ax1, ax2, ax3 = row_axes
#     (means, stds), tgrid = res
#     T, X = jnp.meshgrid(xgrid, tgrid)
#     plot_contour(
#         ax1,
#         X,
#         T,
#         means,
#         cmap="Greys",
#         alpha=0.45,
#         linewidths=0.8,
#         vmin=0,
#         vmax=0.5,
#     )
#     colors = plot_contour(
#         ax2,
#         X,
#         T,
#         stds,
#         cmap="seismic",
#         alpha=0.45,
#         linewidths=0.8,
#         vmin=vmin_std,
#         vmax=vmax_std,
#     )
#     n = jnp.minimum(len(means), len(means_ref))
#     plot_contour(
#         ax3,
#         X[:n],
#         T[:n],
#         jnp.abs(means[:n] - means_ref[:n]),
#         cmap="seismic",
#         alpha=0.45,
#         linewidths=0.8,
#         vmin=vmin_std,
#         vmax=vmax_std,
#     )
#
# # Create a colorbar based on the PNMOL output (which is why we reversed the loop above)
# fig.colorbar(
#     colors,
#     ,
#     ,
# )

#
# # Column titles
# top_row_axis = axes[0]
# ax1, ax2, ax3 = top_row_axis
# ax1.set_title(r"$\bf a.$ " + "Mean", loc="left")
# ax2.set_title(r"$\bf b.$ " + "Std.-dev.", loc="left")
# ax3.set_title(r"$\bf c.$ " + "Error", loc="left")
#
# # x-labels
# bottom_row_axis = axes[1]
# for ax in bottom_row_axis:
#     ax.set_xlabel("Time, $t$")
#
# # y-labels
# left_column_axis = axes[:, 0]
# for ax, label in zip(left_column_axis, ["PNMOL", "MOL"]):
#     ax.set_ylabel("Space, $x$ " + f"({label})")
#
# # Common settings for all plots
# for ax in axes.flatten():
#     ax.set_xticks(tgrid)
#     ax.set_yticks(xgrid)
#     ax.set_xticklabels(())
#     ax.set_yticklabels(())
#     ax.grid(which="major", color="k", alpha=0.25, linestyle="dotted")
#
# # Save the figure if desired and plot
# if SAVE:
#     plt.savefig("means_and_stds.pdf")
# plt.show()
