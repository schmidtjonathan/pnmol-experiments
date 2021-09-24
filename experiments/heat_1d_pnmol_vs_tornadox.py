"""A simple sanity check that the meascov solvers work on the 1d heat equation."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox

import pnmol

SAVE = True


def solve_pde_pnmol(pde, steprule, nu, progressbar):
    # Solve the discretised PDE
    ek1 = pnmol.solver.MeasurementCovarianceEK1(num_derivatives=nu, steprule=steprule)
    sol = ek1.solve(pde, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    return read_mean_and_std(sol, E0), sol.t


def solve_pde_tornadox(pde, steprule, nu, progressbar):
    ivp = pde.to_tornadox_ivp_1d()
    ek1 = tornadox.ek1.ReferenceEK1(num_derivatives=nu, steprule=steprule)
    sol = ek1.solve(ivp, progressbar=progressbar)
    E0 = ek1.iwp.projection_matrix(0)
    means, stds = read_mean_and_std(sol, E0)

    means = jnp.pad(means, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...]
    stds = jnp.pad(stds, pad_width=1, mode="constant", constant_values=0.0)[1:-1, ...]
    return (means, stds), sol.t


def read_mean_and_std(sol, E0):
    means = sol.mean[:, 0]
    cov = sol.cov_sqrtm @ jnp.transpose(sol.cov_sqrtm, axes=(0, 2, 1))
    stds = jnp.sqrt(jnp.diagonal(cov, axis1=1, axis2=2) @ E0.T)
    return means, stds


discretized_pde = pnmol.pde_problems.heat_1d(
    tmax=5.0, dx=0.1, stencil_size=5, diffusion_rate=0.05
)
constant_steps = pnmol.step.ConstantSteps(0.1)
nu = 2
res_pnmol = solve_pde_pnmol(discretized_pde, constant_steps, nu, True)
res_tornadox = solve_pde_tornadox(discretized_pde, constant_steps, nu, True)


# Plot mean and std


def plot_contour(ax, *args, **kwargs):
    """Contour lines with fill color and sharp edges."""
    ax.contour(*args, **kwargs)
    return ax.contourf(*args, **kwargs)


def infer_vrange_std(res):
    (_, stds), _ = res
    return jnp.amin(stds), jnp.amax(stds)


# Need this later
xgrid = discretized_pde.spatial_grid.points.squeeze()

# Create 2x2 grid
fig, axes = plt.subplots(
    ncols=2,
    nrows=2,
    dpi=300,
    figsize=(8, 4),
    sharex=True,
    sharey=True,
    constrained_layout=True,
)

# Plot the means and stds
#
# Reversed so the "last" colormap is the first row (PNMOL),
# which is set to be the figures colorbar below.
vmin_std, vmax_std = infer_vrange_std(res_pnmol)
for row_axes, res in zip(reversed(axes), reversed([res_pnmol, res_tornadox])):
    ax1, ax2 = row_axes
    (means, stds), tgrid = res
    T, X = jnp.meshgrid(xgrid, tgrid)
    plot_contour(
        ax1,
        X,
        T,
        means,
        cmap="Greys",
        levels=10,
        alpha=0.45,
        linewidths=0.8,
        vmin=0,
        vmax=0.5,
    )
    colors = plot_contour(
        ax2,
        X,
        T,
        stds,
        cmap="gist_rainbow",
        levels=10,
        alpha=0.45,
        linewidths=0.8,
        vmin=vmin_std,
        vmax=vmax_std,
    )

# Create a colorbar based on the PNMOL output (which is why we reversed the loop above)
fig.colorbar(
    colors,
    ax=axes[:, 1].ravel().tolist(),
    ticks=(vmin_std, 0.5 * (vmin_std + vmax_std), vmax_std),
)

# Column titles
top_row_axis = axes[0]
ax1, ax2 = top_row_axis
ax1.set_title(r"$\bf a.$ " + "Means", loc="left")
ax2.set_title(r"$\bf b.$ " + "Standard deviations", loc="left")

# x-labels
bottom_row_axis = axes[1]
for ax in bottom_row_axis:
    ax.set_xlabel("Time, $t$")

# y-labels
left_column_axis = axes[:, 0]
for ax, label in zip(left_column_axis, ["PNMOL", "MOL"]):
    ax.set_ylabel("Space, $x$ " + f"({label})")

# Common settings for all plots
for ax in axes.flatten():
    ax.set_xticks(tgrid)
    ax.set_yticks(xgrid)
    ax.set_xticklabels(())
    ax.set_yticklabels(())
    ax.grid(which="major", color="k", alpha=0.25, linestyle="dotted")

# Save the figure if desired and plot
if SAVE:
    plt.savefig("means_and_stds.pdf")
plt.show()
