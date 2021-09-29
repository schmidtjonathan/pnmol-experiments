"""A simple sanity check that the meascov solvers work on the 1d heat equation."""
import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt

import pnmol

SAVE = True

# Create the PDE problem
discretized_pde = pnmol.pde_problems.heat_1d(
    tmax=5.0, dx=0.05, stencil_size=5, diffusion_rate=0.05
)

# Solve the discretised PDE
constant_steps = pnmol.step.ConstantSteps(0.1)
nu = 2
ek1 = pnmol.solver.MeasurementCovarianceEK1(num_derivatives=nu, steprule=constant_steps)
sol = ek1.solve(discretized_pde, progressbar=True)

# Read out mean and std for plotting.
E0 = ek1.iwp.projection_matrix(0)
means = sol.mean[:, 0]
cov = sol.cov_sqrtm @ jnp.transpose(sol.cov_sqrtm, axes=(0, 2, 1))
stds = jnp.sqrt(jnp.diagonal(cov, axis1=1, axis2=2) @ E0.T)

# Plot mean and std

xgrid = discretized_pde.spatial_grid.points.squeeze()
tgrid = sol.t

T, X = jnp.meshgrid(xgrid, tgrid)


def plot_contour(ax, *args, **kwargs):
    """Contour lines with fill color and sharp edges."""
    ax.contour(*args, **kwargs)
    ax.contourf(*args, **kwargs)


fig, (ax1, ax2) = plt.subplots(
    ncols=2, dpi=300, figsize=(8, 3), sharex=True, sharey=True, constrained_layout=True
)
plot_contour(ax1, X, T, means, cmap="gist_rainbow", alpha=0.45, linewidths=0.8)
plot_contour(ax2, X, T, stds, cmap="gist_rainbow", alpha=0.45, linewidths=0.8)


ax1.set_title(r"$\bf a.$ " + "Means", loc="left")
ax2.set_title(r"$\bf b.$ " + "Standard deviations", loc="left")

for ax in [ax1, ax2]:
    ax.set_xlabel("Time, $t$")
    ax.set_xticks(tgrid)
    ax.set_yticks(xgrid)
    ax.set_xticklabels(())
    ax.set_yticklabels(())
ax1.set_ylabel("Space, $x$")

ax1.grid(which="major", color="k", alpha=0.25, linestyle="dotted")
ax2.grid(which="major", color="k", alpha=0.25, linestyle="dotted")

if SAVE:
    plt.savefig("means_and_stds.pdf")
plt.show()
