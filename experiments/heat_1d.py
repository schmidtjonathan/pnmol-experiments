"""A simple sanity check that the meascov solvers work on the 1d heat equation."""

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox
from matplotlib import animation

import pnmol

ANIMATION = False
SAVE = True

# Create the PDE problem
discretized_pde = pnmol.pde_problems.heat_1d(
    tmax=5.0, dx=0.05, stencil_size=5, diffusion_rate=0.05
)

# Solve the discretised PDE
constant_steps = tornadox.step.ConstantSteps(0.1)
nu = 2
ek1 = pnmol.solver.MeasurementCovarianceEK1(num_derivatives=nu, steprule=constant_steps)
sol = ek1.solve(ivp=discretized_pde, compile_step=False, progressbar=True)

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

fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=100, figsize=(8,3), sharex=True, sharey=True, constrained_layout=True)
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


# Create an animation

if ANIMATION:
    plt.rcParams["animation.embed_limit"] = (
        2 * 10 ** 8
    )  # Set the animation max size to 200MB

    grid = discretized_pde.spatial_grid

    fig = plt.figure(figsize=(20, 8))
    ax = fig.add_subplot(1, 1, 1)
    _im1 = ax.plot(grid.points.squeeze(), sol.mean[0, 0])
    _im1 = ax.plot(grid.points.squeeze(), discretized_pde.y0.squeeze())

    ax1ylim = [-0.2, 1.2]
    ax.set_ylim(ax1ylim)

    plt.close()

    cov = sol.cov_sqrtm @ jnp.transpose(sol.cov_sqrtm, axes=(0, 2, 1))

    def animate(i):
        mean = sol.mean[i, 0]
        std = 0 * jnp.sqrt(jnp.diag(cov[i]))[0]

        ax.cla()
        ax.set_title(f"t={sol.t[i]}")
        ax.plot(grid.points.squeeze(), mean, color="C0", label="PN solution")
        ax.fill_between(
            grid.points.squeeze(),
            mean - 2 * std,
            mean + 2 * std,
            color="C0",
            alpha=0.2,
        )
        ax.set_ylim(ax1ylim)
        ax.legend()

    # Animation setup
    anim = animation.FuncAnimation(
        fig,
        func=animate,
        frames=len(sol.t),
        interval=100,
        repeat_delay=4000,
        blit=False,
    )
    anim.save("anim_heat_1d.gif")
