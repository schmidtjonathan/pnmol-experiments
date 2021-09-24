"""A simple sanity check that the meascov solvers work on the 1d heat equation."""

import jax.numpy as jnp

import matplotlib.pyplot as plt
import tornadox
from matplotlib import animation

import pnmol
import jax

ANIMATION = False

# Create the PDE problem
discretized_pde = pnmol.pde_problems.heat_1d(tmax=5., dx=0.05, stencil_size=5, diffusion_rate=0.05)

# Solve the discretised PDE
constant_steps = tornadox.step.ConstantSteps(0.1)
nu = 2
ek1 = pnmol.solver.MeasurementCovarianceEK1(num_derivatives=nu, steprule=constant_steps)
sol = ek1.solve(ivp=discretized_pde, compile_step=False, progressbar=True)

# Read out mean and std for plotting.
E0 = ek1.iwp.projection_matrix(0)
means = sol.mean[:, 0]
cov = sol.cov_sqrtm @ jnp.transpose(sol.cov_sqrtm, axes=(0,2,1))
stds = jnp.sqrt(jnp.diagonal(cov, axis1=1, axis2=2) @ E0.T)

# Plot mean and std
fig, (ax1, ax2) = plt.subplots(ncols=2, dpi=200)
mean_map = ax1.imshow(means.T, cmap="cividis", vmin=0., vmax=1.)
std_map = ax2.imshow(jnp.log(stds.T), cmap="cividis")
plt.tight_layout()
plt.show()



# Create an animation

if ANIMATION:
    plt.rcParams[
        "animation.embed_limit"] = 2 * 10 ** 8  # Set the animation max size to 200MB

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
        fig, func=animate, frames=len(sol.t), interval=100, repeat_delay=4000,
        blit=False
    )
    anim.save("anim_heat_1d.gif")
