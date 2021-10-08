"""Work-precision diagrams and so on."""


import jax.numpy as jnp
import matplotlib.pyplot as plt

import pnmol

k = pnmol.kernels.duplicate(
    pnmol.kernels.Matern52() + pnmol.kernels.WhiteNoise(), num=2
)

pde = pnmol.pde.examples.lotka_volterra_1d_discretized(
    tmax=20.0,
    dx=0.05,
)

for tol in jnp.logspace(-1, -3, 5):

    steps = pnmol.odetools.step.Adaptive(abstol=0.01 * tol, reltol=tol)
    solver = pnmol.white.SemiLinearWhiteNoiseEK1(steprule=steps, spatial_kernel=k)

    sol = solver.solve(pde, progressbar=True)
    print(sol.info)


u, v = jnp.split(sol.mean, 2, axis=-1)

fig, axes = plt.subplots(ncols=6)
ax_dt, ax_sol1, ax_sol2 = axes[:3]
ax_single_curve1, ax_single_curve2, ax_single_curve3 = axes[3:]


ax_dt.semilogy(jnp.diff(sol.t))
ax_dt.set_title("Step-size")

ax_sol1.imshow(u[:, 0, :].T, aspect="auto", vmin=0.0, cmap="Blues")
ax_sol2.imshow(v[:, 0, :].T, aspect="auto", vmin=0.0, cmap="Oranges")

ax_sol1.set_title("Predators")
ax_sol2.set_title("Prey")


ax_single_curve1.plot(sol.t, u[:, 0, 0])
ax_single_curve1.plot(sol.t, v[:, 0, 0])
ax_single_curve2.plot(sol.t, u[:, 0, 10])
ax_single_curve2.plot(sol.t, v[:, 0, 10])
ax_single_curve3.plot(sol.t, u[:, 0, -1])
ax_single_curve3.plot(sol.t, v[:, 0, -1])

for ax in [ax_single_curve1, ax_single_curve2, ax_single_curve3]:
    ax.set_ylim((0.0, 25.0))
plt.show()
