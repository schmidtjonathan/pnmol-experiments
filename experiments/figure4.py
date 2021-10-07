"""Work-precision diagrams and so on."""


import jax.numpy as jnp
import matplotlib.pyplot as plt

import pnmol

steps = pnmol.odetools.step.Adaptive(abstol=1e-5, reltol=1e-7)
k = pnmol.kernels.duplicate(
    pnmol.kernels.Matern52() + pnmol.kernels.WhiteNoise(), num=3
)
solver = pnmol.white.SemiLinearWhiteNoiseEK1(steprule=steps, spatial_kernel=k)


pde = pnmol.pde.examples.sir_1d_discretized(
    tmax=20.0,
    dx=0.2,
    diffusion_rate_S=0.1,
    diffusion_rate_I=0.1,
    diffusion_rate_R=0.1,
)


sol = solver.solve(pde, progressbar=True)

s, i, r = jnp.split(sol.mean, 3, axis=-1)

fig, (
    ax_dt,
    ax_sol,
    ax_single_curve1,
    ax_single_curve2,
    ax_single_curve3,
) = plt.subplots(ncols=5)

ax_dt.semilogy(jnp.diff(sol.t))


# bar = ax_sol.imshow(means[:, 0, :].T, aspect="auto", vmin=0.)
# fig.colorbar(bar, ax=ax_sol)

ax_single_curve1.plot(sol.t, s[:, 0, 0])
ax_single_curve1.plot(sol.t, i[:, 0, 0])
ax_single_curve1.plot(sol.t, r[:, 0, 0])
ax_single_curve2.plot(sol.t, s[:, 0, 2])
ax_single_curve2.plot(sol.t, i[:, 0, 2])
ax_single_curve2.plot(sol.t, r[:, 0, 2])
ax_single_curve3.plot(sol.t, s[:, 0, 4])
ax_single_curve3.plot(sol.t, i[:, 0, 4])
ax_single_curve3.plot(sol.t, r[:, 0, 4])
plt.show()
