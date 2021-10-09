"""Work-precision diagrams and so on."""


import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox
from scipy.integrate import solve_ivp

import pnmol

pde_kwargs = {"t0": 0.0, "tmax": 15.0}
dx = 0.0125
pde = pnmol.pde.examples.lotka_volterra_1d_discretized(
    **pde_kwargs,
    dx=dx,
    stencil_size_interior=3,
    stencil_size_boundary=4,
)
ivp = pde.to_tornadox_ivp()


ref_scale = 1
pde_ref = pnmol.pde.examples.lotka_volterra_1d_discretized(
    **pde_kwargs,
    dx=dx / ref_scale,
)
ivp_ref = pde_ref.to_tornadox_ivp()
print("assembled")

print(pde.mesh_spatial.shape)
print(pde_ref.mesh_spatial.shape)
ref_sol = solve_ivp(
    ivp_ref.f, ivp_ref.t_span, y0=ivp_ref.y0, method="RK45", atol=1e-10, rtol=1e-10
)


print("solved")
u_reference_full, v_reference_full = jnp.split(ref_sol.y[:, -1], 2)
u_reference, v_reference = (
    u_reference_full[(ref_scale - 1) :: ref_scale],
    v_reference_full[(ref_scale - 1) :: ref_scale],
)

k = pnmol.kernels.duplicate(
    pnmol.kernels.Matern52() + pnmol.kernels.WhiteNoise(), num=2
)


for tol in jnp.logspace(-1, -6, 4, endpoint=True):

    # [PNMOL] Solve
    steps = pnmol.odetools.step.Adaptive(abstol=0.01 * tol, reltol=tol)
    solver = pnmol.white.SemiLinearWhiteNoiseEK1(
        num_derivatives=2, steprule=steps, spatial_kernel=k
    )
    sol_pnmol, sol_pnmol_info = solver.simulate_final_state(pde, progressbar=True)

    # [PNMOL] Extract mean
    u_pnmol_full, v_pnmol_full = jnp.split(sol_pnmol.y.mean[0], 2)
    u_pnmol, v_pnmol = u_pnmol_full[1:-1], v_pnmol_full[1:-1]

    # [PNMOL] Extract covariance
    cov_final = sol_pnmol.ycov_sqrtm @ sol_pnmol.y.cov_sqrtm.T
    cov_final_interesting = solver.E0 @ cov_final @ solver.E0.T
    cov_final_u = jnp.split(jnp.split(cov_final_interesting, 2, axis=-1)[0], 2, axis=0)[
        0
    ][1:-1, 1:-1]

    # [PNMOL] Compute error and calibration
    error_pnmol_abs = jnp.abs(u_pnmol - u_reference)
    error_pnmol_rel = error_pnmol_abs / jnp.abs(u_reference)
    rmse_pnmol = jnp.linalg.norm(error_pnmol_rel) / jnp.sqrt(u_pnmol.size)
    chi2_pnmol = (
        error_pnmol_abs
        @ jnp.linalg.solve(cov_final_u, error_pnmol_abs)
        / error_pnmol_abs.shape[0]
    )

    # [MOL] Solve
    steps = tornadox.step.AdaptiveSteps(abstol=0.01 * tol, reltol=tol)
    ek1 = tornadox.ek1.ReferenceEK1(
        num_derivatives=2,
        steprule=steps,
        initialization=tornadox.init.Stack(use_df=False),
    )
    sol_mol, sol_mol_info = ek1.simulate_final_state(ivp)

    # [MOL] Extract mean
    u_mol, v_mol = jnp.split(sol_mol.y.mean[0], 2)

    # [MOL] Extract covariance
    cov_final_mol = sol_mol.y.cov_sqrtm @ sol_mol.y.cov_sqrtm.T
    cov_final_interesting_mol = ek1.P0 @ cov_final_mol @ ek1.P0.T
    cov_final_u_mol = jnp.split(
        jnp.split(cov_final_interesting_mol, 2, axis=-1)[0], 2, axis=0
    )[0]

    # [MOL] Compute error and calibration
    error_mol_abs = jnp.abs(u_mol - u_reference)
    error_mol_rel = error_mol_abs / jnp.abs(u_reference)
    rmse_mol = jnp.linalg.norm(error_mol_rel) / jnp.sqrt(u_mol.size)
    chi2_mol = (
        error_mol_abs
        @ jnp.linalg.solve(cov_final_u_mol, error_mol_abs)
        / error_mol_abs.shape[0]
    )

    # Print results
    print(
        "The error is extremely bounded by the accuracy of dx and ref_scale... if dx_scale=1, then MOL does really well, and PNMOL does not care. if dx_scale=10, both are bad. what are the stdevs??"
    )
    print("MOL", rmse_mol, chi2_mol, sol_mol_info["num_steps"])
    print("PNMOL", rmse_pnmol, chi2_pnmol, sol_pnmol_info["num_steps"])

    print()

u, v = jnp.split(sol_pnmol.mean, 2, axis=-1)

fig, axes = plt.subplots(ncols=6)
ax_dt, ax_sol1, ax_sol2 = axes[:3]
ax_single_curve1, ax_single_curve2, ax_single_curve3 = axes[3:]


ax_dt.semilogy(jnp.diff(sol_pnmol.t))
ax_dt.set_title("Step-size")

ax_sol1.imshow(u[:, 0, :].T, aspect="auto", vmin=0.0, cmap="Blues")
ax_sol2.imshow(v[:, 0, :].T, aspect="auto", vmin=0.0, cmap="Oranges")

ax_sol1.set_title("Predators")
ax_sol2.set_title("Prey")


ax_single_curve1.plot(sol_pnmol.t, u[:, 0, 0])
ax_single_curve1.plot(sol_pnmol.t, v[:, 0, 0])
ax_single_curve2.plot(sol_pnmol.t, u[:, 0, 10])
ax_single_curve2.plot(sol_pnmol.t, v[:, 0, 10])
ax_single_curve3.plot(sol_pnmol.t, u[:, 0, -1])
ax_single_curve3.plot(sol_pnmol.t, v[:, 0, -1])

for ax in [ax_single_curve1, ax_single_curve2, ax_single_curve3]:
    ax.set_ylim((0.0, 25.0))
plt.show()
