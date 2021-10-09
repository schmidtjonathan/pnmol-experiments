"""Work-precision diagrams and so on."""


import time
import warnings

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox
from scipy.integrate import solve_ivp

import pnmol

warnings.filterwarnings("ignore")

pde_kwargs = {"t0": 0.0, "tmax": 6.0}

for dx in [0.01, 0.05]:
    pde = pnmol.pde.examples.lotka_volterra_1d_discretized(
        **pde_kwargs,
        dx=dx,
        stencil_size_interior=3,
        stencil_size_boundary=4,
    )
    ivp = pde.to_tornadox_ivp()

    ref_scale = 6
    pde_ref = pnmol.pde.examples.lotka_volterra_1d_discretized(
        **pde_kwargs,
        dx=dx / ref_scale,
    )
    ivp_ref = pde_ref.to_tornadox_ivp()
    print("assembled")

    print(pde.mesh_spatial.shape)
    print(pde_ref.mesh_spatial.shape)
    ref_sol = solve_ivp(
        jax.jit(ivp_ref.f),
        ivp_ref.t_span,
        y0=ivp_ref.y0,
        method="LSODA",
        atol=1e-10,
        rtol=1e-10,
        t_eval=(ivp_ref.t0, ivp_ref.tmax),
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

    pnmol_rmse = []
    pnmol_nsteps = []
    pnmol_chi2 = []
    pnmol_time = []

    mol_rmse = []
    mol_nsteps = []
    mol_chi2 = []
    mol_time = []

    scipy_rmse = []
    scipy_time = []
    scipy_nsteps = []

    # dts = jnp.logspace(0., -3, 1, endpoint=True)
    dts = jnp.logspace(0.0, -2.5, 12, endpoint=True)
    for dt in dts:

        # [PNMOL] Solve
        steps = pnmol.odetools.step.Constant(dt)
        solver = pnmol.white.SemiLinearWhiteNoiseEK1(
            num_derivatives=2, steprule=steps, spatial_kernel=k
        )
        time_start = time.time()
        sol_pnmol, sol_pnmol_info = solver.simulate_final_state(pde, progressbar=True)
        time_pnmol = time.time() - time_start

        # [PNMOL] Extract mean
        u_pnmol_full, v_pnmol_full = jnp.split(sol_pnmol.y.mean[0], 2)
        u_pnmol, v_pnmol = u_pnmol_full[1:-1], v_pnmol_full[1:-1]

        # [PNMOL] Extract covariance
        cov_final = sol_pnmol.y.cov_sqrtm @ sol_pnmol.y.cov_sqrtm.T
        cov_final_interesting = solver.E0 @ cov_final @ solver.E0.T
        cov_final_u_split_horizontally, _ = jnp.split(cov_final_interesting, 2, axis=-1)
        cov_final_u_split, _ = jnp.split(cov_final_u_split_horizontally, 2, axis=0)
        cov_final_u = cov_final_u_split[1:-1, 1:-1]

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
        steps = tornadox.step.ConstantSteps(dt)
        ek1 = tornadox.ek1.ReferenceEK1(
            num_derivatives=2,
            steprule=steps,
            initialization=tornadox.init.Stack(use_df=False),
        )
        time_start = time.time()
        sol_mol, sol_mol_info = ek1.simulate_final_state(ivp, progressbar=True)
        time_mol = time.time() - time_start

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

        # [SciPy] Solve
        time_start = time.time()
        sol = solve_ivp(
            pde.f,
            pde.t_span,
            y0=pde.y0,
            atol=(0.01 * dt) ** 2,
            rtol=(0.01 * dt) ** 2,
            method="RK45",
            t_eval=(pde.t0, pde.tmax),
        )
        time_scipy = time.time() - time_start
        sol_scipy = sol.y[:, -1]
        u_scipy_full, _ = jnp.split(sol_scipy, 2)
        u_scipy = u_scipy_full[1:-1]

        # [SciPy] Compute error
        error_abs_scipy = jnp.abs(u_scipy - u_reference)
        error_rel_scipy = error_abs_scipy / jnp.abs(u_reference)
        rmse_scipy = jnp.linalg.norm(error_rel_scipy) / jnp.sqrt(
            error_rel_scipy.shape[0]
        )

        # Print results
        print(
            f"MOL:\n\tRMSE={rmse_mol}, chi2={chi2_mol}, nsteps={sol_mol_info['num_steps']}, time={time_mol}"
        )
        print(
            f"PNMOL:\n\tRMSE={rmse_pnmol}, chi2={chi2_pnmol}, nsteps={sol_pnmol_info['num_steps']}, time={time_pnmol}"
        )
        print(f"Scipy:\n\tRMSE={rmse_scipy}, nsteps={sol.nfev}, time={time_scipy}")

        pnmol_rmse.append(rmse_pnmol)
        pnmol_chi2.append(chi2_pnmol)
        pnmol_nsteps.append(sol_pnmol_info["num_steps"])
        pnmol_time.append(time_pnmol)

        mol_rmse.append(rmse_mol)
        mol_chi2.append(chi2_mol)
        mol_nsteps.append(sol_mol_info["num_steps"])
        mol_time.append(time_mol)

        scipy_rmse.append(rmse_scipy)
        scipy_nsteps.append(sol.nfev)
        scipy_time.append(time_scipy)

        print()

    path = "experiments/results/figure4/" + f"dx_{dx}_"

    jnp.save(path + "pnmol_rmse.npy", jnp.asarray(pnmol_rmse))
    jnp.save(path + "pnmol_chi2.npy", jnp.asarray(pnmol_chi2))
    jnp.save(path + "pnmol_nsteps.npy", jnp.asarray(pnmol_nsteps))
    jnp.save(path + "pnmol_time.npy", jnp.asarray(pnmol_time))

    jnp.save(path + "mol_rmse.npy", jnp.asarray(mol_rmse))
    jnp.save(path + "mol_chi2.npy", jnp.asarray(mol_chi2))
    jnp.save(path + "mol_nsteps.npy", jnp.asarray(mol_nsteps))
    jnp.save(path + "mol_time.npy", jnp.asarray(mol_time))

    jnp.save(path + "scipy_rmse.npy", jnp.asarray(scipy_rmse))
    jnp.save(path + "scipy_nsteps.npy", jnp.asarray(scipy_nsteps))
    jnp.save(path + "scipy_time.npy", jnp.asarray(scipy_time))

    jnp.save(path + "dts.npy", jnp.asarray(dts))
