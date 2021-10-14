"""Work-precision diagrams and so on."""


import time

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import tornadox
from scipy.integrate import solve_ivp

import pnmol

pde_kwargs = {"t0": 0.0, "tmax": 6.0}

dts = 2.0 ** jnp.arange(1, -12, step=-1)

num_derivatives = 1

for dx in sorted([0.02, 0.1, 0.2]):
    pde = pnmol.pde.examples.lotka_volterra_1d_discretized(
        **pde_kwargs,
        dx=dx,
        stencil_size_interior=3,
        stencil_size_boundary=5,
    )
    ivp = pde.to_tornadox_ivp()

    ref_scale = 19
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

    pnmol_white_rmse = []
    pnmol_white_nsteps = []
    pnmol_white_chi2 = []
    pnmol_white_time = []

    pnmol_latent_rmse = []
    pnmol_latent_nsteps = []
    pnmol_latent_chi2 = []
    pnmol_latent_time = []

    mol_rmse = []
    mol_nsteps = []
    mol_chi2 = []
    mol_time = []

    # dts = jnp.logspace(0.0, -3.5, -0.5, endpoint=True)
    for dt in sorted(dts):

        if dx > 0.01:
            # [PNMOL-LATENT] Solve
            steps = pnmol.odetools.step.Constant(dt)
            solver = pnmol.latent.SemiLinearLatentForceEK1(
                num_derivatives=num_derivatives, steprule=steps, spatial_kernel=k
            )
            time_start = time.time()
            with jax.disable_jit():
                sol_pnmol_latent, sol_pnmol_latent_info = solver.simulate_final_state(
                    pde, progressbar=True
                )
            time_pnmol_latent = time.time() - time_start

            # [PNMOL-LATENT] Extract mean
            mean_state, _ = jnp.split(
                sol_pnmol_latent.y.mean[0], 2
            )  # ignore mean_latent
            u_pnmol_latent_full, v_pnmol_latent_full = jnp.split(mean_state, 2)
            u_pnmol_latent, v_pnmol_latent = (
                u_pnmol_latent_full[1:-1],
                v_pnmol_latent_full[1:-1],
            )

            # [PNMOL-LATENT] Extract covariance: first remove xi, then remove "v"
            cov_final_latent = (
                sol_pnmol_latent.y.cov_sqrtm @ sol_pnmol_latent.y.cov_sqrtm.T
            )
            cov_final_no_xi = jnp.split(
                jnp.split(cov_final_latent, 2, axis=-1)[0], 2, axis=0
            )[0]
            cov_final_latent_interesting = solver.E0 @ cov_final_no_xi @ solver.E0.T
            cov_final_latent_u_split_horizontally, _ = jnp.split(
                cov_final_latent_interesting, 2, axis=-1
            )
            cov_final_latent_u_split, _ = jnp.split(
                cov_final_latent_u_split_horizontally, 2, axis=0
            )
            cov_final_latent_u = cov_final_latent_u_split[1:-1, 1:-1]

            # [PNMOL-LATENT] Compute error and calibration
            error_pnmol_latent_abs = jnp.abs(u_pnmol_latent - u_reference)
            error_pnmol_latent_rel = error_pnmol_latent_abs / jnp.abs(u_reference)
            rmse_pnmol_latent = jnp.linalg.norm(error_pnmol_latent_rel) / jnp.sqrt(
                u_pnmol_latent.size
            )
            chi2_pnmol_latent = (
                error_pnmol_latent_abs
                @ jnp.linalg.solve(cov_final_latent_u, error_pnmol_latent_abs)
                / error_pnmol_latent_abs.shape[0]
            )
        else:
            print("Skipping Latent...")

        ################################################################
        ################################################################

        # [PNMOL-WHITE] Solve
        steps = pnmol.odetools.step.Constant(dt)
        solver = pnmol.white.SemiLinearWhiteNoiseEK1(
            num_derivatives=num_derivatives, steprule=steps, spatial_kernel=k
        )
        time_start = time.time()
        with jax.disable_jit():
            sol_pnmol_white, sol_pnmol_white_info = solver.simulate_final_state(
                pde, progressbar=True
            )
        time_pnmol_white = time.time() - time_start

        # [PNMOL-WHITE] Extract mean
        u_pnmol_white_full, v_pnmol_white_full = jnp.split(sol_pnmol_white.y.mean[0], 2)
        u_pnmol_white, v_pnmol_white = (
            u_pnmol_white_full[1:-1],
            v_pnmol_white_full[1:-1],
        )

        # [PNMOL-WHITE] Extract covariance
        cov_final_white = sol_pnmol_white.y.cov_sqrtm @ sol_pnmol_white.y.cov_sqrtm.T
        cov_final_white_interesting = solver.E0 @ cov_final_white @ solver.E0.T
        cov_final_white_u_split_horizontally, _ = jnp.split(
            cov_final_white_interesting, 2, axis=-1
        )
        cov_final_white_u_split, _ = jnp.split(
            cov_final_white_u_split_horizontally, 2, axis=0
        )
        cov_final_white_u = cov_final_white_u_split[1:-1, 1:-1]

        # [PNMOL-WHITE] Compute error and calibration
        error_pnmol_white_abs = jnp.abs(u_pnmol_white - u_reference)
        error_pnmol_white_rel = error_pnmol_white_abs / jnp.abs(u_reference)
        rmse_pnmol_white = jnp.linalg.norm(error_pnmol_white_rel) / jnp.sqrt(
            u_pnmol_white.size
        )
        chi2_pnmol_white = (
            error_pnmol_white_abs
            @ jnp.linalg.solve(cov_final_white_u, error_pnmol_white_abs)
            / error_pnmol_white_abs.shape[0]
        )

        ################################################################
        ################################################################

        # [MOL] Solve
        steps = tornadox.step.ConstantSteps(dt)
        ek1 = tornadox.ek1.ReferenceEK1ConstantDiffusion(
            num_derivatives=num_derivatives,
            steprule=steps,
            initialization=tornadox.init.Stack(use_df=False),
        )
        time_start = time.time()
        with jax.disable_jit():
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

        ################################################################
        ################################################################

        # Print results
        print(
            f"MOL:\n\tRMSE={rmse_mol}, chi2={chi2_mol}, nsteps={sol_mol_info['num_steps']}, time={time_mol}"
        )
        print(
            f"PNMOL(white):\n\tRMSE={rmse_pnmol_white}, chi2={chi2_pnmol_white}, nsteps={sol_pnmol_white_info['num_steps']}, time={time_pnmol_white}"
        )
        if dx > 0.01:
            print(
                f"PNMOL(latent):\n\tRMSE={rmse_pnmol_latent}, chi2={chi2_pnmol_latent}, nsteps={sol_pnmol_latent_info['num_steps']}, time={time_pnmol_latent}"
            )

            pnmol_latent_rmse.append(rmse_pnmol_latent)
            pnmol_latent_chi2.append(chi2_pnmol_latent)
            pnmol_latent_nsteps.append(sol_pnmol_latent_info["num_steps"])
            pnmol_latent_time.append(time_pnmol_latent)

        pnmol_white_rmse.append(rmse_pnmol_white)
        pnmol_white_chi2.append(chi2_pnmol_white)
        pnmol_white_nsteps.append(sol_pnmol_white_info["num_steps"])
        pnmol_white_time.append(time_pnmol_white)

        mol_rmse.append(rmse_mol)
        mol_chi2.append(chi2_mol)
        mol_nsteps.append(sol_mol_info["num_steps"])
        mol_time.append(time_mol)

        print()

    path = "experiments/results/figure4/" + f"dx_{dx}_"

    jnp.save(path + "pnmol_white_rmse.npy", jnp.asarray(pnmol_white_rmse))
    jnp.save(path + "pnmol_white_chi2.npy", jnp.asarray(pnmol_white_chi2))
    jnp.save(path + "pnmol_white_nsteps.npy", jnp.asarray(pnmol_white_nsteps))
    jnp.save(path + "pnmol_white_time.npy", jnp.asarray(pnmol_white_time))

    jnp.save(path + "pnmol_latent_rmse.npy", jnp.asarray(pnmol_latent_rmse))
    jnp.save(path + "pnmol_latent_chi2.npy", jnp.asarray(pnmol_latent_chi2))
    jnp.save(path + "pnmol_latent_nsteps.npy", jnp.asarray(pnmol_latent_nsteps))
    jnp.save(path + "pnmol_latent_time.npy", jnp.asarray(pnmol_latent_time))

    jnp.save(path + "mol_rmse.npy", jnp.asarray(mol_rmse))
    jnp.save(path + "mol_chi2.npy", jnp.asarray(mol_chi2))
    jnp.save(path + "mol_nsteps.npy", jnp.asarray(mol_nsteps))
    jnp.save(path + "mol_time.npy", jnp.asarray(mol_time))

    jnp.save(path + "dts.npy", jnp.asarray(dts))
