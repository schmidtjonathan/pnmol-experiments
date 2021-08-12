import functools
import logging
import os

logging.basicConfig(
    level=logging.INFO,
    format="%(levelname)s - 1-D diffusion notebook - %(message)s",
)

os.environ["TF_CPP_MAX_LOG_LEVEL"] = "0"
os.environ["JAX_PLATFORM_NAME"] = "cpu"

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import numpy as np
import scipy.sparse
import scipy.stats
import stackedssm
import tqdm
from matplotlib import animation
from ppde import differential_operator, discretize, kernel, mesh
from probnum import diffeq, filtsmooth, problems, randprocs, randvars, statespace
from scipy.integrate import solve_ivp

jax.config.update("jax_platform_name", "cpu")

M = 50
W = 1

BBOX = np.array([0, W])

dx = (BBOX[1] - BBOX[0]) / M

grid = mesh.RectangularMesh.from_bounding_boxes_1d(bounding_boxes=BBOX, num=M)

# Diffusion parameter
nu = 0.01

time_domain = (0.0, 20.0)


U0 = scipy.stats.norm(0.3, 0.06).pdf(grid.points.reshape(M))
U0 /= U0.max()

interior, interior_idcs = grid.interior
boundary, boundary_idcs = grid.boundary

L_MOL = np.zeros((len(grid), len(grid)))
for point in tqdm.tqdm(interior):

    neighbors, neighbor_idcs = grid.neighbours(point=point, num=3)
    center_idx = neighbor_idcs[0]

    L_MOL[center_idx, neighbor_idcs] = np.array([-2.0, 1.0, 1.0]) / (dx ** 2)

L_MOL[boundary_idcs, boundary_idcs] = 0.0


drift_MOL = L_MOL * nu


stencil_size = 3
lengthscale = dx * int(stencil_size / 2)

pkern = kernel.GaussianKernel(lengthscale=lengthscale)

laplace = differential_operator.laplace()

L_PNMOL, E = discretize.discretize(laplace, grid, pkern, stencil_size=stencil_size)


drift_PNMOL = L_PNMOL * nu


# ## Solve with `scipy`


eval_dt = 0.1


sol_MOL_scipy = solve_ivp(
    lambda t, x: drift_MOL @ x,
    t_span=time_domain,
    y0=U0.copy(),
    t_eval=np.arange(*time_domain, step=eval_dt),
)


sol_PNMOL_scipy = solve_ivp(
    lambda t, x: drift_PNMOL @ x,
    t_span=time_domain,
    y0=U0.copy(),
    t_eval=np.arange(*time_domain, step=eval_dt),
)


# ## Spatial discretization error

solution_prior = statespace.IBM(
    ordint=2,
    spatialdim=len(grid),
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)

solution_prior.dispmat = solution_prior.dispmat * 0.1


error_prior = statespace.IBM(
    ordint=1,
    spatialdim=len(grid),
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)

error_prior.dispmat = error_prior.dispmat * 0.001


prior = stackedssm.stacked_ssm.StackedSDE(
    processes=(solution_prior, error_prior),
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)


def measure_ode(t, y, vectorfield):
    x = prior.proj2process(0) @ y
    eps = prior.proj2process(1) @ y

    E0_x = prior.proj2coord(proc=0, coord=0)
    E1_x = prior.proj2coord(proc=0, coord=1)
    E0_eps = prior.proj2coord(proc=1, coord=0)

    return (E1_x @ x) - vectorfield(t, E0_x @ x) - (E0_eps @ eps)


def measure_ode_jac(t, y, vectorfieldjac):
    x = prior.proj2process(0) @ y
    eps = prior.proj2process(1) @ y

    E0_x = prior.proj2coord(proc=0, coord=0)
    E1_x = prior.proj2coord(proc=0, coord=1)
    E0_eps = prior.proj2coord(proc=1, coord=0)

    d_dx = E1_x - vectorfieldjac(t, E0_x @ x) @ E0_x
    d_deps = -E0_eps
    return np.concatenate([d_dx, d_deps], -1)


measure_ode_PNMOL = functools.partial(
    measure_ode, vectorfield=(lambda t, x: drift_PNMOL @ x)
)
measure_ode_jac_PNMOL = functools.partial(
    measure_ode_jac, vectorfieldjac=(lambda t, x: drift_PNMOL)
)


proc_idcs = prior.state_idcs

init_mean = np.zeros((prior.dimension,))
init_mean[proc_idcs[0]["state_d0"]] = U0.copy()
init_mean[proc_idcs[0]["state_d1"]] = drift_PNMOL @ U0

init_marginal_vars = 0.0 * np.ones((prior.dimension,))

init_cov = np.diag(init_marginal_vars)

initrv = randvars.Normal(init_mean, init_cov)


measurement_matrix_ode = np.zeros((len(grid), len(grid)))
measurement_matrix_ode_chol = np.zeros_like(measurement_matrix_ode)

measurement_model_ode = statespace.DiscreteGaussian(
    input_dim=initrv.mean.size,
    output_dim=len(grid),
    state_trans_fun=measure_ode_PNMOL,
    proc_noise_cov_mat_fun=lambda t: measurement_matrix_ode,
    proc_noise_cov_cholesky_fun=lambda t: measurement_matrix_ode_chol,
    jacob_state_trans_fun=measure_ode_jac_PNMOL,
)

linearized_measurement_model_ode = filtsmooth.gaussian.approx.DiscreteEKFComponent(
    non_linear_model=measurement_model_ode,
    forward_implementation="sqrt",
    backward_implementation="sqrt",
)


locations = np.arange(*time_domain, 0.05 * eval_dt)
zero_data = np.stack([np.zeros((len(grid),)) for _ in range(locations.size)])

regression_problem = problems.TimeSeriesRegressionProblem(
    observations=zero_data,
    locations=locations,
    measurement_models=[linearized_measurement_model_ode] * locations.size,
)


_point = prior.proj2coord(proc=0, coord=0) @ prior.proj2process(0) @ initrv.mean
_t = 0.1
_m = initrv.mean

meas_fct = lambda eps: measure_ode_PNMOL(_t, _m + eps)
meas_jac = measure_ode_jac_PNMOL(_t, _m)

stackedssm.test_jacobian(dim=_m.size, jacobian=meas_jac, function=meas_fct)


gm_process = randprocs.MarkovProcess(
    initarg=locations[0], initrv=initrv, transition=prior
)


kalman_filter = filtsmooth.gaussian.Kalman(gm_process)


filter_posterior, _ = kalman_filter.filtsmooth(regression_problem)


sol_MOL_probnum_y = filter_posterior(np.arange(*time_domain, step=eval_dt))
sol_PNMOL_probnum_y = filter_posterior(np.arange(*time_domain, step=eval_dt))


plt.rcParams["animation.embed_limit"] = (
    2 * 10 ** 8
)  # Set the animation max size to 200MB

fig = plt.figure(figsize=(20, 8))
ax = fig.add_subplot(1, 2, 1)
ax2 = fig.add_subplot(1, 2, 2)
_im1 = ax.plot(
    grid.points.squeeze(),
    prior.proj2coord(proc=0, coord=0)
    @ prior.proj2process(0)
    @ sol_MOL_probnum_y[0].mean,
)
_im1 = ax.plot(grid.points.squeeze(), U0)
_im2 = ax2.plot(
    grid.points.squeeze(),
    prior.proj2coord(proc=1, coord=0)
    @ prior.proj2process(1)
    @ sol_MOL_probnum_y[0].mean,
)

ax1ylim = ax.get_ylim()
ax2ylim = ax2.get_ylim()
ax.set_ylim(ax1ylim)


def animate(i):

    mean = sol_MOL_probnum_y[i].mean
    std = np.sqrt(np.diag(sol_MOL_probnum_y[i].cov))

    sol_mean = prior.proj2coord(proc=0, coord=0) @ prior.proj2process(0) @ mean
    sol_std = prior.proj2coord(proc=0, coord=0) @ prior.proj2process(0) @ std

    eps_mean = prior.proj2coord(proc=1, coord=0) @ prior.proj2process(1) @ mean
    eps_std = prior.proj2coord(proc=1, coord=0) @ prior.proj2process(1) @ std

    ax.cla()
    ax.set_title(f"t={np.round(i*eval_dt, 2)}")
    ax.plot(grid.points.squeeze(), sol_mean, color="C0", label="PN solution")
    ax.plot(
        grid.points.squeeze(),
        sol_MOL_scipy.y[..., i],
        color="C1",
        label="SciPy solution",
    )
    ax.plot(
        grid.points.squeeze()[:M],
        (sol_mean + eps_mean),
        color="C2",
        label=r"PN + $\epsilon$",
    )
    ax.fill_between(
        grid.points.squeeze(),
        sol_mean - 2 * sol_std,
        sol_mean + 2 * sol_std,
        color="C0",
        alpha=0.2,
    )

    ax2.cla()
    ax2.set_title(f"t={np.round(i*eval_dt, 2)}")
    ax2.plot(grid.points.squeeze(), eps_mean, color="C3", label=r"$\epsilon$")
    ax2.fill_between(
        grid.points.squeeze(),
        eps_mean - 2 * eps_std,
        eps_mean + 2 * eps_std,
        color="C1",
        alpha=0.2,
    )

    ax.set_ylim(ax1ylim)
    ax2.set_ylim([-0.003, 0.003])
    ax.legend()
    ax2.legend()


# Animation setup
anim = animation.FuncAnimation(
    fig,
    func=animate,
    frames=sol_PNMOL_probnum_y.shape[0],
    interval=100,
    repeat_delay=4000,
    blit=False,
)
anim.save("heat_1d.gif")
