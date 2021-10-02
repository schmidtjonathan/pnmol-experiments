"""Figure 2."""


from functools import partial

import jax
import jax.numpy as jnp
import matplotlib.pyplot as plt
import plotting
from tqdm import tqdm

import pnmol

# Part 1: Maximum likelihood estimation of the input scale.


def log_likelihood(gram_matrix, y, n):
    a = y @ jnp.linalg.solve(gram_matrix, y)
    b = jnp.log(jnp.linalg.det(gram_matrix))
    c = n * jnp.log(2 * jnp.pi)
    return -0.5 * (a + b + c)


def input_scale_to_log_likelihood_general(l, y, n, mesh_points, log_lkld):
    prior = pnmol.kernels.SquareExponential(input_scale=l)
    K = prior(mesh_points[:, None], mesh_points[None, :])
    return log_lkld(K, y, n)


def input_scale_mle(*, mesh_points, obj_fun, num_trial_points, scale_to_like):
    y = obj_fun(mesh_points[:, None]).squeeze()
    input_scale_to_log_likelihood = jax.vmap(
        partial(
            scale_to_like,
            mesh_points=mesh_points,
            log_lkld=log_likelihood,
            y=y,
            n=num_mesh_points,
        )
    )
    input_scale_trials = jnp.logspace(-3, 3, num_trial_points)
    log_likelihood_values = input_scale_to_log_likelihood(input_scale_trials)
    index_max = jnp.argmax(log_likelihood_values)
    return input_scale_trials[index_max]


def input_scale_to_rmse(scale, stencil_size, diffop, mesh, obj_fun, truth_fun):
    kernel = pnmol.kernels.SquareExponential(input_scale=scale)
    l, e = pnmol.discretize.fd_probabilistic(
        diffop=diffop,
        mesh_spatial=mesh,
        kernel=kernel,
        stencil_size=stencil_size,
    )
    x = mesh.points[1:-1]
    fx = obj_fun(x)
    dfx = truth_fun(x)
    error_abs = jnp.abs(l[1:-1, 1:-1] @ fx - dfx)
    rmse = jnp.linalg.norm(error_abs) / jnp.sqrt(error_abs.size)
    return rmse, (l, e)


def save_array(arr, /, *, suffix, path="experiments/results/figure2/"):
    path_with_suffix = path + suffix
    jnp.save(path_with_suffix, arr)


# Define the basic setup: target function, etc.
obj_fun = jax.vmap(lambda x: jnp.sin(jnp.linalg.norm(x) ** 2))
diffop = pnmol.diffops.laplace()
truth_fun = jax.vmap(diffop(obj_fun))

# Choose a mesh
num_mesh_points = 25
mesh = pnmol.mesh.RectangularMesh(
    jnp.linspace(0, 1, num_mesh_points, endpoint=True)[:, None], bbox=[0.0, 1.0]
)

# Compute the MLE estimate (for comparison)
scale_mle = input_scale_mle(
    mesh_points=mesh.points.squeeze(),
    obj_fun=obj_fun,
    num_trial_points=200,
    scale_to_like=input_scale_to_log_likelihood_general,
)

# Compute all RMSEs
input_scales = jnp.array([0.1, 1.0, 10.0])
stencil_sizes = jnp.array([3, 5, 8, 13, 21])

scale_to_rmse = partial(
    input_scale_to_rmse, diffop=diffop, mesh=mesh, obj_fun=obj_fun, truth_fun=truth_fun
)
scale_to_rmse_vmapped = jax.vmap(scale_to_rmse, in_axes=(0, None))
rmse_all = jnp.stack(
    [scale_to_rmse_vmapped(input_scales, st)[0] for st in stencil_sizes]
)


# Compute L and E for a number of stencil sizes
L_sparse, E_sparse = scale_to_rmse(scale=scale_mle, stencil_size=3)[1]
L_dense, E_dense = scale_to_rmse(scale=scale_mle, stencil_size=300)[1]

# Plotting purposes...
xgrid = jnp.linspace(0, 1, 150)
fx = obj_fun(xgrid[:, None]).squeeze()
dfx = truth_fun(xgrid[:, None]).squeeze()


save_array(rmse_all, suffix="rmse_all")
save_array(input_scales, suffix="input_scales")
save_array(stencil_sizes, suffix="stencil_sizes")
save_array(L_sparse, suffix="L_sparse")
save_array(L_dense, suffix="L_dense")
save_array(E_sparse, suffix="E_sparse")
save_array(xgrid, suffix="xgrid")
save_array(fx, suffix="fx")
save_array(dfx, suffix="dfx")


plotting.figure_2()
