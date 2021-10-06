"""Figure 2."""

from functools import partial

import jax
import jax.numpy as jnp
import jax.random
import matplotlib.pyplot as plt
import plotting
from tqdm import tqdm

import pnmol

# Part 1: Maximum likelihood estimation of the input scale.


def input_scale_mle(*, mesh_points, obj, num_trial_points):
    """Compute the MLE over the input_scale parameter in a square exponential kernel."""
    y = obj(mesh_points[:, None]).squeeze()
    input_scale_trials = jnp.logspace(-3, 3, num_trial_points)

    log_likelihood_values = jnp.stack(
        [
            input_scale_to_log_likelihood(
                l=l,
                y=y,
                n=num_mesh_points,
                mesh_points=mesh_points,
            )
            for l in input_scale_trials
        ]
    )

    index_max = jnp.argmax(log_likelihood_values)
    return input_scale_trials[index_max]


def input_scale_to_log_likelihood(*, l, y, n, mesh_points):
    kernel = pnmol.kernels.SquareExponential(input_scale=l)
    K = kernel(mesh_points[:, None], mesh_points[None, :])
    return log_likelihood(gram_matrix=K, y=y, n=n)


def log_likelihood(*, gram_matrix, y, n):
    a = y @ jnp.linalg.solve(gram_matrix, y)
    b = jnp.log(jnp.linalg.det(gram_matrix))
    c = n * jnp.log(2 * jnp.pi)
    return -0.5 * (a + b + c)


def input_scale_to_rmse(scale, stencil_size, *, diffop, mesh, obj_fun, truth_fun):
    kernel = pnmol.kernels.SquareExponential(input_scale=scale)
    l, e = pnmol.discretize.fd_probabilistic(
        diffop=diffop,
        mesh_spatial=mesh,
        kernel=kernel,
        stencil_size_interior=stencil_size,
        stencil_size_boundary=stencil_size,
    )
    x = mesh.points
    fx = obj_fun(x).squeeze()
    dfx = truth_fun(x).squeeze()
    error_abs = jnp.abs(l @ fx - dfx) / jnp.abs(dfx)
    rmse = jnp.linalg.norm(error_abs) / jnp.sqrt(error_abs.size)
    return rmse, (l, e)


def sample(key, kernel, mesh_points, nugget_gram_matrix=1e-12):
    N = mesh_points.shape[0]
    gram_matrix = kernel(mesh_points, mesh_points.T)
    gram_matrix += nugget_gram_matrix * jnp.eye(N)

    sample = jax.random.normal(key, shape=(N, 2))
    return jnp.linalg.cholesky(gram_matrix) @ sample


def save_array(arr, /, *, suffix, path="experiments/results/figure2/"):
    _assert_not_nan(arr)
    path_with_suffix = path + suffix
    jnp.save(path_with_suffix, arr)


def _assert_not_nan(arr):
    assert not jnp.any(jnp.isnan(arr))


# Define the basic setup: target function, etc.
obj_fun = jax.vmap(lambda x: jnp.sin(x.dot(x)))
diffop = pnmol.diffops.laplace()
truth_fun = jax.vmap(diffop(obj_fun))


# Choose a mesh
num_mesh_points = 25
mesh = pnmol.mesh.RectangularMesh(
    jnp.linspace(0, 1, num_mesh_points, endpoint=True)[:, None],
    bbox=jnp.asarray([0.0, 1.0])[:, None],
)

# Compute the MLE estimate (for comparison)
scale_mle = input_scale_mle(
    mesh_points=mesh.points.squeeze(),
    obj=obj_fun,
    num_trial_points=3,  # 20 was good
)


# Compute all RMSEs
input_scales = jnp.array([0.2, 0.8, 3.2])
stencil_sizes = jnp.arange(3, len(mesh[:5]), step=2)
e = partial(
    input_scale_to_rmse, diffop=diffop, mesh=mesh, obj_fun=obj_fun, truth_fun=truth_fun
)

# The below can be vmapped (with a bit of tinkering) which makes it _much_ faster,
# but also kind of unreadable (and in the experiment code, I prefer readability over speed)
rmse_all = jnp.asarray(
    [[e(l, s)[0] for l in input_scales] for s in tqdm(stencil_sizes)]
)
rmse_all = jnp.nan_to_num(rmse_all, nan=100.0)

# Compute L and E for a number of stencil sizes
L_sparse, E_sparse = e(scale=scale_mle, stencil_size=3)[1]
L_dense, E_dense = pnmol.discretize.collocation_global(
    diffop=diffop,
    mesh_spatial=mesh,
    kernel=pnmol.kernels.SquareExponential(input_scale=scale_mle),
    nugget_cholesky_E=1e-10,
    nugget_gram_matrix=1e-12,
    symmetrize_cholesky_E=True,
)

# Plotting purposes...
xgrid = jnp.linspace(0, 1, 150)[:, None]
fx = obj_fun(xgrid).squeeze()
dfx = truth_fun(xgrid).squeeze()


# Sample from the different priors
key = jax.random.PRNGKey(seed=123)
k1 = pnmol.kernels.SquareExponential(input_scale=input_scales[0])
k2 = pnmol.kernels.SquareExponential(input_scale=input_scales[1])
k3 = pnmol.kernels.SquareExponential(input_scale=input_scales[2])
s1 = sample(key=key, kernel=k1, mesh_points=xgrid)
_, key = jax.random.split(key)
s2 = sample(key=key, kernel=k2, mesh_points=xgrid)
_, key = jax.random.split(key)
s3 = sample(key=key, kernel=k3, mesh_points=xgrid)


save_array(rmse_all, suffix="rmse_all")
save_array(input_scales, suffix="input_scales")
save_array(stencil_sizes, suffix="stencil_sizes")
save_array(L_sparse, suffix="L_sparse")
save_array(L_dense, suffix="L_dense")
save_array(E_sparse, suffix="E_sparse")
save_array(E_dense, suffix="E_dense")
save_array(xgrid, suffix="xgrid")
save_array(fx, suffix="fx")
save_array(dfx, suffix="dfx")
save_array(s1, suffix="s1")
save_array(s2, suffix="s2")
save_array(s3, suffix="s3")


plotting.figure_2()
