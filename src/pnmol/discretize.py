"""Discretise differential operators on a mesh, assuming an underlying function space."""

from functools import partial

import jax
import jax.numpy as jnp
import tqdm

from pnmol import kernels


def fd_probabilistic(
    diffop,
    mesh_spatial,
    kernel=None,
    stencil_size=3,
    nugget_gram_matrix=0.0,
):
    """
    Discretize a differential operator with probabilistic finite differences.

    Turn something that acts on functions (callables) into something that acts on
    vectors (arrays).

    Parameters
    ---------
    diffop
        Differential operator.
    mesh_spatial
        On which mesh shall the differential operator be discretized.
    kernel
        Representative kernel object for an underlying reproducing kernel Hilbert
        space (RKHS). Recover classical finite differences
    stencil_size
        Number of local neighbours to use for localised discretization.
        Optional. Default is ``None``, which implies that all points are used.
    nugget_gram_matrix
        Damping value to be added to the diagonal of each kernel Gram matrix.

    Returns
    -------
    np.ndarray
        The discretized linear operator as a ``np.ndarray``.
    """

    if kernel is None:
        kernel = kernels.SquareExponential()

    # Fix kernel arguments in FD function
    L_kx = kernels.Lambda(diffop(kernel.pairwise, argnums=0))
    LL_kx = kernels.Lambda(diffop(L_kx.pairwise, argnums=1))
    fd_coeff_fun_partial = partial(
        fd_coefficients,
        k=kernel,
        L_k=L_kx,
        LL_k=LL_kx,
        nugget_gram_matrix=nugget_gram_matrix,
    )
    fd_coeff_fun_batched = jax.jit(jax.vmap(fd_coeff_fun_partial))

    # Read off all neighbors in a single batch (the underlying KDTree is vectorized)
    neighbors_all, neighbor_indices_all = mesh_spatial.neighbours(
        point=mesh_spatial.points, num=stencil_size
    )

    # Compute all FD coefficients in a single batch
    weights_all, uncertainties_all = fd_coeff_fun_batched(
        x=mesh_spatial.points, neighbors=neighbors_all
    )

    # Stack the resulting weights and uncertainties into a matrix
    return _weights_to_matrix(
        weights_all,
        uncertainties_all,
        neighbor_indices_all,
    )


@jax.jit
def _weights_to_matrix(weights_all, uncertainties_all, neighbor_indices_all):
    """Stack the FD weights (and uncertainties) into a differentiation matrix."""
    num_mesh_points = weights_all.shape[0]
    L_empty = jnp.zeros((num_mesh_points, num_mesh_points))
    indices_col = neighbor_indices_all
    indices_row = jnp.full(
        weights_all.shape, fill_value=jnp.arange(num_mesh_points)[:, None]
    )
    L = jax.ops.index_update(x=L_empty, idx=(indices_row, indices_col), y=weights_all)
    E_sqrtm = jnp.diag(uncertainties_all)
    return L, jnp.sqrt(jnp.abs(E_sqrtm))


@partial(jax.jit, static_argnums=(2, 3, 4))
def fd_coefficients(x, neighbors, k, L_k, LL_k, nugget_gram_matrix=0.0):
    """Compute kernel-based finite difference coefficients."""

    X, n = neighbors, neighbors.shape[0]
    gram_matrix = k(X, X.T) + nugget_gram_matrix * jnp.eye(n)
    diffop_at_point = L_k(x[None, :], X.T).reshape((-1,))

    weights = jnp.linalg.solve(gram_matrix, diffop_at_point)
    uncertainty = LL_k(x, x).reshape(()) - weights @ diffop_at_point

    return weights, uncertainty
