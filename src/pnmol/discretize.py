"""Discretise differential operators on a mesh, assuming an underlying function space."""

from functools import partial

import jax
import jax.numpy as jnp
import tqdm

from pnmol import kernels


def discretize(
    diffop,
    mesh_spatial,
    kernel,
    stencil_size,
    nugget_gram_matrix=0.0,
    progressbar=False,
):
    """
    Discretize a differential operator.

    Turn something that acts on functions (callables) into something that acts on
    vectors (arrays).

    Parameters
    ---------
    diffop
        Differential operator.
    mesh
        On which mesh shall the differential operator be discretized.
    kernel
        Representative kernel object for an underlying reproducing kernel Hilbert
        space (RKHS). Recover classical finite differences
    stencil_size
        Number of local neighbours to use for localised discretization.
        Optional. Default is ``None``, which implies that all points are used.
    cov_damping
        Damping value to be added to the diagonal of each kernel Gram matrix.

    Returns
    -------
    np.ndarray
        The discretized linear operator as a ``np.ndarray``.
    """

    M = len(mesh_spatial)

    L_kx = kernels.Lambda(diffop(kernel.pairwise, argnums=0))
    LL_kx = kernels.Lambda(diffop(L_kx.pairwise, argnums=1))

    fd_coeff_fun = partial(
        fd_coeff,
        mesh_spatial=mesh_spatial,
        stencil_size=stencil_size,
        k=kernel,
        L_k=L_kx,
        LL_k=LL_kx,
        nugget_gram_matrix=nugget_gram_matrix,
    )

    L_data, L_row, L_col, E_data = [], [], [], []

    range_loop = enumerate(tqdm.tqdm(mesh_spatial.points, disable=not progressbar))
    for i, point in range_loop:

        weights, uncertainty, neighbor_idcs = fd_coeff_fun(x=point)

        L_data.append(weights)
        L_row.append(jnp.full(shape=stencil_size, fill_value=i, dtype=int))
        L_col.append(neighbor_idcs)
        E_data.append(uncertainty)

    L_data = jnp.concatenate(L_data)
    L_row = jnp.concatenate(L_row)
    L_col = jnp.concatenate(L_col)
    E_data = jnp.stack(E_data)

    L = jax.ops.index_update(jnp.zeros((M, M)), (L_row, L_col), L_data)
    E = jnp.diag(E_data)
    return L, jnp.sqrt(jnp.abs(E))


def fd_coeff(x, mesh_spatial, stencil_size, k, L_k, LL_k, nugget_gram_matrix):
    """Compute kernel-based finite difference coefficients."""

    neighbors, neighbor_indices = mesh_spatial.neighbours(point=x, num=stencil_size)

    X = neighbors.points
    gram_matrix = k(X, X.T) + nugget_gram_matrix * jnp.eye(X.shape[0])
    diffop_at_point = L_k(x[None, :], X.T).reshape((-1,))
    weights = jnp.linalg.solve(gram_matrix, diffop_at_point)
    uncertainty = LL_k(x, x).reshape(()) - weights @ diffop_at_point

    assert not jnp.any(jnp.isnan(weights)), (
        weights,
        diffop_at_point,
        gram_matrix,
        jnp.linalg.eigvals(gram_matrix),
        x,
        X,
    )
    assert not jnp.any(jnp.isnan(uncertainty))

    return weights, uncertainty, neighbor_indices
