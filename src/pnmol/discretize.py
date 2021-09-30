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
):
    """
    Discretize a differential operator.

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

    M = len(mesh_spatial)

    L_kx = kernels.Lambda(diffop(kernel.pairwise, argnums=0))
    LL_kx = kernels.Lambda(diffop(L_kx.pairwise, argnums=1))

    fd_coeff_fun = jax.jit(
        jax.vmap(
            partial(
                fd_coeff,
                k=kernel,
                L_k=L_kx,
                LL_k=LL_kx,
                nugget_gram_matrix=nugget_gram_matrix,
            )
        )
    )

    neighbors_all, neighbor_indices_all = mesh_spatial.neighbours(
        point=mesh_spatial.points, num=stencil_size
    )
    weights_all, uncertainties_all = fd_coeff_fun(
        x=mesh_spatial.points, neighbors=neighbors_all
    )

    indices_row = jnp.full(weights_all.shape, fill_value=jnp.arange(M)[:, None])
    L = jax.ops.index_update(
        jnp.zeros((M, M)), (indices_row, neighbor_indices_all), weights_all
    )
    E_sqrtm = jnp.diag(uncertainties_all)
    return L, jnp.sqrt(jnp.abs(E_sqrtm))


@partial(jax.jit, static_argnums=(2, 3, 4))
def fd_coeff(x, neighbors, k, L_k, LL_k, nugget_gram_matrix=0.0):
    """Compute kernel-based finite difference coefficients."""

    X = neighbors
    gram_matrix = k(X, X.T) + nugget_gram_matrix * jnp.eye(X.shape[0])
    diffop_at_point = L_k(x[None, :], X.T).reshape((-1,))
    weights = jnp.linalg.solve(gram_matrix, diffop_at_point)
    uncertainty = LL_k(x, x).reshape(()) - weights @ diffop_at_point

    return weights, uncertainty
