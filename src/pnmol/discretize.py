"""Discretise differential operators on a mesh, assuming an underlying function space."""

from functools import partial

import jax
import jax.numpy as jnp
import tqdm

from pnmol import diffops, kernels


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
        kernel = kernels.SquareExponential(input_scale=1.0, output_scale=1.0)

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


def fd_probabilistic_neumann_1d(
    mesh_spatial,
    kernel=None,
    stencil_size=2,
    nugget_gram_matrix=0.0,
):
    if stencil_size != 2:
        raise NotImplementedError
    if kernel is None:
        kernel = kernels.SquareExponential(input_scale=1.0, output_scale=1.0)

    # Differentiate the kernels
    D = diffops.gradient()  # 1d, so gradient works.
    k = kernel
    Lk = kernels.Lambda(D(k.pairwise, argnums=0))
    LLk = kernels.Lambda(D(Lk.pairwise, argnums=1))

    # Fix the inputs of the coefficient function
    fd_coeff_neumann_fixed = partial(
        _fd_coefficients_neumann,
        mesh_spatial=mesh_spatial,
        k=k,
        L_k=Lk,
        LL_k=LLk,
        nugget_gram_matrix=nugget_gram_matrix,
    )
    fd_coeff_neumann = jax.jit(fd_coeff_neumann_fixed)

    # Compute left and right weights and uncertainties
    weights_left, uncertainty_left = fd_coeff_neumann(idx_x=0, idx_neighbors=(0, 1))
    weights_right, uncertainty_right = fd_coeff_neumann(
        idx_x=-1, idx_neighbors=(-1, -2)
    )

    # Projection matrix to the boundaries _including_ their neighbors
    # The order of the indices of the right boundary reflects their locations in 'neighbors'
    B = jnp.eye(len(mesh_spatial))[((0, 1, -1, -2),)]

    # negate the left weights, because the normal derivative at the left boundary
    # "points to the left", whereas standard derivatives "point to the right"
    diffmatrix = jax.scipy.linalg.block_diag(-weights_left, weights_right)
    errormatrix = jnp.diag(jnp.array([uncertainty_left, uncertainty_right]))
    return diffmatrix @ B, errormatrix


def _fd_coefficients_neumann(
    idx_x, idx_neighbors, mesh_spatial, k, L_k, LL_k, nugget_gram_matrix
):
    """Compute finite difference coefficients for Neumann (normal) derivative."""
    x = mesh_spatial[idx_x]
    neighbors = mesh_spatial[(idx_neighbors,)]
    return fd_coefficients(
        x=x,
        neighbors=neighbors,
        k=k,
        L_k=L_k,
        LL_k=LL_k,
        nugget_gram_matrix=nugget_gram_matrix,
    )


@partial(jax.jit, static_argnums=(2, 3, 4))
def fd_coefficients(x, neighbors, k, L_k, LL_k, nugget_gram_matrix=0.0):
    """Compute kernel-based finite difference coefficients."""

    X, n = neighbors, neighbors.shape[0]
    gram_matrix = k(X, X.T) + nugget_gram_matrix * jnp.eye(n)
    diffop_at_point = L_k(x[None, :], X.T).reshape((-1,))

    weights = jnp.linalg.solve(gram_matrix, diffop_at_point)
    uncertainty = LL_k(x, x).reshape(()) - weights @ diffop_at_point

    return weights, uncertainty


def collocation_global(
    diffop,
    mesh_spatial,
    kernel=None,
    nugget_gram_matrix=0.0,
):
    """Discretize a differential operator with global, unsymmetric collocation."""

    if kernel is None:
        kernel = kernels.SquareExponential(input_scale=1.0, output_scale=1.0)

    # Fix kernel arguments in FD function
    L_kx = kernels.Lambda(diffop(kernel.pairwise, argnums=0))
    LL_kx = kernels.Lambda(diffop(L_kx.pairwise, argnums=1))

    gram_matrix = kernel(mesh_spatial.points, mesh_spatial.points.T)
    gram_matrix += nugget_gram_matrix * jnp.eye(mesh_spatial.shape[0])

    diffop_at_set = L_kx(mesh_spatial.points, mesh_spatial.points.T)

    D = jnp.linalg.solve(gram_matrix, diffop_at_set)

    x = LL_kx(mesh_spatial.points, mesh_spatial.points.T)
    E = x - D.T @ diffop_at_set
    return D, jnp.linalg.cholesky(E)
