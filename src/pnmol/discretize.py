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
    stencil_size_interior=3,
    stencil_size_boundary=3,
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

    # Select points on the boundary and in the interior
    points_interior, _, indices_interior = mesh_spatial.interior
    points_boundary, _, indices_boundary = mesh_spatial.boundary

    # Read off all neighbors in a single batch (the underlying KDTree is vectorized)
    neighbors_interior, neighbor_indices_interior = mesh_spatial.neighbours(
        point=points_interior, num=stencil_size_interior
    )
    neighbors_boundary, neighbor_indices_boundary = mesh_spatial.neighbours(
        point=points_boundary, num=stencil_size_boundary
    )

    # Compute all FD coefficients in a single batch
    weights_interior, uncertainties_interior = fd_coeff_fun_batched(
        x=points_interior, neighbors=neighbors_interior
    )
    weights_boundary, uncertainties_boundary = fd_coeff_fun_batched(
        x=points_boundary, neighbors=neighbors_boundary
    )

    # Stack the resulting weights and uncertainties into a matrix
    L = jnp.zeros((mesh_spatial.shape[0], mesh_spatial.shape[0]), dtype=jnp.float64)
    E_sqrtm = jnp.zeros(
        (mesh_spatial.shape[0], mesh_spatial.shape[0]), dtype=jnp.float64
    )
    L_boundary_weights, E_sqrtm_boundary_weights = _weights_to_matrix(
        L=L,
        E_sqrtm=E_sqrtm,
        weights=weights_boundary,
        uncertainties=uncertainties_boundary,
        indices_column=neighbor_indices_boundary,
        indices_row=indices_boundary[:, None],
    )
    L_all_weights, E_sqrtm_all_weights = _weights_to_matrix(
        L=L_boundary_weights,
        E_sqrtm=E_sqrtm_boundary_weights,
        weights=weights_interior,
        uncertainties=uncertainties_interior,
        indices_column=neighbor_indices_interior,
        indices_row=indices_interior[:, None],
    )
    return L_all_weights, E_sqrtm_all_weights


@jax.jit
def _weights_to_matrix(L, E_sqrtm, weights, uncertainties, indices_column, indices_row):
    """Stack the FD weights (and uncertainties) into a differentiation matrix."""
    L_new = jax.ops.index_update(x=L, idx=(indices_row, indices_column), y=weights)
    E_sqrtm_new = jax.ops.index_update(
        x=E_sqrtm, idx=(indices_row, indices_row.T), y=uncertainties
    )
    return L_new, E_sqrtm_new


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
    nugget_cholesky_E=0.0,
    symmetrize_cholesky_E=False,
):
    """Discretize a differential operator with global, unsymmetric collocation."""

    if kernel is None:
        kernel = kernels.SquareExponential(input_scale=1.0, output_scale=1.0)

    # Differentiate kernel
    k = kernel
    L_kx = kernels.Lambda(diffop(k.pairwise, argnums=0))
    LL_kx = kernels.Lambda(diffop(L_kx.pairwise, argnums=1))

    # Assemble Gram matrices
    gram_matrix_k = kernel(mesh_spatial.points, mesh_spatial.points.T)
    gram_matrix_k += nugget_gram_matrix * jnp.eye(mesh_spatial.shape[0])
    gram_matrix_Lk = L_kx(mesh_spatial.points, mesh_spatial.points.T)
    gram_matrix_LLk = LL_kx(mesh_spatial.points, mesh_spatial.points.T)

    # Compute differentiation matrix and error covariance matrix
    D = jnp.linalg.solve(gram_matrix_k, gram_matrix_Lk.T).T
    E = gram_matrix_LLk - D @ gram_matrix_Lk.T

    # Symmetrize and add nugget
    if symmetrize_cholesky_E:
        E = 0.5 * (E + E.T)
    E += nugget_cholesky_E * jnp.eye(mesh_spatial.shape[0])
    return D, jnp.linalg.cholesky(E)
