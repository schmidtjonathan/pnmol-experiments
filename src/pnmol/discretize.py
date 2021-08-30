"""Discretise differential operators on a mesh, assuming an underlying function space."""

import numpy as np
import scipy.sparse
import tqdm

from pnmol import kernels


def discretize(diffop, mesh, kernel, stencil_size):
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
    num_neighbors
        Number of local neighbours to use for localised discretization.
        Optional. Default is ``None``, which implies that all points are used.

    Returns
    -------
    np.ndarray
        The discretized linear operator as a ``np.ndarray``.
    """

    M = len(mesh)
    interior_points, _ = mesh.interior

    L_k = diffop(kernel, argnums=0)  # derivative function of kernel
    LL_k = diffop(L_k, argnums=1)
    L_kx = kernels.Kernel(fun=L_k)
    LL_kx = kernels.Kernel(fun=LL_k)

    L_data = []
    L_row = []
    L_col = []

    E_diag = np.zeros(M)
    progressbar = tqdm.tqdm(interior_points)

    for point in progressbar:

        neighbors, neighbor_idcs = mesh.neighbours(point=point, num=stencil_size)

        gram_matrix = kernel(
            neighbors, neighbors, as_matrix=True
        )  # [stencil_size, stencil_size]
        diffop_at_point = L_kx(
            point.reshape(1, -1), neighbors, as_matrix=True
        ).squeeze()  # [stencil_size, ]

        # weights = diffop_at_point @ np.linalg.inv(gram_matrix)  # [stencil_size,]
        weights = np.linalg.solve(gram_matrix, diffop_at_point)  # [stencil_size,]
        L_data.append(weights)
        L_row.append(
            np.full(shape=stencil_size, fill_value=neighbor_idcs[0], dtype=int)
        )
        L_col.append(neighbor_idcs)

        E_term1 = LL_kx(
            point.reshape(1, -1), point.reshape(1, -1), as_matrix=True
        ).squeeze()
        E_term2 = (
            weights @ L_kx(neighbors, point.reshape(1, -1), as_matrix=True).squeeze()
        )
        E_diag[neighbor_idcs[0]] = E_term1 - E_term2

        progressbar.set_description(str(np.array(point).round(3)))

    L_data = np.concatenate(L_data)
    L_row = np.concatenate(L_row)
    L_col = np.concatenate(L_col)

    L = scipy.sparse.bsr_matrix((L_data, (L_row, L_col)), shape=(M, M))
    E = scipy.sparse.diags(E_diag)

    return L, E
