import jax.numpy as jnp
import jax.scipy.linalg
import tornadox


class BlockDiagonal:
    def __init__(self, array_list):
        self._array_list = array_list

    @property
    def array_list(self):
        return self._array_list

    @property
    def num_blocks(self):
        return len(self._array_list)

    @property
    def shape(self):
        return (
            sum((a.shape[0] for a in self._array_list)),
            sum((a.shape[1] for a in self._array_list)),
        )

    @property
    def T(self):
        transposed_array_list = [a.T for a in self._array_list]
        return BlockDiagonal(array_list=transposed_array_list)

    def todense(self):
        return jax.scipy.linalg.block_diag(*self.array_list)

    def __matmul__(self, other):
        if isinstance(other, jnp.ndarray):
            split_indices_colwise = jnp.cumsum(
                jnp.array([a.shape[1] for a in self._array_list])
            )[:-1]
            if other.ndim == 1:
                split_array = jnp.split(other, split_indices_colwise)
                individual_products = [
                    self._array_list[i] @ split_array[i] for i in range(self.num_blocks)
                ]
                return jnp.concatenate(individual_products)

            elif other.ndim == 2:
                in_block_rows = jnp.split(other, split_indices_colwise, axis=0)
                out_block_rows = [
                    self._array_list[block_diag_index] @ r
                    for block_diag_index, r in enumerate(in_block_rows)
                ]
                return jnp.concatenate(out_block_rows, axis=0)

            else:
                raise ValueError("BlockDiagonal matmul is not supported for n_dim > 2.")

        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, jnp.ndarray):
            split_indices_rowwise = jnp.cumsum(
                jnp.array([a.shape[0] for a in self._array_list])
            )[:-1]
            if other.ndim == 1:
                split_array = jnp.split(other, split_indices_rowwise)
                individual_products = [
                    split_array[i] @ self._array_list[i] for i in range(self.num_blocks)
                ]
                return jnp.concatenate(individual_products)

            elif other.ndim == 2:
                in_block_cols = jnp.split(other, split_indices_rowwise, axis=1)
                out_block_cols = [
                    c @ self._array_list[block_diag_index]
                    for block_diag_index, c in enumerate(in_block_cols)
                ]
                return jnp.concatenate(out_block_cols, axis=1)

            else:
                raise ValueError("BlockDiagonal matmul is not supported for n_dim > 2.")

        return NotImplemented


class StackedSSM:
    def __init__(self, processes) -> None:
        self.processes = tuple(processes)
        self._dims = tuple((p.state_dimension for p in self.processes))

    @property
    def state_dimension(self):
        return sum(self._dims)

    @property
    def preconditioned_discretize(self):
        As, Qs = [], []
        for p in self.processes:
            A, SQ = p.preconditioned_discretize
            As.append(A)
            Qs.append(SQ @ SQ.T)

        # A = jax.scipy.linalg.block_diag(*As)
        Q = jax.scipy.linalg.block_diag(*Qs)
        return As, jnp.linalg.cholesky(Q)

    def non_preconditioned_discretize(self, dt):
        As, Qs = [], []
        for p in self.processes:
            A, SQ = p.non_preconditioned_discretize(dt)
            As.append(A)
            Qs.append(SQ @ SQ.T)

        # A = jax.scipy.linalg.block_diag(*As)
        Q = jax.scipy.linalg.block_diag(*Qs)
        return As, jnp.linalg.cholesky(Q)

    def nordsieck_preconditioner(self, dt):
        Ps, P_invs = [], []
        for p in self.processes:
            prec, prec_inv = p.nordsieck_preconditioner(dt)
            Ps.append(prec)
            P_invs.append(prec_inv)
        # P = jax.scipy.linalg.block_diag(*Ps)
        # P_inv = jax.scipy.linalg.block_diag(*P_invs)
        return Ps, P_invs

    def projection_matrix(
        self, derivative_to_project_onto, process_to_project_onto=None
    ):
        if process_to_project_onto is None:
            return [
                p.projection_matrix(derivative_to_project_onto) for p in self.processes
                ]

        assert isinstance(process_to_project_onto, int)
        proj_to_proc = self.projection_to_process(process_to_project_onto)
        proj_to_deriv = self.processes[process_to_project_onto].projection_matrix(
            derivative_to_project_onto
        )
        return proj_to_deriv @ proj_to_proc

    def projection_to_process(self, process_to_project_onto: int):
        start = (
            sum(self._dims[0:process_to_project_onto])
            if process_to_project_onto > 0
            else 0
        )
        stop = (
            start + self._dims[process_to_project_onto]
            if process_to_project_onto < len(self.processes)
            else None
        )
        return jnp.eye(self.state_dimension)[start:stop, :]
