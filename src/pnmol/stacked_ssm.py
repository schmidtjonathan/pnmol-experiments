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
