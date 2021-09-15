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
    def T(self):
        transposed_array_list = [a.T for a in self._array_list]
        return BlockDiagonal(array_list=transposed_array_list)

    def todense(self):
        return jax.scipy.linalg.block_diag(*self.array_list)

    def __matmul__(self, other):
        if isinstance(other, jnp.ndarray) and other.ndim == 1:
            split_array = jnp.split(
                other,
                jnp.cumsum(jnp.array([a.shape[1] for a in self._array_list]))[:-1],
            )

            individual_products = [
                self._array_list[i] @ split_array[i] for i in range(self.num_blocks)
            ]

            return jnp.concatenate(individual_products)

        return NotImplemented

    def __rmatmul__(self, other):
        if isinstance(other, jnp.ndarray) and other.ndim == 1:
            split_array = jnp.split(
                other,
                jnp.cumsum(jnp.array([a.shape[0] for a in self._array_list]))[:-1],
            )

            individual_products = [
                split_array[i] @ self._array_list[i] for i in range(self.num_blocks)
            ]

            return jnp.concatenate(individual_products)

        return NotImplemented
