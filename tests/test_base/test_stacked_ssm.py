import jax
import jax.numpy as jnp
import pytest

import pnmol.base


def genkey(key):
    _, res = jax.random.split(key)
    return res


@pytest.fixture
def seed():
    return 41  # @__@


@pytest.fixture
def base_rng(seed):
    return jax.random.PRNGKey(seed)


@pytest.fixture(params=[3, 50, 100])
def n_blocks(request):
    return request.param


@pytest.fixture
def blocks_and_shape(n_blocks, base_rng):
    blocks = []
    total_shape = jnp.zeros(2, dtype=int)
    for _ in range(n_blocks):
        key = genkey(base_rng)
        shape = jax.random.randint(key, shape=(2,), minval=2, maxval=10)
        key = genkey(key)
        blocks.append(jax.random.normal(key, shape=shape))
        total_shape = total_shape + shape

    return blocks, total_shape, key


def test_matmul(blocks_and_shape):
    blocks, shape, rng = blocks_and_shape
    shape = tuple(shape)
    block_diag = pnmol.base.stacked_ssm.BlockDiagonal(blocks)
    dense_block_diag = jax.scipy.linalg.block_diag(*blocks)

    assert isinstance(block_diag, pnmol.base.stacked_ssm.BlockDiagonal)
    assert isinstance(dense_block_diag, jnp.ndarray)
    assert isinstance(block_diag.todense(), jnp.ndarray)
    assert block_diag.shape == shape == dense_block_diag.shape
    assert jnp.allclose(block_diag.todense(), dense_block_diag)

    # Create arrays to multiply the block diagonal matrix with
    u = jax.random.normal(rng, shape=(shape[1],))
    rng = genkey(rng)
    v = jax.random.normal(rng, shape=(shape[0],))

    rng = genkey(rng)
    out_dim = int(jax.random.randint(rng, shape=(), minval=4, maxval=10))
    rng = genkey(rng)
    M = jax.random.normal(rng, shape=(shape[1], out_dim))
    rng = genkey(rng)
    K = jax.random.normal(rng, shape=(shape[0], out_dim))

    # Test correct outcomes
    # 1. __matmul__:
    # 1.1. mat-vec
    assert jnp.allclose(block_diag @ u, dense_block_diag @ u)
    assert jnp.allclose(block_diag.T @ v, dense_block_diag.T @ v)
    # 1.2 mat-mat
    assert jnp.allclose(block_diag @ M, dense_block_diag @ M)
    assert jnp.allclose(block_diag.T @ K, dense_block_diag.T @ K)
    # 2. __rmatmul__:
    # 2.1. vec-mat
    assert jnp.allclose(v @ block_diag, v @ dense_block_diag)
    assert jnp.allclose(u @ block_diag.T, u @ dense_block_diag.T)
    # 2.2 mat-mat
    assert jnp.allclose(M.T @ block_diag, M.T @ dense_block_diag)
    assert jnp.allclose(K.T @ block_diag.T, K.T @ dense_block_diag.T)
