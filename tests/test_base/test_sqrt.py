"""Tests for square-root utilities."""

import jax.numpy as jnp
import pytest

import pnmol
import pnmol.base.iwp


@pytest.fixture
def iwp():
    """Steal system matrices from an IWP transition."""
    return pnmol.base.iwp.IntegratedWienerTransition(
        wiener_process_dimension=1,
        num_derivatives=1,
        wp_diffusion_sqrtm=jnp.eye(1),
    )


@pytest.fixture
def H_and_SQ(iwp, measurement_style):
    """Measurement model via IWP system matrices."""
    H, SQ = iwp.preconditioned_discretize_1d

    if measurement_style == "full":
        return H, SQ
    return H[:1], SQ[:1, :1]


@pytest.fixture
def SC(iwp):
    """Initial covariance via IWP process noise."""
    return iwp.preconditioned_discretize_1d[1]


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_propagate_cholesky_factor(H_and_SQ, SC, measurement_style):
    """Assert that sqrt propagation coincides with non-sqrt propagation."""
    H, SQ = H_and_SQ

    # First test: Non-optional S2
    chol = pnmol.base.sqrt.propagate_cholesky_factor(S1=(H @ SC), S2=SQ)
    cov = H @ SC @ SC.T @ H.T + SQ @ SQ.T
    assert jnp.allclose(chol @ chol.T, cov)
    assert jnp.allclose(jnp.tril(chol), chol)


@pytest.mark.parametrize("measurement_style", ["full", "partial"])
def test_update_sqrt(H_and_SQ, SC, measurement_style):
    """Sqrt-update coincides with non-square-root update."""

    H, SQ = H_and_SQ

    SC_new, kalman_gain, innov_chol = pnmol.base.sqrt.update_sqrt(
        H, SC, meascov_sqrtm=SQ
    )
    assert isinstance(SC_new, jnp.ndarray)
    assert isinstance(kalman_gain, jnp.ndarray)
    assert isinstance(innov_chol, jnp.ndarray)
    assert SC_new.shape == SC.shape
    assert kalman_gain.shape == (H.shape[1], H.shape[0])
    assert innov_chol.shape == (H.shape[0], H.shape[0])

    # expected:
    S = H @ SC @ SC.T @ H.T + SQ @ SQ.T
    K = SC @ SC.T @ H.T @ jnp.linalg.inv(S)
    C = SC @ SC.T - K @ S @ K.T

    # Test SC
    assert jnp.allclose(SC_new @ SC_new.T, C)
    assert jnp.allclose(SC_new, jnp.tril(SC_new))

    # Test K
    assert jnp.allclose(K, kalman_gain)

    # Test S
    assert jnp.allclose(innov_chol @ innov_chol.T, S)
    assert jnp.allclose(innov_chol, jnp.tril(innov_chol))
