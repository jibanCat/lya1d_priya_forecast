"""Tests for KSDataLikelihood — pieces testable without lyaemu.

The KSData covariance load itself requires `lyaemu` (cluster-only); the
end-to-end fit-test for KSDataLikelihood is gated on that import. Here
we test the layout helper `_build_z_blocks` which doesn't need lyaemu.

Cluster smoke (when lyaemu is present): construct KSDataLikelihood with
mock_data='gp', verify shape of `inputs.d`, that `cov_chol` is real
upper/lower-tri Cholesky, and that `model_at(theta_fid) == d` exactly.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.ksdata_likelihood import KSDataLikelihood, _build_z_blocks


def test_z_blocks_simple():
    """Standard z-major layout: each unique z is one consecutive block."""
    z = np.array([2.6, 2.6, 2.6, 2.8, 2.8, 3.0, 3.0, 3.0, 3.0])
    blocks = _build_z_blocks(z)
    assert len(blocks) == 3
    assert blocks[0] == (2.6, slice(0, 3))
    assert blocks[1] == (2.8, slice(3, 5))
    assert blocks[2] == (3.0, slice(5, 9))


def test_z_blocks_empty():
    """Empty input → empty list."""
    assert _build_z_blocks(np.array([])) == []


def test_z_blocks_single():
    z = np.array([3.6, 3.6, 3.6])
    blocks = _build_z_blocks(z)
    assert blocks == [(3.6, slice(0, 3))]


def test_z_blocks_each_z_singleton():
    """If every row has a unique z, each block has length 1."""
    z = np.array([2.6, 2.8, 3.0, 3.2])
    blocks = _build_z_blocks(z)
    assert blocks == [(2.6, slice(0, 1)), (2.8, slice(1, 2)),
                      (3.0, slice(2, 3)), (3.2, slice(3, 4))]


def test_z_blocks_handles_float_tolerance():
    """Tiny float jitter (e.g. from loadtxt) shouldn't split a z bin."""
    z = np.array([2.6, 2.6 + 1e-9, 2.6 - 1e-9, 2.8])
    blocks = _build_z_blocks(z)
    assert len(blocks) == 2
    assert blocks[0][1] == slice(0, 3)
    assert blocks[1][1] == slice(3, 4)


def test_z_blocks_non_contiguous_z_raises_no_error_but_fragments():
    """If the same z appears non-contiguously (unusual), each occurrence
    becomes its own block. Document the behavior — KSData is z-major
    so this shouldn't happen in practice, but we don't sort/group across
    discontinuities to preserve cov-matrix row alignment.
    """
    z = np.array([2.6, 2.6, 3.0, 2.6])
    blocks = _build_z_blocks(z)
    assert blocks == [(2.6, slice(0, 2)), (3.0, slice(2, 3)), (2.6, slice(3, 4))]


# ------------------------------------------------------------------
# k_min validation: pure (no lyaemu) — validators run before the import.
# ------------------------------------------------------------------


def test_kmin_negative_rejected():
    """k_min must be >= 0."""
    with pytest.raises(ValueError, match="k_min must be >= 0"):
        KSDataLikelihood(model=None, k_min=-0.001, k_max=0.04)  # type: ignore[arg-type]


def test_kmin_ge_kmax_rejected():
    """k_min must be strictly less than k_max."""
    with pytest.raises(ValueError, match="k_min .* must be < k_max"):
        KSDataLikelihood(model=None, k_min=0.04, k_max=0.04)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="k_min .* must be < k_max"):
        KSDataLikelihood(model=None, k_min=0.05, k_max=0.04)  # type: ignore[arg-type]


# ------------------------------------------------------------------
# End-to-end smoke test: requires lyaemu. Skipped if not installed.
# ------------------------------------------------------------------

try:
    import lyaemu  # noqa: F401
    LYAEMU_AVAILABLE = True
except ImportError:
    LYAEMU_AVAILABLE = False


@pytest.mark.skipif(not LYAEMU_AVAILABLE, reason="requires lyaemu")
def test_ksdata_likelihood_construction_smoke():
    """End-to-end: construct KSDataLikelihood with mock_data='gp', verify
    `model_at(fid) == inputs.d` exactly and `cov_chol` is positive-definite
    Cholesky."""
    from priya_forecast.models.gp_model import MockGPModel
    from priya_forecast.parameters import fiducial_vector

    fid = np.array(fiducial_vector(), dtype=float)
    lk = KSDataLikelihood(
        model=MockGPModel(),
        z_min=2.6, z_max=4.2, k_max=0.064,
        mock_data="gp", theta_fid=fid,
    )
    n = len(lk.kept_z)
    assert n > 0, "KSData should have rows in [2.6, 4.2] × k≤0.064"
    # mock=gp → d == model.predict at fid stacked → identical to model_at(fid).
    np.testing.assert_allclose(
        lk.model_at(fid), lk.inputs.d, rtol=1e-12, atol=1e-12,
    )
    # cov_chol is lower-triangular and positive (diag entries > 0).
    assert lk.inputs.cov_chol.shape == (n, n)
    assert np.all(np.diag(lk.inputs.cov_chol) > 0)
    # Reconstruct cov from chol.
    L = lk.inputs.cov_chol
    np.testing.assert_allclose(L @ L.T, lk.inputs.cov, rtol=1e-10, atol=1e-10)


@pytest.mark.skipif(not LYAEMU_AVAILABLE, reason="requires lyaemu")
def test_kmin_drops_lowest_bins():
    """Setting k_min above the lowest retained k should drop those rows.

    With conservative=True, the lowest retained k is ≈0.0055. Setting
    k_min=0.01 should drop the first ~2 bins (k=0.0055, 0.0077) while
    k_min=0.0 keeps all of them.
    """
    from priya_forecast.models.gp_model import MockGPModel
    from priya_forecast.parameters import fiducial_vector

    fid = np.array(fiducial_vector(), dtype=float)
    lk_all = KSDataLikelihood(
        model=MockGPModel(),
        z_min=3.6, z_max=3.6, k_max=0.04, k_min=0.0,
        mock_data="gp", theta_fid=fid,
    )
    lk_cut = KSDataLikelihood(
        model=MockGPModel(),
        z_min=3.6, z_max=3.6, k_max=0.04, k_min=0.01,
        mock_data="gp", theta_fid=fid,
    )
    assert len(lk_cut.kept_k) < len(lk_all.kept_k)
    assert lk_cut.kept_k.min() >= 0.01 - 1e-6
    assert lk_all.kept_k.min() < 0.01


@pytest.mark.skipif(not LYAEMU_AVAILABLE, reason="requires lyaemu")
@settings(max_examples=15, deadline=None)
@given(
    k_min=st.floats(min_value=0.0, max_value=0.025),
    k_max_offset=st.floats(min_value=0.005, max_value=0.05),
)
def test_kept_k_within_bounds(k_min: float, k_max_offset: float):
    """Invariant: every kept_k satisfies k_min <= k <= k_max for any
    valid (k_min, k_max) range that produces non-empty kept rows."""
    from priya_forecast.models.gp_model import MockGPModel
    from priya_forecast.parameters import fiducial_vector

    k_max = k_min + k_max_offset
    fid = np.array(fiducial_vector(), dtype=float)
    try:
        lk = KSDataLikelihood(
            model=MockGPModel(),
            z_min=3.6, z_max=3.6, k_min=k_min, k_max=k_max,
            mock_data="gp", theta_fid=fid,
        )
    except ValueError:
        # Empty kept set is a valid outcome for some narrow ranges; only
        # the bounds invariant is what we're checking.
        return
    assert lk.kept_k.min() >= k_min - 1e-6
    assert lk.kept_k.max() <= k_max + 1e-6
