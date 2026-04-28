"""Unit + hypothesis tests for `priya_forecast.models.normalization`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.models import NormalizationSpec
from priya_forecast.models.normalization import (
    DEFAULT_K_MAX,
    DEFAULT_K_MIN,
    derive_from_gp,
    from_files,
    identity,
)
from priya_forecast.parameters import fiducial_vector


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _spec(nk: int = 35) -> NormalizationSpec:
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, nk)
    return NormalizationSpec(
        param_min=0.8,
        param_max=1.05,
        k_min=DEFAULT_K_MIN,
        k_max=DEFAULT_K_MAX,
        mean_flux=np.linspace(2.0, 3.0, nk),
        std_flux=np.linspace(0.5, 0.7, nk),
        k_grid=k_grid,
    )


# ---------------------------------------------------------------------------
# NormalizationSpec construction + validation
# ---------------------------------------------------------------------------


def test_spec_constructs_with_valid_inputs():
    s = _spec()
    assert s.mean_flux.shape == s.std_flux.shape == s.k_grid.shape == (35,)


def test_spec_rejects_inverted_param_range():
    with pytest.raises(ValueError, match="param_max"):
        NormalizationSpec(
            param_min=1.0,
            param_max=0.5,
            k_min=DEFAULT_K_MIN,
            k_max=DEFAULT_K_MAX,
            mean_flux=np.zeros(5),
            std_flux=np.ones(5),
            k_grid=np.linspace(0.001, 0.02, 5),
        )


def test_spec_rejects_zero_std():
    with pytest.raises(ValueError, match="std_flux"):
        NormalizationSpec(
            param_min=0,
            param_max=1,
            k_min=DEFAULT_K_MIN,
            k_max=DEFAULT_K_MAX,
            mean_flux=np.zeros(5),
            std_flux=np.array([1.0, 0.0, 1.0, 1.0, 1.0]),
            k_grid=np.linspace(0.001, 0.02, 5),
        )


def test_spec_rejects_mismatched_shapes():
    with pytest.raises(ValueError, match="must match"):
        NormalizationSpec(
            param_min=0,
            param_max=1,
            k_min=DEFAULT_K_MIN,
            k_max=DEFAULT_K_MAX,
            mean_flux=np.zeros(5),
            std_flux=np.ones(6),
            k_grid=np.linspace(0.001, 0.02, 5),
        )


def test_spec_rejects_non_monotone_k_grid():
    with pytest.raises(ValueError, match="strictly increasing"):
        NormalizationSpec(
            param_min=0,
            param_max=1,
            k_min=DEFAULT_K_MIN,
            k_max=DEFAULT_K_MAX,
            mean_flux=np.zeros(5),
            std_flux=np.ones(5),
            k_grid=np.array([0.001, 0.002, 0.0015, 0.003, 0.004]),
        )


# ---------------------------------------------------------------------------
# Forward/inverse — unit tests
# ---------------------------------------------------------------------------


def test_normalize_param_endpoints():
    s = _spec()
    assert s.normalize_param(s.param_min) == pytest.approx(0.0)
    assert s.normalize_param(s.param_max) == pytest.approx(1.0)


def test_normalize_k_endpoints():
    s = _spec()
    assert np.allclose(s.normalize_k(s.k_min), 0.0)
    assert np.allclose(s.normalize_k(s.k_max), 1.0)


def test_denormalize_zero_norm_recovers_mean():
    """flux_norm = 0 must yield P_F = mean_k(k_target)."""
    s = _spec()
    out = s.denormalize_flux(np.zeros(s.k_grid.size), s.k_grid)
    np.testing.assert_allclose(out, s.mean_flux)


def test_denormalize_unit_norm_recovers_mean_plus_std():
    s = _spec()
    out = s.denormalize_flux(np.ones(s.k_grid.size), s.k_grid)
    np.testing.assert_allclose(out, s.mean_flux + s.std_flux)


def test_denormalize_interpolates_to_target_grid():
    """Off-grid k values use linear interp on mean_k / std_k."""
    s = _spec()
    k_target = np.array([0.005, 0.015])  # not in s.k_grid (which is uniform 0.001..0.02)
    flux_norm = np.array([1.0, -1.0])
    out = s.denormalize_flux(flux_norm, k_target)
    expected_mean = np.interp(k_target, s.k_grid, s.mean_flux)
    expected_std = np.interp(k_target, s.k_grid, s.std_flux)
    np.testing.assert_allclose(out, flux_norm * expected_std + expected_mean)


def test_denormalize_rejects_shape_mismatch():
    s = _spec()
    with pytest.raises(ValueError, match="must match"):
        s.denormalize_flux(np.zeros(5), np.zeros(7))


# ---------------------------------------------------------------------------
# from_files / identity
# ---------------------------------------------------------------------------


def test_from_files_loads_student_txt_format(tmp_path: Path):
    nk = 35
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, nk)
    mean_path = tmp_path / "mean_flux_low_ns.txt"
    std_path = tmp_path / "std_flux_low_ns.txt"
    np.savetxt(mean_path, np.linspace(2, 3, nk))
    np.savetxt(std_path, np.linspace(0.5, 0.7, nk))
    spec = from_files(
        param_name="ns", mean_flux_path=mean_path, std_flux_path=std_path, k_grid=k_grid,
    )
    assert spec.param_min == 0.8  # ns prior lower
    assert spec.param_max == 1.05
    np.testing.assert_allclose(spec.mean_flux, np.linspace(2, 3, nk))


def test_from_files_loads_real_student_2pvar_file():
    """Load one of the student's actual 2pvar mean/std files and round-trip."""
    mean_path = Path(
        "/home/mfho/student_projects/InferenceLyaData/2pvar/mean_flux_low_ns-hub.txt"
    )
    std_path = Path(
        "/home/mfho/student_projects/InferenceLyaData/2pvar/std_flux_low_ns-hub.txt"
    )
    if not mean_path.exists():
        pytest.skip("Student 2pvar files not present in this environment.")
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, 35)
    spec = from_files(
        param_name="ns",
        mean_flux_path=mean_path,
        std_flux_path=std_path,
        k_grid=k_grid,
    )
    assert spec.mean_flux.shape == (35,)
    assert spec.std_flux.shape == (35,)
    # Sanity: positive P_F means; non-trivial spread.
    assert np.all(spec.mean_flux > 0)
    assert np.all(spec.std_flux > 0)
    # Round-trip: feed unit flux_norm, recover mean + std.
    out = spec.denormalize_flux(np.ones(35), k_grid)
    np.testing.assert_allclose(out, spec.mean_flux + spec.std_flux, rtol=1e-12)


def test_derive_from_gp_matches_student_file_format():
    """`derive_from_gp` must produce the same `(mean, std)` shape the student
    saves to `mean_flux_low_*.txt` / `std_flux_low_*.txt` — so the two
    normalization sources are interchangeable in the YAML."""
    from priya_forecast.models import MockGPModel

    nk = 35
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, nk)
    spec = derive_from_gp(
        gp_model=MockGPModel(),
        param_name="ns",
        z=3.6,
        k_grid=k_grid,
        n_samples=50,  # student uses npoints=50 in pysr_mf_given.py
    )
    assert spec.mean_flux.shape == (nk,)
    assert spec.std_flux.shape == (nk,)
    # Mock returns positive P_F, so per-k means must be positive.
    assert np.all(spec.mean_flux > 0)
    # Spread must be a small fraction of mean (mock has only 5-20% sensitivity).
    assert np.all(spec.std_flux < spec.mean_flux)


def test_from_files_rejects_unknown_param(tmp_path: Path):
    p = tmp_path / "x.txt"
    np.savetxt(p, np.zeros(5))
    with pytest.raises(KeyError):
        from_files(
            param_name="not_a_real_param",
            mean_flux_path=p,
            std_flux_path=p,
            k_grid=np.linspace(0.001, 0.02, 5),
        )


def test_identity_round_trip_is_pass_through():
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, 10)
    s = identity(k_grid)
    out = s.denormalize_flux(np.full(10, 5.0), k_grid)
    np.testing.assert_allclose(out, 5.0)


# ---------------------------------------------------------------------------
# derive_from_gp uses a mock GP — fully synthetic
# ---------------------------------------------------------------------------


class _MockGP:
    """Trivial mock: P_F(theta, k, z) = a * theta_idx + b*k for some idx."""

    def __init__(self, varying_idx: int = 2, slope: float = 10.0, k_slope: float = 100.0):
        self.idx = varying_idx
        self.slope = slope
        self.k_slope = k_slope

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        return self.slope * theta[self.idx] + self.k_slope * k


def test_derive_from_gp_recovers_correct_per_k_mean_and_std():
    """For a linear-in-theta GP, mean_k and std_k are analytic."""
    nk = 20
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, nk)
    gp = _MockGP(varying_idx=2, slope=10.0, k_slope=100.0)
    spec = derive_from_gp(
        gp_model=gp,
        param_name="ns",
        z=3.6,
        k_grid=k_grid,
        n_samples=4096,
        seed=0,
    )
    # ns prior is (0.8, 1.05) → uniform mean = 0.925 → contribution to flux mean = 9.25
    expected_mean = 9.25 + 100.0 * k_grid
    np.testing.assert_allclose(spec.mean_flux, expected_mean, rtol=2e-3)
    # uniform std on (lo, hi) = (hi-lo) / sqrt(12); slope-amplified
    expected_std = 10.0 * (1.05 - 0.8) / np.sqrt(12)
    np.testing.assert_allclose(spec.std_flux, expected_std, rtol=2e-2)


def test_derive_from_gp_uses_provided_fiducial_for_other_params():
    """Other-parameter values come from `fiducial_theta`."""
    nk = 5
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, nk)

    # Mock GP returns theta[3] (Ap idx) regardless of varied param.
    class FidProbe:
        def predict(self, theta, k, z):
            return np.full_like(k, theta[3])

    fid = np.array(fiducial_vector(), dtype=float)
    spec = derive_from_gp(
        gp_model=FidProbe(),
        param_name="ns",
        z=3.6,
        k_grid=k_grid,
        n_samples=10,
        fiducial_theta=fid,
    )
    np.testing.assert_allclose(spec.mean_flux, fid[3])


def test_derive_from_gp_seed_reproducibility():
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, 8)
    gp = _MockGP()
    s1 = derive_from_gp(gp_model=gp, param_name="ns", z=3.6, k_grid=k_grid, seed=42)
    s2 = derive_from_gp(gp_model=gp, param_name="ns", z=3.6, k_grid=k_grid, seed=42)
    np.testing.assert_allclose(s1.mean_flux, s2.mean_flux)
    np.testing.assert_allclose(s1.std_flux, s2.std_flux)


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(
    flux_norm=st.lists(
        st.floats(min_value=-10, max_value=10, allow_nan=False, allow_infinity=False),
        min_size=5, max_size=5,
    ),
)
@settings(max_examples=30, deadline=None)
def test_property_round_trip_inverse_of_forward(flux_norm: list[float]):
    """For any (theta, k) in their training ranges, normalize then denormalize
    is the identity (modulo interp at the same grid)."""
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, 5)
    s = NormalizationSpec(
        param_min=0,
        param_max=1,
        k_min=DEFAULT_K_MIN,
        k_max=DEFAULT_K_MAX,
        mean_flux=np.linspace(2.0, 3.0, 5),
        std_flux=np.linspace(0.5, 0.7, 5),
        k_grid=k_grid,
    )
    arr = np.asarray(flux_norm)
    p = s.denormalize_flux(arr, k_grid)
    # round-trip: subtract the per-k mean, divide by std → recover input
    recovered = (p - s.mean_flux) / s.std_flux
    np.testing.assert_allclose(recovered, arr, atol=1e-12, rtol=1e-12)


@given(
    param=st.floats(min_value=0.0, max_value=1.0, allow_nan=False),
)
@settings(max_examples=50, deadline=None)
def test_property_normalize_param_in_unit_interval(param: float):
    s = NormalizationSpec(
        param_min=0, param_max=1,
        k_min=DEFAULT_K_MIN, k_max=DEFAULT_K_MAX,
        mean_flux=np.zeros(3), std_flux=np.ones(3),
        k_grid=np.linspace(0.001, 0.02, 3),
    )
    n = s.normalize_param(param)
    assert 0.0 <= n <= 1.0


@given(
    n_samples=st.integers(min_value=200, max_value=500),
    seed=st.integers(min_value=0, max_value=99),
)
@settings(max_examples=8, deadline=None)
def test_property_derive_from_gp_mean_converges(n_samples: int, seed: int):
    """As n_samples grows the empirical mean approaches the analytic mean."""
    nk = 6
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, nk)
    gp = _MockGP(varying_idx=2, slope=10.0, k_slope=0.0)
    spec = derive_from_gp(
        gp_model=gp, param_name="ns", z=3.6, k_grid=k_grid,
        n_samples=n_samples, seed=seed,
    )
    expected_mean = 10.0 * 0.5 * (0.8 + 1.05)  # uniform mean of slope*theta
    # Per-k mean is constant in this mock; tolerance loosens with n_samples
    tol = 5.0 / np.sqrt(n_samples)  # ~few sigma
    np.testing.assert_allclose(spec.mean_flux, expected_mean, atol=tol)
