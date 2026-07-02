# tests/test_stage10_gate.py
"""Tests for Stage 10: derivative_faithful_multiz + build_refit_from_pareto_multiz_gated."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from priya_forecast.derivative_gate import derivative_faithful_multiz
from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.multi_z.refit import build_refit_from_pareto_multiz_gated


# ---------------------------------------------------------------------------
# Shared fixtures / helpers
# ---------------------------------------------------------------------------

def _make_k_grid(n=6):
    return np.linspace(0.005, 0.04, n)


def _make_z_grid():
    return np.array([3.4, 3.6], dtype=float)


def _make_fid():
    """11D fiducial vector (internal-unit order from PARAMS_11D)."""
    from priya_forecast.parameters import fiducial_vector
    return np.asarray(fiducial_vector(), dtype=float)


class _StubRefit:
    """Minimal stand-in for Refit1DResult with a controllable gradient slope."""

    def __init__(self, slope: float = 1.0):
        self.slope = slope

    def predict(self, theta_phys, k, resolution=0.8, z=None):
        k = np.asarray(k, dtype=float)
        return self.slope * float(theta_phys) * np.ones_like(k)


class _StubGP:
    """Minimal stand-in for GPModel.

    predict(theta, k, z) returns slope_gp * theta[param_idx] * ones(k).
    param_idx=2 corresponds to 'ns' in PARAMS_11D.
    """

    def __init__(self, slope: float = 1.0, param_idx: int = 2):
        self.slope = slope
        self.param_idx = param_idx

    def predict(self, theta, k, z):
        k = np.asarray(k, dtype=float)
        return self.slope * float(np.asarray(theta)[self.param_idx]) * np.ones_like(k)


# ---------------------------------------------------------------------------
# Part A: derivative_faithful_multiz unit tests
# ---------------------------------------------------------------------------

def test_derivative_faithful_multiz_faithful():
    """Equation gradient equals GP gradient: should pass."""
    k_grid = _make_k_grid()
    z_grid = _make_z_grid()
    fid = _make_fid()
    # ns is index 2
    gp = _StubGP(slope=1.0, param_idx=2)
    refit = _StubRefit(slope=1.0)
    result = derivative_faithful_multiz(
        refit=refit, gp=gp, fid=fid, fid_value=fid[2],
        k_grid=k_grid, z_grid=z_grid, param_idx=2,
        tol=0.25,
    )
    assert result is True


def test_derivative_faithful_multiz_unfaithful():
    """Equation gradient is 2x the GP gradient: ratio deviates by 1.0, fails tol=0.25."""
    k_grid = _make_k_grid()
    z_grid = _make_z_grid()
    fid = _make_fid()
    gp = _StubGP(slope=1.0, param_idx=2)
    refit = _StubRefit(slope=2.0)   # double the slope -> ratio=2, |ratio-1|=1
    result = derivative_faithful_multiz(
        refit=refit, gp=gp, fid=fid, fid_value=fid[2],
        k_grid=k_grid, z_grid=z_grid, param_idx=2,
        tol=0.25,
    )
    assert result is False


def test_derivative_faithful_multiz_near_zero_gp_gradient():
    """GP gradient is identically zero -> all bins masked -> returns False."""
    k_grid = _make_k_grid()
    z_grid = _make_z_grid()
    fid = _make_fid()

    class _ZeroGP:
        def predict(self, theta, k, z):
            return np.zeros_like(np.asarray(k, dtype=float))

    refit = _StubRefit(slope=1.0)
    result = derivative_faithful_multiz(
        refit=refit, gp=_ZeroGP(), fid=fid, fid_value=fid[2],
        k_grid=k_grid, z_grid=z_grid, param_idx=2,
        tol=0.25,
    )
    assert result is False


def test_derivative_faithful_multiz_aggregates_over_z():
    """One z passes, one z fails: the median over all (k,z) governs the decision.

    Setup:
      - Equation: slope=1.0  (refit.predict returns 1.0 * theta_phys * ones(k))
      - GP at z=3.4 (good): slope_gp=1.0  -> rel=|1/1 - 1|=0.0 per k-bin (passes)
      - GP at z=3.6 (bad):  slope_gp=0.5  -> rel=|1/0.5 - 1|=1.0 per k-bin (fails)

    With n=4 k-bins:
      - good z only  (4 values of 0.0):  median=0.0  <= tol=0.25  -> True
      - bad  z only  (4 values of 1.0):  median=1.0  >  tol=0.25  -> False
      - both z's     (4×0 + 4×1 = 8 values): median=0.5 > tol=0.25 -> False
    """
    k_grid = _make_k_grid(n=4)
    z_good = 3.4
    z_bad = 3.6
    fid = _make_fid()

    class _PerZGP:
        """GP whose effective slope depends on z: 1.0 at z_good, 0.5 at z_bad."""
        def predict(self, theta, k, z):
            k = np.asarray(k, dtype=float)
            slope = 1.0 if float(z) == z_good else 0.5
            return slope * float(np.asarray(theta)[2]) * np.ones_like(k)

    refit = _StubRefit(slope=1.0)
    gp = _PerZGP()

    # (a) good z only -> passes
    result_good = derivative_faithful_multiz(
        refit=refit, gp=gp, fid=fid, fid_value=fid[2],
        k_grid=k_grid, z_grid=np.array([z_good]), param_idx=2, tol=0.25,
    )
    assert result_good is True, "good-z-only: expected True (median rel=0.0)"

    # (b) bad z only -> fails
    result_bad = derivative_faithful_multiz(
        refit=refit, gp=gp, fid=fid, fid_value=fid[2],
        k_grid=k_grid, z_grid=np.array([z_bad]), param_idx=2, tol=0.25,
    )
    assert result_bad is False, "bad-z-only: expected False (median rel=1.0)"

    # (c) both z's: 4 rel=0 from good-z + 4 rel=1 from bad-z,
    #     median of [0,0,0,0,1,1,1,1] = 0.5 > tol=0.25 -> False
    result_both = derivative_faithful_multiz(
        refit=refit, gp=gp, fid=fid, fid_value=fid[2],
        k_grid=k_grid, z_grid=np.array([z_good, z_bad]), param_idx=2, tol=0.25,
    )
    assert result_both is False, (
        "both z's: expected False — median(0×4 + 1×4)=0.5 exceeds tol=0.25"
    )


# ---------------------------------------------------------------------------
# Part B: build_refit_from_pareto_multiz_gated
# ---------------------------------------------------------------------------

def _write_artifacts(tmp_path, param="ns"):
    """Write a Pareto CSV + norm sidecar usable by the gated builder.

    Two equations:
      row 0: loss=0.01, "x0 + x1 + 0.1 * x3"   (best loss, x0-dependent -> Fisher-safe)
      row 1: loss=1.00, "0.5"                     (no x0 -> filtered out)
    """
    df = pd.DataFrame({
        "Complexity": [3, 1],
        "Loss": [0.01, 1.0],
        "Equation": ["x0 + x1 + 0.1 * x3", "0.5"],
    })
    csv = tmp_path / f"pareto_{param}.csv"
    df.to_csv(csv, index=False)

    z_grid = np.array([3.4, 3.6])
    k_grid = np.linspace(0.005, 0.04, 6)
    spec = MultiZNormalizationSpec(
        param_min=0.0, param_max=2.0,
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        z_grid=z_grid,
        mean_flux=np.outer(1.0 + 0.1 * z_grid, 1.0 + k_grid),
        std_flux=0.2 * np.ones((2, 6)),
        k_grid=k_grid,
    )
    norm_path = tmp_path / f"norm_{param}.npz"
    np.savez(
        norm_path,
        param_min=spec.param_min, param_max=spec.param_max,
        k_min=spec.k_min, k_max=spec.k_max,
        z_grid=spec.z_grid, mean_flux=spec.mean_flux,
        std_flux=spec.std_flux, k_grid=spec.k_grid,
        x_param_min=0.1, x_param_max=1.9,
        result_k_min=spec.k_min, result_k_max=spec.k_max,
    )
    return csv, norm_path, k_grid, z_grid


def _make_matching_gp(pareto_csv, norm_npz, k_grid, z_grid, fid, param_name="ns"):
    """Build a GP stub whose gradient exactly matches the first Fisher-safe equation.

    Uses the actual Refit1DResult.predict() to compute finite-difference
    gradients, then returns a GP that produces values consistent with those
    gradients (ratio ≡ 1, median deviation 0.0 <= tol).
    """
    from priya_forecast.multi_z.refit import build_refit_from_pareto_multiz
    from priya_forecast.derivative_gate import equation_param_gradient
    from priya_forecast.parameters import PARAM_NAMES, get_param

    refit = build_refit_from_pareto_multiz(
        param_name=param_name, z_min=3.4, z_max=3.6,
        pareto_csv=pareto_csv, norm_npz=norm_npz, pick_rule="best_loss",
    )
    meta = get_param(param_name)
    param_idx = list(PARAM_NAMES).index(param_name)

    # Compute what the equation's gradient is per (k, z); the GP stub will
    # return values that produce the same gradient (slope = equation slope).
    # Concretely: make the GP predict refit.predict(...) so the gradient
    # ratio is exactly 1 by construction.
    class _MatchingGP:
        def predict(self, theta, k, z):
            return refit.predict(
                theta_phys=float(np.asarray(theta)[param_idx]),
                k=np.asarray(k, dtype=float), z=float(z),
            )

    return _MatchingGP()


def test_gated_builder_returns_faithful_candidate(tmp_path):
    """A GP whose gradient matches the equation -> faithful -> builder returns it."""
    csv, norm_path, k_grid, z_grid = _write_artifacts(tmp_path)
    fid = _make_fid()
    gp = _make_matching_gp(csv, norm_path, k_grid, z_grid, fid)

    result = build_refit_from_pareto_multiz_gated(
        param_name="ns", z_min=3.4, z_max=3.6,
        pareto_csv=csv, norm_npz=norm_path,
        gp=gp, fid=fid, k_grid=k_grid, z_grid=z_grid,
        derivative_tol=0.25,
    )
    assert result.is_multiz
    assert "x0" in result.equation_str


def test_gated_builder_raises_when_none_pass(tmp_path):
    """When no candidate is derivative-faithful, ValueError is raised."""
    csv, norm_path, k_grid, z_grid = _write_artifacts(tmp_path)
    fid = _make_fid()

    # GP with zero gradient -> all bins masked -> derivative_faithful_multiz
    # returns False for every candidate.
    class _ZeroGP:
        def predict(self, theta, k, z):
            return np.zeros_like(np.asarray(k, dtype=float))

    with pytest.raises(ValueError, match="No derivative-faithful equation"):
        build_refit_from_pareto_multiz_gated(
            param_name="ns", z_min=3.4, z_max=3.6,
            pareto_csv=csv, norm_npz=norm_path,
            gp=_ZeroGP(), fid=fid, k_grid=k_grid, z_grid=z_grid,
            derivative_tol=0.25,
        )


def test_gated_builder_raises_when_no_fisher_safe(tmp_path):
    """When all Pareto rows are x0-free, ValueError is raised before gating."""
    df = pd.DataFrame({
        "Complexity": [1, 2],
        "Loss": [0.1, 0.01],
        "Equation": ["0.5", "x1 + x2"],   # neither uses x0
    })
    csv = tmp_path / "pareto_ns.csv"
    df.to_csv(csv, index=False)

    z_grid = np.array([3.4, 3.6])
    k_grid = np.linspace(0.005, 0.04, 6)
    spec = MultiZNormalizationSpec(
        param_min=0.0, param_max=2.0,
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        z_grid=z_grid,
        mean_flux=np.outer(1.0 + 0.1 * z_grid, 1.0 + k_grid),
        std_flux=0.2 * np.ones((2, 6)),
        k_grid=k_grid,
    )
    norm_path = tmp_path / "norm_ns.npz"
    np.savez(
        norm_path,
        param_min=spec.param_min, param_max=spec.param_max,
        k_min=spec.k_min, k_max=spec.k_max,
        z_grid=spec.z_grid, mean_flux=spec.mean_flux,
        std_flux=spec.std_flux, k_grid=spec.k_grid,
        x_param_min=0.1, x_param_max=1.9,
        result_k_min=spec.k_min, result_k_max=spec.k_max,
    )

    fid = _make_fid()
    gp = _StubGP(slope=1.0, param_idx=2)

    with pytest.raises(ValueError, match="No x0-dependent"):
        build_refit_from_pareto_multiz_gated(
            param_name="ns", z_min=3.4, z_max=3.6,
            pareto_csv=csv, norm_npz=norm_path,
            gp=gp, fid=fid, k_grid=k_grid, z_grid=z_grid,
        )
