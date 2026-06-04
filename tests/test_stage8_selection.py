"""Stage 8 Task 6 — derivative-validation gate in single-z equation selection.

Tests `build_refit_from_pareto_gated`: given two Fisher-safe equations where
the LOWER-loss one has an unfaithful gradient and the HIGHER-loss one has a
faithful gradient, the gated builder must skip the lower-loss equation and
return the faithful higher-loss one.

Construction
------------
Parameter: ``ns``   (fid=0.983, prior=(0.8, 1.05), width=0.25)

Equations (both use x0 and x1, no pathological constants):
  - "2*x0 + x1"  — loss=0.01  (LOWER loss, unfaithful)
  - "5*x0 + x1"  — loss=0.05  (HIGHER loss, faithful)

Gradient derivation (identity-like norm: std_flux≈1, mean_flux≈0):
  predict(θ) ≈ (c * x0_norm + x1_norm) * std_k + mean_k
  ∂predict/∂θ ≈ c / (param_max - param_min) * std_k   [per k-bin]

  For c=2: ∂predict/∂θ ∝ 2/0.25 * std = 8 * std
  For c=5: ∂predict/∂θ ∝ 5/0.25 * std = 20 * std

Setting gp_target_grad = 20 * std_k (matching c=5):
  ratio for c=2: 8/20 = 0.40  → |ratio-1| = 0.60 > tol=0.25  → REJECT
  ratio for c=5: 20/20 = 1.00 → |ratio-1| = 0.00 ≤ tol=0.25  → ACCEPT

The std_k cancels in the ratio check because both equations have the same
x1 contribution which is θ-independent.  The ratio is purely determined by
the x0 coefficient, so the test is robust to the exact flux_lf_z values as
long as std_flux > 0 everywhere.
"""
from __future__ import annotations

import io
import numpy as np
import pandas as pd
import pytest

from priya_forecast.parameters import get_param
from priya_forecast.single_z.forecast import build_refit_from_pareto_gated
from priya_forecast.refit_1d_pysr import HF_RESOLUTION


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

N_K = 10
K_GRID = np.linspace(0.001, 0.04, N_K)

# Flux array with non-trivial std (rows alternate +/-1 around a mean of 2.0
# so std ≈ 1.0 and mean ≈ 2.0; all values strictly positive).
N_POINTS = 20
_base = np.ones((N_POINTS, N_K), dtype=float) * 2.0
_base[::2] += 1.0     # every other row +1
_base[1::2] -= 1.0    # every other row -1
FLUX_LF_Z = _base     # shape (20, 10); std_flux ≈ 1.0, mean_flux ≈ 2.0

# kfkms_lf_z: same k-grid replicated per point
KFKMS = np.broadcast_to(K_GRID, (N_POINTS, N_K)).copy()

FAKE_1PVAR = {
    "flux_lf_z": FLUX_LF_Z,
    "kfkms_lf_z": KFKMS,
}

# Two rows: lower-loss unfaithful (c=2) and higher-loss faithful (c=5).
PARETO_DF = pd.DataFrame({
    "Complexity": [3, 3],
    "Loss":       [0.01, 0.05],   # lower-loss first
    "Equation":   ["2*x0 + x1", "5*x0 + x1"],
})


def _make_pareto_csv(tmp_path) -> str:
    """Write the two-row Pareto CSV to a temp file; return its path."""
    p = tmp_path / "pareto_ns.csv"
    PARETO_DF.to_csv(p, index=False)
    return str(p)


def _make_gp_target_grad() -> np.ndarray:
    """Construct a synthetic GP target gradient matching 'c=5' equation.

    Uses the same norm that `build_refit_from_pareto_gated` will compute from
    FAKE_1PVAR.  The norm's std_flux is FLUX_LF_Z.std(axis=0, ddof=0) ≈ 1.0.

    The analytic gradient for '5*x0 + x1' at any theta is:
        ∂predict/∂θ = 5 / (param_max - param_min) * std_flux
                    = 5 / 0.25 * std_flux
                    = 20 * std_flux
    """
    meta = get_param("ns")
    width = float(meta.prior[1]) - float(meta.prior[0])  # 0.25
    std_flux = FLUX_LF_Z.std(axis=0, ddof=0)
    # Gradient for the '5*x0 + x1' equation
    return (5.0 / width) * std_flux  # shape (N_K,)


# ---------------------------------------------------------------------------
# Main test: gate overrides best_loss
# ---------------------------------------------------------------------------

def test_gate_selects_faithful_over_best_loss(tmp_path, monkeypatch):
    """The gated builder must skip the lower-loss equation (unfaithful gradient)
    and return the higher-loss equation whose gradient matches the GP target.
    """
    # Monkeypatch load_1pvar inside forecast.py so no HDF5 files are needed.
    monkeypatch.setattr(
        "priya_forecast.single_z.forecast.load_1pvar",
        lambda *, param_name, z, data_dir: FAKE_1PVAR,
    )

    pareto_csv = _make_pareto_csv(tmp_path)
    gp_target_grad = _make_gp_target_grad()

    refit = build_refit_from_pareto_gated(
        param_name="ns",
        z=3.6,
        pareto_csv=pareto_csv,
        data_1pvar_dir=tmp_path,  # unused due to monkeypatch
        gp_target_grad=gp_target_grad,
        derivative_tol=0.25,
    )

    # The gate must have chosen '5*x0 + x1', NOT the lower-loss '2*x0 + x1'.
    assert refit.equation_str == "5*x0 + x1", (
        f"Expected '5*x0 + x1' (faithful), got {refit.equation_str!r}. "
        "Gate did not override best_loss selection."
    )
    assert refit.param_name == "ns"
    assert refit.z == 3.6

    # Sanity: the returned refit evaluates without error.
    pred = refit.predict(theta_phys=0.983, k=K_GRID)
    assert pred.shape == (N_K,)
    assert np.all(np.isfinite(pred))


# ---------------------------------------------------------------------------
# Additional: all-unfaithful raises ValueError (triggers GP fallback)
# ---------------------------------------------------------------------------

def test_gate_raises_when_no_faithful_equation(tmp_path, monkeypatch):
    """If no equation passes the gate, ValueError is raised (caller falls back)."""
    monkeypatch.setattr(
        "priya_forecast.single_z.forecast.load_1pvar",
        lambda *, param_name, z, data_dir: FAKE_1PVAR,
    )

    pareto_csv = _make_pareto_csv(tmp_path)
    # Set target_grad to something totally unrelated (zeroed out → always fails).
    # derivative_faithful returns False when target is all-zero (amax==0).
    zero_target = np.zeros(N_K)

    with pytest.raises(ValueError, match="No derivative-faithful equation"):
        build_refit_from_pareto_gated(
            param_name="ns",
            z=3.6,
            pareto_csv=pareto_csv,
            data_1pvar_dir=tmp_path,
            gp_target_grad=zero_target,
            derivative_tol=0.25,
        )


# ---------------------------------------------------------------------------
# Config: derivative_tol field and validation
# ---------------------------------------------------------------------------

def test_pipeline_config_derivative_tol_default():
    """PipelineConfig.derivative_tol defaults to 0.25."""
    from priya_forecast.single_z.config import PipelineConfig
    cfg = PipelineConfig()
    assert cfg.derivative_tol == 0.25


def test_pipeline_config_derivative_tol_validate_rejects_nonpositive():
    """derivative_tol <= 0 raises ValueError in validate()."""
    from priya_forecast.single_z.config import PipelineConfig
    cfg = PipelineConfig(derivative_tol=0.0)
    with pytest.raises(ValueError, match="derivative_tol must be > 0"):
        cfg.validate()
