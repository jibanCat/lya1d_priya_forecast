"""forecast_only mode: Pareto CSVs → equations → combined model → Fisher.

This module reconstructs a `Refit1DResult` per parameter from a picked Pareto
equation plus the regenerated 1pvar training data, builds the combined model,
and runs the three Fisher forecasts (σ_GP, σ_perfect_1D, σ_PySR).
"""

from __future__ import annotations

import numpy as np

from priya_forecast.models.normalization import NormalizationSpec


def per_param_local_norm(
    *,
    flux_lf_z: np.ndarray,
    k_grid: np.ndarray,
    param_min: float,
    param_max: float,
) -> NormalizationSpec:
    """Per-parameter local normalization from a 1pvar LF flux sweep.

    The `local_anchored` combine normalizes each per-param equation with the
    per-k mean/std of that parameter's own LF training flux. Computing it here
    — from the same regenerated 1pvar data Stage C trains on — guarantees the
    forecast-time norm matches the train-time norm.

    Parameters
    ----------
    flux_lf_z : (n_points, n_k) — LF P_F sweep at one z-bin.
    k_grid : (n_k,) — strictly increasing k-grid.
    param_min, param_max : the parameter's prior bounds.
    """
    flux_lf_z = np.asarray(flux_lf_z, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    mean_flux = flux_lf_z.mean(axis=0)
    std_flux = flux_lf_z.std(axis=0, ddof=0)
    std_flux = np.where(std_flux > 0, std_flux, 1.0)
    return NormalizationSpec(
        param_min=float(param_min),
        param_max=float(param_max),
        k_min=float(k_grid.min()),
        k_max=float(k_grid.max()),
        mean_flux=mean_flux,
        std_flux=std_flux,
        k_grid=k_grid,
    )
