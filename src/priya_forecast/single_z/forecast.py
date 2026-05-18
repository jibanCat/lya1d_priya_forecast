"""forecast_only mode: Pareto CSVs → equations → combined model → Fisher.

This module reconstructs a `Refit1DResult` per parameter from a picked Pareto
equation plus the regenerated 1pvar training data, builds the combined model,
and runs the three Fisher forecasts (σ_GP, σ_perfect_1D, σ_PySR).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.models.normalization import NormalizationSpec
from priya_forecast.models.pysr_model import load_pareto_csv, pick_equation
from priya_forecast.pareto_filters import (
    has_pathological_constant,
    is_fisher_stencil_safe,
)
from priya_forecast.parameters import get_param
from priya_forecast.refit_1d_pysr import HF_RESOLUTION, LF_RESOLUTION, Refit1DResult
from priya_forecast.single_z.training_data import load_1pvar


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


def _filter_fisher_safe(df, n_features: int):
    """Drop Fisher-pathological Pareto rows; return the surviving sub-frame.

    Mirrors `scripts/refit_one_param.py`: an equation is kept only if it has
    no pathological constant and is Fisher-stencil-safe — the two guards that
    protect Fisher conditioning.
    """
    eq = df["Equation"].astype(str)
    pathological = eq.apply(has_pathological_constant)
    stencil_safe = eq.apply(
        lambda s: is_fisher_stencil_safe(s, n_features=n_features)
    )
    return df[(~pathological) & stencil_safe].reset_index(drop=True)


def build_refit_from_pareto(
    *,
    param_name: str,
    z: float,
    pareto_csv,
    pick_rule: str,
    data_1pvar_dir,
) -> Refit1DResult:
    """Reconstruct a `Refit1DResult` from a Pareto CSV + regenerated 1pvar data.

    Filter-then-pick: drop Fisher-pathological rows, then apply `pick_rule`.
    The per-parameter normalization comes from `per_param_local_norm` on the
    regenerated LF flux — identical to what Stage C trains with.
    """
    df = load_pareto_csv(pareto_csv)
    # PySR equations here have 3 inputs (x0=θ_norm, x1=k_norm, x2=resolution).
    safe = _filter_fisher_safe(df, n_features=3)
    if safe.empty:
        raise ValueError(
            f"No Fisher-safe equation in Pareto front for ({param_name}, z={z}): "
            f"all {len(df)} rows were pathological or stencil-unsafe."
        )
    equation_str, complexity, loss = pick_equation(safe, pick_rule)

    d = load_1pvar(param_name=param_name, z=z, data_dir=data_1pvar_dir)
    k_grid = d["kfkms_lf_z"][0]
    meta = get_param(param_name)
    norm = per_param_local_norm(
        flux_lf_z=d["flux_lf_z"], k_grid=k_grid,
        param_min=float(meta.prior[0]), param_max=float(meta.prior[1]),
    )
    return Refit1DResult(
        param_name=param_name,
        z=float(z),
        equation_str=equation_str,
        pareto_complexity=int(complexity),
        pareto_loss=float(loss),
        pareto_complexities=[int(c) for c in df["Complexity"]],
        pareto_losses=[float(x) for x in df["Loss"]],
        x_param_min=float(meta.prior[0]),
        x_param_max=float(meta.prior[1]),
        k_min=float(k_grid.min()),
        k_max=float(k_grid.max()),
        lf_resolution=LF_RESOLUTION,
        hf_resolution=HF_RESOLUTION,
        fid_value=float(meta.fid),
        norm=norm,
        k_grid=np.asarray(k_grid, dtype=float),
        wall_time_s=0.0,
        lf_train_mean_rel_err=0.0,
        hf_train_mean_rel_err=0.0,
        lf_train_max_rel_err=0.0,
        hf_train_max_rel_err=0.0,
    )
