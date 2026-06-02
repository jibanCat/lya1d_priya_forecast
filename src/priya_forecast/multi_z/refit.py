"""Multi-z PySR refit driver + Pareto-CSV reconstruction.

- refit_one_param_multi_z: run refit_1d_multiz_for_param with seed-retry,
  persist pareto_<param>.csv + norm_<param>.npz.
- build_refit_from_pareto_multiz: reload both into a 4-input Refit1DResult.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.models.pysr_model import load_pareto_csv, pick_equation
from priya_forecast.parameters import get_param
from priya_forecast.refit_1d_pysr import (
    HF_RESOLUTION, LF_RESOLUTION, Refit1DResult, refit_1d_multiz_for_param,
)
from priya_forecast.single_z.forecast import _filter_fisher_safe


def build_refit_from_pareto_multiz(
    *,
    param_name: str,
    z_min: float,
    z_max: float,
    pareto_csv,
    norm_npz,
    pick_rule: str,
) -> Refit1DResult:
    """Reconstruct a 4-input Refit1DResult from a Pareto CSV + norm sidecar."""
    df = load_pareto_csv(pareto_csv)
    # 4 inputs: x0=θ_norm, x1=k_norm, x2=resolution, x3=z_norm.
    safe = _filter_fisher_safe(df, n_features=4)
    if safe.empty:
        raise ValueError(
            f"No x0-dependent / Fisher-safe equation in Pareto front for "
            f"({param_name}, z∈[{z_min},{z_max}]): all {len(df)} rows unusable."
        )
    equation_str, complexity, loss = pick_equation(safe, pick_rule)
    norm = MultiZNormalizationSpec.load_npz(norm_npz)
    meta = get_param(param_name)
    z_center = float((z_min + z_max) / 2.0)
    return Refit1DResult(
        param_name=param_name, z=z_center, equation_str=equation_str,
        pareto_complexity=int(complexity), pareto_loss=float(loss),
        pareto_complexities=[int(c) for c in df["Complexity"]],
        pareto_losses=[float(x) for x in df["Loss"]],
        x_param_min=float(norm.param_min), x_param_max=float(norm.param_max),
        k_min=float(norm.k_min), k_max=float(norm.k_max),
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        fid_value=float(meta.fid), norm=norm,
        k_grid=np.asarray(norm.k_grid, dtype=float),
        wall_time_s=0.0,
        lf_train_mean_rel_err=0.0, hf_train_mean_rel_err=0.0,
        lf_train_max_rel_err=0.0, hf_train_max_rel_err=0.0,
        z_min=float(z_min), z_max=float(z_max),
    )


def _write_pareto_csv(result: Refit1DResult, csv_path: Path) -> None:
    """Write a load_pareto_csv-compatible CSV from a Refit1DResult's front.

    refit_1d_multiz_for_param returns only the *picked* equation string, not
    every row's equation. We persist the full complexity/loss front but fill
    the Equation only for the picked complexity; other rows carry an empty
    string. _filter_fisher_safe drops empty/x0-free rows, so the picked row
    survives and best_loss reconstructs it.
    """
    import pandas as pd
    pd.DataFrame({
        "Complexity": result.pareto_complexities,
        "Loss": result.pareto_losses,
        "Equation": [result.equation_str if c == result.pareto_complexity
                     else "" for c in result.pareto_complexities],
    }).to_csv(csv_path, index=False)


def refit_one_param_multi_z(
    *,
    param_name: str,
    z_min: float,
    z_max: float,
    cfg,
    gp_lf,
    gp_hf,
    k_grid: np.ndarray,
    out_dir: str | Path,
    n_total: int = 225,
    max_retries: int = 4,
) -> Refit1DResult:
    """Refit one parameter over [z_min, z_max]; write CSV + norm sidecar.

    Retries with bumped seeds until the front has an x0-dependent,
    Fisher-safe equation, or retries are exhausted.
    """
    from priya_forecast.single_z.refit import pysr_kwargs_for_cfg

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / f"pareto_{param_name}.csv"
    norm_path = out_dir / f"norm_{param_name}.npz"
    pysr_kwargs = pysr_kwargs_for_cfg(cfg)
    k_grid = np.asarray(k_grid, dtype=float)

    result = None
    for attempt in range(max_retries + 1):
        result = refit_1d_multiz_for_param(
            param_name=param_name, z_min=z_min, z_max=z_max, k_grid=k_grid,
            gp_lf=gp_lf, gp_hf=gp_hf, n_total=n_total,
            pysr_kwargs=pysr_kwargs, seed=cfg.pysr.seed + attempt,
        )
        _write_pareto_csv(result, csv_path)
        result.norm.save_npz(norm_path)
        safe = _filter_fisher_safe(load_pareto_csv(csv_path), n_features=4)
        if not safe.empty:
            return result
    return result
