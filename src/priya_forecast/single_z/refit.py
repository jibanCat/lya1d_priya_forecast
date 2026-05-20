"""refit_and_forecast mode: single-z PySR refit per parameter.

Thin wrapper over `refit_1d_pysr.refit_1d_for_param` (inline 1pvar path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.refit_1d_pysr import (
    DEFAULT_PYSR_KWARGS,
    SMART_REFIT_PYSR_KWARGS,
    refit_1d_for_param,
)
from priya_forecast.single_z.config import PipelineConfig


def kodiaq_k_grid(kmin: float, kmax: float, nk: int = 48) -> np.ndarray:
    """Log-spaced k-grid (s/km) — the grid the regen + refit share."""
    return np.geomspace(kmin, kmax, nk)


def pysr_kwargs_for_cfg(cfg: PipelineConfig) -> dict:
    """Assemble the PySR kwargs dict from `cfg.pysr`.

    `smart_kwargs` selects SMART (restricted operators + ANOVA loss) vs the
    default operator set; the search-budget fields are taken from `cfg.pysr`.
    """
    base = dict(
        SMART_REFIT_PYSR_KWARGS if cfg.pysr.smart_kwargs else DEFAULT_PYSR_KWARGS
    )
    base["niterations"] = cfg.pysr.niterations
    base["maxsize"] = cfg.pysr.maxsize
    base["populations"] = cfg.pysr.populations
    base["procs"] = cfg.pysr.procs
    return base


def refit_one_param_single_z(
    *,
    param_name: str,
    z: float,
    cfg: PipelineConfig,
    gp_lf,
    gp_hf,
    k_grid: np.ndarray,
    out_dir: str | Path,
    max_retries: int = 4,
):
    """Refit one parameter at one z-bin; write `pareto_{param}.csv`.

    Retries with bumped seeds (cfg.pysr.seed + attempt) until the Pareto
    front contains at least one x0-dependent, Fisher-safe equation, or
    `max_retries` extra attempts are exhausted. Returns the `Refit1DResult`
    of the first attempt that yields a usable front, else the last attempt.
    """
    from priya_forecast.models.pysr_model import load_pareto_csv
    from priya_forecast.single_z.forecast import _filter_fisher_safe

    out_dir = Path(out_dir)
    pareto_csv = out_dir / f"pareto_{param_name}.csv"
    pysr_kwargs = pysr_kwargs_for_cfg(cfg)
    k_grid = np.asarray(k_grid, dtype=float)

    result = None
    for attempt in range(max_retries + 1):
        result = refit_1d_for_param(
            param_name=param_name,
            z=z,
            k_grid=k_grid,
            gp_lf=gp_lf,
            gp_hf=gp_hf,
            pysr_kwargs=pysr_kwargs,
            seed=cfg.pysr.seed + attempt,
            pareto_csv_out=pareto_csv,
            log_space=(cfg.target_space == "log"),
        )
        # PySR equations have 3 inputs (x0=θ_norm, x1=k_norm, x2=resolution).
        safe = _filter_fisher_safe(load_pareto_csv(pareto_csv), n_features=3)
        if not safe.empty:
            return result
    return result
