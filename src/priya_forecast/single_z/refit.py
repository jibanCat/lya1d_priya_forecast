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
):
    """Refit one parameter at one z-bin; write `pareto_{param}.csv`.

    Returns the `Refit1DResult`. `out_dir` is `<output_dir>/refit/z{z}/`.
    """
    out_dir = Path(out_dir)
    return refit_1d_for_param(
        param_name=param_name,
        z=z,
        k_grid=np.asarray(k_grid, dtype=float),
        gp_lf=gp_lf,
        gp_hf=gp_hf,
        pysr_kwargs=pysr_kwargs_for_cfg(cfg),
        seed=cfg.pysr.seed,
        pareto_csv_out=out_dir / f"pareto_{param_name}.csv",
    )
