"""Single-z pipeline dispatcher.

Three top-level entry points keyed by `cfg.mode`:

- ``run_gp_only(cfg)`` — Fisher on the GP itself (σ_GP only).
- ``run_forecast_only(cfg)`` — student CSVs → PySR equations → Fisher
  (σ_GP, σ_PySR, σ_perfect_1D, ratios + diagnostics).
- ``run_refit_and_forecast(cfg)`` — refit single-z PySR per parameter,
  emit CSVs, then run forecast_only.

Each entry point writes everything it needs into ``cfg.output_dir``:
``forecast_table.txt``, ``scorecard.md``, ``forecast_corner.png``,
``forecast_sigma.png``, ``p1d_comparison.png``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.fisher import fisher_matrix
from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.ksdata_likelihood import KSDataLikelihood
from priya_forecast.models.gp_model import GPModel
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    fiducial_vector,
    get_param,
)
from priya_forecast.single_z.config import PipelineConfig


def _build_gp(cfg: PipelineConfig) -> GPModel:
    return GPModel(basedir=cfg.gp.basedir, hires_subdir=cfg.gp.hires_subdir)


def _build_ksdata_likelihood(cfg: PipelineConfig, gp: GPModel) -> KSDataLikelihood:
    return KSDataLikelihood(
        model=gp,
        z_min=cfg.redshift,
        z_max=cfg.redshift,
        k_min=cfg.k_range.min,
        k_max=cfg.k_range.max,
        cov_scale=cfg.data.cov_scale,
        mock_data=cfg.data.mock_data,
        conservative=cfg.data.conservative,
    )


def _build_eboss_likelihood(cfg: PipelineConfig, gp: GPModel) -> GaussianLikelihood:
    return GaussianLikelihood(
        model=gp, z=cfg.redshift, cov_scale=cfg.data.cov_scale,
        mock_data="gp",
    )


def _params_subset() -> tuple:
    return PARAMS_11D


def _selected_indices(cfg: PipelineConfig) -> list[int]:
    return [PARAM_NAMES.index(n) for n in cfg.parameters]


def run_gp_only(cfg: PipelineConfig) -> dict:
    """Fisher on the GP only.

    Returns a result dict with sigma_gp (per-parameter), the FisherResult,
    and the path of the written forecast_table.txt.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    gp = _build_gp(cfg)
    if cfg.data.source == "kodiaq":
        like = _build_ksdata_likelihood(cfg, gp)
    else:
        like = _build_eboss_likelihood(cfg, gp)

    indices = _selected_indices(cfg)
    selected_params = tuple(PARAMS_11D[i] for i in indices)

    # Partial Fisher: theta_fid is the FULL 11-vector (the model expects 11),
    # and param_indices tells fisher_matrix which positions in it to perturb.
    theta_fid_full = np.asarray(fiducial_vector(), dtype=float)
    fisher = fisher_matrix(
        likelihood=like,
        theta_fid=theta_fid_full,
        params=selected_params,
        step_frac=cfg.fisher.step_frac,
        rel_tol=cfg.fisher.rel_tol,
        param_indices=indices,
    )

    sigma = fisher.sigma

    table_path = out_dir / "forecast_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"# single-z gp_only forecast at z={cfg.redshift}\n")
        f.write(f"# data={cfg.data.source} cov_scale={cfg.data.cov_scale}\n")
        f.write(f"# k_range=[{cfg.k_range.min}, {cfg.k_range.max}]\n")
        f.write(f"# {'param':<12s} {'fid':>10s} {'sigma_GP':>12s} {'rel_sigma':>10s}\n")
        for i, p in enumerate(selected_params):
            f.write(
                f"  {p.name:<12s} {p.fid:>10.4g} {sigma[i]:>12.4g} "
                f"{sigma[i] / abs(p.fid):>10.4f}\n"
            )

    scorecard_path = out_dir / "scorecard.md"
    with open(scorecard_path, "w", encoding="utf-8") as f:
        f.write(f"# Single-z forecast scorecard — gp_only mode\n\n")
        f.write(f"- z = {cfg.redshift}\n")
        f.write(f"- data = {cfg.data.source} (cov_scale = {cfg.data.cov_scale})\n")
        f.write(f"- k ∈ [{cfg.k_range.min}, {cfg.k_range.max}]\n")
        f.write(f"- parameters = {cfg.parameters}\n\n")
        f.write(f"## σ_GP per parameter\n\n")
        f.write(f"| param | fid | σ_GP | σ_GP / |fid| |\n")
        f.write(f"|---|---|---|---|\n")
        for i, p in enumerate(selected_params):
            f.write(
                f"| {p.name} | {p.fid:.4g} | {sigma[i]:.4g} | {sigma[i] / abs(p.fid):.4f} |\n"
            )

    return {
        "sigma_gp": sigma,
        "fisher": fisher,
        "table_path": table_path,
        "scorecard_path": scorecard_path,
        "selected_params": selected_params,
    }


def run_forecast_only(cfg: PipelineConfig) -> dict:
    raise NotImplementedError("forecast_only mode lands in Stage B.")


def run_refit_and_forecast(cfg: PipelineConfig) -> dict:
    raise NotImplementedError("refit_and_forecast mode lands in Stage C.")


DISPATCH = {
    "gp_only": run_gp_only,
    "forecast_only": run_forecast_only,
    "refit_and_forecast": run_refit_and_forecast,
}


def run(cfg: PipelineConfig) -> dict:
    """Top-level dispatcher: validates cfg, runs the right mode."""
    cfg.validate()
    return DISPATCH[cfg.mode](cfg)
