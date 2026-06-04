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
from priya_forecast.single_z import forecast as _fc
from priya_forecast.single_z import refit as _refit
from priya_forecast.diagnostics.forecast_plots import plot_fisher_corner


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


def _write_forecast_deliverables(cfg, out_dir, results, *, pysr_available,
                                 dropped: list | None = None):
    """Write corner.png + forecast_table.txt + scorecard.md from the 3 Fisher
    results. Returns {table_path, scorecard_path, corner_path}.

    Parameters
    ----------
    dropped : list of parameter names that had no usable PySR equation and
        fell back to the GP slice.  Written into the scorecard when non-empty.
    """
    if dropped is None:
        dropped = []
    sigmas = {label: fr.sigma for label, fr in results.items()}
    corner_labels = ["GP", "perfect_1D"] + (["PySR"] if pysr_available else [])
    corner_path = out_dir / "corner.png"
    plot_fisher_corner(
        fisher_results={lab: results[lab] for lab in corner_labels},
        outpath=corner_path,
        param_subset=cfg.parameters[: min(5, len(cfg.parameters))],
    )

    table_path = out_dir / "forecast_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"# single-z forecast_only at z={cfg.redshift}\n")
        f.write(f"# combine={cfg.combine}  pysr_equations="
                f"{'yes' if pysr_available else 'NONE'}\n")
        f.write(f"# {'param':<12s} {'sigma_GP':>12s} {'sigma_perf1D':>14s} "
                f"{'sigma_PySR':>12s}\n")
        for i, name in enumerate(cfg.parameters):
            sp = (f"{sigmas['PySR'][i]:>12.4g}" if pysr_available
                  else f"{'n/a':>12s}")
            f.write(f"  {name:<12s} {sigmas['GP'][i]:>12.4g} "
                    f"{sigmas['perfect_1D'][i]:>14.4g} {sp}\n")

    scorecard_path = out_dir / "scorecard.md"
    with open(scorecard_path, "w", encoding="utf-8") as f:
        f.write("# Single-z forecast scorecard — forecast_only\n\n")
        f.write(f"- z = {cfg.redshift}\n")
        f.write(f"- combine = {cfg.combine}\n")
        f.write(f"- PySR equations: "
                f"{'available' if pysr_available else 'NOT available — σ_PySR omitted'}\n\n")
        f.write("## σ per parameter\n\n")
        f.write("| param | σ_GP | σ_perfect_1D | σ_PySR |\n|---|---|---|---|\n")
        for i, name in enumerate(cfg.parameters):
            sp = f"{sigmas['PySR'][i]:.4g}" if pysr_available else "n/a"
            f.write(f"| {name} | {sigmas['GP'][i]:.4g} | "
                    f"{sigmas['perfect_1D'][i]:.4g} | {sp} |\n")
        if dropped:
            f.write(
                f"\n**Parameters with no usable PySR equation (GP-slice fallback):** "
                f"{', '.join(dropped)}\n"
            )

    fisher_npz = {}
    for label, fr in results.items():
        npz_path = out_dir / f"fisher_{label}.npz"
        fr.save_npz(npz_path)
        fisher_npz[label] = npz_path

    return {
        "table_path": table_path,
        "scorecard_path": scorecard_path,
        "corner_path": corner_path,
        "fisher_npz": fisher_npz,
    }


def run_forecast_only(cfg: PipelineConfig) -> dict:
    """Student CSVs → equations → combined model → σ_GP / σ_perfect_1D / σ_PySR.

    σ_GP and σ_perfect_1D need no equations. σ_PySR needs Pareto CSVs; if none
    are available the run still emits σ_GP and σ_perfect_1D and notes σ_PySR
    as unavailable in the scorecard.
    """
    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    gp = _build_gp(cfg)
    fid = np.asarray(fiducial_vector(), dtype=float)

    # Reconstruct per-parameter refits from Pareto CSVs if available.
    refits: dict = {name: None for name in PARAM_NAMES}
    dropped: list[str] = []
    pysr_available = False
    try:
        csv_paths = _fc.resolve_pareto_csvs(cfg)
    except FileNotFoundError:
        csv_paths = {}
    if csv_paths:
        from priya_forecast.derivative_gate import gp_param_gradient
        k_refit = _refit.kodiaq_k_grid(cfg.k_range.min, cfg.k_range.max, 48)
    for param, csv in csv_paths.items():
        try:
            tgt = gp_param_gradient(
                gp=gp, fid=fid, k_grid=k_refit, z=cfg.redshift,
                param_idx=PARAM_NAMES.index(param),
            )
            refits[param] = _fc.build_refit_from_pareto_gated(
                param_name=param, z=cfg.redshift, pareto_csv=csv,
                pick_rule=cfg.pick, data_1pvar_dir="data/single_z_1pvar",
                gp_target_grad=tgt, derivative_tol=cfg.derivative_tol,
                log_space=(cfg.target_space == "log"),
            )
            pysr_available = True
        except ValueError as exc:
            dropped.append(param)
            print(f"[forecast_only] {param}: {exc} — falling back to GP slice.")

    results = _fc.run_three_fisher(cfg=cfg, gp=gp, fid=fid, refits=refits)

    deliverables = _write_forecast_deliverables(
        cfg, out_dir, results, pysr_available=pysr_available, dropped=dropped,
    )
    return {
        "sigmas": {label: fr.sigma for label, fr in results.items()},
        "fisher_results": results,
        "pysr_available": pysr_available,
        **deliverables,
    }


def run_refit_and_forecast(cfg: PipelineConfig) -> dict:
    """Refit single-z PySR per parameter, emit Pareto CSVs, then forecast.

    Loops `refit_one_param_single_z` over `cfg.parameters` in-process, then
    runs the three Fisher forecasts on the fresh refits.
    """
    out_dir = Path(cfg.output_dir)
    refit_dir = out_dir / "refit" / f"z{cfg.redshift}"
    refit_dir.mkdir(parents=True, exist_ok=True)
    fid = np.asarray(fiducial_vector(), dtype=float)

    k_grid = _refit.kodiaq_k_grid(cfg.k_range.min, cfg.k_range.max, 48)
    gp_lf = GPModel(basedir=cfg.gp.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=cfg.gp.basedir, fidelity="hf", kf=k_grid)

    from priya_forecast.derivative_gate import gp_param_gradient

    k_refit = _refit.kodiaq_k_grid(cfg.k_range.min, cfg.k_range.max, 48)
    refits: dict = {name: None for name in PARAM_NAMES}
    dropped: list[str] = []
    for param in cfg.parameters:
        _refit.refit_one_param_single_z(
            param_name=param, z=cfg.redshift, cfg=cfg,
            gp_lf=gp_lf, gp_hf=gp_hf, k_grid=k_grid, out_dir=refit_dir,
        )
        csv = refit_dir / f"pareto_{param}.csv"
        try:
            tgt = gp_param_gradient(
                gp=gp_hf, fid=fid, k_grid=k_refit, z=cfg.redshift,
                param_idx=PARAM_NAMES.index(param),
            )
            refits[param] = _fc.build_refit_from_pareto_gated(
                param_name=param, z=cfg.redshift, pareto_csv=csv,
                pick_rule=cfg.pick, data_1pvar_dir="data/single_z_1pvar",
                gp_target_grad=tgt, derivative_tol=cfg.derivative_tol,
                log_space=(cfg.target_space == "log"),
            )
        except ValueError as exc:
            dropped.append(param)
            print(f"[refit] {param}: {exc} — GP-slice fallback.")

    pysr_available = bool(cfg.parameters) and len(dropped) < len(cfg.parameters)
    results = _fc.run_three_fisher(cfg=cfg, gp=gp_hf, fid=fid, refits=refits)
    deliverables = _write_forecast_deliverables(
        cfg, out_dir, results, pysr_available=pysr_available, dropped=dropped,
    )
    return {
        "sigmas": {label: fr.sigma for label, fr in results.items()},
        "fisher_results": results,
        "pysr_available": pysr_available,
        "refit_dir": refit_dir,
        **deliverables,
    }


DISPATCH = {
    "gp_only": run_gp_only,
    "forecast_only": run_forecast_only,
    "refit_and_forecast": run_refit_and_forecast,
}


def run(cfg: PipelineConfig) -> dict:
    """Top-level dispatcher: validates cfg, runs the right mode."""
    cfg.validate()
    return DISPATCH[cfg.mode](cfg)
