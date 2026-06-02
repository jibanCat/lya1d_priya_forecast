"""Multi-z pipeline dispatcher (Stage 7). Mirrors single_z/pipeline.py."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.fisher import fisher_matrix
from priya_forecast.ksdata_likelihood import KSDataLikelihood
from priya_forecast.models.gp_model import GPModel
from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D, fiducial_vector
from priya_forecast.diagnostics.forecast_plots import plot_fisher_corner
from priya_forecast.multi_z.config import MultiZPipelineConfig
from priya_forecast.multi_z import forecast as _fc
from priya_forecast.multi_z import refit as _refit
from priya_forecast.single_z.forecast import equation_uses_param
from priya_forecast.single_z.refit import kodiaq_k_grid


def _build_gp(cfg):
    return GPModel(basedir=cfg.gp.basedir, hires_subdir=cfg.gp.hires_subdir)


def _selected_indices(cfg):
    return [PARAM_NAMES.index(n) for n in cfg.parameters]


def run_gp_only_multiz(cfg: MultiZPipelineConfig) -> dict:
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gp = _build_gp(cfg)
    like = KSDataLikelihood(
        model=gp, z_min=cfg.z_min, z_max=cfg.z_max,
        k_min=cfg.k_range.min, k_max=cfg.k_range.max,
        cov_scale=cfg.data.cov_scale, mock_data=cfg.data.mock_data,
        conservative=cfg.data.conservative,
    )
    indices = _selected_indices(cfg)
    selected = tuple(PARAMS_11D[i] for i in indices)
    theta_fid_full = np.asarray(fiducial_vector(), dtype=float)
    fisher = fisher_matrix(
        likelihood=like, theta_fid=theta_fid_full, params=selected,
        step_frac=cfg.fisher.step_frac, rel_tol=cfg.fisher.rel_tol,
        param_indices=indices,
    )
    sigma = fisher.sigma
    table_path = out_dir / "forecast_table.txt"
    with open(table_path, "w", encoding="utf-8") as f:
        f.write(f"# multi-z gp_only forecast z in [{cfg.z_min},{cfg.z_max}]\n")
        f.write(f"# data={cfg.data.source} cov_scale={cfg.data.cov_scale}\n")
        f.write(f"# {'param':<12s} {'fid':>10s} {'sigma_GP':>12s} {'rel':>10s}\n")
        for i, p in enumerate(selected):
            f.write(f"  {p.name:<12s} {p.fid:>10.4g} {sigma[i]:>12.4g} "
                    f"{sigma[i]/abs(p.fid):>10.4f}\n")
    return {"sigma_gp": sigma, "fisher": fisher, "table_path": table_path,
            "selected_params": selected}


def _write_forecast_deliverables_multiz(cfg, out_dir, results, *,
                                        pysr_available, dropped=None):
    """Mirror single_z/pipeline.py:_write_forecast_deliverables, with multi-z headers."""
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
        f.write(f"# multi-z forecast_only z in [{cfg.z_min},{cfg.z_max}]\n")
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
        f.write("# Multi-z forecast scorecard — forecast_only\n\n")
        f.write(f"- z in [{cfg.z_min},{cfg.z_max}]\n")
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


def run_forecast_only_multiz(cfg: MultiZPipelineConfig) -> dict:
    out_dir = Path(cfg.output_dir); out_dir.mkdir(parents=True, exist_ok=True)
    gp = _build_gp(cfg)
    fid = np.asarray(fiducial_vector(), dtype=float)
    refits, dropped = _fc.load_refits(cfg)
    pysr_available = any(v is not None for v in refits.values())
    results = _fc.run_three_fisher_multiz(cfg=cfg, gp=gp, fid=fid, refits=refits)
    deliverables = _write_forecast_deliverables_multiz(
        cfg, out_dir, results, pysr_available=pysr_available, dropped=dropped)
    return {"sigmas": {k: fr.sigma for k, fr in results.items()},
            "fisher_results": results, "pysr_available": pysr_available,
            **deliverables}


def run_refit_and_forecast_multiz(cfg: MultiZPipelineConfig) -> dict:
    out_dir = Path(cfg.output_dir)
    refit_dir = out_dir / "refit" / f"z{cfg.z_min}-{cfg.z_max}"
    refit_dir.mkdir(parents=True, exist_ok=True)
    fid = np.asarray(fiducial_vector(), dtype=float)
    k_grid = kodiaq_k_grid(cfg.k_range.min, cfg.k_range.max, 48)
    gp_lf = GPModel(basedir=cfg.gp.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=cfg.gp.basedir, fidelity="hf", kf=k_grid)
    refits = {n: None for n in PARAM_NAMES}
    dropped = []
    for param in cfg.parameters:
        result = _refit.refit_one_param_multi_z(
            param_name=param, z_min=cfg.z_min, z_max=cfg.z_max, cfg=cfg,
            gp_lf=gp_lf, gp_hf=gp_hf, k_grid=k_grid, out_dir=refit_dir,
        )
        if equation_uses_param(result.equation_str):
            refits[param] = result
        else:
            dropped.append(param)
            print(f"[multi_z refit] {param}: no x0 dependence — GP-slice fallback.")
    pysr_available = bool(cfg.parameters) and len(dropped) < len(cfg.parameters)
    results = _fc.run_three_fisher_multiz(cfg=cfg, gp=gp_hf, fid=fid, refits=refits)
    deliverables = _write_forecast_deliverables_multiz(
        cfg, out_dir, results, pysr_available=pysr_available, dropped=dropped)
    return {"sigmas": {k: fr.sigma for k, fr in results.items()},
            "fisher_results": results, "pysr_available": pysr_available,
            "refit_dir": refit_dir, **deliverables}


DISPATCH = {
    "gp_only": run_gp_only_multiz,
    "forecast_only": run_forecast_only_multiz,
    "refit_and_forecast": run_refit_and_forecast_multiz,
}


def run(cfg: MultiZPipelineConfig) -> dict:
    cfg.validate()
    return DISPATCH[cfg.mode](cfg)
