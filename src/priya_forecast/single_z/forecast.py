"""forecast_only mode: Pareto CSVs → equations → combined model → Fisher.

This module reconstructs a `Refit1DResult` per parameter from a picked Pareto
equation plus the regenerated 1pvar training data, builds the combined model,
and runs the three Fisher forecasts (σ_GP, σ_perfect_1D, σ_PySR).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.fisher import FisherResult, fisher_matrix
from priya_forecast.ksdata_likelihood import KSDataLikelihood
from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.models.normalization import NormalizationSpec
from priya_forecast.parameters import get_param, PARAM_NAMES, PARAMS_11D
from priya_forecast.single_z.combine import build_combined_model
from priya_forecast.single_z.config import PipelineConfig
from priya_forecast.models.pysr_model import load_pareto_csv, pick_equation
from priya_forecast.pareto_filters import (
    has_pathological_constant,
    is_fisher_stencil_safe,
)
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


def equation_uses_param(equation_str: str) -> bool:
    """True if the PySR equation depends on x0 (the forecast parameter).

    An x0-free equation has zero ∂P/∂θ → a NaN/infinite Fisher σ. PySR
    sometimes returns such equations when it finds no θ-dependence.
    """
    import re
    # `x0` but not `x0` as a prefix of `x01` etc. (PySR uses x0..x9 here).
    return bool(re.search(r"\bx0\b", str(equation_str)))


def _filter_fisher_safe(df, n_features: int):
    """Drop Fisher-pathological Pareto rows; return the surviving sub-frame.

    Mirrors `scripts/refit_one_param.py`: an equation is kept only if it
    uses x0 (the forecast parameter), has no pathological constant, and is
    Fisher-stencil-safe. An x0-free equation has zero ∂P/∂θ → NaN σ_PySR.
    """
    eq = df["Equation"].astype(str)
    uses_x0 = eq.apply(equation_uses_param)
    pathological = eq.apply(has_pathological_constant)
    stencil_safe = eq.apply(
        lambda s: is_fisher_stencil_safe(s, n_features=n_features)
    )
    return df[uses_x0 & (~pathological) & stencil_safe].reset_index(drop=True)


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
            f"No x0-dependent / Fisher-safe equation in Pareto front for "
            f"({param_name}, z={z}): all {len(df)} rows were x0-free, "
            f"pathological, or stencil-unsafe."
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


# Vendored baseline Pareto CSVs (populated once Stage C produces them).
_BUNDLED_BASELINE_DIR = (
    Path(__file__).resolve().parents[2]
    / "priya_forecast" / "_vendored" / "data" / "pareto_baseline"
)


def resolve_pareto_csvs(cfg: PipelineConfig) -> dict[str, Path]:
    """Map each selected parameter to its Pareto-CSV path, per `pareto_csvs.source`.

    - `per_parameter`    → the path in each `ParetoEntry`.
    - `from_refit`       → `<output_dir>/refit/z{z}/pareto_{param}.csv`.
    - `bundled_baseline` → the vendored `_vendored/data/pareto_baseline/z{z}/`.

    Raises FileNotFoundError naming the parameter if a CSV is absent.
    """
    src = cfg.pareto_csvs.source
    z_tag = f"z{cfg.redshift}"
    out: dict[str, Path] = {}
    for param in cfg.parameters:
        if src == "per_parameter":
            entry = cfg.pareto_csvs.per_parameter[param]
            path = Path(entry.pareto_csv)
        elif src == "from_refit":
            path = Path(cfg.output_dir) / "refit" / z_tag / f"pareto_{param}.csv"
        elif src == "bundled_baseline":
            path = _BUNDLED_BASELINE_DIR / z_tag / f"pareto_{param}.csv"
        else:  # pragma: no cover - config.validate already guards this
            raise ValueError(f"unknown pareto_csvs.source {src!r}.")
        if not path.exists():
            raise FileNotFoundError(
                f"Pareto CSV for parameter {param!r} not found at {path} "
                f"(pareto_csvs.source={src!r})."
            )
        out[param] = path
    return out


def _build_likelihood(cfg, model):
    """Build the likelihood for `model` per `cfg.data.source`.

    kodiaq → KSDataLikelihood (real KODIAQ-SQUAD covariance, Karaçaylı+ 2021).
    eboss_dr14 → GaussianLikelihood (eBOSS DR14).
    """
    if cfg.data.source == "kodiaq":
        return KSDataLikelihood(
            model=model,
            z_min=cfg.redshift,
            z_max=cfg.redshift,
            k_min=cfg.k_range.min,
            k_max=cfg.k_range.max,
            cov_scale=cfg.data.cov_scale,
            mock_data=cfg.data.mock_data,
            conservative=cfg.data.conservative,
        )
    return GaussianLikelihood(
        model=model, z=cfg.redshift, cov_scale=cfg.data.cov_scale,
        mock_data="gp",
    )


def _likelihood_k_grid(like):
    """The k-grid a likelihood evaluates models on (KSData vs eBOSS)."""
    if hasattr(like, "kept_k"):
        return np.asarray(like.kept_k, dtype=float)
    return np.asarray(like.inputs.k_eboss, dtype=float)


def _fisher_for_likelihood(like, *, parameters, step_frac, rel_tol):
    """Run `fisher_matrix` for a pre-built likelihood over a parameter subset."""
    indices = [PARAM_NAMES.index(n) for n in parameters]
    selected = tuple(PARAMS_11D[i] for i in indices)
    theta_fid_full = np.array([p.fid for p in PARAMS_11D], dtype=float)
    return fisher_matrix(
        likelihood=like, theta_fid=theta_fid_full, params=selected,
        step_frac=step_frac, rel_tol=rel_tol, param_indices=indices,
    )


def run_three_fisher(
    *,
    cfg,
    gp,
    fid: np.ndarray,
    refits: dict,
) -> dict[str, FisherResult]:
    """Compute σ_GP, σ_perfect_1D, σ_PySR as a dict of FisherResults.

    All three Fisher forecasts use the SAME likelihood covariance
    (KODIAQ-SQUAD when `cfg.data.source == "kodiaq"`, else eBOSS DR14) and
    the SAME k-grid — they differ only in the forward model — so the σ's
    are directly comparable.
    """
    fid = np.asarray(fid, dtype=float)
    like_gp = _build_likelihood(cfg, gp)
    k_grid = _likelihood_k_grid(like_gp)

    none_refits = {n: None for n in PARAM_NAMES}
    perfect_model = build_combined_model(
        combine_mode=cfg.combine, gp=gp, fid=fid, refits=none_refits,
        k_grid=k_grid, z=cfg.redshift,
    )
    pysr_model = build_combined_model(
        combine_mode=cfg.combine, gp=gp, fid=fid, refits=refits,
        k_grid=k_grid, z=cfg.redshift,
    )
    like_perfect = _build_likelihood(cfg, perfect_model)
    like_pysr = _build_likelihood(cfg, pysr_model)

    common = dict(
        parameters=cfg.parameters,
        step_frac=cfg.fisher.step_frac,
        rel_tol=cfg.fisher.rel_tol,
    )
    return {
        "GP": _fisher_for_likelihood(like_gp, **common),
        "perfect_1D": _fisher_for_likelihood(like_perfect, **common),
        "PySR": _fisher_for_likelihood(like_pysr, **common),
    }
