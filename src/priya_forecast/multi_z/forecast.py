"""Multi-z forecast_only: joint KSDataLikelihood over [z_min, z_max].

Approach A: one z-spanning likelihood + the existing fisher_matrix.
The likelihood loops z_blocks calling model.predict(θ,k,z) and stacks the
joint data vector, so the returned Fisher is F = Σ_z F(z).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from priya_forecast.fisher import FisherResult, fisher_matrix
from priya_forecast.ksdata_likelihood import KSDataLikelihood
from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D
from priya_forecast.multi_z.combine import build_combined_model_multiz
from priya_forecast.multi_z.refit import (
    build_refit_from_pareto_multiz,
    build_refit_from_pareto_multiz_gated,
)


def _build_likelihood(cfg, model):
    if cfg.data.source == "kodiaq":
        return KSDataLikelihood(
            model=model, z_min=cfg.z_min, z_max=cfg.z_max,
            k_min=cfg.k_range.min, k_max=cfg.k_range.max,
            cov_scale=cfg.data.cov_scale, mock_data=cfg.data.mock_data,
            conservative=cfg.data.conservative,
        )
    raise NotImplementedError(
        "multi-z forecast currently supports data.source='kodiaq' only."
    )


def shared_k_and_z_grid(like) -> tuple[np.ndarray, np.ndarray]:
    """Return (k_grid, z_grid) from a KSDataLikelihood's z_blocks.

    Asserts every z-block shares the same k-grid (required because
    MultiZAdditiveTaylorModel is built for one fixed k_grid). Raises a
    clear error if KODIAQ uses a non-uniform per-z binning in range.
    """
    kept_k = np.asarray(like.kept_k, dtype=float)
    z_grid = np.array([zv for zv, _ in like.z_blocks], dtype=float)
    blocks = [kept_k[sl] for _, sl in like.z_blocks]
    k0 = blocks[0]
    for zv, kb in zip(z_grid, blocks):
        if kb.shape != k0.shape or not np.allclose(kb, k0):
            raise ValueError(
                f"KODIAQ k-grid differs across z-blocks (z={zv}); Approach A "
                f"requires a common per-z k-grid. Block k-shapes: "
                f"{[b.shape for b in blocks]}."
            )
    return k0, z_grid


def _fisher_for_likelihood(like, *, parameters, step_frac, rel_tol):
    indices = [PARAM_NAMES.index(n) for n in parameters]
    selected = tuple(PARAMS_11D[i] for i in indices)
    theta_fid_full = np.array([p.fid for p in PARAMS_11D], dtype=float)
    return fisher_matrix(
        likelihood=like, theta_fid=theta_fid_full, params=selected,
        step_frac=step_frac, rel_tol=rel_tol, param_indices=indices,
    )


def run_three_fisher_multiz(
    *, cfg, gp, fid: np.ndarray, refits: dict,
) -> dict[str, FisherResult]:
    """σ_GP, σ_perfect_1D, σ_PySR on the joint multi-z likelihood."""
    fid = np.asarray(fid, dtype=float)
    log_space = (cfg.target_space == "log")

    like_gp = _build_likelihood(cfg, gp)
    k_grid, z_grid = shared_k_and_z_grid(like_gp)

    none_refits = {n: None for n in PARAM_NAMES}
    perfect_model = build_combined_model_multiz(
        combine_mode=cfg.combine, gp=gp, fid=fid, refits=none_refits,
        k_grid=k_grid, z_grid=z_grid, log_space=log_space,
    )
    pysr_model = build_combined_model_multiz(
        combine_mode=cfg.combine, gp=gp, fid=fid, refits=refits,
        k_grid=k_grid, z_grid=z_grid, log_space=log_space,
    )
    like_perfect = _build_likelihood(cfg, perfect_model)
    like_pysr = _build_likelihood(cfg, pysr_model)

    common = dict(parameters=cfg.parameters,
                  step_frac=cfg.fisher.step_frac, rel_tol=cfg.fisher.rel_tol)
    return {
        "GP": _fisher_for_likelihood(like_gp, **common),
        "perfect_1D": _fisher_for_likelihood(like_perfect, **common),
        "PySR": _fisher_for_likelihood(like_pysr, **common),
    }


def resolve_refit_artifacts(cfg) -> dict[str, tuple[Path, Path]]:
    """Map each parameter to (pareto_csv, norm_npz) under <output_dir>/refit.

    Layout: <output_dir>/refit/z{z_min}-{z_max}/pareto_{param}.csv (+ norm_{param}.npz).
    Missing parameters are omitted (caller falls back to GP slice).
    """
    base = Path(cfg.output_dir) / "refit" / f"z{cfg.z_min}-{cfg.z_max}"
    out: dict[str, tuple[Path, Path]] = {}
    for param in cfg.parameters:
        csv = base / f"pareto_{param}.csv"
        norm = base / f"norm_{param}.npz"
        if csv.exists() and norm.exists():
            out[param] = (csv, norm)
    return out


def load_refits(
    cfg,
    *,
    gp=None,
    fid: np.ndarray | None = None,
    k_grid: np.ndarray | None = None,
    z_grid=None,
) -> tuple[dict, list[str]]:
    """Reconstruct refits from artifacts; return (refits, dropped).

    When ``gp``, ``fid``, ``k_grid``, and ``z_grid`` are all provided the
    gated builder ``build_refit_from_pareto_multiz_gated`` is used: candidates
    are iterated in ascending-loss order and the first whose finite-difference
    θ-gradient is faithful to the GP over the (k, z) grid is returned.
    Otherwise the un-gated ``build_refit_from_pareto_multiz`` (best-loss pick)
    is used, preserving backward compatibility for existing callers and tests.
    """
    use_gate = (gp is not None and fid is not None
                and k_grid is not None and z_grid is not None)
    refits: dict = {n: None for n in PARAM_NAMES}
    dropped: list[str] = []
    for param, (csv, norm) in resolve_refit_artifacts(cfg).items():
        try:
            if use_gate:
                refits[param] = build_refit_from_pareto_multiz_gated(
                    param_name=param, z_min=cfg.z_min, z_max=cfg.z_max,
                    pareto_csv=csv, norm_npz=norm, gp=gp, fid=fid,
                    k_grid=k_grid, z_grid=z_grid,
                )
            else:
                refits[param] = build_refit_from_pareto_multiz(
                    param_name=param, z_min=cfg.z_min, z_max=cfg.z_max,
                    pareto_csv=csv, norm_npz=norm, pick_rule=cfg.pick,
                )
        except ValueError as exc:
            dropped.append(param)
            print(f"[multi_z forecast] {param}: {exc} — GP-slice fallback.")
    return refits, dropped
