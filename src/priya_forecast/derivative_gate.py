"""Finite-difference derivative-validation gate for PySR equations.

Compares a candidate equation's central-difference dP/dtheta at fid against
the GP's, per k-bin, and flags equations whose *slope shape* is unfaithful
(the "Fisher's-Mirage" pathology) before best_loss selection.

This is a deliberately covariance-free, legible operating-point metric:
``median_k |dP_eq/dP_GP - 1|`` over non-negligible k-bins, tolerance 0.25.
It is NOT the covariance-weighted Fisher quantity ``sigma_eq/sigma_GP`` and
uses a simpler 3-point stencil (h=1e-3) than ``fisher.py`` (step_frac=0.01).
It is reported as a per-bin slope-shape error so readers apply their own
tolerance; the multi-z forecast's marginalized sigma ratio is the
confirmatory Fisher-level cross-check.
"""
from __future__ import annotations

import numpy as np

from priya_forecast.refit_1d_pysr import HF_RESOLUTION, Refit1DResult


def gp_param_gradient(*, gp, fid: np.ndarray, k_grid: np.ndarray, z: float,
                      param_idx: int, h: float = 1e-3) -> np.ndarray:
    """Central-difference dP_GP/dtheta_param at fid, per k-bin."""
    fid = np.asarray(fid, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    tp, tm = fid.copy(), fid.copy()
    step = h * max(abs(float(fid[param_idx])), 1.0)
    tp[param_idx] += step
    tm[param_idx] -= step
    pp = np.asarray(gp.predict(tp, k_grid, z), dtype=float)
    pm = np.asarray(gp.predict(tm, k_grid, z), dtype=float)
    return (pp - pm) / (2.0 * step)


def equation_param_gradient(*, refit: Refit1DResult, fid_value: float,
                            k_grid: np.ndarray, z: float, h: float = 1e-3,
                            resolution: float = HF_RESOLUTION) -> np.ndarray:
    """Central-difference dP_eq/dtheta at fid via the refit's own predict()."""
    k_grid = np.asarray(k_grid, dtype=float)
    step = h * max(abs(float(fid_value)), 1.0)
    pp = np.asarray(refit.predict(theta_phys=fid_value + step, k=k_grid,
                                  resolution=resolution, z=z), dtype=float)
    pm = np.asarray(refit.predict(theta_phys=fid_value - step, k=k_grid,
                                  resolution=resolution, z=z), dtype=float)
    return (pp - pm) / (2.0 * step)


def derivative_faithful(*, cand_grad: np.ndarray, target_grad: np.ndarray,
                        tol: float = 0.25, floor_frac: float = 1e-3) -> bool:
    """True if median_k |cand/target - 1| <= tol over non-negligible bins.

    Bins where |target_grad| is below `floor_frac` times its own max are
    masked out (a ~zero GP gradient makes the ratio meaningless / explosive).
    If every bin is masked, returns False (no usable gradient to validate).
    """
    cand = np.asarray(cand_grad, dtype=float)
    target = np.asarray(target_grad, dtype=float)
    amax = float(np.max(np.abs(target)))
    if amax == 0.0:
        return False
    keep = np.abs(target) >= floor_frac * amax
    if not np.any(keep):
        return False
    rel = np.abs(cand[keep] / target[keep] - 1.0)
    return bool(np.median(rel) <= tol)


def derivative_faithful_multiz(
    *, refit, gp, fid: np.ndarray, fid_value: float, k_grid: np.ndarray,
    z_grid, param_idx: int, tol: float = 0.25, floor_frac: float = 1e-3,
    h: float = 1e-3,
) -> bool:
    """True if the median over (k, z) of |∂eq/∂θ ÷ ∂P_GP/∂θ − 1| ≤ tol.

    Computes, per z in z_grid, the equation's finite-diff θ-gradient and the
    GP's, masks near-zero GP-gradient bins, and takes the median over all
    kept (k, z) pairs.  Returns False if no usable (k, z) pairs exist.
    """
    fid = np.asarray(fid, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    rel: list[float] = []
    for z in np.asarray(z_grid, dtype=float):
        tgt = gp_param_gradient(gp=gp, fid=fid, k_grid=k_grid, z=float(z),
                                param_idx=param_idx, h=h)
        g = equation_param_gradient(refit=refit, fid_value=fid_value,
                                    k_grid=k_grid, z=float(z), h=h)
        amax = float(np.max(np.abs(tgt)))
        if amax == 0.0:
            continue
        keep = np.abs(tgt) >= floor_frac * amax
        if not np.any(keep):
            continue
        rel.extend(list(np.abs(g[keep] / tgt[keep] - 1.0)))
    if not rel:
        return False
    return bool(np.median(np.asarray(rel)) <= tol)
