"""Finite-difference derivative-validation gate for PySR equations.

Compares a candidate equation's central-difference dP/dtheta at fid against
the GP's, using the SAME stencil the Fisher matrix consumes (fisher.py).
Equations whose gradient is unfaithful (the "Fisher's-Mirage" pathology) are
rejected before best_loss selection.
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
