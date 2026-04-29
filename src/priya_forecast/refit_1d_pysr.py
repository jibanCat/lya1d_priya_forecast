"""Clean reusable 1D PySR refitter.

The architectural fix the user is asking for: **train on flux_norm,
predict on raw P_F**. This decouples the two normalization concerns:

  1. Training target: flux_norm = (P_F - mean_k(k)) / std_k(k)  per k bin.
     This is a near-zero-centered, near-unit-variance target → polynomial /
     PySR fits much faster and avoids fighting the dominant k-shape.

  2. Inference output: raw P_F. We bundle the (mean_k, std_k) arrays
     with the equation and apply the inverse transform inside `predict`.

Concretely:

  P_F(theta, k) = f_pysr(theta_norm, k_norm) · std_k(k) + mean_k(k)

The student's `pysr_mf_given.py` already trains on flux_norm but the
output equations were used inconsistently downstream. This module makes
the round-trip explicit: every refit returns `(equation_str,
NormalizationSpec)` and a `predict` callable that always emits P_F.

Example:
    from priya_forecast.refit_1d_pysr import refit_1d_for_param
    result = refit_1d_for_param(
        gp=gp, param_name="ns", z=3.6, k_grid=k_eboss,
        n_train=128, niter=100, maxsize=25,
    )
    p_f = result.predict(theta_phys=0.99, k=k_eboss)  # raw P_F at ns=0.99
    print(result.equation_str)         # the discovered PySR expression
    print(result.norm.std_flux[0])     # the per-k normalization vector
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Default PySR setup: include exp / log / square because Lyα P_F has
# exp-decay and tilt structure (per H4 in PYSR_HYPOTHESIS.md). Mild
# parsimony (per H2: aggressive parsimony silently drops weakly-coupled
# params).
DEFAULT_PYSR_KWARGS = dict(
    niterations=100,
    maxsize=25,
    populations=30,
    population_size=33,
    parsimony=1e-3,
    binary_operators=["+", "-", "*", "/"],
    unary_operators=["log", "exp", "square"],
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    deterministic=True,
    parallelism="serial",
    verbosity=0,
)


@dataclass
class Refit1DResult:
    """Bundles the PySR-discovered equation + the normalization round-trip
    so callers always get raw P_F predictions.
    """
    param_name: str
    z: float
    equation_str: str          # PySR equation in x0/x1 convention
    pareto_complexity: int
    pareto_loss: float          # PySR's training loss (on flux_norm)
    pareto_complexities: list[int]
    pareto_losses: list[float]
    norm: object                # NormalizationSpec (mean_k, std_k, etc.)
    fid_value: float            # physical fid of this param
    k_grid: np.ndarray
    wall_time_s: float

    def predict(self, theta_phys: float | np.ndarray, k: np.ndarray) -> np.ndarray:
        """P_F(theta_phys, k) = f_pysr(theta_norm, k_norm) · std_k(k) + mean_k(k).

        `theta_phys` is in physical units (NOT normalized). The output is
        always raw P_F, matched onto `k`.
        """
        import sympy as sp
        expr = sp.sympify(self.equation_str)
        # Identify x0, x1, ... by name; default missing inputs to 0.
        x_syms = sorted(
            [s for s in expr.free_symbols if s.name.startswith("x")],
            key=lambda s: int(s.name[1:]),
        )
        # We trained on (x0=theta_norm, x1=k_norm). Bind both.
        x0, x1 = sp.Symbol("x0"), sp.Symbol("x1")
        all_syms = list({*x_syms, x0, x1})
        all_syms.sort(key=lambda s: int(s.name[1:]) if s.name.startswith("x") else 99)
        fn = sp.lambdify(all_syms, expr, modules=["numpy"])

        theta_phys = np.asarray(theta_phys, dtype=float)
        k = np.asarray(k, dtype=float)
        theta_norm = self.norm.normalize_param(theta_phys)
        k_norm = self.norm.normalize_k(k)
        # Broadcast theta_norm onto k.shape for the equation eval.
        if theta_norm.ndim == 0:
            theta_norm_arr = np.full_like(k, float(theta_norm))
        else:
            theta_norm_arr = np.asarray(theta_norm, dtype=float)
        # Pass both x0 and x1 plus zeros for any unused symbols.
        args = []
        for s in all_syms:
            if s.name == "x0":
                args.append(theta_norm_arr)
            elif s.name == "x1":
                args.append(k_norm)
            else:
                args.append(np.zeros_like(k))
        flux_norm = np.broadcast_to(np.asarray(fn(*args), dtype=float), k.shape).copy()
        # Denormalize back to P_F.
        return self.norm.denormalize_flux(flux_norm, k)


def _build_normalized_dataset(*, gp, param_name: str, z: float,
                              k_grid: np.ndarray, n_train: int, seed: int):
    """Sobol-sample one parameter at fid-others, evaluate the GP, normalize
    flux per-k. Returns (X, y_flux_norm, NormalizationSpec)."""
    from priya_forecast.parameters import (
        PARAM_NAMES, fiducial_vector, get_param,
    )
    from priya_forecast.models.normalization import NormalizationSpec
    from scipy.stats import qmc

    fid = np.array(fiducial_vector(), dtype=float)
    p = get_param(param_name)
    sampler = qmc.Sobol(d=1, seed=seed)
    u = sampler.random(n=n_train).ravel()
    thetas = p.prior[0] + p.width() * u
    flux = np.empty((n_train, k_grid.size))
    for i, t in enumerate(thetas):
        theta_full = fid.copy()
        theta_full[PARAM_NAMES.index(param_name)] = t
        flux[i] = gp.predict(theta_full, k_grid, z)

    # Per-k normalization (matches the student's `mf_*.py` recipe).
    mean_k = flux.mean(axis=0)
    std_k = flux.std(axis=0, ddof=0)
    std_k = np.where(std_k > 0, std_k, 1.0)

    spec = NormalizationSpec(
        param_min=p.prior[0], param_max=p.prior[1],
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        mean_flux=mean_k, std_flux=std_k, k_grid=k_grid,
    )

    flux_norm = (flux - mean_k) / std_k
    rows_X, rows_y = [], []
    for i in range(n_train):
        for ki, k in enumerate(k_grid):
            rows_X.append([
                (thetas[i] - p.prior[0]) / p.width(),
                (k - k_grid.min()) / (k_grid.max() - k_grid.min()),
            ])
            rows_y.append(flux_norm[i, ki])
    return np.asarray(rows_X), np.asarray(rows_y), spec


def refit_1d_for_param(
    *,
    gp,
    param_name: str,
    z: float,
    k_grid: np.ndarray,
    n_train: int = 128,
    seed: int = 0,
    pysr_kwargs: dict | None = None,
) -> Refit1DResult:
    """Train a single PySR equation for `param_name` on flux_norm; return a
    Refit1DResult that emits raw P_F via the bundled normalization.

    Defaults follow the empirical findings in `docs/PYSR_HYPOTHESIS.md`:
    operators include `exp`/`log`/`square`, parsimony is mild (1e-3),
    maxsize is 25, niter is 100. Override via `pysr_kwargs`.
    """
    import time
    from pysr import PySRRegressor  # type: ignore[import-not-found]
    from priya_forecast.parameters import get_param

    X, y, spec = _build_normalized_dataset(
        gp=gp, param_name=param_name, z=z, k_grid=k_grid,
        n_train=n_train, seed=seed,
    )
    args = dict(DEFAULT_PYSR_KWARGS)
    args.update(pysr_kwargs or {})
    args["random_state"] = seed
    t0 = time.time()
    model = PySRRegressor(**args)
    model.fit(X, y)
    elapsed = time.time() - t0
    pareto = model.equations_
    best_idx = int(pareto["loss"].idxmin())
    return Refit1DResult(
        param_name=param_name, z=z,
        equation_str=str(pareto.iloc[best_idx]["equation"]),
        pareto_complexity=int(pareto.iloc[best_idx]["complexity"]),
        pareto_loss=float(pareto.iloc[best_idx]["loss"]),
        pareto_complexities=pareto["complexity"].astype(int).tolist(),
        pareto_losses=pareto["loss"].astype(float).tolist(),
        norm=spec,
        fid_value=get_param(param_name).fid,
        k_grid=np.asarray(k_grid, dtype=float),
        wall_time_s=elapsed,
    )
