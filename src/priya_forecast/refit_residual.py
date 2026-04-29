"""Residual learning on top of a 1D-product baseline.

User's Comment 4: "Does it work better if a combination of 1D learned
PySR eqs that we use another PySR to learn the residuals of the 1D
combined?" — the answer is "build it and measure."

Architecture:

  1. Get the per-parameter `Refit1DResult` for each param (from
     `refit_1d_pysr.refit_1d_for_param`).
  2. Build a residual training set by:
        a. Sobol-sample the chosen multi-D parameter subspace via the GP.
        b. Compute P_baseline = P_fid · ∏ f_i(θ_i, k) / f_i(θ_fid_i, k)
           using the cached 1D fits.
        c. Compute residual = P_GP - P_baseline.
        d. Train a SECOND PySR on (theta_subspace, k) → residual.
  3. Final model: P(theta, k) = P_baseline(theta, k) + g_residual(theta, k).

The residual is BY CONSTRUCTION smaller than P_GP itself, so PySR sees
a near-zero target and finds compact correction terms — tracking the
cross-coupling the 1D-product missed.

The same pattern applies to a smaller target subspace (e.g. only the
4 forecast params + k) for tractability.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from itertools import combinations_with_replacement
from pathlib import Path
from typing import Callable

import numpy as np

from priya_forecast.parameters import (
    PARAM_NAMES, fiducial_vector, get_param,
)
from priya_forecast.refit_1d_pysr import Refit1DResult


@dataclass
class ResidualFit:
    """Bundles the trained residual-PySR + the baseline equations + the
    multi-D normalization needed for inference.
    """
    varying_names: list[str]
    refits_1d: dict           # param_name -> Refit1DResult
    residual_equation_str: str
    residual_complexity: int
    residual_loss: float       # PySR training MSE on the residual
    k_grid: np.ndarray
    z: float
    wall_time_s: float

    def predict(self, theta_full: np.ndarray, k: np.ndarray) -> np.ndarray:
        """Full multiplicative-product baseline + residual correction.

        `theta_full` is a length-11 PRIYA parameter vector in physical units.
        Output: raw P_F on `k`.
        """
        # 1D-product baseline
        fid = np.array(fiducial_vector(), dtype=float)
        # Need P_fid from the residual training context — caller can store
        # it or recompute via GP. For self-contained inference, we save the
        # P_fid array in the residual ResidualFit instance via train.
        raise NotImplementedError(
            "Use ResidualFit.predict_with_p_fid(theta, k, p_fid) — we don't "
            "carry P_fid here to avoid pickling the GP. Caller computes it."
        )

    def predict_with_p_fid(
        self,
        theta_full: np.ndarray,
        k: np.ndarray,
        p_fid: np.ndarray,
    ) -> np.ndarray:
        """Same as predict but the caller supplies P_fid(k) from the GP."""
        import sympy as sp

        out = p_fid.copy()
        for pname, r in self.refits_1d.items():
            if r is None:
                continue
            i = PARAM_NAMES.index(pname)
            num = r.predict(theta_phys=float(theta_full[i]), k=k)
            den = r.predict(theta_phys=float(r.fid_value), k=k)
            with np.errstate(divide="ignore", invalid="ignore"):
                out = out * (num / den)

        # Residual correction.
        expr = sp.sympify(self.residual_equation_str)
        n_in = len(self.varying_names) + 1  # + k
        # The PySR residual was trained with x0..xN where x_last = k_norm.
        x_syms = sorted(
            [s for s in expr.free_symbols if s.name.startswith("x")],
            key=lambda s: int(s.name[1:]),
        )
        all_syms = sorted(
            list({sp.Symbol(f"x{i}") for i in range(n_in)} | set(x_syms)),
            key=lambda s: int(s.name[1:]),
        )
        fn = sp.lambdify(all_syms, expr, modules=["numpy"])

        # Build the input vectors at all (varying_norm, k_norm) points.
        args = []
        k_norm = (k - self.k_grid.min()) / (self.k_grid.max() - self.k_grid.min())
        for s in all_syms:
            col = int(s.name[1:])
            if col < len(self.varying_names):
                pname = self.varying_names[col]
                p = get_param(pname)
                idx = PARAM_NAMES.index(pname)
                theta_norm = (theta_full[idx] - p.prior[0]) / p.width()
                args.append(np.full_like(k, float(theta_norm)))
            elif col == len(self.varying_names):
                args.append(k_norm)
            else:
                args.append(np.zeros_like(k))
        residual = np.broadcast_to(np.asarray(fn(*args), dtype=float), k.shape).copy()
        return out + residual


def fit_residual(
    *,
    gp,
    refits_1d: dict,
    varying_names: list[str],
    z: float,
    k_grid: np.ndarray,
    n_train: int = 128,
    seed: int = 0,
    pysr_kwargs: dict | None = None,
) -> ResidualFit:
    """Train a residual PySR on top of the 1D-product baseline.

    `refits_1d` is the dict returned by looping `refit_1d_for_param` over
    `PARAM_NAMES`. Pass `varying_names` = the parameter subset to vary in
    the residual training set; others stay at fid.
    """
    from pysr import PySRRegressor  # type: ignore[import-not-found]
    from scipy.stats import qmc

    fid = np.array(fiducial_vector(), dtype=float)
    p_fid = gp.predict(fid, k_grid, z)

    # Sobol over the varying subspace.
    sampler = qmc.Sobol(d=len(varying_names), seed=seed)
    u = sampler.random(n=n_train)

    # Pre-cache f_i_at_fid for each refit (fixed across thetas).
    f_fid_cache = {
        pn: r.predict(theta_phys=r.fid_value, k=k_grid)
        for pn, r in refits_1d.items() if r is not None
    }

    rows_X, rows_y = [], []
    k_norm = (k_grid - k_grid.min()) / (k_grid.max() - k_grid.min())
    for i in range(n_train):
        theta = fid.copy()
        for col, name in enumerate(varying_names):
            p = get_param(name)
            theta[PARAM_NAMES.index(name)] = p.prior[0] + p.width() * u[i, col]

        # Baseline P_F at theta via 1D-product.
        p_baseline = p_fid.copy()
        for pname, r in refits_1d.items():
            if r is None:
                continue
            idx = PARAM_NAMES.index(pname)
            num = r.predict(theta_phys=float(theta[idx]), k=k_grid)
            den = f_fid_cache[pname]
            with np.errstate(divide="ignore", invalid="ignore"):
                p_baseline = p_baseline * (num / den)
        p_truth = gp.predict(theta, k_grid, z)
        residual = p_truth - p_baseline

        for ki, k_val in enumerate(k_grid):
            row = []
            for col, name in enumerate(varying_names):
                row.append(u[i, col])
            row.append(k_norm[ki])
            rows_X.append(row)
            rows_y.append(residual[ki])

    X = np.asarray(rows_X)
    y = np.asarray(rows_y)
    print(f"  residual training set: X={X.shape}, "
          f"residual range = [{y.min():.3g}, {y.max():.3g}]")

    args = dict(
        niterations=80, maxsize=20, populations=20, parsimony=1e-3,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "square"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=seed, deterministic=True, parallelism="serial", verbosity=0,
    )
    args.update(pysr_kwargs or {})
    t0 = time.time()
    model = PySRRegressor(**args)
    model.fit(X, y)
    elapsed = time.time() - t0
    pareto = model.equations_
    best_idx = int(pareto["loss"].idxmin())

    return ResidualFit(
        varying_names=list(varying_names),
        refits_1d=refits_1d,
        residual_equation_str=str(pareto.iloc[best_idx]["equation"]),
        residual_complexity=int(pareto.iloc[best_idx]["complexity"]),
        residual_loss=float(pareto.iloc[best_idx]["loss"]),
        k_grid=np.asarray(k_grid, dtype=float),
        z=z,
        wall_time_s=elapsed,
    )
