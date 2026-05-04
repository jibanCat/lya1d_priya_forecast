"""Pareto-selection filters for PySR equations.

Used by refit_one_param.py (per-1D) and refit_multi_d.py (multi-D
cross-coupled) to reject equations that LOOK low-loss but blow up at
prior boundaries or contain pathological constants.

Three guards, applied in order:
  1. `has_pathological_constant`: rejects equations whose literal
     coefficients exceed `|c| > threshold` (default 100).
     Catches `(x0 - 3.4e11) / (x3 - 0.23)` failure mode (technically
     uses x0 but is effectively constant in θ via huge offset).
  2. `is_eq_well_behaved`: lambdifies the eq, evaluates over the
     training X, rejects if predictions are non-finite (NaN/inf) or
     exceed `100 × y_range` in magnitude.
     Catches `sqrt(sqrt(k/(θ_herei + θ_Ap)))` failure mode (eq is
     finite at most θ but explodes when θ near 0 → NaN in Fisher
     stencil).
  3. `feature_count`: counts how many `xN` tokens appear in the
     equation; used as a tie-breaker preferring equations that use
     more input features.
"""

from __future__ import annotations

import re

import numpy as np


_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?")


def has_pathological_constant(eq_str: str, threshold: float = 100.0) -> bool:
    """True if any literal constant in `eq_str` has |c| > threshold.

    Catches PySR equations that nominally use a feature but with a
    huge offset (e.g. `(x0 - 3.4e11) / (x3 - 0.23)`) which is
    numerically constant in θ.
    """
    for m in _NUMBER_RE.finditer(eq_str):
        try:
            if abs(float(m.group())) > threshold:
                return True
        except ValueError:
            continue
    return False


def is_eq_well_behaved(
    eq_str: str,
    X: np.ndarray,
    y: np.ndarray,
    n_features: int,
    *,
    range_factor: float = 100.0,
) -> bool:
    """Evaluate the eq over the training set and check finiteness + range.

    Returns True iff:
      - sympy can parse the equation,
      - lambdify-eval over X produces all-finite values, and
      - max(|prediction|) ≤ range_factor × max(|target range|, 1).

    `n_features` is the expected number of `xN` symbols (matches X's
    column count).
    """
    import sympy as sp
    try:
        expr = sp.sympify(eq_str)
    except Exception:
        return False
    all_syms = [sp.Symbol(f"x{i}") for i in range(n_features)]
    try:
        fn = sp.lambdify(
            all_syms, expr,
            modules=[{"inv": lambda x: 1.0 / x}, "numpy"],
        )
    except Exception:
        return False

    cols = [X[:, i] for i in range(n_features)]
    try:
        with np.errstate(all="ignore"):
            pred = np.asarray(fn(*cols), dtype=float)
    except Exception:
        return False
    if pred.ndim == 0:
        # Scalar (eq is a constant); broadcast to match y.
        pred = np.full_like(y, float(pred), dtype=float)
    elif pred.shape != y.shape:
        try:
            pred = np.broadcast_to(pred, y.shape)
        except ValueError:
            return False
    if not np.all(np.isfinite(pred)):
        return False
    y_range = max(float(np.abs(y.max() - y.min())), 1.0)
    if np.max(np.abs(pred)) > range_factor * y_range:
        return False
    return True


def feature_count(eq_str: str, n_features: int) -> int:
    """Number of distinct `xN` features (N < n_features) referenced in the eq."""
    return sum(1 for i in range(n_features) if f"x{i}" in eq_str)
