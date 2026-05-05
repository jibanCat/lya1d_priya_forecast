"""Smoke tests linking the Python reference ANOVA loss to the Julia port.

The full-strength check is an end-to-end PySR fit with
`loss_function=JULIA_LOSS_FUNCTION_ANOVA` on a controlled dataset where
we know which feature the residual depends on. That test is opt-in
(set `RUN_SLOW_PYSR_SMOKE=1`) because cold-starting PySR + Julia takes
60-120 s and is inappropriate for the default suite.

The cheap checks here verify that the Julia source string encodes the
same constants (`α=5`, `B=10`) as the Python reference, so a code-only
review can catch obvious drift without running Julia.
"""

from __future__ import annotations

import os

import pytest

from priya_forecast.dim_balanced_loss import (
    DEFAULT_ALPHA,
    JULIA_LOSS_FUNCTION,
    JULIA_LOSS_FUNCTION_ANOVA,
    JULIA_LOSS_FUNCTION_CORR,
)


def test_julia_default_aliases_anova_not_corr():
    """Production wires the ANOVA loss; the legacy corr² is opt-in."""
    assert JULIA_LOSS_FUNCTION is JULIA_LOSS_FUNCTION_ANOVA
    assert JULIA_LOSS_FUNCTION is not JULIA_LOSS_FUNCTION_CORR


def test_julia_anova_string_encodes_python_ref_constants():
    """The Julia source must pin the same α and B as the Python ref so
    the two versions can't drift silently. The numeric values are the
    explicit knobs in PAPER_NOTES § D3 (α=5, B=10)."""
    src = JULIA_LOSS_FUNCTION_ANOVA
    assert "n_bins = 10" in src, "B=10 quantile bins (PAPER_NOTES § D3)"
    assert "L(5.0) * pen" in src, "α=5 penalty weight (PAPER_NOTES § D3)"
    assert int(DEFAULT_ALPHA) == 5, (
        "Python DEFAULT_ALPHA must match the α=5 hard-coded in the "
        "Julia source above."
    )


def test_julia_anova_string_uses_full_batch_loss_signature():
    """`loss_function` is the full-batch PySR API (vs. per-sample
    `elementwise_loss`). The ANOVA bin means need the full batch, so
    this signature is required."""
    src = JULIA_LOSS_FUNCTION_ANOVA
    assert "function loss_function(tree, dataset::Dataset" in src
    assert "eval_tree_array(tree, dataset.X, options)" in src


@pytest.mark.skipif(
    os.environ.get("RUN_SLOW_PYSR_SMOKE") != "1",
    reason="Cold-starting PySR + Julia is 60-120s; opt-in via env var.",
)
def test_julia_anova_drives_pysr_to_use_x0():
    """End-to-end smoke: when y depends only on x0, PySR with
    `loss_function=JULIA_LOSS_FUNCTION_ANOVA` finds an x0-using eq.

    With plain MSE on a tiny dataset, parsimony pressure can let the
    constant-output baseline win the Pareto front; the ANOVA penalty
    should demote it because the residual would inherit the x0
    dependence (PAPER_NOTES § D3 'why this catches feature-dropping').

    Run via `RUN_SLOW_PYSR_SMOKE=1 pytest tests/test_pysr_dim_balanced_smoke.py`.
    """
    import numpy as np
    from pysr import PySRRegressor

    rng = np.random.default_rng(0)
    n = 200
    X = rng.uniform(0, 1, size=(n, 3))
    y = 2.0 * X[:, 0]  # depends only on x0
    model = PySRRegressor(
        niterations=20,
        binary_operators=["+", "-", "*"],
        unary_operators=["square"],
        loss_function=JULIA_LOSS_FUNCTION_ANOVA,
        deterministic=True,
        parallelism="serial",
        random_state=42,
        verbosity=0,
    )
    model.fit(X, y)
    best = str(model.get_best().equation)
    assert "x0" in best, (
        f"PySR with ANOVA loss should find an x0-using eq; got {best!r}"
    )
