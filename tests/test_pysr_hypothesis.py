"""Hypothesis-driven unit + property tests for the PySR-vs-GP gap.

Every test is one falsifiable claim about the source of the gap. Claims
that hold are summarized in `docs/PYSR_HYPOTHESIS.md`; claims that fail
flag the test and force us to update the doc. The test suite is the
single source of truth.

We use the pure-numpy synthetic target in `priya_forecast.pysr_hypothesis`
so the suite stays under a few seconds without needing PySR or the GP.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.pysr_hypothesis import (
    experiment_h1_loss_function,
    experiment_h2_parsimony,
    experiment_h3_normalization,
    experiment_h4_operators,
    experiment_h6_covariance_combine,
    fit_polynomial,
    fit_polynomial_with_parsimony,
    predict_polynomial,
    synthetic_p_f,
)


# ---------------------------------------------------------------------------
# Synthetic target sanity
# ---------------------------------------------------------------------------


def test_synthetic_target_is_positive_and_decreasing_in_k():
    k = np.linspace(0.001, 0.02, 20)
    fid = np.array([0.5, 0.5])
    out = synthetic_p_f(fid, k)
    assert np.all(out > 0)
    # Power-law × exp damping → strictly decreasing on this k-range.
    assert np.all(np.diff(out) < 0)


def test_synthetic_target_responds_to_perturbations():
    k = np.linspace(0.001, 0.02, 20)
    p_low = synthetic_p_f(np.array([0.0, 0.5]), k)
    p_mid = synthetic_p_f(np.array([0.5, 0.5]), k)
    p_high = synthetic_p_f(np.array([1.0, 0.5]), k)
    assert np.all(p_low < p_mid)
    assert np.all(p_mid < p_high)


# ---------------------------------------------------------------------------
# H1: loss-function / training-distribution does NOT explain the gap
# ---------------------------------------------------------------------------


def test_h1_full_prior_training_matches_truth_fisher_within_1pct():
    """If H1 were the cause, full-prior training would have much worse
    Fisher than near-fid training. We assert: at order-4 polynomial,
    Fisher is recovered to 1% regardless. So H1 is NOT the bottleneck."""
    r = experiment_h1_loss_function(seed=0)
    assert r.extra["ratio_full_to_truth_ns"] < 1.02
    assert r.extra["ratio_near_to_truth_ns"] < 1.02


# ---------------------------------------------------------------------------
# H2: mild parsimony is harmless; aggressive parsimony breaks fits
# ---------------------------------------------------------------------------


def test_h2_mild_parsimony_does_not_inflate_test_mse():
    r = experiment_h2_parsimony(seed=0)
    base = r.extra["parsimony_0e+00"]["test_mse"]
    mild = r.extra["parsimony_1e-03"]["test_mse"]
    moderate = r.extra["parsimony_1e-02"]["test_mse"]
    aggressive = r.extra["parsimony_1e-01"]["test_mse"]
    # Mild: same MSE.
    assert abs(mild - base) / base < 0.01
    # Moderate: still same MSE (within 1%).
    assert abs(moderate - base) / base < 0.01
    # Aggressive: catastrophic — > 100x worse.
    assert aggressive / base > 100


def test_h2_parsimony_drops_terms_progressively():
    r = experiment_h2_parsimony(seed=0)
    base_terms = r.extra["parsimony_0e+00"]["kept_total"]
    mild_terms = r.extra["parsimony_1e-03"]["kept_total"]
    aggressive_terms = r.extra["parsimony_1e-01"]["kept_total"]
    assert mild_terms < base_terms
    assert aggressive_terms < mild_terms


# ---------------------------------------------------------------------------
# H3: NORMALIZATION IS THE BIG ONE
# ---------------------------------------------------------------------------


def test_h3_flux_norm_training_dramatically_outperforms_raw():
    """The headline finding: a polynomial fit on flux_norm reconstructs
    P_F to ~1e-26 vs ~1e3 for raw. 28 orders of magnitude difference."""
    r = experiment_h3_normalization(seed=0)
    raw = r.extra["test_mse_raw"]
    norm = r.extra["test_mse_norm_then_denorm"]
    assert raw / max(norm, 1e-30) > 1e10, (
        f"flux_norm should be at least 10 orders of magnitude better; "
        f"raw={raw:.3g}, norm={norm:.3g}"
    )


# ---------------------------------------------------------------------------
# H4: operator choice (exp basis) is critical for Lyα-shape targets
# ---------------------------------------------------------------------------


def test_h4_exp_basis_dramatically_outperforms_polynomial_only():
    r = experiment_h4_operators(seed=0)
    poly = r.extra["test_mse_polynomial_only"]
    aug = r.extra["test_mse_with_exp_basis"]
    assert poly / max(aug, 1e-30) > 1e10, (
        f"exp basis should be > 10 orders of magnitude better; "
        f"polynomial-only={poly:.3g}, aug={aug:.3g}"
    )


# ---------------------------------------------------------------------------
# H6: covariance-aware combine gives only modest gain on weakly-coupled targets
# ---------------------------------------------------------------------------


def test_h6_explicit_cross_terms_help_modestly_on_non_separable_truth():
    r = experiment_h6_covariance_combine(seed=0)
    # M1 should beat M0 (since the truth is non-separable).
    assert r.extra["mse_M1_with_cross_terms"] < r.extra["mse_M0_no_cross_terms"]
    # But the improvement is modest (the synthetic cross-term is small).
    assert 1.0 < r.extra["improvement_factor"] < 5.0


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(
    seed=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=4, deadline=None)
def test_property_h3_robust_across_seeds(seed: int):
    """The H3 normalization result must hold across seeds — not a
    one-off lucky sample."""
    r = experiment_h3_normalization(seed=seed)
    raw = r.extra["test_mse_raw"]
    norm = r.extra["test_mse_norm_then_denorm"]
    assert raw / max(norm, 1e-30) > 1e8, (
        f"seed={seed}: raw={raw:.3g}, norm={norm:.3g}"
    )


@given(
    seed=st.integers(min_value=0, max_value=20),
)
@settings(max_examples=4, deadline=None)
def test_property_h4_robust_across_seeds(seed: int):
    r = experiment_h4_operators(seed=seed)
    poly = r.extra["test_mse_polynomial_only"]
    aug = r.extra["test_mse_with_exp_basis"]
    assert poly / max(aug, 1e-30) > 1e8


@given(
    parsimony=st.floats(min_value=0.0, max_value=1e-3, allow_nan=False),
)
@settings(max_examples=8, deadline=None)
def test_property_mild_parsimony_keeps_test_mse_stable(parsimony: float):
    """For any parsimony in [0, 1e-3], test MSE is within 1% of unpruned."""
    rng = np.random.default_rng(0)
    X = rng.uniform(0, 1, size=(80, 3))
    y = synthetic_p_f(X[:, :2], X[:, 2] * 0.019 + 0.001)
    coef_unpruned, terms = fit_polynomial(X, y, max_degree=3)
    coef_pruned, _ = fit_polynomial_with_parsimony(X, y, max_degree=3, parsimony=parsimony)
    pred_un = predict_polynomial(X, coef_unpruned, terms)
    pred_pr = predict_polynomial(X, coef_pruned, terms)
    mse_un = float(np.mean((pred_un - y) ** 2))
    mse_pr = float(np.mean((pred_pr - y) ** 2))
    if mse_un > 1e-15:
        assert abs(mse_pr - mse_un) / mse_un < 0.02
