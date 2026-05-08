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
import re

import pytest

from priya_forecast.dim_balanced_loss import (
    DEFAULT_ALPHA,
    JULIA_LOSS_FUNCTION,
    JULIA_LOSS_FUNCTION_ANOVA,
    JULIA_LOSS_FUNCTION_CORR,
)


# ---------------------------------------------------------------------------
# Numeric-knob extraction: parse the Julia source for α and B and assert on
# the *parsed numbers* rather than the literal substrings. This addresses
# GitHub Copilot's review on PR #2 (2026-05-05): a substring assertion on
# `"L(5.0) * pen"` had two failure modes:
#   (a) false positives — fails on harmless reformats like `L(5)` vs
#       `L(5.0)` or whitespace changes (no math change).
#   (b) false negatives — passes on math changes like `L(5.0) * pen / 2`
#       (α effectively halved while the substring still matches).
# Regex extraction with numeric assertions + the post-`pen` lookahead
# below addresses both modes.
# ---------------------------------------------------------------------------

# `L(α) * pen` — α is the penalty weight; bare integer or float, optional
# whitespace inside the L(…) call, optional `.0` suffix.
#
# The negative lookahead `(?!\s*[*/^])` rejects matches where `pen` is
# followed by a scaling operator (`L(5.0) * pen / 2`, `L(5.0) * pen * 2`,
# `L(5.0) * pen ^ 2`). Without this, a math change that effectively
# scales α by an external divisor/multiplier would still parse the
# literal `5.0` from the L(...) call and pass the equality assertion —
# the false-negative class GitHub Copilot called out in the PR #2
# review. Additive followups (`L(5.0) * pen + 0.001`, `... - small`,
# end-of-line, end-of-statement) are *allowed* because `+`/`-` after
# `pen` doesn't scale α.
_ALPHA_RE = re.compile(
    r"L\(\s*(\d+(?:\.\d+)?)\s*\)\s*\*\s*pen\b(?!\s*[*/^])",
    re.IGNORECASE,
)

# `n_bins = B` at module scope; whitespace tolerated.
_NBINS_RE = re.compile(r"n_bins\s*=\s*(\d+)", re.IGNORECASE)


def _extract_alpha(src: str) -> float:
    m = _ALPHA_RE.search(src)
    if m is None:
        raise AssertionError(
            "Could not find `L(<num>) * pen` pattern in Julia source. "
            "Either the regex is wrong (update _ALPHA_RE) or the Julia "
            "code stopped pinning α."
        )
    return float(m.group(1))


def _extract_nbins(src: str) -> int:
    m = _NBINS_RE.search(src)
    if m is None:
        raise AssertionError(
            "Could not find `n_bins = <int>` in Julia source. Either the "
            "regex is wrong (update _NBINS_RE) or B is no longer pinned."
        )
    return int(m.group(1))


def test_julia_default_aliases_anova_not_corr():
    """Production wires the ANOVA loss; the legacy corr² is opt-in."""
    assert JULIA_LOSS_FUNCTION is JULIA_LOSS_FUNCTION_ANOVA
    assert JULIA_LOSS_FUNCTION is not JULIA_LOSS_FUNCTION_CORR


def test_julia_anova_alpha_matches_python_default():
    """The α (penalty weight) parsed from the Julia source must equal
    `DEFAULT_ALPHA` from the Python ref. PAPER_NOTES § D3 pins α=5."""
    alpha = _extract_alpha(JULIA_LOSS_FUNCTION_ANOVA)
    assert alpha == float(DEFAULT_ALPHA), (
        f"Julia α={alpha} disagrees with Python DEFAULT_ALPHA="
        f"{DEFAULT_ALPHA}; one of the two has drifted."
    )
    assert int(DEFAULT_ALPHA) == 5, "PAPER_NOTES § D3 hard-codes α=5."


def test_julia_anova_n_bins_is_10():
    """B=10 quantile bins (PAPER_NOTES § D3 worked example assumes this)."""
    assert _extract_nbins(JULIA_LOSS_FUNCTION_ANOVA) == 10


def test_julia_anova_string_uses_full_batch_loss_signature():
    """`loss_function` is the full-batch PySR API (vs. per-sample
    `elementwise_loss`). The ANOVA bin means need the full batch, so
    this signature is required by PySR's API contract."""
    src = JULIA_LOSS_FUNCTION_ANOVA
    assert "function loss_function(tree, dataset::Dataset" in src
    assert "eval_tree_array(tree, dataset.X, options)" in src


# ---------------------------------------------------------------------------
# Unit tests for the regex extractors themselves: confirm they're robust to
# the formatting variations Copilot called out (`L(5)` vs `L(5.0)`, varied
# whitespace, etc.) AND that they fail loudly when the pattern is absent.
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("snippet,expected", [
    ("L(5.0) * pen", 5.0),
    ("L(5) * pen", 5.0),                    # integer literal also OK
    ("L( 5.0 ) * pen", 5.0),                # whitespace inside L(…)
    ("L(5.0)*pen", 5.0),                    # no surrounding whitespace
    ("return mse + L(5.0) * pen", 5.0),     # the production form
    ("return mse + L(5.0) * pen\n", 5.0),   # trailing newline
    ("L(5.0) * pen + 0.001", 5.0),          # additive followup is fine (no α scale)
    ("L(5.0) * pen - tiny", 5.0),           # subtractive followup is also fine
    ("L(7.5) * pen", 7.5),                  # different value parses correctly
    ("L(2) * pen", 2.0),                    # different value parses correctly
])
def test_alpha_regex_handles_formatting_variations(snippet, expected):
    """The same numeric α must parse identically across plausible Julia
    formatting choices. Documents what the regex tolerates: integer
    vs float literal, whitespace inside `L(…)`, multiplication
    spacing, additive followups (which don't scale α)."""
    assert _extract_alpha(snippet) == expected


@pytest.mark.parametrize("snippet", [
    # Standard "no pattern" cases (regex-not-found).
    "no_alpha_pattern_here",
    "L(5.0)+pen",                           # not a multiplication
    "L(5.0)",                               # missing `* pen`
    "alpha * pen",                          # missing `L(…)` wrapper
    # Critical: the Copilot false-negative class. Multiplicative or
    # divisor followups effectively scale α; the regex MUST reject
    # them so the test fails loudly rather than silently parsing 5.0
    # while the math has α=2.5 / α=10 / etc.
    "L(5.0) * pen / 2",                     # α effectively halved
    "L(5.0) * pen * 2",                     # α effectively doubled
    "L(5.0) * pen ^ 2",                     # α^2 — Julia uses ^ for power
    "L(5.0) * pen/2.0",                     # no whitespace before /
    "L(5.0)*pen*foo",                       # multiplicative chain
])
def test_alpha_regex_raises_when_pattern_absent(snippet):
    """If the Julia source doesn't match `L(<num>) * pen` (anchored
    against post-`pen` scaling operators), the extractor raises
    AssertionError. This is the "no false-negatives" half of the
    Copilot review fix."""
    with pytest.raises(AssertionError, match="L\\(<num>\\) \\* pen"):
        _extract_alpha(snippet)


def test_alpha_regex_rejects_external_multiplicative_scaling():
    """Direct test of GitHub Copilot's exact false-negative example
    (PR #2 review, 2026-05-05): the substring `"L(5.0) * pen"` would
    appear in `L(5.0) * pen / 2` (α effectively halved) and in
    `L(5.0) * pen * 2` (α effectively doubled). The substring-based
    assertion would silently pass on these; the regex with the
    post-`pen` `(?!\\s*[*/^])` lookahead does not."""
    halved = "return mse + L(5.0) * pen / 2"
    doubled = "return mse + L(5.0) * pen * 2"
    powered = "return mse + L(5.0) * pen ^ 2"
    for src in (halved, doubled, powered):
        with pytest.raises(AssertionError, match="L\\(<num>\\) \\* pen"):
            _extract_alpha(src)


@pytest.mark.parametrize("snippet,expected", [
    ("n_bins = 10", 10),
    ("n_bins=10", 10),
    ("n_bins  =  10", 10),
    ("    n_bins = 10", 10),
    ("n_bins = 5", 5),                      # different value parses correctly
    ("    n_bins = 20  # comment", 20),     # trailing comment OK
])
def test_nbins_regex_handles_formatting_variations(snippet, expected):
    """B parses across plausible spellings."""
    assert _extract_nbins(snippet) == expected


@pytest.mark.parametrize("snippet", [
    "no_nbins_here",
    "n_bins = ten",                         # word, not digits
    "bins = 10",                            # missing `n_` prefix
])
def test_nbins_regex_raises_when_pattern_absent(snippet):
    with pytest.raises(AssertionError, match="n_bins = <int>"):
        _extract_nbins(snippet)


def test_alpha_regex_first_match_wins_on_ambiguity():
    """If the Julia source ever contained multiple `L(<num>) * pen`
    patterns, the test extracts the first. This documents the choice
    so nobody is surprised; if a future refactor introduces a second
    pattern this test is the canary."""
    src = "L(5.0) * pen + L(7.5) * pen"
    assert _extract_alpha(src) == 5.0


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
