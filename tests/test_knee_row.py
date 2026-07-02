"""Tests for the Pareto-knee selection (grad_faith_io.knee_row)."""
from __future__ import annotations

import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from priya_forecast.grad_faith_io import knee_row


def test_knee_picks_lowest_complexity_within_tol():
    # min loss 0.45; within 10% -> <=0.495. c10(0.5) excluded; c15(0.46),c20(0.45)
    # eligible -> lowest complexity = 15.
    df = pd.DataFrame({"Complexity": [5, 10, 15, 20], "Loss": [1.0, 0.5, 0.46, 0.45]})
    assert int(knee_row(df, rel_tol=0.1)["Complexity"]) == 15


def test_knee_avoids_the_truncated_ceiling():
    # Loss still falling at the ceiling (no real knee): best-loss(idxmin) would
    # take c=20; the knee takes the simplest within 10% (c=18), not the ceiling.
    df = pd.DataFrame({"Complexity": [16, 18, 20], "Loss": [11.6, 9.75, 8.9]})
    assert int(knee_row(df, rel_tol=0.1)["Complexity"]) == 18


def test_knee_empty_raises():
    with pytest.raises(ValueError, match="empty"):
        knee_row(pd.DataFrame({"Complexity": [], "Loss": []}))


@settings(max_examples=40, deadline=None)
@given(
    losses=st.lists(st.floats(min_value=1e-6, max_value=1e3,
                              allow_nan=False, allow_infinity=False),
                    min_size=1, max_size=12),
    rel_tol=st.floats(min_value=0.0, max_value=0.5),
)
def test_property_knee_within_tol_and_no_more_complex_than_best_loss(losses, rel_tol):
    df = pd.DataFrame({"Complexity": list(range(1, len(losses) + 1)), "Loss": losses})
    r = knee_row(df, rel_tol=rel_tol)
    lmin = min(losses)
    # (a) the knee's loss is within rel_tol of the best loss
    assert float(r["Loss"]) <= lmin * (1.0 + rel_tol) + 1e-9
    # (b) the knee is never MORE complex than the best-loss (idxmin) pick
    best_complexity = int(df.loc[df["Loss"].idxmin(), "Complexity"])
    assert int(r["Complexity"]) <= best_complexity
