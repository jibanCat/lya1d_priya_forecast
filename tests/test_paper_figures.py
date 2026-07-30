"""Tests for the reusable paper-figures module."""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from priya_forecast import paper_figures as pf

_RUN = Path(pf.DEFAULT_RUN_DIR)
_HAS_DATA = (_RUN / "sobolev" / "refit" / "z3.6" / "grad_faith_ns.csv").exists()
needs_data = pytest.mark.skipif(not _HAS_DATA, reason="committed production run not present")


@given(g=st.floats(min_value=0.0, max_value=5.0), gate=st.floats(min_value=0.05, max_value=0.5))
def test_classify_property(g, gate):
    c = pf.classify(g, gate)
    if g <= gate:
        assert c == "faithful"
    elif g > 0.6:
        assert c == "resistant"
    else:
        assert c == "above-gate"


def test_classify_nan():
    assert pf.classify(float("nan")) == "n/a"


def test_load_run_missing_dir_warns(tmp_path):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        run = pf.load_run(tmp_path, z=3.6)
    assert run.sidecars == {}
    assert any("No grad_faith sidecars" in str(x.message) for x in w)


@needs_data
def test_load_run_and_taxonomy_match_paper():
    run = pf.load_run()  # committed default, no paths
    assert len(run.sidecars) == 22  # 11 params x 2 losses
    tax = pf.taxonomy(run).set_index("param")
    assert len(tax) == 11
    assert abs(tax.loc["ns", "sobolev_grad_err"] - 0.160) < 0.01
    assert (tax["class"] == "faithful").sum() == 9
    assert tax.loc["hub", "class"] == "resistant"
    assert tax.loc["bhfeedback", "class"] == "resistant"


@needs_data
def test_plot_functions_return_figures():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib.figure import Figure
    run = pf.load_run()
    with pf.paper_style(usetex=False):  # no TeX needed in CI
        for fn in (pf.plot_scorecard, pf.plot_maxsize_sensitivity,
                   pf.plot_seed_band, pf.plot_ns_budget, pf.plot_crossz, pf.plot_multid):
            assert isinstance(fn(run), Figure)


def test_budget_front_keeps_the_sidecars_real_values():
    """Regression: the placeholder NaN columns must not survive the merge.

    `load_front(path, None)` returns all-NaN grad_err/value_mse; merging without
    dropping them suffixed the real values to `*_y` and plotted 22 NaNs."""
    front = pd.DataFrame({"Complexity": [3, 7], "Loss": [1.0, 0.5],
                          "grad_err": [np.nan, np.nan], "value_mse": [np.nan, np.nan]})
    side = pd.DataFrame({"Complexity": [3, 7], "grad_err": [0.10, 0.29],
                         "value_mse": [2.0, 1.0]})
    out = pf.budget_front(front, side)
    assert not [c for c in out.columns if c.endswith(("_x", "_y"))]
    assert out["grad_err"].tolist() == [0.10, 0.29]
    assert out["value_mse"].tolist() == [2.0, 1.0]


@settings(max_examples=20, deadline=None)
@given(cx=st.lists(st.integers(1, 40), min_size=1, max_size=12, unique=True),
       shift=st.integers(0, 5))
def test_budget_front_never_reintroduces_nan_on_matched_rows(cx, shift):
    """Property: grad_err is NaN exactly where the sidecar has no matching row."""
    side_cx = [c + shift for c in cx]
    front = pd.DataFrame({"Complexity": cx, "grad_err": np.nan, "value_mse": np.nan})
    side = pd.DataFrame({"Complexity": side_cx,
                         "grad_err": np.linspace(0.01, 0.9, len(side_cx)),
                         "value_mse": np.linspace(1.0, 9.0, len(side_cx))})
    out = pf.budget_front(front, side)
    matched = set(cx) & set(side_cx)
    assert set(out.loc[out["grad_err"].notna(), "Complexity"]) == matched


def test_default_budget_arm_is_the_one_the_shipped_figure_used():
    """`load_run`'s budget arm must track `make_diagnostic_figs.DEFAULT_BUDGET`.

    The shipped `ns_budget_panel.pdf` came from the seed-0 arm; the old
    `budget35_value` default is a different maxsize-35 run whose knee sits on
    the other side of the gate, so the API and the figure must not diverge."""
    import inspect
    import re
    src = (Path(__file__).resolve().parents[1] / "scripts"
           / "make_diagnostic_figs.py").read_text()
    m = re.search(r'DEFAULT_BUDGET\s*=\s*f?"\{_PROD\}/(?P<sub>.+?)/refit/', src)
    assert m, "DEFAULT_BUDGET not found in scripts/make_diagnostic_figs.py"
    assert inspect.signature(pf.load_run).parameters["budget_sub"].default == m["sub"]


@needs_data
def test_ns_budget_arm_has_real_points_on_the_committed_run():
    from priya_forecast.pareto_diag import load_front
    run = pf.load_run()
    assert run.budget_ns is not None
    bp = pf._zdir(run.data_dir, run.budget_sub, run.z) / "pareto_ns.csv"
    out = pf.budget_front(load_front(bp, None), run.budget_ns)
    assert out["grad_err"].notna().sum() == 22  # every sidecar row lands


def test_seed_band_labels_say_knee_not_best_loss():
    """Regression: the shipped Fig. 5 legend read "best-loss" while its y-axis, its
    caption and the aggregator all say Pareto-knee. The band IS knee-selected --
    seed_band_summary.json's medians equal knee_row to 6 decimals."""
    assert all("knee" in s for s in pf.SEED_BAND_LABELS)
    assert not any("best-loss" in s or "best_loss" in s for s in pf.SEED_BAND_LABELS)


@needs_data
def test_seed_band_plots_exactly_the_committed_json():
    import matplotlib
    matplotlib.use("Agg")
    run = pf.load_run()
    P = run.seed_band["params"]
    params = [p for p in pf.PARAM_NAMES if p in P]
    with pf.paper_style(usetex=False):
        ax = pf.plot_seed_band(run).axes[0]
    legend = [t.get_text() for t in ax.get_legend().get_texts()]
    assert all("knee" in s for s in legend), legend
    for container, key in zip(ax.containers, ("value", "sobolev")):
        expected = np.clip([P[p][key][0] for p in params], 0, 1.2)
        assert np.allclose(container[0].get_ydata(), expected, atol=1e-12)


@needs_data
def test_pareto_grid_writes_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    run = pf.load_run()
    out = tmp_path / "grid.png"
    with pf.paper_style(usetex=False):
        pf.plot_pareto_faithfulness(run, out)
    assert out.exists() and out.stat().st_size > 0
