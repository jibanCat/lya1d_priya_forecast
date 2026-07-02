"""Tests for the reusable paper-figures module."""
from __future__ import annotations

import warnings
from pathlib import Path

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


@needs_data
def test_pareto_grid_writes_file(tmp_path):
    import matplotlib
    matplotlib.use("Agg")
    run = pf.load_run()
    out = tmp_path / "grid.png"
    with pf.paper_style(usetex=False):
        pf.plot_pareto_faithfulness(run, out)
    assert out.exists() and out.stat().st_size > 0
