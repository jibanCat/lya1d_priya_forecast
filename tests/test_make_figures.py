"""scripts/make_figures.py draws the manuscript's figures from committed, tagged files."""
from __future__ import annotations

import importlib.util
import os
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "scripts" / "make_figures.py"


def _load():
    saved = list(sys.path)
    sys.path.insert(0, str(SCRIPT.parent))          # the script imports figstyle
    try:
        spec = importlib.util.spec_from_file_location("make_figures", SCRIPT)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    finally:
        sys.path[:] = saved
    return mod


mf = _load()
_HAS_TAG = subprocess.run(
    ["git", "-C", str(ROOT), "rev-parse", "-q", "--verify", f"{mf.ANCHOR}^{{commit}}"],
    capture_output=True).returncode == 0
needs_tag = pytest.mark.skipif(not _HAS_TAG, reason=f"needs tag {mf.ANCHOR} (a full git clone)")


def test_one_generator_per_figure_the_paper_renders():
    # fig:multid_bestworst and fig:multid_3d are added with the per-k data they plot.
    assert {"prediction", "param_errors", "seed_band", "pareto", "ns_budget"} <= set(mf.FIGURES)
    assert "resolution" not in mf.FIGURES            # the paper dropped that figure


@needs_tag
def test_accepts_an_input_that_matches_the_tag():
    path = mf.PROD / "seed_band" / "seed_band_summary.json"
    assert mf.at_anchor(path) == path


@needs_tag
def test_refuses_a_file_the_tag_does_not_contain():
    with pytest.raises(SystemExit, match="refusing to plot"):
        mf.at_anchor(SCRIPT)                         # added after the tag


@needs_tag
@pytest.mark.skipif(shutil.which("latex") is None, reason="figures are typeset with LaTeX")
def test_every_figure_builds_without_the_emulator(tmp_path):
    subprocess.run([sys.executable, str(SCRIPT), "--out", str(tmp_path)], check=True,
                   cwd=ROOT, env={**os.environ, "MPLBACKEND": "Agg"})
    built = {p.name for p in tmp_path.glob("*.pdf")}
    assert {"pysr_pred_tau0_Ap.pdf", "param_error_bars.pdf", "seed_band.pdf",
            "pareto_faithfulness.pdf", "ns_budget_panel.pdf"} <= built
