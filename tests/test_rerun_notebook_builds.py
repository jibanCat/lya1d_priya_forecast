import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

NB = Path("notebooks/rerun_paper.ipynb")
BUILDER = Path("notebooks/_build_rerun_paper.py")


def _check_valid(nb):
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10
    srcs = "".join("".join(c["source"]) for c in nb["cells"])
    assert "results/tutorial_reruns" in srcs
    assert "RerunConfig" in srcs and "compare_to_production" in srcs
    assert "load_run" in srcs
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert "'''" not in "".join(c["source"])


def test_committed_notebook_is_valid():
    # Works on a clean checkout: the notebook is committed; the builder is not.
    _check_valid(json.loads(NB.read_text()))


def test_builder_reproduces_valid_notebook_when_present(tmp_path):
    # --out is mandatory here: the builder emits an UNEXECUTED notebook, so
    # letting it default to NB would silently strip the committed copy's 94
    # outputs (it has done exactly that twice).
    if not BUILDER.exists():
        pytest.skip("builder is gitignored; not present on a clean checkout")
    out = tmp_path / "rerun_paper.ipynb"
    before = NB.read_bytes()
    subprocess.run([sys.executable, str(BUILDER), "--out", str(out)], check=True,
                   env={**os.environ, "PYTHONPATH": "src"})
    _check_valid(json.loads(out.read_text()))
    assert NB.read_bytes() == before, "builder overwrote the committed notebook"


def test_committed_notebook_keeps_its_executed_outputs():
    nb = json.loads(NB.read_text())
    n_out = sum(len(c.get("outputs", [])) for c in nb["cells"])
    assert n_out >= 90, f"committed notebook looks output-stripped ({n_out} outputs)"
