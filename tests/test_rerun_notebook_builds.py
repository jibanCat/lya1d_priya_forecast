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


def test_builder_reproduces_valid_notebook_when_present():
    if not BUILDER.exists():
        pytest.skip("builder is gitignored; not present on a clean checkout")
    subprocess.run([sys.executable, str(BUILDER)], check=True,
                   env={**os.environ, "PYTHONPATH": "src"})
    _check_valid(json.loads(NB.read_text()))
