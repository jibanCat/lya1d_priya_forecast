import json, subprocess, sys
from pathlib import Path


def test_builder_produces_valid_notebook(tmp_path):
    subprocess.run([sys.executable, "notebooks/_build_rerun_paper.py"],
                   check=True, env={"PYTHONPATH": "src", "PATH": __import__("os").environ["PATH"]})
    nb = json.loads(Path("notebooks/rerun_paper.ipynb").read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10
    srcs = "".join("".join(c["source"]) for c in nb["cells"])
    assert "results/tutorial_reruns" in srcs
    assert "RerunConfig" in srcs and "compare_to_production" in srcs
    assert "load_run" in srcs
    # no code cell contains a triple-quote (builder invariant)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert "'''" not in "".join(c["source"])
