"""Provenance convention: generated artifacts carry a git stamp (priya_forecast.provenance)."""
import importlib.util
import pathlib

from priya_forecast.grad_faith_io import (
    SIDECAR_COLUMNS, read_grad_faith_sidecar, write_grad_faith_sidecar,
)
from priya_forecast.provenance import git_stamp


def _load_stamp_script():
    """Load scripts/stamp_grad_faith_git.py by path (it's a script, not a package)."""
    scr = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "stamp_grad_faith_git.py"
    spec = importlib.util.spec_from_file_location("stamp_grad_faith_git", scr)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_git_stamp_never_empty():
    s = git_stamp()
    assert isinstance(s, str) and s  # a short hash (opt. +dirty), or "nogit" — never empty


def test_sidecar_header_carries_git_and_still_reads(tmp_path):
    rows = [{c: 0 for c in SIDECAR_COLUMNS}]
    p = write_grad_faith_sidecar(
        tmp_path / "grad_faith_x.csv", rows,
        param="x", z=3.6, tol=0.25, log_space=True, source_pareto="pareto_x.csv",
    )
    header = open(p).readline()
    assert header.startswith("# ")
    assert "git=" in header               # convention: per-file git stamp
    assert "param=x" in header and "source=" in header
    df = read_grad_faith_sidecar(p)       # header still skipped by the reader
    assert list(df.columns) == list(SIDECAR_COLUMNS)


def test_backfill_stamp_header_is_idempotent():
    stamp_header = _load_stamp_script().stamp_header
    h0 = "# param=ns z=3.6 tol=0.25 log_space=True source=foo/pareto_ns.csv\n"
    h1 = stamp_header(h0, "abc1234")
    assert "git=abc1234" in h1 and h1.endswith("\n") and "source=foo" in h1
    assert stamp_header(h1, "abc1234") is None            # already stamped -> no double-stamp
    assert stamp_header("Complexity,Loss\n", "abc1234") is None  # not a header line
