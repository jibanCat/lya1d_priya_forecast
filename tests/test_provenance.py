"""Provenance convention: generated artifacts carry a git stamp (priya_forecast.provenance)."""
import importlib.util
import pathlib

from hypothesis import given, settings
from hypothesis import strategies as st

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


stamp_header = _load_stamp_script().stamp_header


# ---- unit tests ----------------------------------------------------------

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
    h0 = "# param=ns z=3.6 tol=0.25 log_space=True source=foo/pareto_ns.csv\n"
    h1 = stamp_header(h0, "abc1234")
    assert "git=abc1234" in h1 and h1.endswith("\n") and "source=foo" in h1
    assert stamp_header(h1, "abc1234") is None            # already stamped -> no double-stamp
    assert stamp_header("Complexity,Loss\n", "abc1234") is None  # not a header line


# ---- property-based test -------------------------------------------------

_TOKEN = st.text(
    alphabet=st.characters(blacklist_characters=" #\n=\t"), min_size=1, max_size=10,
)
_HEXISH = st.text(alphabet="0123456789abcdef", min_size=4, max_size=16)


@settings(max_examples=40, deadline=None)
@given(param=_TOKEN, git=_HEXISH)
def test_stamp_header_inserts_exactly_once_before_source_and_is_idempotent(param, git):
    """For any well-formed sidecar header, back-filling inserts one space-delimited
    git=<hash> before source=, keeps it a comment line, and never double-stamps."""
    h0 = f"# param={param} z=3.6 tol=0.25 log_space=True source=x/pareto.csv\n"
    h1 = stamp_header(h0, git)
    assert h1 is not None
    assert h1.startswith("# ")
    assert f" git={git} " in h1                         # inserted, space-delimited
    assert h1.count(f"git={git}") == 1                  # exactly once
    assert h1.index(f"git={git}") < h1.index("source=")  # before source=
    assert "source=x/pareto.csv" in h1                  # source preserved
    assert stamp_header(h1, git) is None                # idempotent


def test_headerless_csv_readers_tolerate_git_stamp(tmp_path):
    """pareto/maxsize/multid readers skip a leading '# git=...' provenance line."""
    import pandas as pd

    from priya_forecast.pareto_diag import load_front
    p = tmp_path / "pareto_ns.csv"
    p.write_text("# git=abc1234 source=pysr_hall_of_fame\n"
                 "Complexity,Loss,Equation\n1,0.5,x0\n2,0.4,x0*x0\n")
    front = load_front(p, None)                          # forecast/figure reader
    assert len(front) == 2 and "Complexity" in front.columns

    m = tmp_path / "maxsize_sensitivity.csv"
    m.write_text("# git=abc1234 source=maxsize_sweep\n"
                 "param,loss,maxsize,grad_err,complexity\nns,sobolev,20,0.16,18\n")
    df = pd.read_csv(m, comment="#")
    assert df.iloc[0]["param"] == "ns" and float(df.iloc[0]["grad_err"]) == 0.16
