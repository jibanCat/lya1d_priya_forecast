"""`scripts/run_multid_pysr.py --basedir` and the JSON provenance block.

The committed §5.1 sweep (results/joint_multid_sweep/) ran with
`--basedir data/kodiaq_gp`, so `--basedir` must reach `GPModel` unchanged. The
default stays the bare `GPModel()` call an earlier, superseded sweep used; and
the rank JSON must say which emulator, budget and code produced it.
"""
from __future__ import annotations

import importlib.util
import json
import os
import pathlib
import sys
from types import SimpleNamespace
from unittest import mock

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from priya_forecast.parameters import get_param


def _load_script():
    """Load the script by path, undoing its import-time env/sys.path defaults."""
    scr = pathlib.Path(__file__).resolve().parents[1] / "scripts" / "run_multid_pysr.py"
    saved_path = list(sys.path)
    with mock.patch.dict(os.environ):
        spec = importlib.util.spec_from_file_location("run_multid_pysr", scr)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
    sys.path[:] = saved_path
    return mod


rmp = _load_script()


class _Stop(Exception):
    """Raised by the fake GPModel so main() stops before any real work."""


def _gpmodel_call(monkeypatch, tmp_path, extra_argv):
    import priya_forecast.models.gp_model as gm

    seen = {}

    def fake(*args, **kwargs):
        seen["args"], seen["kwargs"] = args, kwargs
        raise _Stop

    monkeypatch.setattr(gm, "GPModel", fake)
    monkeypatch.setattr(sys, "argv", ["run_multid_pysr.py", "--params", "ns", "Ap",
                                      "--output", str(tmp_path / "out"), *extra_argv])
    with pytest.raises(_Stop):
        rmp.main()
    return seen


# ---- unit tests ----------------------------------------------------------

def test_default_builds_the_bare_gpmodel(monkeypatch, tmp_path):
    """No --basedir -> the bare GPModel() call, unchanged."""
    assert _gpmodel_call(monkeypatch, tmp_path, []) == {"args": (), "kwargs": {}}


def _fake_fisher(n):
    return SimpleNamespace(F=np.diag(np.arange(1.0, n + 1)), sigma=np.ones(n))


def test_rank_json_records_the_run_block(tmp_path):
    params = [get_param("ns"), get_param("Ap")]
    run = {"git": "abc1234", "gp_basedir": "data/kodiaq_gp",
           "gp_basedir_name": "kodiaq_gp", "n_train": 512, "niter": 250, "seed": 3}
    rmp._write_joint_rank_json(
        path=tmp_path / "j.json", params=params, fr_student=_fake_fisher(2),
        fr_gp=_fake_fisher(2), equation="x0 + x1", loss=0.1, complexity=5, z=3.6,
        maxsize=25, run=run)
    assert json.loads((tmp_path / "j.json").read_text())["run"] == run


def test_rank_json_without_run_keeps_the_old_schema(tmp_path):
    """Writers that pass no `run` (every committed JSON) get no new key."""
    params = [get_param("ns"), get_param("Ap")]
    rmp._write_joint_rank_json(
        path=tmp_path / "j.json", params=params, fr_student=_fake_fisher(2),
        fr_gp=_fake_fisher(2), equation="x0 + x1", loss=0.1, complexity=5, z=3.6)
    assert "run" not in json.loads((tmp_path / "j.json").read_text())


# ---- hypothesis property tests -------------------------------------------

@settings(max_examples=10, deadline=None)
@given(basedir=st.text(st.characters(whitelist_categories=("L", "N")), min_size=1, max_size=12))
def test_basedir_reaches_gpmodel_unchanged(tmp_path_factory, basedir):
    with pytest.MonkeyPatch.context() as mp:
        call = _gpmodel_call(mp, tmp_path_factory.mktemp("b"), ["--basedir", basedir])
    assert call == {"args": (), "kwargs": {"basedir": basedir}}
