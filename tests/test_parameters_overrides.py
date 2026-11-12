import dataclasses
import pytest
from priya_forecast import parameters as P


def test_with_overrides_replaces_fid_and_prior_only_named():
    out = P.with_overrides(fiducial={"ns": 0.95}, prior={"Ap": (1.0, 3.0)})
    names = [p.name for p in out]
    assert names == list(P.PARAM_NAMES)                 # order/names unchanged
    assert P.get_param("ns").fid == 0.983               # global untouched
    assert next(p for p in out if p.name == "ns").fid == 0.95
    assert next(p for p in out if p.name == "Ap").prior == (1.0, 3.0)
    # untouched params identical to base
    assert next(p for p in out if p.name == "tau0") == P.get_param("tau0")


def test_with_overrides_unknown_name_raises():
    with pytest.raises(KeyError):
        P.with_overrides(fiducial={"not_a_param": 1.0})


def test_with_overrides_none_returns_equal_copy():
    out = P.with_overrides()
    assert tuple(out) == tuple(P.PARAMS_11D)


def test_override_params_context_swaps_and_restores():
    original = P.PARAMS_11D
    assert P.fiducial_vector()[2] == 0.983              # ns index 2
    with P.override_params(fiducial={"ns": 0.90}):
        assert P.PARAMS_11D is not original
        assert P.get_param("ns").fid == 0.90
        assert P.fiducial_vector()[2] == 0.90           # accessor sees override
    assert P.PARAMS_11D is original                     # restored
    assert P.get_param("ns").fid == 0.983


def test_override_params_restores_on_exception():
    original = P.PARAMS_11D
    with pytest.raises(RuntimeError):
        with P.override_params(prior={"ns": (0.5, 0.6)}):
            raise RuntimeError("boom")
    assert P.PARAMS_11D is original


def test_override_params_noop_when_empty():
    original = P.PARAMS_11D
    with P.override_params():
        assert P.PARAMS_11D is original                 # no swap needed
