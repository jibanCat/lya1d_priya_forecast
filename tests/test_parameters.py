"""Unit + hypothesis tests for `priya_forecast.parameters`."""

from __future__ import annotations

import pytest
from hypothesis import given, strategies as st

from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    Param,
    fiducial_vector,
    get_param,
    prior_bounds,
    validate_priors,
)


# ---------------------------------------------------------------------------
# Unit tests — fixed expected values
# ---------------------------------------------------------------------------


def test_eleven_params_in_canonical_order():
    assert len(PARAMS_11D) == 11
    assert PARAM_NAMES == (
        "dtau0",
        "tau0",
        "ns",
        "Ap",
        "herei",
        "heref",
        "alphaq",
        "hub",
        "omegamh2",
        "hireionz",
        "bhfeedback",
    )


def test_fiducial_vector_matches_dataclass():
    fid = fiducial_vector()
    assert fid == tuple(p.fid for p in PARAMS_11D)
    assert fid[2] == pytest.approx(0.983)  # ns
    assert fid[3] == pytest.approx(1.46e-9)  # Ap


def test_priors_match_known_emulator_limits():
    # From emulator_params.json (the cosmology+IGM 9 params).
    bounds = dict(zip(PARAM_NAMES, prior_bounds()))
    assert bounds["ns"] == (0.8, 1.05)
    assert bounds["Ap"] == (1.2e-9, 2.6e-9)
    assert bounds["hub"] == (0.65, 0.75)
    assert bounds["omegamh2"] == (0.14, 0.146)


def test_get_param_known_and_unknown():
    p = get_param("ns")
    assert p.name == "ns"
    with pytest.raises(KeyError, match="Unknown PRIYA parameter"):
        get_param("not_a_param")


def test_default_priors_validate():
    # Sanity: shipped defaults must be self-consistent.
    validate_priors()


def test_param_width_is_hi_minus_lo():
    ns = get_param("ns")
    assert ns.width() == pytest.approx(0.25)


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(idx=st.integers(min_value=0, max_value=10))
def test_property_fid_strictly_inside_prior(idx: int):
    """Every shipped parameter has lo < fid < hi."""
    p = PARAMS_11D[idx]
    lo, hi = p.prior
    assert lo < p.fid < hi


@given(
    name=st.text(min_size=1, max_size=10).filter(str.isidentifier),
    fid=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    half_width=st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_property_validate_priors_accepts_well_formed(name: str, fid: float, half_width: float):
    """Any param with strictly-bracketing prior validates."""
    p = Param(name=name, fid=fid, prior=(fid - half_width, fid + half_width), latex="x")
    validate_priors((p,))


@given(
    fid=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    half_width=st.floats(min_value=1e-6, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_property_validate_priors_rejects_fid_at_or_outside_bounds(fid: float, half_width: float):
    """fid sitting on or outside the prior boundary must fail validation."""
    p_at_lo = Param("p", fid=fid, prior=(fid, fid + half_width), latex="x")
    p_at_hi = Param("p", fid=fid, prior=(fid - half_width, fid), latex="x")
    p_outside = Param("p", fid=fid + 2 * half_width, prior=(fid - half_width, fid + half_width), latex="x")
    for bad in (p_at_lo, p_at_hi, p_outside):
        with pytest.raises(ValueError, match="not strictly inside"):
            validate_priors((bad,))


@given(
    a=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
    b=st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
)
def test_property_validate_priors_rejects_degenerate_or_inverted(a: float, b: float):
    """lo >= hi must always raise."""
    if a < b:
        a, b = b, a  # now a >= b, so prior=(a, b) is degenerate or inverted
    p = Param("p", fid=0.5 * (a + b), prior=(a, b), latex="x")
    with pytest.raises(ValueError):
        validate_priors((p,))
