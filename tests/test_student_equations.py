"""Sanity-check the student's published PySR equations through the framework.

This test does *not* require the real GP — it asserts that the equations
(a) parse, (b) evaluate without error on the eBOSS k-grid given an auto-
derived normalization, and (c) the alphaq equation has zero gradient with
respect to alphaq (a previously-discovered bug: the published equation
contains no `alphaq` term).
"""

from __future__ import annotations

import numpy as np
import pytest
import sympy as sp

from priya_forecast.models.normalization import (
    DEFAULT_K_MAX,
    DEFAULT_K_MIN,
    NormalizationSpec,
)
from priya_forecast.models.pysr_model import _parse_safely, compile_equation
from priya_forecast.parameters import get_param

# Equations as quoted by the user from the InferenceLyaData write-up.
STUDENT_EQUATIONS = {
    "dtau0":  "(((1.4061172 - k)**(-0.5989224)) * dtau0) - (r * 1.3422583) + dtau0 - 1.3998809",
    "Ap":     "(((2*Ap)**(cos(k))) + ((-0.5290618 - sin(r)) * 1.4107764)) + Ap",
    "ns":     "((ns * k) - r) * 2.3955164",
    "alphaq": "cos(r + 0.7157408 - 1.5351741*k)**4 / 0.47581 - r - 1.04696",
}


def _norm(param_name: str, n_k: int = 35) -> NormalizationSpec:
    """Identity-ish spec with the parameter's prior bounds."""
    p = get_param(param_name)
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, n_k)
    return NormalizationSpec(
        param_min=p.prior[0], param_max=p.prior[1],
        k_min=DEFAULT_K_MIN, k_max=DEFAULT_K_MAX,
        mean_flux=np.full(n_k, 50.0),  # rough scale for eBOSS at z=3.6
        std_flux=np.full(n_k, 1.0),
        k_grid=k_grid,
    )


@pytest.mark.parametrize("pname", ["dtau0", "Ap", "ns", "alphaq"])
def test_student_equation_parses(pname: str):
    """Each quoted equation must parse via the sympy whitelist."""
    expr_str = STUDENT_EQUATIONS[pname]
    syms = {pname: sp.Symbol(pname), "k": sp.Symbol("k"), "r": sp.Symbol("r")}
    expr = _parse_safely(expr_str, syms)
    assert {s.name for s in expr.free_symbols} <= set(syms)


@pytest.mark.parametrize("pname", ["dtau0", "Ap", "ns", "alphaq"])
def test_student_equation_compiles_with_resolution_fixed(pname: str):
    """Each equation must compile with `r` fixed at HF=0.8 — no leftover free symbols."""
    ce = compile_equation(
        param_name=pname,
        raw_expression=STUDENT_EQUATIONS[pname],
        variables=[pname, "k", "r"],
        fix={"r": 0.8},
        norm=_norm(pname),
        fiducial=get_param(pname).fid,
    )
    k = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, 35)
    out = ce.evaluate(theta_i=get_param(pname).fid, k=k)
    assert out.shape == (35,)
    assert np.all(np.isfinite(out))


def test_alphaq_equation_has_no_alphaq_dependence():
    """Locked-in diagnosis: the published `alphaq` equation does NOT contain
    the alphaq symbol. This is a real bug the framework caught — when
    forecasting on alphaq, σ blows up by ~10^11× because df/dalphaq = 0.

    If/when the student fixes the upstream LaTeX or retrains, this test
    flips: drop or invert the assertion.
    """
    syms = {"alphaq": sp.Symbol("alphaq"), "k": sp.Symbol("k"), "r": sp.Symbol("r")}
    expr = _parse_safely(STUDENT_EQUATIONS["alphaq"], syms)
    assert sp.Symbol("alphaq") not in expr.free_symbols, (
        "alphaq appears in the published equation — re-run regen_sample_figures.py "
        "and the student-paper-eqs σ should drop dramatically."
    )


def test_student_equation_gradients_are_nonzero_for_working_three():
    """dtau0/Ap/ns equations must have a finite, non-zero gradient w.r.t. their
    parameter at fid (otherwise the Fisher would be singular and σ would
    explode the same way alphaq does)."""
    for pname in ("dtau0", "Ap", "ns"):
        ce = compile_equation(
            param_name=pname,
            raw_expression=STUDENT_EQUATIONS[pname],
            variables=[pname, "k", "r"],
            fix={"r": 0.8},
            norm=_norm(pname),
            fiducial=get_param(pname).fid,
        )
        k = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, 35)
        fid = get_param(pname).fid
        h = 0.01 * get_param(pname).width()
        f_plus = ce.evaluate(fid + h, k)
        f_minus = ce.evaluate(fid - h, k)
        df = (f_plus - f_minus) / (2 * h)
        assert np.all(np.isfinite(df))
        assert np.linalg.norm(df) > 0, f"Equation for {pname} has zero gradient."
