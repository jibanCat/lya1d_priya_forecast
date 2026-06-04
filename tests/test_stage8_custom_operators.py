# tests/test_stage8_custom_operators.py
import numpy as np
import sympy as sp
from priya_forecast.custom_operators import (
    AQ_JULIA, EXTRA_SYMPY_MAPPINGS, LAMBDIFY_MODULES,
)


def test_aq_julia_def_shape():
    compact = AQ_JULIA.replace(" ", "")
    assert compact.startswith("aq(") and "sqrt(1+y^2)" in compact


def test_lambdify_modules_cover_inv_and_aq():
    assert "inv" in LAMBDIFY_MODULES and "aq" in LAMBDIFY_MODULES
    assert LAMBDIFY_MODULES["inv"](2.0) == 0.5
    np.testing.assert_allclose(LAMBDIFY_MODULES["aq"](1.0, 2.0), 1.0 / np.sqrt(5.0))


def test_extra_sympy_mappings_cover_inv_and_aq():
    assert "inv" in EXTRA_SYMPY_MAPPINGS and "aq" in EXTRA_SYMPY_MAPPINGS
    x, y = sp.symbols("x y")
    assert sp.simplify(EXTRA_SYMPY_MAPPINGS["aq"](x, y) - x / sp.sqrt(1 + y**2)) == 0


def test_aq_roundtrip_through_lambdify():
    expr = sp.sympify("aq(x0, 2*x1)")
    x0, x1 = sp.Symbol("x0"), sp.Symbol("x1")
    fn = sp.lambdify([x0, x1], expr, modules=[LAMBDIFY_MODULES, "numpy"])
    assert np.isclose(float(fn(0.5, 0.5)), 0.5 / np.sqrt(1 + 1.0))
