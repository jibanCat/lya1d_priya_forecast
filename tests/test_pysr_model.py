"""Unit + hypothesis tests for `priya_forecast.models.pysr_model`."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.config import EqnConfig, EqnParam
from priya_forecast.models.normalization import (
    DEFAULT_K_MAX,
    DEFAULT_K_MIN,
    NormalizationSpec,
    identity,
)
from priya_forecast.models.pysr_model import (
    PySRModel,
    _parse_safely,
    compile_equation,
    load_pareto_csv,
    pick_equation,
)
import sympy as sp


# ---------------------------------------------------------------------------
# Sympy whitelist
# ---------------------------------------------------------------------------


def test_parse_safely_accepts_known_symbols_and_funcs():
    syms = {"a": sp.Symbol("a"), "k": sp.Symbol("k")}
    expr = _parse_safely("sin(a) + log(k)*2", syms)
    assert {s.name for s in expr.free_symbols} == {"a", "k"}


def test_parse_safely_rejects_unknown_symbol():
    syms = {"a": sp.Symbol("a")}
    with pytest.raises(ValueError, match="unknown symbols"):
        _parse_safely("a + b", syms)


def test_parse_safely_rejects_garbage():
    syms = {"a": sp.Symbol("a")}
    with pytest.raises(ValueError, match="Failed to parse"):
        _parse_safely("a +)", syms)


def test_parse_safely_handles_pysr_specials():
    """square / inv / pow are part of PySR's native operator set."""
    syms = {"x0": sp.Symbol("x0"), "x1": sp.Symbol("x1")}
    expr = _parse_safely("square(x0) + inv(x1)", syms)
    # square(x0) → x0**2
    assert expr == sp.Symbol("x0") ** 2 + 1 / sp.Symbol("x1")


# ---------------------------------------------------------------------------
# Pareto CSV picking
# ---------------------------------------------------------------------------


def _write_csv(tmp_path: Path, body: str, name: str = "hall_of_fame.csv") -> Path:
    p = tmp_path / name
    p.write_text(body)
    return p


def test_load_pareto_csv_parses_student_format():
    """The student's real CSV at priya_pysr/outputs/.../hall_of_fame.csv."""
    p = Path(
        "/home/mfho/student_projects/priya_pysr/outputs/"
        "20250908_084612_ldBfZE/hall_of_fame.csv"
    )
    if not p.exists():
        pytest.skip("Student CSV not present in this environment.")
    df = load_pareto_csv(p)
    assert set(df.columns) >= {"Complexity", "Loss", "Equation"}
    assert (df["Complexity"] > 0).all()


def test_load_pareto_csv_rejects_missing_columns(tmp_path: Path):
    p = _write_csv(tmp_path, "Complexity,Loss\n1,2\n")
    with pytest.raises(ValueError, match="missing required columns"):
        load_pareto_csv(p)


def test_pick_best_loss(tmp_path: Path):
    p = _write_csv(
        tmp_path,
        "Complexity,Loss,Equation\n1,2.0,x0\n3,0.5,x0+1\n5,0.7,x0*2\n",
    )
    df = load_pareto_csv(p)
    eq, c, l = pick_equation(df, "best_loss")
    assert eq == "x0+1" and c == 3 and l == pytest.approx(0.5)


def test_pick_complexity_le(tmp_path: Path):
    p = _write_csv(
        tmp_path,
        "Complexity,Loss,Equation\n1,2.0,x0\n3,0.5,x0+1\n5,0.3,x0*2\n",
    )
    df = load_pareto_csv(p)
    eq, c, _ = pick_equation(df, "complexity_le:3")
    # best loss with complexity <= 3 is the 3-complexity row
    assert c == 3 and eq == "x0+1"


def test_pick_accuracy_at(tmp_path: Path):
    p = _write_csv(
        tmp_path,
        "Complexity,Loss,Equation\n1,2.0,x0\n3,0.5,x0+1\n5,0.3,x0*2\n",
    )
    df = load_pareto_csv(p)
    eq, c, _ = pick_equation(df, "accuracy_at:0.6")
    # smallest complexity with loss <= 0.6 is the 3-complexity row
    assert c == 3 and eq == "x0+1"


def test_pick_row_index(tmp_path: Path):
    p = _write_csv(
        tmp_path,
        "Complexity,Loss,Equation\n1,2.0,x0\n3,0.5,x0+1\n",
    )
    df = load_pareto_csv(p)
    eq, _, _ = pick_equation(df, "row:0")
    assert eq == "x0"


def test_pick_no_eligible_complexity(tmp_path: Path):
    p = _write_csv(tmp_path, "Complexity,Loss,Equation\n10,0.1,x0\n")
    df = load_pareto_csv(p)
    with pytest.raises(ValueError, match="No equation in Pareto front"):
        pick_equation(df, "complexity_le:5")


def test_pick_no_eligible_accuracy(tmp_path: Path):
    p = _write_csv(tmp_path, "Complexity,Loss,Equation\n10,1.0,x0\n")
    df = load_pareto_csv(p)
    with pytest.raises(ValueError, match="No equation in Pareto front"):
        pick_equation(df, "accuracy_at:0.1")


# ---------------------------------------------------------------------------
# compile_equation: alias rename + fix substitution
# ---------------------------------------------------------------------------


def _identity_norm(nk: int = 5) -> NormalizationSpec:
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, nk)
    return identity(k_grid)


def test_compile_aliases_pysr_x0_to_named_param():
    """`x0` must rename to the first entry in `variables`."""
    norm = _identity_norm()
    ce = compile_equation(
        param_name="ns",
        raw_expression="x0 * 2 + x1",
        variables=["ns", "k"],
        fix=None,
        norm=norm,
        fiducial=0.97,
    )
    k = np.array([0.001, 0.005, 0.01, 0.015, 0.02])
    out = ce.evaluate(theta_i=0.97, k=k)
    # raw eq = ns_norm * 2 + k_norm; identity-normed → param_min=0, param_max=1
    # so ns_norm = 0.97, k_norm in [0,1]
    expected_norm = 0.97 * 2 + (k - DEFAULT_K_MIN) / (DEFAULT_K_MAX - DEFAULT_K_MIN)
    np.testing.assert_allclose(out, expected_norm, rtol=1e-12)


def test_compile_fix_substitutes_resolution():
    """3-input PySR equation with `x2=resolution` collapsed to a constant."""
    norm = _identity_norm()
    ce = compile_equation(
        param_name="bhfeedback",
        raw_expression="x0 + x1 - x2",
        variables=["bhfeedback", "k", "resolution"],
        fix={"resolution": 0.8},
        norm=norm,
        fiducial=0.05,
    )
    assert ce.extra_args == ("resolution",)
    k = np.array([0.005, 0.01, 0.015])
    out = ce.evaluate(theta_i=0.05, k=k)
    expected = 0.05 + (k - DEFAULT_K_MIN) / (DEFAULT_K_MAX - DEFAULT_K_MIN) - 0.8
    np.testing.assert_allclose(out, expected, rtol=1e-12)


def test_compile_rejects_unfixed_extra_variable():
    norm = _identity_norm()
    with pytest.raises(ValueError, match="must be assigned a constant in `fix:`"):
        compile_equation(
            param_name="bhfeedback",
            raw_expression="x0 + x1 - x2",
            variables=["bhfeedback", "k", "resolution"],
            fix=None,
            norm=norm,
            fiducial=0.05,
        )


def test_compile_rejects_param_name_not_in_variables():
    norm = _identity_norm()
    with pytest.raises(ValueError, match="must contain param_name"):
        compile_equation(
            param_name="ns",
            raw_expression="x0 + x1",
            variables=["bhfeedback", "k"],
            fix=None,
            norm=norm,
            fiducial=0.97,
        )


def test_compile_with_real_normalization_round_trips_to_p1d():
    """End-to-end: CompiledEquation.evaluate returns physical-units P_F."""
    nk = 35
    k_grid = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, nk)
    norm = NormalizationSpec(
        param_min=0.8,
        param_max=1.05,
        k_min=DEFAULT_K_MIN,
        k_max=DEFAULT_K_MAX,
        mean_flux=np.full(nk, 5.0),
        std_flux=np.full(nk, 2.0),
        k_grid=k_grid,
    )
    # Equation outputs flux_norm = 0 → P_F = mean_k = 5.0
    ce = compile_equation(
        param_name="ns",
        raw_expression="x0 - x0",
        variables=["ns", "k"],
        fix=None,
        norm=norm,
        fiducial=0.97,
    )
    out = ce.evaluate(theta_i=0.97, k=k_grid)
    np.testing.assert_allclose(out, 5.0)


# ---------------------------------------------------------------------------
# PySRModel — multiplicative / additive combine
# ---------------------------------------------------------------------------


def _make_eqn_cfg(
    *,
    combine: str,
    fiducial_p1d_path: Path,
    expression_per_param: dict[str, str],
    variables_per_param: dict[str, list[str]] | None = None,
) -> EqnConfig:
    from priya_forecast.parameters import get_param

    parameters = {}
    for pname, expr in expression_per_param.items():
        parameters[pname] = EqnParam(
            fiducial=get_param(pname).fid,
            expression=expr,
            variables=(variables_per_param or {}).get(pname, [pname, "k"]),
        )
    return EqnConfig(
        name="t",
        redshift=3.6,
        model="pysr",
        combine=combine,
        fiducial_p1d=str(fiducial_p1d_path),
        parameters=parameters,
    )


def _write_fiducial_p1d(tmp_path: Path, k: np.ndarray, p1d: np.ndarray) -> Path:
    p = tmp_path / "fid.npz"
    np.savez(p, k=k, p1d=p1d)
    return p


def test_pysr_model_multiplicative_at_fiducial_recovers_p_fid(tmp_path: Path):
    """At theta_fid, every per-param ratio equals 1 → predict == P_fid."""
    k_eboss = np.linspace(0.001, 0.02, 35)
    p_fid = 100.0 * np.exp(-50 * k_eboss)
    fid_path = _write_fiducial_p1d(tmp_path, k_eboss, p_fid)

    # One-parameter eqn cfg for simplicity. Equation: P_norm = 1 * x0 (linear)
    cfg = _make_eqn_cfg(
        combine="multiplicative",
        fiducial_p1d_path=fid_path,
        expression_per_param={"ns": "x0"},
    )
    model = PySRModel(eqn_cfg=cfg, k_grid=k_eboss, normalization_block={"mode": "identity"})

    from priya_forecast.parameters import fiducial_vector
    out = model.predict(np.array(fiducial_vector()), k_eboss, 3.6)
    np.testing.assert_allclose(out, p_fid, rtol=1e-10)


def test_pysr_model_additive_at_fiducial_recovers_p_fid(tmp_path: Path):
    k_eboss = np.linspace(0.001, 0.02, 35)
    p_fid = 1.5 + 0.5 * np.cos(50 * k_eboss)
    fid_path = _write_fiducial_p1d(tmp_path, k_eboss, p_fid)
    cfg = _make_eqn_cfg(
        combine="additive",
        fiducial_p1d_path=fid_path,
        expression_per_param={"ns": "x0", "hub": "x0*x1"},
    )
    model = PySRModel(eqn_cfg=cfg, k_grid=k_eboss, normalization_block={"mode": "identity"})
    from priya_forecast.parameters import fiducial_vector
    out = model.predict(np.array(fiducial_vector()), k_eboss, 3.6)
    np.testing.assert_allclose(out, p_fid, rtol=1e-10)


def test_pysr_model_multiplicative_perturbed_param_scales_correctly(tmp_path: Path):
    """A 10% bump in ns and equation `x0` should multiply P_fid by 1.1/0.97."""
    k_eboss = np.linspace(0.001, 0.02, 35)
    p_fid = 7.0 * np.ones_like(k_eboss)
    fid_path = _write_fiducial_p1d(tmp_path, k_eboss, p_fid)
    cfg = _make_eqn_cfg(
        combine="multiplicative",
        fiducial_p1d_path=fid_path,
        expression_per_param={"ns": "x0"},
    )
    model = PySRModel(eqn_cfg=cfg, k_grid=k_eboss, normalization_block={"mode": "identity"})
    from priya_forecast.parameters import fiducial_vector, PARAM_NAMES
    theta = np.array(fiducial_vector())
    theta[PARAM_NAMES.index("ns")] = 1.0  # was 0.983
    out = model.predict(theta, k_eboss, 3.6)
    np.testing.assert_allclose(out, 7.0 * (1.0 / 0.983), rtol=1e-10)


def test_pysr_model_z_mismatch_rejected(tmp_path: Path):
    k_eboss = np.linspace(0.001, 0.02, 35)
    fid_path = _write_fiducial_p1d(tmp_path, k_eboss, np.ones_like(k_eboss))
    cfg = _make_eqn_cfg(
        combine="multiplicative",
        fiducial_p1d_path=fid_path,
        expression_per_param={"ns": "x0"},
    )
    model = PySRModel(eqn_cfg=cfg, k_grid=k_eboss, normalization_block={"mode": "identity"})
    from priya_forecast.parameters import fiducial_vector
    with pytest.raises(ValueError, match="z=3.6"):
        model.predict(np.array(fiducial_vector()), k_eboss, 2.8)


def test_pysr_model_uses_real_pareto_csv_via_yaml(tmp_path: Path):
    """End-to-end: a YAML referencing a CSV picks an equation and evaluates."""
    csv_path = tmp_path / "hall.csv"
    csv_path.write_text(
        "Complexity,Loss,Equation\n1,9.0,x0\n3,1.0,x0*x1+1\n5,0.5,x0*x1+2\n"
    )
    k_eboss = np.linspace(0.001, 0.02, 35)
    fid_path = _write_fiducial_p1d(tmp_path, k_eboss, np.full_like(k_eboss, 4.0))

    from priya_forecast.parameters import get_param

    parameters = {
        "ns": EqnParam(
            fiducial=get_param("ns").fid,
            pareto_csv=str(csv_path),
            pick="best_loss",
            variables=["ns", "k"],
        )
    }
    cfg = EqnConfig(
        name="t",
        redshift=3.6,
        model="pysr",
        combine="multiplicative",
        fiducial_p1d=str(fid_path),
        parameters=parameters,
    )
    model = PySRModel(eqn_cfg=cfg, k_grid=k_eboss, normalization_block={"mode": "identity"})
    assert model.compiled["ns"].complexity == 5
    assert model.compiled["ns"].loss == pytest.approx(0.5)


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(
    a=st.floats(min_value=-2, max_value=2, allow_nan=False),
    b=st.floats(min_value=-2, max_value=2, allow_nan=False),
)
@settings(max_examples=20, deadline=None)
def test_property_compile_linear_equation_preserves_values(a: float, b: float):
    """For raw eq = a*x0 + b*x1 (identity normalization), evaluate matches by hand."""
    norm = _identity_norm()
    ce = compile_equation(
        param_name="ns",
        raw_expression=f"x0*{a} + x1*{b}",
        variables=["ns", "k"],
        fix=None,
        norm=norm,
        fiducial=0.97,
    )
    k = np.array([0.001, 0.005, 0.01, 0.015, 0.02])
    out = ce.evaluate(theta_i=0.5, k=k)
    k_norm = (k - DEFAULT_K_MIN) / (DEFAULT_K_MAX - DEFAULT_K_MIN)
    expected = a * 0.5 + b * k_norm
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)


@given(
    res=st.floats(min_value=0.1, max_value=1.0, allow_nan=False),
)
@settings(max_examples=20, deadline=None)
def test_property_fix_substitution_idempotent(res: float):
    """Fixing `resolution` produces the same result as rewriting the equation by hand."""
    norm = _identity_norm()
    ce = compile_equation(
        param_name="ns",
        raw_expression="x2 * x0 + x1",
        variables=["ns", "k", "resolution"],
        fix={"resolution": res},
        norm=norm,
        fiducial=0.97,
    )
    k = np.array([0.005, 0.01, 0.015])
    out = ce.evaluate(theta_i=0.5, k=k)
    k_norm = (k - DEFAULT_K_MIN) / (DEFAULT_K_MAX - DEFAULT_K_MIN)
    expected = res * 0.5 + k_norm
    np.testing.assert_allclose(out, expected, rtol=1e-10, atol=1e-12)
