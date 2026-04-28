"""Unit + hypothesis tests for `priya_forecast.config`."""

from __future__ import annotations

import textwrap
from pathlib import Path

import pytest
import yaml
from hypothesis import given, strategies as st

from priya_forecast.config import (
    EqnConfig,
    EqnParam,
    KRange,
    RunConfig,
    _is_valid_pick,
    load_diagnostic_config,
    load_eqn_config,
    load_hpo_config,
    load_run_config,
)
from priya_forecast.parameters import PARAM_NAMES


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


# ---------------------------------------------------------------------------
# RunConfig
# ---------------------------------------------------------------------------


def test_default_yaml_loads_and_validates():
    cfg = load_run_config(Path(__file__).parent.parent / "configs" / "default.yaml")
    assert cfg.redshift == 3.6
    assert cfg.mode == "fisher"
    assert cfg.k_range.min == 0.001
    assert cfg.k_range.max == 0.02
    assert cfg.mcmc.walkers_per_dim == 4


def test_run_config_rejects_bad_mode(tmp_path: Path):
    p = _write(tmp_path, "bad.yaml", "redshift: 3.6\nmode: bayesian\n")
    with pytest.raises(ValueError, match="mode must be"):
        load_run_config(p)


def test_run_config_rejects_inverted_k_range(tmp_path: Path):
    p = _write(
        tmp_path,
        "bad.yaml",
        """
        redshift: 3.6
        k_range:
          min: 0.05
          max: 0.01
        """,
    )
    with pytest.raises(ValueError, match="k_range.max"):
        load_run_config(p)


def test_run_config_rejects_z_out_of_eboss_range(tmp_path: Path):
    p = _write(tmp_path, "bad.yaml", "redshift: 5.5\n")
    with pytest.raises(ValueError, match="outside eBOSS"):
        load_run_config(p)


# ---------------------------------------------------------------------------
# EqnConfig — student-facing PySR YAML
# ---------------------------------------------------------------------------


def _eqn_yaml_with_pysr_csv(csv_path: str) -> str:
    pdict = "\n".join(
        f"  {n}:\n"
        f"    pareto_csv: {csv_path}\n"
        f"    pick: best_loss\n"
        f"    fiducial: 1.0"
        for n in PARAM_NAMES
    )
    return (
        "name: t\n"
        "model: pysr\n"
        "redshift: 3.6\n"
        "combine: multiplicative\n"
        "fiducial_p1d: dummy.npz\n"
        "parameters:\n" + pdict + "\n"
    )


def test_eqn_config_pysr_csv_path_loads(tmp_path: Path):
    p = _write(tmp_path, "ok.yaml", _eqn_yaml_with_pysr_csv("/tmp/hall.csv"))
    cfg = load_eqn_config(p)
    assert cfg.model == "pysr"
    assert set(cfg.parameters) == set(PARAM_NAMES)
    assert cfg.parameters["ns"].pareto_csv == "/tmp/hall.csv"
    assert cfg.parameters["ns"].pick == "best_loss"


def test_eqn_config_gp_baseline_skips_param_validation(tmp_path: Path):
    p = _write(tmp_path, "gp.yaml", "name: gp\nmodel: gp\nredshift: 3.6\n")
    cfg = load_eqn_config(p)
    assert cfg.model == "gp"
    assert cfg.parameters == {}


def test_eqn_config_rejects_unknown_param(tmp_path: Path):
    body = (
        "name: t\nmodel: pysr\nredshift: 3.6\ncombine: multiplicative\n"
        "fiducial_p1d: dummy.npz\nparameters:\n"
        "  not_a_real_param:\n    pareto_csv: x.csv\n    pick: best_loss\n    fiducial: 1.0\n"
    )
    p = _write(tmp_path, "bad.yaml", body)
    with pytest.raises(ValueError, match="Unknown parameter names"):
        load_eqn_config(p)


def test_eqn_config_rejects_eqnparam_with_neither_pareto_nor_expression(tmp_path: Path):
    body = (
        "name: t\nmodel: pysr\nredshift: 3.6\ncombine: multiplicative\n"
        "fiducial_p1d: dummy.npz\nparameters:\n"
        "  ns:\n    fiducial: 0.97\n"
    )
    p = _write(tmp_path, "bad.yaml", body)
    with pytest.raises(ValueError, match="must set either"):
        load_eqn_config(p)


def test_eqn_config_rejects_invalid_pick(tmp_path: Path):
    body = (
        "name: t\nmodel: pysr\nredshift: 3.6\ncombine: multiplicative\n"
        "fiducial_p1d: dummy.npz\nparameters:\n"
        "  ns:\n    pareto_csv: hall.csv\n    pick: bogus_rule\n    fiducial: 0.97\n"
    )
    p = _write(tmp_path, "bad.yaml", body)
    with pytest.raises(ValueError, match="invalid `pick`"):
        load_eqn_config(p)


def test_eqn_config_joint_requires_expression(tmp_path: Path):
    body = (
        "name: t\nmodel: pysr\nredshift: 3.6\ncombine: joint\n"
        "fiducial_p1d: dummy.npz\nparameters: {}\n"
    )
    p = _write(tmp_path, "bad.yaml", body)
    with pytest.raises(ValueError, match="joint_expression"):
        load_eqn_config(p)


def test_eqn_config_multiplicative_requires_fiducial_p1d(tmp_path: Path):
    body = (
        "name: t\nmodel: pysr\nredshift: 3.6\ncombine: multiplicative\n"
        "parameters:\n"
        "  ns:\n    pareto_csv: hall.csv\n    pick: best_loss\n    fiducial: 0.97\n"
    )
    p = _write(tmp_path, "bad.yaml", body)
    with pytest.raises(ValueError, match="fiducial_p1d"):
        load_eqn_config(p)


def test_shipped_pysr_v1_yaml_loads(tmp_path: Path):
    """Shipped example config must round-trip — students will start from it."""
    cfg = load_eqn_config(Path(__file__).parent.parent / "configs" / "eqns" / "pysr_v1.yaml")
    assert cfg.model == "pysr"
    assert cfg.combine == "multiplicative"
    assert set(cfg.parameters) == set(PARAM_NAMES)


def test_shipped_gp_baseline_yaml_loads():
    cfg = load_eqn_config(Path(__file__).parent.parent / "configs" / "eqns" / "gp_baseline.yaml")
    assert cfg.model == "gp"


def test_shipped_diagnostic_yaml_loads():
    cfg = load_diagnostic_config(Path(__file__).parent.parent / "configs" / "diagnostic.yaml")
    assert cfg.benchmark_z == 3.6
    assert 3.6 in cfg.redshifts


def test_shipped_hpo_quick_yaml_loads():
    cfg = load_hpo_config(Path(__file__).parent.parent / "configs" / "hpo" / "quick.yaml")
    assert cfg.strategy == "random"
    assert cfg.n_trials == 6


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(
    rule=st.sampled_from(["best_loss"])
    | st.builds(lambda n: f"complexity_le:{n}", st.integers(min_value=1, max_value=50))
    | st.builds(lambda x: f"accuracy_at:{x}", st.floats(min_value=1e-9, max_value=1.0))
    | st.builds(lambda i: f"row:{i}", st.integers(min_value=0, max_value=100)),
)
def test_property_valid_pick_rules_accepted(rule: str):
    assert _is_valid_pick(rule)


@given(rule=st.text(min_size=1, max_size=20))
def test_property_random_strings_mostly_rejected(rule: str):
    """Random strings without a known prefix or value are rejected."""
    if rule == "best_loss":
        return  # not interesting for this test
    if any(rule.startswith(p) and len(rule) > len(p) for p in
           ("complexity_le:", "accuracy_at:", "row:")):
        return  # an accidental valid form — also fine
    assert not _is_valid_pick(rule)


@given(
    z=st.floats(min_value=2.2, max_value=4.6, allow_nan=False),
    cov_scale=st.floats(min_value=0.01, max_value=100, allow_nan=False),
    kmin=st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False),
    kmax_mult=st.floats(min_value=1.1, max_value=100, allow_nan=False),
)
def test_property_well_formed_run_config_validates(
    z: float, cov_scale: float, kmin: float, kmax_mult: float
):
    cfg = RunConfig(
        redshift=z,
        cov_scale=cov_scale,
        k_range=KRange(min=kmin, max=kmin * kmax_mult),
    )
    cfg.validate()
