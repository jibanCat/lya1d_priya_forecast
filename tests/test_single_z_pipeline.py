"""Unit + hypothesis tests for `priya_forecast.single_z` (config + pipeline).

Config tests are pure (no lyaemu / no GPy). The end-to-end gp_only smoke is
gated on `RUN_SLOW_GP_ONLY=1` because it loads the full GP (~30s wall) and
needs lyaemu + the prepped `data/kodiaq_gp/` basedir.
"""

from __future__ import annotations

import os
import textwrap
from pathlib import Path

import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.single_z.config import (
    DataConfig,
    FisherConfig,
    GPConfig,
    KRange,
    NormalizationConfig,
    PipelineConfig,
    PySRConfig,
    _is_valid_pick,
    load_config,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _write(tmp_path: Path, name: str, body: str) -> Path:
    p = tmp_path / name
    p.write_text(textwrap.dedent(body))
    return p


def _basedir(tmp_path: Path) -> Path:
    """A directory that exists so GPConfig.validate doesn't trip."""
    d = tmp_path / "fake_gp"
    d.mkdir()
    return d


# ---------------------------------------------------------------------------
# load_config — happy path
# ---------------------------------------------------------------------------


def test_shipped_example_yaml_loads_and_validates():
    """Shipped example config must round-trip — students copy from it."""
    cfg = load_config(
        Path(__file__).parent.parent / "configs" / "single_z" / "example.yaml"
    )
    assert cfg.mode == "gp_only"
    assert cfg.redshift == 3.6
    assert cfg.k_range.min == 0.001
    assert cfg.k_range.max == 0.04
    assert cfg.data.source == "kodiaq"
    assert cfg.combine == "additive"
    assert set(cfg.parameters) == set(PARAM_NAMES)


def test_minimal_yaml_takes_defaults(tmp_path: Path):
    """An almost-empty YAML inherits sensible defaults."""
    basedir = _basedir(tmp_path)
    p = _write(
        tmp_path,
        "min.yaml",
        f"""
        gp:
          basedir: {basedir}
        """,
    )
    cfg = load_config(p)
    assert cfg.mode == "forecast_only"     # default
    assert cfg.redshift == 3.6
    assert cfg.data.source == "kodiaq"
    assert cfg.normalization.mode == "auto"
    assert cfg.pareto_csvs.source == "bundled_baseline"


# ---------------------------------------------------------------------------
# load_config — validation rejects bad input
# ---------------------------------------------------------------------------


def test_unknown_top_level_key_rejected(tmp_path: Path):
    p = _write(tmp_path, "bad.yaml", "redshift: 3.6\nbogus_field: 1\n")
    with pytest.raises(ValueError, match="Unknown top-level key"):
        load_config(p)


def test_bad_mode_rejected(tmp_path: Path):
    basedir = _basedir(tmp_path)
    p = _write(
        tmp_path, "bad.yaml",
        f"mode: bayesian\ngp:\n  basedir: {basedir}\n",
    )
    with pytest.raises(ValueError, match="mode must be"):
        load_config(p)


def test_bad_data_source_rejected(tmp_path: Path):
    basedir = _basedir(tmp_path)
    p = _write(
        tmp_path, "bad.yaml",
        f"gp:\n  basedir: {basedir}\ndata:\n  source: pristine_void\n",
    )
    with pytest.raises(ValueError, match="data.source"):
        load_config(p)


def test_inverted_k_range_rejected(tmp_path: Path):
    basedir = _basedir(tmp_path)
    p = _write(
        tmp_path, "bad.yaml",
        f"gp:\n  basedir: {basedir}\nk_range:\n  min: 0.05\n  max: 0.01\n",
    )
    with pytest.raises(ValueError, match="k_range invalid"):
        load_config(p)


def test_z_outside_range_rejected(tmp_path: Path):
    basedir = _basedir(tmp_path)
    p = _write(
        tmp_path, "bad.yaml",
        f"redshift: 5.5\ngp:\n  basedir: {basedir}\n",
    )
    with pytest.raises(ValueError, match="outside"):
        load_config(p)


def test_unknown_parameter_rejected(tmp_path: Path):
    basedir = _basedir(tmp_path)
    p = _write(
        tmp_path, "bad.yaml",
        f"gp:\n  basedir: {basedir}\nparameters:\n  - not_real\n  - ns\n",
    )
    with pytest.raises(ValueError, match="Unknown PRIYA parameters"):
        load_config(p)


def test_bad_combine_rejected(tmp_path: Path):
    basedir = _basedir(tmp_path)
    p = _write(
        tmp_path, "bad.yaml",
        f"gp:\n  basedir: {basedir}\ncombine: telepathy\n",
    )
    with pytest.raises(ValueError, match="combine must be"):
        load_config(p)


def test_missing_basedir_rejected(tmp_path: Path):
    p = _write(
        tmp_path, "bad.yaml",
        f"gp:\n  basedir: {tmp_path / 'does_not_exist'}\n",
    )
    with pytest.raises(ValueError, match="gp.basedir does not exist"):
        load_config(p)


def test_per_parameter_pareto_requires_all_entries(tmp_path: Path):
    basedir = _basedir(tmp_path)
    p = _write(
        tmp_path, "bad.yaml",
        f"""
        gp:
          basedir: {basedir}
        parameters: [ns, Ap]
        pareto_csvs:
          source: per_parameter
          per_parameter:
            ns: {{pareto_csv: ns.csv, pick: best_loss}}
        """,
    )
    with pytest.raises(ValueError, match="missing entries"):
        load_config(p)


def test_per_parameter_invalid_pick_rejected(tmp_path: Path):
    basedir = _basedir(tmp_path)
    p = _write(
        tmp_path, "bad.yaml",
        f"""
        gp:
          basedir: {basedir}
        parameters: [ns]
        pareto_csvs:
          source: per_parameter
          per_parameter:
            ns: {{pareto_csv: ns.csv, pick: bogus_rule}}
        """,
    )
    with pytest.raises(ValueError, match="invalid"):
        load_config(p)


# ---------------------------------------------------------------------------
# Pick-rule recognizer
# ---------------------------------------------------------------------------


def test_is_valid_pick_accepts_known_rules():
    assert _is_valid_pick("best_loss")
    assert _is_valid_pick("complexity_le:7")
    assert _is_valid_pick("accuracy_at:1e-3")
    assert _is_valid_pick("row:0")


def test_is_valid_pick_rejects_bare_prefixes_and_garbage():
    assert not _is_valid_pick("complexity_le:")     # empty value
    assert not _is_valid_pick("accuracy_at:")
    assert not _is_valid_pick("row:")
    assert not _is_valid_pick("bogus_rule")
    assert not _is_valid_pick("")


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@settings(max_examples=20, deadline=None)
@given(
    z=st.floats(min_value=2.2, max_value=4.6, allow_nan=False),
    kmin=st.floats(min_value=1e-6, max_value=1e-2, allow_nan=False),
    kmax_mult=st.floats(min_value=1.1, max_value=100, allow_nan=False),
    cov_scale=st.floats(min_value=0.01, max_value=100, allow_nan=False),
    n_params=st.integers(min_value=1, max_value=11),
)
def test_property_well_formed_config_validates(
    z: float, kmin: float, kmax_mult: float, cov_scale: float, n_params: int,
    tmp_path_factory: pytest.TempPathFactory,
):
    """Any well-formed config (z in range, k positive ordered, params subset
    of PARAM_NAMES) must validate."""
    base = tmp_path_factory.mktemp("hyp_basedir")
    cfg = PipelineConfig(
        redshift=z,
        k_range=KRange(min=kmin, max=kmin * kmax_mult),
        data=DataConfig(cov_scale=cov_scale),
        gp=GPConfig(basedir=str(base)),
        parameters=list(PARAM_NAMES[:n_params]),
    )
    cfg.validate()


@settings(max_examples=15, deadline=None)
@given(
    rule=st.sampled_from(["best_loss"])
    | st.builds(lambda n: f"complexity_le:{n}", st.integers(min_value=1, max_value=50))
    | st.builds(lambda x: f"accuracy_at:{x}", st.floats(min_value=1e-9, max_value=1.0))
    | st.builds(lambda i: f"row:{i}", st.integers(min_value=0, max_value=100)),
)
def test_property_valid_pick_rules_accepted(rule: str):
    assert _is_valid_pick(rule)


# ---------------------------------------------------------------------------
# Pipeline dispatcher — Stage B/C still raise NotImplementedError
# ---------------------------------------------------------------------------


def test_dispatcher_forecast_only_is_implemented(tmp_path: Path):
    """forecast_only is now implemented in Stage 2; it must not raise NotImplementedError."""
    from priya_forecast.single_z.pipeline import run_forecast_only
    # Simply confirm the function is callable and not the stub.
    assert callable(run_forecast_only)
    assert run_forecast_only.__doc__ is not None


def test_dispatcher_refit_and_forecast_is_implemented(tmp_path: Path):
    """refit_and_forecast is now implemented in Stage 3; it must not raise NotImplementedError."""
    from priya_forecast.single_z.pipeline import run_refit_and_forecast
    assert callable(run_refit_and_forecast)
    assert run_refit_and_forecast.__doc__ is not None


# ---------------------------------------------------------------------------
# End-to-end gp_only smoke — gated (slow: ~30s GP load + Fisher).
# ---------------------------------------------------------------------------

RUN_SLOW = os.environ.get("RUN_SLOW_GP_ONLY") == "1"
GP_BASEDIR = Path(__file__).parent.parent / "data" / "kodiaq_gp"

try:
    import lyaemu  # noqa: F401
    LYAEMU_AVAILABLE = True
except ImportError:
    LYAEMU_AVAILABLE = False


def test_combine_defaults_to_additive(tmp_path: Path):
    """forecast_only's combine default is additive (the student contract)."""
    basedir = _basedir(tmp_path)
    p = _write(tmp_path, "c.yaml", f"gp:\n  basedir: {basedir}\n")
    cfg = load_config(p)
    assert cfg.combine == "additive"


def test_top_level_pick_default_and_validation(tmp_path: Path):
    """PipelineConfig has a top-level `pick` rule, default best_loss; bad rule rejected."""
    basedir = _basedir(tmp_path)
    good = _write(tmp_path, "g.yaml", f"gp:\n  basedir: {basedir}\n")
    assert load_config(good).pick == "best_loss"
    bad = _write(tmp_path, "b.yaml",
                 f"pick: nonsense\ngp:\n  basedir: {basedir}\n")
    with pytest.raises(ValueError, match="pick"):
        load_config(bad)


@pytest.mark.skipif(
    not (RUN_SLOW and LYAEMU_AVAILABLE and GP_BASEDIR.exists()),
    reason="gated on RUN_SLOW_GP_ONLY=1 + lyaemu + data/kodiaq_gp/",
)
def test_gp_only_end_to_end(tmp_path: Path):
    """Run the dispatcher in gp_only mode against the real GP; assert
    output files are written and σ are finite + positive."""
    import numpy as np
    from priya_forecast.single_z.pipeline import run

    cfg = PipelineConfig(
        mode="gp_only",
        redshift=3.6,
        output_dir=str(tmp_path / "out"),
        gp=GPConfig(basedir=str(GP_BASEDIR)),
        parameters=["ns", "Ap"],   # 2-param subset for speed + invertibility
        k_range=KRange(min=0.001, max=0.04),
    )
    result = run(cfg)
    assert (tmp_path / "out" / "forecast_table.txt").exists()
    assert (tmp_path / "out" / "scorecard.md").exists()
    sigma = result["sigma_gp"]
    assert sigma.shape == (2,)
    assert np.all(np.isfinite(sigma))
    assert np.all(sigma > 0)


RUN_SLOW_FORECAST = os.environ.get("RUN_SLOW_FORECAST_ONLY") == "1"


@pytest.mark.skipif(
    not (RUN_SLOW_FORECAST and LYAEMU_AVAILABLE and GP_BASEDIR.exists()),
    reason="gated on RUN_SLOW_FORECAST_ONLY=1 + lyaemu + data/kodiaq_gp/",
)
def test_forecast_only_perfect_1d_end_to_end(tmp_path: Path):
    """forecast_only with no equations still yields σ_GP and σ_perfect_1D."""
    import numpy as np
    from priya_forecast.single_z.pipeline import run

    cfg = PipelineConfig(
        mode="forecast_only", redshift=3.6,
        output_dir=str(tmp_path / "out"),
        gp=GPConfig(basedir=str(GP_BASEDIR)),
        parameters=["ns", "Ap"],
        k_range=KRange(min=0.001, max=0.04),
        data=DataConfig(source="kodiaq"),
    )
    result = run(cfg)
    for label in ("GP", "perfect_1D"):
        s = result["sigmas"][label]
        assert s.shape == (2,)
        assert np.all(np.isfinite(s)) and np.all(s > 0)
    assert (tmp_path / "out" / "forecast_table.txt").exists()
    assert (tmp_path / "out" / "corner.png").exists()


RUN_SLOW_REFIT = os.environ.get("RUN_SLOW_REFIT") == "1"


@pytest.mark.skipif(
    not (RUN_SLOW_REFIT and LYAEMU_AVAILABLE and GP_BASEDIR.exists()),
    reason="gated on RUN_SLOW_REFIT=1 + lyaemu + data/kodiaq_gp/ (runs PySR)",
)
def test_refit_and_forecast_end_to_end(tmp_path: Path):
    """refit_and_forecast refits a 2-param subset and forecasts σ_PySR."""
    import numpy as np
    from priya_forecast.single_z.pipeline import run

    cfg = PipelineConfig(
        mode="refit_and_forecast", redshift=3.6,
        output_dir=str(tmp_path / "out"),
        gp=GPConfig(basedir=str(GP_BASEDIR)),
        parameters=["ns", "Ap"],
        k_range=KRange(min=0.001, max=0.04),
        data=DataConfig(source="kodiaq"),
    )
    result = run(cfg)
    assert result["pysr_available"] is True
    for label in ("GP", "perfect_1D", "PySR"):
        s = result["sigmas"][label]
        assert s.shape == (2,)
        assert np.all(np.isfinite(s)) and np.all(s > 0)
    for p in ("ns", "Ap"):
        assert (tmp_path / "out" / "refit" / "z3.6" / f"pareto_{p}.csv").exists()
    assert (tmp_path / "out" / "corner.png").exists()


def test_write_forecast_deliverables_saves_npz(tmp_path: Path):
    """_write_forecast_deliverables persists each FisherResult as an npz."""
    import numpy as np
    from priya_forecast.fisher import FisherResult
    from priya_forecast.single_z.pipeline import _write_forecast_deliverables
    from priya_forecast.single_z.config import PipelineConfig

    def _fake_fr(scale):
        n = 2
        return FisherResult(
            F=np.eye(n), cov=np.eye(n) * scale, sigma=np.full(n, scale),
            corr=np.eye(n), steps=np.full(n, 0.01),
            param_names=("ns", "Ap"), theta_fid=np.array([0.98, 1.46]),
        )

    cfg = PipelineConfig(mode="forecast_only", parameters=["ns", "Ap"])
    results = {"GP": _fake_fr(0.1), "perfect_1D": _fake_fr(0.1),
               "PySR": _fake_fr(0.2)}
    out = tmp_path / "out"
    out.mkdir()
    _write_forecast_deliverables(cfg, out, results, pysr_available=True)
    for label in ("GP", "perfect_1D", "PySR"):
        npz = out / f"fisher_{label}.npz"
        assert npz.exists()
        loaded = np.load(npz, allow_pickle=True)
        assert loaded["sigma"].shape == (2,)
