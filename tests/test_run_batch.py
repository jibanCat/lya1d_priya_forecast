"""Unit tests for scripts/run_batch.py."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_batch", Path(__file__).parent.parent / "scripts" / "run_batch.py",
)


def _load():
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def test_derive_z_configs(tmp_path):
    """derive_z_configs fans a base config over the 13 z-bins."""
    mod = _load()
    from priya_forecast.single_z.config import PipelineConfig

    base = PipelineConfig(mode="forecast_only", redshift=3.6,
                          output_dir=str(tmp_path / "run"))
    derived = mod.derive_z_configs(base)
    assert len(derived) == 13
    redshifts = sorted(c.redshift for c in derived)
    assert redshifts[0] == pytest.approx(2.2)
    assert redshifts[-1] == pytest.approx(4.6)
    for c in derived:
        assert c.output_dir.endswith(f"z{c.redshift}")
