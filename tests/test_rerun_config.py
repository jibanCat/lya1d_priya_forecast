from pathlib import Path
import pytest
from priya_forecast.rerun import RerunConfig, PRODUCTION_BUDGET, ALL_PARAMS


def test_quick_preset():
    c = RerunConfig.quick()
    assert c.params == list(ALL_PARAMS) and len(c.params) == 11
    assert c.zs == [3.6]
    assert set(c.arms) == {"value", "sobolev"}
    assert c.niterations < PRODUCTION_BUDGET["niterations"]     # reduced budget
    assert c.label == "quick"


def test_full_preset_matches_production_budget():
    c = RerunConfig.full()
    assert c.zs == [2.6, 3.6, 4.2]
    assert c.niterations == PRODUCTION_BUDGET["niterations"]
    assert c.populations == PRODUCTION_BUDGET["populations"]
    assert c.sobolev_lambda == PRODUCTION_BUDGET["sobolev_lambda"]


def test_run_dir_is_under_tutorial_root():
    c = RerunConfig.quick()
    assert c.run_dir == Path("results/tutorial_reruns/rerun_quick")


def test_run_dir_raises_if_inside_production(tmp_path):
    c = RerunConfig.quick()
    c.out_root = Path("results/paper_production_20260630_perz_sobolev_z2.6-4.2")
    with pytest.raises(ValueError, match="production"):
        _ = c.run_dir


def test_validate_rejects_bad_arm():
    c = RerunConfig.quick()
    c.arms = ["value", "nonsense"]
    with pytest.raises(ValueError):
        c.validate()
