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


def test_presets_accept_field_overrides():
    # regression: quick()/full() must let callers override preset fields
    # without a "multiple values for keyword argument" TypeError.
    c = RerunConfig.quick(label="verify", params=["ns"], niterations=12)
    assert c.label == "verify" and c.params == ["ns"] and c.niterations == 12
    assert c.zs == [3.6]                       # untouched preset default preserved
    f = RerunConfig.full(label="myfull")
    assert f.label == "myfull" and f.zs == [2.6, 3.6, 4.2]


# ---- fiducial / prior overrides are checked before any fit starts -----------------

def test_validate_rejects_a_range_wider_than_the_emulators():
    # Ap's range (1.2, 2.6) is the emulator's training range; outside it there is no
    # emulator, and the run used to die later with an empty AssertionError.
    cfg = RerunConfig.quick(prior_overrides={"Ap": (0.5, 4.0)})
    with pytest.raises(ValueError, match="emulator"):
        cfg.validate()


def test_validate_accepts_a_narrower_range():
    RerunConfig.quick(prior_overrides={"Ap": (1.3, 2.2)}).validate()


def test_validate_rejects_a_fiducial_outside_its_range():
    with pytest.raises(ValueError, match="ns"):
        RerunConfig.quick(fiducial_overrides={"ns": 1.2}).validate()
    with pytest.raises(ValueError, match="Ap"):       # default fiducial 1.46 is now outside
        RerunConfig.quick(prior_overrides={"Ap": (1.6, 2.2)}).validate()


def test_validate_rejects_an_unknown_override_name():
    with pytest.raises(ValueError, match="nope"):
        RerunConfig.quick(fiducial_overrides={"nope": 1.0}).validate()
