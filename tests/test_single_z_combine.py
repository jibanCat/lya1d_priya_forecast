"""Unit tests for `priya_forecast.single_z.combine`.

Pure tests — `MockGPModel` stands in for the emulator, all 11 refits are
`None` (so the combine falls back to GP 1D slices). At θ=fid every per-D
deviation is zero, so the `local_anchored` combine must return the GP anchor
exactly.
"""

from __future__ import annotations

import numpy as np
import pytest

from priya_forecast.models.gp_model import MockGPModel
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector
from priya_forecast.refit_taylor import AdditiveTaylorModel
from priya_forecast.single_z.combine import build_combined_model


def _fid() -> np.ndarray:
    return np.asarray(fiducial_vector(), dtype=float)


def _none_refits() -> dict:
    return {name: None for name in PARAM_NAMES}


def test_build_combined_model_additive_returns_local_anchored():
    gp = MockGPModel()
    k = np.linspace(0.001, 0.04, 20)
    model = build_combined_model(
        combine_mode="additive", gp=gp, fid=_fid(), refits=_none_refits(),
        k_grid=k, z=3.6,
    )
    assert isinstance(model, AdditiveTaylorModel)
    assert model.mode == "local_anchored"


def test_build_combined_model_additive_anchor_identity():
    """At θ=fid the additive/local_anchored combine == the GP anchor."""
    gp = MockGPModel()
    k = np.linspace(0.001, 0.04, 20)
    fid = _fid()
    model = build_combined_model(
        combine_mode="additive", gp=gp, fid=fid, refits=_none_refits(),
        k_grid=k, z=3.6,
    )
    np.testing.assert_allclose(
        model.predict(fid, k, 3.6), gp.predict(fid, k, 3.6), rtol=1e-9,
    )


@pytest.mark.parametrize("mode", ["multiplicative", "joint"])
def test_build_combined_model_unimplemented_modes(mode):
    with pytest.raises(NotImplementedError, match="not implemented"):
        build_combined_model(
            combine_mode=mode, gp=MockGPModel(), fid=_fid(),
            refits=_none_refits(), k_grid=np.linspace(0.001, 0.04, 10), z=3.6,
        )


def test_build_combined_model_unknown_mode():
    with pytest.raises(ValueError, match="unknown combine mode"):
        build_combined_model(
            combine_mode="bogus", gp=MockGPModel(), fid=_fid(),
            refits=_none_refits(), k_grid=np.linspace(0.001, 0.04, 10), z=3.6,
        )
