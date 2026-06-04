"""Combine per-parameter 4-input PySR equations into one multi-z model.

Thin wrapper over `refit_taylor.MultiZAdditiveTaylorModel` (always
local_anchored). Mirrors single_z/combine.py.
"""
from __future__ import annotations

import numpy as np

from priya_forecast.models.base import P1DModel
from priya_forecast.refit_taylor import MultiZAdditiveTaylorModel
from priya_forecast.single_z.config import VALID_COMBINES as VALID_COMBINE_MODES


def build_combined_model_multiz(
    *,
    combine_mode: str,
    gp: P1DModel,
    fid: np.ndarray,
    refits: dict,
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    log_space: bool = False,
) -> P1DModel:
    """Construct the multi-z combined P_F(θ, k, z) model.

    Only `additive` is implemented; `multiplicative`/`joint` raise
    NotImplementedError (mirrors the single-z combine).
    """
    if combine_mode == "additive":
        return MultiZAdditiveTaylorModel(
            gp=gp, fid=np.asarray(fid, dtype=float), refits=refits,
            k_grid=np.asarray(k_grid, dtype=float),
            z_grid=np.asarray(z_grid, dtype=float), log_space=log_space,
        )
    if combine_mode in ("multiplicative", "joint"):
        raise NotImplementedError(
            f"combine mode {combine_mode!r} is not implemented; "
            f"only 'additive' is available."
        )
    raise ValueError(
        f"unknown combine mode {combine_mode!r}; expected one of {VALID_COMBINE_MODES}."
    )
