"""Combine per-parameter PySR equations into one multi-D forward model.

Thin wrapper over `refit_taylor.AdditiveTaylorModel`. The default `additive`
combine uses `local_anchored` mode — the combine is anchored on the GP
prediction at fiducial θ and the per-D equations supply only the deviations
(found to forecast better than the student's `multi_d` formula). The
`multiplicative` and `joint` modes are reserved in the config schema but not
yet implemented.
"""

from __future__ import annotations

import numpy as np

from priya_forecast.models.base import P1DModel
from priya_forecast.refit_taylor import AdditiveTaylorModel
from priya_forecast.single_z.config import VALID_COMBINES as VALID_COMBINE_MODES


def build_combined_model(
    *,
    combine_mode: str,
    gp: P1DModel,
    fid: np.ndarray,
    refits: dict,
    k_grid: np.ndarray,
    z: float,
    global_norm=None,
    log_space: bool = False,
) -> P1DModel:
    """Construct the combined P_F(θ, k) model for the given combine mode.

    Parameters
    ----------
    combine_mode : one of `VALID_COMBINE_MODES`. Only `additive` is
        implemented; `multiplicative` / `joint` raise NotImplementedError.
    gp : the HF GP emulator — the combine anchor and the perfect-1D fallback
        source for any param whose refit is `None`.
    fid : (11,) fiducial parameter vector, canonical order.
    refits : dict mapping each of the 11 param names to a `Refit1DResult`
        or `None` (None → fall back to the GP's 1D slice for that param).
    k_grid, z : the grid and redshift the combine is built on.
    global_norm : `NormalizationSpec`; only the (unimplemented) `multi_d`
        path uses it, so pass `None` for `additive`/`local_anchored`.
    """
    if combine_mode == "additive":
        return AdditiveTaylorModel(
            gp=gp,
            fid=np.asarray(fid, dtype=float),
            refits=refits,
            global_norm=global_norm,
            k_grid=np.asarray(k_grid, dtype=float),
            z=float(z),
            mode="local_anchored",
            log_space=log_space,
        )
    if combine_mode in ("multiplicative", "joint"):
        raise NotImplementedError(
            f"combine mode {combine_mode!r} is not implemented yet; "
            f"only 'additive' is available."
        )
    raise ValueError(
        f"unknown combine mode {combine_mode!r}; "
        f"expected one of {VALID_COMBINE_MODES}."
    )
