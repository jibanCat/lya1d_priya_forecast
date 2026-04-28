"""eBOSS DR14 P1D data + covariance loader.

Thin wrapper around the vendored `BOSSData` class (originally from
`sbird/lya_emulator`). The vendored module finds its data files relative
to its own __file__, so we ship them at
`src/priya_forecast/_vendored/data/boss_dr14_data/`.

Public API:
- ``EBOSS_REDSHIFTS``: the 13 z-bins available in DR14, sorted increasing.
- ``load_eboss(z)``: returns ``(k_eboss, pf_eboss, cov_eboss)`` for one z-bin.
- ``bin_model_to_data(k_model, pf_model, k_eboss)``: bin a fine-grid model
  prediction onto the eBOSS k-grid (top-hat binning, not interpolation).

Units throughout: k in s/km, P_F dimensionless, redshift dimensionless.
"""

from __future__ import annotations

import numpy as np

from priya_forecast._vendored.lyman_data import BOSSData

EBOSS_REDSHIFTS: tuple[float, ...] = tuple(np.round(np.arange(2.2, 4.601, 0.2), 1).tolist())
"""The 13 z-bins eBOSS DR14 reports, sorted in increasing redshift."""


def _select_z(z: float) -> float:
    """Snap to the nearest eBOSS bin and assert it's within 0.01."""
    arr = np.asarray(EBOSS_REDSHIFTS)
    i = int(np.argmin(np.abs(arr - z)))
    if abs(arr[i] - z) > 0.01:
        raise ValueError(
            f"z={z} is not an eBOSS DR14 bin. Allowed bins: {EBOSS_REDSHIFTS}."
        )
    return float(arr[i])


def load_eboss(z: float = 3.6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load eBOSS DR14 P1D data + covariance for one z-bin.

    Parameters
    ----------
    z : float
        Redshift bin. Must be within 0.01 of one of `EBOSS_REDSHIFTS`.

    Returns
    -------
    k_eboss : ndarray, shape (35,)
        Wavenumber grid in s/km, strictly increasing.
    pf_eboss : ndarray, shape (35,)
        P_F(k) measurement for that z-bin (dimensionless).
    cov_eboss : ndarray, shape (35, 35)
        Symmetric positive-definite covariance for that z-bin. Combines the
        DR14 statistical + 8 systematic-uncertainty contributions, scaled by
        the published correlation matrix.
    """
    zsnap = _select_z(z)
    boss = BOSSData()  # default: DR14
    cov = boss.get_covar(zbin=zsnap)
    pf_full = boss.pf
    redshifts = boss.redshifts
    mask = np.abs(redshifts - zsnap) < 0.01
    pf = np.asarray(pf_full[mask])
    k = np.asarray(boss.kf[mask])
    # Sort by k (the upstream layout is already increasing, but be safe).
    order = np.argsort(k)
    return k[order], pf[order], np.asarray(cov)[np.ix_(order, order)]


def bin_model_to_data(
    k_model: np.ndarray,
    pf_model: np.ndarray,
    k_eboss: np.ndarray,
) -> np.ndarray:
    """Bin a fine-grid model prediction onto the eBOSS k-grid.

    Top-hat binning over half-bin-width windows centered on each `k_eboss[i]`.
    Half-widths come from neighbouring bin midpoints, with the boundary
    half-bins extrapolated symmetrically. If a window contains no model
    samples, falls back to nearest-neighbour to that bin centre.

    Parameters
    ----------
    k_model : ndarray, shape (Nm,)
        Strictly increasing model k-grid in s/km.
    pf_model : ndarray, shape (Nm,)
        Model P_F values at `k_model`.
    k_eboss : ndarray, shape (Nd,)
        Strictly increasing eBOSS k-grid (the data's k-bins).

    Returns
    -------
    ndarray, shape (Nd,)
        Top-hat-binned model prediction on the eBOSS k-grid.
    """
    k_model = np.asarray(k_model)
    pf_model = np.asarray(pf_model)
    k_eboss = np.asarray(k_eboss)
    if k_model.ndim != 1 or pf_model.ndim != 1:
        raise ValueError("k_model and pf_model must be 1D.")
    if k_model.shape != pf_model.shape:
        raise ValueError("k_model and pf_model must have matching shape.")
    if k_model.size < 2:
        raise ValueError("k_model must have at least two samples for binning.")
    if not np.all(np.diff(k_model) > 0):
        raise ValueError("k_model must be strictly increasing.")
    if not np.all(np.diff(k_eboss) > 0):
        raise ValueError("k_eboss must be strictly increasing.")

    # Build half-bin edges around each k_eboss centre.
    midpoints = 0.5 * (k_eboss[:-1] + k_eboss[1:])
    edges = np.empty(k_eboss.size + 1)
    edges[1:-1] = midpoints
    # Symmetric extrapolation at the boundaries.
    edges[0] = k_eboss[0] - 0.5 * (k_eboss[1] - k_eboss[0])
    edges[-1] = k_eboss[-1] + 0.5 * (k_eboss[-1] - k_eboss[-2])

    out = np.empty_like(k_eboss, dtype=float)
    for i in range(k_eboss.size):
        lo, hi = edges[i], edges[i + 1]
        in_bin = (k_model >= lo) & (k_model < hi)
        if np.any(in_bin):
            out[i] = float(np.mean(pf_model[in_bin]))
        else:
            # Nearest-neighbour fallback for bins with no samples.
            j = int(np.argmin(np.abs(k_model - k_eboss[i])))
            out[i] = float(pf_model[j])
    return out
