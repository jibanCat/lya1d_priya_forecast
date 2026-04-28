"""Normalization round-trip for the student's PySR pipeline.

The student's `pysr_mf_given.py` / `mf_*.py` scripts train PySR on:
- inputs  : (param_norm, k_norm, [resolution, ...])
            with param_norm = (param - param_min) / (param_max - param_min) ∈ [0,1]
            and  k_norm     = (k - k_min) / (k_max - k_min) ∈ [0,1]
- output  : flux_norm = (P_F - mean_k) / std_k
            with mean_k and std_k computed *per k-bin* over the training set.

`mean_flux_low_<subset>.txt` and `std_flux_low_<subset>.txt` are the files the
student saves alongside each training run (1D vectors of length n_k).

This module exposes:

- `NormalizationSpec`  : the four arrays needed to round-trip an equation.
- `from_files()`       : load the student's `.txt` convention.
- `derive_from_gp()`   : recompute mean_k / std_k by sampling a `P1DModel`
                         along the chosen parameter at fixed-fiducial-rest,
                         used in dev when the student hasn't shipped files.
- `apply_forward()`    : (theta_i, k) -> (theta_norm, k_norm), in [0, 1].
- `apply_inverse()`    : flux_norm -> P_F.

All arrays are stored as `np.float64`. Forward and inverse are pure-numpy.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from priya_forecast.parameters import PARAM_NAMES, get_param

# Default training k-range used by `priya_pysr/pysr_mf_given.py` and the
# 1pvar HDF5 inputs — read off the eBOSS / emulator k-grid.
DEFAULT_K_MIN: float = 1.0e-3
DEFAULT_K_MAX: float = 2.0e-2


@dataclass
class NormalizationSpec:
    """Per-(parameter, z) normalization round-trip data.

    Attributes
    ----------
    param_min, param_max : float
        Forward range used for `param_norm = (param - lo) / (hi - lo)`. By
        default these come from the parameter's prior bounds, but can be
        overridden if the student trained on a narrower or wider range.
    k_min, k_max : float
        Forward range for `k_norm`. Defaults to the upstream training range.
    mean_flux : ndarray, shape (Nk,)
        Per-k mean used to denormalize the equation's output.
    std_flux : ndarray, shape (Nk,)
        Per-k standard deviation. Strictly positive.
    k_grid : ndarray, shape (Nk,)
        The k-grid mean_flux / std_flux were computed on. The model
        interpolates these onto the requested forecast k-grid before
        applying the inverse transform.
    """

    param_min: float
    param_max: float
    k_min: float
    k_max: float
    mean_flux: np.ndarray
    std_flux: np.ndarray
    k_grid: np.ndarray

    def __post_init__(self) -> None:
        if self.param_max <= self.param_min:
            raise ValueError(
                f"param_max ({self.param_max}) must exceed param_min ({self.param_min})."
            )
        if self.k_max <= self.k_min:
            raise ValueError(f"k_max ({self.k_max}) must exceed k_min ({self.k_min}).")
        self.mean_flux = np.asarray(self.mean_flux, dtype=float)
        self.std_flux = np.asarray(self.std_flux, dtype=float)
        self.k_grid = np.asarray(self.k_grid, dtype=float)
        if self.mean_flux.ndim != 1 or self.std_flux.ndim != 1:
            raise ValueError("mean_flux and std_flux must be 1D arrays.")
        if self.mean_flux.shape != self.std_flux.shape:
            raise ValueError(
                f"mean_flux shape {self.mean_flux.shape} must match "
                f"std_flux shape {self.std_flux.shape}."
            )
        if self.k_grid.shape != self.mean_flux.shape:
            raise ValueError(
                f"k_grid shape {self.k_grid.shape} must match "
                f"mean_flux shape {self.mean_flux.shape}."
            )
        if not np.all(self.std_flux > 0):
            raise ValueError("std_flux entries must all be > 0.")
        if not np.all(np.diff(self.k_grid) > 0):
            raise ValueError("k_grid must be strictly increasing.")

    # ------------------------------------------------------------------
    # Forward (forecast value -> normalized PySR input)
    # ------------------------------------------------------------------

    def normalize_param(self, value: float | np.ndarray) -> np.ndarray:
        return (np.asarray(value) - self.param_min) / (self.param_max - self.param_min)

    def normalize_k(self, k: np.ndarray) -> np.ndarray:
        return (np.asarray(k) - self.k_min) / (self.k_max - self.k_min)

    # ------------------------------------------------------------------
    # Inverse (PySR output -> P_F)
    # ------------------------------------------------------------------

    def denormalize_flux(self, flux_norm: np.ndarray, k_target: np.ndarray) -> np.ndarray:
        """Apply `P_F = flux_norm * std_k + mean_k` after interpolating to k_target."""
        flux_norm = np.asarray(flux_norm, dtype=float)
        k_target = np.asarray(k_target, dtype=float)
        if flux_norm.shape != k_target.shape:
            raise ValueError(
                f"flux_norm shape {flux_norm.shape} must match k_target shape {k_target.shape}."
            )
        mean = np.interp(k_target, self.k_grid, self.mean_flux)
        std = np.interp(k_target, self.k_grid, self.std_flux)
        return flux_norm * std + mean


# ---------------------------------------------------------------------------
# Loaders / derivers
# ---------------------------------------------------------------------------


def from_files(
    *,
    param_name: str,
    mean_flux_path: str | Path,
    std_flux_path: str | Path,
    k_grid: np.ndarray,
    param_min: float | None = None,
    param_max: float | None = None,
    k_min: float = DEFAULT_K_MIN,
    k_max: float = DEFAULT_K_MAX,
) -> NormalizationSpec:
    """Load the student's `mean_flux_*.txt` / `std_flux_*.txt` convention.

    `param_min` / `param_max` default to the parameter's prior bounds from
    `PARAMS_11D` — what `mf_*.py` uses unless the student deviates.
    """
    if param_name not in PARAM_NAMES:
        raise KeyError(f"Unknown parameter {param_name!r}.")
    if param_min is None or param_max is None:
        bounds = get_param(param_name).prior
        if param_min is None:
            param_min = bounds[0]
        if param_max is None:
            param_max = bounds[1]
    mean_flux = np.loadtxt(mean_flux_path)
    std_flux = np.loadtxt(std_flux_path)
    return NormalizationSpec(
        param_min=float(param_min),
        param_max=float(param_max),
        k_min=float(k_min),
        k_max=float(k_max),
        mean_flux=mean_flux,
        std_flux=std_flux,
        k_grid=np.asarray(k_grid, dtype=float),
    )


def derive_from_gp(
    *,
    gp_model,  # P1DModel; not type-hinted to dodge a circular import
    param_name: str,
    z: float,
    k_grid: np.ndarray,
    n_samples: int = 64,
    fiducial_theta: np.ndarray | None = None,
    seed: int = 0,
    param_min: float | None = None,
    param_max: float | None = None,
    k_min: float = DEFAULT_K_MIN,
    k_max: float = DEFAULT_K_MAX,
) -> NormalizationSpec:
    """Derive (mean_k, std_k) by sweeping the chosen param via the GP at fixed-fiducial-rest.

    Mirrors `pysr_mf_given.py`'s 1-parameter normalization step but uses the
    GP emulator instead of the multi-fidelity simulation HDF5s. Parameter
    bounds default to the prior. Reproducible via `seed`.
    """
    if param_name not in PARAM_NAMES:
        raise KeyError(f"Unknown parameter {param_name!r}.")
    if fiducial_theta is None:
        from priya_forecast.parameters import fiducial_vector

        fiducial_theta = np.asarray(fiducial_vector(), dtype=float)
    fiducial_theta = np.asarray(fiducial_theta, dtype=float)
    if fiducial_theta.shape != (11,):
        raise ValueError(f"fiducial_theta must be shape (11,), got {fiducial_theta.shape}.")

    if param_min is None or param_max is None:
        bounds = get_param(param_name).prior
        if param_min is None:
            param_min = bounds[0]
        if param_max is None:
            param_max = bounds[1]

    rng = np.random.default_rng(seed)
    samples = rng.uniform(param_min, param_max, size=n_samples)
    idx = PARAM_NAMES.index(param_name)

    k_grid = np.asarray(k_grid, dtype=float)
    flux = np.empty((n_samples, k_grid.size))
    for i, v in enumerate(samples):
        theta = fiducial_theta.copy()
        theta[idx] = v
        flux[i] = gp_model.predict(theta, k_grid, z)

    mean_flux = flux.mean(axis=0)
    std_flux = flux.std(axis=0, ddof=0)
    # Guard against pathological zeros — fall back to a tiny epsilon.
    std_flux = np.where(std_flux > 0, std_flux, 1e-30)

    return NormalizationSpec(
        param_min=float(param_min),
        param_max=float(param_max),
        k_min=float(k_min),
        k_max=float(k_max),
        mean_flux=mean_flux,
        std_flux=std_flux,
        k_grid=k_grid,
    )


def identity(k_grid: np.ndarray, param_min: float = 0.0, param_max: float = 1.0) -> NormalizationSpec:
    """Pass-through normalization: equation is already in physical units.

    Useful when an equation set was trained without normalization, or when
    a YAML supplies an `expression:` override the student wrote by hand in
    physical variables.
    """
    k_grid = np.asarray(k_grid, dtype=float)
    return NormalizationSpec(
        param_min=param_min,
        param_max=param_max,
        k_min=DEFAULT_K_MIN,
        k_max=DEFAULT_K_MAX,
        mean_flux=np.zeros_like(k_grid),
        std_flux=np.ones_like(k_grid),
        k_grid=k_grid,
    )
