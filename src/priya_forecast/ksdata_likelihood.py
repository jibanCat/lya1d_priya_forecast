"""Multi-z Gaussian likelihood with the real KODIAQ-SQUAD covariance.

Replaces the synthetic 5%-of-P_F diagonal covariance with the
production paper's `KSData(conservative=True)` (Karacayli et al. 2021)
182×182 cross-(z, k) covariance.

The KSData layout is z-major: 14 z bins × 13 k bins per z = 182 rows,
ordered (z=2.0; k₀…k₁₂), (z=2.2; k₀…k₁₂), …, (z=4.6; k₀…k₁₂). For our
forecast we filter to z ∈ [z_min, z_max] AND k ≤ k_max, extract the
matched cov sub-matrix, and pre-Cholesky for fast log-likelihood
evaluation.

`model_at(theta)` predicts P_F at each unique z in the kept set (one GP
call per z, predictions match the kept k-bins for that z), concatenated
in the kept-row order so the chi² residual aligns with the cov sub-matrix.

The cross-z structure of the KSData covariance is **NOT** block-diagonal
— so the per-z Fisher aggregation in `combine_fisher_phys_arrays` is
incorrect for KSData. Use a single `fisher_matrix(...)` call with a
`KSDataLikelihood` instance instead.

Tests in `tests/test_ksdata_likelihood.py` validate the layout and
masking. Cluster-only: requires `lyaemu` import.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from priya_forecast.likelihood import LikelihoodInputs
from priya_forecast.models.base import P1DModel
from priya_forecast.parameters import fiducial_vector


@dataclass
class KSDataInputs:
    """Frozen inputs for a multi-z KSData likelihood."""

    kept_z: np.ndarray              # (n_kept,) — z value per kept row
    kept_k: np.ndarray              # (n_kept,) — k value per kept row
    z_blocks: list[tuple[float, slice]]   # per-unique-z (value, row slice)
    d: np.ndarray                   # (n_kept,) — "data" vector (mock=gp or actual)
    cov: np.ndarray                 # (n_kept, n_kept) — sub-cov from KSData
    cov_chol: np.ndarray            # (n_kept, n_kept) lower-triangular Cholesky
    log_norm: float


def _build_z_blocks(kept_z: np.ndarray) -> list[tuple[float, slice]]:
    """Find consecutive runs of equal z in the kept_z array.

    KSData is z-major so kept rows are already grouped by z. Returns
    `[(z_value, slice(start, stop)), ...]` covering all rows.
    """
    blocks: list[tuple[float, slice]] = []
    n = len(kept_z)
    if n == 0:
        return blocks
    i = 0
    while i < n:
        z_i = float(kept_z[i])
        j = i + 1
        while j < n and np.isclose(kept_z[j], z_i, atol=1e-6):
            j += 1
        blocks.append((z_i, slice(i, j)))
        i = j
    return blocks


class KSDataLikelihood:
    """Real KODIAQ-SQUAD multi-z Gaussian likelihood.

    Parameters
    ----------
    model : P1DModel
        Forward model with `predict(theta, k, z)`.
    z_min, z_max : float
        Keep rows with `z_min ≤ z ≤ z_max`. Default kodiaq production
        range 2.6 → 4.2.
    k_max : float
        Discard k bins above this (default 0.064 — production range).
    cov_scale : float
        Multiplies the KSData covariance (sanity-check knob).
    mock_data : {"gp", "kodiaq"}
        "gp" → use the model at fid as the "data" (forecast convention).
        "kodiaq" → use the actual KODIAQ-SQUAD measurement.
    theta_fid : ndarray, shape (11,) | None
        Fiducial point for mock="gp".
    conservative : bool
        Passes through to `KSData(conservative=...)`. Default True
        (matches production paper's chains).
    """

    def __init__(
        self,
        *,
        model: P1DModel,
        z_min: float = 2.6,
        z_max: float = 4.2,
        k_max: float = 0.064,
        cov_scale: float = 1.0,
        mock_data: str = "gp",
        theta_fid: np.ndarray | None = None,
        conservative: bool = True,
    ) -> None:
        try:
            from lyaemu.lyman_data import KSData  # type: ignore[import-not-found]
        except ImportError as e:
            raise ImportError(
                "KSDataLikelihood requires `lyaemu` (sbird/lya_emulator). "
                "Install it and add to PYTHONPATH."
            ) from e

        ks = KSData(scale_covar=cov_scale, conservative=conservative)
        all_z = np.asarray(ks.redshifts, dtype=float)
        all_k = np.asarray(ks.kf, dtype=float)
        all_pf = np.asarray(ks.pf, dtype=float)
        all_cov = np.asarray(ks.covar, dtype=float)

        kept_mask = (
            (all_z >= z_min - 1e-6)
            & (all_z <= z_max + 1e-6)
            & (all_k <= k_max + 1e-6)
        )
        kept_idx = np.where(kept_mask)[0]
        if kept_idx.size == 0:
            raise ValueError(
                f"No KSData rows match z ∈ [{z_min}, {z_max}], k ≤ {k_max}."
            )

        kept_z = all_z[kept_idx]
        kept_k = all_k[kept_idx]
        kept_pf = all_pf[kept_idx]
        cov = all_cov[np.ix_(kept_idx, kept_idx)]
        cov = 0.5 * (cov + cov.T)   # symmetrize against tiny asymmetry

        z_blocks = _build_z_blocks(kept_z)
        self.z_blocks = z_blocks
        self.model = model

        self.theta_fid = (
            np.asarray(fiducial_vector(), dtype=float) if theta_fid is None
            else np.asarray(theta_fid, dtype=float)
        )

        # Mock data.
        if mock_data == "gp":
            d = self._predict_stacked(model, self.theta_fid, kept_k, z_blocks)
            if not np.all(np.isfinite(d)):
                raise FloatingPointError("Model at fid contains NaN/inf.")
        elif mock_data == "kodiaq":
            d = kept_pf.copy()
        else:
            raise ValueError(f"mock_data must be 'gp' or 'kodiaq'; got {mock_data!r}.")

        try:
            cov_chol = la.cholesky(cov, lower=True)
        except la.LinAlgError as e:
            raise ValueError(
                f"KSData covariance not positive-definite at z=[{z_min},{z_max}], "
                f"k_max={k_max}, cov_scale={cov_scale}: {e}"
            ) from e

        n = len(d)
        log_det = 2.0 * np.sum(np.log(np.diag(cov_chol)))
        log_norm = -0.5 * (n * np.log(2 * np.pi) + log_det)

        self.inputs = LikelihoodInputs(
            z=float((z_min + z_max) / 2.0),  # nominal; multi-z internally
            k_eboss=kept_k,                  # actually kodiaq k; kept name for compat
            d=d,
            cov=cov,
            cov_chol=cov_chol,
            log_norm=log_norm,
        )
        # Extra attrs for diagnostics.
        self.kept_z = kept_z
        self.kept_k = kept_k
        self.kept_pf_data = kept_pf
        self.z_min = float(z_min)
        self.z_max = float(z_max)
        self.k_max = float(k_max)
        self.mock_data = mock_data
        self.cov_scale = cov_scale

    @staticmethod
    def _predict_stacked(
        model: P1DModel,
        theta: np.ndarray,
        kept_k: np.ndarray,
        z_blocks: list[tuple[float, slice]],
    ) -> np.ndarray:
        out = np.empty(len(kept_k), dtype=float)
        for z_value, sl in z_blocks:
            k_block = kept_k[sl]
            p_block = np.asarray(model.predict(theta, k_block, z_value), dtype=float)
            if p_block.shape != k_block.shape:
                raise ValueError(
                    f"model.predict returned shape {p_block.shape}, expected "
                    f"{k_block.shape} at z={z_value}."
                )
            out[sl] = p_block
        return out

    def model_at(self, theta: np.ndarray) -> np.ndarray:
        return self._predict_stacked(
            self.model, np.asarray(theta, dtype=float),
            self.kept_k, self.z_blocks,
        )

    def log_likelihood(self, theta: np.ndarray) -> float:
        m = self.model_at(theta)
        r = self.inputs.d - m
        y = la.solve_triangular(self.inputs.cov_chol, r, lower=True)
        chi2 = float(y @ y)
        return float(self.inputs.log_norm - 0.5 * chi2)

    def chi_squared(self, theta: np.ndarray) -> float:
        m = self.model_at(theta)
        r = self.inputs.d - m
        y = la.solve_triangular(self.inputs.cov_chol, r, lower=True)
        return float(y @ y)

    def __call__(self, theta: np.ndarray) -> float:
        return self.log_likelihood(theta)
