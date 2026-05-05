"""Gaussian likelihood for single-z eBOSS P1D forecast.

    log L(theta) = -0.5 * (d - m(theta))^T C^-1 (d - m(theta)) + const

where `d` is the eBOSS DR14 P1D measurement at the chosen z-bin, `C` is its
covariance (optionally scaled by `cov_scale`), and `m(theta)` is the model
prediction binned onto the eBOSS k-grid.

Uses a Cholesky factor of `C` cached at construction so each likelihood call
is O(N^2) (a triangular solve), not O(N^3) (a full inverse). Raises on any
NaN / non-finite model output — silent NaN handling is forbidden per the
project spec.

Public API:

- ``GaussianLikelihood(model, ...)``: callable returning log L.
- ``LogPosterior(likelihood, params)``: adds a uniform prior over the 11
  parameter bounds. Returns -inf outside the prior box (used by emcee).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.linalg as la

from priya_forecast.data import bin_model_to_data, load_eboss
from priya_forecast.models.base import P1DModel
from priya_forecast.parameters import PARAMS_11D, Param, fiducial_vector


@dataclass
class LikelihoodInputs:
    """Frozen inputs the likelihood will reuse on every call."""

    z: float
    k_eboss: np.ndarray
    d: np.ndarray
    cov: np.ndarray
    cov_chol: np.ndarray  # lower-triangular Cholesky factor of `cov`
    log_norm: float       # -0.5 * (N*log(2pi) + log|cov|)


def _build_inputs(
    z: float,
    cov_scale: float,
    mock_data: str,
    model: P1DModel,
    theta_fid: np.ndarray,
    k_grid: np.ndarray | None = None,
    cov_diag_frac: float | None = None,
) -> LikelihoodInputs:
    if (k_grid is None) != (cov_diag_frac is None):
        raise ValueError(
            "_build_inputs: k_grid and cov_diag_frac must be either both "
            "None (eBOSS fallback) or both provided (synthetic diagonal "
            f"cov). Got k_grid={'None' if k_grid is None else 'array'}, "
            f"cov_diag_frac={cov_diag_frac!r}."
        )
    if k_grid is not None and cov_diag_frac is not None:
        # Synthetic diagonal cov for non-eBOSS k-grids (e.g. KODIAQ
        # production: k=0.005-0.064 s/km). σ_k = cov_diag_frac · P_F(fid, k).
        k_eboss = np.asarray(k_grid, dtype=float)
        m_fid = model.predict(theta_fid, k_eboss, z)
        if not np.all(np.isfinite(m_fid)):
            raise FloatingPointError("Model prediction at fid contains NaN/inf.")
        # Floor sigma so a (near-)zero P_F(fid) doesn't produce a singular
        # diagonal cov; use a small fraction of the median |P_F| as the floor.
        med = float(np.median(np.abs(m_fid)))
        sigma_floor = max(1e-30, med * 1e-12)
        sigma = np.maximum(float(cov_diag_frac) * np.abs(m_fid), sigma_floor)
        cov_eboss = np.diag(sigma ** 2)
        pf_eboss = m_fid.copy()
    else:
        k_eboss, pf_eboss, cov_eboss = load_eboss(z=z)
    cov = cov_eboss * float(cov_scale)
    if mock_data == "eboss":
        d = pf_eboss
    elif mock_data == "gp":
        # Use the model's own prediction at fiducial as the "data" — standard
        # forecast setup; ML is the fiducial point by construction.
        m_fid = model.predict(theta_fid, k_eboss, z)
        if not np.all(np.isfinite(m_fid)):
            raise FloatingPointError("Model prediction at fiducial contains NaN / inf.")
        d = m_fid
    else:
        raise ValueError(f"mock_data must be 'gp' or 'eboss', got {mock_data!r}.")

    # Symmetrize against tiny numerical asymmetry, then Cholesky.
    cov = 0.5 * (cov + cov.T)
    try:
        cov_chol = la.cholesky(cov, lower=True)
    except la.LinAlgError as e:
        raise ValueError(
            f"Covariance not positive-definite at z={z}, cov_scale={cov_scale}: {e}"
        ) from e

    n = len(d)
    log_det = 2.0 * np.sum(np.log(np.diag(cov_chol)))
    log_norm = -0.5 * (n * np.log(2 * np.pi) + log_det)
    return LikelihoodInputs(
        z=z, k_eboss=k_eboss, d=d, cov=cov, cov_chol=cov_chol, log_norm=log_norm
    )


class GaussianLikelihood:
    """Single-z Gaussian P1D likelihood.

    Parameters
    ----------
    model : P1DModel
        Forward model with `predict(theta, k, z)`.
    z : float
        eBOSS redshift bin to forecast at.
    cov_scale : float
        Multiplies the eBOSS covariance (sanity-check knob).
    mock_data : {"gp", "eboss"}
        "gp" → use the model at fiducial as the "data" (standard forecast).
        "eboss" → use the actual DR14 measurement.
    theta_fid : ndarray, shape (11,) | None
        Fiducial point for "gp" mock-data mode. Defaults to PARAMS_11D fids.
    """

    def __init__(
        self,
        *,
        model: P1DModel,
        z: float,
        cov_scale: float = 1.0,
        mock_data: str = "gp",
        theta_fid: np.ndarray | None = None,
        k_grid: np.ndarray | None = None,
        cov_diag_frac: float | None = None,
    ) -> None:
        self.model = model
        self.z = z
        self.cov_scale = cov_scale
        self.mock_data = mock_data
        self.theta_fid = (
            np.asarray(fiducial_vector(), dtype=float) if theta_fid is None
            else np.asarray(theta_fid, dtype=float)
        )
        if self.theta_fid.ndim != 1 or self.theta_fid.size == 0:
            raise ValueError(
                f"theta_fid must be 1D non-empty, got shape {self.theta_fid.shape}."
            )
        self.inputs = _build_inputs(
            z, cov_scale, mock_data, model, self.theta_fid,
            k_grid=k_grid, cov_diag_frac=cov_diag_frac,
        )

    # --- forward model ---------------------------------------------------

    def model_at(self, theta: np.ndarray) -> np.ndarray:
        """Return the model binned onto the eBOSS k-grid for one theta."""
        theta = np.asarray(theta, dtype=float)
        k = self.inputs.k_eboss
        m_native = self.model.predict(theta, k, self.z)
        if not np.all(np.isfinite(m_native)):
            raise FloatingPointError(
                f"Model returned non-finite P_F at theta={theta}: contains NaN/inf."
            )
        if m_native.shape == k.shape:
            return m_native
        # Shouldn't happen: predict() docstring requires shape (Nk,).
        raise ValueError(
            f"Model returned shape {m_native.shape}; expected {k.shape}."
        )

    # --- log-likelihood --------------------------------------------------

    def log_likelihood(self, theta: np.ndarray) -> float:
        m = self.model_at(theta)
        r = self.inputs.d - m
        # Solve L y = r for y, then chi2 = ||y||^2 (i.e., r^T C^-1 r).
        y = la.solve_triangular(self.inputs.cov_chol, r, lower=True)
        chi2 = float(y @ y)
        return float(self.inputs.log_norm - 0.5 * chi2)

    def chi_squared(self, theta: np.ndarray) -> float:
        """Return (d - m)^T C^-1 (d - m), without the constant log-det term."""
        m = self.model_at(theta)
        r = self.inputs.d - m
        y = la.solve_triangular(self.inputs.cov_chol, r, lower=True)
        return float(y @ y)

    # Make the class callable so it slots into emcee.EnsembleSampler.
    def __call__(self, theta: np.ndarray) -> float:
        return self.log_likelihood(theta)


# ---------------------------------------------------------------------------
# Posterior wrapper with uniform prior over the 11D parameter box
# ---------------------------------------------------------------------------


class UniformBoxPrior:
    """Flat prior on the 11 parameter bounds. Returns 0 inside the box, -inf
    outside. Used as the default prior for emcee."""

    def __init__(self, params: tuple[Param, ...] = PARAMS_11D) -> None:
        self.params = params
        self.lo = np.array([p.prior[0] for p in params], dtype=float)
        self.hi = np.array([p.prior[1] for p in params], dtype=float)

    def log_prior(self, theta: np.ndarray) -> float:
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self.lo.shape:
            raise ValueError(f"theta shape {theta.shape} != prior {self.lo.shape}.")
        if np.any(theta < self.lo) or np.any(theta > self.hi):
            return -np.inf
        return 0.0

    def __call__(self, theta: np.ndarray) -> float:
        return self.log_prior(theta)


class LogPosterior:
    """log_posterior(theta) = log_prior(theta) + log_likelihood(theta).

    Returns -inf if the prior excludes theta (avoids evaluating the model
    outside its training domain).
    """

    def __init__(
        self,
        likelihood: GaussianLikelihood,
        prior: UniformBoxPrior | None = None,
    ) -> None:
        self.likelihood = likelihood
        self.prior = prior if prior is not None else UniformBoxPrior()

    def __call__(self, theta: np.ndarray) -> float:
        lp = self.prior(theta)
        if not np.isfinite(lp):
            return lp
        return lp + self.likelihood(theta)
