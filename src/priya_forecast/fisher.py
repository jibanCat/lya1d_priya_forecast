"""Fisher forecast via 5-point-stencil derivatives with adaptive step halving.

For a Gaussian likelihood with parameter-independent covariance,

    F_ij = (dm/dtheta_i)^T C^-1 (dm/dtheta_j)

We evaluate `dm/dtheta_i` with a centered 5-point stencil:

    dm/dtheta ≈ (-m(+2h) + 8 m(+h) - 8 m(-h) + m(-2h)) / (12 h)

Step `h_i` starts at `step_frac * (prior_hi - prior_lo)` and halves until the
relative change in F_ii (the diagonal) is below `rel_tol`. Halving
independently per parameter — different parameters have different curvature
scales.

The output bundle exposes:

- ``F``       : Fisher matrix (n, n)
- ``cov``     : F^-1, the parameter covariance
- ``sigma``   : sqrt(diag(cov)), the marginalized 1-sigma errors
- ``corr``    : correlation matrix derived from cov
- ``steps``   : the converged step size per parameter (diagnostic)

Saving:
- ``save_npz(path)``         : F, cov, sigma, corr, steps, param names.
- ``markdown_table()``       : human-readable 1-sigma summary.
"""

from __future__ import annotations

import io
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import scipy.linalg as la

from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.parameters import PARAMS_11D, Param


@dataclass
class FisherResult:
    F: np.ndarray
    cov: np.ndarray
    sigma: np.ndarray
    corr: np.ndarray
    steps: np.ndarray
    param_names: tuple[str, ...]
    theta_fid: np.ndarray

    def save_npz(self, path: str | Path) -> None:
        np.savez(
            path,
            F=self.F,
            cov=self.cov,
            sigma=self.sigma,
            corr=self.corr,
            steps=self.steps,
            param_names=np.array(self.param_names),
            theta_fid=self.theta_fid,
        )

    def markdown_table(self) -> str:
        lines = [
            "| Parameter | Fiducial | sigma | sigma / |fid| |",
            "|---|---|---|---|",
        ]
        for name, fid, s in zip(self.param_names, self.theta_fid, self.sigma):
            ratio = s / abs(fid) if fid != 0 else float("nan")
            lines.append(f"| {name} | {fid:.5g} | {s:.3g} | {ratio:.3g} |")
        return "\n".join(lines)


def _stencil_derivative(
    likelihood: GaussianLikelihood, theta: np.ndarray, i: int, h: float
) -> np.ndarray:
    """5-point stencil for dm/dtheta_i at `theta`, returning a length-Nk array."""
    pts = []
    for w in (-2, -1, 1, 2):
        t = theta.copy()
        t[i] = theta[i] + w * h
        pts.append(likelihood.model_at(t))
    return (-pts[3] + 8 * pts[2] - 8 * pts[1] + pts[0]) / (12 * h)


def fisher_matrix(
    *,
    likelihood: GaussianLikelihood,
    theta_fid: np.ndarray | None = None,
    params: tuple[Param, ...] = PARAMS_11D,
    step_frac: float = 0.01,
    rel_tol: float = 0.01,
    max_halvings: int = 8,
) -> FisherResult:
    """Compute the Fisher matrix at `theta_fid` with adaptive step halving.

    Parameters
    ----------
    likelihood : GaussianLikelihood
        Provides `model_at` and the cached Cholesky of C.
    theta_fid : ndarray, shape (n,) | None
        Linearization point. Defaults to `[p.fid for p in params]`.
    params : tuple of Param
        Parameter metadata (used for prior widths).
    step_frac : float
        Initial step h_i = step_frac * (prior_hi - prior_lo).
    rel_tol : float
        Halve h_i until |F_ii(h) - F_ii(h/2)| / |F_ii(h/2)| < rel_tol.
    max_halvings : int
        Hard cap on halvings to keep this fast; the test suite uses 4-5.
    """
    if theta_fid is None:
        theta_fid = np.array([p.fid for p in params], dtype=float)
    theta_fid = np.asarray(theta_fid, dtype=float)
    n = len(params)
    if theta_fid.shape != (n,):
        raise ValueError(f"theta_fid must be ({n},), got {theta_fid.shape}.")

    # Work in dimensionless `theta_hat = theta / width` so F is well-conditioned
    # regardless of physical-unit spans (Ap ~ 1e-9, ns ~ 0.25, etc.). At the end
    # we scale back: sigma_phys_i = sigma_hat_i * width_i.
    widths = np.array([p.width() for p in params], dtype=float)

    L = likelihood.inputs.cov_chol
    init_steps = np.array([step_frac * w for w in widths], dtype=float)
    converged_steps = np.empty(n)
    derivs: list[np.ndarray] = []

    for i in range(n):
        h = float(init_steps[i])
        d_prev = _stencil_derivative(likelihood, theta_fid, i, h)
        y_prev = la.solve_triangular(L, d_prev, lower=True)
        f_ii_prev = float(y_prev @ y_prev)
        for _ in range(max_halvings):
            h_new = h / 2.0
            d_new = _stencil_derivative(likelihood, theta_fid, i, h_new)
            y_new = la.solve_triangular(L, d_new, lower=True)
            f_ii_new = float(y_new @ y_new)
            if f_ii_new == 0:
                break
            rel = abs(f_ii_new - f_ii_prev) / abs(f_ii_new)
            if rel < rel_tol:
                d_prev, h, f_ii_prev = d_new, h_new, f_ii_new
                break
            d_prev, h, f_ii_prev = d_new, h_new, f_ii_new
        converged_steps[i] = h
        derivs.append(d_prev)

    # Physical-unit Fisher = Y^T Y where Y_i = L^-1 dm/dtheta_i.
    Y = np.stack(
        [la.solve_triangular(L, d, lower=True) for d in derivs], axis=1
    )  # shape (Nk, n)
    F_phys = Y.T @ Y
    # Re-express in dimensionless coords: F_hat_ij = F_phys_ij * width_i * width_j.
    W = np.outer(widths, widths)
    F_hat = F_phys * W
    try:
        cov_hat = la.inv(F_hat)
    except la.LinAlgError as e:
        raise ValueError(f"Fisher matrix not invertible: {e}") from e
    cov = cov_hat * W
    F = F_phys

    sigma = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore"):
        corr = cov / np.outer(sigma, sigma)
    return FisherResult(
        F=F,
        cov=cov,
        sigma=sigma,
        corr=corr,
        steps=converged_steps,
        param_names=tuple(p.name for p in params),
        theta_fid=theta_fid,
    )
