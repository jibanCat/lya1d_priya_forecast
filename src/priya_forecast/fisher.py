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
    likelihood: GaussianLikelihood,
    theta: np.ndarray,
    i: int,
    h: float,
    global_index: int | None = None,
) -> np.ndarray:
    """5-point stencil for dm/dtheta_i at `theta`, returning a length-Nk array.

    `theta` is the full-length vector the model expects; `i` is the index
    within the *varying* params list; `global_index` (default `i`) is the
    position to perturb in `theta`. Pass `global_index` when running a
    partial Fisher — i.e. some params held fixed at fid.
    """
    j = i if global_index is None else global_index
    pts = []
    for w in (-2, -1, 1, 2):
        t = theta.copy()
        t[j] = theta[j] + w * h
        pts.append(likelihood.model_at(t))
    return (-pts[3] + 8 * pts[2] - 8 * pts[1] + pts[0]) / (12 * h)


def compute_fisher_F_phys(
    *,
    likelihood: GaussianLikelihood,
    theta_fid: np.ndarray,
    params: tuple[Param, ...],
    param_indices: list[int],
    step_frac: float = 0.02,
    rel_tol: float = 0.05,
    max_halvings: int = 2,
) -> np.ndarray:
    """Return `F_phys = Y^T Y` without trying to invert.

    Used by the multi-z aggregator: at any single z some parameters may
    have zero gradient, producing a singular per-z Fisher. We only need
    F_phys per z; inversion happens once after summing across z.
    """
    n = len(params)
    widths = np.array([p.width() for p in params], dtype=float)
    L = likelihood.inputs.cov_chol
    init_steps = np.array([step_frac * w for w in widths], dtype=float)
    derivs: list[np.ndarray] = []
    for i in range(n):
        gi = param_indices[i]
        h = float(init_steps[i])
        d_prev = _stencil_derivative(likelihood, theta_fid, i, h, global_index=gi)
        y_prev = la.solve_triangular(L, d_prev, lower=True)
        f_ii_prev = float(y_prev @ y_prev)
        for _ in range(max_halvings):
            h_new = h / 2.0
            d_new = _stencil_derivative(likelihood, theta_fid, i, h_new, global_index=gi)
            y_new = la.solve_triangular(L, d_new, lower=True)
            f_ii_new = float(y_new @ y_new)
            if f_ii_new == 0:
                break
            rel = abs(f_ii_new - f_ii_prev) / abs(f_ii_new)
            if rel < rel_tol:
                d_prev, h, f_ii_prev = d_new, h_new, f_ii_new
                break
            d_prev, h, f_ii_prev = d_new, h_new, f_ii_new
        derivs.append(d_prev)
    Y = np.stack(
        [la.solve_triangular(L, d, lower=True) for d in derivs], axis=1
    )
    return Y.T @ Y


def combine_fisher_phys_arrays(
    F_phys_list: list[np.ndarray],
    *,
    params: tuple[Param, ...],
    theta_fid: np.ndarray,
    priors_sigma: dict[str, float] | None = None,
) -> FisherResult:
    """Sum a list of F_phys matrices, add priors, invert in dim-less coords."""
    n = len(params)
    F_phys = np.zeros((n, n), dtype=float)
    for F in F_phys_list:
        if F.shape != (n, n):
            raise ValueError(f"F_phys shape {F.shape} != ({n}, {n}).")
        F_phys = F_phys + np.asarray(F, dtype=float)
    if priors_sigma:
        for pname, sigma_p in priors_sigma.items():
            if pname not in {p.name for p in params}:
                raise KeyError(f"prior on unknown param {pname!r}.")
            i = next(j for j, p in enumerate(params) if p.name == pname)
            F_phys[i, i] += 1.0 / float(sigma_p) ** 2
    widths = np.array([p.width() for p in params], dtype=float)
    W = np.outer(widths, widths)
    F_hat = F_phys * W
    try:
        cov_hat = la.inv(F_hat)
    except la.LinAlgError as e:
        raise ValueError(f"Combined Fisher not invertible: {e}") from e
    cov = cov_hat * W
    sigma = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore"):
        corr = cov / np.outer(sigma, sigma)
    return FisherResult(
        F=F_phys, cov=cov, sigma=sigma, corr=corr,
        steps=np.zeros(n),
        param_names=tuple(p.name for p in params),
        theta_fid=theta_fid,
    )


def combine_fisher_phys(
    fishers: list[FisherResult],
    *,
    params: tuple[Param, ...],
    priors_sigma: dict[str, float] | None = None,
) -> FisherResult:
    """Sum per-z Fisher matrices (z-independent covariance) and re-invert.

    Multi-z setup: at each z bin, run `fisher_matrix(...)` WITHOUT priors,
    then pass the list of FisherResults here. We sum `F_phys` across z
    (z-bins are independent in the eBOSS / KODIAQ covariance), add the
    priors ONCE (so they're not double-counted), and invert in
    dimensionless coords for stability.

    Parameters
    ----------
    fishers : list of FisherResult
        Per-z results; each must share the same `params` ordering.
    params : tuple of Param
        Same params used in each per-z Fisher (for widths and names).
    priors_sigma : dict | None
        Gaussian priors applied ONCE after summation.
    """
    if not fishers:
        raise ValueError("combine_fisher_phys: no FisherResults provided.")
    n = len(params)
    F_phys = np.zeros((n, n), dtype=float)
    for fr in fishers:
        if fr.F.shape != (n, n):
            raise ValueError(
                f"FisherResult F shape {fr.F.shape} != ({n}, {n})."
            )
        F_phys = F_phys + fr.F
    if priors_sigma:
        for pname, sigma_p in priors_sigma.items():
            if pname not in {p.name for p in params}:
                raise KeyError(f"prior on unknown param {pname!r}.")
            i = next(j for j, p in enumerate(params) if p.name == pname)
            F_phys[i, i] += 1.0 / float(sigma_p) ** 2
    widths = np.array([p.width() for p in params], dtype=float)
    W = np.outer(widths, widths)
    F_hat = F_phys * W
    try:
        cov_hat = la.inv(F_hat)
    except la.LinAlgError as e:
        raise ValueError(f"Combined Fisher not invertible: {e}") from e
    cov = cov_hat * W
    sigma = np.sqrt(np.diag(cov))
    with np.errstate(invalid="ignore"):
        corr = cov / np.outer(sigma, sigma)
    # converged_steps are per-z; pick the first as a placeholder (the
    # diagnostic isn't very meaningful when summing across z anyway).
    return FisherResult(
        F=F_phys, cov=cov, sigma=sigma, corr=corr,
        steps=fishers[0].steps,
        param_names=tuple(p.name for p in params),
        theta_fid=fishers[0].theta_fid,
    )


def fisher_matrix(
    *,
    likelihood: GaussianLikelihood,
    theta_fid: np.ndarray | None = None,
    params: tuple[Param, ...] = PARAMS_11D,
    step_frac: float = 0.01,
    rel_tol: float = 0.01,
    max_halvings: int = 8,
    priors_sigma: dict[str, float] | None = None,
    param_indices: list[int] | None = None,
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
    # Two valid shapes for theta_fid: (n,) — same length as params (model
    # expects exactly this), or longer if running a partial Fisher with
    # fixed-at-fid params. In the longer case, `param_indices` must give
    # the global position of each varying param in `theta_fid`.
    if theta_fid.shape == (n,):
        global_indices = list(range(n)) if param_indices is None else list(param_indices)
    else:
        if param_indices is None:
            raise ValueError(
                f"theta_fid length {theta_fid.shape[0]} != n={n} params; "
                f"pass `param_indices` mapping each varying param to its "
                f"global index in theta_fid."
            )
        if len(param_indices) != n:
            raise ValueError(
                f"param_indices length {len(param_indices)} != n={n} params."
            )
        global_indices = list(param_indices)

    # Work in dimensionless `theta_hat = theta / width` so F is well-conditioned
    # regardless of physical-unit spans (Ap ~ 1e-9, ns ~ 0.25, etc.). At the end
    # we scale back: sigma_phys_i = sigma_hat_i * width_i.
    widths = np.array([p.width() for p in params], dtype=float)

    L = likelihood.inputs.cov_chol
    init_steps = np.array([step_frac * w for w in widths], dtype=float)
    converged_steps = np.empty(n)
    derivs: list[np.ndarray] = []

    for i in range(n):
        gi = global_indices[i]
        h = float(init_steps[i])
        d_prev = _stencil_derivative(likelihood, theta_fid, i, h, global_index=gi)
        y_prev = la.solve_triangular(L, d_prev, lower=True)
        f_ii_prev = float(y_prev @ y_prev)
        for _ in range(max_halvings):
            h_new = h / 2.0
            d_new = _stencil_derivative(likelihood, theta_fid, i, h_new, global_index=gi)
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
    # Add Gaussian priors on selected params: F_phys[i,i] += 1/σ_prior_i².
    # Used to break degeneracies (e.g. tau0 mean-flux at single z) — see
    # `lya_emulator_full/lyaemu/likelihood.py::_resolve_tau_prior` for the
    # production presets (kim, becker, kodiaq, squad).
    if priors_sigma:
        for pname, sigma_p in priors_sigma.items():
            if pname not in {p.name for p in params}:
                raise KeyError(
                    f"prior on unknown param {pname!r}; "
                    f"known: {tuple(p.name for p in params)}"
                )
            i = next(j for j, p in enumerate(params) if p.name == pname)
            F_phys[i, i] += 1.0 / float(sigma_p) ** 2
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
    # theta_fid stored on the result is the per-varying-param view (length n),
    # extracted from the full vector if needed. Matches `param_names`.
    if theta_fid.shape == (n,):
        theta_fid_subset = theta_fid
    else:
        theta_fid_subset = np.array(
            [theta_fid[gi] for gi in global_indices], dtype=float
        )
    return FisherResult(
        F=F,
        cov=cov,
        sigma=sigma,
        corr=corr,
        steps=converged_steps,
        param_names=tuple(p.name for p in params),
        theta_fid=theta_fid_subset,
    )
