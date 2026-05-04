"""Single multi-D PySR fit over a cross-coupled parameter subset.

Replaces the per-1D additive-Taylor combine for parameters that have
**genuine cross-couplings** the per-1D fit can't capture (the
coupling-matrix headline finding flagged `herei × alphaq` as the only
positive coupling in the 11-param prior cube). For the cross-coupled
subset we fit ONE PySR equation that takes all subset params jointly,
so the equation can express interaction terms like `(z − 3.9)² · herei
· alphaq`.

The other 11D parameters are handled outside this fit:
  - **Priored-out** (hub, omegamh2, bhfeedback): GP-slice fallback in
    the combine. Their σ in the Fisher is prior-dominated regardless.
  - **Mean-flux factor** (tau0): GP-slice fallback. Optional Kim
    Gaussian prior on tau0.
  - **Mean-flux slope** (dtau0): held fixed at 0 (production paper's
    `USE_TAU0_ONLY=true` convention).

The default cross-coupled subset is {ns, Ap, herei, heref, alphaq,
hireionz} per the user's direction. Override via `subset_names`.

Multi-z + at-fid anchor + dim-balanced loss (optional) are reused from
`refit_1d_pysr.py`. Inputs are 8-D: 6 θ-norms + k_norm + z_norm + res.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from priya_forecast.dim_balanced_loss import JULIA_LOSS_FUNCTION
from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.parameters import (
    PARAM_NAMES,
    fiducial_vector,
    get_param,
)
from priya_forecast.refit_1d_pysr import (
    DEFAULT_PYSR_KWARGS,
    HF_RESOLUTION,
    LF_RESOLUTION,
)


DEFAULT_SUBSET = ("ns", "Ap", "herei", "heref", "alphaq", "hireionz")
"""Default cross-coupled subset: cosmology + IGM thermal params with
genuine cross-coupling that per-1D can't capture (Phase 5 coupling
matrix headline + the user's redshift-dependence observation)."""


@dataclass
class MultiDRefitResult:
    """Bundles a single multi-D PySR equation + the metadata needed for
    multi-z prediction.

    Inputs are concatenated [θ_norm_subset (length n_subset), k_norm,
    resolution, z_norm] → equation features `x0..x_{n_subset+2}`.
    """

    subset_names: tuple[str, ...]            # which θ params the eq sees
    z_min: float
    z_max: float
    equation_str: str
    pareto_complexity: int
    pareto_loss: float
    pareto_complexities: list[int]
    pareto_losses: list[float]
    # Per-subset min-max from LF Sobol training data range.
    x_param_min: np.ndarray                  # (n_subset,)
    x_param_max: np.ndarray                  # (n_subset,)
    k_min: float
    k_max: float
    lf_resolution: float
    hf_resolution: float
    fid_phys: np.ndarray                     # full 11D fid (subset entries are anchors)
    norm: MultiZNormalizationSpec
    k_grid: np.ndarray
    wall_time_s: float
    lf_train_mean_rel_err: float
    hf_train_mean_rel_err: float
    lf_train_max_rel_err: float
    hf_train_max_rel_err: float

    @property
    def n_subset(self) -> int:
        return len(self.subset_names)

    def predict_normalized(
        self,
        theta_phys_full: np.ndarray,
        k: np.ndarray,
        resolution: float = HF_RESOLUTION,
        z: float | None = None,
    ) -> np.ndarray:
        """Evaluate the multi-D PySR eq in flux_norm space.

        `theta_phys_full` is a length-11 PRIYA vector; we extract the
        subset entries and min-max-normalize them. Other params in
        `theta_phys_full` are ignored by the equation (their effects
        come through the GP-slice fallback in the combine).
        """
        import sympy as sp
        z_eval = self.z_min if z is None else float(z)
        n_sub = self.n_subset

        # Extract subset θ values from the full vector.
        theta_phys_full = np.asarray(theta_phys_full, dtype=float)
        theta_subset = np.array([
            theta_phys_full[PARAM_NAMES.index(name)]
            for name in self.subset_names
        ], dtype=float)
        theta_norm = (theta_subset - self.x_param_min) / (
            self.x_param_max - self.x_param_min
        )

        # Build per-row 8-feature input.
        k = np.asarray(k, dtype=float)
        n_k = k.size
        k_norm = (k - self.k_min) / (self.k_max - self.k_min)
        z_norm = (z_eval - self.z_min) / (self.z_max - self.z_min)

        # Lambdify the equation.
        expr = sp.sympify(self.equation_str)
        n_features = n_sub + 3   # θ_subset + k + res + z
        all_syms = [sp.Symbol(f"x{i}") for i in range(n_features)]
        fn = sp.lambdify(
            all_syms, expr,
            modules=[{"inv": lambda x: 1.0 / x}, "numpy"],
        )

        # Build args: subset thetas (broadcast to k.shape), k_norm, res, z_norm.
        args = []
        for j in range(n_sub):
            args.append(np.full(n_k, float(theta_norm[j])))
        args.append(k_norm)
        args.append(np.full(n_k, float(resolution)))
        args.append(np.full(n_k, float(z_norm)))
        return np.broadcast_to(np.asarray(fn(*args), dtype=float), (n_k,)).copy()

    def predict(
        self,
        theta_phys_full: np.ndarray,
        k: np.ndarray,
        resolution: float = HF_RESOLUTION,
        z: float | None = None,
    ) -> np.ndarray:
        """Raw P_F via the bundled per-z normalization round-trip."""
        flux_norm = self.predict_normalized(
            theta_phys_full, k, resolution=resolution, z=z,
        )
        return self.norm.denormalize_flux(
            flux_norm, np.asarray(k, dtype=float),
            z=(z if z is not None else self.z_min),
        )


def _generate_multi_d_sobol(
    *,
    gp_lf,
    gp_hf,
    subset_names: tuple[str, ...],
    z_min: float,
    z_max: float,
    k_grid: np.ndarray,
    n_total: int = 256,
    seed: int = 0,
) -> dict:
    """Sobol scatter over (θ_subset × z), evaluate LF+HF on k_grid.

    Output: payload dict with `flux_lf_z`, `flux_hf_z` (n_total, n_k),
    `params_full` (n_total, 11), `theta_subset` (n_total, n_subset),
    `z_per_row` (n_total,) snapped to kodiaq z grid.
    """
    from scipy.stats import qmc

    fid = np.array(fiducial_vector(), dtype=float)
    n_sub = len(subset_names)
    bounds = np.array([get_param(n).prior for n in subset_names], dtype=float)
    sub_indices = [PARAM_NAMES.index(n) for n in subset_names]

    # Sobol over (subset_dim + 1) including z as the last column.
    sampler = qmc.Sobol(d=n_sub + 1, seed=seed)
    u = sampler.random(n=n_total)
    theta_subset = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u[:, :n_sub]
    z_continuous = z_min + (z_max - z_min) * u[:, n_sub]
    z_grid_kodiaq = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    z_grid_in_range = z_grid_kodiaq[
        (z_grid_kodiaq >= z_min - 1e-6) & (z_grid_kodiaq <= z_max + 1e-6)
    ]
    z_samples = z_grid_in_range[
        np.argmin(np.abs(z_continuous[:, None] - z_grid_in_range[None, :]), axis=1)
    ]

    k_grid = np.asarray(k_grid, dtype=float)
    n_k = k_grid.size
    flux_lf = np.empty((n_total, n_k), dtype=float)
    flux_hf = np.empty((n_total, n_k), dtype=float)
    params_full = np.tile(fid, (n_total, 1))
    for j, idx in enumerate(sub_indices):
        params_full[:, idx] = theta_subset[:, j]
    for i in range(n_total):
        z_i = float(z_samples[i])
        flux_lf[i] = np.asarray(gp_lf.predict(params_full[i], k_grid, z_i), dtype=float)
        flux_hf[i] = np.asarray(gp_hf.predict(params_full[i], k_grid, z_i), dtype=float)

    return dict(
        flux_lf_z=flux_lf, flux_hf_z=flux_hf,
        params_full=params_full,
        theta_subset=theta_subset,
        z_per_row=z_samples,
        z_grid_in_range=z_grid_in_range,
        sub_indices=sub_indices,
    )


def _build_multi_d_training_matrix(
    *,
    payload: dict,
    norm: MultiZNormalizationSpec,
    z_min: float,
    z_max: float,
    k_grid: np.ndarray,
    bounds: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stack LF + HF into 4-column inputs `[θ_subset, k, res, z]`.

    Returns (X_act, Y_act, ranges) where X_act has columns
    `[θ_norm[0], θ_norm[1], ..., k_norm, resolution, z_norm]`.
    """
    flux_lf = payload["flux_lf_z"]
    flux_hf = payload["flux_hf_z"]
    theta_subset = payload["theta_subset"]
    z_per_row = payload["z_per_row"]
    n_total, n_k = flux_lf.shape
    n_sub = theta_subset.shape[1]

    # Per-row at-fid anchor (norm.mean_flux is shape (n_z, n_k)).
    mean_per_row = np.zeros_like(flux_lf)
    std_per_row = np.zeros_like(flux_lf)
    for r in range(n_total):
        zi = norm._z_index(float(z_per_row[r]))
        mean_per_row[r] = np.interp(k_grid, norm.k_grid, norm.mean_flux[zi])
        std_per_row[r] = np.interp(k_grid, norm.k_grid, norm.std_flux[zi])
    flux_lf_norm = (flux_lf - mean_per_row) / std_per_row
    flux_hf_norm = (flux_hf - mean_per_row) / std_per_row

    # Min-max normalize θ_subset using bounds (n_sub, 2).
    x_param_min = bounds[:, 0]
    x_param_max = bounds[:, 1]
    theta_norm_per_row = (
        theta_subset - x_param_min[None, :]
    ) / (x_param_max - x_param_min)[None, :]

    # k_norm
    k_min = float(k_grid.min())
    k_max = float(k_grid.max())
    k_norm = (k_grid - k_min) / (k_max - k_min)
    z_norm_per_row = (z_per_row - z_min) / (z_max - z_min)

    # Build per-(row, k) X matrix. n_rows × n_k → flatten.
    # Column order: [θ_norm[0..n_sub-1], k_norm, resolution, z_norm].
    rows_lf = np.empty((n_total * n_k, n_sub + 3), dtype=float)
    rows_hf = np.empty((n_total * n_k, n_sub + 3), dtype=float)
    Y_lf = flux_lf_norm.ravel()
    Y_hf = flux_hf_norm.ravel()
    for r in range(n_total):
        s = r * n_k
        e = s + n_k
        rows_lf[s:e, 0:n_sub] = theta_norm_per_row[r][None, :]
        rows_lf[s:e, n_sub] = k_norm
        rows_lf[s:e, n_sub + 1] = LF_RESOLUTION
        rows_lf[s:e, n_sub + 2] = z_norm_per_row[r]
        rows_hf[s:e, 0:n_sub] = theta_norm_per_row[r][None, :]
        rows_hf[s:e, n_sub] = k_norm
        rows_hf[s:e, n_sub + 1] = HF_RESOLUTION
        rows_hf[s:e, n_sub + 2] = z_norm_per_row[r]

    X_act = np.vstack([rows_lf, rows_hf])
    Y_act = np.concatenate([Y_lf, Y_hf])
    return X_act, Y_act, dict(
        x_param_min=x_param_min, x_param_max=x_param_max,
        k_min=k_min, k_max=k_max, z_min=z_min, z_max=z_max,
    )


def refit_multi_d(
    *,
    gp_lf,
    gp_hf,
    subset_names: tuple[str, ...] = DEFAULT_SUBSET,
    z_min: float = 2.6,
    z_max: float = 4.2,
    k_grid: np.ndarray = None,
    n_total: int = 256,
    pysr_kwargs: dict | None = None,
    seed: int = 42,
    use_dim_balanced_loss: bool = True,
) -> MultiDRefitResult:
    """Train ONE multi-D PySR equation over (θ_subset, k, res, z)."""
    from pysr import PySRRegressor  # type: ignore[import-not-found]
    from priya_forecast.refit_1d_pysr import (
        compute_local_normalization_multiz,
    )

    if k_grid is None:
        k_grid = np.linspace(0.005, 0.064, 32)
    fid_full = np.array(fiducial_vector(), dtype=float)
    payload = _generate_multi_d_sobol(
        gp_lf=gp_lf, gp_hf=gp_hf, subset_names=subset_names,
        z_min=z_min, z_max=z_max, k_grid=k_grid,
        n_total=n_total, seed=seed,
    )
    z_grid_in_range = payload["z_grid_in_range"]

    # Per-z at-fid normalization (anchored on FULL fid_phys vector).
    norm = compute_local_normalization_multiz(
        flux_lf_z=payload["flux_lf_z"], z_per_row=payload["z_per_row"],
        z_grid=z_grid_in_range, k_grid=k_grid,
        gp_lf=gp_lf, fid=fid_full,
        param_min=0.0, param_max=1.0,  # not used for multi-D fit (no single param)
    )

    bounds = np.array([get_param(n).prior for n in subset_names], dtype=float)
    X_act, Y_act, ranges = _build_multi_d_training_matrix(
        payload=payload, norm=norm,
        z_min=z_min, z_max=z_max, k_grid=k_grid,
        bounds=bounds,
    )

    args = dict(DEFAULT_PYSR_KWARGS)
    args.update(pysr_kwargs or {})
    args["random_state"] = seed
    if use_dim_balanced_loss:
        args.pop("elementwise_loss", None)
        args["loss_function"] = JULIA_LOSS_FUNCTION

    t0 = time.time()
    model = PySRRegressor(**args)
    model.fit(X_act, Y_act.reshape(-1, 1))
    elapsed = time.time() - t0
    pareto = model.equations_

    # Prefer the lowest-loss Pareto entry whose equation references the most
    # subset θ-features. Each x_i for i < n_subset corresponds to a θ.
    n_sub = len(subset_names)
    def _theta_count(eq_str: str) -> int:
        return sum(1 for i in range(n_sub) if f"x{i}" in str(eq_str))
    pareto["_theta_count"] = pareto["equation"].astype(str).apply(_theta_count)
    max_count = int(pareto["_theta_count"].max())
    if max_count > 0:
        cand = pareto[pareto["_theta_count"] == max_count]
        best_idx = int(cand["loss"].idxmin())
    else:
        best_idx = int(pareto["loss"].idxmin())

    result = MultiDRefitResult(
        subset_names=tuple(subset_names),
        z_min=float(z_min), z_max=float(z_max),
        equation_str=str(pareto.iloc[best_idx]["equation"]),
        pareto_complexity=int(pareto.iloc[best_idx]["complexity"]),
        pareto_loss=float(pareto.iloc[best_idx]["loss"]),
        pareto_complexities=pareto["complexity"].astype(int).tolist(),
        pareto_losses=pareto["loss"].astype(float).tolist(),
        x_param_min=ranges["x_param_min"], x_param_max=ranges["x_param_max"],
        k_min=ranges["k_min"], k_max=ranges["k_max"],
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        fid_phys=fid_full, norm=norm, k_grid=np.asarray(k_grid, dtype=float),
        wall_time_s=elapsed,
        lf_train_mean_rel_err=float("nan"),
        hf_train_mean_rel_err=float("nan"),
        lf_train_max_rel_err=float("nan"),
        hf_train_max_rel_err=float("nan"),
    )
    diag = _validate_per_fidelity_multi_d(result=result, payload=payload)
    result.lf_train_mean_rel_err = diag["lf_mean"]
    result.hf_train_mean_rel_err = diag["hf_mean"]
    result.lf_train_max_rel_err = diag["lf_max"]
    result.hf_train_max_rel_err = diag["hf_max"]
    return result


def _validate_per_fidelity_multi_d(
    *, result: MultiDRefitResult, payload: dict
) -> dict[str, float]:
    flux_lf = payload["flux_lf_z"]
    flux_hf = payload["flux_hf_z"]
    z_per_row = payload["z_per_row"]
    params_full = payload["params_full"]
    n_total = flux_lf.shape[0]
    pred_lf = np.empty_like(flux_lf)
    pred_hf = np.empty_like(flux_hf)
    for r in range(n_total):
        z_r = float(z_per_row[r])
        pred_lf[r] = result.predict(
            theta_phys_full=params_full[r], k=result.k_grid,
            resolution=result.lf_resolution, z=z_r,
        )
        pred_hf[r] = result.predict(
            theta_phys_full=params_full[r], k=result.k_grid,
            resolution=result.hf_resolution, z=z_r,
        )
    rel_lf = np.abs(pred_lf - flux_lf) / np.abs(flux_lf)
    rel_hf = np.abs(pred_hf - flux_hf) / np.abs(flux_hf)
    return dict(
        lf_mean=float(rel_lf.mean()), hf_mean=float(rel_hf.mean()),
        lf_max=float(rel_lf.max()), hf_max=float(rel_hf.max()),
    )
