"""Per-param 1D PySR refit — verbatim replication of the student's recipe.

Source of truth: `student_projects/priya_pysr/pysr_mf_given.py`. We
reproduce its training pipeline literally:

  1. Load LF + HF 1pvar HDF5 from
     `<data_dir>/lf_{param}_npoints50_datacorrFalse.hdf5` and the matching
     `hf_*` file (default data_dir =
     `/home/mfho/student_projects/InferenceLyaData/1pvar/`).
  2. Slice the z-bin of interest from `flux_vectors[:, zindex, :]` and
     `kfkms[:, zindex, :]` (50 sims × 35 k-bins).
  3. **Normalize with the multi-D global `(mean_k, std_k)`** supplied by
     the caller — *not* the local 1D mean/std of the 1pvar slice. The
     student's `pysr_mf_given.py` falls back to the local 1D version
     only when the multi-D file is absent; the intended path uses the
     multi-D normalization, see `mean_flux_low_<subset>.txt` produced
     by `mf_*.py` and the user's explicit clarification.
  4. Min-max normalize `θ` and `k` to [0,1] from LF data extents.
  5. Stack: `X = [(θ_norm, k_norm, 0.4)_LF; (θ_norm, k_norm, 0.8)_HF]`,
     shape (2·50·35, 3) = (3500, 3); `Y` = stacked flux_norm.
  6. PySR config from `priya_pysr/pysr_model.create_model`:
     `niter=20, maxsize=20, maxdepth=10`,
     binary `[+, -, *, /, ^]`,
     unary `[sin, cos, exp, log, square, sqrt, inv]`,
     loss `(x − y)^2`, `random_state=42`, `model_selection="best"`.
  7. Validate <1% mean relative error per fidelity on the training data.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import h5py
import numpy as np

from priya_forecast.models.normalization import NormalizationSpec
from priya_forecast.parameters import (
    PARAM_NAMES,
    fiducial_vector,
    get_param,
)

# Default location of the student's 1pvar HDF5 files (50-point 1-parameter
# variations at LF and HF). Overridable per-call.
DEFAULT_1PVAR_DIR = Path(
    "/home/mfho/student_projects/InferenceLyaData/1pvar"
)

# Resolution feature values from the student's pipeline.
LF_RESOLUTION = 0.4
HF_RESOLUTION = 0.8

# PySR config: like the student's `priya_pysr/pysr_model.create_model`
# but **drops sin/cos** from the unary set per `feedback_pysr_operators.md`.
# Trig-rich equations produce oscillatory derivatives that wreck the
# Fisher matrix's conditioning (10³–10⁶× σ ratios in our z=3.6 baseline).
# Use a "GP-like" smooth-analytic operator set instead.
DEFAULT_PYSR_KWARGS: dict[str, Any] = dict(
    model_selection="best",
    niterations=50,           # was 20 — extra search to chase tail outliers
    maxsize=20,
    maxdepth=10,
    binary_operators=["+", "-", "*", "/", "^"],
    unary_operators=["exp", "log", "square", "sqrt", "inv(x) = 1/x"],
    extra_sympy_mappings={"inv": lambda x: 1 / x},
    elementwise_loss="loss(prediction, target) = (prediction - target)^2",
    # Tame the `^` operator: arbitrary base, simple exponent only. Prevents
    # `(complex)^(complex)` patterns that fit the mean but blow up at
    # prior boundaries (Ap HF max rel-err = 19% in the previous run).
    constraints={"^": (-1, 1)},
    # Discourage `^` slightly to bias toward polynomial / log / exp where
    # those are sufficient.
    complexity_of_operators={"^": 3},
    # Multithreading for ~4-8x faster fits — investigation-loop priority
    # per `feedback_pysr_speed.md`. `deterministic=False` is required when
    # parallelism is on; results may differ slightly across runs but the
    # best-equation under a fixed random_state is reproducible to within
    # genetic-algorithm noise.
    deterministic=False,
    parallelism="multithreading",
    procs=4,
    verbosity=0,
)


# Phase 1.5 "smart refit" config for IGM-thermal params (heref, herei, alphaq).
# Phase-1 closure (results/closure_at_simdat_ind15_ksdata/scorecard.md) showed
# σ_PySR/σ_GP off-fid at 6.4× (heref), 2.7× (herei), 0.20× (alphaq) — all three
# have wrong fid-curvature in their default-MSE per-1D eqs. Two architectural
# fixes per user direction (2026-05-04):
#   - Drop `inv` and `sqrt` from unary operators (the two with sharpest
#     fid-curvature); keep smooth analytic operators only.
#   - Replace MSE elementwise loss with the dim-balanced ANOVA loss
#     (penalizes batch-level main effects on dropped features → forces PySR
#     to use θ even when (k, z, r) alone could lower the per-sample MSE).
# niter stays at 50 — these params are weakly sensitive to P_F at fid; more
# genetic search can't find signal that isn't there. Architectural lever only.
SMART_REFIT_PYSR_KWARGS: dict[str, Any] = dict(DEFAULT_PYSR_KWARGS)
SMART_REFIT_PYSR_KWARGS["unary_operators"] = ["exp", "log", "square"]
SMART_REFIT_PYSR_KWARGS["extra_sympy_mappings"] = {}
# PySR can't take both `elementwise_loss` and `loss_function`; swap them.
SMART_REFIT_PYSR_KWARGS.pop("elementwise_loss", None)
from priya_forecast.dim_balanced_loss import JULIA_LOSS_FUNCTION  # noqa: E402
SMART_REFIT_PYSR_KWARGS["loss_function"] = JULIA_LOSS_FUNCTION

# Default: which params get the smart-refit treatment. Override at call time.
SMART_REFIT_PARAMS: tuple[str, ...] = ("heref", "herei", "alphaq")


@dataclass
class Refit1DResult:
    """Bundles a per-param PySR equation + the metadata needed to evaluate
    it on `(θ_norm, k_norm, resolution[, z_norm])` and de-normalize back to
    raw P_F.

    Two flavors:
      - **Single-z** (legacy): equation has 3 inputs (x0=θ_norm, x1=k_norm,
        x2=resolution). `z` is a scalar; `z_min`, `z_max` unused.
      - **Multi-z** (this session's day-1 extension): 4 inputs
        (x0=θ_norm, x1=k_norm, x2=resolution, x3=z_norm). `z_min`,
        `z_max` set the z range PySR was trained on. predict() /
        predict_normalized() take an explicit `z` argument.
    """

    param_name: str
    z: float                   # single-z: the bin; multi-z: nominal/center z
    equation_str: str
    pareto_complexity: int
    pareto_loss: float
    pareto_complexities: list[int]
    pareto_losses: list[float]
    x_param_min: float
    x_param_max: float
    k_min: float
    k_max: float
    lf_resolution: float
    hf_resolution: float
    fid_value: float
    norm: NormalizationSpec
    k_grid: np.ndarray
    wall_time_s: float
    lf_train_mean_rel_err: float
    hf_train_mean_rel_err: float
    lf_train_max_rel_err: float
    hf_train_max_rel_err: float
    # Multi-z: range of z covered by the training set. None for single-z fits.
    z_min: float | None = None
    z_max: float | None = None

    @property
    def is_multiz(self) -> bool:
        return self.z_min is not None and self.z_max is not None

    def _z_norm(self, z: float | np.ndarray) -> np.ndarray:
        if not self.is_multiz:
            raise ValueError(
                "Refit1DResult was trained single-z; z_norm not defined."
            )
        z_arr = np.asarray(z, dtype=float)
        return (z_arr - self.z_min) / (self.z_max - self.z_min)

    def predict_normalized(
        self,
        theta_phys: float | np.ndarray,
        k: np.ndarray,
        resolution: float = HF_RESOLUTION,
        z: float | None = None,
    ) -> np.ndarray:
        """Evaluate the PySR equation in flux_norm space.

        Inputs: `x0 = (theta_phys - x_param_min)/(x_param_max - x_param_min)`,
        `x1 = (k - k_min)/(k_max - k_min)`, `x2 = resolution`. For multi-z
        fits, also `x3 = (z - z_min)/(z_max - z_min)`.

        `z` defaults to the result's stored `self.z` (single-z fits) or
        the multi-z midpoint.
        """
        import sympy as sp
        expr = sp.sympify(self.equation_str)
        x_syms = sorted(
            [s for s in expr.free_symbols if s.name.startswith("x")],
            key=lambda s: int(s.name[1:]),
        )
        x0, x1, x2, x3 = (
            sp.Symbol("x0"), sp.Symbol("x1"),
            sp.Symbol("x2"), sp.Symbol("x3"),
        )
        all_syms = list({*x_syms, x0, x1, x2, x3})
        all_syms.sort(key=lambda s: int(s.name[1:]) if s.name.startswith("x") else 99)
        fn = sp.lambdify(
            all_syms, expr,
            modules=[{"inv": lambda x: 1.0 / x}, "numpy"],
        )

        theta_phys_arr = np.asarray(theta_phys, dtype=float)
        k = np.asarray(k, dtype=float)
        theta_norm = (theta_phys_arr - self.x_param_min) / (self.x_param_max - self.x_param_min)
        k_norm = (k - self.k_min) / (self.k_max - self.k_min)
        if theta_norm.ndim == 0:
            theta_norm_arr = np.full_like(k, float(theta_norm))
        else:
            theta_norm_arr = np.asarray(theta_norm, dtype=float)
        res_arr = np.full_like(k, float(resolution))
        if self.is_multiz:
            z_eval = self.z if z is None else z
            z_norm_arr = np.full_like(k, float(self._z_norm(z_eval)))
        else:
            z_norm_arr = np.zeros_like(k)
        args = []
        for s in all_syms:
            if s.name == "x0":
                args.append(theta_norm_arr)
            elif s.name == "x1":
                args.append(k_norm)
            elif s.name == "x2":
                args.append(res_arr)
            elif s.name == "x3":
                args.append(z_norm_arr)
            else:
                args.append(np.zeros_like(k))
        return np.broadcast_to(np.asarray(fn(*args), dtype=float), k.shape).copy()

    def predict(
        self,
        theta_phys: float | np.ndarray,
        k: np.ndarray,
        resolution: float = HF_RESOLUTION,
        z: float | None = None,
    ) -> np.ndarray:
        """Raw P_F: `flux_norm · std_k + mean_k` (per-param normalization)."""
        flux_norm = self.predict_normalized(
            theta_phys, k, resolution=resolution, z=z,
        )
        # Multi-z normalization needs the z to look up the right per-z spec;
        # single-z normalization ignores z. denormalize_flux signature unified
        # via the optional `z=` kwarg in NormalizationSpec.
        return self.norm.denormalize_flux(
            flux_norm, np.asarray(k, dtype=float),
            z=(z if z is not None else self.z),
        )


def _load_1pvar(
    *, param_name: str, z: float, data_dir: Path,
) -> dict[str, np.ndarray]:
    """Load LF and HF 1pvar HDF5 for one param, slice the z-bin.

    Returns a dict with keys
      - `params_lf`, `params_hf`        : (50, 11)
      - `kfkms_lf_z`, `kfkms_hf_z`      : (50, 35)
      - `flux_lf_z`, `flux_hf_z`        : (50, 35)
      - `zindex_lf`, `zindex_hf`        : int
      - `kfkms_lf_min`, `kfkms_lf_max`  : float (used for k-norm)

    Raises FileNotFoundError if either HDF5 is absent.
    """
    data_dir = Path(data_dir)
    lf_path = data_dir / f"lf_{param_name}_npoints50_datacorrFalse.hdf5"
    hf_path = data_dir / f"hf_{param_name}_npoints50_datacorrFalse.hdf5"
    for p in (lf_path, hf_path):
        if not p.exists():
            raise FileNotFoundError(f"1pvar HDF5 not found: {p}")

    # IMPORTANT: the 1pvar HDF5 files store `k * P_F / π` (the
    # `sample_1P_predictions` path in `lyaemu.priya_explorer` line 193
    # applies that transform before saving). Our framework, including
    # the global multi-D normalization, works in raw P_F. Undo the
    # transform on load: P_F = stored * π / k.
    out: dict[str, np.ndarray] = {}
    with h5py.File(lf_path, "r") as fh:
        zout_lf = fh["zout"][:]
        zindex_lf = int(np.argmin(np.abs(zout_lf - z)))
        if abs(zout_lf[zindex_lf] - z) > 1e-3:
            raise ValueError(f"z={z} not in {lf_path}'s zout: {zout_lf}")
        out["params_lf"] = fh["params"][:]
        kfkms_lf = fh["kfkms"][:, zindex_lf, :]
        flux_lf_kP = fh["flux_vectors"][:, zindex_lf, :]
        out["kfkms_lf_z"] = kfkms_lf
        out["flux_lf_z"] = flux_lf_kP * np.pi / kfkms_lf
        out["zindex_lf"] = zindex_lf
    with h5py.File(hf_path, "r") as fh:
        zout_hf = fh["zout"][:]
        zindex_hf = int(np.argmin(np.abs(zout_hf - z)))
        if abs(zout_hf[zindex_hf] - z) > 1e-3:
            raise ValueError(f"z={z} not in {hf_path}'s zout: {zout_hf}")
        out["params_hf"] = fh["params"][:]
        kfkms_hf = fh["kfkms"][:, zindex_hf, :]
        flux_hf_kP = fh["flux_vectors"][:, zindex_hf, :]
        out["kfkms_hf_z"] = kfkms_hf
        out["flux_hf_z"] = flux_hf_kP * np.pi / kfkms_hf
        out["zindex_hf"] = zindex_hf
    out["kfkms_lf_min"] = float(out["kfkms_lf_z"].min())
    out["kfkms_lf_max"] = float(out["kfkms_lf_z"].max())
    return out


def _build_training_matrix(
    *,
    payload: dict,
    param_idx: int,
    global_norm: NormalizationSpec,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    """Stack LF + HF into the student's `(X, y)` training matrix.

    `global_norm` provides the MULTI-D `(mean_k, std_k)` used to normalize
    both LF and HF flux. `mean_k` / `std_k` are interpolated onto each
    fidelity's k-grid before per-bin normalization.

    Returns (X_act, Y_act, ranges, fidelity_arrays) where
      - X_act shape (3500, 3) cols = [theta_norm, k_norm, resolution]
      - Y_act shape (3500,)   = stacked flux_norm
      - ranges = {x_param_min/max, k_min/max} (from LF data, used by the
        student's recipe and stored on the Refit1DResult)
      - fidelity_arrays = LF/HF flux_norm reshaped (50, 35) for
        diagnostics + denormalized P_F arrays.
    """
    flux_lf = payload["flux_lf_z"]   # (50, 35)
    flux_hf = payload["flux_hf_z"]   # (50, 35)
    k_lf = payload["kfkms_lf_z"]     # (50, 35)
    k_hf = payload["kfkms_hf_z"]     # (50, 35)
    params_lf = payload["params_lf"] # (50, 11)
    params_hf = payload["params_hf"] # (50, 11)

    # Normalize both fidelities with the GLOBAL multi-D (mean_k, std_k),
    # interpolated onto each fidelity's k-grid. Per the contract: the
    # normalization is from MULTI-D fid, not 1D.
    mean_k_lf = np.interp(k_lf[0], global_norm.k_grid, global_norm.mean_flux)
    std_k_lf = np.interp(k_lf[0], global_norm.k_grid, global_norm.std_flux)
    mean_k_hf = np.interp(k_hf[0], global_norm.k_grid, global_norm.mean_flux)
    std_k_hf = np.interp(k_hf[0], global_norm.k_grid, global_norm.std_flux)

    flux_lf_norm = (flux_lf - mean_k_lf[None, :]) / std_k_lf[None, :]
    flux_hf_norm = (flux_hf - mean_k_hf[None, :]) / std_k_hf[None, :]

    # X_param column from LF data only (matches student script).
    x_param_lf = np.repeat(params_lf[:, param_idx, None], k_lf.shape[1], axis=1)
    x_param_hf = np.repeat(params_hf[:, param_idx, None], k_hf.shape[1], axis=1)
    x_param_min = float(x_param_lf.min())
    x_param_max = float(x_param_lf.max())
    x_param_lf_norm = (x_param_lf - x_param_min) / (x_param_max - x_param_min)
    x_param_hf_norm = (x_param_hf - x_param_min) / (x_param_max - x_param_min)

    k_min = float(k_lf.min())
    k_max = float(k_lf.max())
    k_lf_norm = (k_lf - k_min) / (k_max - k_min)
    k_hf_norm = (k_hf - k_min) / (k_max - k_min)  # use LF range, per student

    n_lf = flux_lf.size  # 50 * 35 = 1750
    n_hf = flux_hf.size

    res_lf = np.full(n_lf, LF_RESOLUTION)
    res_hf = np.full(n_hf, HF_RESOLUTION)

    X_lf = np.column_stack([x_param_lf_norm.ravel(), k_lf_norm.ravel(), res_lf])
    X_hf = np.column_stack([x_param_hf_norm.ravel(), k_hf_norm.ravel(), res_hf])
    Y_lf = flux_lf_norm.ravel()
    Y_hf = flux_hf_norm.ravel()

    X_act = np.vstack([X_lf, X_hf])
    Y_act = np.concatenate([Y_lf, Y_hf])

    return (
        X_act,
        Y_act,
        dict(
            x_param_min=x_param_min, x_param_max=x_param_max,
            k_min=k_min, k_max=k_max,
        ),
        dict(
            flux_lf=flux_lf, flux_hf=flux_hf,
            flux_lf_norm=flux_lf_norm, flux_hf_norm=flux_hf_norm,
            k_lf=k_lf, k_hf=k_hf,
            mean_k_lf=mean_k_lf, std_k_lf=std_k_lf,
            mean_k_hf=mean_k_hf, std_k_hf=std_k_hf,
        ),
    )


def _validate_per_fidelity(
    *,
    result: Refit1DResult,
    fidelity_arrays: dict[str, np.ndarray],
) -> tuple[float, float, float, float]:
    """Compute mean and max relative error per-fidelity on the training set.

    For each (sim, k) of LF and HF: predict the de-normalized P_F via the
    refit's `predict()` and compare to the true P_F from the 1pvar HDF5.
    Returns (lf_mean, hf_mean, lf_max, hf_max) of `|P_pred - P_true| / P_true`.
    """
    flux_lf = fidelity_arrays["flux_lf"]   # (50, 35)
    flux_hf = fidelity_arrays["flux_hf"]
    k_lf = fidelity_arrays["k_lf"]
    k_hf = fidelity_arrays["k_hf"]

    n_sims = flux_lf.shape[0]
    pred_lf = np.empty_like(flux_lf)
    pred_hf = np.empty_like(flux_hf)
    # We need theta_phys per sim — caller should have stored params_lf in a
    # way we can recover it. For now, recover via the X_param_min/max +
    # the fidelity_arrays' shape: we approximate by computing the implied
    # theta from the un-normalized x_param column. Since we passed
    # raw 1pvar values into the loader, just recompute from payload outside.
    # Simpler: caller passes thetas directly.
    raise NotImplementedError(
        "Use _validate_per_fidelity_from_payload — this stub is unused."
    )


def _validate_per_fidelity_from_payload(
    *,
    result: Refit1DResult,
    payload: dict,
    param_idx: int,
) -> dict[str, float]:
    """Mean and max relative error of the refit on each fidelity slice."""
    flux_lf = payload["flux_lf_z"]
    flux_hf = payload["flux_hf_z"]
    k_lf = payload["kfkms_lf_z"]
    k_hf = payload["kfkms_hf_z"]
    thetas_lf = payload["params_lf"][:, param_idx]
    thetas_hf = payload["params_hf"][:, param_idx]

    pred_lf = np.empty_like(flux_lf)
    for i, t in enumerate(thetas_lf):
        pred_lf[i] = result.predict(theta_phys=float(t), k=k_lf[i],
                                     resolution=result.lf_resolution)
    pred_hf = np.empty_like(flux_hf)
    for i, t in enumerate(thetas_hf):
        pred_hf[i] = result.predict(theta_phys=float(t), k=k_hf[i],
                                     resolution=result.hf_resolution)
    rel_lf = np.abs(pred_lf - flux_lf) / np.abs(flux_lf)
    rel_hf = np.abs(pred_hf - flux_hf) / np.abs(flux_hf)
    return dict(
        lf_mean=float(rel_lf.mean()),
        hf_mean=float(rel_hf.mean()),
        lf_max=float(rel_lf.max()),
        hf_max=float(rel_hf.max()),
    )


def _generate_1pvar_inline(
    *,
    gp_lf,
    gp_hf,
    param_name: str,
    z: float,
    k_grid: np.ndarray,
    n_points: int = 50,
) -> dict[str, np.ndarray]:
    """Sweep one param over its prior; collect LF + HF flux from emulators.

    Replaces the student's pre-saved `1pvar/{lf,hf}_<param>_npoints50` HDF5
    files with on-the-fly generation. Mirrors the structure that
    `_build_training_matrix` expects:
      - `params_lf`, `params_hf`        : (n_points, 11) — full PRIYA vector
      - `kfkms_lf_z`, `kfkms_hf_z`      : (n_points, n_k) — replicated k_grid
      - `flux_lf_z`, `flux_hf_z`        : (n_points, n_k) — raw P_F

    Output P_F is in raw units (no `k·P/π` transform). Other params held at
    fid; only `param_name` varies (linspace over its `(prior_lo, prior_hi)`).
    """
    if param_name not in PARAM_NAMES:
        raise KeyError(f"Unknown PRIYA parameter {param_name!r}.")
    p = get_param(param_name)
    fid = np.array(fiducial_vector(), dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    n_k = k_grid.size
    idx = PARAM_NAMES.index(param_name)
    samples = np.linspace(p.prior[0], p.prior[1], n_points)

    flux_lf = np.empty((n_points, n_k), dtype=float)
    flux_hf = np.empty((n_points, n_k), dtype=float)
    params_arr = np.tile(fid, (n_points, 1))
    params_arr[:, idx] = samples
    for i in range(n_points):
        theta = params_arr[i]
        flux_lf[i] = np.asarray(gp_lf.predict(theta, k_grid, z), dtype=float)
        flux_hf[i] = np.asarray(gp_hf.predict(theta, k_grid, z), dtype=float)

    kfkms = np.tile(k_grid, (n_points, 1))
    return dict(
        params_lf=params_arr.copy(),
        params_hf=params_arr.copy(),
        kfkms_lf_z=kfkms.copy(),
        kfkms_hf_z=kfkms.copy(),
        flux_lf_z=flux_lf,
        flux_hf_z=flux_hf,
        zindex_lf=-1,
        zindex_hf=-1,
        kfkms_lf_min=float(k_grid.min()),
        kfkms_lf_max=float(k_grid.max()),
    )


def _generate_1pvar_multiz_inline(
    *,
    gp_lf,
    gp_hf,
    param_name: str,
    z_min: float,
    z_max: float,
    k_grid: np.ndarray,
    n_total: int = 225,
    seed: int = 0,
) -> dict:
    """Multi-z 1pvar sweep via 2D Sobol over `(θ_param, z)` jointly.

    Sobol-scatter `n_total` points in the 2D unit square; map column 0 to
    the param's prior `(lo, hi)` and column 1 to `(z_min, z_max)` (continuous
    z, so PySR can learn continuous z-evolution rather than memorizing per-z
    fits). For each draw, predict P_F at LF and HF emulators on `k_grid`.

    Sobol over linspace because the user asked for it: linspace-per-z would
    just repeat the same θ values across z bins, giving PySR no joint
    information about how θ-effect varies with z.
    """
    if param_name not in PARAM_NAMES:
        raise KeyError(f"Unknown PRIYA parameter {param_name!r}.")
    from scipy.stats import qmc

    p = get_param(param_name)
    fid = np.array(fiducial_vector(), dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    n_k = k_grid.size
    idx = PARAM_NAMES.index(param_name)

    sampler = qmc.Sobol(d=2, seed=seed)
    u = sampler.random(n=n_total)  # (n_total, 2) in [0, 1]
    theta_samples = p.prior[0] + (p.prior[1] - p.prior[0]) * u[:, 0]
    # Snap Sobol z to the kodiaq emulator's discrete z-grid (Δz=0.2, from
    # `lyaemu.gp_wrap`). The (θ, z) decorrelation is preserved — each pair
    # is unique even when z values repeat across rows.
    z_grid_kodiaq = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    z_grid_in_range = z_grid_kodiaq[(z_grid_kodiaq >= z_min - 1e-6) & (z_grid_kodiaq <= z_max + 1e-6)]
    z_continuous = z_min + (z_max - z_min) * u[:, 1]
    z_samples = z_grid_in_range[
        np.argmin(np.abs(z_continuous[:, None] - z_grid_in_range[None, :]), axis=1)
    ]

    flux_lf = np.empty((n_total, n_k), dtype=float)
    flux_hf = np.empty((n_total, n_k), dtype=float)
    params_arr = np.tile(fid, (n_total, 1))
    params_arr[:, idx] = theta_samples
    for i in range(n_total):
        theta = params_arr[i]
        flux_lf[i] = np.asarray(gp_lf.predict(theta, k_grid, float(z_samples[i])), dtype=float)
        flux_hf[i] = np.asarray(gp_hf.predict(theta, k_grid, float(z_samples[i])), dtype=float)

    kfkms = np.tile(k_grid, (n_total, 1))
    return dict(
        params_lf=params_arr.copy(),
        params_hf=params_arr.copy(),
        kfkms_lf_z=kfkms.copy(),
        kfkms_hf_z=kfkms.copy(),
        flux_lf_z=flux_lf,
        flux_hf_z=flux_hf,
        z_per_row=z_samples,
        kfkms_lf_min=float(k_grid.min()),
        kfkms_lf_max=float(k_grid.max()),
        sobol_seed=seed,
    )


def _build_training_matrix_multiz(
    *,
    payload: dict,
    param_idx: int,
    norm,
    z_min: float,
    z_max: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, float], dict[str, np.ndarray]]:
    """Stack LF + HF for multi-z into 4-column inputs.

    Output columns: `[θ_norm, k_norm, resolution, z_norm]`. flux_norm uses
    `norm`, which can be a single `NormalizationSpec` (one (mean, std)
    across all z) OR a `MultiZNormalizationSpec` (per-z (mean, std) — the
    flux_norm target then has uniform amplitude across z bins, which is
    what PySR needs to learn balanced z-dependence).
    """
    from priya_forecast.models.normalization import MultiZNormalizationSpec

    flux_lf = payload["flux_lf_z"]   # (n_total, n_k)
    flux_hf = payload["flux_hf_z"]
    k_lf = payload["kfkms_lf_z"]
    k_hf = payload["kfkms_hf_z"]
    params_lf = payload["params_lf"]
    z_per_row = payload["z_per_row"]

    if isinstance(norm, MultiZNormalizationSpec):
        # Per-(z, k) normalization: each row uses its row's z.
        mean_per_row = np.zeros_like(flux_lf)
        std_per_row = np.zeros_like(flux_lf)
        k_use = k_lf[0]
        for r in range(flux_lf.shape[0]):
            zi = norm._z_index(float(z_per_row[r]))
            mean_per_row[r] = np.interp(k_use, norm.k_grid, norm.mean_flux[zi])
            std_per_row[r] = np.interp(k_use, norm.k_grid, norm.std_flux[zi])
        flux_lf_norm = (flux_lf - mean_per_row) / std_per_row
        flux_hf_norm = (flux_hf - mean_per_row) / std_per_row
    else:
        # Single (mean, std) across all (z, sims) — simpler but produces
        # uneven flux_norm amplitude when P_F scale varies with z.
        mean_k = np.interp(k_lf[0], norm.k_grid, norm.mean_flux)
        std_k = np.interp(k_lf[0], norm.k_grid, norm.std_flux)
        flux_lf_norm = (flux_lf - mean_k[None, :]) / std_k[None, :]
        flux_hf_norm = (flux_hf - mean_k[None, :]) / std_k[None, :]

    x_param_lf = np.repeat(params_lf[:, param_idx, None], k_lf.shape[1], axis=1)
    x_param_min = float(x_param_lf.min())
    x_param_max = float(x_param_lf.max())
    x_param_lf_norm = (x_param_lf - x_param_min) / (x_param_max - x_param_min)
    k_min = float(k_lf.min())
    k_max = float(k_lf.max())
    k_lf_norm = (k_lf - k_min) / (k_max - k_min)

    z_per_row_lf_2d = np.repeat(z_per_row[:, None], k_lf.shape[1], axis=1)
    z_norm_lf_2d = (z_per_row_lf_2d - z_min) / (z_max - z_min)

    n = flux_lf.size
    res_lf = np.full(n, LF_RESOLUTION)
    res_hf = np.full(n, HF_RESOLUTION)
    X_lf = np.column_stack([
        x_param_lf_norm.ravel(), k_lf_norm.ravel(), res_lf, z_norm_lf_2d.ravel(),
    ])
    X_hf = np.column_stack([
        x_param_lf_norm.ravel(), k_lf_norm.ravel(), res_hf, z_norm_lf_2d.ravel(),
    ])
    Y_lf = flux_lf_norm.ravel()
    Y_hf = flux_hf_norm.ravel()

    X_act = np.vstack([X_lf, X_hf])
    Y_act = np.concatenate([Y_lf, Y_hf])

    return (
        X_act, Y_act,
        dict(x_param_min=x_param_min, x_param_max=x_param_max,
             k_min=k_min, k_max=k_max, z_min=z_min, z_max=z_max),
        dict(flux_lf=flux_lf, flux_hf=flux_hf,
             flux_lf_norm=flux_lf_norm, flux_hf_norm=flux_hf_norm,
             k_lf=k_lf, k_hf=k_hf, z_per_row=z_per_row),
    )


def compute_local_normalization_multiz(
    *,
    flux_lf_z: np.ndarray,
    z_per_row: np.ndarray,
    z_grid: np.ndarray,
    k_grid: np.ndarray,
    gp_lf=None,
    fid: np.ndarray | None = None,
    param_min: float = 0.0,
    param_max: float = 1.0,
):
    """Per-(z, k) normalization with **at-fid anchor** for multi-z fits.

    The "mean" we subtract per (z, k) is the LF GP prediction *at the
    fiducial parameter vector*, not the empirical mean across the Sobol
    sims. This anchors the training target at zero when θ = fid_phys, so
    PySR is forced to use x0 (param) dependence to fit non-zero values —
    crucial for multi-z runs where the genetic search would otherwise
    latch onto k/z/res-only patterns and drop x0 (observed: 6/11 params
    in our first multi-z run lost x0 dependence with empirical-mean
    anchoring).

    `std_k(z)` is still the empirical std across the Sobol sample at
    that z.
    """
    from priya_forecast.models.normalization import MultiZNormalizationSpec
    flux_lf_z = np.asarray(flux_lf_z, dtype=float)
    z_per_row = np.asarray(z_per_row, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    z_grid = np.asarray(z_grid, dtype=float)
    n_z = z_grid.size
    n_k = k_grid.size
    if gp_lf is None or fid is None:
        raise ValueError(
            "compute_local_normalization_multiz now requires `gp_lf` and "
            "`fid` so the per-z mean is anchored at LF GP(fid, k, z). "
            "Pass them through from refit_1d_multiz_for_param."
        )
    fid = np.asarray(fid, dtype=float)
    mean = np.zeros((n_z, n_k), dtype=float)
    std = np.zeros((n_z, n_k), dtype=float)
    for zi, z in enumerate(z_grid):
        mean[zi] = np.asarray(gp_lf.predict(fid, k_grid, float(z)), dtype=float)
        mask = np.isclose(z_per_row, z, atol=1e-3)
        if not mask.any():
            raise ValueError(f"No payload rows at z={z}.")
        std[zi] = flux_lf_z[mask].std(axis=0, ddof=0)
    std = np.where(std > 0, std, 1.0)
    return MultiZNormalizationSpec(
        param_min=float(param_min), param_max=float(param_max),
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        z_grid=z_grid, mean_flux=mean, std_flux=std, k_grid=k_grid,
    )


def compute_local_normalization(
    *,
    flux_lf_z: np.ndarray,
    k_grid: np.ndarray,
    mean_flux_global: np.ndarray | None = None,
    param_min: float = 0.0,
    param_max: float = 1.0,
) -> NormalizationSpec:
    """Per-param 1D-local std normalization (Option B).

    Compute `std_k_local` from the LF 1pvar slice (50 sims, n_k bins), and
    optionally use a `mean_flux_global` (from the multi-D Sobol) for the
    mean. If `mean_flux_global` is None, fall back to per-param 1D-local
    mean — which is what the student's `pysr_mf_given.py` does in its
    "fallback" path.

    Returns a `NormalizationSpec` whose `denormalize_flux` round-trips
    `flux_norm = (P_F − mean) / std_local` back to raw P_F.
    """
    flux_lf_z = np.asarray(flux_lf_z, dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)
    if flux_lf_z.shape[1] != k_grid.size:
        raise ValueError(
            f"flux_lf_z width {flux_lf_z.shape[1]} != k_grid size {k_grid.size}."
        )
    std_k_local = flux_lf_z.std(axis=0, ddof=0)
    std_k_local = np.where(std_k_local > 0, std_k_local, 1.0)
    if mean_flux_global is None:
        mean_k = flux_lf_z.mean(axis=0)
    else:
        # mean_flux_global must already live on the same k_grid as the
        # local std. Earlier code attempted np.interp(k_grid,
        # arange(len(mean_flux_global)), mean_flux_global) which mixes
        # index-coordinates and physical k-values — that silently clamps
        # every output to mean_flux_global[0]. We require matching
        # lengths and assign directly; a length mismatch is a caller bug.
        mean_flux_global = np.asarray(mean_flux_global, dtype=float)
        if mean_flux_global.shape != k_grid.shape:
            raise ValueError(
                "mean_flux_global must be on the same k_grid as flux_lf_z; "
                f"got shape {mean_flux_global.shape}, expected {k_grid.shape}."
            )
        mean_k = mean_flux_global
    return NormalizationSpec(
        param_min=float(param_min), param_max=float(param_max),
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        mean_flux=mean_k, std_flux=std_k_local, k_grid=k_grid,
    )


def refit_1d_multiz_for_param(
    *,
    param_name: str,
    z_min: float,
    z_max: float,
    k_grid: np.ndarray,
    gp_lf,
    gp_hf,
    n_total: int = 225,
    pysr_kwargs: dict | None = None,
    seed: int = 42,
) -> Refit1DResult:
    """Multi-z 1D PySR refit: one equation in `(θ_norm, k_norm, res, z_norm)`.

    Training set: 2D Sobol over `(θ_param, z)` with `n_total` draws in
    `(prior_lo, prior_hi) × (z_min, z_max)`. LF + HF stacked → 2·n_total
    rows × n_k k-bins. PySR fits a single equation that captures the
    joint (θ, k, z) structure plus the LF→HF resolution lift.
    """
    if param_name not in PARAM_NAMES:
        raise KeyError(f"Unknown PRIYA parameter {param_name!r}.")
    from pysr import PySRRegressor  # type: ignore[import-not-found]

    payload = _generate_1pvar_multiz_inline(
        gp_lf=gp_lf, gp_hf=gp_hf, param_name=param_name,
        z_min=z_min, z_max=z_max, k_grid=k_grid, n_total=n_total, seed=seed,
    )
    p_meta = get_param(param_name)
    # Per-z, at-fid-anchored normalization. Forces x0 dependence in the eq.
    z_grid_kodiaq = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    z_grid_in_range = z_grid_kodiaq[
        (z_grid_kodiaq >= z_min - 1e-6) & (z_grid_kodiaq <= z_max + 1e-6)
    ]
    fid = np.array(fiducial_vector(), dtype=float)
    norm = compute_local_normalization_multiz(
        flux_lf_z=payload["flux_lf_z"], z_per_row=payload["z_per_row"],
        z_grid=z_grid_in_range, k_grid=k_grid,
        gp_lf=gp_lf, fid=fid,
        param_min=float(p_meta.prior[0]), param_max=float(p_meta.prior[1]),
    )

    param_idx = PARAM_NAMES.index(param_name)
    X_act, Y_act, ranges, fidelity_arrays = _build_training_matrix_multiz(
        payload=payload, param_idx=param_idx, norm=norm,
        z_min=z_min, z_max=z_max,
    )

    args = dict(DEFAULT_PYSR_KWARGS)
    args.update(pysr_kwargs or {})
    args["random_state"] = seed
    t0 = time.time()
    model = PySRRegressor(**args)
    model.fit(X_act, Y_act.reshape(-1, 1))
    elapsed = time.time() - t0
    pareto = model.equations_
    best_idx = int(pareto["loss"].idxmin())

    z_center = float((z_min + z_max) / 2.0)
    result = Refit1DResult(
        param_name=param_name, z=z_center,
        equation_str=str(pareto.iloc[best_idx]["equation"]),
        pareto_complexity=int(pareto.iloc[best_idx]["complexity"]),
        pareto_loss=float(pareto.iloc[best_idx]["loss"]),
        pareto_complexities=pareto["complexity"].astype(int).tolist(),
        pareto_losses=pareto["loss"].astype(float).tolist(),
        x_param_min=ranges["x_param_min"], x_param_max=ranges["x_param_max"],
        k_min=ranges["k_min"], k_max=ranges["k_max"],
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        fid_value=p_meta.fid,
        norm=norm,
        k_grid=np.asarray(k_grid, dtype=float),
        wall_time_s=elapsed,
        lf_train_mean_rel_err=float("nan"),
        hf_train_mean_rel_err=float("nan"),
        lf_train_max_rel_err=float("nan"),
        hf_train_max_rel_err=float("nan"),
        z_min=z_min, z_max=z_max,
    )
    diagnostics = _validate_per_fidelity_from_payload_multiz(
        result=result, payload=payload, param_idx=param_idx,
    )
    result.lf_train_mean_rel_err = diagnostics["lf_mean"]
    result.hf_train_mean_rel_err = diagnostics["hf_mean"]
    result.lf_train_max_rel_err = diagnostics["lf_max"]
    result.hf_train_max_rel_err = diagnostics["hf_max"]
    return result


def _validate_per_fidelity_from_payload_multiz(
    *, result: Refit1DResult, payload: dict, param_idx: int,
) -> dict[str, float]:
    """Mean/max rel-err of the multi-z refit on each fidelity slice.

    Loops over each (sim, z) row in the payload and predicts via
    `result.predict(theta, k, z, resolution)`.
    """
    flux_lf = payload["flux_lf_z"]
    flux_hf = payload["flux_hf_z"]
    k_lf = payload["kfkms_lf_z"]
    k_hf = payload["kfkms_hf_z"]
    z_per_row = payload["z_per_row"]
    thetas = payload["params_lf"][:, param_idx]

    pred_lf = np.empty_like(flux_lf)
    pred_hf = np.empty_like(flux_hf)
    for i in range(flux_lf.shape[0]):
        pred_lf[i] = result.predict(
            theta_phys=float(thetas[i]), k=k_lf[i],
            resolution=result.lf_resolution, z=float(z_per_row[i]),
        )
        pred_hf[i] = result.predict(
            theta_phys=float(thetas[i]), k=k_hf[i],
            resolution=result.hf_resolution, z=float(z_per_row[i]),
        )
    rel_lf = np.abs(pred_lf - flux_lf) / np.abs(flux_lf)
    rel_hf = np.abs(pred_hf - flux_hf) / np.abs(flux_hf)
    return dict(
        lf_mean=float(rel_lf.mean()), hf_mean=float(rel_hf.mean()),
        lf_max=float(rel_lf.max()), hf_max=float(rel_hf.max()),
    )


def refit_1d_for_param(
    *,
    param_name: str,
    z: float,
    k_grid: np.ndarray,
    gp_lf=None,
    gp_hf=None,
    norm: NormalizationSpec | None = None,
    mean_flux_global: np.ndarray | None = None,
    n_points: int = 50,
    data_dir: str | Path | None = None,
    pysr_kwargs: dict | None = None,
    seed: int = 42,
) -> Refit1DResult:
    """Train a 1D PySR equation for `param_name`.

    Two data paths:
      - **Inline 1pvar** (default, used with KODIAQ + custom k-grid):
        pass `gp_lf` and `gp_hf`. The 50-point 1pvar sweep is generated
        on the fly via `_generate_1pvar_inline`.
      - **HDF5 1pvar** (legacy, priya emulator only): pass `data_dir`
        pointing to the student's `1pvar/` HDF5 directory.

    Two normalization paths:
      - **Option B local-std** (default): compute per-param 1D-local
        `std_k` from the LF 1pvar slice. Optionally use a multi-D
        `mean_flux_global` for the mean (else per-param 1D mean).
      - **Multi-D global** (legacy): pass `norm=<global multi-D
        NormalizationSpec>` to use it for both training and predict.

    Returns a `Refit1DResult` that always emits raw P_F via its bundled
    `NormalizationSpec`.
    """
    if param_name not in PARAM_NAMES:
        raise KeyError(f"Unknown PRIYA parameter {param_name!r}.")
    from pysr import PySRRegressor  # type: ignore[import-not-found]

    if data_dir is not None:
        payload = _load_1pvar(param_name=param_name, z=z, data_dir=Path(data_dir))
    else:
        if gp_lf is None or gp_hf is None:
            raise ValueError(
                "Pass either `data_dir` (legacy HDF5) or both `gp_lf` and "
                "`gp_hf` (inline 1pvar generation)."
            )
        payload = _generate_1pvar_inline(
            gp_lf=gp_lf, gp_hf=gp_hf, param_name=param_name, z=z,
            k_grid=k_grid, n_points=n_points,
        )

    if norm is None:
        # Option B: per-param 1D-local std from the LF 1pvar slice; mean
        # either from the multi-D Sobol (if `mean_flux_global` provided)
        # or per-param 1D-local. Resulting NormalizationSpec is bundled
        # on the Refit1DResult and used by predict() / the additive
        # combine for the round-trip.
        p_meta = get_param(param_name)
        norm = compute_local_normalization(
            flux_lf_z=payload["flux_lf_z"], k_grid=k_grid,
            mean_flux_global=mean_flux_global,
            param_min=float(p_meta.prior[0]), param_max=float(p_meta.prior[1]),
        )

    param_idx = PARAM_NAMES.index(param_name)
    X_act, Y_act, ranges, fidelity_arrays = _build_training_matrix(
        payload=payload, param_idx=param_idx, global_norm=norm,
    )

    args = dict(DEFAULT_PYSR_KWARGS)
    args.update(pysr_kwargs or {})
    args["random_state"] = seed
    t0 = time.time()
    model = PySRRegressor(**args)
    model.fit(X_act, Y_act.reshape(-1, 1))
    elapsed = time.time() - t0
    pareto = model.equations_
    best_idx = int(pareto["loss"].idxmin())

    result = Refit1DResult(
        param_name=param_name, z=z,
        equation_str=str(pareto.iloc[best_idx]["equation"]),
        pareto_complexity=int(pareto.iloc[best_idx]["complexity"]),
        pareto_loss=float(pareto.iloc[best_idx]["loss"]),
        pareto_complexities=pareto["complexity"].astype(int).tolist(),
        pareto_losses=pareto["loss"].astype(float).tolist(),
        x_param_min=ranges["x_param_min"], x_param_max=ranges["x_param_max"],
        k_min=ranges["k_min"], k_max=ranges["k_max"],
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        fid_value=get_param(param_name).fid,
        norm=norm,
        k_grid=np.asarray(k_grid, dtype=float),
        wall_time_s=elapsed,
        lf_train_mean_rel_err=float("nan"),
        hf_train_mean_rel_err=float("nan"),
        lf_train_max_rel_err=float("nan"),
        hf_train_max_rel_err=float("nan"),
    )
    diagnostics = _validate_per_fidelity_from_payload(
        result=result, payload=payload, param_idx=param_idx,
    )
    result.lf_train_mean_rel_err = diagnostics["lf_mean"]
    result.hf_train_mean_rel_err = diagnostics["hf_mean"]
    result.lf_train_max_rel_err = diagnostics["lf_max"]
    result.hf_train_max_rel_err = diagnostics["hf_max"]
    return result
