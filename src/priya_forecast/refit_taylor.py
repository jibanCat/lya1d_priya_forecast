"""Additive 1st-order Taylor combine for per-param 1D PySR refits.

Replicates the recipe in the student's
`InferenceLyaData/mf_*.py` scripts (`mf_dtau0_ap.py`,
`mf_dtau0_ap_ns_hf.py`, `mf_herei_alphaq.py` — same pattern):

  1. Per parameter `i`, train a 1D PySR equation on
     `flux_norm = (P_F - mean_k) / std_k` with inputs
     `(theta_i_norm, k_norm)` both in [0, 1]. (See `refit_1d_pysr.py`.)
  2. Combine multi-D **in normalized space**:

         P_norm(theta, k) = Σ_i [ eq_i(theta_i_norm, k_norm)
                                  - eq_i(fid_i_norm,  k_norm) ]
                          + (1/n) Σ_i eq_i(fid_i_norm, k_norm)

     The first sum is a 1st-order Taylor: each parameter contributes its
     own deviation from its own fiducial. The second term is the
     "constant background" — the average of the per-param fiducial
     evaluations — added once.

  3. De-normalize ONCE at the end with a single multi-D `(mean_k, std_k)`:

         P_F(theta, k) = P_norm(theta, k) · std_k_global + mean_k_global

The multi-D `(mean_k, std_k)` are computed by Sobol-sampling the GP over
the full prior cube of the params being varied (see
`compute_global_normalization`).

This is the literal "1st-order combine" baseline the user asked for.
Compared to the multiplicative ratio combine in
`scripts/refit_all_11_params._RefitMultiplicativeModel`, the additive
Taylor combine is what the student's published 2D / 3D scripts actually
do — and is the one to compare against the GP for the paper's headline
forecast number.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from priya_forecast.models.base import P1DModel
from priya_forecast.models.normalization import NormalizationSpec
from priya_forecast.parameters import (
    PARAM_NAMES,
    fiducial_vector,
    get_param,
    prior_bounds,
)
from priya_forecast.refit_1d_pysr import Refit1DResult


def compute_global_normalization(
    *,
    gp,
    k_grid: np.ndarray,
    z: float,
    n_train: int = 256,
    seed: int = 0,
    varying_params: tuple[str, ...] | None = None,
) -> NormalizationSpec:  # noqa: D401
    """Per-k mean and std of P_F across a Sobol sample of the **LF** emulator.

    Per the student's contract: this normalization comes from the multi-D
    fiducial Sobol of the LF emulator, NOT from per-param 1D data. Both
    LF and HF training flux are normalized with this single global
    `(mean_k, std_k)`, and the additive-Taylor combine de-normalizes
    its output with the same pair.

    Pass `gp = GPModel(fidelity="lf")` to match the student exactly.
    Passing the HF GP also works but produces a slightly different
    per-k profile.

    Parameters mirror the previous signature; see body docstring below.
    """
    """Per-k mean and std of P_F across a Sobol sample of the GP.

    Parameters
    ----------
    gp : P1DModel
        The GP emulator (or any forward model with a `predict(theta, k, z)`).
    k_grid : ndarray, shape (Nk,)
        The k-grid on which the GP is evaluated.
    z : float
        Redshift bin.
    n_train : int
        Sobol-sample count over the prior cube. The student typically uses
        50-200 points; 256 is a safe default for a smooth global average.
    seed : int
        Sobol seed (reproducible).
    varying_params : tuple of str | None
        Subset of `PARAM_NAMES` to vary. Default = all 11. Non-varying
        params stay at their fiducial value.

    Returns
    -------
    NormalizationSpec
        `mean_flux` and `std_flux` are the per-k mean and std over the
        Sobol set. `param_min`/`param_max` are placeholder (0, 1) since
        this spec is used only for de-normalization, not for input
        normalization.
    """
    from scipy.stats import qmc

    if varying_params is None:
        varying_params = PARAM_NAMES
    fid = np.array(fiducial_vector(), dtype=float)
    k_grid = np.asarray(k_grid, dtype=float)

    sampler = qmc.Sobol(d=len(varying_params), seed=seed)
    u = sampler.random(n=n_train)
    bounds = np.array([get_param(p).prior for p in varying_params], dtype=float)
    scaled = bounds[:, 0] + (bounds[:, 1] - bounds[:, 0]) * u  # (n_train, n_var)

    indices = [PARAM_NAMES.index(p) for p in varying_params]
    flux = np.empty((n_train, k_grid.size), dtype=float)
    for i in range(n_train):
        theta = fid.copy()
        for j, idx in enumerate(indices):
            theta[idx] = scaled[i, j]
        flux[i] = gp.predict(theta, k_grid, z)

    mean_flux = flux.mean(axis=0)
    std_flux = flux.std(axis=0, ddof=0)
    std_flux = np.where(std_flux > 0, std_flux, 1.0)

    return NormalizationSpec(
        param_min=0.0,
        param_max=1.0,
        k_min=float(k_grid.min()),
        k_max=float(k_grid.max()),
        mean_flux=mean_flux,
        std_flux=std_flux,
        k_grid=k_grid,
    )


# Per-param fid_norm hardcoded as 0.5 in every student `mf_*.py`. Replicate
# the approximation verbatim; see `student_pysr_contract.md` item 5.
STUDENT_FID_NORM = 0.5

# Resolution evaluated at HF in the multi-D combine; see `mf_*.py`.
HF_RESOLUTION_FOR_COMBINE = 0.8


@dataclass
class MultiZAdditiveTaylorModel(P1DModel):
    """Multi-z extension of `AdditiveTaylorModel(mode='local_anchored')`.

    Per-param refits are 4-input PySR equations `eq(θ_norm, k_norm,
    resolution, z_norm)`. The combine at any z in the trained range is

        P_F(θ, k, z) = P_GP(fid, k, z)
                     + Σ_{i ∈ refit}     [r_i.predict(θ_i, k, z, 0.8)
                                        − r_i.predict(fid_i_phys, k, z, 0.8)]
                     + Σ_{i ∈ gp_slice}  [P_GP(fid except θ_i, k, z)
                                        − P_GP(fid, k, z)]

    Each per-param refit contribution is in physical P_F units via its
    bundled `MultiZNormalizationSpec` (per-z mean/std). At fid every
    deviation cancels; the combine recovers P_GP(fid, k, z) exactly.

    Params with `refits[pname] is None` route through the GP-slice
    fallback. This is how the aggregator drops broken refits (no x0,
    bad rel-err) without poisoning the Fisher: pass `refits[pname] =
    None` and the param's gradient comes from the GP — same code path
    as a never-refit param.
    """

    gp: object                          # HF GP, supports predict(theta, k, z)
    fid: np.ndarray                     # shape (11,)
    refits: dict                        # param_name -> Refit1DResult (multi-z)
    k_grid: np.ndarray
    z_grid: np.ndarray                  # discrete z bins this model serves
    log_space: bool = False

    def __post_init__(self) -> None:
        self.fid = np.asarray(self.fid, dtype=float)
        self.k_grid = np.asarray(self.k_grid, dtype=float)
        self.z_grid = np.asarray(self.z_grid, dtype=float)
        if self.fid.shape != (len(PARAM_NAMES),):
            raise ValueError(
                f"fid must be shape ({len(PARAM_NAMES)},), got {self.fid.shape}."
            )
        # Cache P_GP(fid, k_grid, z) for each z in z_grid.
        self._p_gp_fid_per_z: dict[float, np.ndarray] = {
            float(z): np.asarray(self.gp.predict(self.fid, self.k_grid, float(z)),
                                  dtype=float)
            for z in self.z_grid
        }
        # Cache r_i.predict(fid_i_phys, k_grid, z, 0.8) per (param, z) for
        # params with a refit. Params with `r is None` route through
        # the GP-slice fallback path in predict().
        self._eq_at_fid_pf: dict[tuple[str, float], np.ndarray] = {}
        for pname, r in self.refits.items():
            if r is None:
                continue
            i = PARAM_NAMES.index(pname)
            fid_i_phys = float(self.fid[i])
            for z in self.z_grid:
                self._eq_at_fid_pf[(pname, float(z))] = r.predict(
                    theta_phys=fid_i_phys, k=self.k_grid,
                    resolution=HF_RESOLUTION_FOR_COMBINE, z=float(z),
                )
        if self.log_space:
            self._log_p_gp_fid_per_z: dict[float, np.ndarray] = {}
            for z in self.z_grid:
                pf = self._p_gp_fid_per_z[float(z)]
                if np.any(pf <= 0):
                    raise ValueError(
                        f"log_space combine: GP P_F(fid) at z={float(z)} has "
                        f"non-positive entries — cannot take log."
                    )
                self._log_p_gp_fid_per_z[float(z)] = np.log(pf)
            self._eq_at_fid_logpf: dict[tuple[str, float], np.ndarray] = {}
            for pname, r in self.refits.items():
                if r is None:
                    continue
                i = PARAM_NAMES.index(pname)
                fid_i_phys = float(self.fid[i])
                for z in self.z_grid:
                    self._eq_at_fid_logpf[(pname, float(z))] = r.predict_log(
                        theta_phys=fid_i_phys, k=self.k_grid,
                        resolution=HF_RESOLUTION_FOR_COMBINE, z=float(z),
                    )

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        k = np.asarray(k, dtype=float)
        if not np.allclose(k, self.k_grid):
            raise ValueError(
                "MultiZAdditiveTaylorModel built for a fixed k_grid; pass the "
                "same k_grid the model was constructed with."
            )
        z_key = float(z)
        if not np.any(np.isclose(self.z_grid, z_key, atol=1e-3)):
            raise ValueError(
                f"z={z_key} not in this model's z_grid: {self.z_grid}."
            )
        if self.log_space:
            out_log = self._log_p_gp_fid_per_z[z_key].copy()
            for pname, r in self.refits.items():
                if r is None:
                    continue
                i = PARAM_NAMES.index(pname)
                ti, fi = float(theta[i]), float(self.fid[i])
                if abs(ti - fi) <= max(abs(fi), 1.0) * 1e-12:
                    continue
                log_at_theta = r.predict_log(
                    theta_phys=ti, k=self.k_grid,
                    resolution=HF_RESOLUTION_FOR_COMBINE, z=z_key,
                )
                out_log = out_log + (log_at_theta - self._eq_at_fid_logpf[(pname, z_key)])
            for pname, r in self.refits.items():
                if r is not None:
                    continue
                i = PARAM_NAMES.index(pname)
                ti, fi = float(theta[i]), float(self.fid[i])
                if abs(ti - fi) <= max(abs(fi), 1.0) * 1e-12:
                    continue
                t_only = self.fid.copy()
                t_only[i] = theta[i]
                p_slice = np.asarray(
                    self.gp.predict(t_only, self.k_grid, z_key), dtype=float)
                if np.any(p_slice <= 0):
                    raise ValueError(
                        f"log_space combine: GP slice for {pname!r} at z={z_key} "
                        f"has non-positive P_F — cannot take log."
                    )
                out_log = out_log + (np.log(p_slice) - self._log_p_gp_fid_per_z[z_key])
            return np.exp(out_log)
        p_gp_fid = self._p_gp_fid_per_z[z_key]
        out = p_gp_fid.copy()
        # Per-1D Taylor contribution for params with a refit. Use
        # relative tolerance for the at-fid skip (matches the pattern
        # in `Refit2DPairResult.cross_difference`); strict `==` would
        # miss callers passing `theta = fid + 1e-16` from numpy
        # arithmetic and fall through to a redundant predict +
        # subtraction at machine-precision cancellation.
        for pname, r in self.refits.items():
            if r is None:
                continue
            i = PARAM_NAMES.index(pname)
            ti, fi = float(theta[i]), float(self.fid[i])
            if abs(ti - fi) <= max(abs(fi), 1.0) * 1e-12:
                continue
            p_at_theta = r.predict(
                theta_phys=ti, k=self.k_grid,
                resolution=HF_RESOLUTION_FOR_COMBINE, z=z_key,
            )
            out = out + (p_at_theta - self._eq_at_fid_pf[(pname, z_key)])
        # GP-slice fallback for params explicitly routed away from the
        # per-1D Taylor path (refits[pname] is None). This includes
        # never-refit params AND refits dropped by the aggregator's
        # quality gate.
        for pname, r in self.refits.items():
            if r is not None:
                continue
            i = PARAM_NAMES.index(pname)
            ti, fi = float(theta[i]), float(self.fid[i])
            if abs(ti - fi) <= max(abs(fi), 1.0) * 1e-12:
                continue
            t_only = self.fid.copy()
            t_only[i] = theta[i]
            p_slice = np.asarray(
                self.gp.predict(t_only, self.k_grid, z_key), dtype=float
            )
            out = out + (p_slice - p_gp_fid)
        return out


@dataclass
class AdditiveTaylorModel(P1DModel):
    """Multi-D forward model = student's additive-Taylor combine of 1D PySR refits.

    Per `mf_dtau0_ap_ns_hf.py` and siblings, every per-param equation is
    evaluated at the resolution=HF feature (x2=0.8) and at a hardcoded
    fid_norm=0.5 for the constant-reference term. The combine:

        eq_i_at_theta = predict_normalized(θ_i_phys, k, resolution=0.8)
        eq_i_at_fid05 = eq_at_fid05_norm(0.5, k_norm, 0.8)   # ignores θ
        P_norm(θ, k)  = Σ_i [eq_i_at_theta − eq_i_at_fid05]
                      + (1/n) Σ_i eq_i_at_fid05
        P_F(θ, k)     = P_norm · std_k_global + mean_k_global

    `n` = number of refit params; params without a refit fall back to a
    GP-slice contribution so their marginal Fisher sensitivity remains
    correct (this is on top of the student's design — the user wants
    full 11D refit, but we keep the fallback as a defensive default).

    `mean_k_global, std_k_global` come from
    `compute_global_normalization` (Sobol of the LF emulator).
    """

    gp: object                                   # P1DModel (HF GP)
    fid: np.ndarray                              # shape (11,)
    refits: dict                                 # param_name -> Refit1DResult | None
    global_norm: NormalizationSpec | None        # only used in mode="multi_d"
    k_grid: np.ndarray
    z: float
    mode: str = "local_anchored"                 # "local_anchored" | "multi_d"
    log_space: bool = False

    def __post_init__(self) -> None:
        self.fid = np.asarray(self.fid, dtype=float)
        self.k_grid = np.asarray(self.k_grid, dtype=float)
        if self.fid.shape != (len(PARAM_NAMES),):
            raise ValueError(
                f"fid must be shape ({len(PARAM_NAMES)},), got {self.fid.shape}."
            )
        if self.mode not in ("local_anchored", "multi_d"):
            raise ValueError(
                f"mode must be 'local_anchored' or 'multi_d', got {self.mode!r}."
            )
        if self.log_space and self.mode != "local_anchored":
            raise ValueError(
                "log_space=True is only supported with "
                "mode='local_anchored'."
            )
        # Cache P_GP(fid, k_grid) — used as anchor in 'local_anchored' mode
        # and as the GP-slice baseline for un-refit params.
        self._p_gp_fid = np.asarray(
            self.gp.predict(self.fid, self.k_grid, self.z), dtype=float
        )
        if self.mode == "local_anchored":
            # Option B: per-param 1D-local std (and possibly local mean)
            # in `r.norm`. Each per-param eq predicts P_F directly via its
            # own round-trip. Combine = P_GP(fid) + Σ_i [P_F_i(θ_i) − P_F_i(fid_i)].
            # Cache per-param P_F at fid_i_phys (only in linear mode; log_space
            # uses _eq_at_fid_logpf instead, so skip the n_refits wasted calls).
            self._eq_at_fid_pf: dict[str, np.ndarray] = {}
            if not self.log_space:
                for pname, r in self.refits.items():
                    if r is None:
                        continue
                    i = PARAM_NAMES.index(pname)
                    fid_i_phys = float(self.fid[i])
                    self._eq_at_fid_pf[pname] = r.predict(
                        theta_phys=fid_i_phys, k=self.k_grid,
                        resolution=HF_RESOLUTION_FOR_COMBINE,
                    )
            if self.log_space:
                if np.any(self._p_gp_fid <= 0):
                    raise ValueError(
                        "log_space combine: GP P_F(fid) has non-positive "
                        "entries — cannot take log."
                    )
                self._log_p_gp_fid = np.log(self._p_gp_fid)
                self._eq_at_fid_logpf: dict[str, np.ndarray] = {}
                for pname, r in self.refits.items():
                    if r is None:
                        continue
                    i = PARAM_NAMES.index(pname)
                    self._eq_at_fid_logpf[pname] = r.predict_log(
                        theta_phys=float(self.fid[i]), k=self.k_grid,
                        resolution=HF_RESOLUTION_FOR_COMBINE,
                    )
        else:
            # Multi-D mode (legacy / replication-of-student). Cache
            # `eq_i(0.5_norm, k_norm, 0.8)` (in flux_norm space).
            if self.global_norm is None:
                raise ValueError("mode='multi_d' requires global_norm.")
            self._eq_at_fid05: dict[str, np.ndarray] = {}
            for pname, r in self.refits.items():
                if r is None:
                    continue
                theta_at_fid05_phys = (
                    r.x_param_min
                    + STUDENT_FID_NORM * (r.x_param_max - r.x_param_min)
                )
                self._eq_at_fid05[pname] = r.predict_normalized(
                    theta_phys=theta_at_fid05_phys, k=self.k_grid,
                    resolution=HF_RESOLUTION_FOR_COMBINE,
                )
            self._mean_global = np.interp(
                self.k_grid, self.global_norm.k_grid, self.global_norm.mean_flux
            )
            self._std_global = np.interp(
                self.k_grid, self.global_norm.k_grid, self.global_norm.std_flux
            )

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        k = np.asarray(k, dtype=float)
        if not np.allclose(k, self.k_grid):
            raise ValueError(
                "AdditiveTaylorModel is built for a fixed k_grid. Pass the "
                "same k_grid the model was constructed with, or rebuild."
            )
        if abs(z - self.z) > 1e-6:
            raise ValueError(
                f"AdditiveTaylorModel.predict was built for z={self.z}; got z={z}."
            )

        if self.log_space:
            out_log = self._log_p_gp_fid.copy()
            for pname, r in self.refits.items():
                if r is None:
                    continue
                i = PARAM_NAMES.index(pname)
                if float(theta[i]) == float(self.fid[i]):
                    continue
                log_at_theta = r.predict_log(
                    theta_phys=float(theta[i]), k=self.k_grid,
                    resolution=HF_RESOLUTION_FOR_COMBINE,
                )
                out_log = out_log + (log_at_theta - self._eq_at_fid_logpf[pname])
            for pname, r in self.refits.items():
                if r is not None:
                    continue
                i = PARAM_NAMES.index(pname)
                if float(theta[i]) == float(self.fid[i]):
                    continue
                t_only = self.fid.copy()
                t_only[i] = theta[i]
                p_slice = np.asarray(
                    self.gp.predict(t_only, self.k_grid, self.z), dtype=float
                )
                if np.any(p_slice <= 0):
                    raise ValueError(
                        f"log_space combine: GP slice for {pname!r} has "
                        f"non-positive P_F — cannot take log."
                    )
                out_log = out_log + (np.log(p_slice) - self._log_p_gp_fid)
            return np.exp(out_log)

        if self.mode == "local_anchored":
            # P_F(θ, k) = P_GP(fid, k) + Σ_i [eq_i.predict(θ_i, 0.8) − eq_i.predict(fid_i, 0.8)]
            # Each eq_i.predict() is in physical P_F units (round-trip via
            # its own NormalizationSpec). At θ=fid: deviation is zero,
            # output is P_GP(fid) exactly.
            out = self._p_gp_fid.copy()
            for pname, r in self.refits.items():
                if r is None:
                    continue
                i = PARAM_NAMES.index(pname)
                if float(theta[i]) == float(self.fid[i]):
                    continue
                p_at_theta = r.predict(
                    theta_phys=float(theta[i]), k=self.k_grid,
                    resolution=HF_RESOLUTION_FOR_COMBINE,
                )
                out = out + (p_at_theta - self._eq_at_fid_pf[pname])
            # GP-slice fallback for un-refit params (same as multi_d mode).
            for pname, r in self.refits.items():
                if r is not None:
                    continue
                i = PARAM_NAMES.index(pname)
                if float(theta[i]) == float(self.fid[i]):
                    continue
                t_only = self.fid.copy()
                t_only[i] = theta[i]
                p_slice = np.asarray(
                    self.gp.predict(t_only, self.k_grid, self.z), dtype=float
                )
                out = out + (p_slice - self._p_gp_fid)
            return out

        # mode == "multi_d": legacy student-replicate combine.
        flux_norm_total = np.zeros_like(self.k_grid)
        const_term = np.zeros_like(self.k_grid)
        n_refits = 0
        for pname, r in self.refits.items():
            if r is None:
                continue
            n_refits += 1
            i = PARAM_NAMES.index(pname)
            eq_theta = r.predict_normalized(
                theta_phys=float(theta[i]), k=self.k_grid,
                resolution=HF_RESOLUTION_FOR_COMBINE,
            )
            eq_fid = self._eq_at_fid05[pname]
            flux_norm_total = flux_norm_total + (eq_theta - eq_fid)
            const_term = const_term + eq_fid
        if n_refits > 0:
            flux_norm_total = flux_norm_total + const_term / n_refits
        p_refit = flux_norm_total * self._std_global + self._mean_global
        gp_correction = np.zeros_like(self.k_grid)
        for pname, r in self.refits.items():
            if r is not None:
                continue
            i = PARAM_NAMES.index(pname)
            if float(theta[i]) == float(self.fid[i]):
                continue
            t_only = self.fid.copy()
            t_only[i] = theta[i]
            p_slice = np.asarray(
                self.gp.predict(t_only, self.k_grid, self.z), dtype=float
            )
            gp_correction = gp_correction + (p_slice - self._p_gp_fid)
        return p_refit + gp_correction
