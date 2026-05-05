"""Phase 2: per-pair PySR cross-coupling fits + rank-additive ANOVA combine.

Design (from docs/PAIR_FIT_PLAN.md):

  P̂(θ; k, z, r) = P̂_phase1(θ; k, z, r)   ← per-1D + additive Taylor
                + Σ_{(i,j) ∈ pairs} cross_diff_{ij}(θ_i, θ_j; k, z, r)

  cross_diff_{ij}(θ_i, θ_j) = Ĝ_{ij}(θ_i, θ_j)
                            − Ĝ_{ij}(θ_i, fid_j)
                            − Ĝ_{ij}(fid_i, θ_j)
                            + Ĝ_{ij}(fid_i, fid_j)

The cross-difference is the standard ANOVA pure 2-way interaction. Three
properties:

1. **Exact at fid** — every bracketed term is 0; P̂ ≡ P_GP^HF(fid).
2. **Each pair adds an independent gradient direction** in the (θ_i, θ_j)
   plane, orthogonal to all per-1D gradient directions. Fisher rank
   stays full by construction.
3. **Graceful degradation** — if pair signal is weak, Ĝ_ij fits to ≈ 0
   and cross_diff is ≈ 0. Adding a redundant pair costs compute but
   cannot harm Fisher.

Each `Refit2DPairResult` carries one PySR equation over 5 inputs
`(θ_i_norm, θ_j_norm, k_norm, resolution, z_norm)`, trained on the
residual `P_GP(θ_i, θ_j, others=fid; k, z, r) − P̂_phase1(θ_i, θ_j;
k, z, r)` with per-(z, k) normalization (same convention as per-1D).

`MultiZPairCoupledModel` composes a Phase-1 hybrid with a list of pair
refits.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from priya_forecast.models.base import P1DModel
from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.refit_1d_pysr import HF_RESOLUTION, LF_RESOLUTION
from priya_forecast.refit_taylor import MultiZAdditiveTaylorModel


HF_RESOLUTION_FOR_COMBINE = HF_RESOLUTION  # 0.8


@dataclass
class Refit2DPairResult:
    """Per-pair PySR equation Ĝ_{ij}(θ_i, θ_j; k, resolution, z).

    The eq has 5 inputs in normalized space:
        x0 = θ_i_norm   (in [0, 1])
        x1 = θ_j_norm   (in [0, 1])
        x2 = k_norm     (in [0, 1])
        x3 = resolution (LF=0.4, HF=0.8)
        x4 = z_norm     (in [0, 1])

    The output of the eq is normalized flux residual; `predict()`
    de-normalizes via `MultiZNormalizationSpec.denormalize_flux` (same
    per-(z, k) convention as per-1D). `cross_difference()` returns the
    ANOVA pure-2-way-interaction term — zero whenever either θ_i = fid_i
    or θ_j = fid_j.
    """

    pair_names: tuple[str, str]                # ("tau0", "ns")
    equation_str: str                          # raw PySR str (x0..x4)
    pareto_complexity: int
    pareto_loss: float
    pareto_complexities: list[int]
    pareto_losses: list[float]
    x_pair_min: tuple[float, float]            # (θ_i_min, θ_j_min) -- physical
    x_pair_max: tuple[float, float]            # (θ_i_max, θ_j_max)
    k_min: float
    k_max: float
    fid_pair: tuple[float, float]              # (fid_i_phys, fid_j_phys)
    z_min: float
    z_max: float
    norm: MultiZNormalizationSpec              # per-(z, k) de-normalization
    k_grid: np.ndarray                         # training k-grid (for denorm)
    lf_resolution: float = LF_RESOLUTION
    hf_resolution: float = HF_RESOLUTION
    # Diagnostic
    lf_train_mean_rel_err: float = float("nan")
    hf_train_mean_rel_err: float = float("nan")
    lf_train_max_rel_err: float = float("nan")
    hf_train_max_rel_err: float = float("nan")
    wall_time_s: float = float("nan")
    # Cached lambdified callable (lazy; compiled on first call).
    _fn_cache: Any = field(default=None, init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        self.k_grid = np.asarray(self.k_grid, dtype=float)
        if not (len(self.pair_names) == 2 and len(self.x_pair_min) == 2
                and len(self.x_pair_max) == 2 and len(self.fid_pair) == 2):
            raise ValueError("pair_names/x_pair_*/fid_pair must all be length-2.")
        for nm in self.pair_names:
            if nm not in PARAM_NAMES:
                raise ValueError(f"unknown param name {nm!r} in pair_names.")

    # The lambdified callable is rebuilt on first call after unpickling.
    # Strip it from the pickle state so `pickle.dump(self)` doesn't try to
    # serialize a sympy `_lambdifygenerated` function (not picklable).
    def __getstate__(self) -> dict:
        state = self.__dict__.copy()
        state["_fn_cache"] = None
        return state

    def __setstate__(self, state: dict) -> None:
        # Backward-compat: ensure `_fn_cache` is present even if a pre-cache
        # pickle is loaded.
        state.setdefault("_fn_cache", None)
        self.__dict__.update(state)

    # --- internal: lambdify cache --------------------------------------

    def _ensure_fn(self):
        if self._fn_cache is not None:
            return self._fn_cache
        import sympy as sp
        expr = sp.sympify(self.equation_str)
        all_syms = [sp.Symbol(f"x{i}") for i in range(5)]
        fn = sp.lambdify(
            all_syms, expr,
            modules=[{"inv": lambda x: 1.0 / x}, "numpy"],
        )
        self._fn_cache = fn
        return fn

    # --- normalization helpers -----------------------------------------

    def _norm_pair(self, theta_pair_phys: tuple[float, float]) -> tuple[float, float]:
        ti, tj = float(theta_pair_phys[0]), float(theta_pair_phys[1])
        ti_n = (ti - self.x_pair_min[0]) / (self.x_pair_max[0] - self.x_pair_min[0])
        tj_n = (tj - self.x_pair_min[1]) / (self.x_pair_max[1] - self.x_pair_min[1])
        return ti_n, tj_n

    def _norm_k(self, k: np.ndarray) -> np.ndarray:
        if self.k_max == self.k_min:
            return np.zeros_like(np.asarray(k, dtype=float))
        return (np.asarray(k, dtype=float) - self.k_min) / (self.k_max - self.k_min)

    def _norm_z(self, z: float) -> float:
        if self.z_max == self.z_min:
            return 0.0
        return (float(z) - self.z_min) / (self.z_max - self.z_min)

    # --- predict in normalized + physical space ------------------------

    def predict_normalized(
        self, theta_pair_phys: tuple[float, float], k: np.ndarray,
        resolution: float, z: float,
    ) -> np.ndarray:
        fn = self._ensure_fn()
        ti_n, tj_n = self._norm_pair(theta_pair_phys)
        k_arr = np.asarray(k, dtype=float)
        k_n = self._norm_k(k_arr)
        z_n = self._norm_z(z)
        n_k = k_arr.size
        args = [
            np.full(n_k, float(ti_n)),
            np.full(n_k, float(tj_n)),
            k_n,
            np.full(n_k, float(resolution)),
            np.full(n_k, float(z_n)),
        ]
        with np.errstate(all="ignore"):
            out = np.asarray(fn(*args), dtype=float)
        if out.ndim == 0:
            out = np.full(n_k, float(out))
        return np.broadcast_to(out, k_arr.shape).copy()

    def predict(
        self, theta_pair_phys: tuple[float, float], k: np.ndarray,
        resolution: float, z: float,
    ) -> np.ndarray:
        norm_val = self.predict_normalized(theta_pair_phys, k, resolution, z)
        return self.norm.denormalize_flux(norm_val, k, z)

    # --- ANOVA pure 2-way interaction ----------------------------------

    def cross_difference(
        self, theta_pair_phys: tuple[float, float], k: np.ndarray,
        resolution: float, z: float,
    ) -> np.ndarray:
        """Ĝ(θ_i, θ_j) − Ĝ(θ_i, fid_j) − Ĝ(fid_i, θ_j) + Ĝ(fid_i, fid_j).

        Zero whenever either θ_i = fid_i or θ_j = fid_j. Adds an
        independent gradient direction in the (θ_i, θ_j) plane,
        orthogonal to per-1D directions.
        """
        ti, tj = float(theta_pair_phys[0]), float(theta_pair_phys[1])
        fi, fj = float(self.fid_pair[0]), float(self.fid_pair[1])
        # Short-circuit when on either axis (cross_diff is exactly 0).
        if ti == fi or tj == fj:
            return np.zeros_like(np.asarray(k, dtype=float))
        g_ij = self.predict((ti, tj), k, resolution, z)
        g_if = self.predict((ti, fj), k, resolution, z)
        g_fj = self.predict((fi, tj), k, resolution, z)
        g_ff = self.predict((fi, fj), k, resolution, z)
        return g_ij - g_if - g_fj + g_ff

    # --- introspection -------------------------------------------------

    def feature_count(self) -> int:
        """Number of distinct `xN` (N < 5) referenced in the eq.

        Useful for Pareto picks that prefer eqs using both θ features.
        """
        return sum(
            1 for i in range(5)
            if re.search(rf"\bx{i}\b", self.equation_str) is not None
        )


# ----------------------------------------------------------------------
# Composed model: Phase-1 hybrid + pair cross-differences.
# ----------------------------------------------------------------------

@dataclass
class MultiZPairCoupledModel(P1DModel):
    """Phase-2 hybrid: Phase-1 per-1D + Σ pair cross-differences.

    `base` is the Phase-1 `MultiZAdditiveTaylorModel` (per-1D + Taylor).
    `pairs` is a list of `Refit2DPairResult` objects. At each predict()
    call, we sum the base prediction with `Σ pair.cross_difference(θ_i,
    θ_j; k, resolution, z)`.

    Properties (by construction):
    - At fid: every cross_diff is 0; P̂ ≡ base ≡ GP at fid.
    - On any axis (only one θ_i perturbed): every pair involving that
      θ_i either has θ_j = fid_j (so cross_diff = 0) or doesn't include
      θ_i. Phase-1 prediction is exactly recovered.
    - Off-fid: each pair's cross_diff adds an independent gradient
      direction → Fisher rank ≥ rank(Phase 1) + |pairs|.
    """

    base: MultiZAdditiveTaylorModel
    pairs: list[Refit2DPairResult]

    def __post_init__(self) -> None:
        # Sanity: every pair's params must be valid PARAM_NAMES; pair
        # fid_pair must match the base model's `fid` for those params.
        if self.base.fid.shape != (len(PARAM_NAMES),):
            raise ValueError(
                f"base.fid shape {self.base.fid.shape} != ({len(PARAM_NAMES)},)."
            )
        for pair in self.pairs:
            for k_, name in enumerate(pair.pair_names):
                gi = PARAM_NAMES.index(name)
                if not np.isclose(pair.fid_pair[k_], self.base.fid[gi]):
                    raise ValueError(
                        f"pair {pair.pair_names} fid_pair[{k_}]="
                        f"{pair.fid_pair[k_]} != base.fid[{name}]="
                        f"{self.base.fid[gi]}; pair was trained at a "
                        f"different fid than the base model uses."
                    )

    @property
    def fid(self) -> np.ndarray:
        return self.base.fid

    @property
    def k_grid(self) -> np.ndarray:
        return self.base.k_grid

    @property
    def z_grid(self) -> np.ndarray:
        return self.base.z_grid

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        out = np.asarray(self.base.predict(theta, k, z), dtype=float)
        if not self.pairs:
            return out
        theta = np.asarray(theta, dtype=float)
        for pair in self.pairs:
            i_g = PARAM_NAMES.index(pair.pair_names[0])
            j_g = PARAM_NAMES.index(pair.pair_names[1])
            ti = float(theta[i_g])
            tj = float(theta[j_g])
            out = out + pair.cross_difference(
                (ti, tj), k, HF_RESOLUTION_FOR_COMBINE, float(z),
            )
        return out
