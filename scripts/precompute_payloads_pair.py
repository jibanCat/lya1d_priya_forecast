"""Phase 2 / iter 3: precompute pair-fit payloads.

For each pair `(name_i, name_j)`, Sobol-sample 3D over `(θ_i, θ_j, z)`
with all other θ at fid, query LF + HF GPs, compute the Phase-1 hybrid
prediction at the same points, and store the residual

    R(θ_i, θ_j; k, r, z) = P_GP_r(θ_i, θ_j, others=fid; k, z)
                         − P̂_phase1_r(θ_i, θ_j, others=fid; k, z)

separately for LF (r=0.4) and HF (r=0.8). At θ=fid the residual is 0
exactly (Phase-1 hybrid ≡ GP at fid by construction); off-fid the
residual is the cross-coupling signal that per-1D + additive-Taylor
misses.

Per-(z, k) normalization: `std_per_(z, k)` from the empirical
residual sample at each z; `mean_per_(z, k) = 0` since residual is
~0 at fid. Uses the same `MultiZNormalizationSpec` as per-1D so
`Refit2DPairResult.predict` can de-normalize via the existing API.

Saves `payloads_pair/<name_i>_<name_j>.pkl`.

Run:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/precompute_payloads_pair.py \\
          --phase1-refits-dir results/refit_optionC_z2.6-4.2_phase1_5/refits \\
          --pairs tau0,ns herei,alphaq \\
          --output results/refit_pair_z2.6-4.2/payloads
"""

from __future__ import annotations

import argparse
import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(Path.home() / ".julia_env"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(Path.home() / ".julia"))

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.parameters import (
    PARAM_NAMES,
    fiducial_vector,
    get_param,
)
from priya_forecast.refit_1d_pysr import HF_RESOLUTION, LF_RESOLUTION
from priya_forecast.refit_taylor import MultiZAdditiveTaylorModel


def _load_phase1_refits(refits_dir: Path) -> dict:
    """Load all per-1D refits from a directory; missing → None."""
    refits = {pn: None for pn in PARAM_NAMES}
    for pname in PARAM_NAMES:
        path = refits_dir / f"{pname}.pkl"
        if path.exists():
            with open(path, "rb") as fh:
                refits[pname] = pickle.load(fh)
    return refits


def _gate_refits(refits: dict) -> dict:
    """Apply the same gate as multi_z_aggregate.py: drop refits without x0
    or with LF/HF rel-err >= 5%, replacing them with None so the Phase-1
    hybrid routes them through GP-slice fallback."""
    REL_ERR_THRESHOLD = 0.05
    out = dict(refits)
    for pname, r in list(refits.items()):
        if r is None:
            continue
        has_x0 = "x0" in r.equation_str
        lf_ok = (np.isfinite(r.lf_train_mean_rel_err)
                 and r.lf_train_mean_rel_err < REL_ERR_THRESHOLD)
        hf_ok = (np.isfinite(r.hf_train_mean_rel_err)
                 and r.hf_train_mean_rel_err < REL_ERR_THRESHOLD)
        if not (has_x0 and lf_ok and hf_ok):
            out[pname] = None
    return out


def _generate_pair_payload_inline(
    *,
    gp_lf, gp_hf, hybrid_lf, hybrid_hf,
    pair_names: tuple[str, str],
    z_min: float, z_max: float,
    k_grid: np.ndarray,
    n_total: int = 256,
    seed: int = 0,
) -> dict:
    """3D Sobol over `(θ_i, θ_j, z)`; query GPs + Phase-1 hybrids; return residuals.

    The GPs are evaluated at `theta = fid; theta[i_g] = θ_i; theta[j_g] =
    θ_j` (all other params at fid). LF and HF stacks are computed
    separately. Phase-1 hybrid is evaluated at the same points; the
    *residual* (GP − hybrid) is what PySR will fit.
    """
    from scipy.stats import qmc

    name_i, name_j = pair_names
    p_i = get_param(name_i)
    p_j = get_param(name_j)
    fid = np.array(fiducial_vector(), dtype=float)
    i_g = PARAM_NAMES.index(name_i)
    j_g = PARAM_NAMES.index(name_j)
    k_grid = np.asarray(k_grid, dtype=float)
    n_k = k_grid.size

    sampler = qmc.Sobol(d=3, seed=seed)
    u = sampler.random(n=n_total)  # (n_total, 3) in [0, 1]
    theta_i = p_i.prior[0] + (p_i.prior[1] - p_i.prior[0]) * u[:, 0]
    theta_j = p_j.prior[0] + (p_j.prior[1] - p_j.prior[0]) * u[:, 1]
    # Snap to discrete kodiaq z grid (Δz=0.2).
    z_grid_kodiaq = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    z_grid_in_range = z_grid_kodiaq[
        (z_grid_kodiaq >= z_min - 1e-6) & (z_grid_kodiaq <= z_max + 1e-6)
    ]
    z_continuous = z_min + (z_max - z_min) * u[:, 2]
    z_samples = z_grid_in_range[
        np.argmin(np.abs(z_continuous[:, None] - z_grid_in_range[None, :]), axis=1)
    ]

    flux_lf = np.empty((n_total, n_k), dtype=float)
    flux_hf = np.empty((n_total, n_k), dtype=float)
    phase1_lf = np.empty((n_total, n_k), dtype=float)
    phase1_hf = np.empty((n_total, n_k), dtype=float)
    for k_i in range(n_total):
        theta = fid.copy()
        theta[i_g] = theta_i[k_i]
        theta[j_g] = theta_j[k_i]
        z = float(z_samples[k_i])
        flux_lf[k_i] = np.asarray(gp_lf.predict(theta, k_grid, z), dtype=float)
        flux_hf[k_i] = np.asarray(gp_hf.predict(theta, k_grid, z), dtype=float)
        phase1_lf[k_i] = np.asarray(hybrid_lf.predict(theta, k_grid, z), dtype=float)
        phase1_hf[k_i] = np.asarray(hybrid_hf.predict(theta, k_grid, z), dtype=float)

    residual_lf = flux_lf - phase1_lf
    residual_hf = flux_hf - phase1_hf

    return dict(
        pair_names=pair_names,
        theta_i=theta_i, theta_j=theta_j,
        z_per_row=z_samples,
        z_grid_in_range=z_grid_in_range,
        k_grid=k_grid,
        flux_lf_z=flux_lf, flux_hf_z=flux_hf,
        phase1_lf_z=phase1_lf, phase1_hf_z=phase1_hf,
        residual_lf_z=residual_lf, residual_hf_z=residual_hf,
        fid_pair=(float(fid[i_g]), float(fid[j_g])),
        x_pair_min=(float(p_i.prior[0]), float(p_j.prior[0])),
        x_pair_max=(float(p_i.prior[1]), float(p_j.prior[1])),
        z_min=float(z_min), z_max=float(z_max),
    )


def _compute_pair_normalization(
    *, payload: dict,
) -> MultiZNormalizationSpec:
    """Per-(z, k) std of residual; mean=0.

    Residual is ~0 at fid by construction (Phase-1 hybrid ≡ GP at fid).
    For PySR training we want a target with O(1) magnitude per (z, k).

    **Normalization choice (fixed 2026-05-05, per PR #2 review item #3)**:
    use `max(std_LF, std_HF)` per (z, k) so that BOTH the LF and HF
    stacks have a normalized residual amplitude ≤ 1 in absolute value.
    The previous version used `std_LF` only and reused it for HF, which
    biased PySR's loss toward HF when std_HF > std_LF (empirically up
    to 2.5× for some (z, k) bins) — the fit then preferentially
    minimized HF residual at the cost of LF.

    Using `max(std_LF, std_HF)` keeps a single (z, k) std per row (so
    the pair eq's predicted normalized residual is on a single scale),
    and the larger std prevents either stack from being over-weighted.
    """
    z_grid = payload["z_grid_in_range"]
    k_grid = payload["k_grid"]
    n_z, n_k = z_grid.size, k_grid.size
    mean = np.zeros((n_z, n_k), dtype=float)
    std = np.zeros((n_z, n_k), dtype=float)
    for zi, z in enumerate(z_grid):
        mask = np.isclose(payload["z_per_row"], z, atol=1e-3)
        if not mask.any():
            raise ValueError(f"No Sobol rows at z={z}; need n_total ≥ 9 × |z_grid|.")
        std_lf = payload["residual_lf_z"][mask].std(axis=0, ddof=0)
        std_hf = payload["residual_hf_z"][mask].std(axis=0, ddof=0)
        # Per-(z, k) max so neither stack gets over-weighted.
        std[zi] = np.maximum(std_lf, std_hf)
    std = np.where(std > 0, std, 1.0)
    return MultiZNormalizationSpec(
        param_min=float(payload["x_pair_min"][0]),
        param_max=float(payload["x_pair_max"][0]),
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        z_grid=z_grid, mean_flux=mean, std_flux=std, k_grid=k_grid,
    )


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--phase1-refits-dir", type=Path, required=True,
                   help="Dir with cached per-1D refits/<param>.pkl.")
    p.add_argument("--pairs", nargs="+", required=True,
                   help="Pairs as 'name_i,name_j' (e.g. 'tau0,ns').")
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--basedir", type=Path,
                   default=Path("/nfs/turbo/umor-yueyingn/mfho/birdgroup/"
                                "lya_xq100/kodiaq_2_2_4_6-48-48"))
    p.add_argument("--z-min", type=float, default=2.6)
    p.add_argument("--z-max", type=float, default=4.2)
    p.add_argument("--k-min", type=float, default=0.005)
    p.add_argument("--k-max", type=float, default=0.064)
    p.add_argument("--n-k", type=int, default=32)
    p.add_argument("--n-total", type=int, default=256,
                   help="Sobol points per pair (power of 2 recommended).")
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)

    pairs = []
    for raw in args.pairs:
        parts = [s.strip() for s in raw.split(",")]
        if len(parts) != 2 or parts[0] not in PARAM_NAMES or parts[1] not in PARAM_NAMES:
            raise SystemExit(f"Bad --pair {raw!r}; expected 'name_i,name_j'.")
        pairs.append((parts[0], parts[1]))

    k_grid = np.linspace(args.k_min, args.k_max, args.n_k)
    z_grid_kodiaq = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    z_grid_in_range = z_grid_kodiaq[
        (z_grid_kodiaq >= args.z_min - 1e-6) & (z_grid_kodiaq <= args.z_max + 1e-6)
    ]
    fid = np.array(fiducial_vector(), dtype=float)

    print("Loading kodiaq emulators (LF + HF) ...")
    from priya_forecast.models.gp_model import GPModel
    t0 = time.time()
    gp_lf = GPModel(basedir=args.basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    _ = gp_lf.predict(fid, k_grid, 3.6)
    _ = gp_hf.predict(fid, k_grid, 3.6)
    print(f"  loaded in {time.time()-t0:.0f}s.")

    print(f"\nLoading + gating Phase-1 refits from {args.phase1_refits_dir} ...")
    refits = _gate_refits(_load_phase1_refits(args.phase1_refits_dir))
    n_kept = sum(r is not None for r in refits.values())
    print(f"  {n_kept}/{len(PARAM_NAMES)} refits kept (post-gate).")

    print("Building Phase-1 hybrids (LF + HF)...")
    hybrid_lf = MultiZAdditiveTaylorModel(
        gp=gp_lf, fid=fid, refits=refits,
        k_grid=k_grid, z_grid=z_grid_in_range,
    )
    hybrid_hf = MultiZAdditiveTaylorModel(
        gp=gp_hf, fid=fid, refits=refits,
        k_grid=k_grid, z_grid=z_grid_in_range,
    )

    print(f"\nGenerating pair payloads for {len(pairs)} pair(s):")
    print(f"  z=[{args.z_min}, {args.z_max}] (snap to {z_grid_in_range.tolist()})")
    print(f"  k=linspace({args.k_min}, {args.k_max}, {args.n_k}) s/km")
    print(f"  n_total={args.n_total} Sobol per pair (2 × n_total · n_k = "
          f"{2*args.n_total*args.n_k} training rows after LF/HF stack).")
    for pair_names in pairs:
        out_name = f"{pair_names[0]}_{pair_names[1]}.pkl"
        out_path = args.output / out_name
        if out_path.exists():
            print(f"  [skip-cache] {pair_names} → {out_path}")
            continue
        t0 = time.time()
        payload = _generate_pair_payload_inline(
            gp_lf=gp_lf, gp_hf=gp_hf,
            hybrid_lf=hybrid_lf, hybrid_hf=hybrid_hf,
            pair_names=pair_names,
            z_min=args.z_min, z_max=args.z_max,
            k_grid=k_grid, n_total=args.n_total, seed=args.seed,
        )
        norm = _compute_pair_normalization(payload=payload)
        bundle = dict(
            pair_names=pair_names,
            payload=payload,
            norm=norm,
            k_grid=k_grid,
            z_min=float(args.z_min), z_max=float(args.z_max),
            z_grid_in_range=z_grid_in_range,
            lf_resolution=LF_RESOLUTION,
            hf_resolution=HF_RESOLUTION,
        )
        with open(out_path, "wb") as fh:
            pickle.dump(bundle, fh)
        print(f"  [{time.time()-t0:.1f}s] {pair_names} → {out_path}", flush=True)

    print("\nAll pair payloads written. Submit refit_one_pair.py jobs now.")


if __name__ == "__main__":
    main()
