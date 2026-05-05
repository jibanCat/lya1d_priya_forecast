"""Phase 2 / iter 3: per-pair PySR fit on residuals.

Loads a pre-computed pair payload (no GP needed), builds the 5-input
training matrix `(θ_i_norm, θ_j_norm, k_norm, resolution, z_norm)` with
target = residual / std_per_(z, k) (mean = 0 by construction since
residual ≈ 0 at fid). Runs PySR with the smart-refit kwargs (ANOVA
loss + restricted unary operators) — those gave the best Phase-1.5
result and are appropriate for the small-magnitude pair residuals.

Pareto pick: prefer eqs that use BOTH x0 AND x1 (the two θ features) +
finite + Fisher-stencil-safe + no pathological constants.

Saves `refits/<name_i>_<name_j>.pkl` containing a `Refit2DPairResult`.

Run:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/refit_one_pair.py \\
          --pair tau0,ns \\
          --payload-dir results/refit_pair_z2.6-4.2/payloads \\
          --output-dir results/refit_pair_z2.6-4.2
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

from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.refit_1d_pysr import (
    HF_RESOLUTION,
    LF_RESOLUTION,
    SMART_REFIT_PYSR_KWARGS,
)
from priya_forecast.refit_pair import Refit2DPairResult


def _build_pair_training_matrix(
    *, payload: dict, norm,
) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stack LF + HF residuals into a 5-input training matrix.

    Returns (X_act, Y_act, ranges) where:
      X_act[:, 0] = θ_i_norm
      X_act[:, 1] = θ_j_norm
      X_act[:, 2] = k_norm
      X_act[:, 3] = resolution (LF=0.4 or HF=0.8)
      X_act[:, 4] = z_norm
      Y_act       = residual / std_per_(z, k) -- normalized residual
    """
    p = payload  # alias
    n_total = p["theta_i"].size
    n_k = p["k_grid"].size
    k_grid = p["k_grid"]
    z_grid = p["z_grid_in_range"]
    z_per_row = p["z_per_row"]

    # Normalize θ_i, θ_j to [0, 1].
    ti_n = (p["theta_i"] - p["x_pair_min"][0]) / (p["x_pair_max"][0] - p["x_pair_min"][0])
    tj_n = (p["theta_j"] - p["x_pair_min"][1]) / (p["x_pair_max"][1] - p["x_pair_min"][1])
    # Normalize k.
    k_n = (k_grid - k_grid.min()) / (k_grid.max() - k_grid.min())
    # Normalize z (per-row).
    z_n_per_row = np.zeros(n_total, dtype=float)
    z_range = p["z_max"] - p["z_min"]
    if z_range > 0:
        z_n_per_row = (z_per_row - p["z_min"]) / z_range
    # std lookup per row.
    std_per_row = np.empty((n_total, n_k), dtype=float)
    for r_i in range(n_total):
        zi = int(np.argmin(np.abs(z_grid - z_per_row[r_i])))
        std_per_row[r_i] = norm.std_flux[zi]
    # Build LF + HF stacks.
    X_blocks = []
    Y_blocks = []
    for resolution, resid in (
        (LF_RESOLUTION, p["residual_lf_z"]),
        (HF_RESOLUTION, p["residual_hf_z"]),
    ):
        # Each row of residual is a length-n_k array; build (n_total · n_k, 5).
        ti_n_rep = np.repeat(ti_n[:, None], n_k, axis=1)  # (n_total, n_k)
        tj_n_rep = np.repeat(tj_n[:, None], n_k, axis=1)
        z_n_rep = np.repeat(z_n_per_row[:, None], n_k, axis=1)
        k_n_rep = np.tile(k_n[None, :], (n_total, 1))
        r_n_rep = np.full((n_total, n_k), float(resolution))
        X = np.column_stack([
            ti_n_rep.ravel(),
            tj_n_rep.ravel(),
            k_n_rep.ravel(),
            r_n_rep.ravel(),
            z_n_rep.ravel(),
        ])
        # Normalize residual by per-(z, k) std (mean=0 by construction).
        Y_norm = (resid / std_per_row).ravel()
        X_blocks.append(X)
        Y_blocks.append(Y_norm)
    X_act = np.vstack(X_blocks)
    Y_act = np.concatenate(Y_blocks)
    ranges = dict(
        x_pair_min=p["x_pair_min"], x_pair_max=p["x_pair_max"],
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
    )
    return X_act, Y_act, ranges


def _validate_pair(*, result: Refit2DPairResult, payload: dict) -> dict:
    """Compute LF + HF mean/max rel-err of the eq's *residual* prediction
    vs the actual residual on the training Sobol set."""
    p = payload
    n_total = p["theta_i"].size
    z_grid = p["z_grid_in_range"]
    z_per_row = p["z_per_row"]
    diag = {}
    for tag, resolution, resid in (
        ("lf", LF_RESOLUTION, p["residual_lf_z"]),
        ("hf", HF_RESOLUTION, p["residual_hf_z"]),
    ):
        rel_per_row = np.empty(n_total, dtype=float)
        for r_i in range(n_total):
            theta_pair = (float(p["theta_i"][r_i]), float(p["theta_j"][r_i]))
            z = float(z_per_row[r_i])
            pred = result.predict(theta_pair, p["k_grid"], resolution, z)
            true = resid[r_i]
            scale = np.maximum(np.abs(true).max(), np.abs(pred).max(), )
            denom = max(float(scale), 1e-30)
            rel_per_row[r_i] = float(np.mean(np.abs(pred - true)) / denom)
        diag[f"{tag}_mean"] = float(np.mean(rel_per_row))
        diag[f"{tag}_max"] = float(np.max(rel_per_row))
    return diag


def _fit_once(
    *, payload: dict, norm, pysr_kwargs: dict, seed: int,
) -> Refit2DPairResult:
    from pysr import PySRRegressor  # type: ignore[import-not-found]
    X_act, Y_act, ranges = _build_pair_training_matrix(payload=payload, norm=norm)
    args = dict(SMART_REFIT_PYSR_KWARGS)
    args.update(pysr_kwargs or {})
    args["random_state"] = seed
    t0 = time.time()
    model = PySRRegressor(**args)
    model.fit(X_act, Y_act.reshape(-1, 1))
    elapsed = time.time() - t0
    pareto = model.equations_

    # Pareto pick: prefer eqs that use BOTH x0 AND x1; apply same Fisher-stencil
    # safety filters as per-1D.
    from priya_forecast.pareto_filters import (
        has_pathological_constant, is_eq_well_behaved, is_fisher_stencil_safe,
    )
    n_features = X_act.shape[1]  # 5
    eq_strs = pareto["equation"].astype(str)
    import re
    has_x0 = eq_strs.apply(lambda s: re.search(r"\bx0\b", s) is not None)
    has_x1 = eq_strs.apply(lambda s: re.search(r"\bx1\b", s) is not None)
    pathological = eq_strs.apply(has_pathological_constant)
    well_behaved = eq_strs.apply(
        lambda s: is_eq_well_behaved(s, X_act, Y_act.ravel(), n_features=n_features)
    )
    stencil_safe = eq_strs.apply(
        lambda s: is_fisher_stencil_safe(s, n_features=n_features)
    )
    good_safe = (~pathological) & well_behaved & stencil_safe
    both_theta = has_x0 & has_x1 & good_safe
    one_theta = (has_x0 | has_x1) & good_safe
    if bool(both_theta.any()):
        best_idx = int(pareto.loc[both_theta, "loss"].idxmin())
    elif bool(one_theta.any()):
        best_idx = int(pareto.loc[one_theta, "loss"].idxmin())
    else:
        # Fall back to lowest-loss safe eq, then lowest-loss any.
        if bool(good_safe.any()):
            best_idx = int(pareto.loc[good_safe, "loss"].idxmin())
        else:
            best_idx = int(pareto["loss"].idxmin())

    pair_names = payload["pair_names"]
    result = Refit2DPairResult(
        pair_names=pair_names,
        equation_str=str(pareto.iloc[best_idx]["equation"]),
        pareto_complexity=int(pareto.iloc[best_idx]["complexity"]),
        pareto_loss=float(pareto.iloc[best_idx]["loss"]),
        pareto_complexities=pareto["complexity"].astype(int).tolist(),
        pareto_losses=pareto["loss"].astype(float).tolist(),
        x_pair_min=tuple(map(float, ranges["x_pair_min"])),
        x_pair_max=tuple(map(float, ranges["x_pair_max"])),
        k_min=ranges["k_min"], k_max=ranges["k_max"],
        fid_pair=tuple(map(float, payload["fid_pair"])),
        z_min=float(payload["z_min"]), z_max=float(payload["z_max"]),
        norm=norm, k_grid=payload["k_grid"],
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        wall_time_s=elapsed,
    )
    diag = _validate_pair(result=result, payload=payload)
    result.lf_train_mean_rel_err = diag["lf_mean"]
    result.hf_train_mean_rel_err = diag["hf_mean"]
    result.lf_train_max_rel_err = diag["lf_max"]
    result.hf_train_max_rel_err = diag["hf_max"]
    return result


def main():
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--pair", required=True,
                   help="Pair as 'name_i,name_j' (e.g. 'tau0,ns').")
    p.add_argument("--payload-dir", type=Path, required=True,
                   help="Dir with pair payloads (from precompute_payloads_pair.py).")
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--niter", type=int, default=50)
    p.add_argument("--maxsize", type=int, default=20)
    p.add_argument("--maxdepth", type=int, default=10)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--max-retries", type=int, default=2)
    args = p.parse_args()

    parts = [s.strip() for s in args.pair.split(",")]
    if len(parts) != 2:
        raise SystemExit(f"--pair must be 'name_i,name_j', got {args.pair!r}.")
    pair_names = (parts[0], parts[1])
    for nm in pair_names:
        if nm not in PARAM_NAMES:
            raise SystemExit(f"Unknown param name {nm!r}.")

    out_refits = args.output_dir / "refits"
    out_refits.mkdir(parents=True, exist_ok=True)
    out_path = out_refits / f"{pair_names[0]}_{pair_names[1]}.pkl"

    payload_path = args.payload_dir / f"{pair_names[0]}_{pair_names[1]}.pkl"
    if not payload_path.exists():
        raise FileNotFoundError(
            f"Payload not found: {payload_path}. Run precompute_payloads_pair.py first."
        )
    with open(payload_path, "rb") as fh:
        bundle = pickle.load(fh)

    pysr_kwargs = dict(
        niterations=args.niter, maxsize=args.maxsize, maxdepth=args.maxdepth,
    )
    print(f"[pair {pair_names}] SMART pair fit (ANOVA loss + ops {{exp, log, square}})")

    seed = args.seed
    best_result = None
    for retry in range(args.max_retries + 1):
        print(f"[{pair_names}] attempt {retry+1}/{args.max_retries+1} (seed={seed})", flush=True)
        result = _fit_once(
            payload=bundle["payload"], norm=bundle["norm"],
            pysr_kwargs=pysr_kwargs, seed=seed,
        )
        import re
        has_x0 = re.search(r"\bx0\b", result.equation_str) is not None
        has_x1 = re.search(r"\bx1\b", result.equation_str) is not None
        rel_ok = (
            np.isfinite(result.lf_train_mean_rel_err)
            and np.isfinite(result.hf_train_mean_rel_err)
        )
        print(f"  has_x0={has_x0} has_x1={has_x1} rel_ok={rel_ok} | "
              f"LF={result.lf_train_mean_rel_err*100:.2f}% "
              f"HF={result.hf_train_mean_rel_err*100:.2f}% | "
              f"complexity={result.pareto_complexity} | "
              f"eq[:80]={result.equation_str[:80]}", flush=True)
        if has_x0 and has_x1 and rel_ok:
            best_result = result
            break
        # Track best-so-far (most θ-features first, then lowest pareto loss).
        score = (1 if has_x0 else 0) + (1 if has_x1 else 0)
        best_score = -1 if best_result is None else (
            (1 if "x0" in best_result.equation_str else 0)
            + (1 if "x1" in best_result.equation_str else 0)
        )
        if best_result is None or score > best_score or (
            score == best_score and result.pareto_loss < best_result.pareto_loss
        ):
            best_result = result
        seed += 1

    has_x0_final = "x0" in best_result.equation_str
    has_x1_final = "x1" in best_result.equation_str
    if not (has_x0_final and has_x1_final):
        print(f"[{pair_names}] WARNING: all {args.max_retries+1} retries failed "
              f"to use both x0 AND x1; saving best-anyway.", flush=True)

    with open(out_path, "wb") as fh:
        pickle.dump(best_result, fh)
    print(f"[{pair_names}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
