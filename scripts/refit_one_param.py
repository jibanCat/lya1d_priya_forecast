"""Phase 2 of option C: ONE param's multi-z PySR fit, parallel-safe.

Loads a pre-computed payload (no GP emulator!), runs PySR. If the
discovered equation drops x0 (theta dependence), retries with seed+1
up to MAX_RETRIES times — the at-fid anchor doesn't always force x0
for weakly-coupled params, so a small number of retries reliably finds
an x0-using equation.

Run (single param locally):
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/refit_one_param.py \\
          --param ns --payload-dir results/.../payloads \\
          --output-dir results/.../refits

Run (SLURM array via slurm/refit_array.slurm):
  sbatch --array=1-11 slurm/refit_array.slurm
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

from priya_forecast.refit_1d_pysr import (
    DEFAULT_PYSR_KWARGS,
    HF_RESOLUTION,
    LF_RESOLUTION,
    Refit1DResult,
    SMART_REFIT_PARAMS,
    SMART_REFIT_PYSR_KWARGS,
    _build_training_matrix_multiz,
    _validate_per_fidelity_from_payload_multiz,
)
from priya_forecast.parameters import (
    PARAM_NAMES,
    fiducial_vector,
    get_param,
)


def _fit_once(
    *, param_name: str, payload: dict, norm, k_grid: np.ndarray,
    z_min: float, z_max: float, pysr_kwargs: dict, seed: int,
    smart: bool = False,
) -> Refit1DResult:
    from pysr import PySRRegressor  # type: ignore[import-not-found]
    param_idx = PARAM_NAMES.index(param_name)
    X_act, Y_act, ranges, _ = _build_training_matrix_multiz(
        payload=payload, param_idx=param_idx, norm=norm,
        z_min=z_min, z_max=z_max,
    )
    base = SMART_REFIT_PYSR_KWARGS if smart else DEFAULT_PYSR_KWARGS
    args = dict(base)
    args.update(pysr_kwargs or {})
    args["random_state"] = seed
    t0 = time.time()
    model = PySRRegressor(**args)
    model.fit(X_act, Y_act.reshape(-1, 1))
    elapsed = time.time() - t0
    pareto = model.equations_
    # Pareto-pick filters: x0 present + no |c|>100 literals + finite over
    # training X + Fisher-stencil safe. See `pareto_filters.py`.
    from priya_forecast.pareto_filters import (
        has_pathological_constant, is_eq_well_behaved, is_fisher_stencil_safe,
    )
    n_features = X_act.shape[1]
    eq_strs = pareto["equation"].astype(str)
    x0_mask = eq_strs.str.contains("x0")
    pathological = eq_strs.apply(has_pathological_constant)
    well_behaved = eq_strs.apply(
        lambda s: is_eq_well_behaved(s, X_act, Y_act.ravel(), n_features=n_features)
    )
    stencil_safe = eq_strs.apply(
        lambda s: is_fisher_stencil_safe(s, n_features=n_features)
    )
    sane_x0 = x0_mask & (~pathological) & well_behaved & stencil_safe
    if bool(sane_x0.any()):
        best_idx = int(pareto.loc[sane_x0, "loss"].idxmin())
    elif bool((x0_mask & well_behaved & stencil_safe).any()):
        best_idx = int(pareto.loc[x0_mask & well_behaved & stencil_safe, "loss"].idxmin())
    elif bool((x0_mask & well_behaved).any()):
        best_idx = int(pareto.loc[x0_mask & well_behaved, "loss"].idxmin())
    elif bool(x0_mask.any()):
        best_idx = int(pareto.loc[x0_mask, "loss"].idxmin())
    else:
        best_idx = int(pareto["loss"].idxmin())
    p_meta = get_param(param_name)
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
        fid_value=p_meta.fid, norm=norm, k_grid=np.asarray(k_grid, dtype=float),
        wall_time_s=elapsed,
        lf_train_mean_rel_err=float("nan"),
        hf_train_mean_rel_err=float("nan"),
        lf_train_max_rel_err=float("nan"),
        hf_train_max_rel_err=float("nan"),
        z_min=z_min, z_max=z_max,
    )
    diag = _validate_per_fidelity_from_payload_multiz(
        result=result, payload=payload, param_idx=param_idx,
    )
    result.lf_train_mean_rel_err = diag["lf_mean"]
    result.hf_train_mean_rel_err = diag["hf_mean"]
    result.lf_train_max_rel_err = diag["lf_max"]
    result.hf_train_max_rel_err = diag["hf_max"]
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--param", required=True, choices=list(PARAM_NAMES))
    p.add_argument("--payload-dir", type=Path, required=True,
                   help="Directory with <param>.pkl files (from precompute_payloads.py).")
    p.add_argument("--output-dir", type=Path, required=True,
                   help="Where to write refits/<param>.pkl.")
    p.add_argument("--niter", type=int, default=50)
    p.add_argument("--maxsize", type=int, default=20)
    p.add_argument("--maxdepth", type=int, default=10)
    p.add_argument("--seed", type=int, default=42,
                   help="Initial PySR random seed; retries use seed+1, +2, ...")
    p.add_argument("--max-retries", type=int, default=4,
                   help="Number of seed-bumped retries if eq lacks x0.")
    p.add_argument("--smart", action="store_true",
                   help="Use SMART_REFIT_PYSR_KWARGS (ANOVA loss + "
                        "restricted operators {exp, log, square}) — "
                        "Phase 1.5 fix for heref/herei/alphaq.")
    p.add_argument("--auto-smart", action="store_true",
                   help="Enable --smart automatically when --param is in "
                        f"{list(SMART_REFIT_PARAMS)} (default Phase 1.5 set).")
    args = p.parse_args()
    use_smart = bool(args.smart or (args.auto_smart and args.param in SMART_REFIT_PARAMS))
    if use_smart:
        print(f"[{args.param}] SMART refit ENABLED: ANOVA loss + "
              "operators={exp, log, square}")

    out_refits = args.output_dir / "refits"
    out_refits.mkdir(parents=True, exist_ok=True)
    out_path = out_refits / f"{args.param}.pkl"
    # Smart refits intentionally ignore the cache — they're a Phase 1.5
    # rerun of an already-fitted param with different operators/loss.
    if out_path.exists() and not use_smart:
        with open(out_path, "rb") as fh:
            existing = pickle.load(fh)
        if "x0" in existing.equation_str:
            print(f"[cache] {args.param} already done with x0; eq: {existing.equation_str[:80]}")
            return

    payload_path = args.payload_dir / f"{args.param}.pkl"
    if not payload_path.exists():
        raise FileNotFoundError(
            f"Payload not found: {payload_path}. Run precompute_payloads.py first."
        )
    with open(payload_path, "rb") as fh:
        bundle = pickle.load(fh)

    pysr_kwargs = dict(
        niterations=args.niter, maxsize=args.maxsize, maxdepth=args.maxdepth,
    )
    # Acceptance criteria: eq has x0 AND mean rel-err is reasonable. The
    # second guard catches PySR finding x0 with a huge literal constant
    # offset (e.g. `(x0 - 3.4e11) / (x3 - 0.23)`), which technically uses
    # x0 but has a meaningless fit (loss 1e26+, rel-err 1e13%).
    seed = args.seed
    best_result = None
    REL_ERR_GUARD = 0.50  # 50% mean rel-err — if worse, retry.
    for retry in range(args.max_retries + 1):
        print(f"[{args.param}] attempt {retry+1}/{args.max_retries+1} (seed={seed})", flush=True)
        result = _fit_once(
            param_name=args.param,
            payload=bundle["payload"], norm=bundle["norm"],
            k_grid=bundle["k_grid"],
            z_min=bundle["z_min"], z_max=bundle["z_max"],
            pysr_kwargs=pysr_kwargs, seed=seed, smart=use_smart,
        )
        has_x0 = "x0" in result.equation_str
        rel_err_ok = (
            result.lf_train_mean_rel_err < REL_ERR_GUARD
            and result.hf_train_mean_rel_err < REL_ERR_GUARD
        )
        print(f"  has_x0={has_x0} rel_err_ok={rel_err_ok} | "
              f"LF={result.lf_train_mean_rel_err*100:.2f}% "
              f"HF={result.hf_train_mean_rel_err*100:.2f}% | "
              f"eq[:60]={result.equation_str[:60]}", flush=True)
        if has_x0 and rel_err_ok:
            best_result = result
            break
        # Track best-so-far (prefer x0+sane-rel-err > x0+broken > no-x0).
        score = (1 if has_x0 else 0) + (1 if rel_err_ok else 0)
        best_score = -1 if best_result is None else (
            (1 if "x0" in best_result.equation_str else 0)
            + (1 if best_result.lf_train_mean_rel_err < REL_ERR_GUARD
               and best_result.hf_train_mean_rel_err < REL_ERR_GUARD else 0)
        )
        if best_result is None or score > best_score or (
            score == best_score and result.pareto_loss < best_result.pareto_loss
        ):
            best_result = result
        seed += 1
    has_x0_final = "x0" in best_result.equation_str
    rel_ok_final = (
        best_result.lf_train_mean_rel_err < REL_ERR_GUARD
        and best_result.hf_train_mean_rel_err < REL_ERR_GUARD
    )
    if not (has_x0_final and rel_ok_final):
        print(f"[{args.param}] WARNING: all {args.max_retries+1} retries failed "
              f"(x0={has_x0_final}, rel_err_ok={rel_ok_final}); saving best-anyway.",
              flush=True)

    with open(out_path, "wb") as fh:
        pickle.dump(best_result, fh)
    print(f"[{args.param}] wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
