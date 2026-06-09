#!/usr/bin/env python
"""Evaluate per-candidate gradient faithfulness for a 1D Pareto front.

Reproduces the production derivative gate's metric — median_k |∂eq/∂θ ÷
∂P_GP/∂θ − 1| at fid over non-negligible bins — but prints the *numeric*
error for every Fisher-safe candidate (the gate only returns pass/fail).

Decider #1 (budget control): compare the value-loss equation's gradient
error at the certified budget against Stage 9's numbers (ns: 0.69 at
maxsize=20 value-loss; 0.134 with Sobolev).

Usage:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \
  PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
  .venv/bin/python scripts/eval_grad_faithfulness.py \
    --pareto results/decider_budget_z3.6/refit/z3.6/pareto_ns.csv \
    --param ns --z 3.6 --basedir data/kodiaq_gp --log-space
"""
from __future__ import annotations

import argparse
import numpy as np

from priya_forecast.parameters import get_param, PARAM_NAMES, PARAMS_11D
from priya_forecast.single_z import forecast as fc
from priya_forecast.single_z.training_data import load_1pvar
from priya_forecast.models.pysr_model import load_pareto_csv, pick_equation
from priya_forecast.derivative_gate import (
    gp_param_gradient, equation_param_gradient,
)
import priya_forecast.single_z.refit as _refit
from priya_forecast.grad_faith_io import (
    equation_has_x0, write_grad_faith_sidecar,
)
from priya_forecast.refit_1d_pysr import HF_RESOLUTION


def median_rel_error(cand_grad, target_grad, floor_frac=1e-3):
    cand = np.asarray(cand_grad, float)
    target = np.asarray(target_grad, float)
    amax = float(np.max(np.abs(target)))
    if amax == 0.0:
        return np.inf, 0
    keep = np.abs(target) >= floor_frac * amax
    if not np.any(keep):
        return np.inf, 0
    rel = np.abs(cand[keep] / target[keep] - 1.0)
    return float(np.median(rel)), int(np.sum(keep))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pareto", required=True)
    p.add_argument("--param", required=True, choices=list(PARAM_NAMES))
    p.add_argument("--z", type=float, required=True)
    p.add_argument("--basedir", default="data/kodiaq_gp")
    p.add_argument("--kmin", type=float, default=0.001)
    p.add_argument("--kmax", type=float, default=0.04)
    p.add_argument("--tol", type=float, default=0.25)
    p.add_argument("--log-space", action="store_true")
    p.add_argument("--data-1pvar", default="data/single_z_1pvar")
    p.add_argument("--out", default=None,
                   help="write a grad-faith sidecar CSV to this path")
    args = p.parse_args()

    from priya_forecast.models.gp_model import GPModel

    k_grid = _refit.kodiaq_k_grid(args.kmin, args.kmax, 48)
    fid = np.array([pp.fid for pp in PARAMS_11D], dtype=float)
    pidx = PARAM_NAMES.index(args.param)
    meta = get_param(args.param)

    print(f"Loading HF GP from {args.basedir} ...", flush=True)
    gp_hf = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    _ = gp_hf.predict(fid, k_grid, args.z)

    target = gp_param_gradient(gp=gp_hf, fid=fid, k_grid=k_grid, z=args.z,
                               param_idx=pidx)

    df = load_pareto_csv(args.pareto)
    safe = fc._filter_fisher_safe(df, n_features=3)
    d = load_1pvar(param_name=args.param, z=args.z, data_dir=args.data_1pvar)
    kg = d["kfkms_lf_z"][0]
    norm = fc.per_param_local_norm(
        flux_lf_z=d["flux_lf_z"], k_grid=kg,
        param_min=float(meta.prior[0]), param_max=float(meta.prior[1]),
        log_space=args.log_space,
    )

    # Common value-loss reference: the GP's logP over the HF training theta-grid
    # (same fidelity + resolution as the gradient gate). value_mse (below) is the
    # value analog of grad_err and -- unlike the PySR `Loss` column, which is the
    # Sobolev objective for Sobolev runs -- is comparable across runs trained with
    # different objectives. Precompute once (independent of the candidate).
    kg = np.asarray(kg, dtype=float)
    # N1 guard: the candidate gradient uses kg (1pvar k) while the GP target gradient
    # uses k_grid (kodiaq); they coincide at the default bounds but a non-default
    # --kmin/--kmax would silently misalign the elementwise ratio.
    if not np.allclose(np.asarray(k_grid, float), kg):
        raise ValueError(
            "k_grid (kodiaq) and kg (1pvar) differ -- grad_err/value_mse would "
            "misalign. Re-run at the default --kmin/--kmax, or align the grids."
        )
    theta_grid = np.asarray(d["params_hf"][:, pidx], dtype=float)
    logP_gp_grid = np.empty((theta_grid.size, kg.size), dtype=float)
    for i, t in enumerate(theta_grid):
        tv = fid.copy()
        tv[pidx] = float(t)
        logP_gp_grid[i] = np.log(np.asarray(gp_hf.predict(tv, kg, args.z), float))

    rows = []
    for _, row in safe.sort_values("Loss").iterrows():
        cand = fc._refit_from_row(
            equation_str=str(row["Equation"]), complexity=int(row["Complexity"]),
            loss=float(row["Loss"]), df=df, param_name=args.param, z=args.z,
            meta=meta, k_grid=kg, norm=norm, log_space=args.log_space,
        )
        g = equation_param_gradient(refit=cand, fid_value=float(meta.fid),
                                    k_grid=kg, z=args.z)
        err, nkeep = median_rel_error(g, target)
        logP_eq = np.array([
            cand.predict_log(theta_phys=float(t), k=kg,
                             resolution=HF_RESOLUTION, z=args.z)
            for t in theta_grid
        ])
        value_mse = float(np.mean((logP_eq - logP_gp_grid) ** 2))
        rows.append({
            "Complexity": int(row["Complexity"]),
            "Loss": float(row["Loss"]),
            "grad_err": err,
            "value_mse": value_mse,
            "n_keep": int(nkeep),
            "gate_pass": bool(err <= args.tol),
            "x0_enters": bool(equation_has_x0(str(row["Equation"]))),
        })

    print(f"\n=== {args.param} z={args.z}  (Fisher-safe candidates, by loss) ===")
    print(f"{'cmplx':>6} {'loss':>10} {'grad_err':>10} {'value_mse':>10} "
          f"{'gate(<=%.2f)':>12}" % args.tol)
    for r in rows:
        flag = "PASS" if r["gate_pass"] else "fail"
        print(f"{r['Complexity']:>6} {r['Loss']:>10.5f} "
              f"{r['grad_err']:>10.4f} {r['value_mse']:>10.4f} {flag:>12}")

    if rows:
        best_loss = rows[0]  # already sorted by loss asc
        best_faith = min(rows, key=lambda r: r["grad_err"])
        any_pass = any(r["gate_pass"] for r in rows)
        print(f"\nbest_loss pick:   complexity={best_loss['Complexity']} "
              f"loss={best_loss['Loss']:.5f} grad_err={best_loss['grad_err']:.4f}")
        print(f"best faithfulness: complexity={best_faith['Complexity']} "
              f"loss={best_faith['Loss']:.5f} grad_err={best_faith['grad_err']:.4f}")
        print(f"ANY equation passes gate (<= {args.tol}): {any_pass}")

    if args.out:
        path = write_grad_faith_sidecar(
            args.out, rows, param=args.param, z=args.z, tol=args.tol,
            log_space=args.log_space, source_pareto=args.pareto,
        )
        print(f"\nwrote sidecar: {path}")


if __name__ == "__main__":
    main()
