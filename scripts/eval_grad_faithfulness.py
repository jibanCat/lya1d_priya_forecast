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
  PYTHONPATH=src:$LYA_EMULATOR \
  .venv/bin/python scripts/eval_grad_faithfulness.py \
    --pareto results/paper_production_20260630_perz_sobolev_z2.6-4.2/seed_band/z3.6_seed0_budget/refit/z3.6/pareto_ns.csv \
    --param ns --z 3.6 --basedir data/kodiaq_gp --log-space
"""
from __future__ import annotations

import argparse
import json
import os

from priya_forecast.parameters import PARAM_NAMES, override_params
from priya_forecast.grad_faith_io import write_grad_faith_sidecar
# The scoring core is shared with priya_forecast.rerun (in-process, GP-reusing).
# median_rel_error is re-exported for back-compat with existing callers/tests.
from priya_forecast.grad_faith_score import score_pareto, median_rel_error  # noqa: F401


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pareto", required=True)
    p.add_argument("--param", required=True, choices=list(PARAM_NAMES))
    p.add_argument("--z", type=float, required=True)
    p.add_argument("--basedir", default="data/kodiaq_gp")
    p.add_argument("--kmin", type=float, default=0.001)
    p.add_argument("--kmax", type=float, default=0.04)
    p.add_argument("--tol", type=float, default=0.25)
    # Default to log-space (the production gate space); --linear-space opts out.
    # --log-space is kept as a harmless no-op so existing callers still parse.
    p.add_argument("--log-space", action="store_true", default=True,
                   help="(default) score in log-P, matching production.")
    p.add_argument("--linear-space", dest="log_space", action="store_false",
                   help="score in linear P (legacy; not the production gate).")
    p.add_argument("--data-1pvar", default="data/single_z_1pvar")
    p.add_argument("--out", default=None,
                   help="write a grad-faith sidecar CSV to this path")
    args = p.parse_args()

    fid_ov = json.loads(os.environ.get("PRIYA_FIDUCIAL_OVERRIDES", "null"))
    prior_ov = json.loads(os.environ.get("PRIYA_PRIOR_OVERRIDES", "null"))
    with override_params(fid_ov, prior_ov):
        _score(args)


def _score(args):
    # Shared core; loads its own GP (gp_hf=None). run_grid calls score_pareto
    # directly with a pre-loaded GP to avoid reloading per file.
    rows = score_pareto(
        pareto_csv=args.pareto, param=args.param, z=args.z,
        basedir=args.basedir, data_1pvar=args.data_1pvar,
        kmin=args.kmin, kmax=args.kmax, tol=args.tol, log_space=args.log_space,
    )

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
