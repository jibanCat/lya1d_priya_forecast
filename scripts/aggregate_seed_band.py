#!/usr/bin/env python
"""Aggregate the across-seed band: best-loss grad_err median/min/max per parameter,
flag any taxonomy box that flips across seeds. Needs the emulator.

Efficient: loads the GP once and computes each parameter's GP target gradient once
(the GP is seed-independent), then scores the value-optimal (lowest-Loss Fisher-safe)
equation of every (seed, mode) front. Reproduces the production gate metric.

Usage:
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \
  PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
  .venv/bin/python scripts/aggregate_seed_band.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from priya_forecast.parameters import get_param, PARAM_NAMES, PARAMS_11D
from priya_forecast.single_z import forecast as fc
from priya_forecast.single_z.training_data import load_1pvar
from priya_forecast.models.pysr_model import load_pareto_csv
from priya_forecast.derivative_gate import gp_param_gradient, equation_param_gradient
import priya_forecast.single_z.refit as _refit

GATE = 0.25
SEEDS = [0, 1, 2, 3, 4]
Z = 3.6


def median_rel_error(cand, target, floor=1e-3):
    cand = np.asarray(cand, float)
    target = np.asarray(target, float)
    amax = float(np.max(np.abs(target)))
    if amax == 0:
        return np.inf
    keep = np.abs(target) >= floor * amax
    if not np.any(keep):
        return np.inf
    return float(np.median(np.abs(cand[keep] / target[keep] - 1.0)))


def best_loss_grad_err(csv, param, meta, kg, target):
    """grad_err of the value-optimal (lowest-Loss Fisher-safe) equation, or nan."""
    if not Path(csv).exists():
        return np.nan
    df = load_pareto_csv(csv)
    safe = fc._filter_fisher_safe(df, n_features=3)
    if safe.empty:
        return np.nan
    from priya_forecast.grad_faith_io import knee_row
    row = knee_row(safe)  # Pareto-knee pick (see grad_faith_io.knee_row)
    cand = fc._refit_from_row(
        equation_str=str(row["Equation"]), complexity=int(row["Complexity"]),
        loss=float(row["Loss"]), df=df, param_name=param, z=Z, meta=meta,
        k_grid=kg, norm=_NORM[param], log_space=True)
    g = equation_param_gradient(refit=cand, fid_value=float(meta.fid), k_grid=kg, z=Z)
    return median_rel_error(g, target)


_NORM: dict = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--basedir", default="data/kodiaq_gp")
    ap.add_argument("--band-dir", default="results/paper_production_20260630_perz_sobolev_z2.6-4.2/seed_band")
    ap.add_argument("--out", default="results/paper_production_20260630_perz_sobolev_z2.6-4.2/seed_band/seed_band_summary.json")
    args = ap.parse_args()

    from priya_forecast.models.gp_model import GPModel
    k_grid = _refit.kodiaq_k_grid(0.001, 0.04, 48)
    fid = np.array([p.fid for p in PARAMS_11D], float)
    gp = GPModel(basedir=args.basedir, fidelity="hf", kf=k_grid)
    _ = gp.predict(fid, k_grid, Z)
    band = Path(args.band_dir)

    raw = {}
    for p in PARAM_NAMES:
        pidx = PARAM_NAMES.index(p)
        meta = get_param(p)
        d = load_1pvar(param_name=p, z=Z, data_dir="data/single_z_1pvar")
        kg = np.asarray(d["kfkms_lf_z"][0], float)
        _NORM[p] = fc.per_param_local_norm(
            flux_lf_z=d["flux_lf_z"], k_grid=kg, param_min=float(meta.prior[0]),
            param_max=float(meta.prior[1]), log_space=True)
        target = gp_param_gradient(gp=gp, fid=fid, k_grid=kg, z=Z, param_idx=pidx,
                                   log_space=True)
        rec = {"value": [], "sobolev": []}
        for S in SEEDS:
            for mode in ("value", "sobolev"):
                csv = band / f"z3.6_seed{S}_{mode}" / f"refit/z{Z}" / f"pareto_{p}.csv"
                rec[mode].append(best_loss_grad_err(csv, p, meta, kg, target))
        raw[p] = rec

    # ns budget@35
    meta = get_param("ns")
    d = load_1pvar(param_name="ns", z=Z, data_dir="data/single_z_1pvar")
    kg = np.asarray(d["kfkms_lf_z"][0], float)
    target = gp_param_gradient(gp=gp, fid=fid, k_grid=kg, z=Z,
                               param_idx=PARAM_NAMES.index("ns"), log_space=True)
    budget = [best_loss_grad_err(band / f"z3.6_seed{S}_budget" / f"refit/z{Z}/pareto_ns.csv",
                                 "ns", meta, kg, target) for S in SEEDS]

    def stat(a):
        a = np.array([x for x in a if np.isfinite(x)])
        if not a.size:
            return (float("nan"),) * 3 + (0,)
        return float(np.median(a)), float(a.min()), float(a.max()), int(a.size)

    print(f"{'param':>10} | {'value med[min,max] n':>28} | {'sobolev med[min,max] n':>28} | flips")
    from priya_forecast.provenance import git_stamp
    summary = {"gate": GATE, "seeds": SEEDS, "git": git_stamp(), "params": {}}
    for p in PARAM_NAMES:
        vm, vlo, vhi, vn = stat(raw[p]["value"])
        sm, slo, shi, sn = stat(raw[p]["sobolev"])
        vflip = np.isfinite(vlo) and ((vlo <= GATE) != (vhi <= GATE))
        sflip = np.isfinite(slo) and ((slo <= GATE) != (shi <= GATE))
        flips = " ".join(x for x, f in [("value", vflip), ("sobolev", sflip)] if f) or "stable"
        summary["params"][p] = {"value": [vm, vlo, vhi, vn],
                                "sobolev": [sm, slo, shi, sn], "flips": flips}
        print(f"{p:>10} | {vm:>7.3f}[{vlo:.3f},{vhi:.3f}] {vn} | "
              f"{sm:>7.3f}[{slo:.3f},{shi:.3f}] {sn} | {flips}")
    bm, blo, bhi, bn = stat(budget)
    verdict = "all FAIL" if blo > GATE else ("MIXED" if bhi > GATE else "all pass")
    print(f"\nns budget@35 best-loss grad_err across {bn} seeds: "
          f"med {bm:.3f} [{blo:.3f},{bhi:.3f}] vs gate {GATE} -> {verdict}")
    summary["ns_budget35"] = {"median": bm, "min": blo, "max": bhi, "n": bn, "verdict": verdict}
    Path(args.out).write_text(json.dumps(summary, indent=2))
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
