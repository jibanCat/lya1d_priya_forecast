#!/usr/bin/env python
"""Regenerate the three diagnostic paper figures with referee-requested annotations.

Emulator-free: reads the cached grad-faith sidecars. Produces, into --out-dir
(PNG + PDF):
  1. pareto_faithfulness  -- 11-panel grid, y=value_mse, colour=grad_err, with the
     gate made legible on each panel (bold ring = clears 0.25) + an annotated arrow
     on the ns panel at the low-but-red Mirage cluster.
  2. faithfulness_scorecard -- value vs Sobolev best-loss grad_err, with the two
     above-gate parameters labelled with their numbers.
  3. ns_budget_panel -- the paired budget-vs-Sobolev comparison, endpoints labelled.

Usage:
  PYTHONPATH=src python scripts/make_diagnostic_figs.py \
    --out-dir results/single_z_stage_pareto_diag
"""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np

from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.grad_faith_io import read_grad_faith_sidecar
from priya_forecast.pareto_diag import load_front, render_grid, GATE_TOL

VALUE = "results/single_z_stage6_log/refit/z3.6"
SOBOLEV = "results/single_z_stage9/refit/z3.6"
BUDGET = "results/decider_budget_z3.6/refit/z3.6"


def bestloss(d, p, col="grad_err"):
    df = read_grad_faith_sidecar(f"{d}/grad_faith_{p}.csv").sort_values("Loss")
    return float(df.iloc[0][col])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/single_z_stage_pareto_diag")
    ap.add_argument("--also-copy-to", action="append", default=[],
                    help="extra dirs to copy the PNGs into (repeatable)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    # ---------- Figure 1: the grid (gate rings + ns arrow) ----------
    fronts = {}
    for p in PARAM_NAMES:
        rows = [
            {"front": load_front(f"{VALUE}/pareto_{p}.csv", f"{VALUE}/grad_faith_{p}.csv"),
             "label": "value@20", "marker": "o"},
            {"front": load_front(f"{SOBOLEV}/pareto_{p}.csv", f"{SOBOLEV}/grad_faith_{p}.csv"),
             "label": "Sobolev@20", "marker": "s"},
        ]
        if p == "ns":
            rows.append({"front": load_front(f"{BUDGET}/pareto_ns.csv", f"{BUDGET}/grad_faith_ns.csv"),
                         "label": "value@budget", "marker": "^"})
        fronts[p] = rows

    # ns Mirage arrow: point at the lowest-value_mse candidate that still FAILS the gate
    nsf = [r["front"] for r in fronts["ns"]]
    import pandas as pd
    allns = pd.concat(nsf, ignore_index=True)
    fails = allns[(allns["grad_err"] > GATE_TOL) & allns["value_mse"].notna()]
    tgt = fails.sort_values("value_mse").iloc[0]
    annotate = {"ns": dict(
        text="lowest value error,\nstill red = Mirage",
        xy=(float(tgt["Complexity"]), float(tgt["value_mse"])),
        xytext=(float(tgt["Complexity"]) - 14, float(tgt["value_mse"]) * 9),
    )}
    p1 = out / "pareto_faithfulness.pdf"
    render_grid(fronts, p1, y_col="value_mse",
                y_label="value MSE vs GP (log P, HF) — lower is better",
                param_order=list(PARAM_NAMES), annotate=annotate)
    render_grid(fronts, out / "pareto_faithfulness.png", y_col="value_mse",
                y_label="value MSE vs GP (log P, HF) — lower is better",
                param_order=list(PARAM_NAMES), annotate=annotate)

    # ---------- Figure 2: scorecard (numbers on the two resisters) ----------
    rows = sorted(((p, bestloss(VALUE, p), bestloss(SOBOLEV, p)) for p in PARAM_NAMES),
                  key=lambda r: r[2])
    labels = [r[0] for r in rows]
    val = [r[1] for r in rows]
    sob = [r[2] for r in rows]
    x = np.arange(len(labels))
    fig, ax = plt.subplots(figsize=(11, 4.4), layout="constrained")
    for xi, v, s in zip(x, val, sob):
        ax.plot([xi, xi], [min(v, 1.05), min(s, 1.05)], color="0.8", lw=1, zorder=1)
    ax.scatter(x, np.clip(val, 0, 1.05), s=80, marker="o", facecolor="#d6604d",
               edgecolor="k", lw=.5, zorder=3, label="value@20 (value-optimal eq)")
    ax.scatter(x, np.clip(sob, 0, 1.05), s=80, marker="s", facecolor="#1a9850",
               edgecolor="k", lw=.5, zorder=3, label="Sobolev@20 (value-optimal eq)")
    ax.axhline(GATE_TOL, color="k", ls="--", lw=1.2)
    ax.text(0.2, GATE_TOL + 0.02, "gate 0.25", fontsize=9)
    for xi, (p, v, s) in zip(x, rows):
        if s > GATE_TOL:  # the resisters: label both numbers
            ax.annotate(f"{s:.2f}", (xi, min(s, 1.05)), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=8, color="#1a9850",
                        fontweight="bold")
            ax.annotate("resists", (xi, min(s, 1.05)), textcoords="offset points",
                        xytext=(0, 19), ha="center", fontsize=8, color="#7f0000")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel("grad_err of value-optimal eq (clipped 1.05)")
    ax.set_title("Per-parameter derivative faithfulness: value-loss vs Sobolev (z=3.6)")
    ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=.25); ax.legend(loc="upper left", fontsize=9)
    fig.savefig(out / "faithfulness_scorecard.pdf")
    fig.savefig(out / "faithfulness_scorecard.png", dpi=150)
    plt.close(fig)

    # ---------- Figure 3: ns money panel (endpoints labelled) ----------
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=GATE_TOL, vmax=1.0)
    fig, ax = plt.subplots(figsize=(7.8, 5.4), layout="constrained")
    sc = None
    for lab, d, mk in [("value@20", VALUE, "o"), ("value@budget (maxsize 35)", BUDGET, "^"),
                       ("Sobolev@20", SOBOLEV, "s")]:
        df = read_grad_faith_sidecar(f"{d}/grad_faith_ns.csv")
        sc = ax.scatter(df["Complexity"], df["value_mse"], c=np.clip(df["grad_err"], 0, 1),
                        cmap=cmap, norm=norm, marker=mk, s=72, edgecolor="k", lw=.5, label=lab)
    # label the two load-bearing endpoints (best-loss of budget and Sobolev)
    bdf = read_grad_faith_sidecar(f"{BUDGET}/grad_faith_ns.csv").sort_values("Loss").iloc[0]
    sdf = read_grad_faith_sidecar(f"{SOBOLEV}/grad_faith_ns.csv").sort_values("Loss").iloc[0]
    ax.annotate(f"budget: grad_err {bdf['grad_err']:.3f} FAIL\nvalue {bdf['value_mse']:.1e}",
                xy=(bdf["Complexity"], bdf["value_mse"]), xytext=(20, bdf["value_mse"] * 0.45),
                fontsize=8, color="#7f0000",
                arrowprops=dict(arrowstyle="->", color="#7f0000", lw=1.2))
    ax.annotate(f"Sobolev: grad_err {sdf['grad_err']:.3f} PASS\nvalue {sdf['value_mse']:.1e}",
                xy=(sdf["Complexity"], sdf["value_mse"]), xytext=(7, sdf["value_mse"] * 2.6),
                fontsize=8, color="#1a9850",
                arrowprops=dict(arrowstyle="->", color="#1a9850", lw=1.2))
    ax.set_yscale("log"); ax.set_xlabel("complexity")
    ax.set_ylabel("value MSE vs GP (log P, HF) — lower is better")
    ax.set_title("ns — budget reaches the lowest value error but never goes green;\n"
                 "Sobolev clears the gate at a comparable (~24% higher) value error")
    ax.grid(which="both", alpha=.25); ax.legend(loc="upper right", fontsize=9)
    cb = fig.colorbar(sc, ax=ax); cb.set_label("grad_err (clipped 1)")
    cb.ax.axhline(GATE_TOL, color="k", lw=1.2)
    fig.savefig(out / "ns_budget_panel.pdf")
    fig.savefig(out / "ns_budget_panel.png", dpi=150)
    plt.close(fig)

    # ---------- Figure 4: cross-z robustness (Sobolev best-loss grad_err vs z) ----------
    CROSSZ = {
        2.6: "results/single_z_z2.6_sobolev/refit/z2.6",
        3.6: SOBOLEV,
        4.2: "results/single_z_z4.2_sobolev/refit/z4.2",
    }
    zs = [2.6, 3.6, 4.2]
    cmapz = plt.get_cmap("tab20")
    fig, ax = plt.subplots(figsize=(9, 5.2), layout="constrained")
    for i, p in enumerate(PARAM_NAMES):
        s = []
        for z in zs:
            try:
                s.append(bestloss(CROSSZ[z], p))
            except Exception:
                s.append(float("nan"))
        ax.plot(zs, np.clip(s, 0, 1.2), marker="o", lw=1.4, color=cmapz(i % 20), label=p)
    ax.axhline(GATE_TOL, color="k", ls="--", lw=1.3)
    ax.text(4.2, GATE_TOL + 0.01, "gate 0.25", ha="right", fontsize=9)
    ax.set_xticks(zs); ax.set_xlabel("redshift z")
    ax.set_ylabel("Sobolev best-loss grad_err (clipped 1.2)")
    ax.set_title("Redshift robustness of the derivative-faithfulness taxonomy (Sobolev fits)")
    ax.set_ylim(0, 1.25); ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=8, loc="upper center")
    fig.savefig(out / "crossz_faithfulness.pdf")
    fig.savefig(out / "crossz_faithfulness.png", dpi=150)
    plt.close(fig)

    names = ["pareto_faithfulness", "faithfulness_scorecard", "ns_budget_panel",
             "crossz_faithfulness"]
    for dest in args.also_copy_to:
        Path(dest).mkdir(parents=True, exist_ok=True)
        for n in names:
            for ext in ("png", "pdf"):
                src = out / f"{n}.{ext}"
                if src.exists():
                    (Path(dest) / f"{n}.{ext}").write_bytes(src.read_bytes())
    print(f"wrote 4 figures (png+pdf) to {out}" +
          (f" and copied PNG+PDF to {args.also_copy_to}" if args.also_copy_to else ""))


if __name__ == "__main__":
    main()
