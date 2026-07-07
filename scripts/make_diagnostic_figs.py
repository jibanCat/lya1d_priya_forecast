#!/usr/bin/env python
"""Regenerate the four diagnostic paper figures with referee-requested annotations.

Emulator-free: reads the cached PySR Pareto fronts + grad-faith sidecars. Produces,
into --out-dir (PNG + PDF):
  1. pareto_faithfulness  -- 11-panel grid, y=value_mse, colour=grad_err, with the
     gate made legible on each panel (bold ring = clears 0.25) + an annotated arrow
     on the ns panel at the low-but-red Mirage cluster.
  2. faithfulness_scorecard -- value vs Sobolev best-loss grad_err, with the two
     above-gate parameters labelled with their numbers.
  3. ns_budget_panel -- the paired budget-vs-Sobolev comparison, endpoints labelled.
  4. crossz_faithfulness -- redshift robustness of the taxonomy, z=2.6/3.6/4.2.

Usage (defaults read the committed paper-production run):
  PYTHONPATH=src python scripts/make_diagnostic_figs.py \
    --out-dir results/_repro_scratch/diagnostic

Paper production run (results/paper_production_20260630_perz_sobolev_z2.6-4.2),
layout value/refit/z<z>, sobolev/refit/z<z>, seed_band/z3.6_seed0_budget/refit/z3.6:
  PROD=results/paper_production_20260630_perz_sobolev_z2.6-4.2
  PYTHONPATH=src python scripts/make_diagnostic_figs.py \
    --value-dir   $PROD/value/refit/z3.6 \
    --sobolev-dir $PROD/sobolev/refit/z3.6 \
    --budget-dir  $PROD/seed_band/z3.6_seed0_budget/refit/z3.6 \
    --crossz-dirs 2.6=$PROD/sobolev/refit/z2.6 \
                  3.6=$PROD/sobolev/refit/z3.6 \
                  4.2=$PROD/sobolev/refit/z4.2 \
    --out-dir     $PROD/figs
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
from priya_forecast.grad_faith_io import read_grad_faith_sidecar, knee_row
from priya_forecast.pareto_diag import load_front, render_grid, GATE_TOL

# Enlarged, print-legible typography for paper production (column-width PDFs).
# LaTeX-rendered text (usetex) + large fonts so labels are legible in print.
# rcParams set here are global, so the shared pareto-grid renderer in
# priya_forecast.pareto_diag picks them up for any size it does not set explicitly.
plt.rcParams.update({
    "text.usetex": True,
    "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
    "font.family": "serif",
    "font.size": 18,
    "axes.titlesize": 20,
    "axes.labelsize": 22,
    "xtick.labelsize": 17,
    "ytick.labelsize": 17,
    "legend.fontsize": 16,
    "figure.titlesize": 20,
    "savefig.dpi": 150,
})

# Default input dirs reproduce the committed single-z diagnostic figures; each is
# overridable on the CLI so the generators can be pointed at a production run.
_PROD = "results/paper_production_20260630_perz_sobolev_z2.6-4.2"
DEFAULT_VALUE = f"{_PROD}/value/refit/z3.6"
DEFAULT_SOBOLEV = f"{_PROD}/sobolev/refit/z3.6"
DEFAULT_BUDGET = f"{_PROD}/seed_band/z3.6_seed0_budget/refit/z3.6"


def bestloss(d, p, col="grad_err"):
    # Pareto-KNEE pick (lowest complexity within 10% of best loss), not idxmin:
    # on a complexity-truncated front idxmin = the ceiling equation, whose
    # gradient is artificially bad, which would make the Mirage look like a mere
    # selection artifact. The knee is the equation a practitioner would actually
    # use, so a knee that is still derivative-unfaithful is a genuine Mirage.
    df = read_grad_faith_sidecar(f"{d}/grad_faith_{p}.csv")
    return float(knee_row(df)[col])


def parse_crossz(tokens, sobolev_dir):
    """Build the {z: dir} cross-z map. `tokens` is a list of 'z=path' strings;
    when empty, fall back to the committed default (z=3.6 reuses --sobolev-dir)."""
    if not tokens:
        return {
            2.6: f"{_PROD}/sobolev/refit/z2.6",
            3.6: sobolev_dir,
            4.2: f"{_PROD}/sobolev/refit/z4.2",
        }
    out = {}
    for tok in tokens:
        z, _, path = tok.partition("=")
        out[float(z)] = path
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out-dir", default="results/_repro_scratch/diagnostic")
    ap.add_argument("--value-dir", default=DEFAULT_VALUE,
                    help="dir holding value-loss pareto_<p>.csv + grad_faith_<p>.csv")
    ap.add_argument("--sobolev-dir", default=DEFAULT_SOBOLEV,
                    help="dir holding Sobolev-loss pareto_<p>.csv + grad_faith_<p>.csv")
    ap.add_argument("--budget-dir", default=DEFAULT_BUDGET,
                    help="dir holding the value@budget ns front (pareto_ns/grad_faith_ns)")
    ap.add_argument("--crossz-dirs", nargs="*", default=[], metavar="Z=DIR",
                    help="cross-z Sobolev dirs as 'z=path' tokens "
                         "(e.g. 2.6=.../sobolev/refit/z2.6); "
                         "default uses the committed single-z z2.6/z3.6/z4.2 dirs")
    ap.add_argument("--also-copy-to", action="append", default=[],
                    help="extra dirs to copy the PNGs into (repeatable)")
    args = ap.parse_args()
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)

    VALUE = args.value_dir
    SOBOLEV = args.sobolev_dir
    BUDGET = args.budget_dir

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
    y_label_value = r"value MSE vs GP ($\log P$, HF) -- lower is better"
    # The in-figure ns Mirage arrow (`annotate`) collided with the ns panel's
    # legend + data points at print size, so it is dropped from the figure and
    # moved to the caption (see caption sentence in the regen report). `annotate`
    # is still computed above for provenance but no longer drawn.
    p1 = out / "pareto_faithfulness.pdf"
    render_grid(fronts, p1, y_col="value_mse",
                y_label=y_label_value,
                param_order=list(PARAM_NAMES), annotate=None)
    render_grid(fronts, out / "pareto_faithfulness.png", y_col="value_mse",
                y_label=y_label_value,
                param_order=list(PARAM_NAMES), annotate=None)

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
               edgecolor="k", lw=.5, zorder=3, label="value@20 (knee-selected eq)")
    ax.scatter(x, np.clip(sob, 0, 1.05), s=80, marker="s", facecolor="#1a9850",
               edgecolor="k", lw=.5, zorder=3, label="Sobolev@20 (knee-selected eq)")
    ax.axhline(GATE_TOL, color="k", ls="--", lw=1.2)
    ax.text(0.2, GATE_TOL + 0.02, "gate 0.25", fontsize=16)
    for xi, (p, v, s) in zip(x, rows):
        if s > GATE_TOL:  # the resisters: label both numbers
            ax.annotate(f"{s:.2f}", (xi, min(s, 1.05)), textcoords="offset points",
                        xytext=(0, 7), ha="center", fontsize=15, color="#1a9850",
                        fontweight="bold")
            ax.annotate("resists", (xi, min(s, 1.05)), textcoords="offset points",
                        xytext=(0, 21), ha="center", fontsize=15, color="#7f0000")
    ax.set_xticks(x); ax.set_xticklabels(labels, rotation=35, ha="right")
    ax.set_ylabel(r"$\mathrm{grad\_err}$ of knee-selected eq (clipped 1.05)")
    ax.set_title(r"Per-parameter derivative faithfulness: value-loss vs Sobolev ($z=3.6$)")
    ax.set_ylim(0, 1.12); ax.grid(axis="y", alpha=.25); ax.legend(loc="upper left", fontsize=16)
    fig.savefig(out / "faithfulness_scorecard.pdf", bbox_inches="tight")
    fig.savefig(out / "faithfulness_scorecard.png", dpi=150, bbox_inches="tight")
    plt.close(fig)

    # ---------- Figure 3: ns money panel (SINGLE-COLUMN, decluttered) ----------
    # Included at \linewidth in a one-column `figure` (~3.4in wide), so the panel
    # is sized to the column with a small, self-consistent font set (a large
    # 12in figure downscaled to a column rendered ~5pt text). The two endpoint
    # callouts overlapped the point cluster at this size, and the two-line title
    # duplicated the caption, so both are dropped from the figure and moved to
    # the caption (numbers reported in the regen report). Same three fronts, same
    # hard two-tone (green faithful <=0.25 / red Mirage), same data.
    cmap = mcolors.ListedColormap(["#1a9850", "#d6604d"])
    norm = mcolors.BoundaryNorm([0.0, GATE_TOL + 1e-12, 1.0], cmap.N)
    with plt.rc_context({
        "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 11,
        "xtick.labelsize": 9, "ytick.labelsize": 9, "legend.fontsize": 8,
    }):
        fig, ax = plt.subplots(figsize=(3.7, 3.45), layout="constrained")
        sc = None
        for lab, d, mk in [("value@20", VALUE, "o"),
                           ("value@budget", BUDGET, "^"),
                           ("Sobolev@20", SOBOLEV, "s")]:
            df = read_grad_faith_sidecar(f"{d}/grad_faith_ns.csv")
            sc = ax.scatter(df["Complexity"], df["value_mse"],
                            c=np.clip(df["grad_err"], 0, 1), cmap=cmap, norm=norm,
                            marker=mk, s=34, edgecolor="k", lw=.4, label=lab)
        ax.set_yscale("log")
        ax.set_xlabel("complexity")
        ax.set_ylabel(r"value MSE vs GP ($\log P$, HF)")
        ax.grid(which="both", alpha=.25)
        ax.legend(loc="upper right", handletextpad=0.3, borderpad=0.3)
        cb = fig.colorbar(sc, ax=ax, ticks=[GATE_TOL])
        cb.set_label(r"$\mathrm{grad\_err}$ (green $\leq 0.25$, red $> 0.25$)",
                     fontsize=9)
        cb.ax.tick_params(labelsize=8)
        fig.savefig(out / "ns_budget_panel.pdf", bbox_inches="tight")
        fig.savefig(out / "ns_budget_panel.png", dpi=150, bbox_inches="tight")
        plt.close(fig)

    # ---------- Figure 4: cross-z robustness (Sobolev best-loss grad_err vs z) ----------
    CROSSZ = parse_crossz(args.crossz_dirs, SOBOLEV)
    zs = sorted(CROSSZ)
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
    ax.text(max(zs), GATE_TOL + 0.01, "gate 0.25", ha="right", fontsize=16)
    ax.set_xticks(zs); ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"Sobolev best-loss $\mathrm{grad\_err}$ (clipped 1.2)")
    ax.set_title("Redshift robustness of the derivative-faithfulness taxonomy (Sobolev fits)")
    ax.set_ylim(0, 1.25); ax.grid(alpha=.25); ax.legend(ncol=2, fontsize=14, loc="upper center")
    fig.savefig(out / "crossz_faithfulness.pdf", bbox_inches="tight")
    fig.savefig(out / "crossz_faithfulness.png", dpi=150, bbox_inches="tight")
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
