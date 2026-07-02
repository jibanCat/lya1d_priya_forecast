#!/usr/bin/env python
"""Maxsize-sensitivity of derivative faithfulness: value-loss vs Sobolev.

This is the reframe's key piece of evidence. The earlier worry was that the
"ns Fisher-Mirage" (a value-loss equation that nails the P1D *value* but whose
slope dlogP/dtheta is wrong) might be pure search-starvation: give PySR a bigger
complexity budget (`maxsize`) and the value-loss equation would supposedly find
the faithful form. This figure tests that head-on.

For every PRIYA parameter we read the *knee-selected* equation's gradient error
(`grad_err` = median_k |dlogP_eq/dtheta / dlogP_GP/dtheta - 1| at fid, the
production gate metric) from the grad-faith sidecars at four complexity budgets
maxsize in {20, 30, 35, 40}, separately for the value-loss objective and the
Sobolev objective. We do NOT re-run any emulator here -- the sidecars
(grad_faith_<param>.csv) already carry the per-candidate grad_err; we just pick
the Pareto-knee row (priya_forecast.grad_faith_io.knee_row) and tabulate it.

Sidecar sources (all at z=3.6, layout <subdir>/refit/z3.6/grad_faith_<p>.csv):
    maxsize 20 -> value            / sobolev            (the main production fits)
    maxsize 30 -> sens_maxsize30_value / sens_maxsize30_sobolev
    maxsize 35 -> budget35_value (all 11) / budget35_sobolev (only the params
                  that were re-fit at 35 under Sobolev, typically hub+bhfeedback)
    maxsize 40 -> sens_maxsize40_value / sens_maxsize40_sobolev
A missing sidecar (param not re-fit at that budget, or a segfaulted cluster job)
is simply skipped -- it leaves a gap in that series rather than a fake point.

Outputs (into --out-dir):
  * maxsize_sensitivity.csv   columns: param, loss, maxsize, grad_err, complexity
  * maxsize_sensitivity.png / .pdf  two panels (value | Sobolev), shared log-y,
    one line per parameter (grad_err vs maxsize), gate=0.25 drawn on both.

The story the figure makes visible: under the *value* objective the knee
grad_err FALLS as maxsize grows (ns and omegamh2 eventually drop toward/under
the gate -- the equation needed a bigger budget to find a slope-faithful form),
whereas the *Sobolev* objective is already low at maxsize=20 and stays flat ->
Sobolev is search-efficient; value-loss faithfulness is budget-sensitive.

Usage:
  PYTHONPATH=src .venv/bin/python scripts/regen_maxsize_sensitivity.py \
    --prod results/paper_production_20260630_perz_sobolev_z2.6-4.2 \
    --z 3.6 --out-dir <prod>/figures
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np

from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.grad_faith_io import read_grad_faith_sidecar, knee_row

GATE = 0.25

# loss -> {maxsize: subdir under --prod}. The grad-faith sidecar for parameter p
# lives at  <prod>/<subdir>/refit/z<z>/grad_faith_<p>.csv .
SOURCES = {
    "value": {
        20: "value",
        30: "sens_maxsize30_value",
        35: "budget35_value",
        40: "sens_maxsize40_value",
    },
    "sobolev": {
        20: "sobolev",
        30: "sens_maxsize30_sobolev",
        35: "budget35_sobolev",
        40: "sens_maxsize40_sobolev",
    },
}
MAXSIZES = [20, 30, 35, 40]


def knee_grad_err(sidecar: Path):
    """(grad_err, complexity) of the Pareto-knee candidate, or (nan, nan)."""
    if not sidecar.exists():
        return np.nan, np.nan
    try:
        df = read_grad_faith_sidecar(sidecar)
        if df.empty:
            return np.nan, np.nan
        row = knee_row(df)
        return float(row["grad_err"]), int(row["Complexity"])
    except Exception as exc:  # noqa: BLE001 -- diagnostic: skip a bad sidecar
        print(f"  [warn] {sidecar}: {exc}")
        return np.nan, np.nan


def gather(prod: Path, z: float):
    """Long-format records {param, loss, maxsize, grad_err, complexity}."""
    rows = []
    for p in PARAM_NAMES:
        for loss, by_size in SOURCES.items():
            for ms in MAXSIZES:
                sub = by_size.get(ms)
                if sub is None:
                    continue
                sc = prod / sub / "refit" / f"z{z}" / f"grad_faith_{p}.csv"
                ge, cx = knee_grad_err(sc)
                if not np.isfinite(ge):
                    continue
                rows.append({"param": p, "loss": loss, "maxsize": ms,
                             "grad_err": ge, "complexity": cx})
    return rows


def write_csv(out_csv: Path, rows: list[dict]) -> None:
    cols = ["param", "loss", "maxsize", "grad_err", "complexity"]
    from priya_forecast.provenance import git_stamp
    with open(out_csv, "w", newline="") as fh:
        fh.write(f"# git={git_stamp()} source=maxsize_sweep\n")
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in sorted(rows, key=lambda r: (r["loss"], PARAM_NAMES.index(r["param"]), r["maxsize"])):
            w.writerow(r)


def make_figure(out_base: Path, rows: list[dict], *, z: float) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}\usepackage{amssymb}",
        "font.family": "serif",
        "font.size": 18, "axes.titlesize": 22, "axes.labelsize": 22,
        "xtick.labelsize": 17, "ytick.labelsize": 17, "legend.fontsize": 16,
        "figure.titlesize": 20,
    })
    cmap = plt.get_cmap("tab20")
    # Parameters whose value-loss faithfulness is budget-sensitive: highlight.
    HILITE = {"ns", "omegamh2"}

    fig, axes = plt.subplots(1, 2, figsize=(16, 7), sharey=True)
    for ax, loss in zip(axes, ("value", "sobolev")):
        for i, p in enumerate(PARAM_NAMES):
            pts = sorted([r for r in rows if r["loss"] == loss and r["param"] == p],
                         key=lambda r: r["maxsize"])
            if not pts:
                continue
            xs = [r["maxsize"] for r in pts]
            ys = [r["grad_err"] for r in pts]
            hot = p in HILITE
            ax.plot(xs, ys, marker="o", ms=9 if hot else 6,
                    lw=3.2 if hot else 1.6, color=cmap(i % 20),
                    alpha=1.0 if hot else 0.8, zorder=5 if hot else 3,
                    label=p + (" *" if hot else ""))
        ax.axhline(GATE, color="k", ls="--", lw=1.6)
        ax.text(20, GATE * 1.06, "gate 0.25", fontsize=16, va="bottom")
        ax.set_yscale("log")
        ax.set_xticks(MAXSIZES)
        ax.set_xlabel(r"PySR $\mathrm{maxsize}$ (complexity budget)")
        ax.set_title(rf"{loss}-loss")
        ax.grid(alpha=0.3, which="both")
    axes[0].set_ylabel(r"knee-selected $\mathrm{grad\_err}$  (log scale)")
    axes[1].legend(ncol=2, loc="upper right", frameon=True, framealpha=0.9)
    fig.suptitle(
        rf"Derivative faithfulness vs complexity budget ($z={z}$): "
        "value-loss is budget-sensitive, Sobolev is search-efficient",
        fontsize=20,
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prod", type=Path, required=True,
                    help="production root holding value/, sobolev/, sens_*/, budget35_* subdirs")
    ap.add_argument("--z", type=float, default=3.6)
    ap.add_argument("--out-dir", type=Path, required=True)
    args = ap.parse_args()
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = gather(args.prod, args.z)
    out_csv = args.out_dir / "maxsize_sensitivity.csv"
    write_csv(out_csv, rows)
    print(f"wrote {out_csv}  ({len(rows)} rows)")

    out_fig = args.out_dir / "maxsize_sensitivity"
    make_figure(out_fig, rows, z=args.z)
    print(f"wrote {out_fig}.png / {out_fig}.pdf")

    # Console summary for the two headline budget-sensitive params + the two
    # params that have a Sobolev@35 point.
    print(f"\n{'param':>10} {'loss':>8} | " + " ".join(f"ms{m:>2}" for m in MAXSIZES))
    for p in ("ns", "omegamh2", "hub", "bhfeedback"):
        for loss in ("value", "sobolev"):
            cells = []
            for m in MAXSIZES:
                hit = [r for r in rows if r["param"] == p and r["loss"] == loss and r["maxsize"] == m]
                cells.append(f"{hit[0]['grad_err']:5.3f}" if hit else "  -  ")
            print(f"{p:>10} {loss:>8} | " + " ".join(cells))


if __name__ == "__main__":
    main()
