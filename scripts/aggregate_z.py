#!/usr/bin/env python
"""Aggregate per-z single-z forecasts into an across-z view.

Reads `<base>/z{z}/fisher_{GP,perfect_1D,PySR}.npz` for each z-bin present
and writes `<base>/aggregate/`: a σ(z) trend plot and a σ-table.

    python scripts/aggregate_z.py --base results/single_z_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 13 kodiaq z-bins.
Z_BINS_13 = [round(z, 1) for z in np.arange(2.2, 4.601, 0.2)]
LABELS = ("GP", "perfect_1D", "PySR")


def collect_sigma_z(*, base_dir, label: str, z_bins) -> dict:
    """Return {param_name: {z: sigma}} for one label across the z-bins present."""
    base_dir = Path(base_dir)
    out: dict[str, dict[float, float]] = {}
    for z in z_bins:
        npz = base_dir / f"z{z}" / f"fisher_{label}.npz"
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=True)
        names = [str(n) for n in d["param_names"]]
        for name, s in zip(names, d["sigma"]):
            out.setdefault(name, {})[z] = float(s)
    return out


def aggregate(*, base_dir, z_bins=None) -> Path:
    """Write the σ(z) plot + table to `<base_dir>/aggregate/`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_dir = Path(base_dir)
    z_bins = list(Z_BINS_13 if z_bins is None else z_bins)
    per_label = {lab: collect_sigma_z(base_dir=base_dir, label=lab, z_bins=z_bins)
                 for lab in LABELS}
    agg = base_dir / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)

    params = sorted({p for tbl in per_label.values() for p in tbl})
    if not params:
        raise FileNotFoundError(
            f"No fisher_*.npz found under {base_dir}/z*/ — run the pipeline first."
        )
    ncol = min(4, len(params))
    nrow = (len(params) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             dpi=120, squeeze=False)
    for i, param in enumerate(params):
        ax = axes[i // ncol][i % ncol]
        for lab in LABELS:
            tbl = per_label[lab].get(param, {})
            if not tbl:
                continue
            zs = sorted(tbl)
            ax.plot(zs, [tbl[z] for z in zs], "o-", label=lab)
        ax.set_title(param)
        ax.set_xlabel("z")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    for j in range(len(params), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle("Single-z forecast — σ vs redshift")
    fig.tight_layout()
    fig.savefig(agg / "sigma_vs_z.png")
    plt.close(fig)

    lines = ["# Across-z σ table\n"]
    for lab in LABELS:
        tbl = per_label[lab]
        if not tbl:
            continue
        zs = sorted({z for d in tbl.values() for z in d})
        lines.append(f"\n## {lab}\n")
        lines.append("| param | " + " | ".join(f"z={z}" for z in zs) + " |")
        lines.append("|" + "---|" * (len(zs) + 1))
        for param in params:
            row = tbl.get(param, {})
            cells = " | ".join(
                f"{row[z]:.4g}" if z in row else "—" for z in zs
            )
            lines.append(f"| {param} | {cells} |")
    (agg / "sigma_table.md").write_text("\n".join(lines) + "\n")
    return agg


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", required=True,
                   help="Run directory containing z{z}/ subdirs.")
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    out = aggregate(base_dir=args.base)
    print(f"wrote {out}/sigma_vs_z.png and {out}/sigma_table.md")


if __name__ == "__main__":
    main()
