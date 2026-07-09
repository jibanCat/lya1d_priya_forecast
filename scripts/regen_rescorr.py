#!/usr/bin/env python
"""Regenerate the paper's HF/LF resolution-correction figure (`fig:rescorr_plot`).

This is the paper's only active figure that had no committed generator. It is the
``cosmo`` block of `write_param_variation_resolution_correction`, rendered in the
paper style and saved under the name the `.tex` expects.

Tier 1 -- **no GP emulator**. The function only calls ``refit.predict`` on the
committed per-parameter refit pickles, and the k-grid travels inside those pickles.
The GP dependence people assumed came from `multi_z_aggregate.py`, the only other
caller, which loads a `GPModel` for unrelated Fisher work.

Two things the committed path did not do, and this script does:
  * renders under ``text.usetex`` with a serif family, so the labels come out in
    Computer Modern like every other figure in the paper;
  * emits ``resolution_correction.pdf`` rather than
    ``resolution_correction_param_variation_cosmo.pdf``.

Note the style is plain usetex + serif at matplotlib's default sizes, **not**
``paper_figures.paper_style``. That helper's enlarged rcParams shrink the axes
enough to drop a y-tick per panel; the published figure was made without it. The
output of this script is pixel-identical to the published PDF (rasterize both at
``gs -r150`` and compare -- matplotlib stamps a CreationDate, so bytes will differ).

Usage::

    export PATH="$HOME/texlive/2026/bin/x86_64-linux:$PATH"   # usetex needs latex+dvipng
    python scripts/regen_rescorr.py --out-dir figs/
    python scripts/regen_rescorr.py --out-dir /tmp/check --keep-astro
"""
from __future__ import annotations

import argparse
import pickle
from pathlib import Path

import numpy as np

from priya_forecast.deliverables import write_param_variation_resolution_correction
from priya_forecast.parameters import PARAM_NAMES

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_REFITS = (
    ROOT / "results" / "paper_production_20260630_perz_sobolev_z2.6-4.2"
    / "sobolev" / "refit" / "z3.6" / "refits"
)
# The published figure's suptitle reads "z = 3.60". These are single-z refits, so
# z_eval only labels the panel; it is not forwarded to predict().
DEFAULT_Z_EVAL = 3.6


def load_refits(refits_dir: Path) -> dict:
    refits = {}
    for pname in PARAM_NAMES:
        path = refits_dir / f"{pname}.pkl"
        if path.exists():
            with path.open("rb") as fh:
                refits[pname] = pickle.load(fh)
    if not refits:
        raise SystemExit(f"no refit pickles under {refits_dir}")
    return refits


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--refits-dir", type=Path, default=DEFAULT_REFITS)
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--z-eval", type=float, default=DEFAULT_Z_EVAL)
    ap.add_argument("--no-usetex", action="store_true",
                    help="skip LaTeX rendering (produces the sans-serif variant)")
    ap.add_argument("--keep-astro", action="store_true",
                    help="keep the astro-block variant, which the paper does not use")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    refits = load_refits(args.refits_dir)
    k_grid = np.asarray(next(iter(refits.values())).k_grid, dtype=float)
    print(f"loaded {len(refits)}/{len(PARAM_NAMES)} refits from {args.refits_dir}")
    print(f"k=[{k_grid[0]:.4g}, {k_grid[-1]:.4g}] s/km ({len(k_grid)} bins), z_eval={args.z_eval}")

    import matplotlib.pyplot as plt
    rc = {"font.family": "serif", "text.usetex": not args.no_usetex}
    with plt.rc_context(rc):
        write_param_variation_resolution_correction(
            refits, k_grid, args.out_dir, z_eval=args.z_eval,
        )

    stem = "resolution_correction_param_variation"
    for ext in ("pdf", "png"):
        src = args.out_dir / f"{stem}_cosmo.{ext}"
        if src.exists():
            src.replace(args.out_dir / f"resolution_correction.{ext}")
        astro = args.out_dir / f"{stem}_astro.{ext}"
        if astro.exists() and not args.keep_astro:
            astro.unlink()

    print(f"wrote {args.out_dir / 'resolution_correction.pdf'}")


if __name__ == "__main__":
    main()
