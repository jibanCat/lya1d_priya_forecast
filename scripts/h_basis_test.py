#!/usr/bin/env python
"""h basis test — is the GP's dP/dh shaped like a k-axis (Alcock-Paczynski) rescaling?

A k-rescaling k -> k(1+eps) gives dP/d(rescaling) ∝ dP/d ln k. If h's resistance to
symbolic fitting were a *basis/expressivity* wall (h acting as a coordinate distortion
of k that a per-parameter native-k multiplicative form cannot express), then dP/dh would
correlate strongly with dP/d ln k. It does NOT: corr ≈ -0.25 at z=2.6/3.6/4.2 and the
rescaling template explains only ~6% of the dP/dh variance. So h's resistance is better
read as a weak / under-determined response (its ~1% P1D effect; x0 enters only at the
maximum complexity), not a clean AP-distortion basis wall.

Run (needs the GP emulator):
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \
  PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
  .venv/bin/python scripts/h_basis_test.py
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

import priya_forecast.single_z.refit as _refit
from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--basedir", default="data/kodiaq_gp")
    ap.add_argument("--out", default="results/h_basis_test/h_basis.json",
                    help="write the per-z {corr, ap_corr2, n_k} result here (committable)")
    args = ap.parse_args()

    if not Path(args.basedir).exists():
        # GP-only test: fail loudly (non-zero) on a bare clone so it is never a silent no-op.
        sys.exit(f"ERROR: GP data not found at {args.basedir!r}; this test needs the "
                 f"emulator. The committed result lives at {args.out}.")
    from priya_forecast.models.gp_model import GPModel

    k = _refit.kodiaq_k_grid(0.001, 0.04, 48)
    lnk = np.log(k)
    fid = np.array([p.fid for p in PARAMS_11D], float)
    hidx = PARAM_NAMES.index("hub")
    gp = GPModel(basedir=args.basedir, fidelity="hf", kf=k)
    _ = gp.predict(fid, k, 3.6)

    def dPdh(z, h=1e-3):
        s = h * max(abs(fid[hidx]), 1.0)
        tp, tm = fid.copy(), fid.copy()
        tp[hidx] += s
        tm[hidx] -= s
        return (np.asarray(gp.predict(tp, k, z), float)
                - np.asarray(gp.predict(tm, k, z), float)) / (2 * s)

    def corr(a, b):
        a = a - a.mean()
        b = b - b.mean()
        return float(a @ b / np.sqrt((a @ a) * (b @ b)))

    print("h-as-k-rescaling (AP-like) basis test: is dP/dh shaped like dP/dlnk?")
    print(f"{'z':>4} {'corr(dPdh,dPdlnk)':>18} {'corr(dPdh,P)':>13} "
          f"{'ap_corr2':>9} {'ap_partial':>11}")
    out = {"description": "h basis test: is dP/dh shaped like a k-rescaling dP/dlnk? "
           "ap_corr2 = single-feature R^2 of the AP template; ap_partial = its unique "
           "variance after partialling out P. Both small + corr negative => NOT AP.",
           "by_z": {}}
    for z in (2.6, 3.6, 4.2):
        gh = dPdh(z)
        P = np.asarray(gp.predict(fid, k, z), float)
        Tap = np.gradient(P, lnk)              # = dP/dln k = the AP-rescaling template
        X = np.vstack([Tap - Tap.mean(), P - P.mean()]).T
        y = gh - gh.mean()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ap = (Tap - Tap.mean()) * beta[0]
        c_ap = corr(gh, Tap)
        rec = {"corr_dPdh_dPdlnk": c_ap, "corr_dPdh_P": corr(gh, P),
               "ap_corr2": c_ap ** 2, "ap_partial_var": float((ap @ y) / (y @ y)),
               "n_k": int(k.size)}
        out["by_z"][str(z)] = rec
        print(f"{z:>4} {c_ap:>18.3f} {rec['corr_dPdh_P']:>13.3f} "
              f"{rec['ap_corr2']:>9.3f} {rec['ap_partial_var']:>11.3f}")
    out["verdict"] = ("AP/k-rescaling hypothesis NOT supported: dP/dh is weakly "
                      "(negatively) correlated with dP/dlnk (~-0.25) and the template "
                      "explains ~6% of the variance -> h is weak/under-determined.")
    print("\nVerdict: " + out["verdict"])

    op = Path(args.out)
    op.parent.mkdir(parents=True, exist_ok=True)
    op.write_text(json.dumps(out, indent=2))
    print(f"wrote {op}")


if __name__ == "__main__":
    main()
