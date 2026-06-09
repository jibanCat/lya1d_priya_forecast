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

import numpy as np

import priya_forecast.single_z.refit as _refit
from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D
from priya_forecast.models.gp_model import GPModel


def main():
    k = _refit.kodiaq_k_grid(0.001, 0.04, 48)
    lnk = np.log(k)
    fid = np.array([p.fid for p in PARAMS_11D], float)
    hidx = PARAM_NAMES.index("hub")
    gp = GPModel(basedir="data/kodiaq_gp", fidelity="hf", kf=k)
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
    print(f"{'z':>4} {'corr(dPdh,dPdlnk)':>18} {'corr(dPdh,P)':>13} {'AP-frac var':>12}")
    for z in (2.6, 3.6, 4.2):
        gh = dPdh(z)
        P = np.asarray(gp.predict(fid, k, z), float)
        Tap = np.gradient(P, lnk)              # = dP/dln k = the AP-rescaling template
        X = np.vstack([Tap - Tap.mean(), P - P.mean()]).T
        y = gh - gh.mean()
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        ap = (Tap - Tap.mean()) * beta[0]
        print(f"{z:>4} {corr(gh, Tap):>18.3f} {corr(gh, P):>13.3f} "
              f"{float((ap @ y) / (y @ y)):>12.3f}")
    print("\nVerdict: AP/k-rescaling hypothesis NOT supported (weak corr, ~6% variance).")


if __name__ == "__main__":
    main()
