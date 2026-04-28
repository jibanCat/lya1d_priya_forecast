"""Run Fisher (and optionally MCMC) for every YAML in `configs/eqns/`.

Phase 7 will fill this out. Produces:
- summary table: rows = parameters, cols = equation sets, values = sigma_i / sigma_GP.
- 1D marginalized posterior overlays.
- residual plot at fiducial: (P_PySR - P_GP) / sqrt(diag(C_eBOSS)) vs k.
"""


def main():  # pragma: no cover - phase 7
    raise SystemExit("compare_eqn_sets.py is not yet implemented (phase 7).")


if __name__ == "__main__":
    main()
