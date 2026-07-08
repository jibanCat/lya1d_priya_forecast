# §5.1 joint multi-D PySR fit — maxsize sweep (budget vs structural)

Settles whether the joint-fit Fisher rank-deficiency is a search-budget artifact.
Subset `{ns, Ap, herei, heref, alphaq, hireionz}`, z=3.6, MSE objective, 5 seeds/budget.
Jobs 53055690 (ms 25/40/60, niter=400) + 53071661 (ms 100, niter=250, 12h wall).
Diagnostic: `scripts/run_multid_pysr.py` `_front_rank_scan` -> `joint_rank_diagnostic.json`.

GP reference (whitened): rank 6/6, eigenvalues [469,103,1.75,0.41,0.12,0.066], cond 7133.

| maxsize | front-max rank (med, range) | detached / all-6-inputs | median λ5 | median λ6 | #singular | off-fid (med) |
|--------:|:---------------------------:|:-----------------------:|:---------:|:---------:|:---------:|:-------------:|
| 25      | 2 (2-3)                     | 1/5 · 0/5 (z_Hi dropped)| ~0        | ~0        | 5/5       | 0.326         |
| 40      | 2 (2-3)                     | 3/5 · 5/5               | ~0        | ~0        | 5/5       | 0.128         |
| 60      | 4 (3-5)                     | 4/5 · 5/5               | ~0        | ~0        | 5/5       | 0.066         |
| 100     | 5 (4-6)                     | 4/5 · 5/5               | 1.2e-3    | ~0        | 3/5       | 0.047         |

## Verdict (pre-registered)
- STRUCTURAL criterion (rank stays <=3 once front detaches + all 6 inputs): **FAILS** — rank climbs 2->2->4->5, one seed hits 6.
- BUDGET criterion (rank climbs toward 6 with maxsize; cond finite-ing; off-fid -> combine): **largely MET**. The original maxsize=25 rank-2 result was search-starvation (z_Hi dropped; fronts pinned at the cap).
- Residual: even at maxsize=100 the median fit resolves only ~5/6 directions; the sloppiest He II direction (~1e-3 of the Fisher info) never lifts off machine-zero, and 3/5 seeds stay singular — possibly itself budget-limited (ms=100 used the conservative niter=250).
- Clean survivor: the per-parameter combine is rank-6 by construction at ~1/6 the budget.

Reproduce: `scripts/submit_joint_sweep.sh` (see `slurm/joint_multid_sweep.slurm`).
