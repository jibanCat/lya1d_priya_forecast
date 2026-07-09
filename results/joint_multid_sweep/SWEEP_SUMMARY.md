# §5.1 joint multi-D PySR fit — maxsize sweep (budget vs structural)

Settles whether the joint-fit Fisher rank-deficiency is a search-budget artifact.
Subset `{ns, Ap, herei, heref, alphaq, hireionz}`, z=3.6, MSE objective, 5 seeds/budget.
Jobs 53055690 (ms 25/40/60, niter=400) + 53071661 (ms 100, niter=250, 12h wall).
Diagnostic: `scripts/run_multid_pysr.py` `_front_rank_scan` -> `joint_rank_diagnostic.json`.

**Regenerate this table — do not hand-edit it:** `python scripts/aggregate_joint_sweep.py --write-summary`

GP reference (whitened): rank 6/6, eigenvalues [469,103,1.75,0.41,0.12,0.066], cond 7133.
Sloppiest/stiffest = 1.40e-4 (whitened) / 1.23e-5 (physical).

| maxsize | front-max rank (med, range) | selected rank (med, range) | detached / all-6-inputs | median λ5 | median λ6 | rank-deficient | off-fid (med) |
|--------:|:--------------------------:|:--------------------------:|:-----------------------:|:---------:|:---------:|:--------------:|:-------------:|
| 25      | 2 (2-3) | 2 (2-3) | 1/5 · 0/5 | ~0 | ~0 | 5/5 | 0.326 |
| 40      | 2 (2-3) | 2 (2-3) | 3/5 · 5/5 | ~0 | ~0 | 5/5 | 0.128 |
| 60      | 4 (3-5) | 4 (3-5) | 4/5 · 5/5 | ~0 | ~0 | 5/5 | 0.066 |
| 100     | 5 (4-6) | 5 (3-5) | 4/5 · 5/5 | 1.18e-03 | ~0 | 5/5 | 0.047 |

**Two ranks, deliberately.** `front-max` = max rank over *every* Pareto equation; it is biased
upward because the front grows with maxsize (14.6 -> 48.2 rows scanned), so a max over it can climb
from extra draws alone. `selected` = rank of the loss-minimizing equation, the one a forecast would
deploy. They agree here (front-max exceeds selected by +1 in only 3/20 runs, never more), so the
budget trend is not an artifact of the biased statistic. **Quote `selected` in the paper.**

## Verdict (pre-registered)
- STRUCTURAL criterion (rank stays <=3 once front detaches + all 6 inputs): **FAILS** — rank climbs 2->2->4->5.
- BUDGET criterion (rank climbs toward 6 with maxsize; cond finite-ing; off-fid -> combine): **largely MET**. The original maxsize=25 rank-2 result was search-starvation (z_Hi dropped; fronts pinned at the cap).
- Residual: even at maxsize=100 the median fit resolves only ~5/6 directions, and the sloppiest He II direction never lifts off machine-zero (λ6 ~ 2e-15).
- Clean survivor: the per-parameter combine is rank-6 by construction at ~1/6 the *complexity* budget.

## Corrections (2026-07-08 six-agent cross-check; two teams derived independently and agree)

1. **A `#singular` column reading `5/5, 5/5, 5/5, 3/5` was hand-typed into an earlier version of this
   file and is not reproducible.** `rank_deficient_vs_nparams` is True for **5/5 seeds at every
   budget**, including maxsize=100. No metric in these JSONs yields 3/5. The `rank-deficient` column
   above is now derived by the script.
2. **No *selected* fit reaches rank 6 in any of the 20 runs** (max 5). The "one seed hits 6" is
   `ms100_seed1`'s **front-scan** value at complexity 93, a non-selected equation; that seed's
   selected fit is rank 5 and is flagged rank-deficient. Do not conflate the two.
3. **λ5 = 1.18e-3 is an absolute whitened eigenvalue of the marginally-*recovered* 5th direction.**
   It is NOT "~1e-3 of the Fisher information", and it does not describe the *unrecovered* sloppiest
   direction. As an information fraction the sloppiest GP direction is 1.4e-4 (whitened) / 1.2e-5
   (physical). An earlier draft of §5.1 made both errors in one clause.

## Known confound: niter is not held fixed at the top budget

`slurm/joint_multid_sweep.slurm` pre-registers "vary ONLY maxsize ... niter=400 fixed" and defaults
`NITER=400`, but `submit_joint_sweep.sh` takes `NITER` from the environment. What actually ran:

- `53055690_15..19` (ms100, **niter=400**): **TIMEOUT at 04:00:14**, wrote no JSON.
- `53071661_15..19` (ms100, **niter=250**): COMPLETED (2h57m–5h41m). **These are the ms100 JSONs on disk.**

So maxsize and niter co-vary exactly at the top budget. The only clean maxsize step that shows a
rank climb is **40 -> 60** (median 2 -> 4). The residual singularity at ms100 may be iteration-limited
rather than a genuine budget ceiling. The paper (§5.1) now discloses this.

**Provenance gap:** the JSONs record no `niter` / `n_train` / `seed` field. The only record of which
iteration count produced which result is the untracked `slurm-joint_ms_sweep-<jobid>_<task>.out` logs.
Recording those in the diagnostic JSON would prevent a recurrence.

Reproduce: `scripts/submit_joint_sweep.sh` (see `slurm/joint_multid_sweep.slurm`).
Note `results/*` is gitignored (`.gitignore:247`) — only this summary is committed, so the per-seed
JSONs backing it are not independently checkable from a clean checkout.
