# Sec. 5.1 joint six-parameter fit: complexity-limit sweep

**What this is.** One symbolic equation fitted jointly to six parameters and `k`
(`ns, Ap, herei, heref, alphaq, hireionz`), at z = 3.6, high fidelity, plain MSE loss, with
`n_train = 512`. Four complexity limits (`maxsize` 25, 40, 60, 100), five seeds each: 20 fits.
For each fit the diagnostic counts how many independent parameter directions its gradients
carry: the numerical rank of the prior-width-whitened Fisher matrix, counting eigenvalues above
`1e-8` of the largest. A full-rank fit has rank 6.

**Emulator.** The paper's (Sec. 2.1): `data/kodiaq_gp`, 172 k-bins. The emulator itself is
rank 6 (condition number 5731; smallest/largest eigenvalue 1.75e-4).

**Regenerate this table, do not hand-edit it:** `python scripts/aggregate_joint_sweep.py --write-summary`

| maxsize | front-max rank (med, range) | selected rank (med, range) | detached / all-6-inputs | median λ5 | median λ6 | rank-deficient | off-fid (med) |
|--------:|:--------------------------:|:--------------------------:|:-----------------------:|:---------:|:---------:|:--------------:|:-------------:|
| 25      | 3 (2-4) | 2 (2-3) | 1/5 · 0/5 | ~0 | ~0 | 5/5 | 0.292 |
| 40      | 3 (2-4) | 3 (2-4) | 2/5 · 5/5 | ~0 | ~0 | 5/5 | 0.086 |
| 60      | 4 (3-5) | 4 (3-5) | 5/5 · 5/5 | ~0 | ~0 | 5/5 | 0.064 |
| 100     | 5 (5-5) | 5 (4-5) | 5/5 · 5/5 | 1.07e-03 | ~0 | 5/5 | 0.049 |

Columns: `selected` is the rank of each fit's lowest-loss equation (the one a forecast would
use). `front-max` is the highest rank found anywhere on that fit's Pareto front. `detached`
counts fits whose lowest-loss equation is below the complexity limit. `all-6-inputs` counts fits
whose equation uses all six parameters. `off-fid` is the fit's mean-squared error against the
emulator away from the fiducial point.

## What the sweep shows

- The selected rank rises with the complexity limit: median 2, 3, 4, 5 at `maxsize` 25, 40, 60, 100.
- No selected fit reaches rank 6. All 20 fits are rank-deficient at the `1e-8` tolerance.
- At `maxsize = 25` every fit drops at least two of the six parameters; from 40 upward all six are used.

## Limits on what it shows

- **Two things change at the last step.** `maxsize` 25, 40 and 60 ran 400 iterations; `maxsize = 100`
  ran 250, because 400 iterations at that size overran the wall time in the first sweep. The step
  from 60 to 100 therefore changes the complexity limit and the iteration count together.
- **Five seeds.** Ranges overlap between neighbouring limits.
- **The count depends on the tolerance.** The medians are 2, 3, 4, 5 at `1e-6`, `1e-8` and `1e-10`
  alike, but at the emulator's own dynamic range (`1.75e-4`) they are 2, 3, 3, 3. One fit
  (`ms100_seed2`) is rank 6 at `1e-10`.

## Provenance

- Run 2026-08-10 on Great Lakes, code as committed at `0a2db58`, 4 CPUs per fit, serial PySR.
- Jobs: `56895343` (tasks 1-14, 400 iterations), `56897224` (task 0, 400 iterations; a resubmission
  after a file-lock failure), `56895344` (tasks 15-19, 250 iterations). All 20 completed.
- Elapsed per fit: 43-108 min at `maxsize = 25`, 61-124 min at 40, 1.9-3.7 h at 60, 2.4-4.9 h at 100.
- These 20 files predate the `run` block that `scripts/run_multid_pysr.py` now writes, so the
  emulator, iteration count and seed are recorded in the job logs and in this file, not in the JSON.
- An earlier sweep on the 102-bin default emulator (medians 2, 2, 4, 5) is superseded. It remains
  in the history at tag `v1.2-paper`. The run times and job history the manuscript quotes for the
  sweep (44-72 min, 1.9-3.4 h, 3.0-5.7 h; jobs `53055690`, `53071661`) are those of that earlier
  sweep, not of the files here.

Reproduce: `scripts/submit_joint_sweep.sh --dry-run` prints the two submissions; drop `--dry-run` to
submit them (`slurm/joint_multid_sweep_kodiaq.slurm`).
