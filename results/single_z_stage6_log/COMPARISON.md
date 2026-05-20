# Stage 6 — linear vs log(P) PySR target at z=3.6, KODIAQ-SQUAD

Comparison of the Stage 3 linear baseline (`results/single_z_stage3/z3.6/`)
against the Stage 6 log-space refit (this directory).

- **Refit config**: 11 params, z=3.6, KODIAQ-SQUAD, niter=50, smart_kwargs=true, seed=0 (+ seed-retry).
- **Forecast config**: `forecast_only` mode, `additive` combine, KODIAQ-SQUAD cov, mock_data=gp.
- **Only knob changed**: `target_space: linear` → `target_space: log`.

## σ_PySR / σ_GP per parameter

| param      | σ_GP   | linear σ_PySR | linear ratio | log σ_PySR | log ratio | improved? |
|------------|-------:|--------------:|-------------:|-----------:|----------:|:---------:|
| dtau0      |   1.19 |         32.34 |       27.09× |      24.99 |    20.93× | yes (closer to 1) |
| tau0       |   4.64 |         10.83 |        2.34× |       1.63 |     0.35× | no (now sub-1)    |
| ns         |   5.59 |          5.82 |        1.04× |       4.09 |     0.73× | no (was already ~1) |
| Ap         |  36.64 |          7.70 |        0.21× |      12.39 |     0.34× | yes (Mirage attenuated) |
| herei      |  26.68 |         30.41 |        1.14× |      32.94 |     1.23× | no (marginal) |
| heref      |  94.35 |          8.89 |    **0.09×** |      57.29 |     0.61× | **yes** (deep Mirage → mild) |
| alphaq     | 235.20 |         60.11 |        0.26× |     125.40 |     0.53× | yes |
| hub        |   1.46 |          0.24 |    **0.16×** |       0.99 |     0.68× | **yes** (deep Mirage → close to 1) |
| omegamh2   |   0.60 |          0.07 |    **0.12×** |       0.26 |     0.43× | **yes** (deep Mirage attenuated) |
| hireionz   |  86.35 |         56.81 |        0.66× |     105.00 |     1.22× | yes (flipped above 1) |
| bhfeedback |   4.74 |          0.96 |        0.20× |      13.16 |     2.77× | yes (now loose, not tight) |

## Summary statistics

|                                    | linear | log(P) |
|------------------------------------|-------:|-------:|
| mean \|log10(σ_PySR/σ_GP)\|        |  0.615 |  0.366 |
| median \|log10(σ_PySR/σ_GP)\|      |  0.678 |  0.273 |
| max \|log10(σ_PySR/σ_GP)\|         |  1.433 |  1.321 |
| # params with **deep** Mirage (< 0.2×) | 3      |  **0** |
| # params with any Mirage (< 1×)        | 7/11   |  7/11  |

## Interpretation

**The log(P) target dramatically attenuates the worst "Fisher's-Mirage" cases.**

- All three deep-Mirage params in linear mode (heref 0.094×, omegamh2 0.119×, hub 0.162×) move to ratios > 0.4. None of the 11 params stays below 0.2× in log mode.
- The median |log10| ratio is cut by a factor of ~2.5 (0.68 → 0.27).
- 8 of 11 params show ratios closer to 1 in log mode; the remaining 3 (tau0, ns, herei) move modestly in the wrong direction — they were already close to Fisher-faithful in linear mode (1.04× / 1.14× / 2.34×) and the log SR equations introduce small derivative artifacts there.
- Mirage count (sub-1 ratios) is unchanged at 7/11. The improvement is in **severity**, not headcount — exactly what the structural argument predicts (log target reduces fractional-error-driven derivative bias, doesn't eliminate finite-niter SR overfit).

**Bottom line:** log(P) is a partial-but-substantial fix. The structural direction is correct. Stage 8 (Sobolev derivative loss) is needed to close the gap further.

## Files

- `corner.png` — 11-param Fisher ellipses (GP / perfect_1D / PySR).
- `fisher_{GP,perfect_1D,PySR}.npz` — Fisher matrices.
- `forecast_table.txt` — same numbers in plain text.
- `scorecard.md` — single-z forecast summary.
- `refit/z3.6/pareto_*.csv` — 11 PySR Pareto fronts (raw refit output).

Linear baseline for reference: `results/single_z_stage3/z3.6/`.

## Reproduction

```bash
# Refit (11 SLURM tasks, ~5min wall each):
sbatch --export=ALL,REPO=$(pwd),BASEDIR=data/kodiaq_gp,\
       OUTPUT_DIR=results/single_z_stage6_log,Z=3.6,TARGET_SPACE=log \
       --array=0-10 slurm/single_z_refit.slurm

# Forecast (~4min, loads emulator):
python scripts/run_pipeline.py --config configs/single_z/stage6_log_z3.6.yaml
```
