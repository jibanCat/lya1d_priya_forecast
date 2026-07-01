# RUN MANIFEST — prod-20260630-perz-sobolev

- **git:** `fc914fc` @ `stage10-multiz-sobolev` (+uncommitted)
- **submitted:** 2026-06-30 13:48:14 EDT
- **account:** yueyingn0 (expires 2026-07-01)
- **recipe:** one PySR model per (param, z); Sobolev lambda=5, log target, no ANOVA.
- **budget:** maxsize=20, populations=48, niterations=200 (value baseline = plain MSE, same operators/budget).
- **grid:** z = 2.6 3.6 4.2; seed band seeds = 0 1 2 3 4 at z=3.6; ns budget control maxsize=35.
- **layout:** `sobolev/refit/z<z>/`, `value/refit/z<z>/`, `seed_band/z3.6_seed<S>_{value,sobolev,budget}/refit/z3.6/`.

## Submitted jobs
| job | id | array | output |
|---|---|---|---|
| sobolev z=2.6 | 52613569 | 0-10 | results/paper_production_20260630_perz_sobolev_z2.6-4.2/sobolev/refit/z2.6 |
|   sidecar sobolev z=2.6 | 52613570 | afterok:52613569 | grad_faith |
| value z=2.6 | 52613571 | 0-10 | results/paper_production_20260630_perz_sobolev_z2.6-4.2/value/refit/z2.6 |
|   sidecar value z=2.6 | 52613572 | afterok:52613571 | grad_faith |
| sobolev z=3.6 | 52613573 | 0-10 | results/paper_production_20260630_perz_sobolev_z2.6-4.2/sobolev/refit/z3.6 |
|   sidecar sobolev z=3.6 | 52613574 | afterok:52613573 | grad_faith |
| value z=3.6 | 52613575 | 0-10 | results/paper_production_20260630_perz_sobolev_z2.6-4.2/value/refit/z3.6 |
|   sidecar value z=3.6 | 52613576 | afterok:52613575 | grad_faith |
| sobolev z=4.2 | 52613577 | 0-10 | results/paper_production_20260630_perz_sobolev_z2.6-4.2/sobolev/refit/z4.2 |
|   sidecar sobolev z=4.2 | 52613588 | afterok:52613577 | grad_faith |
| value z=4.2 | 52613589 | 0-10 | results/paper_production_20260630_perz_sobolev_z2.6-4.2/value/refit/z4.2 |
|   sidecar value z=4.2 | 52613590 | afterok:52613589 | grad_faith |
| seedband sobolev seed=0 | 52613591 | 0-10 | seed_band/z3.6_seed0_sobolev |
| seedband value seed=0 | 52613592 | 0-10 | seed_band/z3.6_seed0_value |
| seedband ns-budget seed=0 | 52613593 | 2 | seed_band/z3.6_seed0_budget |
|   sidecar ns-budget seed=0 | 52613594 | afterok:52613593 | grad_faith_ns |
| seedband sobolev seed=1 | 52613595 | 0-10 | seed_band/z3.6_seed1_sobolev |
| seedband value seed=1 | 52613596 | 0-10 | seed_band/z3.6_seed1_value |
| seedband ns-budget seed=1 | 52613597 | 2 | seed_band/z3.6_seed1_budget |
| seedband sobolev seed=2 | 52613598 | 0-10 | seed_band/z3.6_seed2_sobolev |
| seedband value seed=2 | 52613599 | 0-10 | seed_band/z3.6_seed2_value |
| seedband ns-budget seed=2 | 52613600 | 2 | seed_band/z3.6_seed2_budget |
| seedband sobolev seed=3 | 52613601 | 0-10 | seed_band/z3.6_seed3_sobolev |
| seedband value seed=3 | 52613602 | 0-10 | seed_band/z3.6_seed3_value |
| seedband ns-budget seed=3 | 52613603 | 2 | seed_band/z3.6_seed3_budget |
| seedband sobolev seed=4 | 52613604 | 0-10 | seed_band/z3.6_seed4_sobolev |
| seedband value seed=4 | 52613605 | 0-10 | seed_band/z3.6_seed4_value |
| seedband ns-budget seed=4 | 52613606 | 2 | seed_band/z3.6_seed4_budget |

## Next (after all jobs finish)
- aggregate seed band: `scripts/aggregate_seed_band.py --band-dir results/paper_production_20260630_perz_sobolev_z2.6-4.2/seed_band --out results/paper_production_20260630_perz_sobolev_z2.6-4.2/seed_band/seed_band_summary.json`
- diagnostic figs (Phase C) read: value=`results/paper_production_20260630_perz_sobolev_z2.6-4.2/value/refit/z3.6`, sobolev=`results/paper_production_20260630_perz_sobolev_z2.6-4.2/sobolev/refit/z3.6`, budget=`results/paper_production_20260630_perz_sobolev_z2.6-4.2/seed_band/z3.6_seed0_budget/refit/z3.6`.

## Budget-control arm (maxsize=35, added 2026-06-30 per Phase-A review must-fix #2)
- bud35_val z3.6 all params: 52617264 | z2.6 ns: 52617265 | z4.2 ns: 52617266
- bud35_sob z3.6 hub+bhfeedback: 52617267
- purpose: generalize the ns budget control to all params (Mirage-not-budget) + show hub/bhfeedback resist even at maxsize=35+Sobolev. Headline run stays maxsize=20.

## z=3.6 sobolev re-run with artifacts (2026-06-30, for prediction figs Fig1/3/4)
- sob_z3.6_art: 52618022 (SAVE_ARTIFACTS=1 -> refits/ + payloads/ pkls); sidecar 52618023

## maxsize-sensitivity sweep (2026-06-30, Phase-C-rework: value-loss budget-dependence table)
- maxsize in {30,40} x {value,sobolev} all 11 params z=3.6 (combine with maxsize=20 main + maxsize=35 budget arm).
- sidecars scored later with the FIXED log-space gate.
