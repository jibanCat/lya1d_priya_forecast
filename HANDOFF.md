# Handoff for the next Claude session

This is the project I (the previous Claude) was building with the user.
Read the user's `CLAUDE_CODE_INSTRUCTIONS.md` (under `/home/mfho/student_projects/`)
for the original spec, then this doc for current state.

---

## Where things are

### Branches

- **`main`** — Phases 0-7 complete. 171 tests pass. Pushed to origin at
  `https://github.com/jibanCat/lya1d_priya_forecast`. Last commit:
  `Phase 7: unified priya-forecast CLI` (d9d8788).

- **`fine-tune-pysr`** — current working branch. Adds the PySR vs GP
  hypothesis investigation: 186 tests, three new HPO metrics
  (`val_mse`, `fisher_agreement`, `sigma_targeted`), a head-to-head
  experiment on the real GP at z=3.6 for the `ns` parameter, and a
  full analysis doc (`docs/PYSR_HYPOTHESIS.md`). Pushed.

### Running jobs

A big-budget σ-targeted HPO sweep was running at session-end:

```
PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \
PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \
    python scripts/run_pysr_hpo.py \
        --param ns --n-train 96 --n-val 256 \
        --space configs/hpo/big_budget.yaml \
        --strategy random --n-trials 8 \
        --metric sigma_targeted --sigma-targeted \
        --output results/pysr_hypothesis/refit_ns_bigbudget
```

Background task IDs: `bq986qky7` (the active one). Output streams to
`/tmp/claude-114399728/-home-mfho-lya1d-priya-forecast/ab7a3a0c-1f74-4732-abcb-458a8274033c/tasks/bq986qky7.output`.

Check with:
```
tail -30 /tmp/claude-114399728/-home-mfho-lya1d-priya-forecast/ab7a3a0c-*/tasks/bq986qky7.output
ps aux | grep run_pysr_hpo
ls results/pysr_hypothesis/refit_ns_bigbudget/
```

The **purpose of this run**: validate the analysis doc's claim that
`(maxsize ≥ 30, niter ≥ 200, metric=sigma_targeted)` is the
combination needed to actually close σ_pysr ≈ σ_GP. Pre-run, all three
prior HPO metrics gave σ_ratio ≤ 0.117 (8.5×+ too tight). This run
should produce σ_ratio close to 1 — or, if it doesn't, the doc's
recommendation needs revision.

### When the run finishes

1. Check `results/pysr_hypothesis/refit_ns_bigbudget/scorecard.md`
   (or `hpo_top10.md`) for the σ_ratio column on the top result.
2. Run `scripts/compare_pysr_winners.py` to update the head-to-head
   figure (it'll auto-pick up the new cache dir if you copy or
   symlink it to the same `refit_ns_sigma` name, or update the
   script's hardcoded paths).
3. Update `docs/PYSR_HYPOTHESIS.md` Q4 section with the result.
4. Commit + push.

If σ_ratio is still ≪ 1 even at big-budget: the analysis is wrong
about which knob matters. Likely culprits in order: (1) PySR can't
find the right operator combination at all (try adding `inv`, `sqrt`,
`pow`); (2) sigma_evaluator is computing σ_pysr from a 1D forecast
that's structurally different from what the equation was trained for
(e.g. sign mismatch); (3) the GP itself has features that are
fundamentally non-symbolic (rare for cosmology emulators).

---

## Phase status (in build order)

| Phase | Description | Status | Notes |
|---|---|---|---|
| 0 | scaffold repo layout | ✅ done | pyproject, configs, stubs |
| 1 | parameters + config (YAML) | ✅ done | 11 PRIYA params, dataclass-based |
| 2 | eBOSS DR14 data loader | ✅ done | vendored from sbird/lya_emulator |
| 3 | P1D models (PySR + GP) | ✅ done | sympy whitelist, GP adapter |
| 4 | likelihood + Fisher + MCMC | ✅ done | Cholesky, dim-less internal Fisher |
| 5 | multi-D PySR diagnostic | ✅ done | coupling matrix headline plot |
| 6 | reusable PySR HPO | ✅ done | now 5 metrics: val_mse, complexity_at_target, pareto_area, fisher_agreement, sigma_targeted |
| 7 | unified CLI + README polish | ✅ partial | priya-forecast CLI works; README still terse |
| 9 | PySR hypothesis investigation | ✅ on branch | docs/PYSR_HYPOTHESIS.md, 11 new tests |

The Phase 7 README is functional but minimal. Worth a polish pass.

---

## Scientific headline findings (the user's actual paper output)

### 1. Coupling matrix (Phase 5, real GP, 11 params, z=3.6)

`results/coupling_matrix/diag3_coupling_matrix.png` and
`results/coupling_matrix_6params/`. **Only one parameter pair has
positive coupling**: `herei × alphaq` (+0.45). All other pairs are
empirically separable at this training budget. The student's
1D-factorization assumption is justified for 54/55 pairs.

`docs/FIGURES.md` § "Multi-D PySR diagnostic" has the full reading.

### 2. Why published PySR equations underperform the GP

(`docs/PYSR_HYPOTHESIS.md`)

The published equations (`mf_*.py` outputs at maxsize=20, niter=20)
give σ_pysr/σ_GP ≈ 2.4-8.4× across 4 forecast params. The
**smoking-gun figure** is `docs/figures/pysr_hypothesis/
fig_published_diagnosis.png` — the published `ns` equation
`((ns·k) - r) · 2.40` has 1/3 of the GP's slope across the prior.

Five hypotheses tested:

- **H1 (loss function)** — partly. Smooth fits don't suffer; real PySR
  with non-smooth wiggles does.
- **H2 (parsimony)** — partly. Mild parsimony is harmless; aggressive
  pruning silently drops weakly-coupled parameters (the published
  `alphaq` equation has no `alphaq` symbol because of this).
- **H3 (output normalization)** — **MASSIVE**. Training on
  `flux_norm = (P_F - mean_k)/std_k` vs raw `P_F` is **28 orders of
  magnitude** difference in test MSE. The student's pipeline does
  this; my framework supports it via `mode: "auto"` or `"files"`.
  **Never use `mode: "identity"` unless your equation outputs
  physical units.**
- **H4 (operator set)** — **MASSIVE for Lyα**. Polynomial-only
  basis vs polynomial + `exp(-c·k)` basis: 22 orders of magnitude.
  Student's PySR includes exp/log; not their problem.
- **H6 (covariance combine)** — modest gain. Adding explicit cross-terms
  for the one coupled pair (`herei × alphaq`) gives ~6% MSE improvement.

### 3. Three HPO metrics on the real GP (Q4 head-to-head)

| Metric | val_mse | σ_ratio | Visual |
|---|---|---|---|
| `val_mse` | 0.636 | 0.08× (12× too tight) | tilted line |
| `fisher_agreement` | 0.882 | 0.02× (50× too tight) | steep line |
| **`sigma_targeted`** | 2.01 | **0.117× (8.5× too tight)** | **shallowest line** |
| GP target | — | 1.00× | quadratic curvature |

`docs/figures/pysr_hypothesis/fig_three_metric_comparison.png`.

**Headline**: at maxsize=15-20 (quick.yaml budget), PySR converges to
**locally-linear approximations regardless of HPO metric**. The
`sigma_targeted` metric correctly picks the closest-to-GP one (even
at the expense of val_mse), but **none of the three reach σ_ratio ≈ 1**
because PySR can't reproduce the GP's quadratic curvature near fid at
small complexity caps.

**Need all three together**: maxsize ≥ 30, niter ≥ 200,
`metric="sigma_targeted"`. **The big-budget run currently in flight
is testing exactly this claim.**

### 4. The published-equations alphaq bug

Locked in as `tests/test_student_equations.py::
test_alphaq_equation_has_no_alphaq_dependence`. The published
"alphaq" equation `cos(r + 0.7158 - 1.535·k)⁴/0.476 - r - 1.047`
contains no `alphaq` symbol. The forecast catches this (σ_alphaq
blows up by 5×10¹¹). When the upstream LaTeX is fixed, invert that
test.

---

## Outstanding work / TODO ordered by importance

1. **WAIT FOR + ANALYZE the big-budget run** (in flight). Update
   PYSR_HYPOTHESIS.md Q4 with the result. If σ_ratio approaches 1,
   commit + close out.
2. **Implement the hybrid combine** (Q6 from the user's questions): a
   `combine: hybrid` mode that does multiplicative for most params
   and explicit joint cross-term for `herei × alphaq` only. Phase 5's
   coupling matrix tells you which pairs need this.
3. **Refit all 4 forecast params** (dtau0, Ap, ns, alphaq) at
   big-budget + sigma_targeted; rerun `train_and_forecast.py
   --equations <new_yaml>` to score the joint forecast. Currently
   only `ns` has been refit.
4. **Phase 5's full PySR backend run on the coupling matrix** — the
   current heatmap was produced by the polynomial surrogate. Replace
   with `pysr_kwargs={...}` for paper-quality coupling numbers.
   (Slow: 55 pairs × ~5 min each = ~5 hours.)
5. **README polish** — currently terse. Should land
   above-the-fold-find-anything-quickly content.
6. **Multi-z extension** — the paper benchmark is z=3.6 but
   `configs/diagnostic.yaml` already lists 2.6→4.2. The
   `run_coupling_matrix.py` script can iterate over z; the
   `run_multid_pysr.py` and `train_and_forecast.py` could too.
7. **Optional: install `optuna`** to enable Bayesian HPO. Currently
   it falls back to random with a warning.
8. **Merge fine-tune-pysr to main** when the big-budget run validates.
   PR is at https://github.com/jibanCat/lya1d_priya_forecast/pull/new/fine-tune-pysr.

---

## Recurring environment / setup notes

The environment is U-Mich Greatlakes (or similar HPC shared filesystem):

- Python: `/sw/pkgs/arc/mamba/py3.11/bin/python` (read-only)
- Test deps installed `--user`: pytest, hypothesis, pandas, h5py, emcee,
  getdist, GPy, emukit, pysr.
- Julia/PySR live under `$HOME/.julia_env` and `$HOME/.julia` because
  `/sw/pkgs/arc/mamba/py3.11/julia_env` is read-only. Always set
  `PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env` and
  `JULIA_DEPOT_PATH=$HOME/.julia` for any PySR run.
- Real GP emulator: `lyaemu` package at
  `/home/mfho/student_projects/lya_emulator_full/`. Add to PYTHONPATH:
  `PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src`.
- `PRIYAEmulatorExplorer` has an upstream bug (`KSData.__init__`
  read-only-pf). Bypass by going to `GPWrap` directly — `GPModel` does
  this already.
- The resolution-correction interpolant in upstream `lyaemu` only spans
  k ≥ 0.003 s/km, so we set `use_res_corr=False` in `GPWrap`.

## Test runner

```
PYTHONPATH=src python -m pytest tests/ -q
```

Should report 186 passing, 1 lyaemu-gated skip on the `fine-tune-pysr`
branch (171 + 1 skip on `main`).

## Sanity-check commands

```
# 1D forecast scoring on the published equations
PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \
    python scripts/train_and_forecast.py \
        --params dtau0 Ap ns alphaq --equations published \
        --output results/published_scorecard

# Coupling matrix on a small subset
PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \
    python scripts/run_coupling_matrix.py \
        --params ns Ap hub omegamh2 herei alphaq \
        --order 4 --n-train 64 --n-test 128 \
        --output results/coupling_matrix_6params

# σ-targeted HPO on one parameter
PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \
PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \
    python scripts/run_pysr_hpo.py \
        --param ns --n-train 64 --n-val 256 \
        --space configs/hpo/quick.yaml \
        --strategy random --n-trials 4 \
        --metric sigma_targeted --sigma-targeted \
        --output results/hpo_demo
```

## Where to look when confused

- `docs/ONBOARDING.md` — student-facing walkthrough.
- `docs/FIGURES.md` — every diagnostic figure explained.
- `docs/PYSR_HYPOTHESIS.md` — the hypothesis investigation (this
  branch). Read top to bottom — the conclusions matter.
- `tests/test_*.py` — every claim made anywhere in the codebase is
  asserted somewhere here.

Good luck. The user is responsive and willing to install deps if
asked. They're a senior researcher; the student they're managing is
the audience for ONBOARDING.md.
