# Fresh-eyes referee report — senior_reproduce

**Role:** senior referee independently verifying the paper's headline claims from a fresh clone.
**Date:** 2026-06-09
**Branch under test:** `stage10-multiz-sobolev`

## Setup (what a real clone gives you)

```
rm -rf /tmp/fe2 && git clone -q /home/mfho/lya1d_priya_forecast /tmp/fe2 \
  && cd /tmp/fe2 && git checkout -q stage10-multiz-sobolev
```

All emulator-free commands run as:
```
PYTHONPATH=src /home/mfho/lya1d_priya_forecast/.venv/bin/python <cmd>
```
(`data/kodiaq_gp/` and the Julia/PySR emulator are gitignored / external, so a clone
has only tracked files. The clone's `data/` contains **only** `priya_fiducial/`.)

---

## VERDICT SUMMARY

| Headline claim | Doc source | Recomputed from clone | Match? |
|---|---|---|---|
| ns Mirage 0.60 → 0.19 | WALKTHROUGH table, HANDOFF | value best-loss **0.603**, Sobolev best-loss **0.193** | YES |
| budget@35 = 0.319 fails | WALKTHROUGH budget control | best-loss cx=35 Loss=0.442 grad_err **0.319**, any-pass **no** | YES |
| Four-way taxonomy | WALKTHROUGH §taxonomy | every best-loss/best-faith/any-pass value reproduces exactly | YES |
| Cross-z He II blow-up @ z=4.2 | WALKTHROUGH z-table | herei 0.060→**0.709**, heref 0.206→**2.690**, alphaq 0.173→**1.556** | YES |
| h basis test corr ≈ −0.25, ~6% var | WALKTHROUGH, HANDOFF | **NOT regenerable from a clone; result not committed** | UNVERIFIABLE |

**Bottom line:** every number that is derivable from the tracked sidecars/CSVs
reproduces *exactly* — all 11 params × {value,Sobolev} × {best-loss,best-faith,x0@},
the ns budget control, the ns value_mse decoupling (3.8e-4 / 4.7e-4 / ~24%), and the
full cross-z table. The figure reproducer is genuinely emulator-free and works as
documented. The **one load-bearing claim that cannot be reproduced or even
spot-checked from a clone is the h basis test** (it needs the GP, and no committed
artifact records its output). Separately, two doc/repo inconsistencies surfaced:
the README's "412 pass" test claim is false on a clone (1 test fails), and the
committed `crossz_faithfulness.png` is stale w.r.t. the current generator.

---

## WHAT WORKED (and was it discoverable?)

### W1. The emulator-free figure reproducer — works, fully discoverable
README §"Regenerate the diagnostic figures" and HANDOFF both give the exact command:
```
PYTHONPATH=src python scripts/make_diagnostic_figs.py --out-dir results/single_z_stage_pareto_diag
```
Ran it to a scratch dir (`--out-dir /tmp/fe2_scratch_figs`); exit 0, wrote 4 PNG+PDF.
All paths it reads (`single_z_stage6_log`, `single_z_stage9`, `decider_budget_z3.6`,
`single_z_z{2.6,4.2}_sobolev`) are tracked in the clone — confirmed by static path
extraction. No guessing required.

### W2. ns Mirage, budget control, taxonomy numbers — all reproduce exactly
Recomputing with the *same* definitions the figure code uses (`bestloss` =
sort sidecar by `Loss`, take row 0's `grad_err`; best-faith = `min(grad_err)`;
any-pass = `any(grad_err <= 0.25)`), every value in WALKTHROUGH's "The numbers
(z=3.6)" table and the taxonomy table matches to 3 d.p.:

- ns: value 0.603 / Sobolev 0.193 → matches "0.60 → 0.19".
- budget@35: cx=35, Loss=0.442, grad_err=0.319, no candidate on the 13→35 front
  passes the gate → matches "0.319 — fails" and "the Mirage is generative, not
  search-starvation."
- Taxonomy categories all hold: robustly-faithful {dtau0,tau0,heref,alphaq,hireionz}
  clear the gate on value best-loss; selection-sensitive {Ap,herei,omegamh2} fail
  best-loss but pass best-faith; ns fails value at every budget but Sobolev passes;
  {hub,bhfeedback} fail under both. Confirmed numerically.
- ns decoupling sub-claim: budget reaches the global-min value_mse **3.82e-4**
  (lower than value@20 1.78e-3 and Sobolev 4.07e-4), Sobolev best-loss value_mse =
  **4.74e-4** = **24.2% higher** → matches "~24% worse."

### W3. Cross-z He II blow-up — reproduces exactly
The full WALKTHROUGH z-table (Sobolev best-loss grad_err by z) matches the tracked
`single_z_z{2.6,4.2}_sobolev` sidecars to 3 d.p., including the headline z=4.2
blow-up of the He II block: herei 0.709, heref 2.690, alphaq 1.556 (all >> gate),
while ns/Ap/omegamh2/dtau0/tau0 and the two resisters stay stable. The claim "the
taxonomy is NOT redshift-uniform; the He II block is faithful at z≤3.6 and blows up
at z=4.2" is fully supported by committed data.

### W4. Committed figures are reproducible (3 of 4 pixel-identical)
Pixel-diffing the committed `results/single_z_stage_pareto_diag/*.png` against a
fresh regeneration: `pareto_faithfulness`, `faithfulness_scorecard`,
`ns_budget_panel` are **byte-for-byte identical** (mean abs pixel diff = 0.000).
(crossz is the exception — see MISLEADING M3.)

### W5. The paper-reproducer notebook runs emulator-free
`jupyter nbconvert --to notebook --execute notebooks/reproduce_paper_figures.ipynb`
→ exit 0. Discoverable from README §Notebooks.

---

## MISSING

### MISS-1 (load-bearing, top finding): the h basis test is not reproducible from a clone, and its result is not committed anywhere
The h refutation claim — "∂P/∂h is NOT a k-rescaling (corr ≈ −0.25, ~6% var) → h
resists because its response is weak/under-determined, not an AP basis wall"
(HANDOFF line 41; WALKTHROUGH lines 178–186) — is a **headline reversal of the
earlier "h = AP" hypothesis**, yet:
- `scripts/h_basis_test.py` requires the GP:
  `GPModel(basedir="data/kodiaq_gp", ...)` + `lya_emulator_full` on PYTHONPATH.
  In the clone it dies with
  `FileNotFoundError: GP emulator basedir does not exist: data/kodiaq_gp`.
- **No committed artifact records its output.** `git ls-files | grep -i basis` →
  only the script. There is no `.json`/`.csv`/`.npz`/`.md` with corr=−0.25 or the
  ~6% variance. (A grep for `-0.25` only hits unrelated resolution-correction
  matrices and equation constants.)

So a referee cannot verify, recompute, or even sanity-check the single number that
overturns the prior hypothesis. The docs do not flag this as a reproducibility gap;
they cite it as settled fact with a script reference, implying it is checkable.

**Fix (concrete):**
1. Commit the basis-test output. Add to `scripts/h_basis_test.py` a
   `--out <path.json>` that writes `{z: {corr, var_explained, n_k}}`, and commit
   `results/h_basis_test/h_basis.json` (it is ~3 floats × 3 redshifts — trivially
   small, like the other sidecars).
2. In WALKTHROUGH §4 (hub) and HANDOFF line 41, add: *"(GP-derived; reproduce via
   `scripts/h_basis_test.py` with the emulator + `data/kodiaq_gp`; cached result in
   `results/h_basis_test/h_basis.json` — not regenerable from a bare clone)."*

### MISS-2: no `data/kodiaq_gp` means the documented "sanity-check on an install" test command is not actually clean on a clone
See MISLEADING M1 for the failing test. The gap itself: the repo ships a config
(`configs/single_z/example.yaml`, `basedir: data/kodiaq_gp`) whose validator
requires a directory that no clone has, and a non-gated test exercises it. There is
a referenced prep script (`scripts/prep_kodiaq_gp.py`) but it needs a `--source`
the clone doesn't have either.

---

## MISLEADING

### MIS-1 (top finding): README/HANDOFF "412 pass, ~13 skip" is wrong on a clone — it's 411 pass / 1 FAIL / 14 skip
README lines 108–113 call the test suite "the quickest sanity check on an install"
and claim "(412 pass, ~13 skip ...)". HANDOFF line 69 repeats "412 pass, ~13 skip."
Actual fresh-clone result:
```
PYTHONPATH=src .venv/bin/python -m pytest tests/ -q -k "not slow"
→ 1 failed, 411 passed, 14 skipped
FAILED tests/test_single_z_pipeline.py::test_shipped_example_yaml_loads_and_validates
       ValueError: gp.basedir does not exist: data/kodiaq_gp.
```
The failing test is **not** emulator-gated (no skip marker); it calls
`load_config(configs/single_z/example.yaml)`, and `GPConfig.validate()`
(`src/priya_forecast/single_z/config.py:110-115`) hard-fails when `data/kodiaq_gp`
is absent. So a clean clone reproducibly fails it. The README's claim that the
suite is a clean emulator-free install check is false.

**Fix (concrete):** In `tests/test_single_z_pipeline.py:54`
(`test_shipped_example_yaml_loads_and_validates`), either (a) skip when
`data/kodiaq_gp` is absent (`pytest.mark.skipif(not Path("data/kodiaq_gp").exists())`),
matching the other GP-gated tests, or (b) better, decouple config-schema validation
from filesystem validation so the YAML round-trip can be checked without the data
(e.g. `load_config(..., validate_paths=False)`), since the test's docstring says it
exists so "students copy from it" — that is a schema check, not a data check. Then
update the count in README:108-113 and HANDOFF:69 to the true emulator-free figure
(411 pass / 14 skip once the test is fixed, or note the 1 expected failure).

### MIS-2: README/HANDOFF describe `test_real_gp_predicts_at_fiducial` as *the* environment-dependent test, but the actual clone failure is a *different* test
README:111-113 and HANDOFF:70-72 specifically call out
`test_real_gp_predicts_at_fiducial` as the one env-dependent test (skips w/o
emulator). That test does skip cleanly. But the test that actually **fails** on a
clone is `test_shipped_example_yaml_loads_and_validates`, which the docs never
mention. A referee following the docs would be blindsided by a red FAIL the docs
say shouldn't happen.

**Fix:** Fold this into the MIS-1 fix; once the YAML test is gated/decoupled, the
docs' "only env-dependent test" framing becomes true again.

### MIS-3: the committed `crossz_faithfulness.png` is stale — the documented reproducer overwrites it with a differently-sized figure
3 of 4 committed figures are pixel-identical to a fresh run, but
`crossz_faithfulness.png` is **shape 900×1650 committed vs 780×1350 fresh**. Git
shows why: the committed PNG was last written at `861bcda` (cross-z feature commit,
a standalone generator), while `make_diagnostic_figs.py` *gained* Figure 4 only at
`1449001` (the PR-#6 consolidation, figsize `(9, 5.2)` → 1350×780 @150dpi) and the
committed PNG was never regenerated. So the "single generator" the commit message
advertises does **not** reproduce the committed crossz figure — it replaces it. The
underlying numbers are unaffected (the z-table matches), but the committed artifact
and the documented command are out of sync.

**Fix:** Regenerate and recommit
`results/single_z_stage_pareto_diag/crossz_faithfulness.{png,pdf}` from the current
`make_diagnostic_figs.py` (run it once with the default `--out-dir`), so the
committed figure matches what the reproducer emits. Add a CI/Make check that
`make_diagnostic_figs.py` output matches the committed PNGs (the other 3 already do).

### MIS-4: `h_basis_test.py` swallows the missing-data error and exits 0
When run in a clone, `scripts/h_basis_test.py` prints a traceback but the process
**exits 0** (no `sys.exit(1)` / the exception is caught at top level by the runner
returning 0). A script that "fails" with exit 0 will pass naive automation and
silently produce nothing — dangerous for the load-bearing claim it backs.

**Fix:** Make `main()` exit non-zero on the missing-data path (or let the
`FileNotFoundError` propagate as the process's exit status), and have it write its
result JSON only on success (ties into MISS-1's `--out`).

### MIS-5 (minor): `configs/default.yaml` points at a non-existent absolute path, masked by the example config
`configs/default.yaml:14` sets
`gp_emulator_basedir: /home/mfho/student_projects/InferenceLyaData/Emulator_Files`,
a user-specific absolute path that won't exist for anyone else, while
`configs/single_z/example.yaml:40` uses the relative `data/kodiaq_gp`. The two
disagree and neither is present in a clone. Not load-bearing for the emulator-free
repro, but a trap for anyone trying the GP path. **Fix:** make both relative to a
documented `data/kodiaq_gp` and note in README §Prerequisites that the absolute
path in `default.yaml` is a local default to be overridden.

---

## Reproducibility-boundary note (where docs blur the clone line)
The README *does* mark the figure reproducer as emulator-free and lists GP
prerequisites separately — good. But it does **not** state that (a) the test suite
is not fully clean on a clone (MIS-1), (b) the h basis test result is GP-only and
uncommitted (MISS-1), or (c) `data/kodiaq_gp` is required even to load the shipped
example config. A one-paragraph "What a bare clone can and cannot reproduce" box in
README would close all three.
