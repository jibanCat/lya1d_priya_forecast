# lya_reviewer — scientific-accuracy review of the housecleaning output

Date: 2026-06-09 · Branch: `stage10-multiz-sobolev` · Reviewer focus: scientific
accuracy of the new `README.md` + the two new notebooks
(`tutorial_01_explore_diagnostic.ipynb`, `reproduce_paper_figures.ipynb`) against
`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` and the committed grad-faith sidecars.

## Verdict: **PASS** (with 2 minor, non-blocking fixes)

Every load-bearing scientific claim in the README and both notebooks is **correct and
matches both the walkthrough and the committed data**. The diagnostic, `grad_err`
metric, `value_mse`, the 0.25 gate, the four-category taxonomy, the ns Mirage, the
budget control, and the cross-z non-uniformity are all accurately described. No wrong
or overstated scientific claim was found. The two issues below are repro-hygiene /
wording, not science.

## What I verified (all green)

- **`grad_err` is linear-P, as both notebooks + README state.** Code path confirmed:
  `derivative_gate.gp_param_gradient` central-differences `gp.predict` (raw `P_F`) and
  `equation_param_gradient` central-differences `refit.predict` (raw `P`); the ratio is
  therefore `∂P_F/∂θ`, not `∂logP`. This matches the walkthrough's corrected
  "Metric space" note (lines 48–54) and the notebooks' `∂P_F/∂θ` / "linear P_F"
  framing. `value_mse` is correctly described as log-P (`mean (logP_eq−logP_GP)²`),
  matching `eval_grad_faithfulness.py:118`.
- **Main per-parameter table (walkthrough 109–119) reproduced exactly** from
  `single_z_stage6_log` (value@20) and `single_z_stage9` (Sobolev@20):
  dtau0 0.214→0.003, tau0 0.160→0.009, ns 0.603(bf .512, FAIL)→0.193(PASS),
  Ap 0.287(bf .108)→0.082, herei 0.251(bf .068)→0.060, heref 0.154→0.206,
  alphaq 0.152→0.173, hub 1.000→0.935(FAIL), omegamh2 0.320(bf .138)→0.198,
  hireionz 0.240→0.090, bhfeedback 1.715→0.946(FAIL). The notebooks' taxonomy
  (robust / selection-sensitive / generative-Mirage=ns / resistant=hub,bhfeedback)
  follows correctly from these.
- **ns budget control reproduced** from `decider_budget_z3.6`: best-loss complexity 35,
  grad_err 0.319 (FAIL), value_mse 3.8×10⁻⁴ (lowest of any series); Sobolev clears at
  0.193 / 4.74×10⁻⁴ (~24% higher value error). Matches README/notebook fig-3 text and
  walkthrough 121–135.
- **Fisher's-Mirage-in-one-table (tutorial cells 12/14) is real:** sorting the Sobolev
  ns front by `value_mse`, the two lowest-value-error candidates (cx13 4.07e-4 / cx12
  4.5e-4) FAIL the gate while cx18 (4.74e-4) passes — "several tiny-value_mse fail" is fair.
- **Cross-z table (walkthrough 234–246 / reproduce fig-4) reproduced exactly** from
  `single_z_z2.6_sobolev` / `stage9` / `single_z_z4.2_sobolev`, incl. the He II blowup
  (herei .186/.060/.709, heref .299/.206/2.69, alphaq .097/.173/1.56) and the stable
  resisters (hub .97/.94/1.22, bhfeedback .67/.95/.37).
- **Both notebooks execute cleanly, emulator-free** (`jupyter nbconvert --execute`,
  exit 0, no GPy/PySR/Julia import). reproduce fig-4 genuinely *generates* crossz from
  sidecars (not a stale-file assert). Tutorial is 32 cells; reproduce is 19 cells.
- **README install/packaging claims correct:** lockfile pins numpy 1.26.4, GPy 1.13.2,
  pysr 1.5.10, pandas 2.3.3; pyproject caps numpy<2 / pandas<3 with the accurate
  GPy-ABI 96→88-byte dtype rationale; extras forecast/pysr/gp/hpo/dev all present;
  `priya-forecast` entry point present; `README_old.md` backup present.
- **Test gate matches:** `PYTHONPATH=src pytest tests/ -q -k "not slow"` →
  **412 passed, 14 skipped**, exit 0.
- **`make_diagnostic_figs.py` runs** → writes exactly **3** figures; README's "three
  diagnostic figures" is correct.

## Issues (prioritised)

### P2 — reproduce notebook overwrites the committed paper figures in-place
`reproduce_paper_figures.ipynb` cell 4 sets `OUT = results/single_z_stage_pareto_diag`
(the committed figure dir under the protected `results/` tree). Executing the notebook
**dirtied 8 tracked files** (`git status --short` showed `M` on all
`pareto_faithfulness/faithfulness_scorecard/ns_budget_panel/crossz_faithfulness.{png,pdf}`).
Not a science error — figures regenerate equivalently — but a "reproducer" that mutates
tracked artifacts is a hygiene hazard for anyone running it. (I restored them with
`git checkout -- results/single_z_stage_pareto_diag/`.)
- **Fix:** point `OUT` at a scratch dir (e.g. `results/_repro_scratch` or a `tempfile`
  dir), mirroring how `tutorial_01` already writes to `results/_tutorial_scratch/`. One
  line. Cell to change: reproduce cell 4 (`OUT = Path("results/single_z_stage_pareto_diag")`).

### P3 — README test caveat says "error" where this env produces a "skip"
README §Usage: *"one pre-existing numpy<2/GPy environment error on
`test_real_gp_predicts_at_fiducial` is expected without the emulator."* In the project
`.venv` the run is clean — **412 passed, 14 skipped, no errors**; that GP test is gated
to a `SKIPPED` (lyaemu not importable), not an error.
- **Fix:** soften to "a handful skip (the GP/emulator-touching tests are gated off when
  `lyaemu` is absent); 0 errors expected." Optional; the "~412 pass, a handful skip"
  headline is already accurate.

## Nit (no action required)
- The cross-task summary line says reproduce "replicates the logic in
  `make_diagnostic_figs.py` to regenerate **all four** diagnostic figures." The script
  emits 3; the notebook adds crossz (fig 4) itself from the z2.6/z4.2 sidecars. The
  notebook's own cell-0 table is honest about this (lists 4, attributes crossz to its
  own sidecars), so it is correct as shipped — only the external summary phrasing is loose.
