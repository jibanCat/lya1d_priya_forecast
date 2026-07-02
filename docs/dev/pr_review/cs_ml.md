# PR #6 review — cs_ml lens (code quality / correctness / reproducibility)

Branch `stage10-multiz-sobolev` → `main`. Reviewed the diagnostic core:
`src/priya_forecast/{grad_faith_io,pareto_diag}.py`,
`scripts/{eval_grad_faithfulness,plot_pareto_faithfulness,make_grad_faith_sidecars,make_diagnostic_figs,h_basis_test}.py`,
`tests/test_{grad_faith_io,pareto_diag}.py`, `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`,
plus `sobolev_loss.py` / `derivative_gate.py` and the committed sidecars/figures.

## (1) Verdict: APPROVE-WITH-NITS

The diagnostic modules are clean, genuinely emulator-free where claimed, and tested.
`grad_err`/`value_mse` metric spaces (linear-P vs log-P) are correctly implemented and
correctly labelled (verified against `refit.predict` returning raw `P_F` and `value_mse`
using `np.log`/`predict_log`). The eval metric provably reproduces the production gate
(same `gp_param_gradient`/`equation_param_gradient` stencil, identical `floor_frac=1e-3`
masking). `make_diagnostic_figs.py` reproduces 3 figures from committed sidecars with no
emulator. Nothing blocks merge, but there is one real reproducibility gap (orphan figures)
and a couple of dead-code/fragility nits.

## (2) What is correct and well-built

- **Emulator-free claims hold.** `grad_faith_io` imports only `re`/`pathlib`/`pandas`;
  `pareto_diag` adds only matplotlib/numpy. Both import with `student_projects` stripped
  from `sys.path`. `make_diagnostic_figs.py` and `plot_pareto_faithfulness.py` have zero
  GP/GPy/lyaemu imports. The "plotter consumes sidecars without GPy" contract is real.
- **Metric correctness (the headline fix is right).** `grad_err` is a ratio of *linear-P*
  slopes: `eval_grad_faithfulness.py` reuses `equation_param_gradient`/`gp_param_gradient`,
  both of which call `.predict` → `refit_1d_pysr.py:275` returns `np.exp(val)` (raw `P_F`),
  and the GP predict returns linear P. `value_mse` (eval lines 101/113-118) uses
  `np.log(...)`/`predict_log` → log-P. The walkthrough's "grad_err is linear `∂P_F`,
  value_mse is log-P" labelling matches the code exactly.
- **The gate-equivalence claim is verifiable, not asserted.** `median_rel_error`
  (eval:38-48) is a byte-for-byte mask/median twin of `derivative_gate.derivative_faithful`
  (gate:42-59): same `amax==0 → inf/False`, same `keep = |target| >= 1e-3*amax`, same
  `median(|cand/target-1|)`. Same h=1e-3 central stencil. So "the same metric the
  production gate uses" is true.
- **Tests are real and pass.** `pytest tests/test_grad_faith_io.py tests/test_pareto_diag.py`
  → 5 passed. They cover the load-bearing edge cases: x0 word-boundary (`x01` not matched),
  comment-header skip on read, `gate_pass` round-trips as real `bool` dtype, gray-fallback
  (no sidecar → all-NaN), and the **left-join NaN** (a complexity with no sidecar row →
  `grad_err` NaN, drawn gray). I independently exercised a mixed seen/unseen front through
  `render_grid` — no crash, unseen rows go gray.
- **Sidecar provenance + completeness.** Every committed sidecar carries a
  `# param=… z=… tol=… log_space=… source=…` header and the full `value_mse` column
  (checked all 22 z=3.6 + 44 cross-z files). `make_diagnostic_figs.py --out-dir /tmp/...`
  reproduces `pareto_faithfulness`, `faithfulness_scorecard`, `ns_budget_panel` (PNG+PDF)
  from those committed CSVs, exit 0, emulator-free.
- **Sobolev wiring is guarded.** `refit_1d_multiz_for_param` raises if `use_sobolev=True`
  without both GPs; the Sobolev `loss_function` deliberately overrides any
  pysr_kwargs-supplied loss and drops `elementwise_loss` (commented). The shell driver's
  hardcoded param list matches `PARAM_NAMES` exactly (sorted-equal).

## (3) Concrete issues (file:line + fix)

1. **[reproducibility — main nit] Three walkthrough figures have NO committed generator.**
   `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` embeds `summary_scorecard.png` (L81),
   `ns_money_panel.png` (L86), and `crossz_faithfulness.png` (L230). All three are committed
   binaries in the PR, but `grep -rn` across `scripts/`/`src/` finds **no code that writes
   those filenames**. `make_diagnostic_figs.py` writes the *differently named*
   `faithfulness_scorecard.png` and `ns_budget_panel.png` (and never the cross-z panel). So
   the four figures a reader sees in the paper's central document are not regenerable from
   this branch — they are orphan artifacts.
   *Fix:* either (a) commit the generators (a cross-z fig script + rename
   `faithfulness_scorecard`→`summary_scorecard`, `ns_budget_panel`→`ns_money_panel` so the
   script's outputs match the embedded names), or (b) repoint the walkthrough's
   `![]()` links to the filenames `make_diagnostic_figs.py` actually emits. Right now the
   doc and the only committed figure script disagree on every filename.

2. **[dead code] Duplicate `return out_path`.** `src/priya_forecast/pareto_diag.py:121-122`
   has two consecutive `return out_path`; line 122 is unreachable. Delete line 122.

3. **[dead code] Unused `bf`.** `scripts/make_diagnostic_figs.py:105`
   `bf = bestloss(SOBOLEV, p)` is assigned and never used (comment even says "same here").
   Delete it (it also does an extra sidecar read per resister for nothing).

4. **[fragility — silent-misalignment risk] eval mixes two k-grids.**
   `eval_grad_faithfulness.py:77` computes `target` on `k_grid = kodiaq_k_grid(kmin,kmax,48)`
   while `g` (line 110-111) is computed on `kg = d["kfkms_lf_z"][0]`. `median_rel_error`
   divides them elementwise, which is only valid if the two grids are identical. They
   *are* today (I verified `max|kg - k_grid| == 0.0` for the default 0.001/0.04/48), but
   that is a coincidence of the default args matching the data grid — change `--kmin/--kmax`
   (or run on data with a different LF grid) and the ratio silently misaligns instead of
   erroring. *Fix:* compute `target` on `kg` (the same grid the candidate uses), or assert
   `np.allclose(k_grid, kg)` before the loop.

5. **[minor] Joined `gate_pass` column is unused by the plotter.** `load_front` joins
   `gate_pass` (pareto_diag.py:34) but `render_grid` derives pass/fail purely from
   `grad_err <= gate_tol` (lines 78/85). The column is dead weight in the plotting path and
   a second source of truth that could drift from `grad_err`. Either drop it from the join
   or use it. (Harmless today; the sidecar `gate_pass` and the derived comparison agree.)

6. **[doc nit, non-blocking] hardcoded `/home/mfho/student_projects/...`** appears in
   `eval_grad_faithfulness.py:14`, `h_basis_test.py:14`, `make_grad_faith_sidecars.sh:13`.
   In the .py files it's docstring-only (fine); in the shell script it's a real
   `export PYTHONPATH`. Matches the documented HPC layout, but it makes the sidecar driver
   non-portable. Consider `${LYA_EMU_PATH:-/home/mfho/...}`.

## (4) Anything that would block merge

Nothing hard-blocks. The only thing approaching a blocker is **issue #1** (orphan figures):
the walkthrough is described as the paper's central instrument, yet 3 of its 4 figures
can't be reproduced from committed code, which undercuts the "reproducible from committed
sidecars" property the rest of the PR earns. I'd gate final approval on either committing
the missing generators or fixing the doc's image paths — but it does not require touching
the (correct, tested) diagnostic modules. Issues #2–#3 are trivial deletions; #4 is a
one-line guard worth adding before anyone reruns the sidecars with non-default k-range.
