# Per-parameter Pareto-faithfulness diagnostic (failure-modes figure)

**Date:** 2026-06-08
**Branch:** `stage10-multiz-sobolev`
**Status:** design approved (user reviews the spec + the walkthrough doc as it builds).

## 1. Motivation — the paper pivots to diagnostic / failure-modes

The arc Stages 6–10 asked *"can a per-parameter 1D SR 'emulator' replace the GP
for Fisher forecasting?"* The standing answer is **no, not cleanly**: 9/11
params can be made derivative-faithful with the full recipe (log target +
multi-z + Sobolev loss), but **hub and bhfeedback resist every method**, and the
4-agent review (2026-06-05, `memory/review_verdict_sr_emulator.md`) showed the
"9/11 faithful" headline is partly GP-contaminated (GP-slice fallback prints
GP-derived σ in the σ_PySR column) and that the "cannot replace" verdict is
**confounded with search budget** (the ladder ran maxsize=20/niter=50; the
project's own `docs/PYSR_HYPOTHESIS.md` shows curvature needs
maxsize≥30/niter≥200).

So we redirect the paper to its honest, defensible core: a **diagnostic /
failure-modes** contribution — *where, why, and how badly per-parameter SR fails
as a Fisher emulator, and what (Sobolev loss) does and does not fix.* The
central instrument is a **per-parameter Pareto-faithfulness figure** modeled on
the syren Pareto-front plots (e.g. arXiv:2506.08783 Fig. A1), extended with the
quantity the syren family never reports: **derivative faithfulness**.

This figure is honest by construction: it plots **raw per-candidate gradient
error with no GP-slice fallback**, so a parameter with no faithful equation
simply shows up all-red. It also defuses the budget confound by overlaying a
**certified-budget value-loss front**, making "even at high complexity the
derivative stays wrong" a visible, referee-proof claim rather than an assertion.

## 2. Scope

In scope:
- A pure plotting script producing the 11-panel per-parameter figure.
- An upgrade to `scripts/eval_grad_faithfulness.py` to emit machine-readable
  per-candidate sidecars the plotter consumes.
- A small driver to produce sidecars for all 11 params × the available fronts.
- The certified-budget control: capture the gradient verdict on the existing
  `results/decider_budget_z3.6` ns front; (optionally) extend to hub/bhfeedback.
- A **living walkthrough doc** with inline figures + per-parameter "why it
  fails" reasoning, for the user to review progress.

Out of scope (this deliverable): any change to the refit/Fisher pipeline; the
multi-z Stage 10 "money plot" (Task 4 stays paused — we are *not* pursuing the
σ_PySR/σ_GP forecast claim in this redirect); new science runs beyond the
budget-control refits.

First cut is **single-z z=3.6, all 11 params**, from Pareto CSVs already on disk
(`results/single_z_stage6_log/refit/z3.6/pareto_*.csv` = value-loss@20;
`results/single_z_stage9/refit/z3.6/pareto_*.csv` = Sobolev@20). No cluster is
needed for the value/complexity axes.

## 3. The figure (locked design)

11 panels (a 4×3 grid with one blank, or 3×4), one per PRIYA parameter, shared
conventions:

- **x** = `Complexity` (PySR node count).
- **y** = `Loss` (PySR value loss), **log scale**.
- **one marker per Pareto-optimal equation** at each complexity.
- **marker color** = `grad_err` = `median_k |∂eq/∂θ ÷ ∂P_GP/∂θ − 1|` evaluated
  at the fiducial parameter vector over non-negligible k-bins (the production
  derivative-gate metric; see §5). Colormap thresholded at the **0.25 gate**:
  green ≤ 0.25 (derivative-faithful) → red ≫ 0.25 (Mirage).
- **series** (distinguished by marker shape/edge, shared colormap for fill):
  1. value-loss @ budget-20 (stage6_log),
  2. **Sobolev @ budget-20 (stage9)**,
  3. value-loss @ **certified budget** (control) — at least ns; others if run.
- **gate line / band** at `grad_err = 0.25` indicated on the colorbar.
- Per-panel annotation: the **lowest complexity at which the parameter's own
  feature (`x0`) first enters** a Pareto equation (surfaces the hub
  "x0 enters only at complexity 6" under-search signal from review #6).

The story each panel tells, read together across the grid:
- **Easy** — loss falls and markers go green at low complexity.
- **Mirage** (ns) — loss is low but value-front markers are red; the Sobolev
  series turns them green → Mirage and its cure in one panel.
- **Resistant** (hub, bhfeedback) — markers stay red at every complexity, under
  every series, including certified budget → the genuine failure mode.

## 4. Architecture — three components, deliberately split

The emulator-dependent computation is isolated from the layout iteration.

### 4.1 `scripts/eval_grad_faithfulness.py` (upgrade)

Currently prints a per-candidate table. Add:
- `--out PATH` → write a sidecar CSV with one row per Fisher-safe candidate:
  `Complexity, Loss, grad_err, n_keep, gate_pass, x0_enters` (and a header
  comment recording `param, z, tol, log_space, source_pareto`).
- Keep the existing stdout table (unchanged behavior when `--out` omitted).
- `x0_enters` is the boolean "does this candidate's equation contain `x0`"
  (the parameter feature), enabling the §3 annotation.
- No change to the metric math (`median_rel_error`, `gp_param_gradient`,
  `equation_param_gradient`) — reuse as-is so the sidecar equals the gate.

*Needs the emulator → runs on the cluster / a GP-capable node.*

### 4.2 `scripts/plot_pareto_faithfulness.py` (new, pure)

- Inputs: a results dir (or explicit lists) of Pareto CSVs + matching
  `grad_faith_*.csv` sidecars per series.
- Reads `Complexity, Loss` from the Pareto CSV; joins `grad_err`/`gate_pass`
  from the sidecar on `Complexity` (+`Loss` tiebreak).
- Renders the 11-panel figure to PNG (+ optionally PDF for the paper).
- **Graceful degradation:** if a sidecar is missing, plot that series in gray
  (value-only) so the **layout can be built and reviewed before the cluster job
  lands**. A small legend notes "gray = derivative not yet evaluated."
- **No emulator import** → iterate layout instantly, anywhere.

### 4.3 Sidecar driver + budget-control

- `scripts/make_grad_faith_sidecars.sh` (or a thin python loop): for each of the
  11 params, run §4.1 against the stage6 value front and the stage9 Sobolev
  front, writing sidecars next to each Pareto CSV. Emulator loads once per
  invocation; params loop cheaply. Can be a `--array=0-10` SLURM job mirroring
  `slurm/single_z_refit.slurm`'s env contract, or a serial login-node loop
  (it is read-only on the GP, no PySR/Julia).
- **Budget-control verdict:** run §4.1 on
  `results/decider_budget_z3.6/refit/z3.6/pareto_ns.csv` → capture the grad_err
  of the certified-budget (complexity-35) value-loss equation. Record whether it
  clears the 0.25 gate. Overlay as series 3 on the ns panel.
- **Optional extension** (decide after seeing the Phase 2 figure): certified-
  budget refits for the resisters hub + bhfeedback, to assert "even at certified
  budget, every complexity stays unfaithful" for the two named exceptions.

### 4.4 Walkthrough doc (living deliverable)

`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` — the artifact the user reviews:
- Embeds the current figure inline (`![](../results/<dir>/pareto_faithfulness.png)`
  with a path that renders in the user's markdown viewer).
- A short methods section (what the axes/colors mean, the grad_err metric, the
  budget control).
- A **per-parameter "why it fails" subsection** with the empirical reading from
  the panel plus the mechanism (see §6). Updated as each phase lands, so the
  doc tracks reasoning *and* progress.
- Closes with the failure-mode taxonomy table (param → category → mechanism →
  what Sobolev does).

## 5. The grad_err metric (pinned, = the gate)

`grad_err(candidate) = median over kept k of |∂eq/∂θ ÷ ∂P_GP/∂θ − 1|`, at the
fiducial θ, where "kept" k-bins are those with `|∂P_GP/∂θ| ≥ 1e-3 · max_k
|∂P_GP/∂θ|` (the gate's `median_rel_error` floor). `∂eq/∂θ` is the candidate
equation's finite-difference gradient (`equation_param_gradient`); `∂P_GP/∂θ` is
the GP's (`gp_param_gradient`). `log_space=True` matches the production target
(stage6/9 fit `log P`). Gate threshold tol = 0.25. This is **the same metric the
production gate uses** — the sidecar must equal what the gate would decide.

## 6. Per-parameter "why" — the reasoning the walkthrough must carry

The mechanisms to explain (verify each against the actual panels before
asserting in the doc):

- **General Mirage:** PySR minimizes value MSE; a low-MSE equation can have the
  wrong *slope* at fid. Fisher sees only the slope, so value-accurate ⇏
  derivative-accurate (arXiv:2406.06067). This is why color (derivative) and
  height (value) disagree.
- **ns** — pivot/tilt response; the value-optimal equation captures P's shape,
  not ∂P/∂ns. Red under value@20 *and* value@certified (budget doesn't fix it),
  green under Sobolev → the clean cure demonstration.
- **hub** — two candidate causes, both checkable on the panel: (a) **under-
  search** — `x0` enters only at high complexity (review #6: complexity ~6),
  signal buried under a resolution offset; (b) **wrong basis** — hub acts like a
  k-rescaling / AP-like distortion, a coordinate transform of k that a per-param
  native-k 1D ansatz cannot express. If hub stays red even under Sobolev at all
  complexities → the basis argument, not just budget.
- **bhfeedback** — weak/near-degenerate gradient (priored out;
  `memory/feedback_anova_loss_impact.md`). The response is tiny, so the equation
  can't lock onto it and grad_err is ill-conditioned.
- **herei, alphaq** — the real cross-term coupling (+0.45,
  `memory/headline_findings.md`) is unrepresentable by per-param-1D + additive
  combine (review #4); these are among the worst-faithfulness params, which the
  figure should corroborate.

## 7. Phases

1. **Local (no cluster):** build §4.2 plotter; render value-only (gray) 11-panel
   figure from existing Pareto CSVs; stand up the walkthrough doc with the
   figure embedded + methods section. → user reviews layout.
2. **Cluster:** ship §4.1 `--out`; run §4.3 sidecar driver for 11 params ×
   {value@20, Sobolev@20}; re-render in color = the money figure; fill the
   per-param "why" subsections.
3. **Cluster:** capture the ns budget-control verdict, overlay series 3; decide
   (with the user) whether to run hub/bhfeedback certified-budget controls.
4. **Writeup:** finalize the taxonomy table; promote the walkthrough into the
   paper's failure-modes section.

## 8. Testing

- `eval_grad_faithfulness --out`: a fast unit test on a **stub Pareto CSV** with
  a known equation + monkeypatched gradients asserting the sidecar columns and
  that `gate_pass == (grad_err <= tol)`. No emulator in the test.
- `plot_pareto_faithfulness`: a unit test that, given a tiny Pareto CSV + a stub
  sidecar (and given a *missing* sidecar), the figure is produced without error
  and the gray-fallback path triggers when the sidecar is absent (assert on the
  matplotlib artists / that a file is written, `Agg` backend).
- No network/emulator in the test suite; the emulator-dependent path stays in
  the driver, exercised by the cluster run, not pytest.

## 9. Provenance / honesty guards

- Each series records its **budget** (maxsize/niter) and **source path** in the
  figure legend and the walkthrough, so value@20 vs Sobolev@20 vs value@certified
  are never silently conflated (the budget confound is the whole point).
- The figure plots **no GP-slice fallback values** — only real per-candidate SR
  gradients. Params with no faithful equation read as all-red, on purpose.
- The walkthrough states plainly that this redirect **drops** the σ_PySR/σ_GP
  forecast claim (review #1/#3) in favor of the derivative-faithfulness
  diagnostic, which the GP-as-oracle setup *can* legitimately support.

## 10. Deliverables

- `scripts/plot_pareto_faithfulness.py` (new)
- `scripts/eval_grad_faithfulness.py` (`--out` upgrade)
- `scripts/make_grad_faith_sidecars.sh` (or python driver)
- `results/single_z_stage_pareto_diag/` — figure PNG/PDF + sidecar CSVs
- `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` (living, inline figures)
- tests in `tests/` per §8
