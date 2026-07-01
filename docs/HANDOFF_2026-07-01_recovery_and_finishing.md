# HANDOFF — 2026-07-01 session recovery + paper finishing pass

**Branch (code):** `stage10-multiz-sobolev` · **HEAD `96e8b16`** (pushed, PR #6 OPEN).
**Paper repo:** `/home/mfho/Latex/Knowledge-Distillation-using-PySR-with-PRIYA-suite`,
branch `paper-additions`, HEAD `b83385b` **+ uncommitted on-disk edits** (this session;
per that repo's policy the author reviews/commits in LaTeX Workshop). Paper builds clean,
**16 pp, no undefined refs**.

Supersedes the state in `HANDOFF_2026-06-30_perz_sobolev.md` (whose HEAD hashes
`061ddc4`/`5e8a43d` were mid-stream and are stale). The live pointer is this file +
the auto-memory `active_work.md`.

---

## 0. TL;DR

The prior session (`434f5621`, 2026-06-30) disconnected mid-task after a **rate-limit
cascade** (two subagents died 429 while writing output READMEs + polishing figures). This
session **recovered its state with a 4-agent + 2-referee cross-check**, then executed a
finishing pass. The paper was already *quantitatively* consistent with the production data;
the work here was consolidation, reproducibility fixes, resolving the one open scientific
question (log-vs-linear grad_err), a real code fix (M2 guard), and applying the BLOCKING
paper corrections + reframe. **What remains is a bounded set of author-decision `\mfho`
points** (see `docs/PAPER_MFHO_DECISION_SHEET_2026-07-01.md`) + Phase 6 (final review, merge,
submit).

---

## 1. Code repo — commits pushed this session (`7aa26af → 96e8b16`)

| Commit | What |
|---|---|
| `e502d9a` | Commit the 3 production-dir READMEs (written-but-uncommitted at the disconnect) + ignore `results/_tutorial_scratch/` |
| `df1d04f` | REPRODUCE.md Tier-2 accuracy: `regen_fig1.py` **does** take `--refit-dir` (the "hard-codes" gotcha was pre-2026-06-30 and stale); added the FIGREPO clone URL; corrected the "committed PDFs" wording |
| `9c2c74e` | VERDICT.md superseded-banner (grad_err is log-space, Fisher-consistent) + **`docs/PAPER_MFHO_DECISION_SHEET_2026-07-01.md`** |
| `3152ea0` | SLURM default account `yueyingn0` (expired 2026-07-01) → `cavestru0` in all executable slurm/scripts |
| `96e8b16` | **M2 guard** on the low-level multi-z Sobolev refit + test (full suite **431 passed, 14 skipped**) |

---

## 2. The one scientific result: grad_err is *exactly* Fisher-consistent

The open contradiction was: `docs/pr_review/VERDICT.md:19-23` called grad_err a **linear-P**
ratio (Fisher-consistent because the KODIAQ covariance is linear), while commit `7aa26af`
switched it to **log-space**. Resolved by reading the code (all three likelihood paths +
`fisher.py` have **no `np.log`** → cov/data are linear P_F):

- The **deployed** emulator is the fiducial-anchored additive combine, for which
  `∂P_F/∂θ|fid = P_F^GP(fid)·∂logP_eq/∂θ` (`refit_taylor.py:262,280,456,474`).
- In the ratio to the GP the anchor `P_F^GP(fid)` **cancels bin-by-bin**, so the equation's
  **log**-slope ratio equals the deployed combine's **linear** (Fisher-space) slope ratio —
  exactly the Fisher-relevant quantity for a linear-P_F covariance.
- ∴ log-space grad_err is **exactly Fisher-consistent for the deployed model, not a proxy**.
  VERDICT.md:19-23 scored the *un-deployed* standalone equation (banner added). No σ-claim
  rides on it (money-plot was dropped). **The paper now states the anchor-cancellation
  identity** at the grad_err definition (see §4).

---

## 3. M2 — multi-z Sobolev log/linear guard (fixed)

`_build_training_matrix_multiz` builds the target in **linear** P_F, but the multi-z Sobolev
target gradient is **log-P** → inconsistent objective. **The driver
`refit_one_param_multi_z` already guarded this** (`multi_z/refit.py:246`, raises
`NotImplementedError`; the referee flagged the low-level *comment* without seeing the driver
guard). This session closed the **defense-in-depth gap**: the low-level
`refit_1d_multiz_for_param` now raises by default too, with an `_allow_unvalidated_sobolev`
opt-in so the wiring test still exercises the loss/weights plumbing. New test
`test_multiz_refit_low_level_sobolev_guard`. **Not** a full fix of multi-z Sobolev (that
needs log-space support in the multi-z builder + normalization — out of scope, the paper is
single-z); it just makes the broken path fail loud instead of silent.

---

## 4. Paper edits applied on disk (uncommitted — review in LaTeX Workshop)

All build clean (16 pp). `\mfho` markers **46 → 42**.

- **Abstract (L94):** reframed to the derivative-faithfulness / Fisher's-Mirage / Sobolev headline.
- **Conclusion (L815, L825):** stale accuracy paragraph ("dτ0 best / n_S worst", specific %s)
  trimmed to a one-sentence value-accuracy bridge; the reframed closing paragraph (L825
  `\additions`) is the new close.
- **L252:** σ_PySR/σ_GP "10³–10⁶×" + Fisher footnote → recast in grad_err terms (retracted quantity gone).
- **L794:** "10⁶–10¹⁹" Fisher factors → recast as parameter-Jacobian rank-collapse (retracted quantity gone).
- **L627:** stale n_S HF 2.03% → **5.16%** (matches `tab:stats_table`) + reframe-coherent softening.
- **grad_err definition (`sec:fisher`):** added the **anchor-cancellation sentence** (§2); kept the one-sentence Fisher motivation (L522/L683 resolved as "keep").
- **L518:** wrote the **`sec:resolution` "Resolution dependence" subsection** — all 11 production
  equations carry the resolution feature `x2` (pinned at LF=0.4/HF=0.8, *not* swept 0→1); the
  LF→HF correction is a bounded few-percent **broadband** suppression. Exact per-parameter
  R(k) magnitudes were left to regenerate (see §6) rather than quote superseded runs.

**Retracted-Fisher magnitudes confirmed gone from live text** (grep); only a `%`-commented
σ-table note remains.

---

## 5. Reproducibility state (student-facing, built + verified)

- **`notebooks/figures_tutorial.ipynb`** (tracked) — the **reproduce-AND-tweak** notebook.
  Emulator-free (light deps), CWD-robust, no hard-coded paths (`pf.load_run()`), reproduces
  every emulator-free figure + Tables 6/7 via `priya_forecast.paper_figures`, with explicit
  knobs (`gate=`, `paper_style(usetex=, scale=)`, `highlight=`, series recolour, reuse-on-new-run).
- **`notebooks/reproduce_paper_figures.ipynb`** (tracked) — straight top-to-bottom repro + Tier-2 command docs.
- **`REPRODUCE.md`** — 3-tier guide (Tier-1 emulator-free ~2 min; Tier-2 GP prediction figs; Tier-3 re-fit).
- Verified this session (referee): `paper_figures` imports in `.venv`; `load_run()` populates all
  sidecars/budget/seed_band/maxsize/multid/crossz; Tier-1 numbers reproduce verbatim; 401 tracked
  files back the light-deps path.
- **Minor follow-ups (non-blocking):** the tutorial writes `pareto_faithfulness_tutorial.png` to the
  repo root (cosmetic — could target `_tutorial_scratch/`); the notebooks' hard-coded FIGREPO path
  for Tier-2 (the builder `_build_reproduce_paper_figures.py`, gitignored) still uses a personal path —
  REPRODUCE.md §2d now gives the clone URL, but the notebook cell could be de-personalized on next regen.

---

## 6. What remains for the final product (author decisions)

All in **`docs/PAPER_MFHO_DECISION_SHEET_2026-07-01.md`** (44→~42 markers, grouped BLOCKING vs optional):
1. **Table 1 priors (L173):** reconcile z_Hei/z_Hef to the emulator box `[3.5,4.5]`/`[2.2,3.2]` (table has 4.1/2.6). *Needs your call on box-vs-physical-prior.*
2. **Eq conventions (L228 ≡ L314):** confirm log-P for the `eq:norm` anchor + the combine (answer both identically).
3. **§cost wall-time (L807):** update niter=50 → the niter=200 production timing (*needs the actual wall-time; not in the committed artifacts*).
4. **~40 remaining `\mfho`:** mostly accept-and-delete FYIs + the keep/drop-float confirmations (Fig 10 maxsize_sens is the one worth considering restoring).
5. **Optional:** regenerate the resolution-correction R(k) table/figure on the production refits
   (`priya_forecast.deliverables.write_resolution_correction_outputs`, Tier-2/GP) so `sec:resolution`
   can quote current per-parameter magnitudes + enable `fig:rescorr` (commented block already in place).

**Phase 6 (final product):** work the decision sheet → final `latexmk` → optional last cross-check
pass → merge PR #6 → submission checklist.

---

## 7. Key pointers
- Decision sheet: `docs/PAPER_MFHO_DECISION_SHEET_2026-07-01.md`
- Production artifacts: `results/paper_production_20260630_perz_sobolev_z2.6-4.2/` (README + RUN_MANIFEST)
- grad_err adjudication evidence: `docs/pr_review/VERDICT.md` (banner) + `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md:48-57`
- Reproduce: `REPRODUCE.md`, `notebooks/figures_tutorial.ipynb`, `priya_forecast.paper_figures`
- Memory: `active_work.md` (2026-07-01 recovery section = live resume pointer)
