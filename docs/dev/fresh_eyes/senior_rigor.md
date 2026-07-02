# Fresh-eyes referee report — scientific honesty, completeness, gaps

**Role:** senior_rigor (referee assessing honesty / completeness / gaps)
**Date:** 2026-06-09
**Method:** genuine fresh clone of the repo, emulator-free path only.

```
git clone -q /home/mfho/lya1d_priya_forecast /tmp/fe_referee
cd /tmp/fe_referee && git checkout -q stage10-multiz-sobolev
# HEAD = 1449001 (branch tip is AHEAD of the orchestrator's stated bbee352)
PYTHONPATH=src /home/mfho/lya1d_priya_forecast/.venv/bin/python <cmd>
```

Bottom line: **the headline emulator-free reproducer works end-to-end and its
provenance data is fully committed and self-consistent** — a genuinely good state.
The honesty problems are in the *surrounding* claims: a stale test-count headline
(fails on a clean clone), undocumented seed/single-run provenance for the entire
taxonomy, dead `memory/` pointers in the *current* source-of-truth doc, and a large
"reproduction" doc that still walks a referee through the *retracted* result.

---

## WHAT WORKED (and was it discoverable?)

1. **Headline figure reproducer — works, fully discoverable.** README §Usage and
   HANDOFF both give the exact command:
   ```
   PYTHONPATH=src python scripts/make_diagnostic_figs.py --out-dir <dir>
   ```
   Ran clean (`wrote 4 figures (png+pdf)`, exit 0). The four PNGs render correctly
   and match the walkthrough's verbal description (hub/bhfeedback all-red; ns panel
   carries the budget triangles; green Sobolev squares for the faithful params).
   Regenerated PNGs are byte-size-identical to the committed
   `results/single_z_stage_pareto_diag/*.png` (differ only in PNG metadata).

2. **Provenance data is committed and self-contained.** The reproducer reads
   `pareto_*.csv` + `grad_faith_*.csv`; a clone ships all of them (67 sidecars; 11
   pareto fronts each for value@20, Sobolev@20; the ns budget pair; and all four
   cross-z stages). **I independently re-derived the walkthrough's headline numbers
   from the raw sidecars**: ns value best-loss 0.603 / best-faith 0.512 / Sobolev
   0.193 — exact match to the table (lines 109–119). This is real, checkable
   provenance and is the strongest honesty point in the repo.

3. **The metric-space claim is TRUE in code.** The walkthrough's prominent "Metric
   space (was mislabeled — fixed)" note claims `grad_err` differences linear/raw
   `P_F`, not `logP`. Verified in `src/priya_forecast/derivative_gate.py`:
   `gp_param_gradient` / `equation_param_gradient` both finite-difference
   `gp.predict` / `refit.predict`, and `Refit1DResult.predict` applies `exp()` when
   `log_space=True` — so the gate truly differences linear `P_F`. The claim holds.

4. **Tests mostly run, env boundary mostly honest.** `pytest -k "not slow"` ran in
   ~40s with the right skips (`lyaemu` absent → skip). The emulator boundary is
   real and mostly well-gated.

Discoverability was good for the headline path: I did **not** have to guess. The
two "Start here" notebooks exist and have proper repo-root chdir + sidecar
existence smoke-checks.

---

## MISSING (could not proceed / had to work around)

### M1. No seed / single-run provenance for the *entire* taxonomy — the biggest gap
PySR is stochastic. The whole scientific claim — "selection-sensitive" vs
"generative Mirage" vs "resistant", the ns 0.603→0.193 cure, "budget proves search
depth alone is not enough", the z=4.2 "reshuffles" — rests on PySR Pareto fronts.
**No seed is recorded anywhere** (not in `pareto_*.csv`, not in `grad_faith_*.csv`
headers) and **no current doc states whether these are single-seed or averaged.**
```
$ grep -rin "seed" README.md HANDOFF.md docs/PARETO_FAITHFULNESS_WALKTHROUGH.md
# (nothing)
$ head -2 results/single_z_stage6_log/refit/z3.6/pareto_ns.csv
Complexity,Loss,Equation,score,sympy_format,lambda_format   # no seed column
```
A referee cannot tell whether "value@budget plateaus at 0.319 > gate" is a robust
fact or one unlucky draw. The only determinism discussion in the repo
(`docs/REPRODUCE.md` §7) is explicitly about the *retracted* σ-result and says the
search is "non-deterministic across runs by default."
**Fix:** add a "Reproducibility / seeds" paragraph to
`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` stating the seed(s) used and whether the
fronts are single-run; ideally record `seed`/`niterations`/`maxsize` in the sidecar
header (`src/priya_forecast/grad_faith_io.py:45`, alongside `tol=`/`log_space=`) and
re-emit. If single-seed, say so and add the across-seed spread for at least ns
(the headline generative-Mirage claim).

### M2. Provenance of the GP data the gate needs is undocumented in current docs
README §Prerequisites(3) says the trained GP "must be present under
`data/kodiaq_gp/`" but **never says where it comes from or how to build it.** A
clone has only `data/priya_fiducial/.gitkeep`. The only pointer to the builder
(`scripts/prep_kodiaq_gp.py`, which exists) is buried inside a test's *error
message*; the script is referenced by **no** current doc.
```
$ grep -rn "prep_kodiaq_gp" README.md HANDOFF.md docs/PARETO_FAITHFULNESS_WALKTHROUGH.md
# (nothing)
```
The source-data location only lives in the superseded `docs/REPRODUCE.md` §2a
(`/nfs/turbo/umor-yueyingn/.../kodiaq_2_2_4_6-48-48/`).
**Fix:** in README §Prerequisites(3), add one line: "build it with
`scripts/prep_kodiaq_gp.py --source <SRC> --dest data/kodiaq_gp`; on Greatlakes
`<SRC>` is `/nfs/turbo/umor-yueyingn/.../kodiaq_2_2_4_6-48-48/`; off-cluster, obtain
the trained pickles from the umor-yueyingn group or rebuild with upstream `lyaemu`."

### M3. `memory/*.md` pointers are dead — including in the CURRENT source-of-truth doc
The walkthrough (the declared source of truth) cites the headline herei×alphaq
coupling as `memory/headline_findings.md` (line 159). **That file is not in the
repo** — it lives in the author's private `~/.claude/.../memory/`. So a referee
cannot verify the +0.45 coupling claim at all. Same dead relative pointers in
`docs/REPRODUCE.md` (`memory/pysr_gp_gotchas.md`, `memory/feedback_pysr_speed.md`).
**Fix:** either commit a `docs/` copy of the cited memory notes, or change the
walkthrough citation to an in-repo artifact (e.g.
`scripts/run_coupling_matrix.py` + its committed output) that actually backs the
+0.45 number. Drop/redirect the `memory/` links in REPRODUCE.md.

---

## MISLEADING (doc says X, reality is Y)

### X1. "412 pass, ~13 skip" — false on a clean clone; **1 test FAILS**
README:111 and HANDOFF:69–72 both headline `(412 pass, ~13 skip)` and claim the
only env-dependent test is `test_real_gp_predicts_at_fiducial`. Reality on a fresh
clone:
```
1 failed, 411 passed, 14 skipped
FAILED tests/test_single_z_pipeline.py::test_shipped_example_yaml_loads_and_validates
  ValueError: gp.basedir does not exist: data/kodiaq_gp.
```
This test is **not gated/skipped** — it requires gitignored `data/kodiaq_gp/` and
hard-fails without it, contradicting the docs' "emulator-touching tests are
gated/skipped." So the documented green-bar sanity check is red on a true clone.
**Fix:** (a) skip the test when `data/kodiaq_gp` is absent (mirror the existing
`lyaemu`/`data` skip guards in the same file), and (b) correct the count in
README:111 + HANDOFF:69 to the post-fix reality (e.g. "411 pass, 15 skip on a clone
without `data/kodiaq_gp`").

### X2. `docs/REPRODUCE.md` reproduces the RETRACTED result, with no in-file banner
REPRODUCE.md is the doc a referee opens to reproduce. Its entire body walks
"§1c (aggregation): expect a table with σ_GP, σ_PySR, σ_PySR/σ_GP" — i.e. the exact
σ-ratio forecast the README says was retracted as "confounded by construction
(σ_perfect_1D ≡ σ_GP is a forced Jacobian identity)." README flags it as superseded,
but **REPRODUCE.md itself carries no banner**, so anyone landing there directly is
walked through the withdrawn claim as if it were the deliverable. (Its result dirs
like `results/refit_phase2_production/` *are* present in a clone, so the steps look
legitimate, which deepens the trap.)
**Fix:** add a one-line banner at the top of `docs/REPRODUCE.md` and `README_v2.md`:
"SUPERSEDED 2026-06-08 — the σ_PySR/σ_GP forecast below was retracted; see
`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`. Retained for history."

### X3. Branch name `stage10-multiz-sobolev` oversells; multi-z Sobolev is DROPPED
The branch (and the implied deliverable) is "multi-z Sobolev," but the actual
headline is **single-z z=3.6** plus a *per-z retrained* cross-z robustness check.
HANDOFF:86 discloses the multi-z "money plot" was **dropped**; README never mentions
it. A referee seeing the branch name expects a multi-z joint Sobolev result that
does not exist. The `derivative_faithful_multiz` code path exists
(`derivative_gate.py:62`) but is not exercised by any committed result.
**Fix:** one sentence in README and the walkthrough intro: "Scope is single-z
z=3.6 (plus per-z-retrained cross-z robustness); the multi-z joint Sobolev plan
was dropped — the multi-z gate code is present but unused."

### X4. Stale `∂logP_GP` label in the multi-z gate docstring — the "fixed everywhere" claim is not quite true
The walkthrough says the `∂P` vs `∂logP` mislabel was "corrected ... everywhere."
But `derivative_gate.py:67` (`derivative_faithful_multiz` docstring) still reads
`|∂eq/∂θ ÷ ∂logP_GP/∂θ − 1|` — it differences raw `P_GP` (uses `gp_param_gradient`,
which is linear `P`), so the docstring is stale/wrong in exactly the way the
walkthrough says was fixed.
**Fix:** edit `src/priya_forecast/derivative_gate.py:67` `∂logP_GP/∂θ` →
`∂P_GP/∂θ`.

### X5. `log_space=True` in sidecar header is a naming trap
`grad_faith_*.csv` headers read `log_space=True`, which a referee reading the
walkthrough's "metric is in linear P_F" note will read as a contradiction. It is
NOT — `log_space` is the *PySR training target* (log-P), independent of the gate's
(linear) metric space. But nothing in the sidecar or doc says so.
**Fix:** note in the walkthrough's metric paragraph: "the sidecar's `log_space=True`
flags the log-P *training target*, not the gate's metric space (which is linear
P_F)."

### X6. "robustly faithful" overstates heref (and partly alphaq)
Taxonomy §1 + table label `heref` "robustly faithful," yet its value→Sobolev
grad_err is **0.154 → 0.206** (Sobolev makes it *worse*, and 0.206 is near the 0.25
gate), and the cross-z section says the He II block incl. heref **"blows up at
z=4.2" (2.69)**. The doc does reconcile this later ("taxonomy is NOT
redshift-uniform"), but the bare word "robustly" in the z=3.6 heading is misleading.
**Fix:** qualify the §1 heading "Robustly faithful **(at z=3.6)**" and footnote
heref/alphaq as "z-localised: faithful at z≤3.6, fails at z=4.2 — see cross-z."

### X7 (minor). README "Repository layout" overstates `data/` and omits the new notebooks
README:188–189 lists `data/ kodiaq_gp/ ..., single_z_1pvar/` as if shipped; a clone
has only `data/priya_fiducial/.gitkeep`. README:183 lists `notebooks/` as "01–03"
only, omitting the two emulator-free "Start here" notebooks the README itself
promotes elsewhere.
**Fix:** mark the `data/` subdirs "(gitignored; not in a clone — see Prerequisites)"
and add the two reproducer notebooks to the layout line.

### X8 (minor). `make_grad_faith_sidecars.sh` hardcodes author paths and silently overrides PYTHONPATH
Lines 11–14 hardcode `$HOME/.julia_env`, `$HOME/.julia`, and
`PYTHONPATH="src:/home/mfho/student_projects/lya_emulator_full"`. README §2 presents
these as user-set exports, but the script overwrites whatever a referee set.
**Fix:** make the upstream-emulator path a parameter/env-var with a default, and
note in README that the script overrides `PYTHONPATH`.

---

## What a referee would still need before calling this "trustworthy & complete"
- Seeds / single-run-vs-averaged disclosure for every Pareto-derived claim (M1).
- A clean-clone-green test bar with an honest count (X1).
- Verifiable provenance for the +0.45 coupling cited in the current doc (M3).
- Clear "superseded" banners so the retracted σ-result can't be mistaken for the
  deliverable (X2), and honest scoping of "multi-z" (X3).
</content>
</invoke>
