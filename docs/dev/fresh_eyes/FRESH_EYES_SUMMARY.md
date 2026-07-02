# Fresh-eyes meta-synthesis — can a new user install, reproduce, and build on this repo from the docs alone?

**Date:** 2026-06-09 · **Branch under test:** `stage10-multiz-sobolev` (HEAD `1449001`)
**Inputs:** four independent fresh-clone reports —
`grad_reproduce.md`, `grad_extend.md`, `senior_reproduce.md`, `senior_rigor.md`.

---

## (1) Overall

**Mostly yes for reproduce, partly no for build-on, and one false headline for install.**
All four reviewers independently confirmed the README's core promise from a genuine
fresh clone: `scripts/make_diagnostic_figs.py` and `notebooks/reproduce_paper_figures.ipynb`
both run **emulator-free** (no GPy / PySR / Julia / `data/kodiaq_gp` loaded — verified
via `sys.modules`), exit 0, and the four diagnostic figures match the walkthrough; the
two senior referees re-derived every headline number (ns Mirage 0.603→0.193, budget@35
grad_err 0.319, the full 4-way taxonomy, the z=4.2 He II blow-up) **exactly** from the
tracked sidecars, and 3 of 4 committed figures are pixel-identical. That is a genuinely
strong, checkable provenance state. **But** the one install-sanity claim the docs lead
with — "412 pass, ~13 skip, emulator-free" — is **false on a clean clone** (it is
1 failed / 411 passed / 14 skipped); a *build-on* user cannot write a new script from
the README alone (the `pareto_diag` library API is undocumented) and falls into an
undocumented NaN trap; and a senior referee cannot verify two load-bearing science
claims (the h basis test and the +0.45 herei×alphaq coupling) because their results
are committed nowhere. Net: reproduction works, the docs over-promise the test bar,
and the library/rigor edges need closing before this is a clean new-user experience.

---

## (2) MISSING PARTS (deduplicated, prioritised)

| P | Item | Who hit it | Why it blocks | Fix (file + change) |
|---|---|---|---|---|
| **P0** | Shipped-config test **fails on a clone** — `test_shipped_example_yaml_loads_and_validates` → `load_config` → `GPConfig.validate()` (`config.py:110-115`) hard-fails: `ValueError: gp.basedir does not exist: data/kodiaq_gp`. The documented "quickest sanity check on an install" is red. | **all 4** (grad_reproduce M1, grad_extend X1, senior_reproduce MIS-1, senior_rigor X1) | The first thing a new user runs (`pytest -k "not slow"`) returns a FAIL the docs say can't happen; un-gated test depends on gitignored emulator data. | `tests/test_single_z_pipeline.py` (~:54): `pytest.mark.skipif(not Path("data/kodiaq_gp").exists())`, **or** better decouple schema from filesystem — `load_config(..., validate_paths=False)` so the YAML round-trip checks parsing, not data. Then correct the count in README:111 + HANDOFF:69. |
| **P1** | **h basis test not reproducible and result committed nowhere** — `scripts/h_basis_test.py` needs the GP (`data/kodiaq_gp`), dies in a clone; no json/csv/npz/md records corr≈−0.25 / ~6% var. It is a *headline reversal* of the "h=AP" hypothesis. | senior_reproduce (MISS-1), senior_rigor (M3 cites same gap via coupling) | A referee cannot verify, recompute, or even spot-check the single number that overturns the prior hypothesis. | Add `--out` to `scripts/h_basis_test.py` writing `{z:{corr,var_explained,n_k}}`; commit `results/h_basis_test/h_basis.json`; flag GP-only in WALKTHROUGH §4 / HANDOFF:41. (Also make it exit non-zero on missing data — currently exits 0; senior_reproduce MIS-4.) |
| **P1** | **No "use it as a library" path** — `load_front` / `render_grid` / the `fronts_by_param` dict shape appear in **zero** prose docs (README names the *file* only; HANDOFF:46 lists bare names, no signatures). | grad_extend (M1) | A build-on user must open `src/priya_forecast/pareto_diag.py` source to write any new script; README alone is insufficient. | Add a short "Use it as a library" snippet to `README.md` after the figure-reproducer section (the minimal `load_front`/`render_grid` call, incl. the dropna-before-idxmin idiom). |
| **P1** | **"value-optimal" silently yields NaN** — sidecar scores only Fisher-safe rows, so naive `front.loc[front["Loss"].idxmin()]` prints NaN grad_err for `ns` (min-Loss row c=19 unscored; correct is c=16 → 0.603). Undocumented. | grad_extend (M2) | The single thing that breaks a naive build-on implementation; invisible on `dtau0`, fatal on `ns`. | `pareto_diag.load_front` docstring + WALKTHROUGH:102: "value-optimal = lowest Loss **among sidecar-scored (Fisher-safe) rows**, e.g. `df.dropna(subset=['grad_err']).sort_values('Loss')`." |
| **P2** | **No seed / single-run-vs-averaged disclosure anywhere** — PySR is stochastic; the entire taxonomy (selection-sensitive vs generative Mirage, ns cure, budget "plateaus", z=4.2 "reshuffles") rests on it. No seed in `pareto_*.csv` / `grad_faith_*.csv` headers; no doc says single-seed or averaged. | senior_rigor (M1) | A referee can't judge whether "0.319 > gate" is robust or one unlucky draw. | Add a "Reproducibility / seeds" paragraph to WALKTHROUGH; record `seed`/`niterations`/`maxsize` in the sidecar header (`grad_faith_io.py:45`) and re-emit; give across-seed spread for at least `ns`. |
| **P2** | **GP data provenance undocumented** — README §Prereq(3) says GP "must be present under `data/kodiaq_gp/`" but never says where it comes from; `scripts/prep_kodiaq_gp.py` (exists) is named only inside a runtime *error message*; source path lives only in superseded REPRODUCE.md. | grad_extend (M3), senior_rigor (M2), senior_reproduce (MISS-2) | Anyone trying the GP path is stuck; the prep tool is undiscoverable from prose. | README §Prereq(3): one line — `python scripts/prep_kodiaq_gp.py --source <SRC> --dest data/kodiaq_gp`; on Greatlakes `<SRC>`=`/nfs/turbo/umor-yueyingn/.../kodiaq_2_2_4_6-48-48/`; off-cluster, obtain from umor-yueyingn group. |
| **P2** | **Dead `memory/*.md` pointers in the CURRENT source-of-truth doc** — WALKTHROUGH:159 cites the +0.45 herei×alphaq coupling as `memory/headline_findings.md`, which lives in the author's private `~/.claude/`, not the repo. Same dead links in REPRODUCE.md. | senior_rigor (M3) | The +0.45 coupling is unverifiable; broken links in the doc a referee is told to trust. | Commit a `docs/` copy of the cited note, **or** repoint the citation to an in-repo artifact (`scripts/run_coupling_matrix.py` + committed output). |
| **P3** | `data/` layout overstated — README "Repository layout" lists `kodiaq_gp/`, `single_z_1pvar/` as if shipped; a clone has only `data/priya_fiducial/.gitkeep`. | grad_reproduce (M2), senior_rigor (X7) | Cosmetic but reads as "this is in the repo"; minor confusion. | Annotate layout entries "(gitignored — not in a clone; see Prerequisites)"; add the two reproducer notebooks to the `notebooks/` line. |

---

## (3) MISLEADING PARTS

| Doc location | Says | Reality | Fix |
|---|---|---|---|
| README:111, HANDOFF:69-70 | "(412 pass, ~13 skip)" emulator-free; only `test_real_gp_predicts_at_fiducial` is env-dependent | **1 failed, 411 passed, 14 skipped** on a clean clone; the *failing* test is a different one (`test_shipped_example_yaml_loads_and_validates`) the docs never mention — contradicts "emulator-touching tests are gated/skipped." | After the P0 test fix, set the true count and keep "only env-dependent test" framing honest. **(all 4 reviewers)** |
| `scripts/make_diagnostic_figs.py:1` (docstring) | "Regenerate the **three** diagnostic paper figures" (list stops at "3. ns_budget_panel") | Code makes **four** (`crossz_faithfulness`, lines 147-172) and prints "wrote 4 figures." *Verified in live repo.* | Docstring → "four"; add bullet `4. crossz_faithfulness — redshift robustness, z=2.6/3.6/4.2`. **(grad_reproduce X1)** |
| `notebooks/reproduce_paper_figures.ipynb` cell 4 | (implicitly) safe to run | Hardcodes `OUT = results/single_z_stage_pareto_diag` (the **committed** dir); executing overwrites 8 committed PNG/PDF in place → dirty tree. | Default `OUT` to a scratch path (`os.environ.get("REPRO_OUT", "results/_repro_scratch")`) + intro note. **(grad_reproduce X2)** |
| `pareto_diag.render_grid` default | `y_col="Loss"` | WALKTHROUGH:56-64 (boxed) says you must **NOT** plot `Loss` (training objective, differs by run; makes Sobolev look worse by construction) — plot `value_mse`. Default gives exactly the axis the docs warn against. | Default `y_col="value_mse"` (or a docstring warning). **(grad_extend X2)** |
| committed `crossz_faithfulness.png` | reproduced by the documented command | **Stale**: committed 900×1650 (from `861bcda`) vs 780×1350 fresh (figsize gained at `1449001`); the "single generator" replaces rather than reproduces it. Numbers unaffected. | Regenerate + recommit `crossz_faithfulness.{png,pdf}`; add a CI check that all 4 PNGs match. **(senior_reproduce MIS-3)** |
| `docs/REPRODUCE.md` (+ `README_v2.md`) | walks a referee through σ_GP / σ_PySR / σ_PySR/σ_GP as the deliverable | That forecast is **retracted** (σ_perfect_1D ≡ σ_GP is a forced Jacobian identity). README flags it superseded; REPRODUCE.md itself carries **no banner**, and its result dirs exist in a clone, deepening the trap. | One-line banner atop REPRODUCE.md / README_v2.md: "SUPERSEDED 2026-06-08 — see WALKTHROUGH; retained for history." **(senior_rigor X2)** |
| branch name `stage10-multiz-sobolev` | implies a multi-z joint Sobolev deliverable | Headline is **single-z z=3.6** + per-z-retrained cross-z check; multi-z "money plot" was **dropped** (HANDOFF:86); `derivative_faithful_multiz` exists but no committed result exercises it. | One sentence in README + WALKTHROUGH intro scoping it to single-z z=3.6; note multi-z gate code is present but unused. **(senior_rigor X3)** |
| `derivative_gate.py:67` docstring | `\|∂eq/∂θ ÷ ∂logP_GP/∂θ − 1\|` | Differences raw linear `P_GP` (uses `gp_param_gradient`); WALKTHROUGH says the `∂P` vs `∂logP` mislabel was "fixed everywhere" — this docstring is still stale. | Edit `∂logP_GP/∂θ` → `∂P_GP/∂θ`. **(senior_rigor X4)** |
| `grad_faith_*.csv` header `log_space=True` | reads as contradicting the "metric is linear P_F" note | Not a contradiction: `log_space` is the PySR *training target*, independent of the gate's (linear) metric space. Nothing says so. | WALKTHROUGH metric paragraph: note `log_space=True` flags the log-P *training target*, not the gate's metric space. **(senior_rigor X5)** |
| WALKTHROUGH §1 heading / table | `heref` "robustly faithful" | value→Sobolev grad_err 0.154→0.206 (Sobolev makes it *worse*, near the 0.25 gate) and the He II block "blows up at z=4.2" (2.69). | Qualify heading "Robustly faithful **(at z=3.6)**"; footnote heref/alphaq as z-localised. **(senior_rigor X6)** |
| `configs/default.yaml:14` | `gp_emulator_basedir: /home/mfho/student_projects/.../Emulator_Files` | User-specific absolute path; disagrees with `example.yaml:40` (`data/kodiaq_gp`); neither present in a clone. | Make both relative to `data/kodiaq_gp`; note in README the absolute path is a local default to override. **(senior_reproduce MIS-5)** |
| `make_grad_faith_sidecars.sh:11-14` | README §2 presents `PYTHONPATH`/Julia exports as user-set | Script hardcodes `$HOME/.julia_env`, `$HOME/.julia`, `/home/mfho/.../lya_emulator_full` and **overrides** whatever the user set. | Parameterise the emulator path (env var w/ default); note the override in README. **(senior_rigor X8)** |
| (environment, not a doc bug) `git clone <local-path>` | implied to work | A plain local clone hardlinks `.git/objects` and dies on a dangling loose object; the failed clone leaves a half-tree whose `ls` lies. Only `git clone file://…` reliably works. | README note for local clones on this HPC: use `file://`. **(grad_reproduce X3, grad_extend X3)** |

---

## (4) What's genuinely good (do NOT over-correct)

- **The emulator-free reproducer is real and the headline promise holds.** Confirmed
  by all 4 reviewers from independent fresh clones: script + notebook both run
  emulator-free, exit 0, four figures match the walkthrough. (grad_reproduce TL;DR,
  grad_extend, both seniors W1/§1).
- **Provenance is committed and self-consistent.** Both senior referees re-derived
  *every* headline number — all 11 params × {value,Sobolev} × {best-loss,best-faith,x0@},
  the ns budget control, the value_mse decoupling (~24%), the full cross-z table —
  **exactly** to 3 d.p. from the tracked sidecars (senior_reproduce W2-W3, senior_rigor §2).
  This is the strongest honesty point in the repo; don't disturb the sidecars.
- **3 of 4 committed figures are byte/pixel-identical** to a fresh run (only crossz is
  stale). (senior_reproduce W4.)
- **The metric-space claim is TRUE in code** — `derivative_gate.py` differences linear
  `P_F` (`Refit1DResult.predict` applies `exp()` when `log_space=True`), matching the
  walkthrough's "fixed" note. The bug here is only a stale *docstring* (X4), not the math.
  (senior_rigor §3.)
- **The reproducer notebooks are robust** — proper repo-root `chdir`, `__file__`-less
  execution handling, sidecar smoke-checks; both validate (nbformat v4.5) and execute
  end-to-end. (grad_reproduce §4, senior_rigor.)
- **The emulator boundary is mostly honestly gated** (`lyaemu` absent → skip cleanly);
  the conceptual ingredients (sidecar columns, 0.25 gate, "value-optimal = lowest loss")
  are all discoverable from the walkthrough. (grad_extend "what worked".)

---

## (5) Ordered TODO for a clean new-user experience

1. **[P0] Make `pytest -k "not slow"` green on a clone** — gate/decouple
   `test_shipped_example_yaml_loads_and_validates` from `data/kodiaq_gp`, then correct
   the count in README:111 + HANDOFF:69 (and the "only env-dependent test" framing).
   *Closes the one false install headline — flagged by all four reviewers.*
2. **[P1] Commit the h basis test result** — add `--out` to `h_basis_test.py`, commit
   `results/h_basis_test/h_basis.json`, make it exit non-zero on missing data, flag it
   GP-only in WALKTHROUGH/HANDOFF. *Restores a verifiable headline refutation (senior).*
3. **[P1] Add a "Use it as a library" snippet to README** (`load_front`/`render_grid`/
   `fronts_by_param` shape) **and** document the value-optimal NaN idiom
   (`dropna(subset=['grad_err']).sort_values('Loss')`) in the `load_front` docstring +
   WALKTHROUGH:102. *Unblocks build-on users — the two things that bit the grad.*
4. **[P1] Fix the docstring/figure mismatches** — `make_diagnostic_figs.py` "three"→"four"
   + 4th bullet; default the notebook `OUT` to a scratch dir; default
   `render_grid(y_col="value_mse")`. *Quick, removes the most visible "huh?" moments.*
5. **[P2] Add a "What a bare clone can and cannot reproduce" box to README** covering:
   test bar, h basis test is GP-only/uncommitted, `data/kodiaq_gp` provenance
   (`prep_kodiaq_gp.py --source <SRC>`), and the local-clone `file://` gotcha.
6. **[P2] Add a "Reproducibility / seeds" paragraph** (WALKTHROUGH) + record
   `seed`/`niterations`/`maxsize` in sidecar headers; give across-seed spread for `ns`.
7. **[P2] Banner the superseded docs** (REPRODUCE.md, README_v2.md) and **scope the
   "multi-z" branch name** honestly in README/WALKTHROUGH intro.
8. **[P3] Tidy** — regenerate+recommit `crossz_faithfulness.{png,pdf}`; fix the stale
   `derivative_gate.py:67` docstring; reconcile `default.yaml` vs `example.yaml` paths;
   parameterise `make_grad_faith_sidecars.sh`; correct the `data/`/`notebooks/` layout
   lines; repoint dead `memory/*.md` links.

---

## Executive summary (8 lines)

1. **Reproduction works:** all four reviewers, from independent fresh clones, confirmed the README's core promise — `make_diagnostic_figs.py` + the reproducer notebook run **emulator-free** (verified no GPy/PySR/Julia loads), exit 0, four figures match the walkthrough.
2. **Provenance is excellent:** both senior referees re-derived *every* headline number (ns 0.603→0.193, budget@35 0.319, full taxonomy, z=4.2 He II blow-up) **exactly** from tracked sidecars; 3 of 4 committed figures are pixel-identical. Don't disturb this.
3. **P0, unanimous:** the lead install claim "412 pass, ~13 skip, emulator-free" is **false on a clone** (1 fail / 411 pass / 14 skip) — `test_shipped_example_yaml_loads_and_validates` hard-fails on gitignored `data/kodiaq_gp`. Fix the test, then the count.
4. **Build-on is blocked for grads:** the `pareto_diag` library API (`load_front`/`render_grid`) is in **zero prose docs**, and "value-optimal" silently returns NaN for `ns` (Fisher-unsafe rows unscored) — undocumented, fatal to a naive script.
5. **Two science claims are unverifiable by a referee:** the h basis test (GP-only, dies on a clone, result committed nowhere) and the +0.45 herei×alphaq coupling (cited to a private `memory/` file not in the repo).
6. **Honesty gaps (senior):** no seed/single-run disclosure under the whole stochastic-PySR taxonomy; `docs/REPRODUCE.md` walks a referee through the **retracted** σ-forecast with no banner; the branch name oversells "multi-z" (the money plot was dropped).
7. **Smaller misleads:** `make_diagnostic_figs.py` docstring says "three" figures but makes four (verified live); the notebook overwrites committed figures; `render_grid` defaults to the `Loss` axis the docs warn against; `crossz_faithfulness.png` is stale.
8. **Net:** the repo is honest and reproducible at its core but over-promises its test bar and under-documents the library/rigor edges — 8 ordered fixes (P0 test → P1 h-basis+library+NaN → P2 boundary box/seeds/banners → P3 tidy) turn it into a clean new-user experience.
