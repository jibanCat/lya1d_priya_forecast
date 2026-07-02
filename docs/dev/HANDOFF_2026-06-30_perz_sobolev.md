# HANDOFF — per-z Sobolev "production mode" + paper reframe (2026-06-30)

**Branch (code):** `stage10-multiz-sobolev` · **Paper repo:** `/home/mfho/Latex/Knowledge-Distillation-using-PySR-with-PRIYA-suite` (uncommitted by policy — edits are on disk).
**Compute node:** the OnDemand session (gl3083) ends ~23:00 on 2026-06-30; the shared filesystem (commits, results/, paper edits) persists. **SLURM allocation `yueyingn0` expires 2026-07-01.**
**No pending compute jobs** — all fits + GP re-scoring finished (see "Production run").

## 2026-06-30 ~21:45 FINAL UPDATE — paper review folded + everything pushed
- **Code repo pushed** → `origin/stage10-multiz-sobolev` HEAD **`061ddc4`** (PR #6). Includes: all Phase A–D + rework commits, the re-derived artifacts, usetex/large-font figure scripts, and the committed prediction-fig pickles (repro must-fix).
- **Paper repo pushed** → `origin/paper-additions` HEAD **`5e8a43d`** (github.com/jibanCat/Knowledge-Distillation-using-PySR-with-PRIYA-suite). The reframe + the author's figure-by-figure review feedback are folded in, ALL wrapped in `\additions{}`/`\mfho{}`; superseded student prose is `\begin{comment}`-blocked, never reworded. latexmk clean, 16 pp, no undefined refs.
- **2-agent cross-check PASSED:** all 12 feedback items APPLIED, no dangling refs, no reworded student prose, build exit 0; every table/figure number verified to MATCH the committed data; figures confirmed large-LaTeX-font. It caught 2 errors in the reframe `\additions` prose (fixed in `5e8a43d`: "six"→"five" value-faithful since z_Hi is Sobolev-rescued; ns budget "three of five"→"two of five" seeds) + `\mfho`-flagged 2 reframe leftovers (stale student 1D-accuracy numbers L627; retracted σ-ratio still live at L252/L791).
- **Author review applied (12 items):** Fig 3/4 widened; Figs 2,5,10,11 + Tables 4,5 + the ANOVA appendix + the joint-multiparam appendix dropped (restorable comment blocks); Table 7 equations rendered in LaTeX (raw sympy kept in `% sympy:` comments); Fisher-forecast presentation removed (§2.3/§4.4 renamed "Derivative faithfulness"; `eq:fisher_singlez` commented; `grad_err`+Sobolev method + the Fisher's-Mirage motivation KEPT); all figures regenerated with `text.usetex` + large fonts.
- **14 `\mfho` author-decision points remain in the paper** (the author's to-do): Fisher-motivation-entirely? · `eq:norm` empirical-vs-at-fid anchor · log-space combine · abstract + conclusion reframe (drafted) · Table 1 priors vs hypercube · Fig 2 holdout keep/refresh · restore maxsize_sens figure? · compact vs c=20 τ0/Ap forms.
- A 2-agent cross-check is verifying feedback fidelity + numbers-match-data + figure legibility (findings to be appended).
- **Repo cleanup DONE** (`5b6d82b`, pushed): removed 150 stale/superseded tracked `results/` files (pysr_hypothesis, `*_ksdata`, published_scorecard, refit_multid_*, holdout_multid_*, closure_*, smoke_*, single_z_stage8, multi_z_stage7, h_basis_test, simdat_ind15_truth) — none in the repro path; recoverable from git history. Tracked `results/` is now the 11 necessary dirs (production, seed_band, single_z_stage{6_log,9}, single_z_stage_pareto_diag, single_z_z{2.6,4.2}_*, decider_budget, refit_phase2_production). Deeper cleanup of stale *scripts/configs* (stage8 config, `scripts/smoke/`, closure scripts, `docs/housecleaning/`) left for a careful pass. Local UNTRACKED stale dirs (single_z_stage1/2/3, multi_z_stage10, _tutorial_scratch) don't reach a clone; left in place.
- **Reproducibility certified:** Tier-1 (emulator-free: diagnostic + maxsize_sensitivity figs) regenerate cleanly from the committed CSVs; Tier-2 (GP-backed multid + prediction figs) validated via the notebook + regen scripts.
- **Reproducibility package (`872db3a`, pushed):** rewrote `REPRODUCE.md` into a 3-tier step-by-step tutorial (env install from scratch → every figure + all 7 tables → optional emulator/refits), added `requirements-figures.txt` (7 pinned light deps), and extended the notebook to recompute Table 1/6/7 inline. **Verified in a fresh throwaway venv** (only the light deps, no GPy/pysr/juliacall/lyaemu): Table 6 taxonomy (ns Sobolev 0.160), Table 7 equations, Tables 1/2, and the 6 Tier-1 figures reproduce. Documented data provenance (`data/kodiaq_gp` is gitignored/43 MB — build via `scripts/prep_kodiaq_gp.py --source <MF GP basedir of Ho+2025>`; emulator = github.com/sbird/lya_emulator) + known gotchas (diagnostic figs need TeX; `regen_table2`/Table 3 has a payload-schema mismatch → `figures/table2_stats.tex` is the artifact of record).
- **PR #6 description updated** to the current state (title "Per-z Sobolev production + Fisher's-Mirage reframe…"). No paper-repo PR opened yet (marked-up WIP with ~45 open `\mfho` author decisions) — paper is on the pushed `paper-additions` branch.
- **Referee-panel reread** (publication-readiness) launched on the final paper; verdict to be appended.

## What this session did
Drove the project into "production mode" (one PySR model per param per z, Sobolev/gradient loss, no ANOVA) and, when the multi-lens review found the production data **refuted a headline claim**, executed an honest reframe + fixed three correctness bugs + re-derived everything + updated the paper and a reproducibility notebook.

## The scientific reframe (READ THIS)
The production **maxsize-sensitivity arm refuted** the draft's central claim that *"the n_S Fisher-Mirage is generative, not search-starvation."* At maxsize=35 the value-loss n_S largely clears the gate (MIXED, 3/5 seeds). Corrected framing (user-approved):
> At a fixed search budget, value-loss SR exhibits Fisher's Mirage (value-accurate, derivative-unfaithful); the **Sobolev loss reaches faithfulness at far lower complexity and more reliably than scaling the search budget**. Deeper budget (maxsize 30–40) does eventually improve value-loss n_S/Ω₀h² but is seed-fragile.

## Final numbers (fixed log-space gate + Pareto-knee selection; z=3.6)
- **Taxonomy (value@20 / sobolev@20, knee grad_err):** dtau0 .009/.003 · tau0 .006/.005 · ns .365/.160 · Ap .298/.155 · herei .079/.084 · heref .105/.045 · alphaq .126/.134 · omegamh2 .697/.063 · hireionz .266/.167 → **9/11 FAITHFUL**; **hub .99/1.00 + bhfeedback 1.42/0.77 RESIST** (budget-insensitive).
- **maxsize sweep (value | sobolev, ms 20/30/35/40):** ns 0.365/0.335/0.149/0.104 | 0.160/0.126/–/0.119 ; omegamh2 0.697/0.113/0.270/0.121 | 0.063/…; hub ~1.0 flat both; bhfeedback ~1.3→0.8 value / ~0.77→0.75 sob.
- **Seed band (median value/sobolev):** ns 0.519/0.212 · omegamh2 0.241/0.096 · herei 0.149/0.081 · alphaq 0.517/0.110 · hub 0.991/0.996 · bhfeedback 1.275/0.778. **ns_budget35 MIXED** median 0.267 [0.059,0.313].
- **Multi-D best/worst (combine vs GP, 256 Sobol):** 2D best tau0-ns 0.34% / worst Ap-ns 0.88%; 3D best tau0-Ap-ns 0.79% / worst Ap-ns-omegamh2 0.92% (all mean <1%).

## Commits (code repo, this session)
- `fc914fc` Phase A — per-z Sobolev plumbing (decouple loss/operators; Sobolev⇄log guard; submit driver).
- `6e69bc2` Phase A.1 — apply Phase-A multi-lens review fixes + self-contained --save-artifacts.
- `bdd74b2` Phase C rework — **fix log-space gate** (dlogP ratio), **Pareto-knee selection**, driver afterany.
- `f9a8e65` Phase C rework — diagnostic + seed-band score the knee equation.
- `f05e8de` Phase C — commit **re-derived production artifacts** (380 files: pareto/grad_faith CSVs, manifest, table fragments) + `scripts/regen_maxsize_sensitivity.py`. **← use this hash for paper %ref provenance.**
- `1f8fbbc` Phase D — reproducibility notebook + REPRODUCE.md.
- `37b718b` Phase C — enlarge render_grid fonts.

## Production run
`results/paper_production_20260630_perz_sobolev_z2.6-4.2/` (RUN_MANIFEST.md has every job id). Layout: `{sobolev,value}/refit/z{2.6,3.6,4.2}/`, `seed_band/z3.6_seed{0..4}_{value,sobolev,budget}/`, `budget35_{value,sobolev}/refit/z3.6/` (maxsize=35 control), `sens_maxsize{30,40}_{value,sobolev}/refit/z3.6/` (sensitivity sweep). `figures/` holds every regenerated fig + table fragment. Recipe: maxsize=20, populations=48, niter=200, λ=5, log target, no ANOVA; sobolev z3.6 also has `refits/`+`payloads/` pkls (--save-artifacts) for the prediction figs.

## Phase status
- **A/A.1/B/C/D: DONE + committed.** Full test suite 428 passed. Two 5-lens reviews run (Phase A ship_after_mustfix, Phase C rework) — both dispositioned.
- **E (paper sweep): IN PROGRESS** in the paper repo (was building via latexmk at 20:43). Executed by an agent from the brief `…/scratchpad/PHASE_E_BRIEF.md` (final numbers + ordered edits) + the full blueprint. **VERIFY on resume** (see below). All text changes wrapped in `\additions{}`/`\mfho{}`; superseded student prose commented (not deleted).
- **PENDING:** (1) review Phase E's `.tex` diff + run a 4-lens paper-review; (2) **repo cleanup** (prune stale `results/` dirs — see below); (3) **Phase F** final cross-validation (numbers in tex match CSVs, notebook runs, PDF clean).

## How to resume
Env: `export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia; export PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full`; python=`.venv/bin/python`. Cluster: `slurm/single_z_refit.slurm` via `scripts/submit_paper_production.sh` (SLURM_ACCOUNT/LYA_EMULATOR overridable).
- **Phase E verify:** open the paper PDF; check every `\additions{}`/`\mfho{}`; confirm Figs 10–17 gone + no dangling `\ref`s; confirm eq:grad_err is the log-derivative ratio; confirm ANOVA fully removed; confirm the ns/budget reframe prose + the maxsize_sensitivity + multid_bestworst figures inserted; confirm tables regenerated from the fixed-gate data.
- **Paper-review:** re-run the saved review workflow `…/workflows/scripts/stage-review-wf_8067282c-0bd.js` with `{stage, context, focus, files}` args pointing at the paper.
- **Repo cleanup (DO CAREFULLY — some old dirs are still referenced):** stale/removable per the recon: `results/single_z_stage1/2/3/3_subset`, `single_z_stage8`, `multi_z_stage10`, `published_scorecard`, and the May `refit_*/holdout_*/closure_*/smoke_*/pysr_hypothesis/h_basis_test/single_z_run/example` dirs. **KEEP:** `results/paper_production_20260630_…` (production), `results/refit_phase2_production` (regen_fig1 hard-codes it), `single_z_stage6_log`+`single_z_stage9` (make_diagnostic_figs DEFAULTS point there — production uses explicit `--*-dir` so defaults could be repointed), `single_z_stage_pareto_diag`, `seed_band`.
- **Phase F:** cross-check numbers in the tex vs the committed CSVs; run the notebook; latexmk clean; then finishing-a-development-branch (PR #6 is open against main).

## Known caveats / user `\mfho` decisions (from the Phase-E blueprint)
- `eq:norm` anchor: production (single-z) uses the **empirical sweep-mean log anchor**, but the paper eq describes the **at-fid LF anchor** (that's the multi-z path). Reconcile — flagged in tex.
- Combine is **log-space** (refit_taylor exponentiates a sum of log-deltas); paper Eqs 7–9 are linear-additive — flagged.
- τ0/Ap recovered equations are near the maxsize=20 ceiling (truncated fronts) → complex; a simpler faithful front row may read better — flagged.
- Table 1 priors differ from the production hypercube (zHei 4.1 vs 4.5, zHef 2.6 vs 2.2, A_P units) — flagged.
- Fig 2 (holdout_validation) has **no source in this run** — keep-or-refresh is a user call.
- Removing Figs 10–17 supersedes the student §4 multi-D narrative — commented with `\mfho`, needs the user's cut/keep call.
- Conclusion/abstract still tell the pre-reframe accuracy story — `\mfho` + draft `\additions{}` for the user to finalize.
- `regen_fig1.py` (paper repo) has no CLI (hard-codes `results/refit_phase2_production`) — REPRODUCE.md documents a symlink; a `--refit-dir` flag would be cleaner.
