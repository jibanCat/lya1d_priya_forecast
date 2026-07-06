# Design — `rerun_paper.ipynb`: re-run and tweak the PySR pipeline

**Date:** 2026-07-06
**Branch:** `rerun-notebook` (off `paper-production`)
**Status:** design approved (awaiting written-spec review)

## Purpose

A **Tier-3** tutorial notebook for a collaborator who already has the GP emulator
installed. It lets them re-run the *exact* per-`(param, z)` PySR pipeline that produced
the paper, tweak knobs and hypotheses, write the output to a **fresh results folder**,
and then regenerate the paper's figures/tables from *their* run via the existing
`reproduce_paper.ipynb`.

`reproduce_paper.ipynb` answers "does the paper reproduce?" (from committed CSVs,
emulator-free). `rerun_paper.ipynb` answers "what happens if I change the pipeline?" —
it re-derives the CSVs from scratch so a collaborator can test their own hypotheses.

### Non-goals (YAGNI)

- Not a from-scratch install guide for the GP emulator / Julia / PySR — that stays in
  `REPRODUCE.md` Tier 2/3. The notebook detects the emulator and points there if missing.
- Not a cluster/SLURM driver. Full mode runs locally/sequentially; the seed-band and
  maxsize-sensitivity arms are *documented as the `submit_paper_production.sh` CLI*, not
  executed inline.
- No new science. It reuses the production pipeline unchanged; the only code touch is a
  backward-compatible override hook.

## Constraints (from the existing repo)

- The atomic production unit is one PySR fit per `(param, z)`:
  `scripts/refit_one_param_single_z.py` → `priya_forecast.single_z.refit.refit_one_param_single_z`.
  Production recipe: `--target-space log --use-sobolev --sobolev-lambda 5 --maxsize 20
  --populations 48 --niterations 200`; the value baseline is the same budget with plain
  MSE (no `--use-sobolev`).
- Re-fitting **requires the GP emulator** (`data/kodiaq_gp` + `GPy`/`lyaemu`) — inherently
  Tier 3. Quick mode only shrinks the search *budget*; it cannot remove the emulator need.
- Fiducial values and prior bounds come from `priya_forecast.parameters.PARAMS_11D`
  (`pp.fid` and the prior min/max). The refit reads them internally; the CLI exposes no
  fiducial/prior flags today.
- Diagnostic figures/tables read a run dir with a fixed layout
  (`<arm>/refit/z<z>/{pareto_<p>.csv, grad_faith_<p>.csv}`). Any run dir with that layout
  is drop-in for the figure code.
- Notebooks in this repo are **generated from a builder** (`notebooks/_build_reproduce_paper.py`,
  gitignored; the `.ipynb` is committed). Match that convention.
- Writing style: `~/Latex/writing.md` (match the kodiaq_emu voice) — American spelling,
  no hyperbole, additions not rewrites.

## Decisions (locked with the user, 2026-07-06)

| Axis | Decision |
|---|---|
| Quick-mode scope | All 11 params @ z=3.6, both `value` + `sobolev` arms, reduced budget (~10–20 min on a laptop). |
| Full-mode scope | 11 params × {2.6, 3.6, 4.2} × {value, sobolev} (the Table 6 headline taxonomy), local/sequential. Seed-band + sensitivity = documented cluster CLI only. |
| Knob depth | Search knobs (loss/λ, maxsize, niter, populations, operators, seed, param/z) **plus** optional fiducial / prior physics overrides. |
| Exec engine | Python API (call the same refit function the CLI calls) **plus** a printed CLI mirror for each run. |
| Location | New branch `rerun-notebook` off `paper-production`; merge by PR after review. |
| Build | `~4`-agent panel + adversarial verify (see Implementation). |

### Refinements (2026-07-06 review — folded in)

1. **Never contaminate production.** All rerun output goes under a dedicated
   `results/tutorial_reruns/` root, isolated from `results/paper_production_*` and
   `results/refit_phase2_production`. `run_grid` hard-refuses (raises) if the resolved
   output path lands inside any production dir. This root is already git-ignored by the
   `results/` whitelist, so tutorial runs can never be committed over tracked artifacts.
2. **Concrete emulator provisioning.** The notebook must spell out exactly how a
   collaborator obtains the GP emulator, not hand-wave to "Tier 2." See §5.
3. **Validation guards (warn, never fail).** After a rerun, print a per-parameter
   deviation report vs the committed production sidecars, plus an under-budget
   disclaimer. See §1 (`compare_to_production`) and §6.

## Architecture

Four new artifacts; the production pipeline is reused unchanged except for one
backward-compatible hook.

### 1. `src/priya_forecast/rerun.py` (new module — the logic)

Keeps orchestration out of notebook JSON so it is unit-testable and the notebook stays
thin.

- **`RerunConfig`** dataclass — every knob with production defaults:
  `params: list[str]`, `zs: list[float]`, `arms: list[str]` (`"value"`, `"sobolev"`),
  `maxsize`, `niterations`, `populations`, `sobolev_lambda`, `binary_operators`,
  `unary_operators`, `seed`, `kmin`, `kmax`,
  `fiducial_overrides: dict[str, float] | None`, `prior_overrides: dict[str, tuple[float,float]] | None`,
  `label: str`, `out_root: Path = Path("results/tutorial_reruns")`.
  - **Isolation:** the effective run dir is `out_root/rerun_<label>/`. `run_grid` **raises**
    if the resolved run dir falls inside `results/paper_production_*` or
    `results/refit_phase2_production`, so tutorial output can never overwrite a production
    run. `results/tutorial_reruns/` is git-ignored by the existing `results/` whitelist, so
    nothing here is committed.
  - `RerunConfig.quick()` → 11 params, `zs=[3.6]`, both arms, `niterations≈30`,
    `populations≈8`, `maxsize=20`. Target wall time a few→~20 min on a laptop.
  - `RerunConfig.full()` → 11 params, `zs=[2.6, 3.6, 4.2]`, both arms, production budget
    (`niterations=200`, `populations=48`, `maxsize=20`, `sobolev_lambda=5`).
- **`run_grid(cfg, gp_lf=None, gp_hf=None, progress=print) -> Path`** — loops
  `(arm, z, param)`, calls `refit_one_param_single_z(...)` (loading the GP once per z and
  reusing it across params/arms), then runs the grad-faith scoring so each fit lands its
  `grad_faith_<p>.csv` sidecar. Writes to `out_root/rerun_<label>/<arm>/refit/z<z>/…`
  (production layout) and a `RUN_MANIFEST.md` stamp (git hash, full knob set, any
  overrides, timestamp passed in by caller). Returns the run dir.
  - `value` arm ⇒ `use_sobolev=False`, `target_space="log"`, plain MSE.
  - `sobolev` arm ⇒ `use_sobolev=True`, `target_space="log"`, `sobolev_lambda=cfg.sobolev_lambda`.
- **`cli_command_for(cfg, param, z, arm) -> str`** — the exact
  `python scripts/refit_one_param_single_z.py …` command equivalent to one grid cell,
  for copy-paste to a cluster. (Physics overrides, if set, are shown as a note that they
  require the Python-API path, since the CLI has no such flag.)
- **Grad-faith scoring reuse:** call the same code path as
  `scripts/eval_grad_faithfulness.py` / `make_grad_faith_sidecars.sh` (log-space,
  gate 0.25) — factor a callable out of that script if it is not already importable.
- **`compare_to_production(run_dir, production_dir=<committed run>) -> Report`** — reads
  the rerun's `grad_faith_<p>.csv` knee rows and the committed production sidecars (Tier-1,
  emulator-free), and returns a per-parameter table: rerun vs production `grad_err` and
  `value_mse`, the signed delta, whether the faithfulness verdict flipped, and a
  worse/better/similar flag. **Print-only; never raises.** Only params/z present in both
  runs are compared (the rest are listed as "not in production baseline").
- **`budget_warnings(cfg) -> list[str]`** — flags any knob below the production budget
  (`niterations<200`, `populations<48`, `maxsize<20`, `sobolev_lambda≠5`,
  `arms`/`zs`/`params` subset). Returns human-readable warning strings for the notebook to
  print — again, warnings, not errors.

### 2. Backward-compatible override hook (the only production-code touch)

- `priya_forecast.parameters.with_overrides(fiducial=None, prior=None) -> list[Param]` —
  returns a copy of `PARAMS_11D` with the named `.fid` / prior bounds replaced; unknown
  keys raise. `PARAMS_11D` itself is untouched.
- Thread an **optional** `params=` argument into `refit_one_param_single_z` (and any
  helper it calls that reads `PARAMS_11D`), defaulting to `PARAMS_11D`. Every existing
  caller (the CLI, all SLURM jobs, all tests) is unaffected because the default preserves
  today's behavior. `rerun.run_grid` passes `with_overrides(...)` when overrides are set.
- Recon (panel step 0) confirms the exact seam; if threading proves invasive, the
  fallback is a documented, run-local monkeypatch inside `run_grid` — but the threaded
  `params=` argument is the target.

### 3. `notebooks/_build_rerun_paper.py` → `notebooks/rerun_paper.ipynb`

Builder mirrors `_build_reproduce_paper.py` (`md()` / `code()` helpers; builder
gitignored, `.ipynb` committed). Notebook flow:

1. **Title + tier banner** — "you need the GP emulator (Tier 3)"; one-paragraph contrast
   with `reproduce_paper.ipynb`.
2. **Get the GP emulator** — a dedicated, copy-pasteable provisioning cell (see §5): try
   to import the emulator; if absent, print the exact clone/install + `kodiaq_gp` fetch
   commands and stop gracefully (never error).
3. **The config cell** — a single `RerunConfig`; `QUICK = True` by default, flip to
   `FULL`; all knobs visible with inline comments; the physics-override dicts empty by
   default with an example commented out. Output goes to `results/tutorial_reruns/` — a
   markdown note states this is isolated from the production run.
4. **What the pipeline does** — plain-language walkthrough of one per-`(param, z)` fit →
   grad-faith gate → additive combine, worded to match the paper's Methods (Sec.
   normalization / algorithm / combine). Short; links the reader to the paper sections.
5. **Budget check (before running)** — print `budget_warnings(cfg)` + the standing
   disclaimer (§6): quick/tweaked runs are illustrative, ran on far less compute than the
   paper, and must not replace the production run without passing a trust budget.
6. **Run (inline)** — `run_dir = run_grid(cfg)`; live progress; for each cell also print
   `cli_command_for(...)` so the reader sees the CLI equivalent.
7. **Full / cluster** — documented block: the `submit_paper_production.sh` invocation and
   the per-arm CLI for the seed-band + sensitivity arms (not run inline).
8. **Compare to production** — print `compare_to_production(run_dir)`: the per-parameter
   deviation table (rerun vs committed production `grad_err`/`value_mse`, signed delta,
   verdict flips, worse/better/similar). Framed as "how far did I move," not pass/fail.
9. **Regenerate the paper outputs from *your* run** — set the figure/table path to
   `run_dir` and call the `reproduce_paper` Tier-1 figure/table functions; render the
   taxonomy table + the pareto/scorecard/ns-budget figures from the collaborator's CSVs
   into `run_dir` (never into the production `figures/` dir).
10. **Knobs to try (hypotheses)** — 3–4 concrete, self-contained experiments with the one
    line to change and what to watch: raise `sobolev_lambda`; drop an operator (e.g. `exp`);
    shift a fiducial value; widen a prior. Each notes the expected direction of the effect.

### 4. Tests + docs

- `tests/test_rerun.py` — GP mocked (a stub `gp_lf`/`gp_hf` with a known analytic response)
  so CI stays emulator-free: `RerunConfig.quick()/full()` shape; `run_grid` writes the
  expected layout + manifest; **`run_grid` raises when `out_root` resolves inside a
  production dir**; `with_overrides` replaces fid/prior and rejects unknown keys;
  `refit_one_param_single_z(params=…)` respects the override; `cli_command_for` round-trips
  the knobs; **`budget_warnings` flags a sub-production config and is silent on a full one**;
  **`compare_to_production` computes correct signed deltas and verdict-flip flags against a
  small fixture**. Aim: no new heavy deps in CI.
- `REPRODUCE.md` + `README.md` — a short Tier-3 pointer: "to re-run and tweak, see
  `notebooks/rerun_paper.ipynb`."

### 5. Emulator provisioning (the "how do I get the GP" story)

The notebook must make Tier-3 turnkey. Recon (panel step 0) pins the exact commands from
the repo; the target shape:

- **Package:** install the emulator Python package the pipeline imports (`lyaemu` / the
  `lya_emulator_full` repo). Document the precise `pip install …` / `git clone …` from
  the repo's own requirements, not a guess.
- **`data/kodiaq_gp/` basedir (~43 MB, git-ignored):** two documented routes, in order —
  1. **Build from source** (fully reproducible, no hosting): clone `lya_emulator_full`,
     then `python scripts/prep_kodiaq_gp.py …` to produce `data/kodiaq_gp/`. This is the
     ground-truth path and always works.
  2. **Fetch a prebuilt archive** (fast path for collaborators): a single
     `curl/​wget` of a hosted `kodiaq_gp.tar.gz` into `data/`. The **URL is a fill-in
     placeholder** the user sets once they upload the archive (Zenodo / GitHub release /
     shared drive — user's call). Clearly marked `# TODO(user): set archive URL`.
- The provisioning cell **auto-detects** which route is already satisfied and only prints
  what is missing. It never downloads silently without the reader running the cell.
- **Open item for the user:** where to host the prebuilt `kodiaq_gp` archive (see Open
  questions). Not blocking — route 1 makes the notebook complete without any hosting.

### 6. Validation guards & disclaimer (warn, never fail)

- **Standing disclaimer** (rendered prominently before and after the run): the production
  run used a far larger search budget (`niter=200`, `populations=48`, 5-seed band); a quick
  or tweaked run here is **illustrative**, shows the *mechanism*, and **must not replace the
  production numbers**. A run should clear a stated **trust budget** (e.g. production
  `niter`/`populations`, ≥1 seed per arm) before its numbers are quoted.
- **`budget_warnings(cfg)`** prints which knobs are below production (§1). Purely
  informational — the run still proceeds.
- **`compare_to_production(run_dir)`** prints the per-parameter deviation table (§1) so the
  reader sees, quantitatively, how far their run moved from the paper — worse, better, or
  within noise — without any assertion or gate. "Better" (lower `grad_err`) is reported as
  neutrally as "worse"; the point is transparency, not a leaderboard.

## Data flow

```
RerunConfig (quick|full + knobs + overrides)
   │
   ├─ with_overrides(fid, prior)?  ─► modified PARAMS list
   ▼
run_grid ──loop (arm,z,param)──► refit_one_param_single_z(params=…) ──► pareto_<p>.csv
   │  (out_root inside a production dir? ─► RAISE)   │
   │                                     └─ grad-faith scoring ──► grad_faith_<p>.csv
   ▼
results/tutorial_reruns/rerun_<label>/<arm>/refit/z<z>/…  (+ RUN_MANIFEST.md)  [production layout, git-ignored]
   ├─► compare_to_production(run_dir) ──► per-param deviation table (warn-only)
   ▼
reproduce_paper figure/table functions (path → run_dir) ──► taxonomy table + diagnostic figs (into run_dir)
```

## Error handling

- **No emulator:** detect at setup; print the exact provisioning commands (§5); the run
  cells short-circuit with a clear message. The notebook always completes.
- **Output would hit production:** `run_grid` raises if `out_root/rerun_<label>` resolves
  inside `results/paper_production_*` or `results/refit_phase2_production`. This is a hard
  error (protecting the paper artifacts is worth failing loudly) — the only raise in the
  run path besides config validation.
- **`use_sobolev` without `target_space="log"`:** already guarded in the refit; `RerunConfig`
  enforces the same invariant at construction so a bad config fails fast in-notebook.
- **Unknown override key:** `with_overrides` raises with the offending name.
- **A single fit raising** (e.g. a pathological operator set) is caught per cell, logged,
  and the grid continues, so one bad param does not kill the whole run.

## Testing strategy

TDD where it pays: `rerun.py` and `with_overrides` are pure/orchestration logic → unit
tests with a mocked GP. The notebook itself is validated by (a) the builder producing a
notebook that `nbconvert --execute` runs clean in **quick** mode *when* an emulator is
present (a manual/opt-in check, not CI), and (b) a static check that it imports and the
config cell constructs a valid `RerunConfig` without the emulator.

## Implementation (panel)

Terminal step of this design is `writing-plans` → an implementation plan; then a ~4-agent
panel executes it:

0. **Recon** — pin the exact `refit`/`parameters` seam for `params=` threading and the
   importable grad-faith scoring entry point.
1. **`rerun.py` + `with_overrides` + `tests/test_rerun.py`** (TDD).
2. **`_build_rerun_paper.py` + `rerun_paper.ipynb`** (depends on 1's API).
3. **Style + docs pass** — match `writing.md` voice on the notebook markdown; add the
   `REPRODUCE.md` / `README.md` pointers (depends on 2).
4. **Adversarial verify** — confirm layout matches production, figures render from a rerun
   dir, existing callers of the touched refit signature are unbroken, full test suite green.

## Open questions

- **Where to host the prebuilt `kodiaq_gp` archive** (fast-path route 2, §5) — Zenodo /
  GitHub release asset / shared drive. **User's call**; not blocking, because the
  build-from-source route (route 1) makes the notebook complete without any hosting. The
  notebook ships a clearly-marked `# TODO(user): set archive URL` placeholder.
- Recon settles the `params=` seam; if it is unexpectedly invasive, fall back to the
  documented monkeypatch (noted in §2).
