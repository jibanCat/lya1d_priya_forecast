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
- **Grad-faith scoring reuse:** recon found no clean importable scorer (the logic is in
  `eval_grad_faithfulness.py:main()`). `run_grid` therefore **subprocesses the existing
  `scripts/eval_grad_faithfulness.py`** per `(param, z, arm)` — byte-identical to the paper
  run, zero refactor of paper-critical code — with `--out <run>/<arm>/refit/z<z>/grad_faith_<p>.csv`
  (exactly as `make_grad_faith_sidecars.sh` does), inheriting the notebook's env
  (`PYTHONPATH=src:$LYA_EMULATOR`). Only the clean sidecar readers
  (`grad_faith_io.read_grad_faith_sidecar`, `knee_row`) are imported.
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

### 2. Physics-override hook — context manager (revised after recon)

**Recon finding:** the fiducial/prior reads are *not* in `single_z/refit.py`; they live in
`refit_1d_pysr.py` across ~10 sites in two core functions (`refit_1d_for_param`,
`_generate_1pvar_inline`) that many other callers share. Threading a `params=` argument
through all of them would be invasive surgery on paper-critical code. **But** those helpers
(`get_param`, `fiducial_vector`) read the module global `parameters.PARAMS_11D` **at call
time**, so a context manager that temporarily rebinds that global flows overrides through
with **zero changes to `refit_1d_pysr.py`**. This is the spec's documented fallback,
promoted to the primary mechanism because recon confirmed the threaded route is invasive.

- `priya_forecast.parameters.with_overrides(fiducial=None, prior=None, base=PARAMS_11D) -> tuple[Param, ...]`
  — pure helper. Returns a copy of `base` with the named params' `.fid` / `.prior` replaced
  via `dataclasses.replace`; unknown names raise `KeyError`. `PARAMS_11D` is untouched.
  Names/order are preserved (only values change), so `PARAM_NAMES`, the `.index()` lookups
  and the 11-vector layout in `refit_1d_pysr.py` stay correct.
- `priya_forecast.parameters.override_params(fiducial=None, prior=None)` — a
  `@contextmanager` that sets `parameters.PARAMS_11D = with_overrides(...)` on entry and
  restores the original in a `finally`. Because `get_param()` / `fiducial_vector()` read the
  module global at call time, the temporarily-overridden fid/prior reach
  `_generate_1pvar_inline` and `refit_1d_for_param` unchanged. No-op (`nullcontext`-like)
  when both overrides are `None`.
- `rerun.run_grid` wraps each refit in `override_params(cfg.fiducial_overrides,
  cfg.prior_overrides)`. Safe for the sequential notebook: the override is only live during
  in-process 1pvar data generation (before PySR's Julia workers consume the arrays), then
  restored. No production caller is affected — the global is only ever swapped inside the
  `with` block.
- **Not touched:** `refit_1d_pysr.py`, the CLI, SLURM, existing tests.

**Post-implementation correction (2026-07-06, from adversarial review + recon):** the
subprocess scorer `eval_grad_faithfulness.py` (a) re-imports the *original* `PARAMS_11D`
so the parent's in-memory override was invisible to it, and (b) reads a git-ignored 1pvar
sweep cache (`data/single_z_1pvar`) that is **not shipped** to collaborators and is at the
original fiducial/prior. Both are fixed together:
- **`run_grid` regenerates the 1pvar sweep run-local, under `override_params`**, into
  `<run_dir>/_1pvar/` via the existing `single_z.training_data.regenerate_param` +
  `write_1pvar_hdf5` (which call `_generate_1pvar_inline` → `fiducial_vector()`/`get_param`
  at call time, so they honor the override). This also removes the hidden dependency on the
  unshipped cache and never touches `data/single_z_1pvar`.
- **`eval_grad_faithfulness.py` gains an env-gated override** (`PRIYA_FIDUCIAL_OVERRIDES` /
  `PRIYA_PRIOR_OVERRIDES`, JSON): its `fid` becomes a call-time `fiducial_vector()` read and
  the scoring body runs inside `override_params(...)`. With no env vars set the context is a
  no-op and the paper output is byte-identical. `run_grid`'s `score_fn` passes the override
  via env + points `--data-1pvar` at the run-local dir.
- **Second production-code touch (accepted by the user):** `eval_grad_faithfulness.py`
  (additive, env-gated, paper-preserving), plus the run-local regen wiring in `rerun.py`.
  `parameters.py` remains the other touched file.

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
- **`data/kodiaq_gp/` basedir (~43 MB, git-ignored):** two documented routes. **Recon
  caveat:** `prep_kodiaq_gp.py --source` reads the *private* upstream PRIYA training set
  (`/nfs/turbo/.../kodiaq_2_2_4_6-48-48`), so build-from-source only works for someone with
  access to that data. **For an external collaborator the hosted archive is the primary
  route.**
  1. **Fetch a prebuilt archive** (primary for collaborators): a single `curl`/`wget` of a
     hosted `kodiaq_gp.tar.gz` into `data/`. The **URL is a fill-in placeholder** the user
     sets once they upload the archive (Zenodo / GitHub release / shared drive — user's
     call). Clearly marked `# TODO(user): set archive URL`.
  2. **Build from source** (only with access to the private PRIYA training set): clone
     `lya_emulator` for the `lyaemu` package, then
     `python scripts/prep_kodiaq_gp.py --source <PRIYA_SET> --dest data/kodiaq_gp`.
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
