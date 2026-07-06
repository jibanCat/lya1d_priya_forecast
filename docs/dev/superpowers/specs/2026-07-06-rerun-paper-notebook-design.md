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
  `label: str`, `out_root: Path`.
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
2. **Setup / emulator detect** — locate repo, try to import the GP; if absent, print the
   `REPRODUCE.md` Tier-2/3 setup command and stop gracefully (never error).
3. **The config cell** — a single `RerunConfig`; `QUICK = True` by default, flip to
   `FULL`; all knobs visible with inline comments; the physics-override dicts empty by
   default with an example commented out.
4. **What the pipeline does** — plain-language walkthrough of one per-`(param, z)` fit →
   grad-faith gate → additive combine, worded to match the paper's Methods (Sec.
   normalization / algorithm / combine). Short; links the reader to the paper sections.
5. **Run (inline)** — `run_dir = run_grid(cfg)`; live progress; for each cell also print
   `cli_command_for(...)` so the reader sees the CLI equivalent.
6. **Full / cluster** — documented block: the `submit_paper_production.sh` invocation and
   the per-arm CLI for the seed-band + sensitivity arms (not run inline).
7. **Regenerate the paper outputs from *your* run** — set the figure/table path to
   `run_dir` and call the `reproduce_paper` Tier-1 figure/table functions; render the
   taxonomy table + the pareto/scorecard/ns-budget figures from the collaborator's CSVs.
8. **Knobs to try (hypotheses)** — 3–4 concrete, self-contained experiments with the one
   line to change and what to watch: raise `sobolev_lambda`; drop an operator (e.g. `exp`);
   shift a fiducial value; widen a prior. Each notes the expected direction of the effect.

### 4. Tests + docs

- `tests/test_rerun.py` — GP mocked (a stub `gp_lf`/`gp_hf` with a known analytic response)
  so CI stays emulator-free: `RerunConfig.quick()/full()` shape; `run_grid` writes the
  expected layout + manifest; `with_overrides` replaces fid/prior and rejects unknown keys;
  `refit_one_param_single_z(params=…)` respects the override; `cli_command_for` round-trips
  the knobs. Aim: no new heavy deps in CI.
- `REPRODUCE.md` + `README.md` — a short Tier-3 pointer: "to re-run and tweak, see
  `notebooks/rerun_paper.ipynb`."

## Data flow

```
RerunConfig (quick|full + knobs + overrides)
   │
   ├─ with_overrides(fid, prior)?  ─► modified PARAMS list
   ▼
run_grid ──loop (arm,z,param)──► refit_one_param_single_z(params=…) ──► pareto_<p>.csv
   │                                     │
   │                                     └─ grad-faith scoring ──► grad_faith_<p>.csv
   ▼
results/rerun_<label>/<arm>/refit/z<z>/…  (+ RUN_MANIFEST.md)   [production layout]
   ▼
reproduce_paper figure/table functions (path → run_dir) ──► taxonomy table + diagnostic figs
```

## Error handling

- **No emulator:** detect at setup; print the exact Tier-2/3 command; the run cells
  short-circuit with a clear message. The notebook always completes.
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

None blocking. Recon settles the `params=` seam; if it is unexpectedly invasive, fall back
to the documented monkeypatch (noted in §2).
