# Fresh-eyes report — role: grad_extend (a grad student using the API to do something new)

**Date:** 2026-06-09
**Branch:** `stage10-multiz-sobolev`
**Vantage:** genuine fresh clone, never-seen-this-repo. Worked from README + docstrings only.
**Task:** standalone script that loads one parameter's Pareto CSV + grad-faith sidecar,
prints the *value-optimal* equation's `grad_err` and `value_mse`, and renders a single-panel
figure with `priya_forecast.pareto_diag`. Did the docs (README + docstrings) suffice to discover
`load_front` / `render_grid` / the sidecar columns / the gate value?

Deliverable script: `/tmp/fe/my_probe.py` (runs clean; output below). Figure: `/tmp/fe/my_probe.png`.

---

## Setup notes (clone fairness)

The clone command in the task brief fails on this filesystem:

```
$ git clone -q /home/mfho/lya1d_priya_forecast /tmp/fe
fatal: failed to copy file to '/tmp/fe/.git/objects/18/48a41a...': No such file or directory
```

A plain local clone tries to **hardlink/copy** `.git/objects` and dies on a stale/missing
loose object (the source has a dangling blob; `git fsck` is otherwise clean). Workaround that
gives a correct, fully-populated checkout:

```
rm -rf /tmp/fe && git clone -q file:///home/mfho/lya1d_priya_forecast /tmp/fe \
  && cd /tmp/fe && git checkout -q stage10-multiz-sobolev
```

`file://` forces a real pack transfer instead of object copying. **First successful-looking
`ls` was a lie**: the failed clone left a half-populated tree (`git status` → "No commits yet"
on `main`) that still listed all the top-level files momentarily. Net: a naive `git clone <path>`
of this repo does NOT reliably produce a working tree — worth a README note for anyone cloning
locally on this HPC.

Confirmed a clone has only tracked files: `data/` contains just `priya_fiducial/.gitkeep`
(no `kodiaq_gp/`), no `.venv`. Emulator-free path only, as expected.

---

## WHAT WORKED

- **The headline figure reproducer runs from a clone exactly as documented** (README §"Regenerate
  the diagnostic figures"). `PYTHONPATH=src python scripts/make_diagnostic_figs.py --out-dir
  /tmp/fe/scratch_figs` → `wrote 4 figures (png+pdf)`, exit 0, no GP/PySR/Julia. The
  "emulator-free" claim holds. Good.
- **My task succeeded end-to-end.** `/tmp/fe/my_probe.py` loads the front, prints the
  value-optimal `grad_err`/`value_mse`, renders a single panel:
  ```
  param          : dtau0
  value-optimal complexity : 20  (Loss=0.08221)
  grad_err       : 0.2143
  value_mse      : 5.799e-05
  gate           : 0.25  -> PASS
  ```
  `grad_err=0.214` matches the walkthrough table (dtau0 value best-loss = 0.214). Cross-checked
  `ns` → 0.603 (FAIL), also matches.
- **Conceptual ingredients WERE discoverable from prose** (`docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`,
  the doc the README points to as source-of-truth):
  - sidecar columns `grad_err` and `value_mse` are named + defined (lines 8, 42, 52-54);
  - the **0.25 gate** is stated repeatedly (line 44 etc.);
  - "**value-optimal = lowest loss**" is defined verbatim (line 102).
- **The run-dir → series mapping is discoverable** by reading `scripts/make_diagnostic_figs.py`
  (lines 32-34: `VALUE=stage6_log`, `SOBOLEV=stage9`, `BUDGET=decider_budget_z3.6`), which the
  README points to. Its line 38 (`read_grad_faith_sidecar(...).sort_values("Loss")`) is also the
  canonical "value-optimal" recipe.

**But the public API of `pareto_diag` (`load_front`, `render_grid`, the `fronts_by_param` shape)
is NOT in the README at all.** I only learned the function names + call signature by opening
`src/priya_forecast/pareto_diag.py` and reading its (good) docstrings. The README's layout block
points at the *file* (`pareto_diag.py`) but never names a function. The only prose mention of
`load_front`/`render_grid` anywhere is one bare line in `HANDOFF.md:46` with no signatures. So:
**from the README alone I could not have written this script — I needed the module docstrings.**
The docstrings, once found, were sufficient (the `fronts_by_param` dict shape is spelled out).

---

## MISSING

### M1 (top). No "use the module as a library" entry point / API doc.
A user wanting to do something *new* with the diagnostic (not just re-run the 4 canned figures)
has no documented path to `load_front`/`render_grid`. They are absent from README and the
walkthrough; `HANDOFF.md:46` lists the names but no signatures, args, or the `fronts_by_param`
schema. I had to read source docstrings to proceed.
**Fix:** add a short "Use it as a library" subsection to `README.md` (after the figure-reproducer
section) showing exactly the minimal call:
```python
from priya_forecast.pareto_diag import load_front, render_grid, GATE_TOL
front = load_front("results/single_z_stage6_log/refit/z3.6/pareto_dtau0.csv",
                   "results/single_z_stage6_log/refit/z3.6/grad_faith_dtau0.csv")
# value-optimal = lowest Loss AMONG sidecar-scored rows (see M2):
best = front.dropna(subset=["grad_err"]).sort_values("Loss").iloc[0]
render_grid({"dtau0": [{"front": front, "label": "value@20", "marker": "o"}]},
            "panel.png", y_col="value_mse", ncol=1)
```

### M2 (the one that actually bit me). "value-optimal" is ambiguous against `load_front`'s output.
The sidecar only scores **Fisher-safe** candidates, so the global-min-`Loss` Pareto row can have
**no `grad_err` (NaN)**. The naive `front.loc[front["Loss"].idxmin()]` therefore prints **NaN**
for some params. Proven for `ns`:
```
naive idxmin(Loss):        complexity 19  grad_err = nan
min-Loss among sidecar rows: complexity 16  grad_err = 0.603   (matches walkthrough)
```
My first draft used `dtau0`, where the min-Loss row happens to be Fisher-safe, so the bug was
invisible. Nothing in README/docstrings warns that "value-optimal" means *lowest-loss among
sidecar-scored rows*. The canonical script sidesteps it by sorting the sidecar alone, but never
says why.
**Fix:** (a) `pareto_diag.load_front` docstring — add a sentence: "lower-complexity rows have
NaN grad_err/value_mse (not Fisher-safe / unscored); pick the value-optimal equation from rows
with a sidecar score, e.g. `df.dropna(subset=['grad_err']).sort_values('Loss')`." (b) Walkthrough
line 102 — append "(among Fisher-safe candidates; non-scored complexities carry no derivative)."

### M3. README says GP data "must be present under `data/kodiaq_gp/`" but never says how to get it.
`README.md:92-94` states the requirement; the only thing that names the populating tool is a
runtime *error message* (`Run scripts/prep_kodiaq_gp.py --source <SRC> --dest data/kodiaq_gp`).
`prep_kodiaq_gp.py` exists in the clone but is mentioned in **zero** prose docs.
**Fix:** in README §"GP data", add: "Populate it with `python scripts/prep_kodiaq_gp.py --source
<SRC> --dest data/kodiaq_gp` (SRC = the shared KODIAQ-SQUAD GP export)."

---

## MISLEADING

### X1 (top). Documented test outcome is wrong on a fresh clone — it's a hard FAIL, not a skip.
`README.md:111-113`: *"(412 pass, ~13 skip. `test_real_gp_predicts_at_fiducial` is
environment-dependent: it **skips** when the upstream emulator is absent...)"* — framing that one
test as the only env-dependent one. Reality on a clean clone:
```
1 failed, 411 passed, 14 skipped
FAILED tests/test_single_z_pipeline.py::test_shipped_example_yaml_loads_and_validates
  ValueError: gp.basedir does not exist: data/kodiaq_gp.
```
This test is named/docstringed as an emulator-free YAML round-trip ("Shipped example config must
round-trip — students copy from it"), but `load_config()` calls `cfg.validate()`, which hard-fails
when the gitignored `data/kodiaq_gp` is absent. So the README's "quickest sanity check on an
install" does NOT pass on the clone it's describing.
**Fix:** either (a) make `test_shipped_example_yaml_loads_and_validates` skip when
`data/kodiaq_gp` is absent (it's testing YAML parsing, not GP data — gate the `validate()` part or
`pytest.skip` on missing basedir), and/or (b) correct README:111 to "411 pass, 1 fail (the
shipped-config test needs `data/kodiaq_gp`), 14 skip" until fixed. Option (a) is better — a config
round-trip test should not require gitignored data.

### X2. `render_grid`'s default y-axis contradicts the walkthrough's central instruction.
`render_grid(..., y_col="Loss", ...)` defaults to `Loss`, but the walkthrough spends a whole boxed
note (lines 56-64) explaining why you must NOT plot `Loss` — it's "the *training objective, which
differs by run*" and "would make Sobolev look like it fits values worse purely by construction" —
and that you should plot `value_mse`. A new user calling `render_grid` with defaults gets exactly
the incomparable axis the docs warn against.
**Fix:** change the default to `y_col="value_mse"` in `pareto_diag.render_grid` (and update the
docstring), or add a one-line warning in the docstring: "Default 'Loss' is the raw training loss
and is NOT cross-run comparable; pass y_col='value_mse' for the figure in the walkthrough."

### X3. The task-brief clone command is itself misleading on this box (see Setup).
`git clone <local-path>` produces a broken half-tree; only `file://` works. Documented above so the
next fresh-eyes run doesn't lose 10 minutes to a phantom "successful" `ls`.

---

## Did the docs suffice? (direct answer to the task)

- `load_front` / `render_grid` / `fronts_by_param` shape: **NO from README; YES from docstrings.**
  Had to open `src/priya_forecast/pareto_diag.py`. Once open, its docstrings were complete and
  correct.
- sidecar columns (`grad_err`, `value_mse`): **YES** — named + defined in the walkthrough and in
  `grad_faith_io.py`'s module docstring.
- gate value (0.25): **YES** — walkthrough prose, and `GATE_TOL = 0.25` in source.
- **Surprising / undocumented:** "value-optimal" silently yields NaN for `ns` (M2) — the single
  thing that would break a naive implementation, and it's nowhere in the docs.
