# Fresh-eyes report — role: grad_reproduce

**Goal:** Brand-new grad student, fresh clone, follow the README to reproduce the
four diagnostic figures. Run `scripts/make_diagnostic_figs.py`, validate + execute
`notebooks/reproduce_paper_figures.ipynb`, confirm the four figures match the
walkthrough, and check whether I can get a figure WITHOUT the emulator (as claimed).

**Setup actually used (a genuine clone — `data/` and `.venv` are gitignored):**
```
rm -rf /tmp/fe && git clone -q file:///home/mfho/lya1d_priya_forecast /tmp/fe
cd /tmp/fe && git checkout -q stage10-multiz-sobolev          # HEAD = 1449001
# emulator-free runs, against the CLONE's src, with the existing venv interpreter:
PYTHONPATH=src /home/mfho/lya1d_priya_forecast/.venv/bin/python <command>
```

---

## TL;DR verdict

The README's headline promise **holds**: I reproduced all four diagnostic figures
from a fresh clone, **emulator-free** (no GPy / PySR / Julia / `data/kodiaq_gp`
imported or read — verified by inspecting `sys.modules`). The script runs clean
(exit 0), the notebook validates (`nbformat` v4.5, 19 cells) and **executes
end-to-end with zero errors** via `nbclient`, self-confirming all four PNG+PDF.
Figures match the walkthrough byte-for-byte in size and visually in content.

But two things a real cloner hits are wrong or under-documented (details below): a
**default-suite test that is NOT emulator-free** (so the README's "412 pass" is
unreachable from a clone — you get 1 failure), and the **figure script's own
docstring says "three figures" while it makes four**.

---

## WHAT WORKED (and was it discoverable?)

1. **`scripts/make_diagnostic_figs.py` — fully discoverable, worked verbatim.**
   README §"Regenerate the diagnostic figures" gives the exact command. Ran:
   ```
   PYTHONPATH=src .../python scripts/make_diagnostic_figs.py --out-dir /tmp/fe_figs_scratch
   -> "wrote 4 figures (png+pdf) to /tmp/fe_figs_scratch"  (exit 0)
   ```
   All 4 PNGs are identical in size to the committed copies in
   `results/single_z_stage_pareto_diag/` (content matches; not byte-identical only
   because matplotlib PNGs carry nondeterministic metadata — not a real diff).
   All four input sidecar dirs the script needs are present in the clone
   (`single_z_stage6_log`, `single_z_stage9`, `decider_budget_z3.6`,
   `single_z_z2.6_sobolev`, `single_z_z4.2_sobolev`).

2. **Emulator-free claim is genuine.** Importing the script and scanning
   `sys.modules` for `lyaemu/GPy/pysr/emukit/juliacall/juliapkg` returns **NONE**.
   The README §"Prerequisites … (NOT needed for the figure reproducer)" is accurate.

3. **The four figures match the walkthrough.** Visual check of each PNG:
   - `pareto_faithfulness` — 11-panel grid, y=value MSE, RdYlGn color=grad_err,
     gate rings, `ns` Mirage arrow. ✓
   - `faithfulness_scorecard` — value@20 vs Sobolev@20 per param, gate 0.25 line,
     the two "resisters" (hub 0.94, bhfeedback 0.95) labelled. ✓
   - `ns_budget_panel` — budget(FAIL 0.319) vs Sobolev(PASS 0.193) endpoints. ✓
   - `crossz_faithfulness` — z=2.6/3.6/4.2 Sobolev grad_err, gate line. ✓
   `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` names all four figures.

4. **`notebooks/reproduce_paper_figures.ipynb` — validates AND executes.**
   - `nbformat.validate` → VALID v4.5, 19 cells.
   - `nbclient` IS in the venv (`nbclient 0.11.0`, `ipykernel 7.2.0`,
     `jupyter_client 8.9.1`), so I executed it headless. Every code cell ran,
     no errors; final cell printed "All four diagnostic figures reproduced
     EMULATOR-FREE …". Cell 2 robustly handles `__file__`-less execution and
     `os.chdir(REPO_ROOT)` so relative `results/...` paths resolve.

5. **`requirements.lock.txt` exists in the clone** (README §Installation references
   it by name — the reference is valid, file is tracked).

---

## MISSING / BROKEN parts

### M1 (blocking the README's test claim). A default-suite test is NOT emulator-free.
README §"Run the tests" says:
> "Emulator-free; this is the quickest sanity check on an install … (412 pass,
> ~13 skip.)"
Reality on a fresh clone:
```
PYTHONPATH=src python -m pytest tests/ -q -k "not slow"
-> 1 failed, 411 passed, 14 skipped
FAILED tests/test_single_z_pipeline.py::test_shipped_example_yaml_loads_and_validates
   ValueError: gp.basedir does not exist: data/kodiaq_gp. Run
   `python scripts/prep_kodiaq_gp.py --source <SRC> --dest data/kodiaq_gp`.
```
The test calls `load_config(configs/single_z/example.yaml)` → `cfg.validate()` →
`gp.validate()` (src/priya_forecast/single_z/config.py:112), which hard-fails when
`data/kodiaq_gp/` is absent. `data/` is gitignored, so it is ALWAYS absent in a
clone. So the README's "412 pass, emulator-free" was measured on the author's
machine (where `data/kodiaq_gp/` happens to exist) and is **unreachable from a
clone**. This is precisely the docs-don't-mark-the-emulator-boundary problem: a
test billed as emulator-free silently depends on emulator data.

**Fix (pick one):**
- (a) Make the test clone-safe: in
  `tests/test_single_z_pipeline.py::test_shipped_example_yaml_loads_and_validates`,
  skip the `gp.basedir` existence check — e.g. load with validation disabled, or
  `pytest.skip("needs data/kodiaq_gp")` when `Path("data/kodiaq_gp").exists()` is
  False. The test's intent ("YAML round-trips") doesn't need the GP data.
- (b) OR decouple path-existence from schema validation: have `GPConfig.validate()`
  gate the `basedir`-exists check behind a flag (e.g. `validate(require_data=False)`)
  and have config round-trip tests pass `require_data=False`.
- After fixing, update the README count line (§"Run the tests") to the true
  clone numbers, or change it to "all non-skipped tests pass" so the figure (which
  drifts as tests are added) isn't load-bearing.

### M2 (minor, documentation of `data/` layout). README "Repository layout" lists
`data/    kodiaq_gp/ (trained GP + KODIAQ-SQUAD inputs), priya_fiducial/, single_z_1pvar/`
as if present. A clone only has `data/priya_fiducial/`; `kodiaq_gp/` and
`single_z_1pvar/` are gitignored and absent. The README does flag elsewhere that GP
data is needed only for emulator runs, but the layout table reads as "here's what's
in the repo". **Fix:** annotate the layout entry, e.g.
`data/  priya_fiducial/ (tracked); kodiaq_gp/, single_z_1pvar/ (gitignored — created
by scripts/prep_kodiaq_gp.py, not in a clone)`.

---

## MISLEADING parts

### X1. `make_diagnostic_figs.py` docstring says "three" figures; it makes four.
`scripts/make_diagnostic_figs.py` line 1:
> "Regenerate the **three** diagnostic paper figures …"
and the docstring's numbered list stops at "3. ns_budget_panel". But the code
(and the README, correctly) produces **four**: it also writes `crossz_faithfulness`
(lines 147–172) and prints "wrote 4 figures". A new student reading the script
docstring will think one figure is missing. **Fix:** update the module docstring in
`scripts/make_diagnostic_figs.py` to "four" and add the 4th bullet
(`4. crossz_faithfulness -- redshift robustness, z=2.6/3.6/4.2`).

### X2. Notebook hardcodes the COMMITTED output dir → running it dirties the tree.
The README tells you to pass a scratch `--out-dir` "if you only want to inspect
without touching" the committed copies — but that advice is only wired for the
*script*. The notebook (cell 4) hardcodes
`OUT = Path("results/single_z_stage_pareto_diag")` (the committed dir). Executing
the reproducer notebook overwrites all 8 committed PNG/PDF in place; `git status`
afterward shows them all modified. A student "just reproducing" ends up with a
dirty working tree they may commit by accident. **Fix:** in
`notebooks/reproduce_paper_figures.ipynb` cell 4, default `OUT` to a scratch path
(e.g. `Path(os.environ.get("REPRO_OUT", "results/_repro_scratch"))`) and say so in
the intro markdown, OR add a one-line note in README §Notebooks that running the
notebook regenerates the committed figures in place.

### X3. (Environmental, not a doc bug — recorded for completeness.) The README's
`git clone <repo-url>` is fine, but the literal LOCAL-path clone
`git clone /home/mfho/lya1d_priya_forecast /tmp/fe` failed once with
`fatal: update_ref … nonexistent object 1449001868…` — a transient race because a
background/scheduled task in the live repo was mid-write on the HEAD object (note
`.claude/scheduled_tasks.lock`). A re-run succeeded, and `git clone file://…`
always worked. No README change needed; just be aware local hardlink clones of a
*live, being-written* repo can flake. If you hit it, retry or use `file://`.

---

## Bottom line for my task
- Did the README install/usage steps work as written? **Figure path: yes.** The
  figure script and the reproduce notebook both worked verbatim and emulator-free.
  **Test path: no** — one default test fails on a clone, contradicting the
  "412 pass, emulator-free" line (M1).
- Could I get a figure without the emulator, as claimed? **Yes — all four**,
  verified no emulator module loads.
- Four figures produced and match the walkthrough? **Yes** (script + notebook,
  both routes).
