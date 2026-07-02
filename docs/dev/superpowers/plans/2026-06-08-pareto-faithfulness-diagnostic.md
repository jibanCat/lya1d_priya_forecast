# Per-parameter Pareto-faithfulness diagnostic — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the paper's central failure-modes figure — a per-parameter Pareto front (loss vs complexity) with markers colored by derivative faithfulness — plus the emulator-side eval that feeds it and a living walkthrough doc that explains why PySR mis-fits each parameter's values and derivatives.

**Architecture:** Split emulator-dependent computation from layout iteration. A pure IO module (`grad_faith_io.py`) defines the sidecar format; `eval_grad_faithfulness.py` (cluster, needs the GP) writes sidecars; a pure plotting module (`pareto_diag.py`, no emulator) reads Pareto CSVs + sidecars and renders the grid, degrading to gray when a sidecar is absent so the layout can be built before the cluster job lands. A walkthrough markdown embeds the figure inline and carries the per-parameter reasoning.

**Tech Stack:** Python, pandas, numpy, matplotlib (Agg), pytest. Reuses `priya_forecast.derivative_gate`, `priya_forecast.single_z.forecast`, `priya_forecast.models.pysr_model`. Spec: `docs/superpowers/specs/2026-06-08-pareto-faithfulness-diagnostic-design.md`.

**Param order (PARAM_NAMES):** `dtau0, tau0, ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback`.

**Phase map:** Tasks 1–4 are **local, no cluster** (produce the gray first-cut figure). Tasks 5–6 are **cluster** (sidecars + budget control → color figure). Task 7 is **analysis** (fill the walkthrough). Commit after every task.

---

### Task 1: `grad_faith_io.py` — the sidecar format (pure, emulator-free)

**Files:**
- Create: `src/priya_forecast/grad_faith_io.py`
- Test: `tests/test_grad_faith_io.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_grad_faith_io.py
from priya_forecast.grad_faith_io import (
    equation_has_x0, write_grad_faith_sidecar, read_grad_faith_sidecar,
    SIDECAR_COLUMNS,
)


def test_equation_has_x0_word_boundary():
    assert equation_has_x0("x0 * 2.6589415")
    assert equation_has_x0("log(x0 + 0.1623519)")
    assert not equation_has_x0("x1 * 3.0")
    assert not equation_has_x0("2.5")
    # must not match a longer feature name that merely starts with x0
    assert not equation_has_x0("x01 + 1.0")


def test_sidecar_roundtrip_preserves_columns_and_bool(tmp_path):
    rows = [
        {"Complexity": 1, "Loss": 24.636, "grad_err": 0.90,
         "n_keep": 40, "gate_pass": False, "x0_enters": True},
        {"Complexity": 3, "Loss": 10.020, "grad_err": 0.134,
         "n_keep": 40, "gate_pass": True, "x0_enters": True},
    ]
    out = write_grad_faith_sidecar(
        tmp_path / "grad_faith_ns.csv", rows,
        param="ns", z=3.6, tol=0.25, log_space=True,
        source_pareto="results/x/pareto_ns.csv",
    )
    df = read_grad_faith_sidecar(out)
    assert list(df.columns) == SIDECAR_COLUMNS
    # the leading "# param=..." comment line must be skipped, not parsed as data
    assert len(df) == 2
    # gate_pass must round-trip as a real boolean column
    assert df["gate_pass"].dtype == bool
    assert bool(df.loc[df.Complexity == 3, "gate_pass"].item()) is True
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_grad_faith_io.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'priya_forecast.grad_faith_io'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/priya_forecast/grad_faith_io.py
"""Read/write per-candidate gradient-faithfulness sidecars.

A sidecar pairs 1:1 with a PySR Pareto CSV and records, for every
Fisher-safe candidate, the derivative-faithfulness metric the production
gate uses (median_k |d_eq/d_theta / d_P_GP/d_theta - 1| at fid). Kept
emulator-free so the plotter can consume it without GPy/lyaemu.
"""
from __future__ import annotations

import re
from pathlib import Path

import pandas as pd

SIDECAR_COLUMNS = [
    "Complexity", "Loss", "grad_err", "n_keep", "gate_pass", "x0_enters",
]

_X0 = re.compile(r"\bx0\b")


def equation_has_x0(equation_str: str) -> bool:
    """True if the PySR equation references the parameter feature x0.

    Word-boundary match so a different feature like x01 is not counted.
    """
    return _X0.search(str(equation_str)) is not None


def write_grad_faith_sidecar(out_path, rows, *, param, z, tol,
                             log_space, source_pareto):
    """Write a sidecar CSV (one row per candidate) with a provenance header.

    rows: iterable of dicts keyed by SIDECAR_COLUMNS.
    Returns the written Path.
    """
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(list(rows), columns=SIDECAR_COLUMNS)
    header = (f"# param={param} z={z} tol={tol} log_space={log_space} "
              f"source={source_pareto}\n")
    with open(out_path, "w") as fh:
        fh.write(header)
        df.to_csv(fh, index=False)
    return out_path


def read_grad_faith_sidecar(path) -> pd.DataFrame:
    """Read a sidecar CSV, skipping the leading '#' provenance comment."""
    return pd.read_csv(path, comment="#")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_grad_faith_io.py -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/grad_faith_io.py tests/test_grad_faith_io.py
git commit -m "feat: grad-faithfulness sidecar IO (pure, emulator-free)"
```

---

### Task 2: `eval_grad_faithfulness.py --out` — write the sidecar (cluster path)

**Files:**
- Modify: `scripts/eval_grad_faithfulness.py`

This script needs the GP emulator, so it is not unit-tested here (its math is the production gate's, already covered). The change is mechanical: collect `x0_enters` + `gate_pass` per row and, when `--out` is given, write the sidecar via Task 1's module.

- [ ] **Step 1: Add the import and the `--out` argument**

In `scripts/eval_grad_faithfulness.py`, add to the imports near the top:

```python
from priya_forecast.grad_faith_io import (
    equation_has_x0, write_grad_faith_sidecar,
)
```

In `main()`, add the argument next to the others:

```python
    p.add_argument("--out", default=None,
                   help="write a grad-faith sidecar CSV to this path")
```

- [ ] **Step 2: Capture x0_enters and gate_pass while looping candidates**

Replace the existing per-candidate loop body so each appended row is a dict (the current code appends a tuple `(complexity, loss, err)`):

```python
    rows = []
    for _, row in safe.sort_values("Loss").iterrows():
        cand = fc._refit_from_row(
            equation_str=str(row["Equation"]), complexity=int(row["Complexity"]),
            loss=float(row["Loss"]), df=df, param_name=args.param, z=args.z,
            meta=meta, k_grid=kg, norm=norm, log_space=args.log_space,
        )
        g = equation_param_gradient(refit=cand, fid_value=float(meta.fid),
                                    k_grid=np.asarray(kg, float), z=args.z)
        err, nkeep = median_rel_error(g, target)
        rows.append({
            "Complexity": int(row["Complexity"]),
            "Loss": float(row["Loss"]),
            "grad_err": err,
            "n_keep": int(nkeep),
            "gate_pass": bool(err <= args.tol),
            "x0_enters": bool(equation_has_x0(str(row["Equation"]))),
        })
```

- [ ] **Step 3: Update the stdout table to read dict fields, and write the sidecar**

Replace the print loop and summary so it indexes dict keys instead of tuple positions, and append the sidecar write at the end of `main()`:

```python
    print(f"\n=== {args.param} z={args.z}  (Fisher-safe candidates, by loss) ===")
    print(f"{'cmplx':>6} {'loss':>10} {'grad_err':>10} {'gate(<=%.2f)':>12}"
          % args.tol)
    for r in rows:
        flag = "PASS" if r["gate_pass"] else "fail"
        print(f"{r['Complexity']:>6} {r['Loss']:>10.5f} "
              f"{r['grad_err']:>10.4f} {flag:>12}")

    if rows:
        best_loss = rows[0]  # already sorted by loss asc
        best_faith = min(rows, key=lambda r: r["grad_err"])
        any_pass = any(r["gate_pass"] for r in rows)
        print(f"\nbest_loss pick:   complexity={best_loss['Complexity']} "
              f"loss={best_loss['Loss']:.5f} grad_err={best_loss['grad_err']:.4f}")
        print(f"best faithfulness: complexity={best_faith['Complexity']} "
              f"loss={best_faith['Loss']:.5f} grad_err={best_faith['grad_err']:.4f}")
        print(f"ANY equation passes gate (<= {args.tol}): {any_pass}")

    if args.out:
        path = write_grad_faith_sidecar(
            args.out, rows, param=args.param, z=args.z, tol=args.tol,
            log_space=args.log_space, source_pareto=args.pareto,
        )
        print(f"\nwrote sidecar: {path}")
```

- [ ] **Step 4: Syntax-check (no emulator needed for this)**

Run: `python -m py_compile scripts/eval_grad_faithfulness.py && echo OK`
Expected: `OK`

- [ ] **Step 5: Commit**

```bash
git add scripts/eval_grad_faithfulness.py
git commit -m "feat: eval_grad_faithfulness --out writes grad-faith sidecar"
```

---

### Task 3: `pareto_diag.py` — load fronts + render the grid (pure, emulator-free)

**Files:**
- Create: `src/priya_forecast/pareto_diag.py`
- Test: `tests/test_pareto_diag.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/test_pareto_diag.py
import numpy as np
import pandas as pd

from priya_forecast.grad_faith_io import write_grad_faith_sidecar
from priya_forecast.pareto_diag import load_front, render_grid


def _write_pareto(path):
    pd.DataFrame({
        "Complexity": [1, 3, 6],
        "Loss": [24.6, 10.0, 4.2],
        "Equation": ["x0", "x0 * 2.66", "log(x0 + 0.16)"],
    }).to_csv(path, index=False)


def test_load_front_without_sidecar_is_all_nan(tmp_path):
    pareto = tmp_path / "pareto_ns.csv"
    _write_pareto(pareto)
    front = load_front(pareto, None)
    assert list(front["Complexity"]) == [1, 3, 6]
    assert front["grad_err"].isna().all()


def test_load_front_with_sidecar_joins_grad_err(tmp_path):
    pareto = tmp_path / "pareto_ns.csv"
    _write_pareto(pareto)
    side = write_grad_faith_sidecar(
        tmp_path / "grad_faith_ns.csv",
        [
            {"Complexity": 1, "Loss": 24.6, "grad_err": 0.90,
             "n_keep": 40, "gate_pass": False, "x0_enters": True},
            {"Complexity": 3, "Loss": 10.0, "grad_err": 0.13,
             "n_keep": 40, "gate_pass": True, "x0_enters": True},
        ],
        param="ns", z=3.6, tol=0.25, log_space=True, source_pareto=str(pareto),
    )
    front = load_front(pareto, side)
    # complexity 6 has no sidecar row -> grad_err NaN (left join)
    assert np.isnan(front.loc[front.Complexity == 6, "grad_err"].item())
    assert front.loc[front.Complexity == 1, "grad_err"].item() == 0.90


def test_render_grid_writes_nonempty_png(tmp_path):
    pareto = tmp_path / "pareto_ns.csv"
    _write_pareto(pareto)
    front = load_front(pareto, None)  # gray-fallback path
    out = tmp_path / "fig.png"
    render_grid(
        {"ns": [{"front": front, "label": "value@20", "marker": "o"}]},
        out, param_order=["ns"],
    )
    assert out.exists() and out.stat().st_size > 0
```

- [ ] **Step 2: Run test to verify it fails**

Run: `PYTHONPATH=src pytest tests/test_pareto_diag.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'priya_forecast.pareto_diag'`

- [ ] **Step 3: Write minimal implementation**

```python
# src/priya_forecast/pareto_diag.py
"""Render the per-parameter Pareto-faithfulness diagnostic figure.

Pure-plotting: reads PySR Pareto CSVs + grad-faith sidecars, no emulator.
A front whose sidecar is missing is drawn gray (value-only) so the layout
can be iterated before the cluster gradient eval lands.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from priya_forecast.grad_faith_io import read_grad_faith_sidecar

GATE_TOL = 0.25


def load_front(pareto_csv, sidecar_csv=None) -> pd.DataFrame:
    """Return a DataFrame[Complexity, Loss, grad_err, gate_pass].

    grad_err/gate_pass are NaN/NA when no sidecar is supplied or a given
    complexity has no sidecar row (left join).
    """
    pareto = pd.read_csv(pareto_csv)[["Complexity", "Loss"]].copy()
    if sidecar_csv is not None and Path(sidecar_csv).exists():
        side = read_grad_faith_sidecar(sidecar_csv)[
            ["Complexity", "grad_err", "gate_pass"]]
        return pareto.merge(side, on="Complexity", how="left")
    return pareto.assign(grad_err=np.nan, gate_pass=pd.NA)


def render_grid(fronts_by_param, out_path, *, gate_tol=GATE_TOL,
                param_order=None, ncol=4):
    """Render one panel per parameter; color = grad_err (clipped to [0,1]).

    fronts_by_param: {param: [ {front: DataFrame, label: str, marker: str}, ... ]}
    """
    params = list(param_order) if param_order else list(fronts_by_param)
    nrow = int(np.ceil(len(params) / ncol))
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             squeeze=False)
    cmap = plt.get_cmap("RdYlGn_r")
    norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=gate_tol, vmax=1.0)
    last_sc = None

    for i, p in enumerate(params):
        ax = axes[i // ncol][i % ncol]
        for series in fronts_by_param.get(p, []):
            df = series["front"]
            marker = series.get("marker", "o")
            ge = df["grad_err"].to_numpy(dtype=float)
            seen = ~np.isnan(ge)
            if seen.any():
                last_sc = ax.scatter(
                    df["Complexity"][seen], df["Loss"][seen],
                    c=np.clip(ge[seen], 0.0, 1.0), cmap=cmap, norm=norm,
                    marker=marker, edgecolor="k", linewidth=0.4, s=44,
                    zorder=3, label=series.get("label"))
            if (~seen).any():
                ax.scatter(df["Complexity"][~seen], df["Loss"][~seen],
                           color="0.75", marker=marker, s=44, zorder=2,
                           label=series.get("label") if not seen.any() else None)
        ax.set_yscale("log")
        ax.set_title(p)
        ax.set_xlabel("complexity")
        ax.set_ylabel("loss")
        ax.grid(True, which="both", alpha=0.2)
        if any(s.get("label") for s in fronts_by_param.get(p, [])):
            ax.legend(fontsize=7, loc="best")

    for j in range(len(params), nrow * ncol):
        axes[j // ncol][j % ncol].axis("off")

    if last_sc is not None:
        cbar = fig.colorbar(last_sc, ax=axes.ravel().tolist(),
                            fraction=0.025, pad=0.01)
        cbar.set_label("grad_err  (median |d_eq / d_GP - 1|, clipped at 1)")
        cbar.ax.axhline(gate_tol, color="k", lw=1.2)

    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    return out_path
```

- [ ] **Step 4: Run test to verify it passes**

Run: `PYTHONPATH=src pytest tests/test_pareto_diag.py -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/pareto_diag.py tests/test_pareto_diag.py
git commit -m "feat: pareto_diag load_front + render_grid (gray-fallback, emulator-free)"
```

---

### Task 4: `plot_pareto_faithfulness.py` CLI + the gray first-cut figure (local)

**Files:**
- Create: `scripts/plot_pareto_faithfulness.py`
- Create (output): `results/single_z_stage_pareto_diag/pareto_faithfulness.png`

- [ ] **Step 1: Write the CLI**

```python
#!/usr/bin/env python
"""Render the per-parameter Pareto-faithfulness figure from cached CSVs.

Local / emulator-free. Each --series points at a dir of pareto_<param>.csv;
if a matching grad_faith_<param>.csv sidecar sits beside it (or in
--sidecar-dir), points are colored by derivative faithfulness, else gray.

Example (Phase-1 gray first cut, no cluster):
  PYTHONPATH=src python scripts/plot_pareto_faithfulness.py \
    --series value@20=results/single_z_stage6_log/refit/z3.6 \
    --series Sobolev@20=results/single_z_stage9/refit/z3.6 \
    --out results/single_z_stage_pareto_diag/pareto_faithfulness.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.pareto_diag import load_front, render_grid

_MARKERS = ["o", "s", "^", "D", "v"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--series", action="append", required=True,
                    help="LABEL=PARETO_DIR (repeatable)")
    ap.add_argument("--sidecar-dir", action="append", default=None,
                    help="optional LABEL=SIDECAR_DIR overrides (repeatable)")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    sidecar_override = {}
    for s in (args.sidecar_dir or []):
        label, d = s.split("=", 1)
        sidecar_override[label] = Path(d)

    series_specs = []
    for i, s in enumerate(args.series):
        label, d = s.split("=", 1)
        series_specs.append((label, Path(d), _MARKERS[i % len(_MARKERS)]))

    fronts_by_param = {}
    for param in PARAM_NAMES:
        rows = []
        for label, pareto_dir, marker in series_specs:
            pareto = pareto_dir / f"pareto_{param}.csv"
            if not pareto.exists():
                continue
            sdir = sidecar_override.get(label, pareto_dir)
            sidecar = sdir / f"grad_faith_{param}.csv"
            rows.append({
                "front": load_front(pareto, sidecar if sidecar.exists() else None),
                "label": label, "marker": marker,
            })
        if rows:
            fronts_by_param[param] = rows

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    render_grid(fronts_by_param, args.out, param_order=list(PARAM_NAMES))
    print(f"wrote {args.out}  ({len(fronts_by_param)} params)")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Produce the Phase-1 gray figure (no cluster)**

Run:
```bash
PYTHONPATH=src python scripts/plot_pareto_faithfulness.py \
  --series value@20=results/single_z_stage6_log/refit/z3.6 \
  --series Sobolev@20=results/single_z_stage9/refit/z3.6 \
  --out results/single_z_stage_pareto_diag/pareto_faithfulness.png
```
Expected: `wrote results/single_z_stage_pareto_diag/pareto_faithfulness.png  (11 params)`

- [ ] **Step 3: Eyeball the figure**

Open the PNG. Confirm: 11 panels in the PARAM_NAMES order, log-y, two marker shapes (value@20 circle, Sobolev@20 square), all gray (no sidecars yet), per-panel legend. This is the layout the user reviews.

- [ ] **Step 4: Commit**

```bash
git add scripts/plot_pareto_faithfulness.py
git commit -m "feat: plot_pareto_faithfulness CLI + Phase-1 gray figure"
```

---

### Task 5: Walkthrough doc scaffold with inline figure (local)

**Files:**
- Create: `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`

- [ ] **Step 1: Write the scaffold**

Create `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md` with: a one-paragraph framing (diagnostic/failure-modes paper; this figure is the central instrument); a Methods section defining the axes, the `grad_err` color metric and the 0.25 gate, and the three series (value@20, Sobolev@20, value@certified-budget); the inline figure embed:

```markdown
![Per-parameter Pareto-faithfulness](../results/single_z_stage_pareto_diag/pareto_faithfulness.png)
```

then a **Per-parameter reading** section with one `### <param>` subsection per PARAM_NAMES entry, each currently a stub: `_Pending color figure (Phase 2)._`; and a closing **Failure-mode taxonomy** table with columns `param | category | mechanism | what Sobolev does` and rows for all 11 params, values `TBD (Phase 2)` until the color figure exists. State explicitly that this redirect **drops** the σ_PySR/σ_GP forecast claim in favor of the derivative-faithfulness diagnostic.

- [ ] **Step 2: Verify the embed resolves**

Run: `ls results/single_z_stage_pareto_diag/pareto_faithfulness.png && echo EMBED_OK`
Expected: the path lists + `EMBED_OK` (the `../results/...` relative path resolves from `docs/`).

- [ ] **Step 3: Commit**

```bash
git add docs/PARETO_FAITHFULNESS_WALKTHROUGH.md
git commit -m "docs: walkthrough scaffold with inline Pareto-faithfulness figure"
```

---

### Task 6: Sidecar driver + budget-control verdict (cluster)

**Files:**
- Create: `scripts/make_grad_faith_sidecars.sh`

This runs on a GP-capable node. It is orchestration over Task 2's CLI; no unit test.

- [ ] **Step 1: Write the driver**

```bash
#!/usr/bin/env bash
# Produce grad-faith sidecars for every param in one (or more) Pareto dirs.
# Usage: scripts/make_grad_faith_sidecars.sh <pareto_dir> <z> [extra eval args...]
set -euo pipefail
DIR="$1"; Z="$2"; shift 2
export PYTHON_JULIAPKG_PROJECT="$HOME/.julia_env"
export JULIA_DEPOT_PATH="$HOME/.julia"
export PYTHONPATH="src:/home/mfho/student_projects/lya_emulator_full"
PY="${PYTHON:-.venv/bin/python}"
PARAMS="dtau0 tau0 ns Ap herei heref alphaq hub omegamh2 hireionz bhfeedback"
for p in $PARAMS; do
  csv="$DIR/pareto_${p}.csv"
  [ -f "$csv" ] || { echo "skip $p (no $csv)"; continue; }
  echo "=== $p ==="
  "$PY" scripts/eval_grad_faithfulness.py \
    --pareto "$csv" --param "$p" --z "$Z" --basedir data/kodiaq_gp \
    --log-space --out "$DIR/grad_faith_${p}.csv" "$@"
done
```

- [ ] **Step 2: Run the value@20 and Sobolev@20 sidecars**

```bash
chmod +x scripts/make_grad_faith_sidecars.sh
scripts/make_grad_faith_sidecars.sh results/single_z_stage6_log/refit/z3.6 3.6
scripts/make_grad_faith_sidecars.sh results/single_z_stage9/refit/z3.6 3.6
```
Expected: a `grad_faith_<param>.csv` beside every `pareto_<param>.csv` in both dirs.

- [ ] **Step 3: Capture the budget-control verdict (ns)**

```bash
PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \
PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full \
.venv/bin/python scripts/eval_grad_faithfulness.py \
  --pareto results/decider_budget_z3.6/refit/z3.6/pareto_ns.csv \
  --param ns --z 3.6 --basedir data/kodiaq_gp --log-space \
  --out results/decider_budget_z3.6/refit/z3.6/grad_faith_ns.csv \
  | tee results/decider_budget_z3.6/ns_grad_verdict.log
```
Record in the walkthrough: does the certified-budget (complexity ~35) value-loss equation clear the 0.25 gate? (Expected from the review: no — budget alone does not cure the derivative.)

- [ ] **Step 4: Re-render in color (now sidecars exist)**

```bash
PYTHONPATH=src python scripts/plot_pareto_faithfulness.py \
  --series value@20=results/single_z_stage6_log/refit/z3.6 \
  --series Sobolev@20=results/single_z_stage9/refit/z3.6 \
  --series value@budget=results/decider_budget_z3.6/refit/z3.6 \
  --out results/single_z_stage_pareto_diag/pareto_faithfulness.png
```
Expected: same panels, now colored; ns shows red value@20 + red value@budget + green Sobolev@20.

- [ ] **Step 5: Commit**

```bash
git add scripts/make_grad_faith_sidecars.sh \
  results/single_z_stage_pareto_diag/pareto_faithfulness.png \
  results/*/refit/z3.6/grad_faith_*.csv \
  results/decider_budget_z3.6/refit/z3.6/grad_faith_ns.csv \
  results/decider_budget_z3.6/ns_grad_verdict.log
git commit -m "feat: grad-faith sidecar driver + color figure + ns budget-control verdict"
```

---

### Task 7: Fill the walkthrough — per-parameter reasoning + taxonomy (analysis)

**Files:**
- Modify: `docs/PARETO_FAITHFULNESS_WALKTHROUGH.md`

- [ ] **Step 1: Read each panel and the sidecars; fill the per-param subsections**

For each param, state the empirical reading (does grad_err fall with complexity? does Sobolev turn it green? does the budget series stay red? at what complexity does `x0_enters` first become true?) and tie it to the mechanism from spec §6. Verify before asserting:
- **ns** — red under value@20 *and* value@budget, green under Sobolev → Mirage curable by the loss, not by budget.
- **hub** — check `x0_enters` first-true complexity (under-search signal) and whether it stays red under Sobolev at all complexities (basis argument: k-rescaling / AP-like distortion a per-param native-k ansatz can't express).
- **bhfeedback** — weak/degenerate gradient (priored out); expect ill-conditioned grad_err.
- **herei, alphaq** — among worst faithfulness; tie to the unrepresentable +0.45 cross-coupling.
- the remaining params — classify easy vs mirage from the panel.

- [ ] **Step 2: Fill the taxonomy table**

Replace each `TBD (Phase 2)` row with `category` ∈ {easy, mirage-cured-by-Sobolev, resistant} + the one-line mechanism + what Sobolev does.

- [ ] **Step 3: Commit**

```bash
git add docs/PARETO_FAITHFULNESS_WALKTHROUGH.md
git commit -m "docs: per-parameter failure-mode reasoning + taxonomy from the color figure"
```

---

## Self-Review

**Spec coverage:**
- §3 figure (loss-vs-complexity, color=grad_err, gate line, 3 series, x0-enters annotation) → Tasks 3 (render_grid color + gate line), 4 (series via CLI), 2 (x0_enters in sidecar), 7 (x0-enters reading). ✓
- §4.1 eval `--out` sidecar → Task 2. ✓
- §4.2 pure plotter + gray degradation → Task 3 (`load_front` NaN path, `render_grid` gray scatter) + Task 4 CLI. ✓
- §4.3 sidecar driver + budget control → Task 6. ✓
- §4.4 walkthrough doc → Tasks 5 (scaffold) + 7 (fill). ✓
- §5 grad_err metric = the gate → Task 2 reuses `median_rel_error`/`equation_param_gradient`/`gp_param_gradient` unchanged; `gate_pass = err <= tol`. ✓
- §8 testing (sidecar columns + gate_pass bool; plotter writes PNG + gray fallback) → Tasks 1 and 3 tests. ✓
- §9 provenance (series budget/source in legend + header) → sidecar header (Task 1), series labels carry budget (Task 4), no GP-slice values plotted (raw sidecar only). ✓

**Placeholder scan:** No TBD/TODO in code steps. The walkthrough's `TBD (Phase 2)` strings are intentional document placeholders filled in Task 7, not plan placeholders.

**Type consistency:** `SIDECAR_COLUMNS` defined in Task 1 is the exact column set written by Task 2's rows and read by Task 3's `load_front` (which selects `Complexity, grad_err, gate_pass`). `render_grid` signature `(fronts_by_param, out_path, *, gate_tol, param_order, ncol)` matches the calls in the Task 3 test and Task 4 CLI. `load_front(pareto_csv, sidecar_csv=None)` matches both. `write_grad_faith_sidecar(out_path, rows, *, param, z, tol, log_space, source_pareto)` matches the Task 2 call. ✓

**Note on Task 6 Step 3:** the `PYTHON_JULIAPKG_PROJECT=...` env vars are shown inline for readability but must be exported (as the driver in Step 1 does) or prefixed before `.venv/bin/python`, not passed as argv — set them on the command line before the `python` token when running ad-hoc.
