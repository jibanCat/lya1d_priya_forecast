# Single-z Stage 4 (batch + aggregate) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: superpowers:subagent-driven-development. Checkbox (`- [ ]`) steps.

**Goal:** Fan the single-z pipeline over all 13 z-bins and aggregate the per-z forecasts into an across-z view (σ(z) trends).

**Architecture:** `_write_forecast_deliverables` also saves each `FisherResult` as an `.npz` (machine-readable). `scripts/aggregate_z.py` reads the 13 per-z `.npz` files and writes σ(z) plots + tables. `scripts/run_batch.py` derives one config per z-bin and either loops the pipeline in-process (`gp_only`/`forecast_only`) or submits the SLURM array (`refit_and_forecast`), then aggregates.

**Tech Stack:** Python 3.11, numpy, matplotlib. Reuses `single_z.pipeline.run`, `single_z.config.load_config`, `FisherResult.save_npz`.

**Spec:** `docs/superpowers/specs/2026-05-18-single-z-stage-bc-design.md` §6.

**Branch:** `single_z_forecast_clean`. Test command: `PYTHONPATH=src pytest <file> -v`.

---

## Task 1: persist FisherResults as npz

**Files:** Modify `src/priya_forecast/single_z/pipeline.py`, `tests/test_single_z_pipeline.py`.

- [ ] **Step 1** — read `_write_forecast_deliverables` in `pipeline.py`. `FisherResult` has a `.save_npz(path)` method (writes `F, cov, sigma, corr, steps, param_names, theta_fid`).

- [ ] **Step 2: Implement** — in `_write_forecast_deliverables`, after the corner plot is written, add a loop that saves each result:

```python
    for label, fr in results.items():
        fr.save_npz(out_dir / f"fisher_{label}.npz")
```

Add `fisher_npz` to the returned dict: `{"GP": out_dir/"fisher_GP.npz", ...}` — i.e. add a key `"fisher_npz": {label: out_dir / f"fisher_{label}.npz" for label in results}`.

- [ ] **Step 3: Test** — append to `tests/test_single_z_pipeline.py`:

```python
def test_write_forecast_deliverables_saves_npz(tmp_path: Path):
    """_write_forecast_deliverables persists each FisherResult as an npz."""
    import numpy as np
    from priya_forecast.fisher import FisherResult
    from priya_forecast.single_z.pipeline import _write_forecast_deliverables
    from priya_forecast.single_z.config import PipelineConfig

    def _fake_fr(scale):
        n = 2
        return FisherResult(
            F=np.eye(n), cov=np.eye(n) * scale, sigma=np.full(n, scale),
            corr=np.eye(n), steps=np.full(n, 0.01),
            param_names=("ns", "Ap"), theta_fid=np.array([0.98, 1.46]),
        )

    cfg = PipelineConfig(mode="forecast_only", parameters=["ns", "Ap"])
    results = {"GP": _fake_fr(0.1), "perfect_1D": _fake_fr(0.1),
               "PySR": _fake_fr(0.2)}
    out = tmp_path / "out"
    out.mkdir()
    _write_forecast_deliverables(cfg, out, results, pysr_available=True)
    for label in ("GP", "perfect_1D", "PySR"):
        npz = out / f"fisher_{label}.npz"
        assert npz.exists()
        loaded = np.load(npz, allow_pickle=True)
        assert loaded["sigma"].shape == (2,)
```

- [ ] **Step 4: Run** — `PYTHONPATH=src pytest tests/test_single_z_pipeline.py -k write_forecast_deliverables -v` → PASS. Full file → no regression.

- [ ] **Step 5: Commit** — `git add -A && git commit -m "Stage 4: persist FisherResults as npz in deliverables"`.

---

## Task 2: `scripts/aggregate_z.py`

**Files:** Create `scripts/aggregate_z.py`, `tests/test_aggregate_z.py`.

- [ ] **Step 1: Write the failing test** — create `tests/test_aggregate_z.py`:

```python
"""Unit tests for scripts/aggregate_z.py (run as a module via importlib)."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import numpy as np
import pytest

_SPEC = importlib.util.spec_from_file_location(
    "aggregate_z",
    Path(__file__).parent.parent / "scripts" / "aggregate_z.py",
)


def _load_module():
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def _write_fisher_npz(path, sigma, names=("ns", "Ap")):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path, F=np.eye(len(names)), cov=np.diag(np.array(sigma) ** 2),
        sigma=np.array(sigma, dtype=float), corr=np.eye(len(names)),
        steps=np.full(len(names), 0.01), param_names=np.array(names),
        theta_fid=np.array([0.98, 1.46]),
    )


def test_collect_sigma_z(tmp_path):
    mod = _load_module()
    for z in (3.4, 3.6):
        _write_fisher_npz(tmp_path / f"z{z}" / "fisher_GP.npz", [0.1 * z, 0.2 * z])
    table = mod.collect_sigma_z(base_dir=tmp_path, label="GP",
                                z_bins=[3.4, 3.6])
    # table maps param -> {z: sigma}
    assert table["ns"][3.6] == pytest.approx(0.1 * 3.6)
    assert table["Ap"][3.4] == pytest.approx(0.2 * 3.4)


def test_aggregate_writes_outputs(tmp_path):
    mod = _load_module()
    for z in (3.4, 3.6):
        for lab in ("GP", "perfect_1D", "PySR"):
            _write_fisher_npz(tmp_path / f"z{z}" / f"fisher_{lab}.npz",
                              [0.1, 0.2])
    out = mod.aggregate(base_dir=tmp_path, z_bins=[3.4, 3.6])
    assert (tmp_path / "aggregate" / "sigma_vs_z.png").exists()
    assert (tmp_path / "aggregate" / "sigma_table.md").exists()
    assert out == tmp_path / "aggregate"
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — create `scripts/aggregate_z.py`:

```python
#!/usr/bin/env python
"""Aggregate per-z single-z forecasts into an across-z view.

Reads `<base>/z{z}/fisher_{GP,perfect_1D,PySR}.npz` for each z-bin present
and writes `<base>/aggregate/`: a σ(z) trend plot and a σ-table.

    python scripts/aggregate_z.py --base results/single_z_run
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

# 13 kodiaq z-bins.
Z_BINS_13 = [round(z, 1) for z in np.arange(2.2, 4.601, 0.2)]
LABELS = ("GP", "perfect_1D", "PySR")


def collect_sigma_z(*, base_dir, label: str, z_bins) -> dict:
    """Return {param_name: {z: sigma}} for one label across the z-bins present."""
    base_dir = Path(base_dir)
    out: dict[str, dict[float, float]] = {}
    for z in z_bins:
        npz = base_dir / f"z{z}" / f"fisher_{label}.npz"
        if not npz.exists():
            continue
        d = np.load(npz, allow_pickle=True)
        names = [str(n) for n in d["param_names"]]
        for name, s in zip(names, d["sigma"]):
            out.setdefault(name, {})[z] = float(s)
    return out


def aggregate(*, base_dir, z_bins=None) -> Path:
    """Write the σ(z) plot + table to `<base_dir>/aggregate/`."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    base_dir = Path(base_dir)
    z_bins = list(Z_BINS_13 if z_bins is None else z_bins)
    per_label = {lab: collect_sigma_z(base_dir=base_dir, label=lab, z_bins=z_bins)
                 for lab in LABELS}
    agg = base_dir / "aggregate"
    agg.mkdir(parents=True, exist_ok=True)

    params = sorted({p for tbl in per_label.values() for p in tbl})
    if not params:
        raise FileNotFoundError(
            f"No fisher_*.npz found under {base_dir}/z*/ — run the pipeline first."
        )
    ncol = min(4, len(params))
    nrow = (len(params) + ncol - 1) // ncol
    fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow),
                             dpi=120, squeeze=False)
    for i, param in enumerate(params):
        ax = axes[i // ncol][i % ncol]
        for lab in LABELS:
            tbl = per_label[lab].get(param, {})
            if not tbl:
                continue
            zs = sorted(tbl)
            ax.plot(zs, [tbl[z] for z in zs], "o-", label=lab)
        ax.set_title(param)
        ax.set_xlabel("z")
        ax.set_ylabel(r"$\sigma$")
        ax.set_yscale("log")
        ax.legend(fontsize=7)
    for j in range(len(params), nrow * ncol):
        axes[j // ncol][j % ncol].set_visible(False)
    fig.suptitle("Single-z forecast — σ vs redshift")
    fig.tight_layout()
    fig.savefig(agg / "sigma_vs_z.png")
    plt.close(fig)

    lines = ["# Across-z σ table\n"]
    for lab in LABELS:
        tbl = per_label[lab]
        if not tbl:
            continue
        zs = sorted({z for d in tbl.values() for z in d})
        lines.append(f"\n## {lab}\n")
        lines.append("| param | " + " | ".join(f"z={z}" for z in zs) + " |")
        lines.append("|" + "---|" * (len(zs) + 1))
        for param in params:
            row = tbl.get(param, {})
            cells = " | ".join(
                f"{row[z]:.4g}" if z in row else "—" for z in zs
            )
            lines.append(f"| {param} | {cells} |")
    (agg / "sigma_table.md").write_text("\n".join(lines) + "\n")
    return agg


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--base", required=True,
                   help="Run directory containing z{z}/ subdirs.")
    args = p.parse_args()
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    out = aggregate(base_dir=args.base)
    print(f"wrote {out}/sigma_vs_z.png and {out}/sigma_table.md")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Run, expect PASS.**

- [ ] **Step 5: Commit** — `git add scripts/aggregate_z.py tests/test_aggregate_z.py && git commit -m "Stage 4: aggregate_z.py — across-z sigma(z) view"`.

---

## Task 3: `scripts/run_batch.py`

**Files:** Create `scripts/run_batch.py`, `tests/test_run_batch.py`.

- [ ] **Step 1: Write the failing test** — create `tests/test_run_batch.py`:

```python
"""Unit tests for scripts/run_batch.py."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

_SPEC = importlib.util.spec_from_file_location(
    "run_batch", Path(__file__).parent.parent / "scripts" / "run_batch.py",
)


def _load():
    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
    mod = importlib.util.module_from_spec(_SPEC)
    _SPEC.loader.exec_module(mod)
    return mod


def test_derive_z_configs(tmp_path):
    """derive_z_configs fans a base config over the 13 z-bins."""
    mod = _load()
    from priya_forecast.single_z.config import PipelineConfig

    base = PipelineConfig(mode="forecast_only", redshift=3.6,
                          output_dir=str(tmp_path / "run"))
    derived = mod.derive_z_configs(base)
    assert len(derived) == 13
    redshifts = sorted(c.redshift for c in derived)
    assert redshifts[0] == pytest.approx(2.2)
    assert redshifts[-1] == pytest.approx(4.6)
    # each derived config writes into its own z-subdir
    for c in derived:
        assert c.output_dir.endswith(f"z{c.redshift}")
```

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** — create `scripts/run_batch.py`:

```python
#!/usr/bin/env python
"""Fan the single-z pipeline over all 13 z-bins.

gp_only / forecast_only  — loops the pipeline in-process, then aggregates.
refit_and_forecast       — two phases:
    --phase submit  : submit 13 SLURM array jobs (one per z-bin).
    --phase collect : forecast per z-bin from the refits, then aggregate.

    python scripts/run_batch.py --config configs/single_z/example.yaml
    python scripts/run_batch.py --config c.yaml --mode refit_and_forecast --phase submit
"""

from __future__ import annotations

import argparse
import dataclasses
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from priya_forecast.single_z.config import PipelineConfig, load_config

Z_BINS_13 = [round(z, 1) for z in np.arange(2.2, 4.601, 0.2)]


def derive_z_configs(base: PipelineConfig) -> list[PipelineConfig]:
    """One PipelineConfig per z-bin: override redshift + output_dir."""
    base_out = base.output_dir.rstrip("/")
    derived = []
    for z in Z_BINS_13:
        c = dataclasses.replace(
            base, redshift=z, output_dir=f"{base_out}/z{z}",
        )
        derived.append(c)
    return derived


def run_inprocess(base: PipelineConfig) -> Path:
    """gp_only / forecast_only: run all 13 z-bins in-process, then aggregate."""
    from priya_forecast.single_z.pipeline import run
    from aggregate_z import aggregate  # type: ignore

    for cfg in derive_z_configs(base):
        print(f"[batch] z={cfg.redshift} ...", flush=True)
        run(cfg)
    out = aggregate(base_dir=base.output_dir.rstrip("/"), z_bins=Z_BINS_13)
    print(f"[batch] aggregated → {out}")
    return out


def submit_slurm(base: PipelineConfig, repo: Path) -> None:
    """refit_and_forecast --phase submit: one SLURM array job per z-bin."""
    base_out = base.output_dir.rstrip("/")
    for z in Z_BINS_13:
        cmd = [
            "sbatch",
            f"--export=ALL,REPO={repo},BASEDIR={base.gp.basedir},"
            f"OUTPUT_DIR={base_out}/z{z},Z={z}",
            "--array=0-10",
            str(repo / "slurm" / "single_z_refit.slurm"),
        ]
        print("[batch submit]", " ".join(cmd), flush=True)
        subprocess.run(cmd, check=True)


def main() -> None:
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--config", required=True, type=Path)
    p.add_argument("--mode", default=None,
                   help="Override the YAML mode.")
    p.add_argument("--phase", choices=["submit", "collect"], default=None,
                   help="refit_and_forecast only: submit the SLURM array, "
                        "or collect+forecast+aggregate.")
    args = p.parse_args()

    base = load_config(args.config)
    if args.mode is not None:
        base.mode = args.mode

    repo = Path(__file__).resolve().parent.parent
    if base.mode in ("gp_only", "forecast_only"):
        run_inprocess(base)
    elif base.mode == "refit_and_forecast":
        if args.phase == "submit":
            submit_slurm(base, repo)
        elif args.phase == "collect":
            run_inprocess(base)  # forecasts from from_refit CSVs + aggregates
        else:
            raise SystemExit(
                "refit_and_forecast needs --phase submit or --phase collect."
            )
    else:
        raise SystemExit(f"unknown mode {base.mode!r}.")


if __name__ == "__main__":
    main()
```

Note: `dataclasses.replace` on `PipelineConfig` performs a shallow copy — the nested config objects (`gp`, `data`, …) are shared, which is fine here since the per-z configs only differ in `redshift`/`output_dir`. Confirm `PipelineConfig` is a plain `@dataclass` (it is) so `dataclasses.replace` works.

- [ ] **Step 4: Run, expect PASS** — `PYTHONPATH=src pytest tests/test_run_batch.py -v`.

- [ ] **Step 5: Commit** — `git add scripts/run_batch.py tests/test_run_batch.py && git commit -m "Stage 4: run_batch.py — fan pipeline over 13 z-bins"`.

---

## Task 4: verification

- [ ] **Step 1** — `PYTHONPATH=src pytest tests/ -q -k "single_z or aggregate_z or run_batch or refit_1d_pysr_pareto"` — all pure tests pass, gated SKIP, no regression.
- [ ] **Step 2** — `git status --short` clean under `src/`, `scripts/`, `tests/`.

## Done criteria

- Per-z forecasts persist `fisher_{label}.npz`.
- `aggregate_z.py` produces `sigma_vs_z.png` + `sigma_table.md`.
- `run_batch.py` fans gp_only/forecast_only in-process and submits the SLURM array for refit_and_forecast.
- No regression.
