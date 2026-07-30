# rerun_paper.ipynb Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** A collaborator-facing `notebooks/rerun_paper.ipynb` that re-runs the exact per-`(param, z)` PySR pipeline, tweaks knobs/hypotheses, writes to an isolated `results/tutorial_reruns/` dir, and regenerates the paper's figures/tables from that run via `reproduce_paper`'s plotting API.

**Architecture:** A new `src/priya_forecast/rerun.py` orchestration module drives the *same* `refit_one_param_single_z` the CLI calls and subprocesses the *same* `eval_grad_faithfulness.py` scorer, into a production-layout run dir. Physics (fiducial/prior) overrides flow through a `parameters.override_params` context manager that temporarily rebinds the module global — zero changes to the paper-critical `refit_1d_pysr.py`. A gitignored builder emits the notebook; `paper_figures.load_run(run_dir)` retargets all Tier-1 figures.

**Tech Stack:** Python 3, numpy/pandas, dataclasses, subprocess; PySR/GPy only at *run* time (mocked in CI). Notebook built via the repo's `md()`/`code()` builder pattern.

**Spec:** `docs/dev/superpowers/specs/2026-07-06-rerun-paper-notebook-design.md`

## Global Constraints

- **Branch:** `rerun-notebook` (already created off `paper-production`).
- **Never write into production:** all output under `results/tutorial_reruns/`; hard-raise if a run dir resolves inside `results/paper_production_*` or `results/refit_phase2_production`.
- **Guards warn, never fail** (except the two hard errors: production-path guard, and config validation). No new heavy CI deps — every test mocks the GP.
- **Reuse, don't reimplement:** the refit is `priya_forecast.single_z.refit.refit_one_param_single_z`; the scorer is `scripts/eval_grad_faithfulness.py`; figures are `priya_forecast.paper_figures.load_run` + `plot_*`/`taxonomy`.
- **Style (notebook markdown only):** match `~/Latex/writing.md` (kodiaq_emu voice), American spelling, no hyperbole.
- **Commit cadence:** commit after each task's tests pass. Commit trailer: `Co-Authored-By: Claude Opus 4.8 (1M context) <noreply@anthropic.com>`.

### Recon facts (exact signatures — do not re-derive)

- `parameters.Param` is a `@dataclass(frozen=True)`: fields `name:str, fid:float, prior:tuple[float,float], latex:str, unit_scale:float=1.0`. `PARAMS_11D: tuple[Param,...]` (order: dtau0, tau0, ns, Ap, herei, heref, alphaq, hub, omegamh2, hireionz, bhfeedback). `PARAM_NAMES = tuple(p.name ...)`. Accessors `get_param(name)`, `fiducial_vector()` read the module global at call time. Existing optional-arg pattern to mirror: `validate_priors(params=PARAMS_11D)`.
- `single_z.refit.refit_one_param_single_z(*, param_name, z, cfg, gp_lf, gp_hf, k_grid, out_dir, max_retries=4, save_artifacts=False)` → `Refit1DResult` (fields incl. `pareto_complexity:int`, `pareto_loss:float`). It writes `pareto_<param>.csv` directly into `out_dir` (does NOT append `refit/z`). `kodiaq_k_grid(kmin, kmax, nk=48)`.
- `single_z.config`: `PipelineConfig(mode, redshift, output_dir, gp=GPConfig(basedir), pysr=PySRConfig(...), target_space, ...)`; `PySRConfig(niterations, populations, procs, maxsize, seed, use_sobolev, sobolev_lambda, use_anova_loss, smart_kwargs)`; `GPConfig(basedir, hires_subdir)`. Operators are chosen by `pysr.smart_kwargs` (True→SMART), NOT config fields. `cfg.validate()` exists.
- GP model: `from priya_forecast.models.gp_model import GPModel`; `GPModel(basedir=..., fidelity="lf"|"hf", kf=k_grid)`; `.predict(fid_vec, k_grid, z)`.
- Scorer CLI: `python scripts/eval_grad_faithfulness.py --pareto <p.csv> --param <name> --z <z> --basedir data/kodiaq_gp --out <grad_faith_name.csv>` (log-space + gate 0.25 are defaults; keep `--kmin/--kmax` default or it raises).
- Sidecar readers: `grad_faith_io.read_grad_faith_sidecar(path)->DataFrame`; `grad_faith_io.knee_row(df, *, rel_tol=0.1)->pd.Series`. Sidecar columns: `Complexity,Loss,grad_err,value_mse,n_keep,gate_pass,x0_enters` (with a leading `#` provenance line). Metrics: `knee_row(df)["grad_err"]`, `["value_mse"]`, `["gate_pass"]`.
- Figures: `paper_figures.load_run(data_dir=DEFAULT_RUN_DIR, z=3.6, *, value_sub="value", sobolev_sub="sobolev", ...) -> PaperRun`; `taxonomy(run, gate=0.25)->DataFrame`; `plot_scorecard(run)`; `plot_pareto_faithfulness(run, out_path)`; `plot_ns_budget(run)`. `load_run` warns-and-skips missing arms → a value+sobolev-only quick run loads fine for taxonomy/scorecard/pareto.
- Layout `load_run` expects: `<data_dir>/value/refit/z<z>/{pareto_<p>,grad_faith_<p>}.csv` and `<data_dir>/sobolev/refit/z<z>/...`.
- Emulator: package `lyaemu` via `PYTHONPATH=src:$LYA_EMULATOR` (NOT pip; clone `github.com/sbird/lya_emulator`) + `pip install GPy emukit` (numpy<2). `data/kodiaq_gp` built by `scripts/prep_kodiaq_gp.py --source <PRIVATE_PRIYA_SET> --dest data/kodiaq_gp` — the `--source` is private, so collaborators need a **hosted archive** (fill-in URL). `results/tutorial_reruns/` is already git-ignored by `.gitignore` `results/*` (verified via `git check-ignore`).
- Builder: `notebooks/_build_reproduce_paper.py` uses `md(src)`/`code(src)` helpers returning nbformat cell dicts; writes JSON to `notebooks/reproduce_paper.ipynb`. `notebooks/_build_*.py` is gitignored (`.gitignore:257`).

---

### Task 1: `parameters.with_overrides` + `override_params` context manager

**Files:**
- Modify: `src/priya_forecast/parameters.py` (append two functions after `validate_priors`)
- Test: `tests/test_parameters_overrides.py`

**Interfaces:**
- Produces:
  - `with_overrides(fiducial: dict[str,float] | None = None, prior: dict[str,tuple[float,float]] | None = None, base: tuple[Param,...] = PARAMS_11D) -> tuple[Param,...]`
  - `override_params(fiducial=None, prior=None) -> contextmanager` (temporarily rebinds `parameters.PARAMS_11D`)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_parameters_overrides.py
import dataclasses
import pytest
from priya_forecast import parameters as P


def test_with_overrides_replaces_fid_and_prior_only_named():
    out = P.with_overrides(fiducial={"ns": 0.95}, prior={"Ap": (1.0, 3.0)})
    names = [p.name for p in out]
    assert names == list(P.PARAM_NAMES)                 # order/names unchanged
    assert P.get_param("ns").fid == 0.983               # global untouched
    assert next(p for p in out if p.name == "ns").fid == 0.95
    assert next(p for p in out if p.name == "Ap").prior == (1.0, 3.0)
    # untouched params identical to base
    assert next(p for p in out if p.name == "tau0") == P.get_param("tau0")


def test_with_overrides_unknown_name_raises():
    with pytest.raises(KeyError):
        P.with_overrides(fiducial={"not_a_param": 1.0})


def test_with_overrides_none_returns_equal_copy():
    out = P.with_overrides()
    assert tuple(out) == tuple(P.PARAMS_11D)


def test_override_params_context_swaps_and_restores():
    original = P.PARAMS_11D
    assert P.fiducial_vector()[2] == 0.983              # ns index 2
    with P.override_params(fiducial={"ns": 0.90}):
        assert P.PARAMS_11D is not original
        assert P.get_param("ns").fid == 0.90
        assert P.fiducial_vector()[2] == 0.90           # accessor sees override
    assert P.PARAMS_11D is original                     # restored
    assert P.get_param("ns").fid == 0.983


def test_override_params_restores_on_exception():
    original = P.PARAMS_11D
    with pytest.raises(RuntimeError):
        with P.override_params(prior={"ns": (0.5, 0.6)}):
            raise RuntimeError("boom")
    assert P.PARAMS_11D is original


def test_override_params_noop_when_empty():
    original = P.PARAMS_11D
    with P.override_params():
        assert P.PARAMS_11D is original                 # no swap needed
```

- [ ] **Step 2: Run — verify fail**

Run: `PYTHONPATH=src pytest tests/test_parameters_overrides.py -v`
Expected: FAIL (`AttributeError: module ... has no attribute 'with_overrides'`).

- [ ] **Step 3: Implement in `parameters.py`**

Add near the top imports: `import dataclasses` and `from contextlib import contextmanager`. Append after `validate_priors`:

```python
def with_overrides(fiducial=None, prior=None, base=None):
    """Return a copy of `base` (default PARAMS_11D) with named params' fid/prior
    replaced. Names and order are preserved; unknown names raise KeyError.
    PARAMS_11D itself is never mutated."""
    base = PARAMS_11D if base is None else base
    fiducial = fiducial or {}
    prior = prior or {}
    known = {p.name for p in base}
    unknown = (set(fiducial) | set(prior)) - known
    if unknown:
        raise KeyError(f"unknown parameter name(s): {sorted(unknown)}")
    out = []
    for p in base:
        changes = {}
        if p.name in fiducial:
            changes["fid"] = float(fiducial[p.name])
        if p.name in prior:
            lo, hi = prior[p.name]
            changes["prior"] = (float(lo), float(hi))
        out.append(dataclasses.replace(p, **changes) if changes else p)
    return tuple(out)


@contextmanager
def override_params(fiducial=None, prior=None):
    """Temporarily rebind the module-global PARAMS_11D so that get_param() /
    fiducial_vector() (read at call time) return overridden fid/prior. Restores
    the original on exit. No-op when no overrides are given."""
    global PARAMS_11D
    if not fiducial and not prior:
        yield
        return
    original = PARAMS_11D
    PARAMS_11D = with_overrides(fiducial=fiducial, prior=prior, base=original)
    try:
        yield
    finally:
        PARAMS_11D = original
```

- [ ] **Step 4: Run — verify pass**

Run: `PYTHONPATH=src pytest tests/test_parameters_overrides.py -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Regression — existing param tests still green**

Run: `PYTHONPATH=src pytest tests/ -k parameter -q`
Expected: PASS (no existing test references `PARAMS_11D` mutation).

- [ ] **Step 6: Commit**

```bash
git add src/priya_forecast/parameters.py tests/test_parameters_overrides.py
git commit -m "feat(parameters): with_overrides + override_params context manager"
```

---

### Task 2: `RerunConfig` dataclass + presets + isolation guard

**Files:**
- Create: `src/priya_forecast/rerun.py`
- Test: `tests/test_rerun_config.py`

**Interfaces:**
- Produces:
  - `PRODUCTION_BUDGET = {"niterations": 200, "populations": 48, "maxsize": 20, "sobolev_lambda": 5.0}`
  - `ALL_PARAMS: tuple[str,...]` (= `parameters.PARAM_NAMES`)
  - `@dataclass RerunConfig` with fields: `params: list[str]`, `zs: list[float]`, `arms: list[str]`, `maxsize: int`, `niterations: int`, `populations: int`, `sobolev_lambda: float`, `seed: int`, `kmin: float=0.001`, `kmax: float=0.04`, `smart_kwargs: bool=True`, `fiducial_overrides: dict|None=None`, `prior_overrides: dict|None=None`, `label: str="quick"`, `out_root: Path=Path("results/tutorial_reruns")`, `basedir: str="data/kodiaq_gp"`
  - classmethods `RerunConfig.quick()`, `RerunConfig.full()`
  - method `RerunConfig.run_dir -> Path` (= `out_root / f"rerun_{label}"`), raising `ValueError` if it resolves inside a production dir
  - method `RerunConfig.validate()` (arms subset of {"value","sobolev"}; sobolev requires log target — enforced implicitly since we always use log; niterations>=1 etc.)

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rerun_config.py
from pathlib import Path
import pytest
from priya_forecast.rerun import RerunConfig, PRODUCTION_BUDGET, ALL_PARAMS


def test_quick_preset():
    c = RerunConfig.quick()
    assert c.params == list(ALL_PARAMS) and len(c.params) == 11
    assert c.zs == [3.6]
    assert set(c.arms) == {"value", "sobolev"}
    assert c.niterations < PRODUCTION_BUDGET["niterations"]     # reduced budget
    assert c.label == "quick"


def test_full_preset_matches_production_budget():
    c = RerunConfig.full()
    assert c.zs == [2.6, 3.6, 4.2]
    assert c.niterations == PRODUCTION_BUDGET["niterations"]
    assert c.populations == PRODUCTION_BUDGET["populations"]
    assert c.sobolev_lambda == PRODUCTION_BUDGET["sobolev_lambda"]


def test_run_dir_is_under_tutorial_root():
    c = RerunConfig.quick()
    assert c.run_dir == Path("results/tutorial_reruns/rerun_quick")


def test_run_dir_raises_if_inside_production(tmp_path):
    c = RerunConfig.quick()
    c.out_root = Path("results/paper_production_20260630_perz_sobolev_z2.6-4.2")
    with pytest.raises(ValueError, match="production"):
        _ = c.run_dir


def test_validate_rejects_bad_arm():
    c = RerunConfig.quick()
    c.arms = ["value", "nonsense"]
    with pytest.raises(ValueError):
        c.validate()
```

- [ ] **Step 2: Run — verify fail** — `PYTHONPATH=src pytest tests/test_rerun_config.py -v` → FAIL (import error).

- [ ] **Step 3: Implement `rerun.py` (config portion)**

```python
"""Re-run the paper's per-(param, z) PySR pipeline into an isolated tutorial dir.

Collaborator-facing: drives the SAME refit_one_param_single_z the CLI calls and
the SAME eval_grad_faithfulness.py scorer, so a rerun is byte-comparable to the
paper. Output goes under results/tutorial_reruns/ (git-ignored, never overwrites
the production run). See notebooks/rerun_paper.ipynb.
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path

from priya_forecast.parameters import PARAM_NAMES

ALL_PARAMS = tuple(PARAM_NAMES)
PRODUCTION_BUDGET = {"niterations": 200, "populations": 48, "maxsize": 20,
                     "sobolev_lambda": 5.0}
_PRODUCTION_MARKERS = ("paper_production", "refit_phase2_production")
_VALID_ARMS = ("value", "sobolev")


@dataclass
class RerunConfig:
    params: list = field(default_factory=lambda: list(ALL_PARAMS))
    zs: list = field(default_factory=lambda: [3.6])
    arms: list = field(default_factory=lambda: ["value", "sobolev"])
    maxsize: int = 20
    niterations: int = 30
    populations: int = 8
    sobolev_lambda: float = 5.0
    seed: int = 0
    kmin: float = 0.001
    kmax: float = 0.04
    smart_kwargs: bool = True
    fiducial_overrides: dict | None = None
    prior_overrides: dict | None = None
    label: str = "quick"
    out_root: Path = field(default_factory=lambda: Path("results/tutorial_reruns"))
    basedir: str = "data/kodiaq_gp"

    @classmethod
    def quick(cls, **kw):
        return cls(label="quick", zs=[3.6], niterations=30, populations=8, **kw)

    @classmethod
    def full(cls, **kw):
        return cls(label="full", zs=[2.6, 3.6, 4.2],
                   niterations=PRODUCTION_BUDGET["niterations"],
                   populations=PRODUCTION_BUDGET["populations"],
                   maxsize=PRODUCTION_BUDGET["maxsize"],
                   sobolev_lambda=PRODUCTION_BUDGET["sobolev_lambda"], **kw)

    @property
    def run_dir(self) -> Path:
        d = Path(self.out_root) / f"rerun_{self.label}"
        resolved = str(d.resolve())
        if any(m in resolved for m in _PRODUCTION_MARKERS):
            raise ValueError(
                f"refusing to write into a production results dir: {d}. "
                "Tutorial reruns must stay under results/tutorial_reruns/.")
        return d

    def validate(self) -> None:
        bad = set(self.arms) - set(_VALID_ARMS)
        if bad:
            raise ValueError(f"invalid arm(s) {sorted(bad)}; use {_VALID_ARMS}.")
        if not self.params:
            raise ValueError("params is empty.")
        if self.niterations < 1 or self.populations < 1 or self.maxsize < 5:
            raise ValueError("budget knobs out of range.")
        unknown = set(self.params) - set(ALL_PARAMS)
        if unknown:
            raise ValueError(f"unknown param(s): {sorted(unknown)}")
```

- [ ] **Step 4: Run — verify pass** — `PYTHONPATH=src pytest tests/test_rerun_config.py -v` → PASS (5).

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/rerun.py tests/test_rerun_config.py
git commit -m "feat(rerun): RerunConfig with quick/full presets + isolation guard"
```

---

### Task 3: `cli_command_for` + `budget_warnings`

**Files:**
- Modify: `src/priya_forecast/rerun.py`
- Test: `tests/test_rerun_cli_and_warnings.py`

**Interfaces:**
- Consumes: `RerunConfig` (Task 2)
- Produces:
  - `cli_command_for(cfg: RerunConfig, param: str, z: float, arm: str) -> str`
  - `budget_warnings(cfg: RerunConfig) -> list[str]`

- [ ] **Step 1: Write failing tests**

```python
# tests/test_rerun_cli_and_warnings.py
from priya_forecast.rerun import RerunConfig, cli_command_for, budget_warnings


def test_cli_command_sobolev_arm_has_flags():
    cmd = cli_command_for(RerunConfig.full(), "ns", 3.6, "sobolev")
    assert "scripts/refit_one_param_single_z.py" in cmd
    assert "--param ns" in cmd and "--z 3.6" in cmd
    assert "--target-space log" in cmd and "--use-sobolev" in cmd
    assert "--sobolev-lambda 5" in cmd
    assert "--niterations 200" in cmd and "--populations 48" in cmd


def test_cli_command_value_arm_no_sobolev():
    cmd = cli_command_for(RerunConfig.full(), "tau0", 2.6, "value")
    assert "--use-sobolev" not in cmd
    assert "--target-space log" in cmd            # value baseline is log-target, plain MSE


def test_cli_command_notes_overrides_need_api():
    c = RerunConfig.quick(fiducial_overrides={"ns": 0.9})
    cmd = cli_command_for(c, "ns", 3.6, "value")
    assert "override" in cmd.lower()              # a comment noting CLI can't take overrides


def test_budget_warnings_flags_quick():
    w = budget_warnings(RerunConfig.quick())
    assert any("niter" in s.lower() for s in w)
    assert any("illustrative" in s.lower() or "production" in s.lower() for s in w)


def test_budget_warnings_silent_on_full():
    # full at production budget, all params, all z, both arms -> no budget shortfall
    w = budget_warnings(RerunConfig.full())
    assert w == []
```

- [ ] **Step 2: Run — verify fail** — FAIL (functions undefined).

- [ ] **Step 3: Implement (append to `rerun.py`)**

```python
def cli_command_for(cfg, param, z, arm):
    """The exact refit_one_param_single_z.py command equivalent to one grid cell."""
    parts = [
        "python scripts/refit_one_param_single_z.py",
        f"--param {param}", f"--z {z}",
        f"--basedir {cfg.basedir}",
        f"--output-dir {cfg.run_dir}/{arm}",
        "--target-space log",
        f"--maxsize {cfg.maxsize}",
        f"--niterations {cfg.niterations}",
        f"--populations {cfg.populations}",
        f"--seed {cfg.seed}",
    ]
    if arm == "sobolev":
        parts += ["--use-sobolev", f"--sobolev-lambda {cfg.sobolev_lambda:g}"]
    cmd = " \\\n    ".join(parts)
    if cfg.fiducial_overrides or cfg.prior_overrides:
        cmd += ("\n# NOTE: fiducial/prior overrides are set in this run; the CLI has "
                "no such flag. Use the notebook's run_grid() (Python API) for those.")
    return cmd


def budget_warnings(cfg):
    """Human-readable warnings when the config is below the production budget.
    Warnings only — never raises. Empty list means the run meets production budget."""
    w = []
    pb = PRODUCTION_BUDGET
    if cfg.niterations < pb["niterations"]:
        w.append(f"niterations={cfg.niterations} < production {pb['niterations']} "
                 "(fewer generations -> less-converged equations).")
    if cfg.populations < pb["populations"]:
        w.append(f"populations={cfg.populations} < production {pb['populations']}.")
    if cfg.maxsize < pb["maxsize"]:
        w.append(f"maxsize={cfg.maxsize} < production {pb['maxsize']}.")
    if "sobolev" in cfg.arms and cfg.sobolev_lambda != pb["sobolev_lambda"]:
        w.append(f"sobolev_lambda={cfg.sobolev_lambda} != production {pb['sobolev_lambda']}.")
    if set(cfg.params) != set(ALL_PARAMS):
        w.append(f"param subset ({len(cfg.params)}/11) -- not the full taxonomy.")
    if set(cfg.zs) != {2.6, 3.6, 4.2}:
        w.append(f"redshift subset {cfg.zs} -- production spans 2.6/3.6/4.2.")
    if cfg.fiducial_overrides or cfg.prior_overrides:
        w.append("fiducial/prior overrides are active -- results are for YOUR "
                 "hypothesis, not the paper's fiducial setup.")
    if w:
        w.append("These runs are ILLUSTRATIVE. The production numbers used a far larger "
                 "search budget (niter=200, populations=48, 5-seed band). Do not replace "
                 "the production results with a quick/tweaked run without meeting the "
                 "production budget on >=1 seed per arm.")
    return w
```

- [ ] **Step 4: Run — verify pass** — PASS (5).

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/rerun.py tests/test_rerun_cli_and_warnings.py
git commit -m "feat(rerun): cli_command_for + budget_warnings (warn-only)"
```

---

### Task 4: `run_grid` — drive the pipeline into the isolated dir

**Files:**
- Modify: `src/priya_forecast/rerun.py`
- Test: `tests/test_rerun_run_grid.py`

**Interfaces:**
- Consumes: `RerunConfig`; `parameters.override_params`; `single_z.refit.refit_one_param_single_z`; `single_z.config.{PipelineConfig,GPConfig,PySRConfig}`; `models.gp_model.GPModel`; `subprocess`.
- Produces: `run_grid(cfg, *, gp_loader=None, refit_fn=None, score_fn=None, progress=print, stamp="") -> Path` — returns `cfg.run_dir`. The three injectable callables default to the real implementations; tests inject fakes so CI needs no GP.
  - `gp_loader(basedir, z, k_grid) -> (gp_lf, gp_hf)`
  - `refit_fn(*, param_name, z, cfg, gp_lf, gp_hf, k_grid, out_dir) -> result`
  - `score_fn(pareto_csv, param, z, out_csv, basedir) -> None`

- [ ] **Step 1: Write failing tests (GP fully mocked)**

```python
# tests/test_rerun_run_grid.py
from pathlib import Path
import pandas as pd
import pytest
from priya_forecast.rerun import RerunConfig, run_grid


def _fake_gp_loader(basedir, z, k_grid):
    return ("gp_lf", "gp_hf")                       # opaque; refit_fn is faked too


def _make_refit_fn(record):
    def refit_fn(*, param_name, z, cfg, gp_lf, gp_hf, k_grid, out_dir):
        record.append((param_name, z, str(out_dir)))
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Complexity": [5], "Loss": [0.1],
                      "Equation": ["x0"]}).to_csv(Path(out_dir)/f"pareto_{param_name}.csv",
                                                  index=False)
        return object()
    return refit_fn


def _fake_score_fn(pareto_csv, param, z, out_csv, basedir):
    pd.DataFrame({"Complexity": [5], "Loss": [0.1], "grad_err": [0.2],
                  "value_mse": [1e-4], "n_keep": [40], "gate_pass": [True],
                  "x0_enters": [True]}).to_csv(out_csv, index=False)


def test_run_grid_writes_production_layout(tmp_path):
    rec = []
    cfg = RerunConfig.quick(params=["ns", "tau0"])
    cfg.out_root = tmp_path
    run_dir = run_grid(cfg, gp_loader=_fake_gp_loader,
                       refit_fn=_make_refit_fn(rec), score_fn=_fake_score_fn,
                       progress=lambda *_: None)
    assert run_dir == tmp_path / "rerun_quick"
    for arm in ("value", "sobolev"):
        d = run_dir / arm / "refit" / "z3.6"
        assert (d / "pareto_ns.csv").exists()
        assert (d / "grad_faith_ns.csv").exists()
        assert (d / "pareto_tau0.csv").exists()
    assert (run_dir / "RUN_MANIFEST.md").exists()
    # 2 params x 2 arms x 1 z = 4 refit calls
    assert len(rec) == 4


def test_run_grid_refuses_production_dir(tmp_path):
    cfg = RerunConfig.quick()
    cfg.out_root = Path("results/paper_production_20260630_perz_sobolev_z2.6-4.2")
    with pytest.raises(ValueError, match="production"):
        run_grid(cfg, gp_loader=_fake_gp_loader,
                 refit_fn=_make_refit_fn([]), score_fn=_fake_score_fn)


def test_run_grid_applies_overrides(monkeypatch, tmp_path):
    # override_params must be active during the refit call
    seen = {}
    from priya_forecast import parameters as P

    def refit_fn(*, param_name, z, cfg, gp_lf, gp_hf, k_grid, out_dir):
        seen["ns_fid"] = P.get_param("ns").fid          # read under the context
        Path(out_dir).mkdir(parents=True, exist_ok=True)
        pd.DataFrame({"Complexity": [5], "Loss": [0.1], "Equation": ["x0"]}
                     ).to_csv(Path(out_dir)/f"pareto_{param_name}.csv", index=False)
        return object()

    cfg = RerunConfig.quick(params=["ns"], arms=["value"],
                            fiducial_overrides={"ns": 0.90})
    cfg.out_root = tmp_path
    run_grid(cfg, gp_loader=_fake_gp_loader, refit_fn=refit_fn,
             score_fn=_fake_score_fn, progress=lambda *_: None)
    assert seen["ns_fid"] == 0.90
    assert P.get_param("ns").fid == 0.983               # restored after
```

- [ ] **Step 2: Run — verify fail** — FAIL (`run_grid` undefined).

- [ ] **Step 3: Implement `run_grid` (+ default real callables) in `rerun.py`**

Add imports at top of `rerun.py`: `import subprocess, sys, os` and `from priya_forecast.parameters import override_params`. Implement:

```python
def _real_gp_loader(basedir, z, k_grid):
    import numpy as np
    from priya_forecast.models.gp_model import GPModel
    from priya_forecast.parameters import fiducial_vector
    gp_lf = GPModel(basedir=basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=basedir, fidelity="hf", kf=k_grid)
    fid = np.asarray(fiducial_vector(), float)
    gp_lf.predict(fid, k_grid, z); gp_hf.predict(fid, k_grid, z)   # warm/validate
    return gp_lf, gp_hf


def _pipeline_cfg(cfg, arm, z, out_root_for_arm):
    from priya_forecast.single_z.config import PipelineConfig, GPConfig, PySRConfig
    use_sob = (arm == "sobolev")
    return PipelineConfig(
        mode="refit_and_forecast", redshift=z, output_dir=str(out_root_for_arm),
        gp=GPConfig(basedir=cfg.basedir),
        pysr=PySRConfig(niterations=cfg.niterations, maxsize=cfg.maxsize,
                        populations=cfg.populations, seed=cfg.seed,
                        smart_kwargs=cfg.smart_kwargs,
                        use_sobolev=use_sob, sobolev_lambda=cfg.sobolev_lambda,
                        use_anova_loss=False),
        target_space="log",                       # both arms train on log(P)
    )


def _real_refit_fn(*, param_name, z, cfg, gp_lf, gp_hf, k_grid, out_dir):
    from priya_forecast.single_z import refit as _refit
    return _refit.refit_one_param_single_z(
        param_name=param_name, z=z, cfg=cfg, gp_lf=gp_lf, gp_hf=gp_hf,
        k_grid=k_grid, out_dir=out_dir, save_artifacts=False)


def _real_score_fn(pareto_csv, param, z, out_csv, basedir):
    env = dict(os.environ)
    lya = env.get("LYA_EMULATOR", "/home/mfho/student_projects/lya_emulator_full")
    env["PYTHONPATH"] = f"src:{lya}" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    subprocess.run(
        [sys.executable, "scripts/eval_grad_faithfulness.py",
         "--pareto", str(pareto_csv), "--param", param, "--z", str(z),
         "--basedir", basedir, "--log-space", "--out", str(out_csv)],
        check=True, env=env)


def run_grid(cfg, *, gp_loader=None, refit_fn=None, score_fn=None,
             progress=print, stamp=""):
    """Run the grid (arm x z x param) into cfg.run_dir; return the run dir.
    Injectable callables default to the real GP-backed implementations; tests
    pass fakes so no emulator is needed in CI."""
    from priya_forecast.single_z.refit import kodiaq_k_grid
    cfg.validate()
    run_dir = cfg.run_dir                          # raises if inside production
    gp_loader = gp_loader or _real_gp_loader
    refit_fn = refit_fn or _real_refit_fn
    score_fn = score_fn or _real_score_fn
    run_dir.mkdir(parents=True, exist_ok=True)
    k_grid = kodiaq_k_grid(cfg.kmin, cfg.kmax, 48)

    n_done = 0
    for z in cfg.zs:
        progress(f"[z={z}] loading GP emulator ...")
        gp_lf, gp_hf = gp_loader(cfg.basedir, z, k_grid)
        for arm in cfg.arms:
            for param in cfg.params:
                out_dir = run_dir / arm / "refit" / f"z{z}"
                out_dir.mkdir(parents=True, exist_ok=True)
                progress(f"  fit {arm} {param} z={z} ...")
                try:
                    pcfg = _pipeline_cfg(cfg, arm, z, run_dir / arm)
                    with override_params(cfg.fiducial_overrides, cfg.prior_overrides):
                        refit_fn(param_name=param, z=z, cfg=pcfg,
                                 gp_lf=gp_lf, gp_hf=gp_hf, k_grid=k_grid,
                                 out_dir=out_dir)
                    pareto = out_dir / f"pareto_{param}.csv"
                    if pareto.exists():
                        score_fn(pareto, param, z, out_dir / f"grad_faith_{param}.csv",
                                 cfg.basedir)
                    n_done += 1
                except Exception as e:                # one bad param must not kill the grid
                    progress(f"  !! {arm} {param} z={z} FAILED: {e} (continuing)")
    _write_manifest(cfg, run_dir, n_done, stamp)
    progress(f"done: {n_done} fits -> {run_dir}")
    return run_dir


def _write_manifest(cfg, run_dir, n_done, stamp):
    lines = [
        f"# RERUN MANIFEST — {cfg.label}", "",
        f"- fits completed: {n_done}",
        f"- params: {cfg.params}", f"- zs: {cfg.zs}", f"- arms: {cfg.arms}",
        f"- budget: niter={cfg.niterations} pop={cfg.populations} maxsize={cfg.maxsize}",
        f"- sobolev_lambda: {cfg.sobolev_lambda}", f"- seed: {cfg.seed}",
        f"- fiducial_overrides: {cfg.fiducial_overrides}",
        f"- prior_overrides: {cfg.prior_overrides}",
        f"- stamp: {stamp}", "",
        "Illustrative tutorial run — NOT the production result. See "
        "results/paper_production_20260630_perz_sobolev_z2.6-4.2/ for the paper run.",
    ]
    (run_dir / "RUN_MANIFEST.md").write_text("\n".join(lines))
```

- [ ] **Step 4: Run — verify pass** — `PYTHONPATH=src pytest tests/test_rerun_run_grid.py -v` → PASS (3).

- [ ] **Step 5: Commit**

```bash
git add src/priya_forecast/rerun.py tests/test_rerun_run_grid.py
git commit -m "feat(rerun): run_grid drives refit+scorer into isolated tutorial dir"
```

---

### Task 5: `compare_to_production`

**Files:**
- Modify: `src/priya_forecast/rerun.py`
- Test: `tests/test_rerun_compare.py`

**Interfaces:**
- Consumes: `grad_faith_io.read_grad_faith_sidecar`, `knee_row`.
- Produces: `compare_to_production(run_dir, production_dir=DEFAULT_PRODUCTION_DIR, *, zs=None, arms=None, gate=0.25) -> pandas.DataFrame` with columns `param, z, arm, grad_err_rerun, grad_err_prod, d_grad_err, value_mse_rerun, value_mse_prod, verdict_rerun, verdict_prod, flipped, flag` where `flag in {"worse","better","similar","n/a"}`. Never raises; missing sidecars → row with `flag="n/a"`.
- Also: `DEFAULT_PRODUCTION_DIR = Path("results/paper_production_20260630_perz_sobolev_z2.6-4.2")`.

- [ ] **Step 1: Write failing tests (fixture sidecars, no GP)**

```python
# tests/test_rerun_compare.py
from pathlib import Path
import pandas as pd
from priya_forecast.rerun import compare_to_production


def _sidecar(path, grad_err, value_mse):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as fh:
        fh.write("# fixture\n")
    pd.DataFrame({"Complexity": [8], "Loss": [0.1], "grad_err": [grad_err],
                  "value_mse": [value_mse], "n_keep": [40], "gate_pass": [grad_err <= 0.25],
                  "x0_enters": [True]}).to_csv(path, mode="a", index=False)


def test_compare_reports_deltas_and_flags(tmp_path):
    run = tmp_path / "rerun"; prod = tmp_path / "prod"
    _sidecar(run / "sobolev/refit/z3.6/grad_faith_ns.csv", 0.40, 2e-4)   # worse
    _sidecar(prod / "sobolev/refit/z3.6/grad_faith_ns.csv", 0.19, 1e-4)
    df = compare_to_production(run, prod, zs=[3.6], arms=["sobolev"])
    row = df[df.param == "ns"].iloc[0]
    assert round(row.d_grad_err, 3) == round(0.40 - 0.19, 3)
    assert row.flag == "worse"                    # higher grad_err
    assert bool(row.flipped) is True              # prod faithful(<=.25), rerun not


def test_compare_missing_sidecar_is_na_not_error(tmp_path):
    run = tmp_path / "rerun"; prod = tmp_path / "prod"
    _sidecar(prod / "sobolev/refit/z3.6/grad_faith_ns.csv", 0.19, 1e-4)  # only prod
    df = compare_to_production(run, prod, zs=[3.6], arms=["sobolev"], )
    row = df[df.param == "ns"].iloc[0]
    assert row.flag == "n/a"
```

- [ ] **Step 2: Run — verify fail** — FAIL.

- [ ] **Step 3: Implement (append to `rerun.py`)**

```python
DEFAULT_PRODUCTION_DIR = Path("results/paper_production_20260630_perz_sobolev_z2.6-4.2")


def _knee_metrics(run_dir, arm, z, param):
    from priya_forecast.grad_faith_io import read_grad_faith_sidecar, knee_row
    path = Path(run_dir) / arm / "refit" / f"z{z}" / f"grad_faith_{param}.csv"
    if not path.exists():
        return None
    try:
        row = knee_row(read_grad_faith_sidecar(path))
        return float(row["grad_err"]), float(row["value_mse"])
    except Exception:
        return None


def compare_to_production(run_dir, production_dir=DEFAULT_PRODUCTION_DIR, *,
                          zs=None, arms=None, gate=0.25, params=None):
    """Per-parameter deviation of a rerun vs the committed production sidecars.
    Print-friendly DataFrame; never raises. flag: worse/better/similar/n_a."""
    import numpy as np
    import pandas as pd
    zs = zs or [3.6]
    arms = arms or ["value", "sobolev"]
    params = params or list(ALL_PARAMS)
    rows = []
    for arm in arms:
        for z in zs:
            for p in params:
                r = _knee_metrics(run_dir, arm, z, p)
                q = _knee_metrics(production_dir, arm, z, p)
                if r is None or q is None:
                    rows.append(dict(param=p, z=z, arm=arm, grad_err_rerun=np.nan,
                                     grad_err_prod=(q[0] if q else np.nan),
                                     d_grad_err=np.nan, value_mse_rerun=np.nan,
                                     value_mse_prod=(q[1] if q else np.nan),
                                     verdict_rerun="n/a", verdict_prod=(
                                         "faithful" if q and q[0] <= gate else
                                         ("unfaithful" if q else "n/a")),
                                     flipped=False, flag="n/a"))
                    continue
                ge_r, vm_r = r; ge_q, vm_q = q
                d = ge_r - ge_q
                vr = "faithful" if ge_r <= gate else "unfaithful"
                vq = "faithful" if ge_q <= gate else "unfaithful"
                flag = ("similar" if abs(d) < 0.05 else
                        ("worse" if d > 0 else "better"))
                rows.append(dict(param=p, z=z, arm=arm, grad_err_rerun=ge_r,
                                 grad_err_prod=ge_q, d_grad_err=d,
                                 value_mse_rerun=vm_r, value_mse_prod=vm_q,
                                 verdict_rerun=vr, verdict_prod=vq,
                                 flipped=(vr != vq), flag=flag))
    return pd.DataFrame(rows)
```

- [ ] **Step 4: Run — verify pass** — PASS (2).

- [ ] **Step 5: Full-module regression**

Run: `PYTHONPATH=src pytest tests/test_rerun_config.py tests/test_rerun_cli_and_warnings.py tests/test_rerun_run_grid.py tests/test_rerun_compare.py tests/test_parameters_overrides.py -q`
Expected: PASS (all).

- [ ] **Step 6: Commit**

```bash
git add src/priya_forecast/rerun.py tests/test_rerun_compare.py
git commit -m "feat(rerun): compare_to_production deviation report (warn-only)"
```

---

### Task 6: Builder + notebook (`notebooks/_build_rerun_paper.py` → `rerun_paper.ipynb`)

**Files:**
- Create: `notebooks/_build_rerun_paper.py` (gitignored by `notebooks/_build_*.py`)
- Create (generated, committed): `notebooks/rerun_paper.ipynb`
- Test: `tests/test_rerun_notebook_builds.py`

**Interfaces:**
- Consumes: all of `rerun.py` + `paper_figures.load_run`.

- [ ] **Step 1: Write the builder** `notebooks/_build_rerun_paper.py`

Mirror `_build_reproduce_paper.py` exactly (copy its `md()`/`code()` helpers and the JSON-write footer). Cell sequence (each `code()` cell must contain only `"..."` strings and `#` comments — never `'''`):

1. `md` — title + banner: this is **Tier 3** (needs the GP emulator); one paragraph contrasting with `reproduce_paper.ipynb` (that one reproduces from committed CSVs; this one *re-derives* them so you can tweak). Match `writing.md` voice.
2. `md` — "## 1. Get the GP emulator" + `code` detection cell:

```python
import os, sys
from pathlib import Path
_root = Path.cwd()
if (_root / "src").is_dir(): sys.path.insert(0, str(_root / "src"))
LYA_EMULATOR = os.environ.get("LYA_EMULATOR", "/home/mfho/student_projects/lya_emulator_full")
GP_BASEDIR = "data/kodiaq_gp"
if LYA_EMULATOR not in sys.path: sys.path.insert(0, LYA_EMULATOR)
os.environ["PYTHONPATH"] = "src:" + LYA_EMULATOR          # for the scorer subprocess
have_pkg = Path(LYA_EMULATOR, "lyaemu").is_dir()
try:
    import GPy; have_gpy = True
except Exception: have_gpy = False
have_gp = Path(GP_BASEDIR).is_dir()
EMULATOR_READY = have_pkg and have_gpy and have_gp
print("lyaemu package:", have_pkg, "| GPy:", have_gpy, "| data/kodiaq_gp:", have_gp)
if not EMULATOR_READY:
    print(chr(10).join([
        "Emulator not ready. To provision (Tier 3):",
        "  # 1. package (NOT pip-installed): clone + PYTHONPATH",
        "  git clone https://github.com/sbird/lya_emulator $LYA_EMULATOR",
        "  pip install GPy emukit   # numpy must stay <2 (GPy ABI)",
        "  # 2. data/kodiaq_gp (~43 MB):",
        "  #    PRIMARY (collaborators): fetch the hosted archive:",
        "  #    curl -L <ARCHIVE_URL> -o kodiaq_gp.tar.gz && tar xzf kodiaq_gp.tar.gz -C data/",
        "  #    (only if you have the PRIVATE PRIYA training set:)",
        "  #    python scripts/prep_kodiaq_gp.py --source <PRIYA_SET> --dest data/kodiaq_gp",
    ]))
```

   Add a `# TODO(user): set <ARCHIVE_URL> once the kodiaq_gp archive is hosted (Zenodo / GitHub release / shared drive)` comment right above the curl line.
3. `md` — "## 2. What the pipeline does" — plain-language walkthrough of one per-`(param, z)` fit → grad-faith gate (0.25) → additive combine, referencing the paper's Methods sections. Match `writing.md`.
4. `md` + `code` — "## 3. Configure your run":

```python
from priya_forecast.rerun import RerunConfig, budget_warnings, cli_command_for
QUICK = True     # flip to False for the full 11 x {2.6,3.6,4.2} x {value,sobolev} run
cfg = RerunConfig.quick() if QUICK else RerunConfig.full()
# --- knobs you can tweak (uncomment / edit) ---
# cfg.sobolev_lambda = 5.0          # derivative-matching strength
# cfg.maxsize = 20                   # equation complexity ceiling
# cfg.niterations = 30; cfg.populations = 8   # search budget (quick defaults)
# cfg.params = ["ns", "tau0", "Ap", "hub"]    # subset for speed
# cfg.seed = 0
# --- physics overrides: test your own hypothesis (Python-API only) ---
# cfg.fiducial_overrides = {"ns": 0.95}                 # move the fiducial n_s
# cfg.prior_overrides   = {"Ap": (1.0, 3.0)}            # widen the A_P prior
cfg.validate()
print("run dir:", cfg.run_dir)      # under results/tutorial_reruns/ (isolated, git-ignored)
```

5. `md` + `code` — "## 4. Budget check" — print `budget_warnings(cfg)` and the standing disclaimer.
6. `md` + `code` — "## 5. Run":

```python
from priya_forecast.rerun import run_grid
if EMULATOR_READY:
    run_dir = run_grid(cfg)
else:
    run_dir = cfg.run_dir
    print("Emulator not ready -- skipping the run. Equivalent CLI for one cell:")
    print(cli_command_for(cfg, cfg.params[0], cfg.zs[0], cfg.arms[0]))
```

7. `md` + `code` — "## 6. Full run on a cluster" — documented (not executed): print `submit_paper_production.sh` usage + the per-arm `cli_command_for(...)` for the seed-band/sensitivity arms.
8. `md` + `code` — "## 7. How far did your run move? (vs production)":

```python
from priya_forecast.rerun import compare_to_production
if EMULATOR_READY:
    cmp = compare_to_production(run_dir, zs=cfg.zs, arms=cfg.arms, params=cfg.params)
    import pandas as pd; pd.set_option("display.width", 160)
    print(cmp[["param","z","arm","grad_err_rerun","grad_err_prod","d_grad_err","flag","flipped"]].to_string(index=False))
```

9. `md` + `code` — "## 8. Regenerate the paper figures from YOUR run":

```python
import priya_forecast.paper_figures as pf
if EMULATOR_READY:
    run = pf.load_run(str(run_dir), z=cfg.zs[0])      # retarget the paper's plotting API
    display(pf.taxonomy(run))                          # the per-param taxonomy table
    pf.plot_scorecard(run)
    out = run_dir / "figures"; out.mkdir(exist_ok=True)
    pf.plot_pareto_faithfulness(run, out / "pareto_faithfulness.png")
    print("figures written under", out)
```

10. `md` — "## 9. Knobs to try" — 3–4 concrete hypotheses, each with the one line to change + expected direction: (a) raise `sobolev_lambda` → lower `grad_err` on borderline params like `ns`; (b) drop `exp` from operators (via `smart_kwargs=False` note) → simpler but possibly less faithful; (c) shift `fiducial_overrides={"tau0": ...}` → see the taxonomy move; (d) widen a prior → weaker per-param slope. Match `writing.md`.

Footer: same JSON write as `_build_reproduce_paper.py` (`NB.write_text(json.dumps(nb, indent=1))` with the same nbformat 4 skeleton).

- [ ] **Step 2: Build the notebook**

Run: `PYTHONPATH=src python notebooks/_build_rerun_paper.py`
Expected: writes `notebooks/rerun_paper.ipynb`, prints a success line.

- [ ] **Step 3: Write the static-validity test**

```python
# tests/test_rerun_notebook_builds.py
import json, subprocess, sys
from pathlib import Path


def test_builder_produces_valid_notebook(tmp_path):
    subprocess.run([sys.executable, "notebooks/_build_rerun_paper.py"],
                   check=True, env={"PYTHONPATH": "src", "PATH": __import__("os").environ["PATH"]})
    nb = json.loads(Path("notebooks/rerun_paper.ipynb").read_text())
    assert nb["nbformat"] == 4
    assert len(nb["cells"]) >= 10
    srcs = "".join("".join(c["source"]) for c in nb["cells"])
    assert "results/tutorial_reruns" in srcs
    assert "RerunConfig" in srcs and "compare_to_production" in srcs
    assert "load_run" in srcs
    # no code cell contains a triple-quote (builder invariant)
    for c in nb["cells"]:
        if c["cell_type"] == "code":
            assert "'''" not in "".join(c["source"])
```

- [ ] **Step 4: Run — verify pass**

Run: `PYTHONPATH=src pytest tests/test_rerun_notebook_builds.py -v`
Expected: PASS.

- [ ] **Step 5: Confirm the config cell imports without an emulator**

Run: `PYTHONPATH=src python -c "from priya_forecast.rerun import RerunConfig; c=RerunConfig.quick(); c.validate(); print(c.run_dir)"`
Expected: prints `results/tutorial_reruns/rerun_quick` (no GP needed).

- [ ] **Step 6: Commit**

```bash
git add notebooks/rerun_paper.ipynb tests/test_rerun_notebook_builds.py
git commit -m "feat(notebook): rerun_paper.ipynb builder + generated notebook"
# NOTE: notebooks/_build_rerun_paper.py is gitignored (notebooks/_build_*.py) -- do not force-add.
```

---

### Task 7: Docs + gitignore

**Files:**
- Modify: `.gitignore` (add explicit tutorial-reruns ignore line for clarity)
- Modify: `REPRODUCE.md` (Tier-3 pointer + the emulator-archive note)
- Modify: `README.md` (one-line pointer)

- [ ] **Step 1: `.gitignore` — add explicit clarity line** (already ignored by `results/*`, but make intent visible). Find the block with `results/_repro_scratch/` and add beside it:

```
results/tutorial_reruns/
```

- [ ] **Step 2: `REPRODUCE.md`** — add a short "## Tier 3 — re-run and tweak (`rerun_paper.ipynb`)" subsection: what it is, that it needs the emulator, the two provisioning routes (hosted archive PRIMARY for collaborators; build-from-source needs the private PRIYA set), and that output is isolated under `results/tutorial_reruns/`. Include the `# TODO(user): set <ARCHIVE_URL>` note.

- [ ] **Step 3: `README.md`** — one line under the reproduce pointer: "To re-run the PySR pipeline and test your own hypotheses, see `notebooks/rerun_paper.ipynb` (Tier 3; needs the GP emulator)."

- [ ] **Step 4: Verify gitignore**

Run: `git check-ignore -v results/tutorial_reruns/foo.png`
Expected: matches a `.gitignore` rule (ignored).

- [ ] **Step 5: Commit**

```bash
git add .gitignore REPRODUCE.md README.md
git commit -m "docs: Tier-3 rerun_paper pointer + tutorial_reruns ignore"
```

---

### Task 8: Verify (adversarial)

**Files:** none (verification only)

- [ ] **Step 1: Full suite green**

Run: `PYTHONPATH=src pytest tests/ -q`
Expected: all prior tests still pass + the 5 new test files pass. Report the pass/skip counts.

- [ ] **Step 2: No production code broken** — confirm the only production-source diffs are `parameters.py` (two additive functions) and the new `rerun.py`; `refit_1d_pysr.py`, `single_z/refit.py`, the CLI, and SLURM are untouched.

Run: `git diff --stat main..HEAD -- src/ scripts/`
Expected: shows `parameters.py` (+~40 lines), `rerun.py` (new). No other `src/`/`scripts/` files.

- [ ] **Step 3: Isolation guard holds** — a config pointed at a production dir raises.

Run: `PYTHONPATH=src python -c "from priya_forecast.rerun import RerunConfig; from pathlib import Path; c=RerunConfig.quick(); c.out_root=Path('results/paper_production_20260630_perz_sobolev_z2.6-4.2'); c.run_dir"`
Expected: raises `ValueError` mentioning "production".

- [ ] **Step 4: Figures retarget check (no GP)** — build a tiny fake run dir with 2 params' sidecars for value+sobolev at z=3.6, call `pf.load_run(fake_dir)` + `pf.taxonomy(run)`; confirm it returns rows without needing the emulator. (This proves the notebook's step-8 wiring against a rerun-shaped dir.)

- [ ] **Step 5: Adversarial review** — re-read `run_grid` for: (a) does the per-param `except` swallow the isolation-guard/validate errors? (No — those fire before the loop.) (b) does `override_params` restore on a mid-grid exception? (finally in the CM.) (c) is `target_space="log"` set for the value arm too (required so the scorer's log-space gate is consistent)? (Yes.) Fix anything found; re-run Step 1.

---

## Self-Review

**Spec coverage:** §1 rerun.py → Tasks 2–5; §2 override hook → Task 1; §3 builder/notebook → Task 6; §4 tests+docs → embedded per task + Task 7; §5 provisioning → Task 6 cell 2 + Task 7; §6 guards → Tasks 3 (budget_warnings) + 5 (compare) + Task 6 cells 4/7; isolation → Task 2 guard + Task 6 config note + Task 7 gitignore. All covered.

**Placeholder scan:** every code step carries real code; the only deliberate fill-in is `<ARCHIVE_URL>`, which is a *user* action explicitly marked `TODO(user)` (the hosting decision is theirs, per the spec's open question) — not an engineer placeholder.

**Type consistency:** `RerunConfig` fields/classmethods identical across Tasks 2–6; `run_grid(cfg, *, gp_loader, refit_fn, score_fn, progress, stamp)` signature matches the injected fakes in Task 4 tests; `compare_to_production(run_dir, production_dir, *, zs, arms, gate, params)` columns match the Task 5 test assertions; `override_params`/`with_overrides` signatures match Task 1 tests.
