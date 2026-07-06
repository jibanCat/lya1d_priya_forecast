"""Re-run the paper's per-(param, z) PySR pipeline into an isolated tutorial dir.

Collaborator-facing: drives the SAME refit_one_param_single_z the CLI calls and
the SAME eval_grad_faithfulness.py scorer, so a rerun is byte-comparable to the
paper. Output goes under results/tutorial_reruns/ (git-ignored, never overwrites
the production run). See notebooks/rerun_paper.ipynb.
"""
from __future__ import annotations

import os
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path

from priya_forecast.parameters import PARAM_NAMES, override_params

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
        f"--kmin {cfg.kmin}", f"--kmax {cfg.kmax}",
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


def _real_gp_loader(basedir, k_grid):
    from priya_forecast.models.gp_model import GPModel
    gp_lf = GPModel(basedir=basedir, fidelity="lf", kf=k_grid)
    gp_hf = GPModel(basedir=basedir, fidelity="hf", kf=k_grid)
    return gp_lf, gp_hf


def _real_regen_fn(gp_lf, gp_hf, params, zs, k_grid, out_dir):
    """Regenerate the LF/HF 1pvar sweep the scorer loads, into out_dir.
    Called by run_grid INSIDE override_params, so the sweep matches this run's
    fiducial/prior. Reuses the production regenerate_param + write_1pvar_hdf5."""
    import numpy as np
    from priya_forecast.single_z.training_data import regenerate_param, write_1pvar_hdf5
    z_grid = np.asarray(zs, dtype=float)
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    for pname in params:
        gen = regenerate_param(gp_lf=gp_lf, gp_hf=gp_hf, param_name=pname,
                               z_grid=z_grid, k_grid=k_grid)
        for fidelity in ("lf", "hf"):
            write_1pvar_hdf5(
                Path(out_dir) / f"{fidelity}_{pname}_npoints50.hdf5",
                params=gen[f"params_{fidelity}"], kfkms=gen[f"kfkms_{fidelity}"],
                flux_vectors=gen[f"flux_{fidelity}"], zout=gen["zout"])


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


def _real_score_fn(pareto_csv, param, z, out_csv, basedir, data_1pvar,
                   fid_ov=None, prior_ov=None, kmin=0.001, kmax=0.04):
    import json
    env = dict(os.environ)
    lya = env.get("LYA_EMULATOR", "/home/mfho/student_projects/lya_emulator_full")
    env["PYTHONPATH"] = f"src:{lya}" + (":" + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    if fid_ov:
        env["PRIYA_FIDUCIAL_OVERRIDES"] = json.dumps(fid_ov)
    if prior_ov:
        env["PRIYA_PRIOR_OVERRIDES"] = json.dumps(prior_ov)
    subprocess.run(
        [sys.executable, "scripts/eval_grad_faithfulness.py",
         "--pareto", str(pareto_csv), "--param", param, "--z", str(z),
         "--basedir", basedir, "--data-1pvar", str(data_1pvar),
         "--kmin", str(kmin), "--kmax", str(kmax),
         "--log-space", "--out", str(out_csv)],
        check=True, env=env)


def run_grid(cfg, *, gp_loader=None, regen_fn=None, refit_fn=None, score_fn=None,
             progress=print, stamp=""):
    """Run the grid (arm x z x param) into cfg.run_dir; return the run dir.
    Injectable callables default to the real GP-backed implementations; tests
    pass fakes so no emulator is needed in CI."""
    from priya_forecast.single_z.refit import kodiaq_k_grid
    cfg.validate()
    run_dir = cfg.run_dir                          # raises if inside production
    gp_loader = gp_loader or _real_gp_loader
    regen_fn = regen_fn or _real_regen_fn
    refit_fn = refit_fn or _real_refit_fn
    score_fn = score_fn or _real_score_fn
    run_dir.mkdir(parents=True, exist_ok=True)
    k_grid = kodiaq_k_grid(cfg.kmin, cfg.kmax, 48)
    data_1pvar = run_dir / "_1pvar"

    progress("loading GP emulator ...")
    gp_lf, gp_hf = gp_loader(cfg.basedir, k_grid)

    progress("regenerating run-local 1pvar sweep ...")
    with override_params(cfg.fiducial_overrides, cfg.prior_overrides):
        regen_fn(gp_lf, gp_hf, cfg.params, cfg.zs, k_grid, data_1pvar)

    n_done = 0
    for arm in cfg.arms:
        for z in cfg.zs:
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
                        score_fn(pareto, param, z,
                                 out_dir / f"grad_faith_{param}.csv",
                                 cfg.basedir, data_1pvar,
                                 cfg.fiducial_overrides, cfg.prior_overrides,
                                 cfg.kmin, cfg.kmax)
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
    Print-friendly DataFrame; never raises. flag: worse/better/similar/n/a."""
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
