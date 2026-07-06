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
