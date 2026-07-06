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
