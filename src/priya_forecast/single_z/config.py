"""YAML schema for the single-z student pipeline.

One file controls everything: mode, redshift, parameter subset, k-range,
data source, GP basedir, normalization, PySR refit knobs, Pareto-pick
rules, combine method, output dir.

Schema (with defaults):

    mode: forecast_only            # gp_only | forecast_only | refit_and_forecast
    redshift: 3.6
    output_dir: results/single_z_run/
    parameters: [ns, Ap, hub, omegamh2, herei, heref, alphaq, hireionz, bhfeedback, dtau0, tau0]
    k_range: {min: 0.001, max: 0.04}

    data:
      source: kodiaq               # kodiaq | eboss_dr14
      cov_scale: 1.0
      conservative: true           # KSData conservative=True (drops first 4 bins)
      mock_data: gp                # gp | kodiaq (for KSData only)

    gp:
      basedir: data/kodiaq_gp
      hires_subdir: hires

    normalization:
      mode: auto                   # auto | identity | mean_flux
      fix: {r: 0.8}

    pysr:                          # only used when mode == refit_and_forecast
      smart_kwargs: true
      use_anova_loss: false
      niterations: 50
      populations: 24
      procs: 4
      maxsize: 20
      seed: 0

    combine: multiplicative        # multiplicative | additive | joint
    fiducial_p1d_cache: null       # path to .npz; auto-create if null

    pareto_csvs:
      source: bundled_baseline     # bundled_baseline | per_parameter | from_refit
      per_parameter: {}            # only when source == per_parameter

    fisher:
      step_frac: 0.01
      rel_tol: 0.01

The schema is intentionally flat at the top — the YAML reads top-down as
"what does the run look like". Subblocks group settings that belong together.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from priya_forecast.parameters import PARAM_NAMES

VALID_MODES = ("gp_only", "forecast_only", "refit_and_forecast")
VALID_DATA_SOURCES = ("kodiaq", "eboss_dr14")
VALID_NORM_MODES = ("auto", "identity", "mean_flux")
VALID_COMBINES = ("multiplicative", "additive", "joint")
VALID_PARETO_SOURCES = ("bundled_baseline", "per_parameter", "from_refit")
VALID_PICK_RULES = ("best_loss",)
VALID_PICK_PREFIXES = ("complexity_le:", "accuracy_at:", "row:")


def _is_valid_pick(rule: str) -> bool:
    if rule in VALID_PICK_RULES:
        return True
    return any(rule.startswith(p) and rule[len(p):] for p in VALID_PICK_PREFIXES)


@dataclass
class KRange:
    min: float = 0.001
    max: float = 0.04

    def validate(self) -> None:
        if self.min <= 0 or self.max <= self.min:
            raise ValueError(f"k_range invalid: min={self.min}, max={self.max}.")


@dataclass
class DataConfig:
    source: str = "kodiaq"
    cov_scale: float = 1.0
    conservative: bool = True
    mock_data: str = "gp"          # KSData only: gp | kodiaq

    def validate(self) -> None:
        if self.source not in VALID_DATA_SOURCES:
            raise ValueError(f"data.source must be one of {VALID_DATA_SOURCES}.")
        if self.cov_scale <= 0:
            raise ValueError(f"data.cov_scale must be > 0.")
        if self.mock_data not in {"gp", "kodiaq"}:
            raise ValueError("data.mock_data must be 'gp' or 'kodiaq'.")


@dataclass
class GPConfig:
    basedir: str = "data/kodiaq_gp"
    hires_subdir: str = "hires"

    def validate(self) -> None:
        if not Path(self.basedir).exists():
            raise ValueError(
                f"gp.basedir does not exist: {self.basedir}. "
                f"Run `python scripts/prep_kodiaq_gp.py --source <SRC> --dest data/kodiaq_gp`."
            )


@dataclass
class NormalizationConfig:
    mode: str = "auto"
    fix: dict[str, float] = field(default_factory=lambda: {"r": 0.8})

    def validate(self) -> None:
        if self.mode not in VALID_NORM_MODES:
            raise ValueError(f"normalization.mode must be one of {VALID_NORM_MODES}.")


@dataclass
class PySRConfig:
    smart_kwargs: bool = True
    use_anova_loss: bool = False
    niterations: int = 50
    populations: int = 24
    procs: int = 4
    maxsize: int = 20
    seed: int = 0

    def validate(self) -> None:
        if self.niterations < 1:
            raise ValueError("pysr.niterations must be >= 1.")
        if self.populations < 1 or self.procs < 1 or self.maxsize < 5:
            raise ValueError("pysr.{populations,procs,maxsize} must be reasonable.")


@dataclass
class FisherConfig:
    step_frac: float = 0.01
    rel_tol: float = 0.01

    def validate(self) -> None:
        if not 0 < self.step_frac < 1 or not 0 < self.rel_tol < 1:
            raise ValueError("fisher.{step_frac,rel_tol} must be in (0, 1).")


@dataclass
class ParetoEntry:
    pareto_csv: str
    pick: str = "best_loss"
    variables: list[str] | None = None
    fiducial: float | None = None

    def validate(self, param_name: str) -> None:
        if not _is_valid_pick(self.pick):
            raise ValueError(
                f"parameters.{param_name}.pick={self.pick!r} invalid. "
                f"Valid: best_loss / complexity_le:N / accuracy_at:tol / row:I."
            )


@dataclass
class ParetoCSVsConfig:
    source: str = "bundled_baseline"
    per_parameter: dict[str, ParetoEntry] = field(default_factory=dict)

    def validate(self, parameters: list[str]) -> None:
        if self.source not in VALID_PARETO_SOURCES:
            raise ValueError(f"pareto_csvs.source must be one of {VALID_PARETO_SOURCES}.")
        if self.source == "per_parameter":
            missing = set(parameters) - set(self.per_parameter)
            if missing:
                raise ValueError(
                    f"pareto_csvs.source=per_parameter but missing entries for: {sorted(missing)}."
                )
            for name, entry in self.per_parameter.items():
                if name not in PARAM_NAMES:
                    raise ValueError(f"Unknown PRIYA parameter in pareto_csvs: {name!r}.")
                entry.validate(name)


@dataclass
class PipelineConfig:
    mode: str = "forecast_only"
    redshift: float = 3.6
    output_dir: str = "results/single_z_run/"
    parameters: list[str] = field(
        default_factory=lambda: list(PARAM_NAMES)
    )
    k_range: KRange = field(default_factory=KRange)
    data: DataConfig = field(default_factory=DataConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    pysr: PySRConfig = field(default_factory=PySRConfig)
    combine: str = "multiplicative"
    fiducial_p1d_cache: str | None = None
    pareto_csvs: ParetoCSVsConfig = field(default_factory=ParetoCSVsConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}.")
        if not 2.2 <= self.redshift <= 4.6:
            raise ValueError(f"redshift {self.redshift} outside [2.2, 4.6].")
        unknown = set(self.parameters) - set(PARAM_NAMES)
        if unknown:
            raise ValueError(f"Unknown PRIYA parameters: {sorted(unknown)}.")
        if self.combine not in VALID_COMBINES:
            raise ValueError(f"combine must be one of {VALID_COMBINES}.")
        self.k_range.validate()
        self.data.validate()
        self.gp.validate()
        self.normalization.validate()
        self.pysr.validate()
        self.fisher.validate()
        self.pareto_csvs.validate(self.parameters)


def _build_pareto_entries(raw: dict[str, Any]) -> dict[str, ParetoEntry]:
    out: dict[str, ParetoEntry] = {}
    for name, body in raw.items():
        if not isinstance(body, dict):
            raise ValueError(f"pareto_csvs.per_parameter.{name} must be a mapping.")
        if "pareto_csv" not in body:
            raise ValueError(f"pareto_csvs.per_parameter.{name} missing pareto_csv.")
        out[name] = ParetoEntry(**body)
    return out


def load_config(path: str | Path) -> PipelineConfig:
    """Load + validate a single-z pipeline YAML."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}

    cfg = PipelineConfig()
    for key, value in raw.items():
        if not hasattr(cfg, key):
            raise ValueError(f"Unknown top-level key in config: {key!r}.")
        if key == "k_range":
            cfg.k_range = KRange(**value)
        elif key == "data":
            cfg.data = DataConfig(**value)
        elif key == "gp":
            cfg.gp = GPConfig(**value)
        elif key == "normalization":
            cfg.normalization = NormalizationConfig(**value)
        elif key == "pysr":
            cfg.pysr = PySRConfig(**value)
        elif key == "fisher":
            cfg.fisher = FisherConfig(**value)
        elif key == "pareto_csvs":
            entries = _build_pareto_entries(value.get("per_parameter", {}))
            cfg.pareto_csvs = ParetoCSVsConfig(
                source=value.get("source", "bundled_baseline"),
                per_parameter=entries,
            )
        else:
            setattr(cfg, key, value)

    cfg.validate()
    return cfg
