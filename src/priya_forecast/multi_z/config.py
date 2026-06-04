"""YAML schema for the multi-z forecast pipeline (Stage 7).

Mirrors single_z/config.py, replacing the scalar `redshift` with a
`z_min`/`z_max` range. The Fisher is computed on one z-spanning
KSDataLikelihood; the returned Fisher is F = Σ_z F(z).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from priya_forecast.parameters import PARAM_NAMES
from priya_forecast.single_z.config import (
    DataConfig, FisherConfig, GPConfig, KRange, NormalizationConfig,
    ParetoCSVsConfig, PySRConfig, VALID_COMBINES, VALID_DATA_SOURCES,
    VALID_MODES, VALID_PARETO_SOURCES, VALID_TARGET_SPACES, _build_pareto_entries,
    _is_valid_pick,
)


@dataclass
class MultiZPipelineConfig:
    mode: str = "forecast_only"
    z_min: float = 2.6
    z_max: float = 4.2
    output_dir: str = "results/multi_z_run/"
    parameters: list[str] = field(default_factory=lambda: list(PARAM_NAMES))
    k_range: KRange = field(default_factory=KRange)
    data: DataConfig = field(default_factory=DataConfig)
    gp: GPConfig = field(default_factory=GPConfig)
    normalization: NormalizationConfig = field(default_factory=NormalizationConfig)
    pysr: PySRConfig = field(default_factory=PySRConfig)
    combine: str = "additive"
    pick: str = "best_loss"
    target_space: str = "linear"
    fiducial_p1d_cache: str | None = None
    pareto_csvs: ParetoCSVsConfig = field(default_factory=ParetoCSVsConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)

    def validate(self) -> None:
        if self.mode not in VALID_MODES:
            raise ValueError(f"mode must be one of {VALID_MODES}.")
        if not 2.2 <= self.z_min <= 4.6 or not 2.2 <= self.z_max <= 4.6:
            raise ValueError(f"z_min/z_max must lie in [2.2, 4.6].")
        if self.z_min > self.z_max:
            raise ValueError(f"z_min ({self.z_min}) must be <= z_max ({self.z_max}).")
        unknown = set(self.parameters) - set(PARAM_NAMES)
        if unknown:
            raise ValueError(f"Unknown PRIYA parameters: {sorted(unknown)}.")
        if self.combine not in VALID_COMBINES:
            raise ValueError(f"combine must be one of {VALID_COMBINES}.")
        if not _is_valid_pick(self.pick):
            raise ValueError(f"pick={self.pick!r} invalid.")
        if self.target_space not in VALID_TARGET_SPACES:
            raise ValueError(f"target_space must be one of {VALID_TARGET_SPACES}.")
        if self.target_space == "log" and self.combine != "additive":
            raise ValueError(
                "target_space='log' requires combine='additive' "
                "(log-space only supports the local_anchored combine)."
            )
        self.k_range.validate()
        self.data.validate()
        self.gp.validate()
        self.normalization.validate()
        self.pysr.validate()
        self.fisher.validate()
        self.pareto_csvs.validate(self.parameters)


def load_config(path: str | Path) -> MultiZPipelineConfig:
    """Load + validate a multi-z pipeline YAML."""
    with open(path, "r", encoding="utf-8") as f:
        raw = yaml.safe_load(f) or {}
    cfg = MultiZPipelineConfig()
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
