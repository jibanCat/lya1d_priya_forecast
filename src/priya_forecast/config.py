"""YAML config loader + dataclass-based validation.

Three top-level configs are loaded by the CLI:

- `RunConfig`        — `configs/default.yaml`: run-wide knobs (z, mode, k range,
                       cov_scale, GP basedir, MCMC/Fisher parameters).
- `EqnConfig`        — `configs/eqns/<name>.yaml`: which model to use and, for
                       PySR, which Pareto CSV / pick rule per parameter.
- `DiagnosticConfig` — `configs/diagnostic.yaml`: multi-D PySR diagnostic.
- `HPOConfig`        — `configs/hpo/{quick,full}.yaml`: PySR HPO budget.

We deliberately use plain dataclasses + manual validation. The YAML schema
is small and fixed; pulling in pydantic/attrs would just be ceremony.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from priya_forecast.parameters import PARAM_NAMES


# ---------------------------------------------------------------------------
# Run config
# ---------------------------------------------------------------------------


@dataclass
class KRange:
    min: float
    max: float

    def validate(self) -> None:
        if self.min <= 0:
            raise ValueError(f"k_range.min must be > 0, got {self.min}.")
        if self.max <= self.min:
            raise ValueError(f"k_range.max ({self.max}) must exceed min ({self.min}).")


@dataclass
class MCMCConfig:
    n_steps: int = 5000
    walkers_per_dim: int = 4
    burn_in_frac: float = 0.2
    backend_path: str = "results/mcmc/chain.h5"

    def validate(self) -> None:
        if self.n_steps < 1:
            raise ValueError(f"mcmc.n_steps must be >= 1, got {self.n_steps}.")
        if self.walkers_per_dim < 2:
            raise ValueError(f"mcmc.walkers_per_dim must be >= 2, got {self.walkers_per_dim}.")
        if not 0.0 <= self.burn_in_frac < 1.0:
            raise ValueError(f"mcmc.burn_in_frac must be in [0,1), got {self.burn_in_frac}.")


@dataclass
class FisherConfig:
    step_frac: float = 0.01
    rel_tol: float = 0.01

    def validate(self) -> None:
        if not 0 < self.step_frac < 1:
            raise ValueError(f"fisher.step_frac must be in (0,1), got {self.step_frac}.")
        if not 0 < self.rel_tol < 1:
            raise ValueError(f"fisher.rel_tol must be in (0,1), got {self.rel_tol}.")


@dataclass
class RunConfig:
    redshift: float = 3.6
    mode: str = "fisher"  # "fisher" | "mcmc"
    include_T0_prior: bool = False
    marginalize_dla: bool = False
    cov_scale: float = 1.0
    mock_data: str = "gp"  # "gp" | "eboss"
    k_range: KRange = field(default_factory=lambda: KRange(min=0.001, max=0.02))
    gp_emulator_basedir: str = (
        "/home/mfho/student_projects/InferenceLyaData/Emulator_Files"
    )
    mcmc: MCMCConfig = field(default_factory=MCMCConfig)
    fisher: FisherConfig = field(default_factory=FisherConfig)

    def validate(self) -> None:
        if self.mode not in {"fisher", "mcmc"}:
            raise ValueError(f"mode must be 'fisher' or 'mcmc', got {self.mode!r}.")
        if self.mock_data not in {"gp", "eboss"}:
            raise ValueError(f"mock_data must be 'gp' or 'eboss', got {self.mock_data!r}.")
        if self.cov_scale <= 0:
            raise ValueError(f"cov_scale must be > 0, got {self.cov_scale}.")
        if not 2.2 <= self.redshift <= 4.6:
            raise ValueError(
                f"redshift {self.redshift} outside eBOSS DR14 range [2.2, 4.6]."
            )
        self.k_range.validate()
        self.mcmc.validate()
        self.fisher.validate()


# ---------------------------------------------------------------------------
# Equation-set config (the student-facing PySR YAML)
# ---------------------------------------------------------------------------


VALID_PICK_RULES = ("best_loss",)
VALID_PICK_PREFIXES = ("complexity_le:", "accuracy_at:", "row:")


@dataclass
class EqnParam:
    """One entry inside `EqnConfig.parameters[name]`.

    Either `pareto_csv` (PySR Pareto file + pick rule) or `expression` (direct
    sympy override) must be set. `expression` takes precedence if both are.
    """

    fiducial: float
    pareto_csv: str | None = None
    pick: str = "best_loss"
    variables: list[str] | None = None
    expression: str | None = None

    def validate(self, param_name: str) -> None:
        if self.expression is None and self.pareto_csv is None:
            raise ValueError(
                f"parameters.{param_name}: must set either `pareto_csv` or `expression`."
            )
        if self.pareto_csv is not None and not _is_valid_pick(self.pick):
            raise ValueError(
                f"parameters.{param_name}: invalid `pick` rule {self.pick!r}. "
                f"Valid: {VALID_PICK_RULES} or one of "
                f"{VALID_PICK_PREFIXES} with a value."
            )


def _is_valid_pick(pick: str) -> bool:
    if pick in VALID_PICK_RULES:
        return True
    return any(pick.startswith(p) and len(pick) > len(p) for p in VALID_PICK_PREFIXES)


@dataclass
class EqnConfig:
    name: str
    redshift: float
    model: str = "pysr"  # "pysr" | "gp"
    description: str = ""
    combine: str = "multiplicative"  # "multiplicative" | "additive" | "joint"
    fiducial_p1d: str | None = None
    parameters: dict[str, EqnParam] = field(default_factory=dict)
    joint_expression: str | None = None

    def validate(self) -> None:
        if self.model not in {"pysr", "gp"}:
            raise ValueError(f"model must be 'pysr' or 'gp', got {self.model!r}.")

        if self.model == "gp":
            # GP baseline — `parameters`/`combine` etc. are ignored.
            return

        if self.combine not in {"multiplicative", "additive", "joint"}:
            raise ValueError(
                f"combine must be 'multiplicative' | 'additive' | 'joint', "
                f"got {self.combine!r}."
            )

        if self.combine == "joint":
            if not self.joint_expression:
                raise ValueError("combine='joint' requires `joint_expression` to be set.")
        else:
            if not self.parameters:
                raise ValueError(
                    f"combine={self.combine!r} requires `parameters` to be populated."
                )
            unknown = set(self.parameters) - set(PARAM_NAMES)
            if unknown:
                raise ValueError(
                    f"Unknown parameter names in eqn config: {sorted(unknown)}. "
                    f"Valid: {PARAM_NAMES}."
                )
            for pname, ep in self.parameters.items():
                ep.validate(pname)
            if self.fiducial_p1d is None:
                raise ValueError(
                    f"combine={self.combine!r} requires `fiducial_p1d` (path to .npz)."
                )


# ---------------------------------------------------------------------------
# Diagnostic + HPO configs (lightweight — used by phases 5 & 6)
# ---------------------------------------------------------------------------


@dataclass
class DiagnosticConfig:
    redshifts: list[float]
    benchmark_z: float
    regimes: list[str]
    n_train: int
    n_test: int
    seed: int
    param_names: list[str]
    pysr_kwargs: dict[str, Any]
    output_dir: str

    def validate(self) -> None:
        if self.benchmark_z not in self.redshifts:
            raise ValueError(
                f"benchmark_z={self.benchmark_z} must appear in redshifts={self.redshifts}."
            )
        for r in self.regimes:
            if r not in {"1D", "2D_pairs", "full_kD"}:
                raise ValueError(f"Unknown regime {r!r} in diagnostic config.")
        unknown = set(self.param_names) - set(PARAM_NAMES)
        if unknown:
            raise ValueError(f"Unknown param_names in diagnostic: {sorted(unknown)}.")
        if self.n_train < 8 or self.n_test < 8:
            raise ValueError("n_train and n_test must be at least 8.")


@dataclass
class HPOConfig:
    strategy: str
    n_trials: int
    metric: str
    target_loss: float
    seed: int
    n_jobs: int
    cache_dir: str | None
    space: dict[str, Any]

    def validate(self) -> None:
        if self.strategy not in {"grid", "random", "bayesian"}:
            raise ValueError(f"strategy must be grid|random|bayesian, got {self.strategy!r}.")
        if self.metric not in {"val_mse", "complexity_at_target", "pareto_area"}:
            raise ValueError(f"unknown metric {self.metric!r}.")
        if self.n_trials < 1:
            raise ValueError(f"n_trials must be >= 1, got {self.n_trials}.")


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, "r") as f:
        data = yaml.safe_load(f)
    if not isinstance(data, dict):
        raise ValueError(f"YAML at {path!r} must be a mapping at top level, got {type(data)}.")
    return data


def load_run_config(path: str | Path) -> RunConfig:
    raw = _read_yaml(path)
    sub: dict[str, Any] = {}
    if "k_range" in raw:
        sub["k_range"] = KRange(**raw.pop("k_range"))
    if "mcmc" in raw:
        sub["mcmc"] = MCMCConfig(**raw.pop("mcmc"))
    if "fisher" in raw:
        sub["fisher"] = FisherConfig(**raw.pop("fisher"))
    cfg = RunConfig(**sub, **raw)
    cfg.validate()
    return cfg


def load_eqn_config(path: str | Path) -> EqnConfig:
    raw = _read_yaml(path)
    params_raw = raw.pop("parameters", {}) or {}
    params: dict[str, EqnParam] = {}
    for pname, pdata in params_raw.items():
        if not isinstance(pdata, dict):
            raise ValueError(
                f"parameters.{pname} must be a mapping, got {type(pdata).__name__}."
            )
        params[pname] = EqnParam(**pdata)
    cfg = EqnConfig(parameters=params, **raw)
    cfg.validate()
    return cfg


def load_diagnostic_config(path: str | Path) -> DiagnosticConfig:
    raw = _read_yaml(path)
    cfg = DiagnosticConfig(**raw)
    cfg.validate()
    return cfg


def load_hpo_config(path: str | Path) -> HPOConfig:
    raw = _read_yaml(path)
    cfg = HPOConfig(**raw)
    cfg.validate()
    return cfg
