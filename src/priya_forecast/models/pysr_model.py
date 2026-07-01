"""PySR-equation P1D model.

Reads a YAML pointing at PySR `hall_of_fame_*.csv` Pareto-front files (one per
parameter), picks one equation per a `pick:` rule, parses it through a sympy
whitelist, applies the student's normalization round-trip, and combines the
per-parameter contributions per `combine: {multiplicative|additive|joint}`.

Math (multiplicative combine):

    P(theta, k) = P_fid(k) * prod_i [ P_F_i(theta_i, k) / P_F_i(theta_i_fid, k) ]

where each `P_F_i = f_i(theta_i_norm, k_norm, ...) * std_k + mean_k` and
`f_i` is the PySR equation. `additive` is the analogue with sums and
differences. `joint` uses a single sympy expression in (theta_1, ..., theta_11, k).

Trust boundary: PySR equations are parsed via a strict sympy whitelist of
allowed symbols and functions (no `__import__`, no Python eval). Any unknown
identifier fails YAML loading rather than reaching sympy unprotected.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import sympy as sp

from priya_forecast.config import EqnConfig, EqnParam
from priya_forecast.models.base import P1DModel
from priya_forecast.models.normalization import (
    NormalizationSpec,
    derive_from_gp,
    from_files,
    identity,
)
from priya_forecast.parameters import PARAM_NAMES, get_param

# ---------------------------------------------------------------------------
# Sympy whitelist
# ---------------------------------------------------------------------------

# Functions and symbol-builders allowed inside PySR equation strings.
# Anything outside this whitelist fails parsing immediately. Note: PySR's
# `square(x)` is mapped to `x**2` and `inv(x)` to `1/x` so the resulting
# sympy expression is a clean polynomial/transcendental form.
_SAFE_FUNCS: dict[str, object] = {
    "sin": sp.sin,
    "cos": sp.cos,
    "tan": sp.tan,
    "exp": sp.exp,
    "log": sp.log,
    "sqrt": sp.sqrt,
    "square": lambda x: x**2,
    "inv": lambda x: 1 / x,
    "abs": sp.Abs,
    "Abs": sp.Abs,
    "pow": lambda a, b: a**b,
}


def _parse_safely(expr_str: str, allowed_symbols: dict[str, sp.Symbol]) -> sp.Expr:
    """Parse a PySR-style expression with a strict whitelist.

    Only the symbols in `allowed_symbols` plus `_SAFE_FUNCS` are exposed.
    Any other identifier triggers a ValueError. We pass `evaluate=False` to
    avoid sympy auto-simplifications that could hide an injection.
    """
    local_dict = dict(_SAFE_FUNCS)
    local_dict.update(allowed_symbols)
    # Use sympy's default global_dict (the `from sympy import *` namespace) so
    # numeric literals, Symbol, and the standard transformations all resolve.
    # That namespace does NOT contain Python builtins like __import__/eval, so
    # arbitrary code execution via parse_expr is not exposed. We harden this
    # further with a post-parse free-symbol whitelist + AppliedUndef check.
    try:
        from sympy.parsing.sympy_parser import (
            parse_expr,
            standard_transformations,
        )

        # Use *only* standard_transformations: implicit_multiplication_application
        # would split PySR's `x0`, `x1`, ... tokens into `x*0`, `x*1`, etc.
        transformations = standard_transformations
        expr = parse_expr(
            expr_str,
            local_dict=local_dict,
            transformations=transformations,
            evaluate=True,
        )
    except Exception as e:
        raise ValueError(f"Failed to parse PySR equation {expr_str!r}: {e}") from e

    free = expr.free_symbols
    unknown = {s for s in free if s.name not in allowed_symbols}
    if unknown:
        names = sorted(s.name for s in unknown)
        raise ValueError(
            f"PySR equation {expr_str!r} references unknown symbols {names}. "
            f"Allowed: {sorted(allowed_symbols)} plus functions {sorted(_SAFE_FUNCS)}."
        )
    # Catch un-resolved function calls (e.g., `f(x)` where f isn't a known func).
    from sympy.core.function import AppliedUndef

    undef = expr.atoms(AppliedUndef)
    if undef:
        bad = sorted({u.func.__name__ for u in undef})
        raise ValueError(
            f"PySR equation {expr_str!r} calls unknown functions {bad}. "
            f"Allowed: {sorted(_SAFE_FUNCS)}."
        )
    return expr


# ---------------------------------------------------------------------------
# Pareto CSV picking
# ---------------------------------------------------------------------------


def load_pareto_csv(path: str | Path) -> pd.DataFrame:
    """Read a PySR `hall_of_fame_*.csv` and return a normalized DataFrame.

    Returns columns ['Complexity', 'Loss', 'Equation'] with capitalization
    coerced. Robust to PySR version drift across `Complexity/complexity` etc.
    """
    df = pd.read_csv(path, comment="#")
    rename = {}
    for col in df.columns:
        cl = col.strip().lower()
        if cl == "complexity":
            rename[col] = "Complexity"
        elif cl == "loss":
            rename[col] = "Loss"
        elif cl == "equation":
            rename[col] = "Equation"
        elif cl == "score":
            rename[col] = "Score"
    df = df.rename(columns=rename)
    missing = {"Complexity", "Loss", "Equation"} - set(df.columns)
    if missing:
        raise ValueError(
            f"PySR CSV {path!r} missing required columns {sorted(missing)}. "
            f"Found: {list(df.columns)}."
        )
    return df.reset_index(drop=True)


def pick_equation(df: pd.DataFrame, rule: str) -> tuple[str, int, float]:
    """Apply a `pick:` rule to a Pareto-front DataFrame.

    Returns
    -------
    expression : str
        The chosen `Equation` string.
    complexity : int
        The chosen Pareto point's complexity.
    loss : float
        The chosen Pareto point's loss.
    """
    if rule == "best_loss":
        i = int(df["Loss"].idxmin())
    elif rule.startswith("complexity_le:"):
        n = int(rule.split(":", 1)[1])
        eligible = df[df["Complexity"] <= n]
        if eligible.empty:
            raise ValueError(
                f"No equation in Pareto front with Complexity <= {n}. "
                f"Available: {df['Complexity'].tolist()}."
            )
        i = int(eligible["Loss"].idxmin())
    elif rule.startswith("accuracy_at:"):
        tol = float(rule.split(":", 1)[1])
        eligible = df[df["Loss"] <= tol]
        if eligible.empty:
            raise ValueError(
                f"No equation in Pareto front with Loss <= {tol}. "
                f"min loss = {df['Loss'].min()}."
            )
        i = int(eligible["Complexity"].idxmin())
    elif rule.startswith("row:"):
        i = int(rule.split(":", 1)[1])
        if not 0 <= i < len(df):
            raise ValueError(
                f"row index {i} out of range for Pareto front of length {len(df)}."
            )
    else:
        raise ValueError(f"Unknown pick rule {rule!r}.")
    row = df.iloc[i]
    return str(row["Equation"]), int(row["Complexity"]), float(row["Loss"])


# ---------------------------------------------------------------------------
# Equation compilation
# ---------------------------------------------------------------------------


@dataclass
class CompiledEquation:
    """A single per-parameter equation, ready to evaluate on (theta_i, k)."""

    param_name: str
    expression: sp.Expr
    fn: Callable[..., np.ndarray]  # vectorized (theta_norm, k_norm) -> flux_norm
    norm: NormalizationSpec
    fiducial: float
    raw_expression: str
    complexity: int | None = None
    loss: float | None = None
    extra_args: tuple[str, ...] = field(default_factory=tuple)
    """Names of any equation inputs beyond (param_name, k) that were declared
    in `fix:` and substituted before lambdification — kept here for diagnostics."""

    def evaluate(self, theta_i: float, k: np.ndarray) -> np.ndarray:
        """Return P_F_i(theta_i, k) on the requested k-grid (denormalized)."""
        k = np.asarray(k, dtype=float)
        theta_norm = self.norm.normalize_param(theta_i)
        k_norm = self.norm.normalize_k(k)
        flux_norm = self.fn(np.full_like(k, theta_norm, dtype=float), k_norm)
        # If the equation collapses to a scalar (e.g., `x0 - x0 == 0`), broadcast
        # back to k.shape so the inverse transform sees a per-k vector.
        flux_norm = np.broadcast_to(np.asarray(flux_norm, dtype=float), k.shape).copy()
        return self.norm.denormalize_flux(flux_norm, k)


def compile_equation(
    *,
    param_name: str,
    raw_expression: str,
    variables: list[str],
    fix: dict[str, float] | None,
    norm: NormalizationSpec,
    fiducial: float,
    complexity: int | None = None,
    loss: float | None = None,
) -> CompiledEquation:
    """Parse a PySR equation string, substitute fixed inputs, and lambdify.

    Parameters
    ----------
    param_name : str
        The forecast parameter this equation models. Must appear in `variables`.
    raw_expression : str
        Raw PySR equation string. May reference `x0, x1, ...` (PySR's default)
        or any name in `variables`. The first `len(variables)` `xN` symbols
        are aliased to the entries in `variables`.
    variables : list[str]
        Column order PySR was trained on. E.g. ["bhfeedback", "k", "resolution"].
    fix : dict[str, float] | None
        Values to substitute for any non-(param_name, k) entries in `variables`
        before lambdification. Required if `variables` contains anything beyond
        {param_name, k}.
    norm : NormalizationSpec
    fiducial : float
        Fiducial value of `param_name` (physical units).
    """
    if param_name not in variables:
        raise ValueError(
            f"`variables` ({variables}) must contain param_name {param_name!r}."
        )
    if "k" not in variables:
        raise ValueError(f"`variables` ({variables}) must contain 'k'.")

    extras = [v for v in variables if v not in {param_name, "k"}]
    fix = dict(fix or {})
    for ex in extras:
        if ex not in fix:
            raise ValueError(
                f"Variable {ex!r} in `variables` must be assigned a constant in `fix:` "
                f"(only {param_name!r} and 'k' may vary). Got fix={fix}."
            )

    # PySR writes equations using x0, x1, ... — declare those AND the named
    # variables as allowed (the student can also write the named form directly).
    alias_pairs = [(sp.Symbol(f"x{i}"), sp.Symbol(name)) for i, name in enumerate(variables)]
    alias_map = {old: new for old, new in alias_pairs}

    allowed_symbols: dict[str, sp.Symbol] = {name: sp.Symbol(name) for name in variables}
    for i in range(len(variables)):
        allowed_symbols[f"x{i}"] = sp.Symbol(f"x{i}")
    expr = _parse_safely(raw_expression, allowed_symbols)
    expr = expr.subs(alias_map)

    # Substitute fixed inputs. We tolerate `fix:` entries naming variables
    # not in this equation's variable list — that lets a global YAML fix:
    # {r: 0.8} apply uniformly across an equation set even when only some
    # equations actually use `r`.
    for name, value in fix.items():
        if name not in variables:
            continue
        expr = expr.subs(sp.Symbol(name), sp.Float(value))

    # Verify only (param_name, k) remain.
    remaining = {s.name for s in expr.free_symbols}
    expected = {param_name, "k"}
    if not remaining.issubset(expected):
        raise ValueError(
            f"After fix-substitution, equation still references {remaining - expected}. "
            f"Expected only {expected}."
        )

    fn = sp.lambdify((sp.Symbol(param_name), sp.Symbol("k")), expr, modules=["numpy"])
    return CompiledEquation(
        param_name=param_name,
        expression=expr,
        fn=fn,
        norm=norm,
        fiducial=fiducial,
        raw_expression=raw_expression,
        complexity=complexity,
        loss=loss,
        extra_args=tuple(extras),
    )


# ---------------------------------------------------------------------------
# YAML → CompiledEquation per-parameter
# ---------------------------------------------------------------------------


def _resolve_normalization(
    *,
    eqn_param: EqnParam,
    norm_block: dict | None,
    param_name: str,
    k_grid: np.ndarray,
    gp_model: P1DModel | None,
    z: float,
) -> NormalizationSpec:
    """Pick a NormalizationSpec for one parameter given the YAML's `normalization:` block.

    Modes:
    - None / mode='none' / mode='identity'  → identity()
    - mode='files'    → from_files(mean_flux, std_flux)
    - mode='auto'     → derive_from_gp(gp_model)  [requires `gp_model`]
    """
    if norm_block is None:
        return identity(k_grid)
    mode = (norm_block.get("mode") or "").lower()
    if mode in ("", "none", "identity"):
        return identity(k_grid)
    if mode == "files":
        return from_files(
            param_name=param_name,
            mean_flux_path=norm_block["mean_flux"],
            std_flux_path=norm_block["std_flux"],
            k_grid=k_grid,
            param_min=norm_block.get("param_min"),
            param_max=norm_block.get("param_max"),
            k_min=norm_block.get("k_min", 1e-3),
            k_max=norm_block.get("k_max", 2e-2),
        )
    if mode == "auto":
        if gp_model is None:
            raise ValueError(
                f"normalization.mode='auto' for {param_name!r} requires a GP model. "
                f"Pass `gp_model=` when building the PySRModel."
            )
        return derive_from_gp(
            gp_model=gp_model,
            param_name=param_name,
            z=z,
            k_grid=k_grid,
            n_samples=norm_block.get("n_samples", 64),
            seed=norm_block.get("seed", 0),
        )
    raise ValueError(f"Unknown normalization.mode {mode!r}.")


# ---------------------------------------------------------------------------
# PySRModel
# ---------------------------------------------------------------------------


class PySRModel(P1DModel):
    """Forecast P1D model built from per-parameter PySR equations.

    Construction flow:

    1. For each parameter in `eqn_cfg.parameters`, resolve the equation
       string (either `expression:` override or `pareto_csv` + `pick:`).
    2. Build a NormalizationSpec from the YAML's `normalization:` block.
    3. Compile the equation with sympy (whitelisted) + lambdify on numpy.
    4. Cache the resulting `CompiledEquation` keyed by parameter name.
    5. Load `fiducial_p1d` (an .npz with arrays `k` and `p1d`) for the
       multiplicative/additive combine.

    `predict(theta, k, z)` then walks each compiled equation and applies
    the YAML's `combine:` rule.
    """

    def __init__(
        self,
        *,
        eqn_cfg: EqnConfig,
        k_grid: np.ndarray,
        gp_model: P1DModel | None = None,
        normalization_block: dict | None = None,
    ) -> None:
        if eqn_cfg.model != "pysr":
            raise ValueError(f"PySRModel requires eqn_cfg.model == 'pysr', got {eqn_cfg.model!r}.")
        self.cfg = eqn_cfg
        self.z = eqn_cfg.redshift
        self.k_grid = np.asarray(k_grid, dtype=float)
        self.compiled: dict[str, CompiledEquation] = {}

        self.fiducial_p1d_k: np.ndarray | None = None
        self.fiducial_p1d: np.ndarray | None = None
        if eqn_cfg.combine in ("multiplicative", "additive"):
            d = np.load(eqn_cfg.fiducial_p1d)
            self.fiducial_p1d_k = np.asarray(d["k"], dtype=float)
            self.fiducial_p1d = np.asarray(d["p1d"], dtype=float)

        if eqn_cfg.combine == "joint":
            self._compile_joint(gp_model, normalization_block)
        else:
            for pname, ep in eqn_cfg.parameters.items():
                self._compile_one(pname, ep, gp_model, normalization_block)

    def _compile_one(
        self,
        param_name: str,
        ep: EqnParam,
        gp_model: P1DModel | None,
        normalization_block: dict | None,
    ) -> None:
        # 1. Resolve the equation string (expression override OR pareto_csv).
        if ep.expression is not None:
            raw_expr = ep.expression
            complexity, loss = None, None
        else:
            df = load_pareto_csv(ep.pareto_csv)  # type: ignore[arg-type]
            raw_expr, complexity, loss = pick_equation(df, ep.pick)

        # 2. Variables: default = [param_name, k].
        variables = ep.variables or [param_name, "k"]

        # 3. Normalization: per-param block in YAML overrides global block.
        per_param_norm = None
        # (We don't currently expose per-parameter `normalization:` in the YAML
        # to keep the schema small; equation sets typically use one convention.)

        norm_block = per_param_norm if per_param_norm is not None else normalization_block
        norm = _resolve_normalization(
            eqn_param=ep,
            norm_block=norm_block,
            param_name=param_name,
            k_grid=self.k_grid,
            gp_model=gp_model,
            z=self.z,
        )

        # 4. fix: dict for non-(param_name, k) variables (e.g., resolution).
        # We accept it as a top-level mapping inside the YAML's normalization
        # block (.fix) so the schema stays minimal, but for clarity also
        # accept ep.variables[i] not in {param_name, k} only if `norm_block`
        # carries `fix` for it.
        fix: dict[str, float] = {}
        if norm_block is not None and "fix" in norm_block and norm_block["fix"]:
            fix.update(norm_block["fix"])

        # 5. Compile.
        self.compiled[param_name] = compile_equation(
            param_name=param_name,
            raw_expression=raw_expr,
            variables=variables,
            fix=fix,
            norm=norm,
            fiducial=ep.fiducial,
            complexity=complexity,
            loss=loss,
        )

    def _compile_joint(
        self,
        gp_model: P1DModel | None,
        normalization_block: dict | None,
    ) -> None:
        """Compile a single joint equation in (theta_subset, k).

        The YAML's `joint_expression` is a sympy string in some subset of the
        11 forecast parameters plus `k` (and optionally extra fixed
        variables). The forecast pipeline then evaluates this single
        equation rather than a product/sum of per-param equations.

        Per-parameter `fiducial` is taken from `parameters[name].fiducial`
        when present, otherwise from `PARAMS_11D[name].fid`. The variables
        list defaults to `[<varying params...>, k]` and can be overridden
        via `parameters['__joint__'].variables` if extra inputs (resolution,
        etc.) need to be declared and fixed.
        """
        if not self.cfg.joint_expression:
            raise ValueError("combine='joint' requires `joint_expression` in YAML.")

        # Sniff variables: any of the 11 forecast names that appear free in
        # the parsed expression, plus `k`. Plus anything declared in
        # parameters['__joint__'].variables / .fix for extra fixed inputs.
        joint_extra = self.cfg.parameters.get("__joint__", None)
        declared_vars = (joint_extra.variables if joint_extra is not None else None) or []
        declared_fix = {}
        if normalization_block is not None and "fix" in (normalization_block or {}):
            declared_fix.update(normalization_block["fix"] or {})
        norm_block = normalization_block or {"mode": "identity"}
        # Build a NormalizationSpec that uses physical-unit pass-through:
        # the joint equation should produce P_F directly (after optional
        # mean/std denorm). For the joint case we don't have a single
        # parameter to derive normalization from, so default to identity.
        from priya_forecast.models.normalization import identity as _identity
        norm = _identity(self.k_grid)

        # First parse to discover which forecast params are referenced.
        candidate_syms = {n: sp.Symbol(n) for n in PARAM_NAMES}
        candidate_syms["k"] = sp.Symbol("k")
        for v in declared_vars:
            candidate_syms[v] = sp.Symbol(v)
        # Also allow x0..xN aliases.
        for i in range(20):
            candidate_syms[f"x{i}"] = sp.Symbol(f"x{i}")
        expr = _parse_safely(self.cfg.joint_expression, candidate_syms)

        # If declared_vars is given, alias x0..xN to the declared variables.
        if declared_vars:
            alias_pairs = [(sp.Symbol(f"x{i}"), sp.Symbol(name))
                           for i, name in enumerate(declared_vars)]
            for old, new in alias_pairs:
                expr = expr.subs(old, new)

        # Substitute fixed inputs (e.g., resolution).
        for name, value in declared_fix.items():
            expr = expr.subs(sp.Symbol(name), sp.Float(value))

        # The remaining free symbols must be a subset of forecast params + 'k'.
        remaining = {s.name for s in expr.free_symbols}
        allowed = set(PARAM_NAMES) | {"k"}
        if not remaining.issubset(allowed):
            raise ValueError(
                f"Joint expression references {remaining - allowed} after fix-substitution; "
                f"only forecast params {PARAM_NAMES} and 'k' may remain free."
            )
        # Lock the parameter symbols + k in a stable order matching PARAM_NAMES.
        param_syms = [sp.Symbol(n) for n in PARAM_NAMES]
        self._joint_fn = sp.lambdify(param_syms + [sp.Symbol("k")], expr, modules=["numpy"])
        self._joint_expr = expr
        self._joint_norm = norm

    # ------------------------------------------------------------------
    # Forward model
    # ------------------------------------------------------------------

    def _theta_dict(self, theta: np.ndarray) -> dict[str, float]:
        if theta.shape != (11,):
            raise ValueError(f"theta must be shape (11,), got {theta.shape}.")
        return dict(zip(PARAM_NAMES, theta.tolist()))

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        if abs(z - self.z) > 1e-3:
            raise ValueError(
                f"PySRModel was built for z={self.z}, called with z={z}."
            )
        k = np.asarray(k, dtype=float)
        td = self._theta_dict(np.asarray(theta, dtype=float))

        if self.cfg.combine == "joint":
            theta_full = np.asarray(theta, dtype=float)
            # Lambdified with symbols in PARAM_NAMES order + 'k'. Broadcast
            # the parameter values to k.shape so the call returns a per-k array.
            args = list(theta_full.tolist()) + [k]
            flux = np.asarray(self._joint_fn(*args), dtype=float)
            # If the equation is k-independent (e.g., constant), broadcast.
            flux = np.broadcast_to(flux, k.shape).copy()
            return self._joint_norm.denormalize_flux(flux, k)

        # Interpolate the cached fiducial P1D onto the requested k-grid.
        assert self.fiducial_p1d_k is not None and self.fiducial_p1d is not None
        p_fid = np.interp(k, self.fiducial_p1d_k, self.fiducial_p1d)

        if self.cfg.combine == "multiplicative":
            out = p_fid.copy()
            for pname, ce in self.compiled.items():
                num = ce.evaluate(td[pname], k)
                den = ce.evaluate(ce.fiducial, k)
                if np.any(den == 0):
                    raise FloatingPointError(
                        f"Per-parameter fiducial evaluation yielded zero for {pname!r}; "
                        f"multiplicative combine is undefined."
                    )
                out = out * (num / den)
            return out

        if self.cfg.combine == "additive":
            out = p_fid.copy()
            for pname, ce in self.compiled.items():
                num = ce.evaluate(td[pname], k)
                ref = ce.evaluate(ce.fiducial, k)
                out = out + (num - ref)
            return out

        raise NotImplementedError(f"combine={self.cfg.combine!r} not implemented.")
