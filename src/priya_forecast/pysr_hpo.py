"""Reusable PySR hyperparameter optimizer.

Independent of the rest of the forecast on purpose: takes raw `(X_train,
y_train, X_val, y_val)` arrays so the student can drop this into any
future symbolic-regression project.

Three strategies:
  - "grid"      : exhaustive (capped at 200 configs).
  - "random"    : uniform random sample of `n_trials` configs.
  - "bayesian"  : optuna if installed, else falls back to "random" with a
                  warning. Optuna is NOT a hard dependency.

Three evaluation metrics on the held-out validation set:
  - "val_mse"              : raw MSE.
  - "complexity_at_target" : minimum Pareto complexity such that
                             val loss <= target_loss. Best for the paper's
                             "smallest interpretable equation" objective.
  - "pareto_area"          : area under the (complexity, log-loss) curve.
                             Captures full-tradeoff quality.

Caching: when `cache_dir` is set, each trained model is keyed by a hash
of (config, X_train, y_train) and reused on rerun. PySR fits are
expensive — caching is the difference between exploratory and useless.
"""

from __future__ import annotations

import hashlib
import itertools
import json
import pickle
import time
import warnings
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------


@dataclass
class HPOSearchSpace:
    """Categorical/grid search space for PySR hyperparameters.

    Each entry is a list of candidate values. Defaults give a small but
    diverse search; override fields for tighter or wider sweeps.
    """

    niterations:        list[int]       = field(default_factory=lambda: [40, 100, 200])
    populations:        list[int]       = field(default_factory=lambda: [15, 30])
    population_size:    list[int]       = field(default_factory=lambda: [33, 50])
    maxsize:            list[int]       = field(default_factory=lambda: [15, 20, 30])
    parsimony:          list[float]     = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    binary_operators:   list[list[str]] = field(default_factory=lambda: [
        ["+", "-", "*", "/"],
        ["+", "-", "*", "/", "pow"],
    ])
    unary_operators:    list[list[str]] = field(default_factory=lambda: [
        [],
        ["log", "exp"],
        ["log", "exp", "sqrt"],
    ])

    def all_configs(self) -> list[dict[str, Any]]:
        """Cartesian product of every field. Used by `strategy='grid'`."""
        keys = [
            "niterations", "populations", "population_size", "maxsize",
            "parsimony", "binary_operators", "unary_operators",
        ]
        choices = [getattr(self, k) for k in keys]
        return [
            dict(zip(keys, combo)) for combo in itertools.product(*choices)
        ]

    def sample_random(self, *, n: int, rng: np.random.Generator) -> list[dict[str, Any]]:
        """Uniform random sample of `n` configs."""
        keys = [
            "niterations", "populations", "population_size", "maxsize",
            "parsimony", "binary_operators", "unary_operators",
        ]
        out = []
        for _ in range(n):
            cfg = {}
            for k in keys:
                opts = getattr(self, k)
                cfg[k] = opts[rng.integers(0, len(opts))]
            out.append(cfg)
        return out


@dataclass
class HPOResult:
    config: dict[str, Any]
    train_loss: float
    val_loss: float
    test_loss: float | None
    pareto_complexity: int
    pareto_loss: float
    best_expression: str
    wall_time_s: float
    pareto_complexities: list[int] = field(default_factory=list)
    pareto_losses: list[float] = field(default_factory=list)
    extra_metrics: dict[str, float] = field(default_factory=dict)

    def metric(self, name: str, *, target_loss: float | None = None,
               fisher_residual: float | None = None) -> float:
        """Score this trial under the named metric (lower = better).

        Recognised names:
          - "val_mse"             : raw validation MSE.
          - "complexity_at_target" : min Pareto complexity with loss ≤ target.
          - "pareto_area"          : ∫ log10(loss) d(complexity) along Pareto.
          - "fisher_agreement"     : mean-square error of the equation's
                                     gradient at fid vs the GP's. Requires
                                     callers to populate
                                     ``self.extra_metrics["fisher_residual"]``
                                     (or pass it explicitly here).
        """
        if name == "val_mse":
            return float(self.val_loss)
        if name == "complexity_at_target":
            if target_loss is None:
                raise ValueError("complexity_at_target requires `target_loss`.")
            ok = [c for c, l in zip(self.pareto_complexities, self.pareto_losses)
                  if l <= target_loss]
            return float(min(ok)) if ok else float("inf")
        if name == "pareto_area":
            if not self.pareto_complexities:
                return float("inf")
            cs = np.asarray(self.pareto_complexities, dtype=float)
            ls = np.log10(np.maximum(np.asarray(self.pareto_losses), 1e-30))
            order = np.argsort(cs)
            return float(np.trapz(ls[order], cs[order]))
        if name == "fisher_agreement":
            v = fisher_residual
            if v is None:
                v = (self.extra_metrics or {}).get("fisher_residual")
            if v is None:
                raise ValueError(
                    "fisher_agreement requires fisher_residual either as "
                    "kwarg or stored in self.extra_metrics."
                )
            return float(v)
        if name == "sigma_targeted":
            ratio = (self.extra_metrics or {}).get("sigma_ratio")
            if ratio is None:
                raise ValueError(
                    "sigma_targeted requires `sigma_ratio` in extra_metrics. "
                    "Use make_sigma_targeted_trainer() to populate it."
                )
            if not np.isfinite(ratio):
                return float("inf")
            return float((ratio - 1.0) ** 2)
        raise ValueError(f"Unknown metric {name!r}.")


# ---------------------------------------------------------------------------
# Hashing + caching
# ---------------------------------------------------------------------------


def _hash_config(config: dict, X_train: np.ndarray, y_train: np.ndarray) -> str:
    """Stable hash of (config + training data). Used as the cache key."""
    payload = json.dumps(config, sort_keys=True, default=str).encode()
    h = hashlib.sha256()
    h.update(payload)
    h.update(np.ascontiguousarray(X_train, dtype=np.float64).tobytes())
    h.update(np.ascontiguousarray(y_train, dtype=np.float64).tobytes())
    return h.hexdigest()[:16]


# ---------------------------------------------------------------------------
# Default trainer (real PySR) + injectable backend for tests
# ---------------------------------------------------------------------------


def _default_pysr_trainer(
    *, X_train, y_train, X_val, y_val, config, seed,
):
    """Train one PySR config.

    Returns a 6-tuple:
      (train_loss, val_loss, pareto_complexities, pareto_losses,
       best_expression, extra_metrics)

    `extra_metrics` is an optional dict of additional metrics computed
    post-fit (empty by default; populated by `make_fisher_aware_trainer`).
    Older trainers returning only the first 5 elements are accepted by
    `run_hpo`.
    """
    from pysr import PySRRegressor  # type: ignore[import-not-found]

    args = dict(
        niterations=config["niterations"],
        populations=config["populations"],
        population_size=config["population_size"],
        maxsize=config["maxsize"],
        parsimony=config["parsimony"],
        binary_operators=config["binary_operators"],
        unary_operators=config["unary_operators"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=seed, deterministic=True, parallelism="serial", verbosity=0,
    )
    model = PySRRegressor(**args)
    model.fit(X_train, y_train)
    pareto = model.equations_
    train_pred = np.asarray(model.predict(X_train)).ravel()
    val_pred = np.asarray(model.predict(X_val)).ravel()
    train_loss = float(np.mean((train_pred - y_train) ** 2))
    val_loss = float(np.mean((val_pred - y_val) ** 2))
    return (
        train_loss, val_loss,
        pareto["complexity"].astype(int).tolist(),
        pareto["loss"].astype(float).tolist(),
        str(pareto.iloc[pareto["loss"].idxmin()]["equation"]),
        {},
    )


def make_sigma_targeted_trainer(
    *,
    base_trainer: Callable | None = None,
    sigma_evaluator: Callable[[str], float],
):
    """Wrap a base PySR trainer so each fit ALSO computes σ_pysr / σ_GP via
    a user-supplied evaluator that runs the actual forecast Fisher.

    `sigma_evaluator(equation_str)` should return the σ ratio (1.0 = perfect).
    Stored in `extra_metrics["sigma_ratio"]` so the metric
    `"sigma_targeted"` can sort by `(sigma_ratio - 1)²`.

    This is more expensive than `make_fisher_aware_trainer` (one extra
    Fisher solve per config) but it directly optimizes what the forecast
    cares about. For 1D forecasts this is ~ms; for higher-D, seconds.
    """
    if base_trainer is None:
        base_trainer = _default_pysr_trainer

    def _trainer(*, X_train, y_train, X_val, y_val, config, seed):
        out = base_trainer(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, config=config, seed=seed,
        )
        if len(out) == 5:
            train_loss, val_loss, complexities, losses, best_expr = out
            extra: dict[str, float] = {}
        else:
            train_loss, val_loss, complexities, losses, best_expr, extra = out
        try:
            ratio = float(sigma_evaluator(best_expr))
            extra["sigma_ratio"] = ratio
        except Exception as e:  # noqa: BLE001
            extra["sigma_ratio"] = float("inf")
            extra["sigma_eval_error"] = str(e)
        return train_loss, val_loss, complexities, losses, best_expr, extra

    return _trainer


def make_fisher_aware_trainer(
    *,
    base_trainer: Callable | None = None,
    gradient_target: np.ndarray,
    fid_X: np.ndarray,
    h: float = 1e-3,
):
    """Wrap a base PySR trainer so each fit also computes the gradient
    residual at fid vs a target (e.g. the GP's gradient).

    Parameters
    ----------
    base_trainer : callable, optional
        Underlying trainer; defaults to `_default_pysr_trainer`.
    gradient_target : ndarray, shape (n_eval_points,)
        The reference gradient ∂P/∂θ to match, evaluated at each row of
        `fid_X`. For 1D forecast, this is the GP's df/dθ along the
        eBOSS k-grid (one number per k-bin) at fid.
    fid_X : ndarray, shape (n_eval_points, n_features)
        Inputs at which to evaluate the equation's gradient. Convention:
        column 0 is the parameter (held at fid_norm), columns 1+ are k_norm
        and any other inputs. Gradient is taken w.r.t. column 0 only.
    h : float
        Finite-difference step in normalized-input units.

    Returns
    -------
    trainer callable that returns the same 6-tuple as `_default_pysr_trainer`
    but with `extra_metrics={"fisher_residual": ...}`.
    """
    if base_trainer is None:
        base_trainer = _default_pysr_trainer

    def _trainer(*, X_train, y_train, X_val, y_val, config, seed):
        # Train normally.
        out = base_trainer(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, config=config, seed=seed,
        )
        # Older trainers return 5-tuples; pad to 6.
        if len(out) == 5:
            train_loss, val_loss, complexities, losses, best_expr = out
            extra: dict[str, float] = {}
        else:
            train_loss, val_loss, complexities, losses, best_expr, extra = out

        # Need the trained PySRRegressor to evaluate gradients at fid.
        # Re-train just at the best Pareto point — cheap because we cache
        # the same config + data hash; default trainer holds the model in
        # memory but doesn't return it. Cleanest is to re-import and refit
        # at the chosen complexity. For simplicity here, reconstruct
        # gradients by parsing the equation through sympy.
        try:
            import sympy as sp
            expr = sp.sympify(best_expr)
            # Map x0, x1, ... to their column indices. Any other free symbols
            # (e.g. `r` in published equations) are kept as additional inputs
            # that the caller can fix via `fid_X` extra columns or — if
            # absent from fid_X — pinned to 0 here.
            xcols = sorted(
                [s for s in expr.free_symbols if s.name.startswith("x")],
                key=lambda s: int(s.name[1:]),
            )
            other = sorted(
                [s for s in expr.free_symbols if not s.name.startswith("x")],
                key=lambda s: s.name,
            )
            eval_points = np.asarray(fid_X)
            if eval_points.ndim == 1:
                eval_points = eval_points[:, None]
            n_in = eval_points.shape[1]
            if not xcols:
                extra["fisher_residual"] = float("inf")
            else:
                # Build a vectorized callable taking the n_in columns of
                # fid_X. Any extra `other` symbols default to 0.
                all_syms = list(xcols) + other
                fn = sp.lambdify(all_syms, expr, modules=["numpy"])

                def _eval(X):
                    args = []
                    for s in xcols:
                        col = int(s.name[1:])
                        if col < n_in:
                            args.append(X[:, col])
                        else:
                            args.append(np.zeros(X.shape[0]))
                    for _ in other:
                        args.append(np.zeros(X.shape[0]))
                    return np.asarray(fn(*args)).ravel()

                Xp = eval_points.copy(); Xp[:, 0] += h
                Xm = eval_points.copy(); Xm[:, 0] -= h
                df = (_eval(Xp) - _eval(Xm)) / (2 * h)
                target = np.asarray(gradient_target).ravel()
                if df.shape == target.shape:
                    extra["fisher_residual"] = float(np.mean((df - target) ** 2))
                else:
                    extra["fisher_residual"] = float("inf")
                    extra["fisher_shape_mismatch"] = f"df={df.shape} target={target.shape}"
        except Exception as e:  # noqa: BLE001
            extra["fisher_residual"] = float("inf")
            extra["fisher_residual_error"] = str(e)

        return (train_loss, val_loss, complexities, losses, best_expr, extra)

    return _trainer


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def run_hpo(
    *,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    space: HPOSearchSpace,
    strategy: str = "random",
    n_trials: int = 20,
    metric: str = "val_mse",
    target_loss: float | None = None,
    seed: int = 0,
    cache_dir: str | Path | None = None,
    trainer: Callable | None = None,
    grid_cap: int = 200,
) -> list[HPOResult]:
    """Run an HPO sweep over `space` and return results sorted by `metric`.

    Parameters
    ----------
    X_train, y_train : training data (whatever shape PySR expects).
    X_val,   y_val   : held-out validation set.
    space            : HPOSearchSpace.
    strategy         : "grid" | "random" | "bayesian".
    n_trials         : number of configs to try (ignored for "grid").
    metric           : key for sorting the returned list.
    target_loss      : required when metric == "complexity_at_target".
    seed             : RNG seed for "random" sampling and PySR's RNG.
    cache_dir        : if set, caches each trained model under
                       `<cache_dir>/<hash>.pkl`. Cache hit = skip training.
    trainer          : injectable callable for unit tests; defaults to
                       _default_pysr_trainer (real PySR).
    grid_cap         : safety cap on grid configurations to prevent
                       accidentally launching a 5000-config sweep.
    """
    rng = np.random.default_rng(seed)
    trainer = trainer if trainer is not None else _default_pysr_trainer

    if strategy == "grid":
        configs = space.all_configs()
        if len(configs) > grid_cap:
            raise ValueError(
                f"grid would launch {len(configs)} configs; pass `grid_cap=` to confirm."
            )
    elif strategy == "random":
        configs = space.sample_random(n=n_trials, rng=rng)
    elif strategy == "bayesian":
        try:
            import optuna  # type: ignore[import-not-found]
            configs = _bayesian_configs(space, n_trials=n_trials, seed=seed,
                                        trainer=trainer, X_train=X_train,
                                        y_train=y_train, X_val=X_val, y_val=y_val,
                                        cache_dir=cache_dir)
            # The bayesian path runs trainer inside its trial loop; we
            # short-circuit the rest here.
            return _sort_results(configs, metric=metric, target_loss=target_loss)
        except ImportError:
            warnings.warn(
                "optuna not installed; falling back to random search.",
                UserWarning, stacklevel=2,
            )
            configs = space.sample_random(n=n_trials, rng=rng)
    else:
        raise ValueError(f"Unknown strategy {strategy!r}.")

    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)

    results: list[HPOResult] = []
    for cfg in configs:
        cache_path = None
        if cache_dir is not None:
            key = _hash_config(cfg, X_train, y_train)
            cache_path = cache_dir / f"{key}.pkl"
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                results.append(cached)
                continue

        t0 = time.time()
        out = trainer(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, config=cfg, seed=seed,
        )
        wt = time.time() - t0
        if len(out) == 5:
            train_loss, val_loss, complexities, losses, best_expr = out
            extra_metrics: dict[str, float] = {}
        else:
            train_loss, val_loss, complexities, losses, best_expr, extra_metrics = out
        best_idx = int(np.argmin(losses)) if losses else 0
        result = HPOResult(
            config=cfg,
            train_loss=train_loss,
            val_loss=val_loss,
            test_loss=None,
            pareto_complexity=int(complexities[best_idx]) if complexities else 0,
            pareto_loss=float(losses[best_idx]) if losses else float("inf"),
            best_expression=best_expr,
            wall_time_s=wt,
            pareto_complexities=list(complexities),
            pareto_losses=list(losses),
            extra_metrics=dict(extra_metrics or {}),
        )
        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(result, f)
        results.append(result)

    return _sort_results(results, metric=metric, target_loss=target_loss)


def _sort_results(results, *, metric, target_loss=None):
    return sorted(results, key=lambda r: r.metric(metric, target_loss=target_loss))


def _bayesian_configs(
    space, *, n_trials, seed, trainer, X_train, y_train, X_val, y_val, cache_dir,
):
    """Optuna-driven Bayesian search. Returns HPOResult list directly."""
    import optuna

    cache_dir = Path(cache_dir) if cache_dir is not None else None
    if cache_dir is not None:
        cache_dir.mkdir(parents=True, exist_ok=True)
    results: list[HPOResult] = []

    def _objective(trial):
        cfg = {
            "niterations":     trial.suggest_categorical("niterations",     space.niterations),
            "populations":     trial.suggest_categorical("populations",     space.populations),
            "population_size": trial.suggest_categorical("population_size", space.population_size),
            "maxsize":         trial.suggest_categorical("maxsize",         space.maxsize),
            "parsimony":       trial.suggest_categorical("parsimony",       space.parsimony),
            # Categorical-of-list isn't supported; index instead.
            "binary_operators": space.binary_operators[
                trial.suggest_int("binary_idx", 0, len(space.binary_operators) - 1)],
            "unary_operators":  space.unary_operators[
                trial.suggest_int("unary_idx", 0, len(space.unary_operators) - 1)],
        }
        cache_path = None
        if cache_dir is not None:
            key = _hash_config(cfg, X_train, y_train)
            cache_path = cache_dir / f"{key}.pkl"
            if cache_path.exists():
                with open(cache_path, "rb") as f:
                    cached = pickle.load(f)
                results.append(cached)
                return cached.val_loss

        t0 = time.time()
        train_loss, val_loss, complexities, losses, best_expr = trainer(
            X_train=X_train, y_train=y_train,
            X_val=X_val, y_val=y_val, config=cfg, seed=seed,
        )
        wt = time.time() - t0
        best_idx = int(np.argmin(losses)) if losses else 0
        r = HPOResult(
            config=cfg, train_loss=train_loss, val_loss=val_loss, test_loss=None,
            pareto_complexity=int(complexities[best_idx]) if complexities else 0,
            pareto_loss=float(losses[best_idx]) if losses else float("inf"),
            best_expression=best_expr, wall_time_s=wt,
            pareto_complexities=list(complexities), pareto_losses=list(losses),
        )
        if cache_path is not None:
            with open(cache_path, "wb") as f:
                pickle.dump(r, f)
        results.append(r)
        return val_loss

    study = optuna.create_study(direction="minimize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(_objective, n_trials=n_trials, show_progress_bar=False)
    return results


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------


def plot_hpo_results(results: list[HPOResult], *, outdir: str | Path,
                     metric: str = "val_mse", target_loss: float | None = None,
                     top_k: int = 10) -> Path:
    """Four diagnostic figures:
    1. Bar chart: top-k configs by metric.
    2. Hyperparam scatter: each numeric hyperparam vs metric.
    3. Pareto-front overlay of top-5 configs.
    4. Wall-time vs metric (efficiency frontier).
    """
    import matplotlib.pyplot as plt

    outdir = Path(outdir); outdir.mkdir(parents=True, exist_ok=True)
    sorted_r = sorted(results, key=lambda r: r.metric(metric, target_loss=target_loss))

    # --- 1. Top-k bar chart ---
    top = sorted_r[:top_k]
    labels = [
        f"n={r.config['niterations']}, sz={r.config['maxsize']}, "
        f"pa={r.config['parsimony']:.0e}"
        for r in top
    ]
    vals = [r.metric(metric, target_loss=target_loss) for r in top]
    fig, ax = plt.subplots(figsize=(8, max(3.5, 0.3 * len(top))), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.barh(range(len(top)), vals, color="C0")
    ax.set_yticks(range(len(top)))
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel(metric)
    ax.set_title(f"HPO top-{len(top)} by {metric}")
    if metric == "val_mse":
        ax.set_xscale("log")
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout(); fig.savefig(outdir / "hpo1_top.png"); plt.close(fig)

    # --- 2. Hyperparam scatter ---
    numeric_keys = ["niterations", "populations", "population_size", "maxsize", "parsimony"]
    fig, axes = plt.subplots(1, len(numeric_keys), figsize=(3 * len(numeric_keys), 3.5), dpi=120)
    fig.patch.set_facecolor("white")
    for ax, key in zip(axes, numeric_keys):
        ax.set_facecolor("white")
        xs = [r.config[key] for r in sorted_r]
        ys = [r.metric(metric, target_loss=target_loss) for r in sorted_r]
        ax.scatter(xs, ys, alpha=0.6, s=20)
        ax.set_xlabel(key)
        ax.set_ylabel(metric)
        if metric == "val_mse":
            ax.set_yscale("log")
        if key == "parsimony":
            ax.set_xscale("log")
        ax.grid(alpha=0.3)
    fig.suptitle(f"HPO sweep: {metric} vs hyperparams", fontsize=11)
    fig.tight_layout(); fig.savefig(outdir / "hpo2_hyperparam_scatter.png"); plt.close(fig)

    # --- 3. Pareto-front overlay ---
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    palette = plt.get_cmap("tab10").colors
    for i, r in enumerate(sorted_r[:5]):
        if not r.pareto_complexities:
            continue
        order = np.argsort(r.pareto_complexities)
        cs = np.asarray(r.pareto_complexities)[order]
        ls = np.asarray(r.pareto_losses)[order]
        label = f"#{i+1} (val={r.val_loss:.3g})"
        ax.plot(cs, ls, "o-", color=palette[i], label=label, alpha=0.8)
    ax.set_yscale("log")
    ax.set_xlabel("complexity"); ax.set_ylabel("training loss")
    ax.set_title(f"HPO top-5 Pareto fronts (sorted by {metric})")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "hpo3_pareto.png"); plt.close(fig)

    # --- 4. Wall-time vs metric ---
    fig, ax = plt.subplots(figsize=(6, 4.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    xs = [r.wall_time_s for r in sorted_r]
    ys = [r.metric(metric, target_loss=target_loss) for r in sorted_r]
    ax.scatter(xs, ys, alpha=0.7, s=22, color="C0")
    ax.set_xlabel("wall time per fit  [s]")
    ax.set_ylabel(metric)
    if metric == "val_mse":
        ax.set_yscale("log")
    ax.set_title("HPO efficiency frontier: cost vs quality")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outdir / "hpo4_walltime.png"); plt.close(fig)

    return outdir
