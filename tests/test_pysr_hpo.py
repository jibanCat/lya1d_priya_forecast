"""Unit + hypothesis tests for `priya_forecast.pysr_hpo`.

We use a fast stub trainer (polynomial least-squares) instead of real
PySR so tests run in seconds. The HPO machinery — search-space sampling,
caching, sorting by metric, plotting — is identical regardless of the
trainer.
"""

from __future__ import annotations

import warnings
from itertools import combinations_with_replacement
from pathlib import Path

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from priya_forecast.pysr_hpo import (
    HPOResult,
    HPOSearchSpace,
    _hash_config,
    plot_hpo_results,
    run_hpo,
)


# ---------------------------------------------------------------------------
# Stub trainer: polynomial fits at "complexity" determined by the config.
# ---------------------------------------------------------------------------


def _stub_trainer(*, X_train, y_train, X_val, y_val, config, seed):
    """Tiny PySR-emulating fit: total-degree polynomial whose order grows
    with maxsize. Pareto front = orders 1..max_order with loss decreasing.
    """
    max_order = max(2, min(config["maxsize"] // 5, 6))
    n_dim = X_train.shape[1] if X_train.ndim > 1 else 1
    Xtr = X_train.reshape(-1, n_dim)
    Xva = X_val.reshape(-1, n_dim)

    def _terms(d):
        out = []
        for total in range(d + 1):
            for c in combinations_with_replacement(range(n_dim), total):
                out.append(tuple(c.count(i) for i in range(n_dim)))
        return out

    def _design(X, terms):
        return np.column_stack([np.prod(X ** np.asarray(t), axis=1) for t in terms])

    pareto_complexities, pareto_losses = [], []
    best_loss_val = np.inf
    best_loss_train = np.inf
    best_expr = "0"
    for d in range(1, max_order + 1):
        terms = _terms(d)
        A = _design(Xtr, terms)
        coeffs, *_ = np.linalg.lstsq(A, y_train, rcond=None)
        train_loss = float(np.mean((A @ coeffs - y_train) ** 2))
        val_loss = float(np.mean((_design(Xva, terms) @ coeffs - y_val) ** 2))
        complexity = len(terms)
        pareto_complexities.append(complexity)
        pareto_losses.append(train_loss)
        if val_loss < best_loss_val:
            best_loss_val = val_loss
            best_loss_train = train_loss
            best_expr = f"poly_d{d}_{n_dim}D"
    return best_loss_train, best_loss_val, pareto_complexities, pareto_losses, best_expr


# ---------------------------------------------------------------------------
# Toy 1D problem: y = sin(x) + x^2  (the spec's smoke test)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def toy_1d():
    rng = np.random.default_rng(0)
    X = rng.uniform(-3, 3, size=(200, 1))
    y = np.sin(X[:, 0]) + X[:, 0] ** 2
    return X[:150], y[:150], X[150:], y[150:]


# ---------------------------------------------------------------------------
# HPOSearchSpace
# ---------------------------------------------------------------------------


def test_search_space_default_grid_has_expected_size():
    space = HPOSearchSpace()
    configs = space.all_configs()
    # 3 * 2 * 2 * 3 * 3 * 2 * 3 = 648 — not a small grid, exercises grid_cap.
    assert len(configs) == 3 * 2 * 2 * 3 * 3 * 2 * 3


def test_search_space_random_sampling_returns_valid_configs():
    space = HPOSearchSpace()
    rng = np.random.default_rng(42)
    cfgs = space.sample_random(n=5, rng=rng)
    assert len(cfgs) == 5
    for cfg in cfgs:
        assert cfg["niterations"] in space.niterations
        assert cfg["maxsize"] in space.maxsize
        assert cfg["binary_operators"] in space.binary_operators


def test_search_space_random_with_seed_is_reproducible():
    space = HPOSearchSpace()
    a = space.sample_random(n=3, rng=np.random.default_rng(7))
    b = space.sample_random(n=3, rng=np.random.default_rng(7))
    assert a == b


# ---------------------------------------------------------------------------
# _hash_config
# ---------------------------------------------------------------------------


def test_hash_config_stable_under_same_inputs():
    cfg = {"niterations": 100, "maxsize": 20}
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)
    assert _hash_config(cfg, X, y) == _hash_config(cfg, X, y)


def test_hash_config_changes_with_different_data():
    cfg = {"niterations": 100, "maxsize": 20}
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)
    h1 = _hash_config(cfg, X, y)
    h2 = _hash_config(cfg, X + 1, y)
    assert h1 != h2


def test_hash_config_changes_with_different_config():
    X = np.arange(20).reshape(10, 2).astype(float)
    y = np.arange(10).astype(float)
    h1 = _hash_config({"niterations": 100}, X, y)
    h2 = _hash_config({"niterations": 200}, X, y)
    assert h1 != h2


# ---------------------------------------------------------------------------
# run_hpo — random strategy
# ---------------------------------------------------------------------------


def test_run_hpo_random_returns_n_results(toy_1d):
    Xtr, ytr, Xva, yva = toy_1d
    space = HPOSearchSpace(maxsize=[10, 20])  # tiny space
    out = run_hpo(
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        space=space, strategy="random", n_trials=3,
        metric="val_mse", trainer=_stub_trainer, seed=0,
    )
    assert len(out) == 3
    for r in out:
        assert isinstance(r, HPOResult)
        assert np.isfinite(r.val_loss)
    # Sorted ascending by val_loss
    losses = [r.val_loss for r in out]
    assert losses == sorted(losses)


def test_run_hpo_grid_strategy_within_cap(toy_1d):
    Xtr, ytr, Xva, yva = toy_1d
    space = HPOSearchSpace(
        niterations=[40], populations=[15], population_size=[33],
        maxsize=[10, 20], parsimony=[1e-3],
        binary_operators=[["+", "-", "*"]], unary_operators=[[]],
    )
    out = run_hpo(
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        space=space, strategy="grid", trainer=_stub_trainer,
    )
    assert len(out) == 2  # 1 * 1 * 1 * 2 * 1 * 1 * 1


def test_run_hpo_grid_rejects_too_large_space(toy_1d):
    Xtr, ytr, Xva, yva = toy_1d
    space = HPOSearchSpace()  # 648 configs > default cap
    with pytest.raises(ValueError, match="grid would launch"):
        run_hpo(
            X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
            space=space, strategy="grid", trainer=_stub_trainer,
        )


def test_run_hpo_caches_to_disk(toy_1d, tmp_path):
    Xtr, ytr, Xva, yva = toy_1d
    space = HPOSearchSpace(
        niterations=[40], populations=[15], population_size=[33],
        maxsize=[15], parsimony=[1e-3],
        binary_operators=[["+", "-", "*"]], unary_operators=[[]],
    )
    cache = tmp_path / "hpo_cache"
    out_a = run_hpo(
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        space=space, strategy="grid",
        trainer=_stub_trainer, cache_dir=cache,
    )
    files_after_first = list(cache.glob("*.pkl"))
    assert len(files_after_first) == len(out_a)

    # Second run with the same inputs should hit the cache and not retrain.
    sentinel = []

    def _spy_trainer(**kw):
        sentinel.append(1)
        return _stub_trainer(**kw)

    out_b = run_hpo(
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        space=space, strategy="grid",
        trainer=_spy_trainer, cache_dir=cache,
    )
    assert sentinel == [], "cache hit should skip training"
    assert out_b[0].val_loss == out_a[0].val_loss


def test_run_hpo_metric_complexity_at_target(toy_1d):
    Xtr, ytr, Xva, yva = toy_1d
    space = HPOSearchSpace(maxsize=[10, 20], niterations=[40], populations=[15],
                           population_size=[33], parsimony=[1e-3],
                           binary_operators=[["+"]], unary_operators=[[]])
    out = run_hpo(
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        space=space, strategy="grid", trainer=_stub_trainer,
        metric="complexity_at_target", target_loss=2.0,
    )
    # Sorted by min complexity such that loss <= 2.0.
    metrics = [r.metric("complexity_at_target", target_loss=2.0) for r in out]
    assert metrics == sorted(metrics)


def test_run_hpo_metric_pareto_area(toy_1d):
    Xtr, ytr, Xva, yva = toy_1d
    space = HPOSearchSpace(maxsize=[15], niterations=[40], populations=[15],
                           population_size=[33], parsimony=[1e-3],
                           binary_operators=[["+"]], unary_operators=[[]])
    out = run_hpo(
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        space=space, strategy="grid", trainer=_stub_trainer,
        metric="pareto_area",
    )
    assert len(out) == 1
    assert np.isfinite(out[0].metric("pareto_area"))


def test_run_hpo_bayesian_falls_back_to_random_when_optuna_missing(toy_1d, monkeypatch):
    Xtr, ytr, Xva, yva = toy_1d
    space = HPOSearchSpace(maxsize=[10, 20])
    # Force the import inside _bayesian_configs to fail.
    import builtins
    real_import = builtins.__import__

    def _no_optuna(name, *a, **kw):
        if name == "optuna":
            raise ImportError("simulated missing optuna")
        return real_import(name, *a, **kw)

    monkeypatch.setattr(builtins, "__import__", _no_optuna)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = run_hpo(
            X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
            space=space, strategy="bayesian", n_trials=2,
            trainer=_stub_trainer, seed=0,
        )
    assert any("optuna" in str(x.message) for x in w)
    assert len(out) == 2


def test_run_hpo_unknown_metric_raises(toy_1d):
    Xtr, ytr, Xva, yva = toy_1d
    space = HPOSearchSpace(maxsize=[10])
    out = run_hpo(
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        space=space, strategy="random", n_trials=1,
        metric="val_mse", trainer=_stub_trainer,
    )
    with pytest.raises(ValueError, match="Unknown metric"):
        out[0].metric("not_a_real_metric")


# ---------------------------------------------------------------------------
# Plotting smoke
# ---------------------------------------------------------------------------


def test_plot_hpo_results_writes_four_figures(toy_1d, tmp_path):
    Xtr, ytr, Xva, yva = toy_1d
    space = HPOSearchSpace(
        maxsize=[10, 15, 20], niterations=[40, 100],
        populations=[15], population_size=[33],
        parsimony=[1e-3], binary_operators=[["+", "-", "*"]],
        unary_operators=[[]],
    )
    out = run_hpo(
        X_train=Xtr, y_train=ytr, X_val=Xva, y_val=yva,
        space=space, strategy="random", n_trials=4,
        trainer=_stub_trainer, seed=0,
    )
    plot_hpo_results(out, outdir=tmp_path, metric="val_mse")
    for name in ("hpo1_top.png", "hpo2_hyperparam_scatter.png",
                 "hpo3_pareto.png", "hpo4_walltime.png"):
        assert (tmp_path / name).exists(), f"missing {name}"


# ---------------------------------------------------------------------------
# Property-based — hypothesis
# ---------------------------------------------------------------------------


@given(
    n=st.integers(min_value=1, max_value=8),
    seed=st.integers(min_value=0, max_value=99),
)
@settings(max_examples=8, deadline=None)
def test_property_random_sample_size_matches_request(n: int, seed: int):
    space = HPOSearchSpace()
    cfgs = space.sample_random(n=n, rng=np.random.default_rng(seed))
    assert len(cfgs) == n


@given(seed=st.integers(min_value=0, max_value=999))
@settings(max_examples=5, deadline=None)
def test_property_hpo_results_sorted_by_metric(seed: int):
    rng = np.random.default_rng(seed)
    X = rng.uniform(-1, 1, size=(80, 1))
    y = X[:, 0] ** 2
    space = HPOSearchSpace(
        maxsize=[10, 15, 20], niterations=[40, 100],
        populations=[15], population_size=[33],
        parsimony=[1e-3], binary_operators=[["+"]],
        unary_operators=[[]],
    )
    out = run_hpo(
        X_train=X[:60], y_train=y[:60], X_val=X[60:], y_val=y[60:],
        space=space, strategy="random", n_trials=3,
        trainer=_stub_trainer, seed=seed,
    )
    losses = [r.val_loss for r in out]
    assert losses == sorted(losses)
