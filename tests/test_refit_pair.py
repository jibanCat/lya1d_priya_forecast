"""Unit tests for the Phase-2 pair-coupling module."""

from __future__ import annotations

import numpy as np
import pytest

from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D, fiducial_vector, get_param
from priya_forecast.refit_pair import (
    HF_RESOLUTION_FOR_COMBINE,
    MultiZPairCoupledModel,
    Refit2DPairResult,
)
from priya_forecast.refit_taylor import MultiZAdditiveTaylorModel


# ---------------------------------------------------------------------
# Test fixtures: build minimal pair refit + Phase-1 base + GP mock.
# ---------------------------------------------------------------------


def _make_norm(*, k_grid: np.ndarray, z_grid: np.ndarray,
               param_min: float, param_max: float) -> MultiZNormalizationSpec:
    n_z, n_k = len(z_grid), len(k_grid)
    rng = np.random.default_rng(0)
    return MultiZNormalizationSpec(
        param_min=param_min, param_max=param_max,
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        z_grid=z_grid,
        mean_flux=rng.uniform(0.5, 1.0, size=(n_z, n_k)),
        std_flux=rng.uniform(0.1, 0.3, size=(n_z, n_k)),
        k_grid=k_grid,
    )


def _make_pair(
    *, pair_names: tuple[str, str],
    eq: str = "x0 * x1 + 0.1 * x2",
    k_grid: np.ndarray | None = None,
    z_grid: np.ndarray | None = None,
) -> Refit2DPairResult:
    if k_grid is None:
        k_grid = np.linspace(0.005, 0.064, 8)
    if z_grid is None:
        z_grid = np.array([2.6, 3.6, 4.2])
    pi = get_param(pair_names[0])
    pj = get_param(pair_names[1])
    norm = _make_norm(
        k_grid=k_grid, z_grid=z_grid,
        param_min=pi.prior[0], param_max=pi.prior[1],
    )
    return Refit2DPairResult(
        pair_names=pair_names,
        equation_str=eq,
        pareto_complexity=5,
        pareto_loss=0.01,
        pareto_complexities=[1, 5],
        pareto_losses=[1.0, 0.01],
        x_pair_min=(pi.prior[0], pj.prior[0]),
        x_pair_max=(pi.prior[1], pj.prior[1]),
        k_min=float(k_grid.min()),
        k_max=float(k_grid.max()),
        fid_pair=(pi.fid, pj.fid),
        z_min=float(z_grid.min()),
        z_max=float(z_grid.max()),
        norm=norm,
        k_grid=k_grid,
    )


class _MockGP:
    """Predictable GP for closure tests: linear in θ, z-independent."""

    def predict(self, theta, k, z):
        return 1.0 + 0.1 * float(np.asarray(theta).sum()) + 0.0 * float(z) \
            + 0.01 * np.asarray(k, dtype=float)


# ---------------------------------------------------------------------
# Refit2DPairResult invariants.
# ---------------------------------------------------------------------


def test_pair_predict_normalized_shape():
    """Output broadcasts to k.shape."""
    pair = _make_pair(pair_names=("tau0", "ns"))
    k = np.linspace(0.005, 0.064, 8)
    out = pair.predict_normalized((1.0, 0.9), k, HF_RESOLUTION_FOR_COMBINE, 3.6)
    assert out.shape == k.shape
    assert np.all(np.isfinite(out))


def test_pair_predict_denormalizes_with_norm_spec():
    """predict() = predict_normalized() · std + mean (per (z, k))."""
    pair = _make_pair(pair_names=("tau0", "ns"))
    k = pair.k_grid
    z = float(pair.norm.z_grid[1])  # mid z
    norm_val = pair.predict_normalized((1.0, 0.9), k, HF_RESOLUTION_FOR_COMBINE, z)
    physical_val = pair.predict((1.0, 0.9), k, HF_RESOLUTION_FOR_COMBINE, z)
    expected = pair.norm.denormalize_flux(norm_val, k, z)
    np.testing.assert_allclose(physical_val, expected)


def test_cross_difference_is_zero_when_either_theta_at_fid():
    """The defining ANOVA invariant: cross_diff = 0 if θ_i = fid_i OR θ_j = fid_j."""
    pair = _make_pair(pair_names=("tau0", "ns"))
    k = pair.k_grid
    z = float(pair.norm.z_grid[1])
    fi, fj = pair.fid_pair

    # On θ_i = fid_i axis (any θ_j).
    cd_i = pair.cross_difference((fi, 0.95), k, HF_RESOLUTION_FOR_COMBINE, z)
    np.testing.assert_array_equal(cd_i, np.zeros_like(k))

    # On θ_j = fid_j axis (any θ_i).
    cd_j = pair.cross_difference((1.05, fj), k, HF_RESOLUTION_FOR_COMBINE, z)
    np.testing.assert_array_equal(cd_j, np.zeros_like(k))


def test_cross_difference_is_nonzero_off_axis():
    """When both θ are perturbed, cross_diff is nonzero (for a pair eq that uses x0 AND x1)."""
    pair = _make_pair(pair_names=("tau0", "ns"), eq="x0 * x1")  # pure cross-coupling
    k = pair.k_grid
    z = float(pair.norm.z_grid[1])
    cd = pair.cross_difference((1.05, 0.95), k, HF_RESOLUTION_FOR_COMBINE, z)
    assert np.any(np.abs(cd) > 1e-6), \
        f"cross_diff should be nonzero off-axis for x0*x1 eq; got {cd}"


def test_cross_difference_short_circuits_under_float_jitter():
    """Per PR #2 review item #5: the short-circuit must use np.isclose-style
    tolerance, not strict ==. A caller passing `theta = fid + 1e-16` from
    numpy arithmetic should still hit the short-circuit (returning exact
    zeros) instead of falling through to a 4-predict difference, which can
    suffer catastrophic cancellation at near-fid points.
    """
    pair = _make_pair(pair_names=("tau0", "ns"), eq="x0 * x1")
    k = pair.k_grid
    z = float(pair.norm.z_grid[1])
    fi, fj = pair.fid_pair
    # θ_i jittered, θ_j exactly at fid → must short-circuit.
    cd1 = pair.cross_difference((1.05, fj + 1e-16), k, HF_RESOLUTION_FOR_COMBINE, z)
    np.testing.assert_array_equal(cd1, np.zeros_like(k))
    # θ_i exactly at fid + jitter, θ_j perturbed → must short-circuit.
    cd2 = pair.cross_difference((fi + 1e-16, 0.95), k, HF_RESOLUTION_FOR_COMBINE, z)
    np.testing.assert_array_equal(cd2, np.zeros_like(k))


def test_cross_difference_nan_guard():
    """Per PR #2 review item #6: if any of the 4 predict() calls produces
    NaN/inf (e.g. a bad eq evaluating `log(negative)` near a boundary), the
    cross_diff must be replaced with zeros rather than NaN-poisoning the
    Fisher stencil. PySR Pareto filters catch most of these at fit time;
    this is a runtime backstop.
    """
    # An eq whose lambdified result is NaN at a specific input combo.
    # `log(-x0 + 0.5)` is NaN for x0 > 0.5 in normalized space.
    pair = _make_pair(
        pair_names=("tau0", "ns"),
        eq="log(-(x0) + 0.5) + x1",
    )
    k = pair.k_grid
    z = float(pair.norm.z_grid[1])
    # θ_tau0 in upper half of prior → θ_tau0_norm > 0.5 → eq returns NaN.
    cd = pair.cross_difference((1.20, 0.95), k, HF_RESOLUTION_FOR_COMBINE, z)
    # Guard kicks in → zeros instead of NaN.
    assert np.all(np.isfinite(cd)), "cross_diff must NOT propagate NaN"
    np.testing.assert_array_equal(cd, np.zeros_like(k))


def test_cross_difference_is_zero_for_axis_separable_eq():
    """If the eq has NO θ_i × θ_j coupling (e.g. x0 + x1), cross_diff = 0 by construction."""
    # f(x, y) = x + y → cross_diff(x, y) = (x+y) - (x+fj) - (fi+y) + (fi+fj) = 0
    pair = _make_pair(pair_names=("tau0", "ns"), eq="x0 + x1 + 0.1 * x2")
    k = pair.k_grid
    z = float(pair.norm.z_grid[1])
    cd = pair.cross_difference((1.05, 0.95), k, HF_RESOLUTION_FOR_COMBINE, z)
    np.testing.assert_allclose(cd, np.zeros_like(k), atol=1e-12)


def test_cross_difference_is_zero_for_no_x0_eq():
    """PAPER_NOTES § D8.5 'no-x0 (pair) failure mode': if the saved eq
    drops x0 entirely (uses only x1, k, r, z), cross_diff is identically
    0 by symbolic algebra — the saved pair is a graceful no-op,
    contributing nothing to Fisher in the θ_i direction.

    Concretely the v2 `tau0×Ap` production fit fell into this mode
    (after 3 retries failed to find an eq using both x0 AND x1, the
    best-anyway eq used x1 but not x0)."""
    pair = _make_pair(pair_names=("tau0", "Ap"), eq="x1 + 0.1 * x2 - x4")
    k = pair.k_grid
    z = float(pair.norm.z_grid[1])
    cd = pair.cross_difference((1.05, 0.95), k, HF_RESOLUTION_FOR_COMBINE, z)
    np.testing.assert_allclose(cd, np.zeros_like(k), atol=1e-12)


def test_cross_difference_is_zero_for_no_x1_eq():
    """Symmetric to the no-x0 case (PAPER_NOTES § D8.5): an eq dropping
    x1 still produces an identically-zero cross_diff. The v1 tau0×Ap
    fit took this route before the LF/HF normalization fix (PR #2
    review item #3)."""
    pair = _make_pair(pair_names=("tau0", "Ap"), eq="x0 + 0.1 * x2 - x4")
    k = pair.k_grid
    z = float(pair.norm.z_grid[1])
    cd = pair.cross_difference((1.05, 0.95), k, HF_RESOLUTION_FOR_COMBINE, z)
    np.testing.assert_allclose(cd, np.zeros_like(k), atol=1e-12)


def test_pickle_round_trip_strips_lambdify_cache():
    """`__getstate__` / `__setstate__` must drop the lambdified callable
    from the pickle (lambdas don't pickle reliably, and loading a stale
    cache from another process is a footgun). After round-trip, predict()
    rebuilds the cache lazily and produces the same values as the
    original."""
    import pickle as _pickle
    pair = _make_pair(pair_names=("tau0", "ns"), eq="x0 * x1 + 0.1 * x2")
    k = pair.k_grid
    z = float(pair.norm.z_grid[1])
    # Populate the lambdify cache on the source.
    out_src = pair.predict_normalized((1.05, 0.95), k, HF_RESOLUTION_FOR_COMBINE, z)
    assert pair._fn_cache is not None
    # Round-trip.
    restored = _pickle.loads(_pickle.dumps(pair))
    assert restored._fn_cache is None, (
        "Pickled pair must drop _fn_cache (lambdas are not portable "
        "across processes)."
    )
    out_restored = restored.predict_normalized(
        (1.05, 0.95), k, HF_RESOLUTION_FOR_COMBINE, z,
    )
    np.testing.assert_allclose(out_restored, out_src)
    assert restored._fn_cache is not None, "predict() should rebuild cache lazily."


def test_feature_count_uses_word_boundary():
    """`feature_count` should count distinct xN tokens, with word-boundary
    (so x1 doesn't match x10, x11, etc.)."""
    pair = _make_pair(pair_names=("tau0", "ns"), eq="x0 * x1 + 0.1 * x2")
    assert pair.feature_count() == 3
    # An eq with x4 (z) but not x3 (resolution).
    pair2 = _make_pair(pair_names=("tau0", "ns"), eq="x0 + x4")
    assert pair2.feature_count() == 2


def test_lambdify_cache_is_lazy_and_reused():
    """The lambdified callable is built on first call; repeated calls reuse it."""
    pair = _make_pair(pair_names=("tau0", "ns"))
    k = pair.k_grid
    assert pair._fn_cache is None
    pair.predict_normalized((1.0, 0.9), k, HF_RESOLUTION_FOR_COMBINE, 3.6)
    fn1 = pair._fn_cache
    assert fn1 is not None
    pair.predict_normalized((1.0, 0.9), k, HF_RESOLUTION_FOR_COMBINE, 3.6)
    assert pair._fn_cache is fn1, "lambdified callable should be cached"


# ---------------------------------------------------------------------
# MultiZPairCoupledModel composition.
# ---------------------------------------------------------------------


def _make_phase1_base(k_grid, z_grid, gp):
    """Build a minimal Phase-1 base with all 11 refits = None (pure GP)."""
    fid = np.array(fiducial_vector(), dtype=float)
    refits = {pn: None for pn in PARAM_NAMES}
    return MultiZAdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits, k_grid=k_grid, z_grid=z_grid,
    )


def test_pair_model_at_fid_matches_base():
    """At fid, every cross_diff is 0 → pair-coupled model ≡ base ≡ GP."""
    k_grid = np.linspace(0.005, 0.064, 8)
    z_grid = np.array([2.6, 3.6, 4.2])
    gp = _MockGP()
    base = _make_phase1_base(k_grid, z_grid, gp)
    pair = _make_pair(pair_names=("tau0", "ns"), k_grid=k_grid, z_grid=z_grid)
    model = MultiZPairCoupledModel(base=base, pairs=[pair])

    fid = np.array(fiducial_vector(), dtype=float)
    z = float(z_grid[1])
    p_base = base.predict(fid, k_grid, z)
    p_paired = model.predict(fid, k_grid, z)
    np.testing.assert_allclose(p_paired, p_base)


def test_pair_model_on_axis_matches_base():
    """If only θ_i is perturbed, the (i, j)-pair's cross_diff = 0; pair model ≡ base."""
    k_grid = np.linspace(0.005, 0.064, 8)
    z_grid = np.array([2.6, 3.6, 4.2])
    gp = _MockGP()
    base = _make_phase1_base(k_grid, z_grid, gp)
    pair = _make_pair(pair_names=("tau0", "ns"), k_grid=k_grid, z_grid=z_grid)
    model = MultiZPairCoupledModel(base=base, pairs=[pair])

    fid = np.array(fiducial_vector(), dtype=float)
    theta = fid.copy()
    theta[PARAM_NAMES.index("tau0")] = 1.05  # only τ0 perturbed
    z = float(z_grid[1])
    p_base = base.predict(theta, k_grid, z)
    p_paired = model.predict(theta, k_grid, z)
    np.testing.assert_allclose(p_paired, p_base)


def test_pair_model_off_axis_adds_cross_difference():
    """When BOTH τ0 and ns are perturbed, the pair's cross_diff contributes."""
    k_grid = np.linspace(0.005, 0.064, 8)
    z_grid = np.array([2.6, 3.6, 4.2])
    gp = _MockGP()
    base = _make_phase1_base(k_grid, z_grid, gp)
    # Use a strongly-coupled eq so the cross_diff is detectably nonzero.
    pair = _make_pair(
        pair_names=("tau0", "ns"), eq="x0 * x1",
        k_grid=k_grid, z_grid=z_grid,
    )
    model = MultiZPairCoupledModel(base=base, pairs=[pair])

    fid = np.array(fiducial_vector(), dtype=float)
    theta = fid.copy()
    theta[PARAM_NAMES.index("tau0")] = 1.05
    theta[PARAM_NAMES.index("ns")] = 0.95
    z = float(z_grid[1])
    p_base = base.predict(theta, k_grid, z)
    p_paired = model.predict(theta, k_grid, z)
    diff = p_paired - p_base
    expected = pair.cross_difference(
        (1.05, 0.95), k_grid, HF_RESOLUTION_FOR_COMBINE, z,
    )
    np.testing.assert_allclose(diff, expected)
    assert np.any(np.abs(diff) > 1e-6), \
        "cross_diff contribution should be nonzero off-axis"


def test_pair_model_rejects_pair_with_mismatched_fid():
    """Constructing with a pair whose fid_pair disagrees with base.fid raises."""
    k_grid = np.linspace(0.005, 0.064, 8)
    z_grid = np.array([2.6, 3.6, 4.2])
    gp = _MockGP()
    base = _make_phase1_base(k_grid, z_grid, gp)
    # Build a pair with a deliberately-wrong fid for tau0.
    pair = _make_pair(pair_names=("tau0", "ns"), k_grid=k_grid, z_grid=z_grid)
    pair.fid_pair = (1.5, pair.fid_pair[1])  # tau0 fid is normally 1.0
    with pytest.raises(ValueError, match="fid_pair"):
        MultiZPairCoupledModel(base=base, pairs=[pair])


def test_pair_model_with_no_pairs_is_identity_to_base():
    """`pairs=[]` → model.predict() ≡ base.predict() everywhere."""
    k_grid = np.linspace(0.005, 0.064, 8)
    z_grid = np.array([2.6, 3.6, 4.2])
    gp = _MockGP()
    base = _make_phase1_base(k_grid, z_grid, gp)
    model = MultiZPairCoupledModel(base=base, pairs=[])

    fid = np.array(fiducial_vector(), dtype=float)
    theta = fid.copy()
    theta[PARAM_NAMES.index("tau0")] = 1.1
    theta[PARAM_NAMES.index("ns")] = 0.93
    for z in z_grid:
        p_base = base.predict(theta, k_grid, float(z))
        p_paired = model.predict(theta, k_grid, float(z))
        np.testing.assert_allclose(p_paired, p_base)


def test_pair_coupled_fisher_at_fid_matches_base_fisher():
    """End-to-end integration check: at θ=fid, the Fisher matrix is
    *identical* whether we use the Phase-1 base or its
    `MultiZPairCoupledModel` wrap. This locks in the
    rank-additivity claim from `refit_pair.py:258` ("Fisher rank
    ≥ rank(Phase 1) + |pairs|") at the boundary case where no
    pair contributes.

    Argument: each Fisher diagonal entry F_ii = (∂_i m)·C⁻¹·(∂_i m)
    perturbs only θ_i; that keeps θ_j = fid_j ∀ j ≠ i, so every
    pair containing θ_i has cross_diff(θ_i, fid_j) ≡ 0 by the ANOVA
    identity. Off-diagonal F_ij is computed from the same
    single-axis ∂_i m and ∂_j m, so identity holds there too.
    """
    from priya_forecast.fisher import fisher_matrix
    from priya_forecast.likelihood import GaussianLikelihood

    k_grid = np.linspace(0.005, 0.064, 16)
    z_grid = np.array([2.6, 3.6, 4.2])
    gp = _MockGP()
    base = _make_phase1_base(k_grid, z_grid, gp)
    pair = _make_pair(
        pair_names=("tau0", "ns"), eq="x0 * x1",
        k_grid=k_grid, z_grid=z_grid,
    )
    coupled = MultiZPairCoupledModel(base=base, pairs=[pair])

    fid = np.array(fiducial_vector(), dtype=float)
    params_active = tuple(p for p in PARAMS_11D if p.name != "dtau0")
    param_indices = [PARAM_NAMES.index(p.name) for p in params_active]
    z_eval = float(z_grid[1])

    # Pass our k_grid + a synthetic diagonal cov so the likelihood
    # doesn't try to load eBOSS data on a 35-bin grid the model wasn't
    # built for.
    lk_base = GaussianLikelihood(
        model=base, z=z_eval, mock_data="gp", theta_fid=fid,
        k_grid=k_grid, cov_diag_frac=0.05,
    )
    lk_coupled = GaussianLikelihood(
        model=coupled, z=z_eval, mock_data="gp", theta_fid=fid,
        k_grid=k_grid, cov_diag_frac=0.05,
    )
    fr_base = fisher_matrix(
        likelihood=lk_base, theta_fid=fid, params=params_active,
        param_indices=param_indices, step_frac=0.01, max_halvings=1,
    )
    fr_coupled = fisher_matrix(
        likelihood=lk_coupled, theta_fid=fid, params=params_active,
        param_indices=param_indices, step_frac=0.01, max_halvings=1,
    )
    np.testing.assert_allclose(fr_coupled.F, fr_base.F, rtol=1e-10, atol=1e-12)
