"""Unit tests for the Phase-2 pair-coupling module."""

from __future__ import annotations

import numpy as np
import pytest

from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector, get_param
from priya_forecast.refit_1d_pysr import Refit1DResult
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


def test_cross_difference_is_zero_for_axis_separable_eq():
    """If the eq has NO θ_i × θ_j coupling (e.g. x0 + x1), cross_diff = 0 by construction."""
    # f(x, y) = x + y → cross_diff(x, y) = (x+y) - (x+fj) - (fi+y) + (fi+fj) = 0
    pair = _make_pair(pair_names=("tau0", "ns"), eq="x0 + x1 + 0.1 * x2")
    k = pair.k_grid
    z = float(pair.norm.z_grid[1])
    cd = pair.cross_difference((1.05, 0.95), k, HF_RESOLUTION_FOR_COMBINE, z)
    np.testing.assert_allclose(cd, np.zeros_like(k), atol=1e-12)


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
