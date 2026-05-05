"""Tests for the multi-D cross-coupled forecast pipeline.

Validates the math of `MultiDRefitResult` (single eq over multiple θ +
k + res + z) and `MultiDCrossCoupledModel` (combine: multi-D eq for
the cross-coupled subset + GP-slice for everything else).

Uses `MockGPModel` to avoid the lyaemu dependency. Hand-built
`MultiDRefitResult` instances with known sympy strings let us check
the math without invoking PySR.
"""

from __future__ import annotations

import numpy as np
import pytest

from priya_forecast.models.gp_model import MockGPModel
from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.parameters import (
    PARAM_NAMES,
    fiducial_vector,
    get_param,
)
from priya_forecast.refit_1d_pysr import HF_RESOLUTION, LF_RESOLUTION
from priya_forecast.refit_multi_d import (
    DEFAULT_SUBSET,
    MultiDCrossCoupledModel,
    MultiDRefitResult,
)


def _hand_multid_refit(
    *,
    subset_names: tuple[str, ...],
    equation_str: str,
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    p_gp_lf_per_z: np.ndarray,    # shape (n_z, n_k) — at-fid anchor
    std_per_z: np.ndarray | None = None,
) -> MultiDRefitResult:
    """Build a MultiDRefitResult with a hand-written equation.

    Default `std_per_z` = 1.0 everywhere (so flux_norm == P_F).
    `p_gp_lf_per_z` plays the role of the `mean_flux` in the
    MultiZNormalizationSpec.
    """
    if std_per_z is None:
        std_per_z = np.ones((len(z_grid), len(k_grid)), dtype=float)
    bounds = np.array([get_param(n).prior for n in subset_names], dtype=float)
    norm = MultiZNormalizationSpec(
        param_min=0.0, param_max=1.0,
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        z_grid=np.asarray(z_grid, dtype=float),
        mean_flux=np.asarray(p_gp_lf_per_z, dtype=float),
        std_flux=np.asarray(std_per_z, dtype=float),
        k_grid=np.asarray(k_grid, dtype=float),
    )
    fid = np.array(fiducial_vector(), dtype=float)
    return MultiDRefitResult(
        subset_names=subset_names,
        z_min=float(z_grid.min()), z_max=float(z_grid.max()),
        equation_str=equation_str,
        pareto_complexity=5, pareto_loss=0.0,
        pareto_complexities=[5], pareto_losses=[0.0],
        x_param_min=bounds[:, 0], x_param_max=bounds[:, 1],
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        fid_phys=fid, norm=norm, k_grid=k_grid, wall_time_s=0.0,
        lf_train_mean_rel_err=0.0, hf_train_mean_rel_err=0.0,
        lf_train_max_rel_err=0.0, hf_train_max_rel_err=0.0,
    )


def test_predict_normalized_evaluates_equation_correctly():
    """Equation = c · x_idx_of_k → eval should give c · k_norm regardless of θ."""
    subset = ("ns", "Ap")
    z_grid = np.array([3.0, 3.6, 4.0])
    k = np.linspace(0.005, 0.064, 8)
    fid = np.array(fiducial_vector(), dtype=float)
    p_gp_anchor = np.zeros((len(z_grid), len(k)), dtype=float)

    # x0=θ_ns_norm, x1=θ_Ap_norm, x2=k_norm, x3=res, x4=z_norm.
    # Equation: 3.0 * x2 → output = 3 · k_norm.
    r = _hand_multid_refit(
        subset_names=subset, equation_str="3.0 * x2",
        k_grid=k, z_grid=z_grid, p_gp_lf_per_z=p_gp_anchor,
    )
    out = r.predict_normalized(theta_phys_full=fid, k=k, resolution=0.8, z=3.6)
    k_norm = (k - r.k_min) / (r.k_max - r.k_min)
    np.testing.assert_allclose(out, 3.0 * k_norm, rtol=1e-10, atol=1e-10)


def test_predict_normalized_uses_subset_thetas():
    """eq = c · x0 → prediction depends on θ_ns (the first subset entry)."""
    subset = ("ns", "Ap")
    z_grid = np.array([3.0, 3.6, 4.0])
    k = np.linspace(0.005, 0.064, 8)
    fid = np.array(fiducial_vector(), dtype=float)
    p_gp_anchor = np.zeros((len(z_grid), len(k)), dtype=float)

    r = _hand_multid_refit(
        subset_names=subset, equation_str="2.5 * x0",
        k_grid=k, z_grid=z_grid, p_gp_lf_per_z=p_gp_anchor,
    )
    p_ns = get_param("ns")
    theta_full = fid.copy()
    theta_full[PARAM_NAMES.index("ns")] = p_ns.prior[0] + 0.5 * p_ns.width()
    out = r.predict_normalized(theta_phys_full=theta_full, k=k, resolution=0.8, z=3.6)
    # θ_ns_norm = 0.5 (we placed θ at midpoint). x0 = 0.5. eq = 2.5 · 0.5 = 1.25.
    np.testing.assert_allclose(out, np.full_like(k, 1.25), rtol=1e-10, atol=1e-10)


def test_cross_coupled_combine_exact_at_fid():
    """At θ=fid, the multi-D combine recovers P_GP(fid, k, z) exactly."""
    gp = MockGPModel()
    k = np.linspace(0.005, 0.064, 16)
    z_grid = np.array([3.0, 3.6, 4.0])
    fid = np.array(fiducial_vector(), dtype=float)

    # Fill the at-fid anchor from the GP.
    p_gp_anchor = np.array([gp.predict(fid, k, float(z)) for z in z_grid])

    subset = ("ns", "Ap", "herei", "heref", "alphaq", "hireionz")
    r = _hand_multid_refit(
        subset_names=subset, equation_str="0.4 * x0 - 0.2 * x1 + 0.1 * x6",
        k_grid=k, z_grid=z_grid, p_gp_lf_per_z=p_gp_anchor,
    )
    model = MultiDCrossCoupledModel(
        multi_d_refit=r, gp=gp, fid=fid, k_grid=k, z_grid=z_grid,
        fixed_params=("dtau0",),
    )
    for z in z_grid:
        p_hy = model.predict(fid, k, float(z))
        p_gp = gp.predict(fid, k, float(z))
        np.testing.assert_allclose(p_hy, p_gp, rtol=1e-10, atol=1e-10)


def test_subset_param_shift_moves_via_multi_d_eq():
    """Shifting a subset param moves P_F by the multi-D eq deviation."""
    gp = MockGPModel()
    k = np.linspace(0.005, 0.064, 16)
    z_grid = np.array([3.0, 3.6, 4.0])
    fid = np.array(fiducial_vector(), dtype=float)

    # std=1 per-(z, k); mean = GP at fid → predict = eq_value + p_gp_fid.
    p_gp_anchor = np.array([gp.predict(fid, k, float(z)) for z in z_grid])
    subset = ("ns", "Ap", "herei", "heref", "alphaq", "hireionz")

    # eq = 4.0 · x0 (depends only on θ_ns_norm). After de-norm with std=1,
    # the deviation = 4 · (θ_ns_norm − θ_ns_fid_norm).
    r = _hand_multid_refit(
        subset_names=subset, equation_str="4.0 * x0",
        k_grid=k, z_grid=z_grid, p_gp_lf_per_z=p_gp_anchor,
    )
    model = MultiDCrossCoupledModel(
        multi_d_refit=r, gp=gp, fid=fid, k_grid=k, z_grid=z_grid,
        fixed_params=("dtau0",),
    )

    # Shift ns to half its prior.
    p_ns = get_param("ns")
    theta = fid.copy()
    theta[PARAM_NAMES.index("ns")] = p_ns.prior[0] + 0.5 * p_ns.width()
    z_check = 3.6
    p_hy = model.predict(theta, k, z_check)
    p_gp_fid = gp.predict(fid, k, z_check)

    # Hybrid = GP(fid) + (eq(θ) − eq(fid)) · std_per_z + 0 from non-subset.
    # eq(θ_ns=mid) = 4·0.5 = 2.0 (other θ_norm at fid_norm don't matter).
    # eq(fid):
    fid_norm = (p_ns.fid - p_ns.prior[0]) / p_ns.width()
    eq_at_fid = 4.0 * fid_norm
    eq_at_theta = 4.0 * 0.5
    expected_delta = (eq_at_theta - eq_at_fid)  # std=1 → P_F units
    expected = p_gp_fid + expected_delta
    np.testing.assert_allclose(p_hy, expected, rtol=1e-10, atol=1e-10)


def test_non_subset_param_shift_uses_gp_slice():
    """Shifting a non-subset, non-fixed param moves P_F by the GP-slice deviation."""
    gp = MockGPModel()
    k = np.linspace(0.005, 0.064, 16)
    z_grid = np.array([3.0, 3.6, 4.0])
    fid = np.array(fiducial_vector(), dtype=float)
    p_gp_anchor = np.array([gp.predict(fid, k, float(z)) for z in z_grid])

    subset = ("ns", "Ap", "herei", "heref", "alphaq", "hireionz")
    # eq = 0 → multi-D contribution always 0.
    r = _hand_multid_refit(
        subset_names=subset, equation_str="0",
        k_grid=k, z_grid=z_grid, p_gp_lf_per_z=p_gp_anchor,
    )
    model = MultiDCrossCoupledModel(
        multi_d_refit=r, gp=gp, fid=fid, k_grid=k, z_grid=z_grid,
        fixed_params=("dtau0",),
    )

    # Shift hub (not in subset, not fixed). Hybrid = GP(fid) + GP-slice(hub).
    p_hub = get_param("hub")
    theta = fid.copy()
    theta[PARAM_NAMES.index("hub")] = p_hub.prior[0] + 0.7 * p_hub.width()
    z_check = 3.0
    p_hy = model.predict(theta, k, z_check)
    p_gp_fid = gp.predict(fid, k, z_check)
    p_gp_slice_hub = gp.predict(theta, k, z_check)
    expected = p_gp_fid + (p_gp_slice_hub - p_gp_fid)
    np.testing.assert_allclose(p_hy, expected, rtol=1e-10, atol=1e-10)


def test_fixed_param_shift_invariant():
    """Shifting a fixed param (dtau0) leaves the prediction unchanged.

    Even if `theta[i_dtau0] != fid[i_dtau0]`, the model treats it as
    fid because dtau0 is in the fixed_params set. The GP-slice fallback
    skips it entirely.
    """
    gp = MockGPModel()
    k = np.linspace(0.005, 0.064, 16)
    z_grid = np.array([3.6])
    fid = np.array(fiducial_vector(), dtype=float)
    p_gp_anchor = np.array([gp.predict(fid, k, float(z)) for z in z_grid])

    subset = ("ns", "Ap", "herei", "heref", "alphaq", "hireionz")
    r = _hand_multid_refit(
        subset_names=subset, equation_str="0",
        k_grid=k, z_grid=z_grid, p_gp_lf_per_z=p_gp_anchor,
    )
    model = MultiDCrossCoupledModel(
        multi_d_refit=r, gp=gp, fid=fid, k_grid=k, z_grid=z_grid,
        fixed_params=("dtau0",),
    )
    z_check = 3.6
    p_at_fid = model.predict(fid, k, z_check)

    # dtau0 strongly perturbed — model output must NOT change.
    theta = fid.copy()
    theta[PARAM_NAMES.index("dtau0")] = -0.4   # extreme
    p_perturbed = model.predict(theta, k, z_check)
    np.testing.assert_allclose(p_perturbed, p_at_fid, rtol=1e-10, atol=1e-10)


def test_combine_aggregates_multiple_non_subset_shifts():
    """Hybrid = GP(fid) + multi-D + Σ_j GP-slice(non-subset j)."""
    gp = MockGPModel()
    k = np.linspace(0.005, 0.064, 16)
    z_grid = np.array([3.6])
    fid = np.array(fiducial_vector(), dtype=float)
    p_gp_anchor = np.array([gp.predict(fid, k, float(z)) for z in z_grid])

    subset = ("ns", "Ap")  # narrow subset, so hub + omegamh2 + ... use GP-slice
    r = _hand_multid_refit(
        subset_names=subset, equation_str="0",
        k_grid=k, z_grid=z_grid, p_gp_lf_per_z=p_gp_anchor,
    )
    model = MultiDCrossCoupledModel(
        multi_d_refit=r, gp=gp, fid=fid, k_grid=k, z_grid=z_grid,
        fixed_params=("dtau0",),
    )

    # Move two non-subset params: hub and omegamh2.
    theta = fid.copy()
    p_hub = get_param("hub"); p_om = get_param("omegamh2")
    theta[PARAM_NAMES.index("hub")] = p_hub.prior[0] + 0.6 * p_hub.width()
    theta[PARAM_NAMES.index("omegamh2")] = p_om.prior[0] + 0.4 * p_om.width()

    z_check = 3.6
    p_hy = model.predict(theta, k, z_check)
    p_gp_fid = gp.predict(fid, k, z_check)

    # Each non-subset param's individual slice deviation summed.
    t_hub = fid.copy()
    t_hub[PARAM_NAMES.index("hub")] = theta[PARAM_NAMES.index("hub")]
    t_om = fid.copy()
    t_om[PARAM_NAMES.index("omegamh2")] = theta[PARAM_NAMES.index("omegamh2")]
    expected = (
        p_gp_fid
        + (gp.predict(t_hub, k, z_check) - p_gp_fid)
        + (gp.predict(t_om, k, z_check) - p_gp_fid)
    )
    np.testing.assert_allclose(p_hy, expected, rtol=1e-10, atol=1e-10)


def test_default_subset_is_user_spec():
    """Sanity: the default cross-coupled subset matches the user's plan."""
    assert DEFAULT_SUBSET == ("ns", "Ap", "herei", "heref", "alphaq", "hireionz")
