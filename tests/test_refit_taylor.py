"""Smoke tests for the additive-Taylor combiner with the resolution feature.

The student's `mf_*.py` evaluates each per-param equation at fid_norm=0.5
and resolution=0.8 in the multi-D combine. These tests construct
hand-built `Refit1DResult`s with 3-input expressions (x0=θ_norm,
x1=k_norm, x2=resolution) and verify the combiner reproduces the
expected formula:

    P_norm(θ, k) = Σ_i [eq_i(θ_i, k, 0.8) − eq_i(0.5, k, 0.8)]
                 + (1/n) Σ_i eq_i(0.5, k, 0.8)
    P_F(θ, k)   = P_norm(θ, k) · std_k_global + mean_k_global
"""

from __future__ import annotations

import numpy as np

from priya_forecast.models.gp_model import MockGPModel
from priya_forecast.models.normalization import NormalizationSpec
from priya_forecast.parameters import (
    PARAM_NAMES,
    fiducial_vector,
    get_param,
)
from priya_forecast.refit_1d_pysr import Refit1DResult, HF_RESOLUTION, LF_RESOLUTION
from priya_forecast.refit_taylor import (
    AdditiveTaylorModel,
    HF_RESOLUTION_FOR_COMBINE,
    STUDENT_FID_NORM,
    compute_global_normalization,
)


def _hand_refit(
    param_name: str,
    *,
    equation_str: str,
    k_grid: np.ndarray,
    global_norm: NormalizationSpec,
) -> Refit1DResult:
    """Build a Refit1DResult with a hand-written equation; no PySR call.

    The training-data range used for input min-max normalization defaults
    to the parameter's prior bounds (so θ_norm at fid is the actual
    fid_norm, not 0.5 — but the multi-D combine uses 0.5 by hardcode).
    """
    p = get_param(param_name)
    return Refit1DResult(
        param_name=param_name,
        z=3.6,
        equation_str=equation_str,
        pareto_complexity=3,
        pareto_loss=0.0,
        pareto_complexities=[3],
        pareto_losses=[0.0],
        x_param_min=float(p.prior[0]),
        x_param_max=float(p.prior[1]),
        k_min=float(k_grid.min()),
        k_max=float(k_grid.max()),
        lf_resolution=LF_RESOLUTION,
        hf_resolution=HF_RESOLUTION,
        fid_value=p.fid,
        norm=global_norm,
        k_grid=np.asarray(k_grid, dtype=float),
        wall_time_s=0.0,
        lf_train_mean_rel_err=0.0,
        hf_train_mean_rel_err=0.0,
        lf_train_max_rel_err=0.0,
        hf_train_max_rel_err=0.0,
    )


def test_compute_global_normalization_shapes():
    gp = MockGPModel()
    k = np.linspace(0.001, 0.02, 35)
    spec = compute_global_normalization(gp=gp, k_grid=k, z=3.6, n_train=64, seed=0)
    assert spec.mean_flux.shape == k.shape
    assert spec.std_flux.shape == k.shape
    assert np.all(spec.std_flux > 0)
    assert spec.mean_flux.min() > 0


def test_additive_taylor_at_fid_uses_fid_norm_05():
    """The constant reference is `eq_i(0.5, k_norm, 0.8)`, not `eq_i(fid_phys)`.

    With an equation that depends only on θ_norm (eq = c·x0), the
    constant term is `c · 0.5` and is k-independent.
    """
    gp = MockGPModel()
    k = np.linspace(0.001, 0.02, 35)
    z = 3.6
    fid = np.array(fiducial_vector(), dtype=float)

    global_norm = compute_global_normalization(gp=gp, k_grid=k, z=z, n_train=64, seed=0)

    # Equation depends only on x0 (θ_norm), with coefficient 4.0.
    refits = {pn: None for pn in PARAM_NAMES}
    refits["ns"] = _hand_refit("ns", equation_str="4.0 * x0",
                               k_grid=k, global_norm=global_norm)
    refits["Ap"] = _hand_refit("Ap", equation_str="-2.0 * x0",
                               k_grid=k, global_norm=global_norm)

    model = AdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits, global_norm=global_norm, k_grid=k, z=z,
        mode="multi_d",
    )
    p_at_fid = model.predict(fid, k, z)

    # At θ=fid_phys: θ_norm differs from 0.5 (since fid is not the prior midpoint),
    # so eq_i(θ_fid_phys) − eq_i(fid_norm=0.5) is non-zero.
    p_ns = get_param("ns"); p_ap = get_param("Ap")
    fid_ns_norm = (p_ns.fid - p_ns.prior[0]) / p_ns.width()
    fid_ap_norm = (p_ap.fid - p_ap.prior[0]) / p_ap.width()
    delta_ns = 4.0 * (fid_ns_norm - STUDENT_FID_NORM)
    delta_ap = -2.0 * (fid_ap_norm - STUDENT_FID_NORM)
    eq_at_05 = (4.0 * STUDENT_FID_NORM + (-2.0) * STUDENT_FID_NORM) / 2.0  # mean
    expected_norm = (delta_ns + delta_ap) + eq_at_05
    expected = global_norm.mean_flux + global_norm.std_flux * expected_norm
    np.testing.assert_allclose(p_at_fid, expected, rtol=1e-10, atol=1e-10)


def test_additive_taylor_responds_to_param_change():
    """Δθ on a refit param shifts P_F by std_k_global · Δeq."""
    gp = MockGPModel()
    k = np.linspace(0.001, 0.02, 35)
    z = 3.6
    fid = np.array(fiducial_vector(), dtype=float)
    global_norm = compute_global_normalization(gp=gp, k_grid=k, z=z, n_train=64, seed=0)

    refits = {pn: None for pn in PARAM_NAMES}
    refits["ns"] = _hand_refit("ns", equation_str="3.0 * x0",
                               k_grid=k, global_norm=global_norm)
    model = AdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits, global_norm=global_norm, k_grid=k, z=z,
        mode="multi_d",
    )
    p_fid = model.predict(fid, k, z)

    p_ns = get_param("ns")
    theta = fid.copy()
    theta[PARAM_NAMES.index("ns")] = p_ns.prior[0] + 0.5 * p_ns.width()
    p_shifted = model.predict(theta, k, z)

    fid_ns_norm = (p_ns.fid - p_ns.prior[0]) / p_ns.width()
    expected_delta_norm = 3.0 * (0.5 - fid_ns_norm)
    expected_delta_pf = global_norm.std_flux * expected_delta_norm
    np.testing.assert_allclose(p_shifted - p_fid, expected_delta_pf,
                               rtol=1e-10, atol=1e-10)


def test_resolution_feature_used():
    """When eq depends on x2 (resolution), combine evaluates at HF=0.8."""
    gp = MockGPModel()
    k = np.linspace(0.001, 0.02, 35)
    z = 3.6
    fid = np.array(fiducial_vector(), dtype=float)
    global_norm = compute_global_normalization(gp=gp, k_grid=k, z=z, n_train=64, seed=0)

    # eq depends ONLY on x2 (resolution): eq = 7 * x2. At combine time,
    # this evaluates to 7·0.8 regardless of θ.
    refits = {pn: None for pn in PARAM_NAMES}
    refits["ns"] = _hand_refit("ns", equation_str="7.0 * x2",
                               k_grid=k, global_norm=global_norm)
    model = AdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits, global_norm=global_norm, k_grid=k, z=z,
        mode="multi_d",
    )
    p_at_fid = model.predict(fid, k, z)
    # eq_i(any θ, any k, 0.8) = 5.6 → P_norm = 0 + 5.6 = 5.6.
    assert HF_RESOLUTION_FOR_COMBINE == 0.8
    expected_norm = 7.0 * HF_RESOLUTION_FOR_COMBINE
    expected = global_norm.mean_flux + global_norm.std_flux * expected_norm
    np.testing.assert_allclose(p_at_fid, expected, rtol=1e-10, atol=1e-10)


def test_local_anchored_mode_exact_at_fid():
    """Option B: mode='local_anchored' must reproduce P_GP(fid) exactly at fid."""
    gp = MockGPModel()
    k = np.linspace(0.001, 0.02, 35)
    z = 3.6
    fid = np.array(fiducial_vector(), dtype=float)

    # Build refits with proper per-param NormalizationSpec (Option B).
    # Each spec uses arbitrary mean/std; at fid the deviation cancels.
    refits = {pn: None for pn in PARAM_NAMES}
    for pname in ("ns", "Ap"):
        p = get_param(pname)
        per_param_norm = NormalizationSpec(
            param_min=float(p.prior[0]), param_max=float(p.prior[1]),
            k_min=float(k.min()), k_max=float(k.max()),
            mean_flux=np.full_like(k, 50.0), std_flux=np.full_like(k, 5.0),
            k_grid=k,
        )
        refits[pname] = Refit1DResult(
            param_name=pname, z=z,
            equation_str="2.5 * x0 + 0.7 * x1 + 0.3 * x2",
            pareto_complexity=5, pareto_loss=0.0,
            pareto_complexities=[5], pareto_losses=[0.0],
            x_param_min=float(p.prior[0]), x_param_max=float(p.prior[1]),
            k_min=float(k.min()), k_max=float(k.max()),
            lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
            fid_value=p.fid, norm=per_param_norm, k_grid=k, wall_time_s=0.0,
            lf_train_mean_rel_err=0.0, hf_train_mean_rel_err=0.0,
            lf_train_max_rel_err=0.0, hf_train_max_rel_err=0.0,
        )
    model = AdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits, global_norm=None,
        k_grid=k, z=z, mode="local_anchored",
    )
    p_at_fid = model.predict(fid, k, z)
    p_gp_fid = gp.predict(fid, k, z)
    np.testing.assert_allclose(p_at_fid, p_gp_fid, rtol=1e-12, atol=1e-12)


def test_local_anchored_mode_deviation_in_pf_units():
    """Option B: shifting one refit param changes P_F by [predict(θ) − predict(fid_phys)]."""
    gp = MockGPModel()
    k = np.linspace(0.001, 0.02, 35)
    z = 3.6
    fid = np.array(fiducial_vector(), dtype=float)
    p_ns = get_param("ns")

    # Linear-in-x0 equation, std_flux=5, mean_flux=50.
    norm = NormalizationSpec(
        param_min=float(p_ns.prior[0]), param_max=float(p_ns.prior[1]),
        k_min=float(k.min()), k_max=float(k.max()),
        mean_flux=np.full_like(k, 50.0), std_flux=np.full_like(k, 5.0),
        k_grid=k,
    )
    r = Refit1DResult(
        param_name="ns", z=z,
        equation_str="3.0 * x0",  # eq depends only on θ_norm
        pareto_complexity=3, pareto_loss=0.0,
        pareto_complexities=[3], pareto_losses=[0.0],
        x_param_min=float(p_ns.prior[0]), x_param_max=float(p_ns.prior[1]),
        k_min=float(k.min()), k_max=float(k.max()),
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        fid_value=p_ns.fid, norm=norm, k_grid=k, wall_time_s=0.0,
        lf_train_mean_rel_err=0.0, hf_train_mean_rel_err=0.0,
        lf_train_max_rel_err=0.0, hf_train_max_rel_err=0.0,
    )
    refits = {pn: None for pn in PARAM_NAMES}
    refits["ns"] = r
    model = AdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits, global_norm=None,
        k_grid=k, z=z, mode="local_anchored",
    )

    # Shift ns: deviation in P_F = (eq(θ_new) − eq(fid_phys)) · std + 0
    theta = fid.copy()
    theta[PARAM_NAMES.index("ns")] = p_ns.prior[0] + 0.5 * p_ns.width()
    p_shifted = model.predict(theta, k, z)
    p_gp_fid = gp.predict(fid, k, z)
    fid_norm = (p_ns.fid - p_ns.prior[0]) / p_ns.width()
    expected_delta_pf = (3.0 * (0.5 - fid_norm)) * 5.0  # std=5 from norm
    np.testing.assert_allclose(p_shifted - p_gp_fid, expected_delta_pf,
                               rtol=1e-10, atol=1e-10)


def test_gp_slice_fallback_recovers_gp_marginal_for_unrefit_param():
    gp = MockGPModel()
    k = np.linspace(0.001, 0.02, 35)
    z = 3.6
    fid = np.array(fiducial_vector(), dtype=float)
    global_norm = compute_global_normalization(gp=gp, k_grid=k, z=z, n_train=64, seed=0)

    refits = {pn: None for pn in PARAM_NAMES}
    model = AdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits, global_norm=global_norm, k_grid=k, z=z,
        mode="multi_d",
    )
    # No refits: at fid → P_F = mean_k_global.
    p_at_fid = model.predict(fid, k, z)
    np.testing.assert_allclose(p_at_fid, global_norm.mean_flux,
                               rtol=1e-10, atol=1e-10)

    p_ns = get_param("ns")
    theta = fid.copy()
    theta[PARAM_NAMES.index("ns")] = p_ns.prior[0] + 0.7 * p_ns.width()
    p_shifted = model.predict(theta, k, z)
    p_gp_fid = gp.predict(fid, k, z)
    p_gp_slice = gp.predict(theta, k, z)
    expected = global_norm.mean_flux + (p_gp_slice - p_gp_fid)
    np.testing.assert_allclose(p_shifted, expected, rtol=1e-10, atol=1e-10)
