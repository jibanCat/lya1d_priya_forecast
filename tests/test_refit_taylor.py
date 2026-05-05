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
from priya_forecast.models.normalization import MultiZNormalizationSpec
from priya_forecast.refit_1d_pysr import Refit1DResult, HF_RESOLUTION, LF_RESOLUTION
from priya_forecast.refit_taylor import (
    AdditiveTaylorModel,
    HF_RESOLUTION_FOR_COMBINE,
    MultiZAdditiveTaylorModel,
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


# -----------------------------------------------------------------------
# MultiZAdditiveTaylorModel: partial-param routing (per-1D Taylor for
# good refits, GP-slice fallback for refits explicitly set to None,
# e.g. by the aggregator's quality gate).
# -----------------------------------------------------------------------


def _hand_multiz_refit(
    pname: str,
    *,
    equation_str: str,
    k_grid: np.ndarray,
    z_grid: np.ndarray,
    mean_flux: float = 50.0,
    std_flux: float = 5.0,
) -> Refit1DResult:
    """Build a multi-z Refit1DResult with a hand-written equation.

    The bundled `MultiZNormalizationSpec` is constant across (z, k) so the
    test math stays simple: predict() = (eq_value · std + mean).
    """
    p = get_param(pname)
    n_z, n_k = z_grid.size, k_grid.size
    norm = MultiZNormalizationSpec(
        param_min=float(p.prior[0]), param_max=float(p.prior[1]),
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        z_grid=np.asarray(z_grid, dtype=float),
        mean_flux=np.full((n_z, n_k), mean_flux, dtype=float),
        std_flux=np.full((n_z, n_k), std_flux, dtype=float),
        k_grid=np.asarray(k_grid, dtype=float),
    )
    return Refit1DResult(
        param_name=pname,
        z=float((z_grid[0] + z_grid[-1]) / 2.0),
        equation_str=equation_str,
        pareto_complexity=3, pareto_loss=0.0,
        pareto_complexities=[3], pareto_losses=[0.0],
        x_param_min=float(p.prior[0]), x_param_max=float(p.prior[1]),
        k_min=float(k_grid.min()), k_max=float(k_grid.max()),
        lf_resolution=LF_RESOLUTION, hf_resolution=HF_RESOLUTION,
        fid_value=p.fid, norm=norm, k_grid=np.asarray(k_grid, dtype=float),
        wall_time_s=0.0,
        lf_train_mean_rel_err=0.0, hf_train_mean_rel_err=0.0,
        lf_train_max_rel_err=0.0, hf_train_max_rel_err=0.0,
        z_min=float(z_grid[0]), z_max=float(z_grid[-1]),
    )


def test_multi_z_partial_routing_gp_slice_for_unrefit_param():
    """Refits set to None route through the GP-slice fallback.

    Construct a MultiZAdditiveTaylorModel with one good refit ('ns') and
    one explicitly-None refit ('Ap'). Vary 'Ap': the change in predicted
    P_F should equal the GP-slice deviation `P_GP(fid|Ap=θ_Ap) −
    P_GP(fid)`. Vary 'ns': should equal the per-1D Taylor deviation.
    """
    gp = MockGPModel()
    k = np.linspace(0.001, 0.02, 35)
    z_grid = np.array([3.4, 3.6, 3.8])
    z = 3.6
    fid = np.array(fiducial_vector(), dtype=float)

    # 'ns' has a good refit (linear in x0). 'Ap' is set to None → GP-slice.
    refits = {pn: None for pn in PARAM_NAMES}
    refits["ns"] = _hand_multiz_refit(
        "ns", equation_str="3.0 * x0",
        k_grid=k, z_grid=z_grid, mean_flux=50.0, std_flux=5.0,
    )
    # All others (incl. 'Ap') stay None.

    model = MultiZAdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits, k_grid=k, z_grid=z_grid,
    )

    # 1) At fid: hybrid == P_GP(fid) exactly (every deviation cancels).
    p_at_fid = model.predict(fid, k, z)
    p_gp_fid = gp.predict(fid, k, z)
    np.testing.assert_allclose(p_at_fid, p_gp_fid, rtol=1e-12, atol=1e-12)

    # 2) Vary 'Ap' (no refit → GP-slice path): hybrid - P_GP(fid)
    #    should equal P_GP(fid|Ap=θ_Ap) - P_GP(fid).
    p_ap = get_param("Ap")
    theta = fid.copy()
    theta[PARAM_NAMES.index("Ap")] = p_ap.prior[0] + 0.7 * p_ap.width()
    p_hy = model.predict(theta, k, z)
    p_gp_slice_ap = gp.predict(theta, k, z)
    np.testing.assert_allclose(
        p_hy - p_gp_fid, p_gp_slice_ap - p_gp_fid,
        rtol=1e-10, atol=1e-10,
    )

    # 3) Vary 'ns' (per-1D Taylor path): hybrid - P_GP(fid) should equal
    #    [eq_ns(θ_ns) - eq_ns(fid_ns)] · std + 0  =  3·Δ(x0) · std.
    p_ns = get_param("ns")
    theta2 = fid.copy()
    theta2[PARAM_NAMES.index("ns")] = p_ns.prior[0] + 0.5 * p_ns.width()
    p_hy2 = model.predict(theta2, k, z)
    fid_ns_norm = (p_ns.fid - p_ns.prior[0]) / p_ns.width()
    expected_delta = (3.0 * (0.5 - fid_ns_norm)) * 5.0
    np.testing.assert_allclose(
        p_hy2 - p_gp_fid, expected_delta, rtol=1e-10, atol=1e-10,
    )


def test_multi_z_partial_routing_gated_refit_matches_gp_slice():
    """Simulating the aggregator's gate: a previously-loaded refit set to
    None must route through GP-slice — its Fisher gradient becomes the
    GP's gradient, NOT the broken eq's gradient.

    Compare two models built on the same fid but with different refit
    dicts:
      A) refits['Ap'] = good_refit            -> per-1D Taylor for Ap
      B) refits['Ap'] = None                  -> GP-slice for Ap
    Both should agree at fid. Off-fid for Ap, A and B should disagree by
    exactly  [eq_Ap(θ_Ap) - eq_Ap(fid_Ap)]·std  -  [P_GP_slice - P_GP(fid)].
    """
    gp = MockGPModel()
    k = np.linspace(0.001, 0.02, 35)
    z_grid = np.array([3.4, 3.6, 3.8])
    z = 3.6
    fid = np.array(fiducial_vector(), dtype=float)

    good_refit = _hand_multiz_refit(
        "Ap", equation_str="2.0 * x0",
        k_grid=k, z_grid=z_grid, mean_flux=30.0, std_flux=4.0,
    )
    refits_A = {pn: None for pn in PARAM_NAMES}
    refits_A["Ap"] = good_refit
    refits_B = {pn: None for pn in PARAM_NAMES}  # 'Ap' explicitly None

    model_A = MultiZAdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits_A, k_grid=k, z_grid=z_grid,
    )
    model_B = MultiZAdditiveTaylorModel(
        gp=gp, fid=fid, refits=refits_B, k_grid=k, z_grid=z_grid,
    )

    # At fid: both must equal P_GP(fid).
    p_gp_fid = gp.predict(fid, k, z)
    np.testing.assert_allclose(model_A.predict(fid, k, z), p_gp_fid,
                                rtol=1e-12, atol=1e-12)
    np.testing.assert_allclose(model_B.predict(fid, k, z), p_gp_fid,
                                rtol=1e-12, atol=1e-12)

    # Off-fid: A uses per-1D Taylor, B uses GP-slice.
    p_ap = get_param("Ap")
    theta = fid.copy()
    theta[PARAM_NAMES.index("Ap")] = p_ap.prior[0] + 0.6 * p_ap.width()
    p_A = model_A.predict(theta, k, z)
    p_B = model_B.predict(theta, k, z)

    # B should equal the GP-slice deviation exactly.
    p_gp_slice = gp.predict(theta, k, z)
    np.testing.assert_allclose(p_B, p_gp_slice, rtol=1e-10, atol=1e-10)

    # A should equal the per-1D Taylor deviation: linear-in-x0, std=4.
    fid_ap_norm = (p_ap.fid - p_ap.prior[0]) / p_ap.width()
    expected_taylor_delta = (2.0 * (0.6 - fid_ap_norm)) * 4.0
    np.testing.assert_allclose(p_A - p_gp_fid, expected_taylor_delta,
                                rtol=1e-10, atol=1e-10)
