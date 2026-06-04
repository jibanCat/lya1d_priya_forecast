# tests/test_multi_z_A_equals_B.py
"""Cross-z diagnostic: joint Fisher (Approach A) vs per-z-sum (Approach B).

A: run_three_fisher_multiz builds one z-spanning KSDataLikelihood + fisher_matrix.
B: per-z KSDataLikelihood -> compute_fisher_F_phys -> combine_fisher_phys_arrays.

Approach A is CORRECT for KSData: the KODIAQ-SQUAD covariance is NOT block-diagonal
in z (see ksdata_likelihood.py docstring), so the per-z aggregation in Approach B is
the biased legacy path.  This test is therefore a DIAGNOSTIC, NOT an equality check:
it asserts the production joint path (A) runs end-to-end and yields sane sigma, then
REPORTS the per-parameter A/B sigma-ratio as the cross-z bias deliverable (feeds the
Stage 7 COMPARISON.md).
"""
import os
import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_FORECAST_ONLY"),
    reason="needs lyaemu + data/kodiaq_gp; set RUN_SLOW_FORECAST_ONLY=1",
)


def test_joint_vs_per_z_sum_diagnostic():
    from priya_forecast.models.gp_model import GPModel
    from priya_forecast.parameters import PARAM_NAMES, PARAMS_11D, fiducial_vector
    from priya_forecast.ksdata_likelihood import KSDataLikelihood
    from priya_forecast.fisher import compute_fisher_F_phys, combine_fisher_phys_arrays
    from priya_forecast.multi_z.config import MultiZPipelineConfig
    from priya_forecast.multi_z.forecast import run_three_fisher_multiz

    params_subset = ["ns", "Ap", "tau0"]
    cfg = MultiZPipelineConfig(
        mode="forecast_only", z_min=3.2, z_max=3.6, parameters=params_subset,
    )
    cfg.gp.basedir = "data/kodiaq_gp"
    cfg.validate()
    gp = GPModel(basedir=cfg.gp.basedir)
    fid = np.asarray(fiducial_vector(), float)

    # --- Approach A: one z-spanning KSDataLikelihood via run_three_fisher_multiz ---
    res_A = run_three_fisher_multiz(
        cfg=cfg, gp=gp, fid=fid, refits={n: None for n in PARAM_NAMES},
    )
    sigma_A = res_A["GP"].sigma

    # --- Approach B: per-z KSDataLikelihood -> F_phys -> sum + invert ---
    indices = [PARAM_NAMES.index(n) for n in params_subset]
    selected = tuple(PARAMS_11D[i] for i in indices)
    # theta_fid for combine_fisher_phys_arrays: per-param subset values
    theta_fid_subset = np.array([fid[i] for i in indices], dtype=float)

    # Discover the z-bins in range from a spanning likelihood's z_blocks.
    like_span = KSDataLikelihood(
        model=gp, z_min=cfg.z_min, z_max=cfg.z_max,
        k_min=cfg.k_range.min, k_max=cfg.k_range.max,
        cov_scale=cfg.data.cov_scale, mock_data=cfg.data.mock_data,
        conservative=cfg.data.conservative,
    )
    z_values = [float(zv) for zv, _ in like_span.z_blocks]

    F_list = []
    for z in z_values:
        like_z = KSDataLikelihood(
            model=gp, z_min=z, z_max=z,
            k_min=cfg.k_range.min, k_max=cfg.k_range.max,
            cov_scale=cfg.data.cov_scale, mock_data=cfg.data.mock_data,
            conservative=cfg.data.conservative,
        )
        # compute_fisher_F_phys(*, likelihood, theta_fid, params, param_indices,
        #                        step_frac=0.02, rel_tol=0.05, max_halvings=2)
        # theta_fid is the FULL 11D vector; param_indices map each varying param
        # to its global position.  step_frac/rel_tol match cfg.fisher defaults.
        F_list.append(compute_fisher_F_phys(
            likelihood=like_z,
            theta_fid=fid,
            params=selected,
            param_indices=indices,
            step_frac=cfg.fisher.step_frac,
            rel_tol=cfg.fisher.rel_tol,
        ))

    # combine_fisher_phys_arrays(F_phys_list, *, params, theta_fid, priors_sigma=None)
    # theta_fid here is the per-varying-param subset (length == len(params)).
    fr_B = combine_fisher_phys_arrays(
        F_list,
        params=selected,
        theta_fid=theta_fid_subset,
    )
    sigma_B = fr_B.sigma

    # KSData covariance is NOT block-diagonal in z (see ksdata_likelihood.py
    # docstring): Approach A (joint) is correct; Approach B (per-z sum) is the
    # biased legacy path. We therefore do NOT assert equality — we assert the
    # production path is sane and REPORT the cross-z bias ratio.
    assert np.all(np.isfinite(sigma_A)), "joint Fisher (A) produced non-finite sigma"
    assert np.all(sigma_A > 0), "joint Fisher (A) produced non-positive sigma"
    ratio = np.asarray(sigma_A) / np.asarray(sigma_B)
    print("\n[A-vs-B cross-z diagnostic] sigma_A/sigma_B per param:")
    for name, r in zip(cfg.parameters, ratio):
        print(f"  {name:<10s} A/B = {r:.4f}")
    # Sanity floor/ceiling only to catch gross wiring bugs (NOT an equality check):
    assert np.all(np.isfinite(ratio)) and np.all(ratio > 0), \
        "A/B ratio non-finite or non-positive — likely a wiring bug, not physics"
