# tests/test_stage9_end_to_end.py
"""Gated end-to-end: the Sobolev loss makes ns gradient-faithful at λ=5 —
the thing the value loss could not achieve (ns: all Fisher-safe equations had
69-97% gradient error → gate rejected every one). Validated locally
2026-06-04: ns best gradient error 0.134 at λ=5 (vs 0.433 at λ=1).
"""
import os

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("RUN_SLOW_REFIT"),
    reason="needs PySR/Julia + emulator; set RUN_SLOW_REFIT=1",
)


def test_sobolev_makes_ns_gradient_faithful(tmp_path):
    from pathlib import Path

    from priya_forecast.models.gp_model import GPModel
    from priya_forecast.parameters import PARAM_NAMES, get_param, fiducial_vector
    from priya_forecast.single_z.config import PipelineConfig
    from priya_forecast.single_z import refit as _r
    from priya_forecast.single_z.refit import kodiaq_k_grid
    from priya_forecast.models.pysr_model import load_pareto_csv
    from priya_forecast.single_z.forecast import (
        _filter_fisher_safe, per_param_local_norm, _refit_from_row)
    from priya_forecast.single_z.training_data import load_1pvar
    from priya_forecast.derivative_gate import (
        gp_param_gradient, equation_param_gradient, derivative_faithful)

    k = kodiaq_k_grid(0.001, 0.04, 48)
    gp_lf = GPModel(basedir="data/kodiaq_gp", fidelity="lf", kf=k)
    gp_hf = GPModel(basedir="data/kodiaq_gp", fidelity="hf", kf=k)
    cfg = PipelineConfig(mode="refit_and_forecast", redshift=3.6, target_space="log")
    cfg.pysr.use_sobolev = True
    cfg.pysr.sobolev_lambda = 5.0
    cfg.pysr.niterations = 40
    _r.refit_one_param_single_z(param_name="ns", z=3.6, cfg=cfg, gp_lf=gp_lf,
                                gp_hf=gp_hf, k_grid=k, out_dir=str(tmp_path))

    df = load_pareto_csv(Path(tmp_path) / "pareto_ns.csv")
    safe = _filter_fisher_safe(df, n_features=3).sort_values("Loss")
    meta = get_param("ns")
    fid = np.asarray(fiducial_vector(), float)
    tgt = gp_param_gradient(gp=gp_hf, fid=fid, k_grid=k, z=3.6,
                            param_idx=PARAM_NAMES.index("ns"))
    d = load_1pvar(param_name="ns", z=3.6, data_dir="data/single_z_1pvar")
    kg = d["kfkms_lf_z"][0]
    norm = per_param_local_norm(flux_lf_z=d["flux_lf_z"], k_grid=kg,
        param_min=float(meta.prior[0]), param_max=float(meta.prior[1]), log_space=True)
    passed = False
    for _, row in safe.iterrows():
        cand = _refit_from_row(equation_str=str(row["Equation"]),
            complexity=int(row["Complexity"]), loss=float(row["Loss"]), df=df,
            param_name="ns", z=3.6, meta=meta, k_grid=kg, norm=norm, log_space=True)
        g = equation_param_gradient(refit=cand, fid_value=float(meta.fid),
                                    k_grid=np.asarray(kg, float), z=3.6)
        if derivative_faithful(cand_grad=g, target_grad=tgt, tol=0.25):
            passed = True
            break
    assert passed, "Sobolev (λ=5) produced no derivative-faithful ns equation"
