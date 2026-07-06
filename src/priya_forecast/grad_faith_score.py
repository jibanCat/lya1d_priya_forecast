"""Shared grad-faithfulness scoring core.

One implementation used by BOTH the CLI (`scripts/eval_grad_faithfulness.py`)
and the tutorial `rerun.run_grid`, so a rerun scores exactly as the paper did.
`run_grid` passes a pre-loaded `gp_hf` so it scores in-process without reloading
the emulator per Pareto file (the CLI, run per-file on a cluster, loads its own).

Reads `get_param()` / `fiducial_vector()` at call time, so an active
`parameters.override_params(...)` context is honored (physics-override runs).
"""
from __future__ import annotations

import numpy as np

from priya_forecast.parameters import get_param, PARAM_NAMES, fiducial_vector
from priya_forecast.single_z import forecast as fc
from priya_forecast.single_z.training_data import load_1pvar
from priya_forecast.models.pysr_model import load_pareto_csv
from priya_forecast.derivative_gate import gp_param_gradient, equation_param_gradient
import priya_forecast.single_z.refit as _refit
from priya_forecast.grad_faith_io import equation_has_x0
from priya_forecast.refit_1d_pysr import HF_RESOLUTION


def median_rel_error(cand_grad, target_grad, floor_frac=1e-3):
    """Median over non-negligible k-bins of |cand/target - 1| (grad_err)."""
    cand = np.asarray(cand_grad, float)
    target = np.asarray(target_grad, float)
    amax = float(np.max(np.abs(target)))
    if amax == 0.0:
        return np.inf, 0
    keep = np.abs(target) >= floor_frac * amax
    if not np.any(keep):
        return np.inf, 0
    rel = np.abs(cand[keep] / target[keep] - 1.0)
    return float(np.median(rel)), int(np.sum(keep))


def score_pareto(*, pareto_csv, param, z, basedir="data/kodiaq_gp",
                 data_1pvar="data/single_z_1pvar", kmin=0.001, kmax=0.04,
                 tol=0.25, log_space=True, gp_hf=None):
    """Score every Fisher-safe candidate in a Pareto CSV.

    Returns a list of sidecar-row dicts with keys
    ``Complexity, Loss, grad_err, value_mse, n_keep, gate_pass, x0_enters``.

    If ``gp_hf`` is None the HF GP is loaded from ``basedir``; pass a pre-loaded
    ``GPModel`` to avoid reloading the emulator (the tutorial reuses one GP for
    the whole grid). Raises ValueError if the kodiaq and 1pvar k-grids differ
    (non-default ``kmin/kmax``).
    """
    k_grid = _refit.kodiaq_k_grid(kmin, kmax, 48)
    fid = np.asarray(fiducial_vector(), dtype=float)
    pidx = PARAM_NAMES.index(param)
    meta = get_param(param)

    if gp_hf is None:
        from priya_forecast.models.gp_model import GPModel
        gp_hf = GPModel(basedir=basedir, fidelity="hf", kf=k_grid)
        gp_hf.predict(fid, k_grid, z)

    target = gp_param_gradient(gp=gp_hf, fid=fid, k_grid=k_grid, z=z,
                               param_idx=pidx, log_space=log_space)

    df = load_pareto_csv(pareto_csv)
    safe = fc._filter_fisher_safe(df, n_features=3)
    d = load_1pvar(param_name=param, z=z, data_dir=data_1pvar)
    kg = np.asarray(d["kfkms_lf_z"][0], dtype=float)
    norm = fc.per_param_local_norm(
        flux_lf_z=d["flux_lf_z"], k_grid=kg,
        param_min=float(meta.prior[0]), param_max=float(meta.prior[1]),
        log_space=log_space,
    )
    # The candidate gradient uses kg (1pvar k) while the GP target gradient uses
    # k_grid (kodiaq); they coincide at the default bounds but a non-default
    # kmin/kmax would silently misalign the elementwise ratio.
    if not np.allclose(np.asarray(k_grid, float), kg):
        raise ValueError(
            "k_grid (kodiaq) and kg (1pvar) differ -- grad_err/value_mse would "
            "misalign. Re-run at the default kmin/kmax, or align the grids."
        )
    theta_grid = np.asarray(d["params_hf"][:, pidx], dtype=float)
    logP_gp_grid = np.empty((theta_grid.size, kg.size), dtype=float)
    for i, t in enumerate(theta_grid):
        tv = fid.copy()
        tv[pidx] = float(t)
        logP_gp_grid[i] = np.log(np.asarray(gp_hf.predict(tv, kg, z), float))

    rows = []
    for _, row in safe.sort_values("Loss").iterrows():
        cand = fc._refit_from_row(
            equation_str=str(row["Equation"]), complexity=int(row["Complexity"]),
            loss=float(row["Loss"]), df=df, param_name=param, z=z,
            meta=meta, k_grid=kg, norm=norm, log_space=log_space,
        )
        g = equation_param_gradient(refit=cand, fid_value=float(meta.fid),
                                    k_grid=kg, z=z, log_space=log_space)
        err, nkeep = median_rel_error(g, target)
        logP_eq = np.array([
            cand.predict_log(theta_phys=float(t), k=kg,
                             resolution=HF_RESOLUTION, z=z)
            for t in theta_grid
        ])
        value_mse = float(np.mean((logP_eq - logP_gp_grid) ** 2))
        rows.append({
            "Complexity": int(row["Complexity"]),
            "Loss": float(row["Loss"]),
            "grad_err": err,
            "value_mse": value_mse,
            "n_keep": int(nkeep),
            "gate_pass": bool(err <= tol),
            "x0_enters": bool(equation_has_x0(str(row["Equation"]))),
        })
    return rows
