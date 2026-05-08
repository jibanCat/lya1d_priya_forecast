"""Smoke test: log-target vs linear-target for the Ap per-1D PySR fit.

Per user direction (2026-05-05): try training PySR on `log(P_F)` instead
of `(P_F − mean)/std` for amplitude-like params (Ap = primordial scalar
amplitude, near-pure linear amplitude in P_F). The fundamental
relationship `P_F ∝ A_s · shape(k, z)` becomes additive in log-space:
`log(P_F) = log(A_s) + log(shape)`, which PySR may find easier than the
nonlinear `(k^θ_Ap)`-style patterns that the linear target induces.

Smoke (NOT touching production pipeline):
  - Load existing Ap payload (`results/refit_phase2_production/payloads/Ap.pkl`).
  - Build TWO training matrices:
    (a) linear: `Y = (P_F − mean_per_(z,k)) / std_per_(z,k)` (current).
    (b) log:    `Y = (log P_F − log mean_per_(z,k)) / std_log_per_(z,k)`.
  - Fit PySR on each with `SMART_REFIT_PYSR_KWARGS` (option B).
  - Convert log-target predictions back to linear-P_F via exp().
  - Compare LF/HF rel-err in linear-P_F space.
  - Also report the eq's gradient at fid w.r.t. θ_Ap_norm.

Run (~10-20 min on login node):
  PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia \\
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/smoke/refit_ap_log_target_smoke.py
"""

from __future__ import annotations

import os
import pickle
import sys
import time
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(Path.home() / ".julia_env"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(Path.home() / ".julia"))

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.parameters import PARAM_NAMES, fiducial_vector, get_param
from priya_forecast.refit_1d_pysr import (
    HF_RESOLUTION, LF_RESOLUTION, SMART_REFIT_PYSR_KWARGS,
)


def _build_log_training_matrix(*, payload: dict, norm) -> tuple[np.ndarray, np.ndarray, dict]:
    """Stack LF + HF in *log* space.

    `mean_per_(z, k) = log(LF_GP(fid, k, z))` (consistent with the linear
    at-fid anchor: at θ=fid, P_F equals LF_GP(fid), so log-residual is 0).
    `std_per_(z, k) = empirical std of log(LF flux) at that (z, k)`.
    """
    flux_lf = payload["flux_lf_z"]
    flux_hf = payload["flux_hf_z"]
    k_lf = payload["kfkms_lf_z"]
    params_lf = payload["params_lf"]
    z_per_row = payload["z_per_row"]
    z_grid = np.array([2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
    z_grid = z_grid[(z_grid >= z_per_row.min()-1e-6) & (z_grid <= z_per_row.max()+1e-6)]

    log_flux_lf = np.log(flux_lf)
    log_flux_hf = np.log(flux_hf)
    # Per-(z, k) stats from log(LF flux). mean = log(LF_GP(fid)) lookup;
    # std = empirical std of log(LF flux) at that z bin.
    n_total, n_k = flux_lf.shape
    k_use = k_lf[0]
    log_mean_per_row = np.zeros_like(flux_lf)
    log_std_per_row = np.zeros_like(flux_lf)
    log_mean_lf_at_fid_per_z = np.log(norm.mean_flux)  # uses the at-fid LF anchor
    for r in range(n_total):
        zi = int(np.argmin(np.abs(z_grid - z_per_row[r])))
        log_mean_per_row[r] = np.interp(k_use, norm.k_grid, log_mean_lf_at_fid_per_z[zi])
        # Use empirical std of log-flux at this z bin.
        mask = (np.abs(z_per_row - z_grid[zi]) < 1e-3)
        log_std_per_row[r] = log_flux_lf[mask].std(axis=0, ddof=0)
    log_std_per_row = np.where(log_std_per_row > 0, log_std_per_row, 1.0)

    log_flux_lf_norm = (log_flux_lf - log_mean_per_row) / log_std_per_row
    log_flux_hf_norm = (log_flux_hf - log_mean_per_row) / log_std_per_row

    p_idx = PARAM_NAMES.index("Ap")
    x_param_lf = np.repeat(params_lf[:, p_idx, None], n_k, axis=1)
    x_param_min = float(x_param_lf.min())
    x_param_max = float(x_param_lf.max())
    x_param_norm = (x_param_lf - x_param_min) / (x_param_max - x_param_min)
    k_min = float(k_lf.min())
    k_max = float(k_lf.max())
    k_norm = (k_lf - k_min) / (k_max - k_min)
    z_min = float(z_per_row.min())
    z_max = float(z_per_row.max())
    z_2d = np.repeat(z_per_row[:, None], n_k, axis=1)
    z_norm_2d = (z_2d - z_min) / (z_max - z_min) if z_max > z_min else np.zeros_like(z_2d)

    n = flux_lf.size
    X_lf = np.column_stack([
        x_param_norm.ravel(), k_norm.ravel(), np.full(n, LF_RESOLUTION), z_norm_2d.ravel(),
    ])
    X_hf = np.column_stack([
        x_param_norm.ravel(), k_norm.ravel(), np.full(n, HF_RESOLUTION), z_norm_2d.ravel(),
    ])
    Y_lf = log_flux_lf_norm.ravel()
    Y_hf = log_flux_hf_norm.ravel()
    X_act = np.vstack([X_lf, X_hf])
    Y_act = np.concatenate([Y_lf, Y_hf])
    return X_act, Y_act, dict(
        log_mean_per_row=log_mean_per_row, log_std_per_row=log_std_per_row,
        x_param_min=x_param_min, x_param_max=x_param_max,
        k_min=k_min, k_max=k_max, z_min=z_min, z_max=z_max,
        n_total=n_total, n_k=n_k,
    )


def _eval_eq_at_fid(eq_str: str, x_param_min, x_param_max, fid_phys,
                    n_features: int = 4) -> float:
    """Evaluate the eq at θ_norm=fid_norm, k_norm=0.5, r=0.8, z_norm=0.5
    and return the slope w.r.t. θ_norm at that point (finite-diff)."""
    import sympy as sp
    expr = sp.sympify(eq_str)
    syms = [sp.Symbol(f"x{i}") for i in range(n_features)]
    fn = sp.lambdify(syms, expr, modules=[{"inv": lambda x: 1/x}, "numpy"])
    fid_norm = (fid_phys - x_param_min) / (x_param_max - x_param_min)
    h = 0.01
    y_plus = float(fn(fid_norm + h, 0.5, 0.8, 0.5))
    y_minus = float(fn(fid_norm - h, 0.5, 0.8, 0.5))
    return (y_plus - y_minus) / (2 * h)


def main():
    payload_path = Path("results/refit_phase2_production/payloads/Ap.pkl")
    if not payload_path.exists():
        raise SystemExit(f"Missing payload: {payload_path}")
    with open(payload_path, "rb") as fh:
        bundle = pickle.load(fh)
    payload = bundle["payload"]
    norm = bundle["norm"]

    print(f"Loaded Ap payload: {payload['flux_lf_z'].shape[0]} Sobol pts × {payload['flux_lf_z'].shape[1]} k bins.")
    p_meta = get_param("Ap")
    print(f"Ap prior: [{p_meta.prior[0]}, {p_meta.prior[1]}], fid={p_meta.fid}.")

    # ---- Build the LOG-target training matrix ----
    print("\n=== Building log-target training matrix ===")
    X_log, Y_log, ranges = _build_log_training_matrix(payload=payload, norm=norm)
    print(f"  X_log shape: {X_log.shape}, Y_log range: "
          f"[{Y_log.min():.3f}, {Y_log.max():.3f}], std={Y_log.std():.3f}")

    # ---- Fit PySR on log target ----
    from pysr import PySRRegressor
    args = dict(SMART_REFIT_PYSR_KWARGS)
    args["random_state"] = 42
    args["niterations"] = 50
    print("\n=== Fitting PySR on LOG target (smart, option B) ===")
    t0 = time.time()
    model = PySRRegressor(**args)
    model.fit(X_log, Y_log.reshape(-1, 1))
    elapsed = time.time() - t0
    print(f"  PySR fit done in {elapsed:.1f}s.")
    pareto = model.equations_
    eq_strs = pareto["equation"].astype(str)
    has_x0 = eq_strs.str.contains("x0")
    if bool(has_x0.any()):
        best_idx = int(pareto.loc[has_x0, "loss"].idxmin())
    else:
        best_idx = int(pareto["loss"].idxmin())
    log_eq = str(pareto.iloc[best_idx]["equation"])
    log_complexity = int(pareto.iloc[best_idx]["complexity"])
    log_loss = float(pareto.iloc[best_idx]["loss"])
    print(f"  Selected eq (complexity {log_complexity}, loss {log_loss:.4g}):\n    {log_eq}")

    # ---- Evaluate log-eq predictions back to linear P_F ----
    print("\n=== Evaluating log-target eq vs truth (in linear P_F space) ===")
    import sympy as sp
    expr = sp.sympify(log_eq)
    syms = [sp.Symbol(f"x{i}") for i in range(4)]
    fn = sp.lambdify(syms, expr, modules=[{"inv": lambda x: 1/x}, "numpy"])

    n_total = ranges["n_total"]
    n_k = ranges["n_k"]
    flux_lf_truth = payload["flux_lf_z"]
    flux_hf_truth = payload["flux_hf_z"]
    log_mean = ranges["log_mean_per_row"]
    log_std = ranges["log_std_per_row"]

    rel_err_lf = []
    rel_err_hf = []
    for r in range(n_total):
        x0 = (payload["params_lf"][r, PARAM_NAMES.index("Ap")] - ranges["x_param_min"]) / (ranges["x_param_max"] - ranges["x_param_min"])
        k_norm = (payload["kfkms_lf_z"][r] - ranges["k_min"]) / (ranges["k_max"] - ranges["k_min"])
        z_norm = (payload["z_per_row"][r] - ranges["z_min"]) / (ranges["z_max"] - ranges["z_min"]) if ranges["z_max"] > ranges["z_min"] else 0.0
        # LF: r=0.4
        pred_log_norm_lf = np.asarray(fn(np.full(n_k, x0), k_norm, np.full(n_k, LF_RESOLUTION), np.full(n_k, z_norm)), dtype=float)
        pred_log_lf = pred_log_norm_lf * log_std[r] + log_mean[r]
        pred_lf = np.exp(pred_log_lf)
        rel_err_lf.append(np.abs(pred_lf - flux_lf_truth[r]) / np.abs(flux_lf_truth[r]))
        # HF
        pred_log_norm_hf = np.asarray(fn(np.full(n_k, x0), k_norm, np.full(n_k, HF_RESOLUTION), np.full(n_k, z_norm)), dtype=float)
        pred_log_hf = pred_log_norm_hf * log_std[r] + log_mean[r]
        pred_hf = np.exp(pred_log_hf)
        rel_err_hf.append(np.abs(pred_hf - flux_hf_truth[r]) / np.abs(flux_hf_truth[r]))
    rel_err_lf = np.asarray(rel_err_lf)
    rel_err_hf = np.asarray(rel_err_hf)
    print(f"  LOG-TARGET: LF rel-err mean/max: {rel_err_lf.mean()*100:.2f}% / {rel_err_lf.max()*100:.2f}%")
    print(f"  LOG-TARGET: HF rel-err mean/max: {rel_err_hf.mean()*100:.2f}% / {rel_err_hf.max()*100:.2f}%")

    # ---- Compare to existing linear-target Ap fit ----
    linear_fit_path = Path("results/refit_phase2_production/refits/Ap.pkl")
    if linear_fit_path.exists():
        with open(linear_fit_path, "rb") as fh:
            r_linear = pickle.load(fh)
        print(f"\n=== Reference: linear-target Ap eq from {linear_fit_path} ===")
        print(f"  eq: {r_linear.equation_str}")
        print(f"  LF rel-err mean/max: {r_linear.lf_train_mean_rel_err*100:.2f}% / "
              f"{r_linear.lf_train_max_rel_err*100:.2f}%")
        print(f"  HF rel-err mean/max: {r_linear.hf_train_mean_rel_err*100:.2f}% / "
              f"{r_linear.hf_train_max_rel_err*100:.2f}%")

    # ---- Gradient at fid (slope w.r.t. θ_Ap_norm) ----
    print("\n=== Slope at fid w.r.t. θ_Ap_norm (mid k, mid z, HF) ===")
    log_grad = _eval_eq_at_fid(log_eq, ranges["x_param_min"], ranges["x_param_max"], p_meta.fid)
    print(f"  log-eq slope (in normalized log-space): {log_grad:.4g}")
    if linear_fit_path.exists():
        lin_grad = _eval_eq_at_fid(
            r_linear.equation_str,
            r_linear.x_param_min, r_linear.x_param_max, p_meta.fid,
        )
        print(f"  linear-eq slope (in normalized space):    {lin_grad:.4g}")

    # ---- Save smoke results ----
    out_dir = Path("results/smoke_ap_log_target")
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        out_dir / "smoke.npz",
        log_eq=log_eq, log_complexity=log_complexity, log_loss=log_loss,
        rel_err_lf=rel_err_lf, rel_err_hf=rel_err_hf,
        log_grad=log_grad,
    )
    md = [
        "# Smoke: log-target vs linear-target for Ap",
        f"emulator: kodiaq_2_2_4_6-48-48; Ap payload: {payload_path}",
        f"fit time: {elapsed:.1f}s with niter=50, smart kwargs (option B).",
        "",
        f"## log-target eq",
        f"```\n{log_eq}\n```",
        f"complexity={log_complexity}, normalized loss={log_loss:.4g}",
        "",
        "## rel-err (linear P_F space, after exp())",
        f"- LF: mean={rel_err_lf.mean()*100:.2f}%, max={rel_err_lf.max()*100:.2f}%",
        f"- HF: mean={rel_err_hf.mean()*100:.2f}%, max={rel_err_hf.max()*100:.2f}%",
        "",
        "## slope at fid (θ_Ap_norm direction)",
        f"- log-eq:    {log_grad:.4g}",
    ]
    if linear_fit_path.exists():
        md += [
            f"- linear-eq: {lin_grad:.4g}",
            "",
            f"## reference linear-target eq (from {linear_fit_path})",
            f"```\n{r_linear.equation_str}\n```",
            f"- LF rel-err mean/max: {r_linear.lf_train_mean_rel_err*100:.2f}% / "
            f"{r_linear.lf_train_max_rel_err*100:.2f}%",
            f"- HF rel-err mean/max: {r_linear.hf_train_mean_rel_err*100:.2f}% / "
            f"{r_linear.hf_train_max_rel_err*100:.2f}%",
        ]
    md += [
        "",
        "## interpretation",
        "- If log-eq has lower **max** rel-err → log-target is more Lipschitz off-fid.",
        "- If |log-eq slope| ≈ |linear-eq slope| × |∂P_F/∂log P_F| → log-target gradient",
        "  matches linear's at-fid Fisher contribution (to first order).",
        "- Gradient mismatch in the *current* linear-eq drives the σ_PySR/σ_GP=2.62×",
        "  Ap regression at fid; if log-eq has comparable shape but better Lipschitz",
        "  off-fid, switching to log-target is the cleaner production target.",
    ]
    (out_dir / "smoke.md").write_text("\n".join(md) + "\n")
    print(f"\nWrote {out_dir / 'smoke.md'}")


if __name__ == "__main__":
    main()
