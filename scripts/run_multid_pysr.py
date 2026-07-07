"""Train multi-D PySR equations on a Sobol sweep of the GP, score them.

This is the full reward loop the student wants:

    1. Sample n_train Sobol points in the chosen parameter subspace.
    2. Evaluate the GP at each point on the eBOSS k-grid.
    3. Train PySR (real, not surrogate) on (theta_subset, k) → flux_norm.
    4. Pick a Pareto-front equation by best_loss / complexity_le / etc.
    5. Build a PySRModel with combine='joint' from that equation.
    6. Compare against the GP forecast — same scorecard as
       train_and_forecast.py but with REAL PySR.

Run:
    PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env \\
    JULIA_DEPOT_PATH=$HOME/.julia \\
    PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
        python scripts/run_multid_pysr.py \\
            --params dtau0 Ap ns alphaq \\
            --n-train 64 \\
            --niter 30 --maxsize 25 \\
            --output results/multid_pysr_run/

Caveat: PySR is slow. 30 iterations × 64 samples × 35 k-bins is ~minutes;
200 iterations × 256 samples is ~tens of minutes. Tune budget for the
turn-around you want.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(Path.home() / ".julia_env"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(Path.home() / ".julia"))

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.config import EqnConfig
from priya_forecast.data import load_eboss
from priya_forecast.diagnostics.compare import EqSetEntry, compare_equation_sets
from priya_forecast.models import PySRModel
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    fiducial_vector,
    get_param,
)


def _sobol_design(*, varying_names, n, seed=0):
    from scipy.stats import qmc

    fid = np.array(fiducial_vector(), dtype=float)
    sampler = qmc.Sobol(d=len(varying_names), seed=seed)
    u = sampler.random(n=n)
    out = np.tile(fid, (n, 1))
    for col, name in enumerate(varying_names):
        lo, hi = get_param(name).prior
        out[:, PARAM_NAMES.index(name)] = lo + (hi - lo) * u[:, col]
    return out


def _build_training(*, gp, varying_names, n, k_grid, z, seed=0):
    """Stack Sobol points × k → flat (X, y). Inputs are normalized to [0,1];
    output is raw P_F (we deliberately skip the (mean_k, std_k) normalization
    so PySR's equation is in physical units directly)."""
    thetas = _sobol_design(varying_names=varying_names, n=n, seed=seed)
    flux = np.stack([gp.predict(t, k_grid, z) for t in thetas], axis=0)  # (n, nk)
    rows_X, rows_y = [], []
    k_min, k_max = float(k_grid.min()), float(k_grid.max())
    for i, t in enumerate(thetas):
        for ki, k in enumerate(k_grid):
            row = []
            for name in varying_names:
                lo, hi = get_param(name).prior
                row.append((t[PARAM_NAMES.index(name)] - lo) / (hi - lo))
            row.append((k - k_min) / (k_max - k_min))
            rows_X.append(row)
            rows_y.append(flux[i, ki])
    return np.asarray(rows_X), np.asarray(rows_y)


def _spectrum(F, widths):
    """Eigen-spectrum + condition number + numerical rank of a Fisher block.

    `FisherResult.F` is `F_phys = Y^T Y` with `Y_i = L^-1 dP_F/dtheta_i` — i.e.
    the Gram matrix of the whitened parameter Jacobian ``J = dP_F/dtheta``. Its
    eigen-spectrum therefore mirrors the singular spectrum of J: a rank-deficient
    Jacobian (parameters folding into one shared sub-expression, §5.1) shows up
    as eigenvalues collapsing toward zero. Reported in two bases:

    - ``physical``: F as-is (per-parameter internal units).
    - ``whitened``: ``F_hat_ij = F_ij * w_i * w_j`` with ``w = prior width`` — the
      dimensionless form the forecast itself inverts, so genuine direction
      collapse is separated from trivial per-parameter unit-scale spread.

    Numerical rank at tolerance ``tol`` counts eigenvalues ``> tol * lambda_max``.
    """
    F = np.asarray(F, dtype=float)
    W = np.outer(widths, widths)
    out = {}
    for label, M in (("physical", F), ("whitened", F * W)):
        eig = np.sort(np.linalg.eigvalsh(0.5 * (M + M.T)))[::-1]
        lam_max = float(eig[0]) if eig.size else 0.0
        lam_min = float(eig[-1]) if eig.size else 0.0
        cond = (lam_max / lam_min) if lam_min > 0 else float("inf")
        ranks = {
            f"{tol:.0e}": (int(np.sum(eig > tol * lam_max)) if lam_max > 0 else 0)
            for tol in (1e-6, 1e-8, 1e-10)
        }
        out[label] = {
            "eigenvalues": eig.tolist(),
            "lambda_max": lam_max,
            "lambda_min": lam_min,
            "condition_number": cond,
            "numerical_rank": ranks,
        }
    return out


def _write_joint_rank_json(*, path, params, fr_student, fr_gp,
                           equation, loss, complexity, z):
    """Dump the §5.1 rank diagnostic (equation + loss + Fisher/Jacobian rank
    spectrum for the joint PySR fit and the GP reference) to `path` as JSON."""
    widths = np.array([p.width() for p in params], dtype=float)
    n = len(params)
    sig = lambda fr: [None if not np.isfinite(s) else float(s) for s in fr.sigma]
    diag = {
        "params": [p.name for p in params],
        "n_params": n,
        "z": float(z),
        "joint_equation": str(equation),
        "joint_loss": float(loss),
        "joint_complexity": int(complexity),
        "joint_pysr": {**_spectrum(fr_student.F, widths), "sigma": sig(fr_student)},
        "gp_reference": {**_spectrum(fr_gp.F, widths), "sigma": sig(fr_gp)},
    }
    rk8 = diag["joint_pysr"]["whitened"]["numerical_rank"]["1e-08"]
    diag["joint_pysr"]["rank_deficient_vs_nparams"] = bool(rk8 < n)

    def _san(o):  # JSON-safe: non-finite floats (e.g. inf cond) -> null
        if isinstance(o, float):
            return o if np.isfinite(o) else None
        if isinstance(o, dict):
            return {k: _san(v) for k, v in o.items()}
        if isinstance(o, list):
            return [_san(v) for v in o]
        return o

    Path(path).write_text(json.dumps(_san(diag), indent=2) + "\n")
    return diag


def main():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--params", nargs="+", required=True)
    parser.add_argument("--n-train", type=int, default=64)
    parser.add_argument("--niter", type=int, default=30)
    parser.add_argument("--maxsize", type=int, default=25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--z", type=float, default=3.6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    args.output.mkdir(parents=True, exist_ok=True)

    print("Loading real PRIYA GP emulator...")
    from priya_forecast.models.gp_model import GPModel
    gp = GPModel()
    z = args.z
    k_eboss, _, cov = load_eboss(z=z)
    fid = np.array(fiducial_vector(), dtype=float)

    print(f"Building Sobol training set ({args.n_train} samples × {k_eboss.size} k-bins)...")
    X, y = _build_training(
        gp=gp, varying_names=args.params, n=args.n_train,
        k_grid=k_eboss, z=z, seed=args.seed,
    )
    print(f"X={X.shape}, y={y.shape}")

    print(f"Training PySR (niter={args.niter}, maxsize={args.maxsize})...")
    from pysr import PySRRegressor

    model = PySRRegressor(
        niterations=args.niter,
        maxsize=args.maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "square"],
        elementwise_loss="loss(prediction, target) = (prediction - target)^2",
        random_state=args.seed,
        deterministic=True,
        parallelism="serial",
        verbosity=0,
    )
    model.fit(X, y)
    pareto = model.equations_
    print("\nPareto front (top 5):")
    print(pareto[["complexity", "loss", "equation"]].tail(5).to_string())

    best_idx = int(pareto["loss"].idxmin())
    best_eq = pareto.iloc[best_idx]
    print(f"\nBest equation (complexity={best_eq['complexity']}, loss={best_eq['loss']:.3g}):")
    print(f"  {best_eq['equation']}")

    # Save the Pareto CSV for posterity.
    pareto[["complexity", "loss", "equation"]].to_csv(args.output / "hall_of_fame.csv", index=False)

    # Build the equation expression in the form the framework expects.
    # PySR returns equation strings using x0, x1, ..., xN where N = len(params)
    # (last one is k_norm). We rewrite to physical-units form.
    expr_str = str(best_eq["equation"])
    # Substitute x_i → ((param_i) - lo_i)/(hi_i - lo_i)
    # and last x_N → ((k) - k_min)/(k_max - k_min).
    k_min, k_max = float(k_eboss.min()), float(k_eboss.max())
    pieces = []
    for i, name in enumerate(args.params):
        lo, hi = get_param(name).prior
        pieces.append((f"x{i}", f"(({name}) - ({lo}))/({hi - lo})"))
    pieces.append((f"x{len(args.params)}", f"((k) - ({k_min}))/({k_max - k_min})"))
    # Substitute right-to-left to avoid x10 → x1 0 issues.
    pieces.sort(key=lambda p: -int(p[0][1:]))
    physical_expr = expr_str
    for old, new in pieces:
        physical_expr = physical_expr.replace(old, f"({new})")

    print(f"\nExpression (physical-units form):\n  {physical_expr}\n")

    # Build a PySRModel with combine='joint'.
    cfg = EqnConfig(
        name=f"multid_pysr_{len(args.params)}D",
        redshift=z, model="pysr", combine="joint",
        joint_expression=physical_expr, parameters={},
    )
    pysr_model = PySRModel(eqn_cfg=cfg, k_grid=k_eboss,
                           normalization_block={"mode": "identity"})

    # Build references and score.
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from train_and_forecast import (
        build_perfect_1d_reference, build_gp_reference, _scorecard,
    )
    pysr_perfect, cfg_perfect = build_perfect_1d_reference(
        gp=gp, fid=fid, k_grid=k_eboss, z=z, varying_names=args.params,
    )
    pysr_gp, cfg_gp = build_gp_reference(gp=gp, fid=fid, k_grid=k_eboss, z=z)

    forecast_params = tuple(p for p in PARAMS_11D if p.name in args.params)
    sets = [
        EqSetEntry(name="GP_reference",      model=pysr_gp,      eqn_cfg=cfg_gp),
        EqSetEntry(name="perfect_1D_slices", model=pysr_perfect, eqn_cfg=cfg_perfect),
        EqSetEntry(name=cfg.name,            model=pysr_model,   eqn_cfg=cfg),
    ]
    out = compare_equation_sets(
        gp_model=gp, pysr_sets=sets, z=z, k_eboss=k_eboss, cov_eboss=cov,
        forecast_params=forecast_params, outdir=args.output,
    )
    print(f"\nFigures + scorecard at {out}")

    from priya_forecast.diagnostics.compare import _fisher_for as fisher_for
    fr_gp = fisher_for(model=gp, fid=fid, k=k_eboss, z=z, params=forecast_params)
    fr_perfect = fisher_for(model=pysr_perfect, fid=fid, k=k_eboss, z=z, params=forecast_params)
    fr_student = fisher_for(model=pysr_model, fid=fid, k=k_eboss, z=z, params=forecast_params)

    # --- §5.1 rank diagnostic: joint-PySR Jacobian/Fisher rank vs the GP. ---
    # `FisherResult.F = Y^T Y` (Y = whitened dP_F/dtheta), so its eigen-spectrum
    # is the parameter-Jacobian singular spectrum. Dump spectrum + condition
    # number + numerical rank (tol 1e-6/1e-8/1e-10) for the joint fit and the GP.
    rank_diag = _write_joint_rank_json(
        path=out / "joint_rank_diagnostic.json",
        params=forecast_params, fr_student=fr_student, fr_gp=fr_gp,
        equation=best_eq["equation"], loss=best_eq["loss"],
        complexity=best_eq["complexity"], z=z,
    )
    jp = rank_diag["joint_pysr"]["whitened"]
    gpw = rank_diag["gp_reference"]["whitened"]
    print("\n=== joint-fit rank diagnostic (whitened Fisher/Jacobian) ===")
    print(f"  n_params            : {rank_diag['n_params']}")
    print(f"  joint loss          : {rank_diag['joint_loss']:.3g}  "
          f"(complexity {rank_diag['joint_complexity']})")
    print(f"  joint-PySR numerical rank (tol 1e-6/1e-8/1e-10): "
          f"{jp['numerical_rank']['1e-06']}/{jp['numerical_rank']['1e-08']}/"
          f"{jp['numerical_rank']['1e-10']}  of {rank_diag['n_params']}")
    print(f"  joint-PySR condition number : {jp['condition_number']:.3e}")
    print(f"  GP-ref    numerical rank (tol 1e-6/1e-8/1e-10): "
          f"{gpw['numerical_rank']['1e-06']}/{gpw['numerical_rank']['1e-08']}/"
          f"{gpw['numerical_rank']['1e-10']}  of {rank_diag['n_params']}")
    print(f"  GP-ref    condition number : {gpw['condition_number']:.3e}")
    print(f"  rank-deficient vs n_params : "
          f"{rank_diag['joint_pysr']['rank_deficient_vs_nparams']}")
    print(f"  written -> {out / 'joint_rank_diagnostic.json'}")

    md = (out / "summary.md").read_text()
    full = _scorecard(
        summary_md=md, fr_perfect=fr_perfect, fr_gp=fr_gp,
        fr_student=fr_student, label=cfg.name,
        k=k_eboss, fid=fid, gp=gp,
        perfect_model=pysr_perfect, student_model=pysr_model, z=z,
    )
    (out / "scorecard.md").write_text(full)
    print("\n=== scorecard ===")
    print(full)


if __name__ == "__main__":
    main()
