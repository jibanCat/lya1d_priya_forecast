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
import re
import sys
from pathlib import Path

import numpy as np

# Why the maxsize sweep has no Sobolev arm. The Sobolev derivative-matching loss
# (priya_forecast.sobolev_loss.make_sobolev_loss) is well-defined ONLY for a
# single-input (per-parameter 1D) fit: inside the Julia loss it perturbs exactly
# one feature row (`X[1, :] .+= h`) and matches that one finite-difference
# gradient against the ONE per-point target carried by PySR's `dataset.weights`
# channel. A joint 6-input equation would need to match dP_F/dtheta_i for all six
# params x0..x5 at once -> six independent target-gradient channels and six
# perturbation directions. PySR exposes a single per-point `weights` vector only,
# so the six targets cannot be delivered and the loss hardcodes perturbing just
# the first feature. Generalizing would require packing 6 target-gradient arrays
# into extra X columns and rewriting the loss to perturb each of the 6
# theta-feature rows against those columns -- a redesign absent from the current
# machinery. Joint-Sobolev is therefore NOT well-defined here; MSE only.
SOBOLEV_ARM_NOTE = {
    "feasible": False,
    "reason": (
        "Sobolev loss perturbs one feature row (X[1,:]+=h) and matches the single "
        "`dataset.weights` channel = one param's target gradient. A joint 6-input eq "
        "needs dP/dtheta_i for all 6 params (6 gradient channels + 6 perturb dirs); "
        "PySR has only one weights vector, so joint-Sobolev is not well-defined with "
        "the current machinery. This sweep uses MSE only."
    ),
}

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


def _to_physical_expr(expr_str, param_names, k_min, k_max):
    """Rewrite a PySR ``x0..xN`` equation into physical-units form: each ``x_i``
    -> the normalized parameter ``(param_i - lo_i)/(hi_i - lo_i)`` and the final
    ``x_N`` -> normalized k ``(k - k_min)/(k_max - k_min)``. Substituted
    high-index-first so ``x10`` is never mangled into ``x1``."""
    pieces = []
    for i, name in enumerate(param_names):
        lo, hi = get_param(name).prior
        pieces.append((f"x{i}", f"(({name}) - ({lo}))/({hi - lo})"))
    pieces.append((f"x{len(param_names)}", f"((k) - ({k_min}))/({k_max - k_min})"))
    pieces.sort(key=lambda p: -int(p[0][1:]))
    out = str(expr_str)
    for old, new in pieces:
        out = out.replace(old, f"({new})")
    return out


def _inputs_present(expr_str, param_names):
    """Which of the len(param_names) parameter inputs (x0..x_{n-1}) actually
    appear in `expr_str` (the last feature x_n is k, ignored here). Returns
    (n_present, absent_names)."""
    used = {int(m) for m in re.findall(r"x(\d+)", str(expr_str))}
    present = [i for i in range(len(param_names)) if i in used]
    absent = [param_names[i] for i in range(len(param_names)) if i not in used]
    return len(present), absent


def _front_rank_scan(*, pareto, param_names, k_grid, k_min, k_max, fid, z,
                     forecast_params, fisher_for, min_complexity=10):
    """Reconstruct EVERY Pareto-front equation with complexity >= min_complexity,
    build its joint PySRModel, compute the whitened Fisher at fid, and return the
    MAX whitened numerical rank (tol 1e-8) reached by ANY front equation plus the
    complexity at which it is reached. The idxmin (loss-min) rank alone can hide a
    higher-rank equation sitting elsewhere on the front — this is the
    discriminator between structural collapse and a loss-surface accident."""
    widths = np.array([p.width() for p in forecast_params], dtype=float)
    W = np.outer(widths, widths)
    rows = []
    best_rank, best_rank_cx = -1, None
    for _, r in pareto.iterrows():
        cx = int(r["complexity"])
        if cx < min_complexity:
            continue
        try:
            phys = _to_physical_expr(str(r["equation"]), param_names, k_min, k_max)
            cfg = EqnConfig(name=f"front_c{cx}", redshift=z, model="pysr",
                            combine="joint", joint_expression=phys, parameters={})
            m = PySRModel(eqn_cfg=cfg, k_grid=k_grid,
                          normalization_block={"mode": "identity"})
            fr = fisher_for(model=m, fid=fid, k=k_grid, z=z, params=forecast_params)
            Fw = fr.F * W
            eig = np.sort(np.linalg.eigvalsh(0.5 * (Fw + Fw.T)))[::-1]
            lam_max = float(eig[0]) if eig.size else 0.0
            rank = int(np.sum(eig > 1e-8 * lam_max)) if lam_max > 0 else 0
            rows.append({"complexity": cx, "loss": float(r["loss"]), "rank_1e8": rank})
            if rank > best_rank:
                best_rank, best_rank_cx = rank, cx
        except Exception as e:  # a front eq may fail to compile / give a NaN Fisher
            rows.append({"complexity": cx, "loss": float(r["loss"]),
                         "rank_1e8": None, "error": str(e)[:160]})
    return {
        "front_max_rank_1e8": (best_rank if best_rank >= 0 else None),
        "front_max_rank_complexity": best_rank_cx,
        "min_complexity_scanned": int(min_complexity),
        "n_front_scanned": int(sum(1 for x in rows if x.get("rank_1e8") is not None)),
        "per_row": rows,
    }


def _offfid_thetas(*, forecast_params, fid, n, seed=0, frac=0.3):
    """n random theta vectors: forecast params jittered uniformly by +/-frac*width
    about fid (clamped strictly inside their priors so the GP stays in range);
    all other params fixed at fid."""
    rng = np.random.default_rng(seed)
    idxs = [PARAM_NAMES.index(p.name) for p in forecast_params]
    out = []
    for _ in range(n):
        theta = np.array(fid, dtype=float).copy()
        for p, gi in zip(forecast_params, idxs):
            lo, hi = p.prior
            w = hi - lo
            val = fid[gi] + rng.uniform(-frac, frac) * w
            theta[gi] = min(max(val, lo + 1e-6 * w), hi - 1e-6 * w)
        out.append(theta)
    return out


def _accuracy_mses(*, gp, student_model, perfect_model, k, z, thetas, sigma_eboss):
    """Mean residual MSE in eBOSS-sigma^2 units vs the GP truth over `thetas`, for
    the student joint eq and the perfect-1D reference."""
    s_mses, p_mses = [], []
    for theta in thetas:
        try:
            p_truth = gp.predict(theta, k, z)
            p_stu = student_model.predict(theta, k, z)
            p_prf = perfect_model.predict(theta, k, z)
            s_mses.append(float(np.mean(((p_stu - p_truth) / sigma_eboss) ** 2)))
            p_mses.append(float(np.mean(((p_prf - p_truth) / sigma_eboss) ** 2)))
        except Exception:
            continue
    _m = lambda v: (float(np.mean(v)) if v else float("nan"))
    return {"student_vs_gp": _m(s_mses), "perfect1d_vs_gp": _m(p_mses),
            "n_points": len(s_mses)}


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
                           equation, loss, complexity, z,
                           maxsize=None, front_scan=None, n_inputs_present=None,
                           absent_inputs=None, accuracy_insample=None,
                           accuracy_offfid=None, sobolev_arm=None):
    """Dump the §5.1 rank diagnostic to `path` as JSON.

    Base fields (loss-min / idxmin equation): equation, loss, complexity, the
    whitened+physical Fisher/Jacobian eigen-spectrum, condition number, and
    numerical rank (tol 1e-6/1e-8/1e-10) for the joint fit and the GP reference.

    Maxsize-sweep upgrade (all optional; populated by the sweep runner):
    - front_scan: MAX whitened rank over the WHOLE Pareto front (the discriminator).
    - n_inputs_present / absent_inputs: how many of the params the loss-min eq uses.
    - pinned_at_cap: best_loss_complexity == maxsize (front still budget-saturated).
    - accuracy_insample / accuracy_offfid: student-vs-GP & perfect1D-vs-GP residual MSE.
    - sobolev_arm: feasibility note for the (not-run) joint-Sobolev arm.
    A flat `preregistered` block mirrors the fields the pass/fail criteria read."""
    widths = np.array([p.width() for p in params], dtype=float)
    n = len(params)
    sig = lambda fr: [None if not np.isfinite(s) else float(s) for s in fr.sigma]
    diag = {
        "params": [p.name for p in params],
        "n_params": n,
        "z": float(z),
        "maxsize": (int(maxsize) if maxsize is not None else None),
        "joint_equation": str(equation),
        "joint_loss": float(loss),
        "joint_complexity": int(complexity),
        "joint_pysr": {**_spectrum(fr_student.F, widths), "sigma": sig(fr_student)},
        "gp_reference": {**_spectrum(fr_gp.F, widths), "sigma": sig(fr_gp)},
    }
    rk8 = diag["joint_pysr"]["whitened"]["numerical_rank"]["1e-08"]
    diag["joint_pysr"]["rank_deficient_vs_nparams"] = bool(rk8 < n)
    if front_scan is not None:
        diag["front_scan"] = front_scan
    if n_inputs_present is not None:
        diag["n_inputs_present"] = int(n_inputs_present)
        diag["absent_inputs"] = list(absent_inputs or [])
    if accuracy_insample is not None:
        diag["accuracy_insample"] = accuracy_insample
    if accuracy_offfid is not None:
        diag["accuracy_offfid"] = accuracy_offfid
    if sobolev_arm is not None:
        diag["sobolev_arm"] = sobolev_arm

    # Flat pre-registered block: every field the STRUCTURAL/BUDGET/INCONCLUSIVE
    # decision reads, so the cross-maxsize analysis is mechanical.
    pinned = (bool(int(complexity) == int(maxsize)) if maxsize is not None else None)
    diag["preregistered"] = {
        "maxsize": (int(maxsize) if maxsize is not None else None),
        "n_params": n,
        "best_loss_complexity": int(complexity),
        "pinned_at_cap": pinned,
        "n_inputs_present": (int(n_inputs_present) if n_inputs_present is not None else None),
        "absent_inputs": list(absent_inputs or []),
        "idxmin_rank_whitened_1e8": rk8,
        "front_max_rank_1e8": (front_scan or {}).get("front_max_rank_1e8"),
        "front_max_rank_complexity": (front_scan or {}).get("front_max_rank_complexity"),
        "joint_condition_number_whitened": diag["joint_pysr"]["whitened"]["condition_number"],
        "gp_condition_number_whitened": diag["gp_reference"]["whitened"]["condition_number"],
        "offfid_student_vs_gp": (accuracy_offfid or {}).get("student_vs_gp"),
        "offfid_perfect1d_vs_gp": (accuracy_offfid or {}).get("perfect1d_vs_gp"),
        "insample_student_vs_gp": (accuracy_insample or {}).get("student_vs_gp"),
        "insample_perfect1d_vs_gp": (accuracy_insample or {}).get("perfect1d_vs_gp"),
    }

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
    k_min, k_max = float(k_eboss.min()), float(k_eboss.max())
    physical_expr = _to_physical_expr(expr_str, args.params, k_min, k_max)

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
    #
    # Maxsize-sweep upgrade: (1) MAX whitened rank over the WHOLE Pareto front,
    # (2) which of the params the loss-min eq uses + pinned-at-cap, (3) in-sample
    # and off-fid residual MSE vs the GP (student & perfect-1D), so the
    # STRUCTURAL-vs-BUDGET-ARTIFACT question is answered mechanically.
    print("\nScanning Pareto front for max Fisher rank (complexity >= 10)...")
    front_scan = _front_rank_scan(
        pareto=pareto, param_names=args.params, k_grid=k_eboss,
        k_min=k_min, k_max=k_max, fid=fid, z=z,
        forecast_params=forecast_params, fisher_for=fisher_for, min_complexity=10,
    )
    n_present, absent = _inputs_present(best_eq["equation"], args.params)
    sigma_eboss = np.sqrt(np.diag(cov))
    insample_thetas = _sobol_design(varying_names=args.params, n=args.n_train, seed=args.seed)
    offfid_thetas = _offfid_thetas(forecast_params=forecast_params, fid=fid,
                                   n=32, seed=0, frac=0.3)
    acc_in = _accuracy_mses(gp=gp, student_model=pysr_model, perfect_model=pysr_perfect,
                            k=k_eboss, z=z, thetas=list(insample_thetas), sigma_eboss=sigma_eboss)
    acc_off = _accuracy_mses(gp=gp, student_model=pysr_model, perfect_model=pysr_perfect,
                             k=k_eboss, z=z, thetas=offfid_thetas, sigma_eboss=sigma_eboss)

    rank_diag = _write_joint_rank_json(
        path=out / "joint_rank_diagnostic.json",
        params=forecast_params, fr_student=fr_student, fr_gp=fr_gp,
        equation=best_eq["equation"], loss=best_eq["loss"],
        complexity=best_eq["complexity"], z=z, maxsize=args.maxsize,
        front_scan=front_scan, n_inputs_present=n_present, absent_inputs=absent,
        accuracy_insample=acc_in, accuracy_offfid=acc_off,
        sobolev_arm=SOBOLEV_ARM_NOTE,
    )
    jp = rank_diag["joint_pysr"]["whitened"]
    gpw = rank_diag["gp_reference"]["whitened"]
    pre = rank_diag["preregistered"]
    print("\n=== joint-fit rank diagnostic (whitened Fisher/Jacobian) ===")
    print(f"  n_params            : {rank_diag['n_params']}   maxsize={args.maxsize}")
    print(f"  joint loss          : {rank_diag['joint_loss']:.3g}  "
          f"(complexity {rank_diag['joint_complexity']}, "
          f"pinned_at_cap={pre['pinned_at_cap']})")
    print(f"  n_inputs_present    : {pre['n_inputs_present']}/{rank_diag['n_params']}  "
          f"(absent: {pre['absent_inputs']})")
    print(f"  joint-PySR idxmin rank (1e-6/1e-8/1e-10): "
          f"{jp['numerical_rank']['1e-06']}/{jp['numerical_rank']['1e-08']}/"
          f"{jp['numerical_rank']['1e-10']}  of {rank_diag['n_params']}")
    print(f"  FRONT max rank (1e-8): {pre['front_max_rank_1e8']} of {rank_diag['n_params']}  "
          f"(at complexity {pre['front_max_rank_complexity']}, "
          f"{front_scan['n_front_scanned']} front eqs scanned)")
    print(f"  joint-PySR condition number : {jp['condition_number']:.3e}   "
          f"GP condition number : {gpw['condition_number']:.3e}")
    print(f"  off-fid MSE  student/GP={pre['offfid_student_vs_gp']:.3g}  "
          f"perfect1D/GP={pre['offfid_perfect1d_vs_gp']:.3g}")
    print(f"  in-sample MSE student/GP={pre['insample_student_vs_gp']:.3g}  "
          f"perfect1D/GP={pre['insample_perfect1d_vs_gp']:.3g}")
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
