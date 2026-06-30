"""Consolidated multi-D (2D + 3D) prediction-error diagnostic.

This is the paper's headline multi-parameter figure + Table-3 backing CSV. It
answers a single question: when we take the per-z **1D** PySR equations (one
per PRIYA parameter) and assemble a multi-parameter P1D prediction with the
additive-1st-order-Taylor combine, how well does that prediction track the GP
"truth" once *several* parameters move at once?

The 1D equations were each trained with one parameter varied and the rest at
fiducial, so a multi-parameter test probes the cross-coupling regime the 1D
fits never saw. We sweep a BROAD set of 2D pairs and 3D triples (Sobol points
across each combo's joint prior sub-cube), compare the combine against the GP
over the kodiaq k-grid, and report per-combo mean / 90th-percentile / max
relative error. The best- and worst-case combos per dimensionality become the
figure; every combo lands in the CSV.

How the machinery is reused (nothing is reinvented here):
  * Per-param equation reload + normalization:
        priya_forecast.single_z.forecast.build_refit_from_pareto_gated
    — the SAME derivative-gated loader the Fisher forecast uses, so a param
    whose equation fails the derivative gate falls back to its GP slice
    exactly as in the paper's forecast (transparent: logged + flagged in CSV).
  * Additive-Taylor combine:
        priya_forecast.single_z.combine.build_combined_model (mode="additive")
    — assembles P_F(theta,k) = P_GP(fid,k) + Sum_i [eq_i(theta_i) - eq_i(fid_i)].
    Because non-varied params sit at fid, their deviation is zero, so ONE
    combine built with all 11 refits serves every combo.
  * GP truth: priya_forecast.models.gp_model.GPModel(fidelity="hf").

Run (validation against the existing single-z stage-9 refit dir, z=3.6):
  export PYTHON_JULIAPKG_PROJECT=$HOME/.julia_env JULIA_DEPOT_PATH=$HOME/.julia
  export PYTHONPATH=src:/home/mfho/student_projects/lya_emulator_full
  .venv/bin/python scripts/regen_multid.py \\
      --refit-dir results/single_z_stage9 --z 3.6 \\
      --basedir data/kodiaq_gp --out-dir <out> --n-sobol 256

Production (per-z Sobolev refits, run once per z bin):
  .venv/bin/python scripts/regen_multid.py \\
      --refit-dir results/paper_production_20260630_perz_sobolev_z2.6-4.2/sobolev \\
      --z 3.6 --basedir data/kodiaq_gp \\
      --out-dir results/paper_production_20260630_perz_sobolev_z2.6-4.2/sobolev/multid_z3.6 \\
      --n-sobol 256
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

# --- Environment defaults (Julia/PySR + the upstream emulator import path) ---
os.environ.setdefault("PYTHON_JULIAPKG_PROJECT", str(Path.home() / ".julia_env"))
os.environ.setdefault("JULIA_DEPOT_PATH", str(Path.home() / ".julia"))

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists() and str(_LYAEMU) not in sys.path:
    sys.path.insert(0, str(_LYAEMU))
_SRC = Path(__file__).resolve().parents[1] / "src"
if _SRC.exists() and str(_SRC) not in sys.path:
    sys.path.insert(0, str(_SRC))

from priya_forecast.derivative_gate import gp_param_gradient
from priya_forecast.models.gp_model import GPModel
from priya_forecast.parameters import (
    PARAM_NAMES,
    fiducial_vector,
    get_param,
)
from priya_forecast.single_z.combine import build_combined_model
from priya_forecast.single_z.forecast import build_refit_from_pareto_gated
from priya_forecast.single_z.refit import kodiaq_k_grid


# Default broad sweep. "-"-joined param names. Edit via --pairs / --triples.
# Includes the three the paper calls out explicitly: dtau0-Ap, dtau0-Ap-ns,
# and the science-relevant thermal pair herei-alphaq.
# NOTE: dtau0 (mean-flux slope) and tau0 (mean-flux factor) are DEGENERATE in
# the GP — upstream collapses them into one effective mean-flux value bounded to
# the dense cube [~0.66, ~1.36]. Varying both at once is not a coupling test (it
# is the same GP direction) and large parts of that joint cube fall outside the
# emulator, so the defaults never co-vary the two. Any residual out-of-domain
# Sobol point (e.g. tau0 near its edge) is dropped by the per-sample guard in
# `_eval_combo`.
DEFAULT_PAIRS = [
    "dtau0-Ap",        # mean-flux slope x amplitude (paper headline pair)
    "herei-alphaq",    # HeII reionization onset x quasar spectral slope (thermal)
    "Ap-ns",           # amplitude x tilt (cosmology)
    "tau0-ns",         # mean-flux factor x tilt
    "herei-heref",     # HeII onset x completion (thermal pair)
    "tau0-Ap",         # mean-flux factor x amplitude
]
DEFAULT_TRIPLES = [
    "dtau0-Ap-ns",            # paper headline triple
    "herei-heref-alphaq",     # full thermal trio
    "tau0-Ap-ns",             # mean-flux factor + amplitude + tilt
    "Ap-ns-omegamh2",         # cosmology trio
]


def _resolve_refit_z_dir(refit_dir: Path, z: float) -> Path:
    """Find the `refit/z<z>/` sub-directory holding pareto_<param>.csv.

    Tries `z{z}` and `z{z:.1f}` first, then globs `refit/z*` and picks the
    z-tag numerically closest to the requested z (within 1e-3).
    """
    base = refit_dir / "refit"
    for tag in (f"z{z}", f"z{z:.1f}", f"z{z:g}"):
        cand = base / tag
        if cand.is_dir():
            return cand
    # Fall back to a numeric match over all z* dirs.
    best, best_d = None, np.inf
    for cand in sorted(base.glob("z*")):
        if not cand.is_dir():
            continue
        try:
            zval = float(cand.name[1:])
        except ValueError:
            continue
        d = abs(zval - z)
        if d < best_d:
            best, best_d = cand, d
    if best is not None and best_d < 1e-3:
        return best
    raise FileNotFoundError(
        f"No refit z-dir for z={z} under {base} "
        f"(looked for z{z}/z{z:.1f}; closest was {best} at delta={best_d:g})."
    )


def _load_refits(
    *, refit_z_dir: Path, z: float, gp_hf, fid: np.ndarray,
    k_grid: np.ndarray, data_1pvar_dir: Path, derivative_tol: float,
    log_space: bool,
) -> tuple[dict, list[str]]:
    """Reconstruct one derivative-gated Refit1DResult per parameter.

    Returns (refits, gp_sliced) where `refits[name]` is a Refit1DResult or
    None (GP-slice fallback), and `gp_sliced` lists the params that fell back
    (missing CSV, no Fisher-safe equation, or failed the derivative gate).
    """
    refits: dict = {name: None for name in PARAM_NAMES}
    gp_sliced: list[str] = []
    for param in PARAM_NAMES:
        csv_path = refit_z_dir / f"pareto_{param}.csv"
        if not csv_path.exists():
            gp_sliced.append(param)
            print(f"  [refit] {param}: no pareto CSV -> GP-slice fallback.")
            continue
        try:
            tgt = gp_param_gradient(
                gp=gp_hf, fid=fid, k_grid=k_grid, z=z,
                param_idx=PARAM_NAMES.index(param),
                log_space=log_space,
            )
            refits[param] = build_refit_from_pareto_gated(
                param_name=param, z=z, pareto_csv=csv_path,
                data_1pvar_dir=str(data_1pvar_dir),
                gp_target_grad=tgt, derivative_tol=derivative_tol,
                log_space=log_space,
            )
            print(f"  [refit] {param}: loaded "
                  f"(complexity={refits[param].pareto_complexity}, "
                  f"loss={refits[param].pareto_loss:.4g}).")
        except (ValueError, FileNotFoundError) as exc:
            gp_sliced.append(param)
            print(f"  [refit] {param}: {exc} -> GP-slice fallback.")
    return refits, gp_sliced


def _parse_combos(spec: list[str]) -> list[tuple[str, ...]]:
    """Turn ["dtau0-Ap", ...] into [("dtau0", "Ap"), ...] with validation."""
    out: list[tuple[str, ...]] = []
    for s in spec:
        names = tuple(p for p in s.split("-") if p)
        for n in names:
            if n not in PARAM_NAMES:
                raise SystemExit(
                    f"Unknown parameter {n!r} in combo {s!r}. "
                    f"Known: {', '.join(PARAM_NAMES)}"
                )
        if len(names) < 2:
            raise SystemExit(f"Combo {s!r} must name >=2 parameters.")
        out.append(names)
    return out


# The PRIYA priors in parameters.py are exactly the GP emulator's training
# hypercube (`_param_limits`). A Sobol point landing on the prior edge can
# overshoot that bound by ~machine-epsilon, and the upstream
# `map_to_unit_cube` assertion only tolerates 1e-16 — so sample the OPEN
# interior, inset by a negligible fraction of each prior range.
_EDGE_MARGIN_FRAC = 1e-9


def _sobol_samples(combo: tuple[str, ...], fid: np.ndarray, n: int, seed: int):
    """Sobol points over the combo's joint prior sub-cube (others at fid)."""
    from scipy.stats import qmc

    sampler = qmc.Sobol(d=len(combo), seed=seed)
    u = sampler.random(n=n)
    theta = np.tile(np.asarray(fid, dtype=float), (n, 1))
    for col, name in enumerate(combo):
        lo, hi = get_param(name).prior
        span = hi - lo
        vals = lo + u[:, col] * span
        m = _EDGE_MARGIN_FRAC * span
        theta[:, PARAM_NAMES.index(name)] = np.clip(vals, lo + m, hi - m)
    return theta


def _eval_combo(*, combo, fid, combine, gp_hf, k_grid, z, n_sobol, seed):
    """Sobol-sweep one combo; return (rel-err array, n_skipped).

    A Sobol point can land outside the GP's training cube (e.g. the
    degenerate mean-flux direction, or a param at its prior edge). The
    upstream emulator raises on those; we skip them per-sample so a few
    edge points never sink a whole combo, and report the count.
    """
    thetas = _sobol_samples(combo, fid, n_sobol, seed)
    rel_rows = []
    n_skip = 0
    for i in range(n_sobol):
        try:
            truth = np.asarray(gp_hf.predict(thetas[i], k_grid, z), dtype=float)
            pred = np.asarray(combine.predict(thetas[i], k_grid, z), dtype=float)
        except (AssertionError, ValueError):
            n_skip += 1
            continue
        rel_rows.append(np.abs(pred - truth) / np.abs(truth))
    return np.asarray(rel_rows, dtype=float), n_skip


def _summarize(rel: np.ndarray, k_grid: np.ndarray) -> dict:
    """Per-combo scalars (in %) + the per-k mean curve for plotting."""
    flat = rel.ravel()
    per_k_mean = rel.mean(axis=0)
    per_k_max = rel.max(axis=0)
    return {
        "mean_pct": 100.0 * float(flat.mean()),
        "p90_pct": 100.0 * float(np.percentile(flat, 90)),
        "max_pct": 100.0 * float(flat.max()),
        "max_k": float(k_grid[int(np.argmax(per_k_max))]),
        "per_k_mean_pct": 100.0 * per_k_mean,
        "per_k_p90_pct": 100.0 * np.percentile(rel, 90, axis=0),
    }


def _write_csv(out_csv: Path, rows: list[dict]) -> None:
    cols = ["dim", "params", "n_varied", "mean_pct", "p90_pct", "max_pct",
            "max_k", "gp_sliced", "role"]
    with open(out_csv, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r[c] for c in cols})


def _make_figure(out_base: Path, rows: list[dict], summaries: dict,
                 k_grid: np.ndarray, *, z: float, n_sobol: int) -> None:
    """Two-panel print figure: rel-err vs k (best/worst per dim) + summary bars."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plt.rcParams.update({
        "font.size": 15, "axes.titlesize": 17, "axes.labelsize": 16,
        "xtick.labelsize": 13, "ytick.labelsize": 13, "legend.fontsize": 12.5,
    })

    # Pick best (lowest mean) and worst (highest mean) per dimensionality.
    picks = {}  # (dim, role) -> row
    for dim in ("2D", "3D"):
        sub = [r for r in rows if r["dim"] == dim]
        if not sub:
            continue
        sub_sorted = sorted(sub, key=lambda r: r["mean_pct"])
        picks[(dim, "best")] = sub_sorted[0]
        picks[(dim, "worst")] = sub_sorted[-1]

    fig, (axL, axR) = plt.subplots(
        1, 2, figsize=(16, 7), gridspec_kw={"width_ratios": [1.15, 1.0]},
    )

    # --- Left: rel-err(%) vs k for best & worst per dim ---
    style = {
        ("2D", "best"): dict(color="#1f77b4", ls="-", marker="o"),
        ("2D", "worst"): dict(color="#1f77b4", ls="--", marker="s"),
        ("3D", "best"): dict(color="#d62728", ls="-", marker="o"),
        ("3D", "worst"): dict(color="#d62728", ls="--", marker="s"),
    }
    for (dim, role), row in picks.items():
        s = summaries[row["params"]]
        axL.plot(
            k_grid, s["per_k_mean_pct"], lw=2.4, ms=5, markevery=4,
            label=f"{dim} {role}: {row['params']}  (mean {row['mean_pct']:.1f}%)",
            **style[(dim, role)],
        )
    axL.set_xscale("log")
    axL.set_yscale("log")
    axL.set_xlabel("k  [s/km]")
    axL.set_ylabel("mean relative error  [%]")
    axL.set_title(f"Multi-D combine vs GP   (z={z}, n_Sobol={n_sobol})")
    axL.axhline(1.0, ls=":", color="0.4", lw=1.0)
    axL.text(k_grid[0], 1.05, "1%", color="0.4", fontsize=11, va="bottom")
    axL.grid(alpha=0.3, which="both")
    axL.legend(loc="upper left", frameon=False)

    # --- Right: horizontal summary bars (mean), with p90 + max markers ---
    rows_sorted = sorted(rows, key=lambda r: (r["dim"], r["mean_pct"]))
    labels = [f"{r['params']}" for r in rows_sorted]
    means = [r["mean_pct"] for r in rows_sorted]
    p90s = [r["p90_pct"] for r in rows_sorted]
    maxs = [r["max_pct"] for r in rows_sorted]
    colors = ["#1f77b4" if r["dim"] == "2D" else "#d62728" for r in rows_sorted]
    y = np.arange(len(rows_sorted))
    axR.barh(y, means, color=colors, alpha=0.85, label="mean")
    axR.scatter(p90s, y, color="0.15", marker="|", s=260, lw=2.4,
                label="p90", zorder=5)
    axR.scatter(maxs, y, color="0.15", marker="x", s=70, lw=2.0,
                label="max", zorder=5)
    axR.set_yticks(y)
    axR.set_yticklabels(labels, fontsize=12)
    axR.invert_yaxis()
    axR.set_xscale("log")
    axR.set_xlabel("relative error  [%]")
    axR.set_title("Per-combo error (mean / p90 / max)")
    axR.grid(alpha=0.3, axis="x", which="both")
    # Legend entries: dim colors + marker meanings.
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    handles = [
        Patch(facecolor="#1f77b4", alpha=0.85, label="2D pair"),
        Patch(facecolor="#d62728", alpha=0.85, label="3D triple"),
        Line2D([0], [0], color="0.15", marker="|", lw=0, ms=14, label="p90"),
        Line2D([0], [0], color="0.15", marker="x", lw=0, ms=9, label="max"),
    ]
    axR.legend(handles=handles, loc="lower right", frameon=False)

    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(out_base.with_suffix(f".{ext}"), dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    ap.add_argument(
        "--refit-dir", type=Path, default=Path("results/single_z_stage9"),
        help="Dir containing refit/z<z>/pareto_<param>.csv (default: stage9).",
    )
    ap.add_argument("--z", type=float, default=3.6)
    ap.add_argument("--basedir", type=Path, default=Path("data/kodiaq_gp"),
                    help="GP emulator basedir.")
    ap.add_argument("--hires-subdir", type=str, default="hires")
    ap.add_argument("--data-1pvar-dir", type=Path,
                    default=Path("data/single_z_1pvar"),
                    help="Regenerated 1pvar HDF5 dir (per-param normalization).")
    ap.add_argument("--out-dir", type=Path, required=True)
    ap.add_argument("--n-sobol", type=int, default=256,
                    help="Sobol points per combo (use a power of 2).")
    ap.add_argument("--seed", type=int, default=1234, help="Sobol seed.")
    ap.add_argument("--k-min", type=float, default=0.001)
    ap.add_argument("--k-max", type=float, default=0.04)
    ap.add_argument("--n-k", type=int, default=48)
    ap.add_argument("--derivative-tol", type=float, default=0.25,
                    help="Derivative-gate tolerance (matches the forecast).")
    ap.add_argument("--target-space", choices=("linear", "log"), default="log",
                    help="Equation training space; 'log' for Sobolev refits "
                         "(stage9 + the sobolev production both use 'log').")
    ap.add_argument("--pairs", nargs="*", default=DEFAULT_PAIRS,
                    help="2D pairs as 'a-b' tokens.")
    ap.add_argument("--triples", nargs="*", default=DEFAULT_TRIPLES,
                    help="3D triples as 'a-b-c' tokens.")
    args = ap.parse_args()

    args.out_dir.mkdir(parents=True, exist_ok=True)
    log_space = args.target_space == "log"
    k_grid = kodiaq_k_grid(args.k_min, args.k_max, args.n_k)
    fid = np.asarray(fiducial_vector(), dtype=float)

    pairs = _parse_combos(args.pairs)
    triples = _parse_combos(args.triples)
    combos = [("2D", c) for c in pairs] + [("3D", c) for c in triples]
    print(f"Multi-D best/worst diagnostic: z={args.z}, n_sobol={args.n_sobol}, "
          f"k=[{args.k_min}, {args.k_max}] s/km ({args.n_k} bins), "
          f"target_space={args.target_space}")
    print(f"  {len(pairs)} pairs + {len(triples)} triples = {len(combos)} combos")

    refit_z_dir = _resolve_refit_z_dir(args.refit_dir, args.z)
    print(f"Refit CSVs: {refit_z_dir}")

    print("Loading HF GP emulator...")
    gp_hf = GPModel(basedir=args.basedir, hires_subdir=args.hires_subdir,
                    fidelity="hf", kf=k_grid)

    print("Reconstructing per-parameter refits (derivative-gated)...")
    refits, gp_sliced = _load_refits(
        refit_z_dir=refit_z_dir, z=args.z, gp_hf=gp_hf, fid=fid,
        k_grid=k_grid, data_1pvar_dir=args.data_1pvar_dir,
        derivative_tol=args.derivative_tol, log_space=log_space,
    )
    n_kept = sum(r is not None for r in refits.values())
    print(f"  refits kept (PySR): {n_kept}/{len(PARAM_NAMES)}; "
          f"GP-sliced: {gp_sliced or 'none'}")

    print("Building additive-Taylor combine...")
    combine = build_combined_model(
        combine_mode="additive", gp=gp_hf, fid=fid, refits=refits,
        k_grid=k_grid, z=args.z, log_space=log_space,
    )

    rows: list[dict] = []
    summaries: dict[str, dict] = {}
    for dim, combo in combos:
        rel, n_skip = _eval_combo(
            combo=combo, fid=fid, combine=combine, gp_hf=gp_hf,
            k_grid=k_grid, z=args.z, n_sobol=args.n_sobol, seed=args.seed,
        )
        params_str = "-".join(combo)
        if rel.shape[0] == 0:
            print(f"  [{dim}] {params_str:<22s} SKIPPED — all "
                  f"{args.n_sobol} Sobol points out of GP domain.")
            continue
        s = _summarize(rel, k_grid)
        sliced_here = [p for p in combo if p in gp_sliced]
        summaries[params_str] = s
        rows.append({
            "dim": dim, "params": params_str, "n_varied": len(combo),
            "mean_pct": round(s["mean_pct"], 4),
            "p90_pct": round(s["p90_pct"], 4),
            "max_pct": round(s["max_pct"], 4),
            "max_k": round(s["max_k"], 5),
            "gp_sliced": "|".join(sliced_here),
            "role": "",
        })
        skip_note = f"  [{n_skip}/{args.n_sobol} OOD skipped]" if n_skip else ""
        print(f"  [{dim}] {params_str:<22s} "
              f"mean={s['mean_pct']:6.2f}%  p90={s['p90_pct']:6.2f}%  "
              f"max={s['max_pct']:7.2f}% @ k={s['max_k']:.4f}"
              + (f"  (GP-sliced: {','.join(sliced_here)})" if sliced_here else "")
              + skip_note)

    # Tag best/worst per dimensionality (the figure's four highlighted combos).
    for dim in ("2D", "3D"):
        sub = [r for r in rows if r["dim"] == dim]
        if not sub:
            continue
        sub.sort(key=lambda r: r["mean_pct"])
        sub[0]["role"] = "best"
        sub[-1]["role"] = "worst"

    out_csv = args.out_dir / "multid_bestworst.csv"
    _write_csv(out_csv, rows)
    print(f"\nWrote {out_csv}")

    out_fig = args.out_dir / "multid_bestworst"
    _make_figure(out_fig, rows, summaries, k_grid, z=args.z, n_sobol=args.n_sobol)
    print(f"Wrote {out_fig}.png / {out_fig}.pdf")

    # Headline summary to stdout.
    for dim in ("2D", "3D"):
        sub = [r for r in rows if r["dim"] == dim]
        if not sub:
            continue
        best = min(sub, key=lambda r: r["mean_pct"])
        worst = max(sub, key=lambda r: r["mean_pct"])
        print(f"\n{dim}: best  = {best['params']:<22s} mean {best['mean_pct']:.2f}% "
              f"(p90 {best['p90_pct']:.2f}%, max {best['max_pct']:.2f}%)")
        print(f"{dim}: worst = {worst['params']:<22s} mean {worst['mean_pct']:.2f}% "
              f"(p90 {worst['p90_pct']:.2f}%, max {worst['max_pct']:.2f}%)")


if __name__ == "__main__":
    main()
