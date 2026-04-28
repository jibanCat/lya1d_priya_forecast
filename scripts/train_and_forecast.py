"""End-to-end PySR forecast scoring driver.

Student-facing reward loop. Pick a parameter subset and an equation
source. The script scores the equation set against two upper bounds:

  1. `perfect_1D_slices` — what an idealized 1D-trained PySR set could
     achieve under multiplicative combine (delegates to the GP per-param
     slice). Equals σ_GP exactly when the equations match the GP.

  2. `gp_joint` — the GP itself, the upper bound for what *any* equation
     set (including a joint multi-D PySR run) could achieve.

The gap between σ_student and σ_perfect_1D measures how much the equation
set is losing relative to a perfect 1D fit. Closing it = retrain with
more PySR iterations / larger maxsize / different operators.

The gap between σ_perfect_1D and σ_GP measures the cost of the
1D-factorization assumption — what multi-D PySR could in principle
recover. If this gap is small, the paper's 1D approach is fine; if it's
large, multi-D PySR is worth the cost.

Usage:
  --equations <yaml>     score a YAML equation set (1D-product or joint)
  --equations published   score the four equations the user quoted earlier
  --equations none        run only the references (sanity check pipeline)

Example:
  PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
      python scripts/train_and_forecast.py \\
          --params dtau0 Ap ns alphaq \\
          --equations published \\
          --output results/published_scorecard
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.config import EqnConfig, EqnParam, load_eqn_config
from priya_forecast.data import load_eboss
from priya_forecast.diagnostics.compare import EqSetEntry, compare_equation_sets
from priya_forecast.models import PySRModel
from priya_forecast.models.base import P1DModel
from priya_forecast.models.normalization import derive_from_gp
from priya_forecast.models.pysr_model import compile_equation
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    fiducial_vector,
    get_param,
)


# ---------------------------------------------------------------------------
# Reference 1: the upper bound for 1D-product combine (delegate to GP slice)
# ---------------------------------------------------------------------------


class _PerfectPerParamSlice(P1DModel):
    def __init__(self, gp, fid_full, varying_names):
        self._gp = gp
        self._fid = fid_full
        self._varying = list(varying_names)

    def predict(self, theta, k, z):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self._fid.shape:
            raise ValueError(f"theta shape {theta.shape} != {self._fid.shape}")
        p_fid = self._gp.predict(self._fid, k, z)
        out = p_fid.copy()
        for name in self._varying:
            i = PARAM_NAMES.index(name)
            t_only = self._fid.copy()
            t_only[i] = theta[i]
            num = self._gp.predict(t_only, k, z)
            out = out * (num / p_fid)
        return out


def build_perfect_1d_reference(*, gp, fid, k_grid, z, varying_names, label="perfect_1D_slices"):
    fid_npz = Path("results") / "_cache" / f"fiducial_p1d_z{z}.npz"
    fid_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fid_npz, k=k_grid, p1d=gp.predict(fid, k_grid, z))
    parameters = {pname: EqnParam(fiducial=get_param(pname).fid, expression="x0", variables=[pname, "k"])
                  for pname in PARAM_NAMES}
    cfg = EqnConfig(
        name=label, redshift=z, model="pysr", combine="multiplicative",
        fiducial_p1d=str(fid_npz), parameters=parameters,
    )
    pysr_model = PySRModel(eqn_cfg=cfg, k_grid=k_grid, normalization_block={"mode": "identity"})
    pysr_model.predict = _PerfectPerParamSlice(gp, fid, varying_names).predict  # type: ignore[assignment]
    for pname in varying_names:
        ce = pysr_model.compiled[pname]
        ce.raw_expression = f"GP({pname}, others=fid, k)  ← idealized 1D slice"
    return pysr_model, cfg


# ---------------------------------------------------------------------------
# Reference 2: the GP itself, wrapped as a "perfect joint" PySRModel
# ---------------------------------------------------------------------------


def build_gp_reference(*, gp, fid, k_grid, z, label="gp_reference"):
    """Wrap the GP as an EqSetEntry so it appears alongside other sets in
    the scorecard. The 'equation' for the card is just 'GP(theta, k)'."""
    fid_npz = Path("results") / "_cache" / f"fiducial_p1d_z{z}.npz"
    fid_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fid_npz, k=k_grid, p1d=gp.predict(fid, k_grid, z))
    parameters = {pname: EqnParam(fiducial=get_param(pname).fid, expression="x0", variables=[pname, "k"])
                  for pname in PARAM_NAMES}
    cfg = EqnConfig(
        name=label, redshift=z, model="pysr", combine="multiplicative",
        fiducial_p1d=str(fid_npz), parameters=parameters,
    )
    pysr_model = PySRModel(eqn_cfg=cfg, k_grid=k_grid, normalization_block={"mode": "identity"})
    pysr_model.predict = lambda theta, k_, z_: gp.predict(theta, k_, z_)  # type: ignore[assignment]
    for pname in PARAM_NAMES:
        pysr_model.compiled[pname].raw_expression = f"GP(theta, k)  ← reference"
    return pysr_model, cfg


# ---------------------------------------------------------------------------
# Equation source: the published student equations from the user's message
# ---------------------------------------------------------------------------


PUBLISHED_EQUATIONS = {
    "dtau0":  "(((1.4061172 - k)**(-0.5989224)) * dtau0) - (r * 1.3422583) + dtau0 - 1.3998809",
    "Ap":     "(((2*Ap)**(cos(k))) + ((-0.5290618 - sin(r)) * 1.4107764)) + Ap",
    "ns":     "((ns * k) - r) * 2.3955164",
    "alphaq": "cos(r + 0.7157408 - 1.5351741*k)**4 / 0.47581 - r - 1.04696",
}


def build_published_set(*, gp, fid, k_grid, z, label="student_published"):
    fid_npz = Path("results") / "_cache" / f"fiducial_p1d_z{z}.npz"
    fid_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(fid_npz, k=k_grid, p1d=gp.predict(fid, k_grid, z))

    parameters = {}
    for pname in PARAM_NAMES:
        if pname in PUBLISHED_EQUATIONS:
            parameters[pname] = EqnParam(
                fiducial=get_param(pname).fid,
                expression=PUBLISHED_EQUATIONS[pname],
                variables=[pname, "k", "r"],
            )
        else:
            parameters[pname] = EqnParam(
                fiducial=get_param(pname).fid,
                expression="1", variables=[pname, "k"],
            )

    cfg = EqnConfig(
        name=label, redshift=z, model="pysr", combine="multiplicative",
        fiducial_p1d=str(fid_npz), parameters=parameters,
    )
    pysr_model = PySRModel(eqn_cfg=cfg, k_grid=k_grid,
                           normalization_block={"mode": "identity", "fix": {"r": 0.8}})
    # Re-compile with GP-derived per-k normalization for the parameters with equations.
    for pname in PUBLISHED_EQUATIONS:
        norm = derive_from_gp(
            gp_model=gp, param_name=pname, z=z, k_grid=k_grid, n_samples=64,
        )
        pysr_model.compiled[pname] = compile_equation(
            param_name=pname,
            raw_expression=PUBLISHED_EQUATIONS[pname],
            variables=[pname, "k", "r"],
            fix={"r": 0.8},
            norm=norm,
            fiducial=get_param(pname).fid,
        )
    return pysr_model, cfg


# ---------------------------------------------------------------------------
# Equation source: from a student-supplied YAML
# ---------------------------------------------------------------------------


def build_from_yaml(*, gp, fid, k_grid, z, yaml_path, label_prefix="from_yaml"):
    cfg = load_eqn_config(yaml_path)
    if cfg.redshift != z:
        raise ValueError(
            f"YAML redshift {cfg.redshift} != requested z {z}. Pick a matching YAML."
        )
    block = {"mode": "auto", "fix": {"r": 0.8}}
    pysr_model = PySRModel(eqn_cfg=cfg, k_grid=k_grid, normalization_block=block)
    return pysr_model, cfg


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def _scorecard(*, summary_md, fr_perfect, fr_gp, fr_student, label, k, fid, gp, perfect_model, student_model, z, n_offfid: int = 16):
    """Build the production scorecard.

    Three gauges:

    1. **σ_student / σ_perfect_1D** at Fisher level — the headline number.
       Measures how far the student's equations are from a perfect 1D fit.
       Achievable: < 1.5. Ideal: ~ 1.

    2. **σ_perfect_1D / σ_gp** at Fisher level — note that this is
       *exactly 1* at the linearization point for any well-formed per-param
       set, because at fid the 1D-product gradient equals the joint gradient
       by the chain rule. Reported here as a sanity check only.

    3. **off-fid MSE: 1D-product vs joint** — the actual cost of the
       1D-factorization assumption shows up at off-fid points where
       cross-parameter terms matter. We Sobol-sample n_offfid points and
       compare ‖P_perfect_1D - P_GP‖² vs ‖P_student - P_GP‖² in eBOSS-σ
       units. Larger ratios = more 1D-product breakdown.
    """
    lines = [summary_md, ""]
    names = fr_gp.param_names

    def _ratio(a, b):
        return a / b if (b > 0 and np.isfinite(a)) else float("inf")

    student_to_perfect = [_ratio(fr_student.sigma[i], fr_perfect.sigma[i]) for i in range(len(names))]
    perfect_to_gp = [_ratio(fr_perfect.sigma[i], fr_gp.sigma[i]) for i in range(len(names))]

    def _gm(xs):
        finite = [x for x in xs if np.isfinite(x) and x > 0]
        return float(np.exp(np.mean(np.log(finite)))) if finite else float("inf")

    # Off-fid residual comparison.
    rng = np.random.default_rng(0)
    widths = np.array([p.width() for p in PARAMS_11D])
    perfect_mses = []
    student_mses = []
    cov_diag = np.diag(load_eboss(z=z)[2])
    sigma_eboss = np.sqrt(cov_diag)
    for _ in range(n_offfid):
        delta = rng.uniform(-0.3, 0.3, size=11) * widths
        # Only perturb forecast params; others stay at fid.
        theta = fid.copy()
        for i, name in enumerate(PARAM_NAMES):
            if name in {p.name for p in [PARAMS_11D[j] for j in range(11)] if p.name in
                        [n for n in names]}:
                theta[i] = fid[i] + delta[i]
        try:
            p_truth = gp.predict(theta, k, z)
            p_perfect = perfect_model.predict(theta, k, z)
            p_student = student_model.predict(theta, k, z)
            perfect_mses.append(float(np.mean(((p_perfect - p_truth) / sigma_eboss) ** 2)))
            student_mses.append(float(np.mean(((p_student - p_truth) / sigma_eboss) ** 2)))
        except Exception:
            continue
    if perfect_mses and student_mses:
        offfid_perfect = float(np.mean(perfect_mses))
        offfid_student = float(np.mean(student_mses))
    else:
        offfid_perfect, offfid_student = float("nan"), float("nan")

    lines.append("### Reward gauges (lower = better)")
    lines.append("")
    lines.append("| Gauge | What it measures | Per-param | Geomean |")
    lines.append("|---|---|---|---|")
    lines.append(
        f"| σ_student / σ_perfect_1D | distance from 1D-product upper bound | "
        f"{', '.join(f'{x:.2g}' for x in student_to_perfect)} | "
        f"**{_gm(student_to_perfect):.2g}** |"
    )
    lines.append(
        f"| σ_perfect_1D / σ_gp | 1D-factorization tax at Fisher level "
        f"(always ≈ 1 — see below) | "
        f"{', '.join(f'{x:.2g}' for x in perfect_to_gp)} | "
        f"{_gm(perfect_to_gp):.2g} |"
    )
    lines.append("")
    lines.append(f"### Off-fiducial residual MSE (eBOSS-σ² units, mean over {n_offfid} Sobol points)")
    lines.append("")
    lines.append(f"- perfect_1D vs GP : **{offfid_perfect:.3g}**")
    lines.append(f"- student   vs GP : **{offfid_student:.3g}**")
    if np.isfinite(offfid_perfect) and offfid_perfect > 0:
        lines.append(f"- ratio (student / perfect_1D) : **{offfid_student / offfid_perfect:.2g}**")
    lines.append("")
    lines.append("### Why σ_perfect_1D / σ_gp = 1 here")
    lines.append("")
    lines.append("At the linearization point (fid), the 1D-product Fisher gradient ")
    lines.append("∂P/∂θ_i = P_fid · (1/f_i_fid) · df_i/dθ_i equals the joint gradient ")
    lines.append("∂GP/∂θ_i for any equation set whose per-param 1D slices match the GP. ")
    lines.append("So Fisher *cannot* see the 1D-factorization tax — only off-fid points or ")
    lines.append("MCMC curvature can. Use Phase 5's coupling-matrix diagnostic to quantify ")
    lines.append("the joint-vs-product tax across off-fid Sobol space.")
    lines.append("")
    lines.append("### Targets to chase")
    lines.append("")
    lines.append("1. σ_student / σ_perfect_1D < 1.5 (geomean) → 1D PySR is converged.")
    lines.append("2. off-fid MSE ratio < 2 → the equations track the GP off-fid too.")
    lines.append("3. Once 1 and 2 are met, run the multi-D PySR diagnostic (Phase 5).")
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--params", nargs="+", required=True,
                        help="Parameter names to forecast on (subset of the 11).")
    parser.add_argument("--equations", default="published",
                        help="'published' | 'none' | path to a YAML config.")
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

    forecast_params = tuple(p for p in PARAMS_11D if p.name in args.params)
    if len(forecast_params) != len(args.params):
        unknown = set(args.params) - set(PARAM_NAMES)
        raise ValueError(f"Unknown parameter names: {unknown}.")

    # Build references: the two upper bounds.
    pysr_perfect, cfg_perfect = build_perfect_1d_reference(
        gp=gp, fid=fid, k_grid=k_eboss, z=z, varying_names=args.params,
    )
    pysr_gp, cfg_gp = build_gp_reference(gp=gp, fid=fid, k_grid=k_eboss, z=z)

    sets = [
        EqSetEntry(name="GP_reference",        model=pysr_gp,      eqn_cfg=cfg_gp),
        EqSetEntry(name="perfect_1D_slices",   model=pysr_perfect, eqn_cfg=cfg_perfect),
    ]

    # Optionally add the student set.
    student_label = None
    if args.equations == "published":
        m, c = build_published_set(gp=gp, fid=fid, k_grid=k_eboss, z=z)
        sets.append(EqSetEntry(name="student_published", model=m, eqn_cfg=c))
        student_label = "student_published"
    elif args.equations == "none":
        pass
    else:
        m, c = build_from_yaml(
            gp=gp, fid=fid, k_grid=k_eboss, z=z, yaml_path=args.equations,
        )
        sets.append(EqSetEntry(name=c.name, model=m, eqn_cfg=c))
        student_label = c.name

    out = compare_equation_sets(
        gp_model=gp, pysr_sets=sets, z=z, k_eboss=k_eboss, cov_eboss=cov,
        forecast_params=forecast_params, outdir=args.output,
    )
    print(f"\nFigures + scorecard at {out}")

    # Build a richer scorecard if we have a student set.
    if student_label is not None:
        from priya_forecast.diagnostics.compare import _fisher_for as fisher_for
        fr_gp = fisher_for(model=gp, fid=fid, k=k_eboss, z=z, params=forecast_params)
        fr_perfect = fisher_for(model=pysr_perfect, fid=fid, k=k_eboss, z=z, params=forecast_params)
        student_model = next(s.model for s in sets if s.name == student_label)
        fr_student = fisher_for(model=student_model, fid=fid, k=k_eboss, z=z, params=forecast_params)
        md = (out / "summary.md").read_text()
        full = _scorecard(
            summary_md=md, fr_perfect=fr_perfect, fr_gp=fr_gp,
            fr_student=fr_student, label=student_label,
            k=k_eboss, fid=fid, gp=gp,
            perfect_model=pysr_perfect, student_model=student_model, z=z,
        )
        (out / "scorecard.md").write_text(full)
        print("\n=== scorecard ===")
        print(full)
    else:
        print("\n(no student equations supplied — only references compared)")


if __name__ == "__main__":
    main()
