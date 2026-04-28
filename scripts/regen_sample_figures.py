"""Regenerate the sample figures committed under `docs/figures/`.

Uses the *real* PRIYA GP emulator at fiducial as ground truth and the
eBOSS DR14 covariance for the noise model. Builds a synthetic Taylor-
expansion PySR equation set so the comparison figures (PySR vs GP
forecast) are exercised end-to-end.

Run from repo root with the upstream lyaemu repo on PYTHONPATH:

    PYTHONPATH=/home/mfho/student_projects/lya_emulator_full:src \\
        python scripts/regen_sample_figures.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

_LYAEMU = Path("/home/mfho/student_projects/lya_emulator_full")
if _LYAEMU.exists():
    sys.path.insert(0, str(_LYAEMU))

from priya_forecast.config import EqnConfig, EqnParam
from priya_forecast.data import load_eboss
from priya_forecast.diagnostics.compare import EqSetEntry, compare_equation_sets
from priya_forecast.fisher import fisher_matrix
from priya_forecast.likelihood import GaussianLikelihood
from priya_forecast.models.base import P1DModel
from priya_forecast.models.pysr_model import PySRModel
from priya_forecast.parameters import (
    PARAM_NAMES,
    PARAMS_11D,
    Param,
    fiducial_vector,
    get_param,
)

OUT = Path(__file__).resolve().parent.parent / "docs" / "figures"
OUT.mkdir(parents=True, exist_ok=True)

FORECAST_NAMES = ("dtau0", "Ap", "ns", "alphaq")
FORECAST_PARAMS = tuple(p for p in PARAMS_11D if p.name in FORECAST_NAMES)


# The student's actual PySR-learned equations from the InferenceLyaData
# write-up (one per parameter, trained on a multi-D Sobol sweep where the
# inputs are normalized to [0,1] and the output is normalized flux:
#   flux_norm = (P_F - mean_k) / std_k
# The `r` variable is the multi-fidelity resolution flag (0.4 = LF, 0.8 = HF).
# We fix r=0.8 since the forecast targets eBOSS (high-fidelity) data.
STUDENT_EQUATIONS = {
    # dtau0 1D from the {dtau0, Ap} subset (paper Eq. dtau_2d):
    "dtau0": "(((1.4061172 - k)**(-0.5989224)) * dtau0) - (r * 1.3422583) + dtau0 - 1.3998809",
    # Ap 1D from the same subset (paper Eq. ap_2d):
    "Ap":    "(((2*Ap)**(cos(k))) + ((-0.5290618 - sin(r)) * 1.4107764)) + Ap",
    # ns 1D from the {ns, hub} subset:
    "ns":    "((ns * k) - r) * 2.3955164",
    # alphaq 1D from the {herei, alphaq} subset:
    "alphaq": "cos(r + 0.7157408 - 1.5351741*k)**4 / 0.47581 - r - 1.04696",
}


# ---------------------------------------------------------------------------
# 1. Setup
# ---------------------------------------------------------------------------


def _setup():
    from priya_forecast.models.gp_model import GPModel

    print("Loading real PRIYA GP emulator...")
    gp = GPModel()
    z = 3.6
    k_eboss, pf_eboss, cov = load_eboss(z=z)
    fid = np.array(fiducial_vector(), dtype=float)
    p_fid = gp.predict(fid, k_eboss, z)
    return gp, z, k_eboss, pf_eboss, cov, fid, p_fid


# ---------------------------------------------------------------------------
# 2. Synthetic PySR equation set: Taylor-expand the GP per parameter
# ---------------------------------------------------------------------------


class _PerfectPerParamSlice(P1DModel):
    """The "ideal PySR equation set" surrogate: each parameter's response is
    delegated to the GP with everything else held at fiducial. The multi-
    plicative combine of these slices is what a perfect 1D-trained PySR set
    would produce — its mismatch vs the joint GP shows the cost of the
    1D-factorization assumption (which is the multi-D diagnostic's headline)."""

    def __init__(self, gp, fid, forecast_names):
        self._gp = gp; self._fid = fid; self._forecast_names = forecast_names

    def predict(self, theta, k, z):
        theta = np.asarray(theta, dtype=float)
        if theta.shape != self._fid.shape:
            raise ValueError(f"theta shape {theta.shape} != {self._fid.shape}")
        # Multiplicative combine of per-parameter GP slices:
        #   P(theta) ≈ P_fid * ∏_i [P_GP(others=fid, theta_i) / P_GP(fid)]
        p_fid = self._gp.predict(self._fid, k, z)
        out = p_fid.copy()
        for name in self._forecast_names:
            i = PARAM_NAMES.index(name)
            t_only = self._fid.copy(); t_only[i] = theta[i]
            num = self._gp.predict(t_only, k, z)
            out = out * (num / p_fid)
        return out


def _build_perfect_pysr_set(*, gp, fid, k_eboss, z, label: str):
    """Wrap the per-param-slice surrogate as a PySRModel for the comparison
    pipeline. We use the `expression: x0` trick + a custom predict-override
    by attaching a `_PerfectPerParamSlice` directly so the diagnostic loop
    treats it like any other model."""
    fid_npz = OUT / "fiducial_p1d_z3.6.npz"
    np.savez(fid_npz, k=k_eboss, p1d=gp.predict(fid, k_eboss, z))
    parameters = {}
    for pname in PARAM_NAMES:
        parameters[pname] = EqnParam(
            fiducial=get_param(pname).fid,
            expression="x0",  # placeholder; the real predict comes from the surrogate
            variables=[pname, "k"],
        )
    cfg = EqnConfig(
        name=label, redshift=z, model="pysr", combine="multiplicative",
        fiducial_p1d=str(fid_npz), parameters=parameters,
    )
    # Compile the PySRModel for the equation card metadata, then swap predict.
    pysr_model = PySRModel(eqn_cfg=cfg, k_grid=k_eboss, normalization_block={"mode": "identity"})
    surrogate = _PerfectPerParamSlice(gp, fid, FORECAST_NAMES)
    pysr_model.predict = lambda theta, k, z: surrogate.predict(theta, k, z)  # type: ignore[assignment]
    # Replace the equation strings on the compiled cache for the card.
    for pname in FORECAST_NAMES:
        ce = pysr_model.compiled[pname]
        ce.raw_expression = f"GP({pname}, others=fid, k)  ← idealized 1D slice"
    return pysr_model, cfg


def _build_student_pysr_set(*, gp, fid, k_eboss, z, label: str = "student_paper_eqs"):
    """Build a PySRModel from the student's quoted paper equations.

    Inputs are normalized to [0,1]; outputs are normalized flux. We use
    the framework's `mode='auto'` to derive (mean_k, std_k) by sweeping
    each parameter via the GP at fixed-fiducial-rest — exactly the
    convention `pysr_mf_given.py` uses on its own training data.

    The `r` (resolution) variable is collapsed to 0.8 (HF / eBOSS-like).
    """
    fid_npz = OUT / "fiducial_p1d_z3.6.npz"
    np.savez(fid_npz, k=k_eboss, p1d=gp.predict(fid, k_eboss, z))

    # Build per-param normalization specs by sweeping the GP.
    from priya_forecast.models.normalization import derive_from_gp

    parameters: dict[str, EqnParam] = {}
    norm_specs: dict[str, "NormalizationSpec"] = {}
    for pname in PARAM_NAMES:
        if pname in STUDENT_EQUATIONS:
            parameters[pname] = EqnParam(
                fiducial=get_param(pname).fid,
                expression=STUDENT_EQUATIONS[pname],
                variables=[pname, "k", "r"],
            )
            norm_specs[pname] = derive_from_gp(
                gp_model=gp, param_name=pname, z=z, k_grid=k_eboss,
                n_samples=64, seed=0,
            )
        else:
            parameters[pname] = EqnParam(
                fiducial=get_param(pname).fid,
                expression="1",
                variables=[pname, "k"],
            )

    cfg = EqnConfig(
        name=label, redshift=z, model="pysr", combine="multiplicative",
        fiducial_p1d=str(fid_npz), parameters=parameters,
    )

    # Build with a dummy normalization first; we'll patch per-param specs in.
    from priya_forecast.models.pysr_model import compile_equation
    from priya_forecast.models.normalization import identity

    pysr_model = PySRModel(
        eqn_cfg=cfg, k_grid=k_eboss,
        normalization_block={"mode": "identity", "fix": {"r": 0.8}},
    )
    # Recompile only the student-equation parameters with the GP-derived norm.
    for pname in STUDENT_EQUATIONS:
        ep = parameters[pname]
        pysr_model.compiled[pname] = compile_equation(
            param_name=pname,
            raw_expression=STUDENT_EQUATIONS[pname],
            variables=[pname, "k", "r"],
            fix={"r": 0.8},
            norm=norm_specs[pname],
            fiducial=ep.fiducial,
        )
    return pysr_model, cfg


def _build_taylor_pysr_set(*, gp, fid, k_eboss, z, label: str, order: int = 2):
    """Per-parameter polynomial fit to the GP across the prior AND k.

    For each parameter, we sweep p_norm ∈ {0, fid_norm, 1} and treat the
    resulting GP outputs as a 2D function of (p_norm, k_norm). A bivariate
    polynomial up to total degree `order` is fit, then written as a sympy
    expression in (x0, x1). This mimics what a competent PySR run would
    produce (rich in both p and k), so the comparison figures actually
    exercise the multi-D structure of the forecast.
    """
    fid_npz = OUT / "fiducial_p1d_z3.6.npz"
    np.savez(fid_npz, k=k_eboss, p1d=gp.predict(fid, k_eboss, z))

    k_norm = (k_eboss - k_eboss.min()) / (k_eboss.max() - k_eboss.min())

    parameters = {}
    for pname in FORECAST_NAMES:
        idx = PARAM_NAMES.index(pname)
        lo, hi = get_param(pname).prior
        sample_thetas = np.linspace(lo, hi, 7)  # denser than 3 points for k-shape recovery
        ps = []
        for v in sample_thetas:
            tt = fid.copy(); tt[idx] = v
            ps.append(gp.predict(tt, k_eboss, z))
        ps = np.stack(ps, axis=0)  # (Nsamp, nk)
        p_norm_samples = (sample_thetas - lo) / (hi - lo)

        # Build (Nsamp * nk, ?) design matrix for total-degree-`order` polynomial.
        P_grid, K_grid = np.meshgrid(p_norm_samples, k_norm, indexing="ij")
        P_flat, K_flat = P_grid.ravel(), K_grid.ravel()
        y_flat = ps.ravel()
        terms: list[tuple[int, int]] = []
        for dp in range(order + 1):
            for dk in range(order + 1 - dp):
                terms.append((dp, dk))
        A = np.column_stack([P_flat ** dp * K_flat ** dk for dp, dk in terms])
        coeffs, *_ = np.linalg.lstsq(A, y_flat, rcond=None)

        # Build sympy-readable polynomial in (x0=p_norm, x1=k_norm).
        parts = []
        for c, (dp, dk) in zip(coeffs, terms):
            if abs(c) < 1e-12:
                continue
            term = f"{c:.6g}"
            if dp > 0:
                term += f"*x0**{dp}" if dp > 1 else "*x0"
            if dk > 0:
                term += f"*x1**{dk}" if dk > 1 else "*x1"
            parts.append(term)
        expr = " + ".join(parts) if parts else "0"
        parameters[pname] = EqnParam(
            fiducial=get_param(pname).fid,
            expression=expr,
            variables=[pname, "k"],
        )

    # Add the other 7 params with `f_i = 1` so they don't perturb the combine.
    for pname in PARAM_NAMES:
        if pname in FORECAST_NAMES:
            continue
        parameters[pname] = EqnParam(
            fiducial=get_param(pname).fid,
            expression="1",
            variables=[pname, "k"],
        )

    cfg = EqnConfig(
        name=label,
        redshift=z,
        model="pysr",
        combine="multiplicative",
        fiducial_p1d=str(fid_npz),
        parameters=parameters,
    )
    model = PySRModel(eqn_cfg=cfg, k_grid=k_eboss, normalization_block={"mode": "identity"})
    return model, cfg


# ---------------------------------------------------------------------------
# 3. Standalone diagnostics (GP-only)
# ---------------------------------------------------------------------------


def fig_gp_at_fiducial(*, k, p_fid, pf_eboss, cov, z, outpath):
    import matplotlib.pyplot as plt

    sigma = np.sqrt(np.diag(cov))
    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    ax.errorbar(k, pf_eboss, yerr=sigma, fmt=".", capsize=2, color="black",
                label=f"eBOSS DR14 (z={z})")
    ax.plot(k, p_fid, "C0-", lw=2, label="GP at fiducial")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel(r"$k$ [s/km]"); ax.set_ylabel(r"$P_F(k)$")
    ax.set_title(f"GP prediction at fiducial vs eBOSS DR14, z={z}")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)
    print(f"wrote {outpath}")


def fig_param_sensitivity(*, gp, k, z, fid, outpath):
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    p_fid = gp.predict(fid, k, z)
    palette = plt.get_cmap("tab20").colors
    for i, p in enumerate(PARAMS_11D):
        h = 0.02 * p.width()
        t_p = fid.copy(); t_p[i] += h
        t_m = fid.copy(); t_m[i] -= h
        dp = (gp.predict(t_p, k, z) - gp.predict(t_m, k, z)) / (2 * h)
        sens = dp * p.width() / p_fid
        ax.plot(k, sens, lw=1.4, color=palette[i % len(palette)], label=p.name)
    ax.axhline(0, color="black", lw=0.5)
    ax.set_xscale("log")
    ax.set_xlabel(r"$k$ [s/km]")
    ax.set_ylabel(r"$d\,\ln P_F / d\hat\theta$  (per prior-width)")
    ax.set_title(f"GP per-parameter sensitivity at fiducial, z={z}")
    ax.legend(fontsize=7, ncol=3, loc="best")
    ax.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)
    print(f"wrote {outpath}")


# ---------------------------------------------------------------------------
# 4. Fisher 1D / 2D / 3D / 4D vs prior width
# ---------------------------------------------------------------------------


def _proj(base, fid_full, sub_idx):
    class _P(P1DModel):
        def predict(self, ts, k, z):
            full = fid_full.copy()
            for i, idx in enumerate(sub_idx):
                full[idx] = ts[i]
            return base.predict(full, k, z)
    return _P()


def _fisher_subset(*, gp, fid, z, names: list[str]):
    sub = tuple(p for p in PARAMS_11D if p.name in names)
    sub_idx = [PARAM_NAMES.index(p.name) for p in sub]
    fid_sub = np.array([fid[i] for i in sub_idx])
    lk = GaussianLikelihood(
        model=_proj(gp, fid, sub_idx), z=z, mock_data="gp", theta_fid=fid_sub,
    )
    return fisher_matrix(
        likelihood=lk, theta_fid=fid_sub, params=sub,
        step_frac=0.02, rel_tol=0.05, max_halvings=2,
    )


def fig_fisher_dimensions(*, gp, fid, z, outpath):
    """Bar chart with grouped bars: same param across 1D/2D/3D/4D Fisher."""
    import matplotlib.pyplot as plt

    setups = [
        ("1D", [[n] for n in FORECAST_NAMES]),
        ("2D", [list(FORECAST_NAMES[:2])]),
        ("3D", [list(FORECAST_NAMES[:3])]),
        ("4D", [list(FORECAST_NAMES)]),
    ]
    # For 1D, run separately per param; for joint, run the joint Fisher.
    sigma_per_param: dict[str, dict[str, float]] = {n: {} for n in FORECAST_NAMES}
    for label, name_lists in setups:
        for names in name_lists:
            res = _fisher_subset(gp=gp, fid=fid, z=z, names=names)
            for i, n in enumerate(names):
                sigma_per_param[n][label] = res.sigma[i] / get_param(n).width()

    labels = ["1D", "2D", "3D", "4D"]
    n_params = len(FORECAST_NAMES)
    bar_w = 0.8 / len(labels)
    x = np.arange(n_params)
    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=120)
    fig.patch.set_facecolor("white"); ax.set_facecolor("white")
    palette = plt.get_cmap("viridis")(np.linspace(0.15, 0.85, len(labels)))
    for i, lab in enumerate(labels):
        vals = [sigma_per_param[n].get(lab, 0.0) for n in FORECAST_NAMES]
        ax.bar(x + i * bar_w, vals, width=bar_w, label=lab, color=palette[i])
    ax.set_xticks(x + 0.4 - bar_w / 2)
    ax.set_xticklabels(FORECAST_NAMES)
    ax.set_ylabel(r"$\sigma_{\rm marg}\,/\,$prior width")
    ax.set_title(f"GP Fisher: 1D → 2D → 3D → 4D parameter constraints, z={z}")
    ax.legend(title="dim", fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout(); fig.savefig(outpath); plt.close(fig)
    print(f"wrote {outpath}")


# ---------------------------------------------------------------------------
# 5. Driver
# ---------------------------------------------------------------------------


def main():
    gp, z, k_eboss, pf_eboss, cov, fid, p_fid = _setup()

    # --- standalone GP diagnostics ---
    fig_gp_at_fiducial(
        k=k_eboss, p_fid=p_fid, pf_eboss=pf_eboss, cov=cov, z=z,
        outpath=OUT / "fig01_gp_at_fiducial.png",
    )
    fig_param_sensitivity(
        gp=gp, k=k_eboss, z=z, fid=fid,
        outpath=OUT / "fig02_gp_param_sensitivity.png",
    )

    # --- 1D → 4D Fisher progression ---
    fig_fisher_dimensions(
        gp=gp, fid=fid, z=z, outpath=OUT / "fig03_fisher_1d_to_4d.png",
    )

    # --- Build the two demonstration PySR sets:
    #     - "perfect_1D_slices" : delegates to the GP per-parameter slice.
    #       The floor of what a 1D-trained PySR set can achieve.
    #     - "student_paper_eqs" : the actual equations the student published
    #       for {dtau0, Ap, ns, alphaq}. The forecast subspace is exactly
    #       these four params so we can score the equations end-to-end.
    print("Building demonstration PySR equation sets...")
    pysr_perfect, cfg_perfect = _build_perfect_pysr_set(
        gp=gp, fid=fid, k_eboss=k_eboss, z=z, label="perfect_1D_slices",
    )
    pysr_student, cfg_student = _build_student_pysr_set(
        gp=gp, fid=fid, k_eboss=k_eboss, z=z, label="student_paper_eqs",
    )

    out = compare_equation_sets(
        gp_model=gp,
        pysr_sets=[
            EqSetEntry(name="perfect_1D_slices", model=pysr_perfect, eqn_cfg=cfg_perfect),
            EqSetEntry(name="student_paper_eqs", model=pysr_student, eqn_cfg=cfg_student),
        ],
        z=z, k_eboss=k_eboss, cov_eboss=cov,
        forecast_params=FORECAST_PARAMS,
        outdir=OUT / "compare_loop",
    )
    print(f"comparison loop output → {out}")
    print((out / "summary.md").read_text())


if __name__ == "__main__":
    main()
