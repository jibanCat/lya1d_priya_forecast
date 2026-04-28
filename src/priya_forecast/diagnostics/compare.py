"""Production diagnostic loop: compare multiple PySR equation sets vs the GP.

`compare_equation_sets` takes:
- A real GP model (the reference)
- A list of pre-built PySRModel instances + their EqnConfigs
- An eBOSS k-grid + covariance

and produces a directory of figures:
  - `eq_card_<name>.png`     — one card per equation set with LaTeX equations
  - `forecast_corner.png`    — Fisher corner: GP + each PySR set overlaid
  - `forecast_sigma.png`     — bar chart: 1σ per parameter, all sets side-by-side
  - `residual_<name>.png`    — (P_PySR - P_GP) / σ_eBOSS per equation set
  - `summary.md`             — markdown table of σ ratios PySR/GP per param

This is the "reward loop": the student trains a new PySR run, drops the
resulting YAML in `configs/eqns/`, and reruns this comparison to see
exactly how the new equations stack up against the GP forecast.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from priya_forecast.config import EqnConfig
from priya_forecast.diagnostics.equation_card import plot_equation_card
from priya_forecast.diagnostics.forecast_plots import (
    plot_fisher_corner,
    plot_fisher_sigma_table,
    plot_residuals_at_fiducial,
)
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


@dataclass
class EqSetEntry:
    """One PySR equation set in the comparison."""

    name: str
    model: PySRModel
    eqn_cfg: EqnConfig


def _proj_model(base: P1DModel, fid_full: np.ndarray, sub_idx: list[int]) -> P1DModel:
    """Wrap a model so it accepts a sub-vector of length len(sub_idx)."""

    class _Proj(P1DModel):
        def predict(self, theta_sub, k, z):
            full = fid_full.copy()
            for i, idx in enumerate(sub_idx):
                full[idx] = theta_sub[i]
            return base.predict(full, k, z)

    return _Proj()


def _fisher_for(*, model, fid, k, z, params: tuple[Param, ...]):
    sub_idx = [PARAM_NAMES.index(p.name) for p in params]
    fid_sub = np.array([fid[i] for i in sub_idx])
    proj = _proj_model(model, fid, sub_idx)
    lk = GaussianLikelihood(model=proj, z=z, mock_data="gp", theta_fid=fid_sub)
    return fisher_matrix(
        likelihood=lk, theta_fid=fid_sub, params=params,
        step_frac=0.02, rel_tol=0.05, max_halvings=2,
    )


def _equations_dict_for_card(eq_cfg: EqnConfig, model: PySRModel) -> dict:
    """Build the parameters mapping for plot_equation_card."""
    out = {}
    for pname, ce in model.compiled.items():
        out[pname] = {
            "raw_expression": ce.raw_expression,
            "variables": [pname, "k"] + list(ce.extra_args),
            "complexity": ce.complexity,
            "loss": ce.loss,
            "fiducial": ce.fiducial,
        }
    return out


def _summary_markdown(
    *, gp_fisher, pysr_fishers: dict, params: tuple[Param, ...]
) -> str:
    """One row per parameter; columns = GP σ + each PySR σ + ratio."""
    lines = ["| Parameter | GP σ | "]
    for label in pysr_fishers:
        lines[0] += f"{label} σ | {label} / GP | "
    lines[0] = lines[0].rstrip(" |") + " |"
    lines.append("|---|---|" + "---|" * (2 * len(pysr_fishers)))
    for i, p in enumerate(params):
        row = f"| {p.name} | {gp_fisher.sigma[i]:.3g} | "
        for label, fr in pysr_fishers.items():
            ratio = fr.sigma[i] / gp_fisher.sigma[i] if gp_fisher.sigma[i] > 0 else float("nan")
            row += f"{fr.sigma[i]:.3g} | {ratio:.3f} | "
        lines.append(row.rstrip(" |") + " |")
    return "\n".join(lines)


def compare_equation_sets(
    *,
    gp_model: P1DModel,
    pysr_sets: Sequence[EqSetEntry],
    z: float,
    k_eboss: np.ndarray,
    cov_eboss: np.ndarray,
    forecast_params: tuple[Param, ...] = tuple(p for p in PARAMS_11D if p.name in {"ns", "Ap", "hub", "omegamh2"}),
    outdir: str | Path,
) -> Path:
    """Run the full PySR-vs-GP comparison loop.

    Returns the output directory path. Generates a fresh set of figures
    each call — safe to invoke after every PySR retraining.
    """
    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fid = np.array(fiducial_vector(), dtype=float)

    # --- Equation cards (one per PySR set) ---
    for entry in pysr_sets:
        plot_equation_card(
            name=entry.name,
            redshift=entry.eqn_cfg.redshift,
            combine=entry.eqn_cfg.combine,
            parameters=_equations_dict_for_card(entry.eqn_cfg, entry.model),
            outpath=outdir / f"eq_card_{entry.name}.png",
        )

    # --- Residuals at fid (P_PySR - P_GP) / sigma_eBOSS per set ---
    cov_diag = np.diag(cov_eboss)
    for entry in pysr_sets:
        plot_residuals_at_fiducial(
            pysr_model=entry.model, gp_model=gp_model, k=k_eboss, z=z,
            theta_fid=fid, cov_diag=cov_diag,
            outpath=outdir / f"residual_{entry.name}.png",
        )

    # --- Fisher per set on the chosen forecast subspace ---
    fr_gp = _fisher_for(model=gp_model, fid=fid, k=k_eboss, z=z, params=forecast_params)
    pysr_fishers = {}
    for entry in pysr_sets:
        pysr_fishers[entry.name] = _fisher_for(
            model=entry.model, fid=fid, k=k_eboss, z=z, params=forecast_params,
        )

    plot_fisher_sigma_table(
        fisher_results={"GP": fr_gp, **pysr_fishers},
        outpath=outdir / "forecast_sigma.png",
    )
    plot_fisher_corner(
        fisher_results={"GP": fr_gp, **pysr_fishers},
        outpath=outdir / "forecast_corner.png",
        param_subset=None,
    )

    # --- Markdown summary ---
    md = _summary_markdown(gp_fisher=fr_gp, pysr_fishers=pysr_fishers, params=forecast_params)
    (outdir / "summary.md").write_text(md + "\n")

    return outdir
