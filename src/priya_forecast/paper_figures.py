"""Reusable, path- and format-decoupled API for the paper's figures + tables.

This is the *reusable* layer a student (or a future paper) imports to regenerate
and **tweak** the emulator-free figures and tables. Design goals:

- **No hard-coded local paths.** Everything hangs off a ``data_dir`` argument
  (defaulting to the committed production run, fully overridable). Nothing here
  references a personal home directory or the GP emulator.
- **Decoupled from the on-disk data format.** :func:`load_run` reads the CSVs /
  JSON / sidecars *once* into plain in-memory structures (pandas DataFrames and
  dicts) held on a :class:`PaperRun`. Every plot/table function takes that
  ``run`` object, never a file path — so if the on-disk schema changes, only
  ``load_run`` changes, not the figures.
- **Tweakable.** Each ``plot_*`` returns a Matplotlib ``Figure`` (except the
  11-panel grid, which delegates to the shared renderer and writes a file), and
  takes a ``style`` dict + kwargs, so a student changes colours / sizes / which
  parameters and re-runs. :func:`paper_style` gives the paper look (large
  LaTeX-rendered labels) and is fully overridable.

Emulator-free scope: the diagnostic figures (pareto faithfulness, scorecard,
ns-budget, maxsize-sensitivity, seed band, cross-z) and Tables 1/6/7 + the
multi-D summary. The prediction figures (Fig 1/3/4) and re-generating the
multi-D table need the GP emulator and live in the ``regen_fig*`` / ``regen_multid``
scripts; this module only *reads* the committed multi-D CSV.

Quick start
-----------
>>> from priya_forecast import paper_figures as pf
>>> run = pf.load_run()                      # committed run; or load_run("/path/to/other_run")
>>> tax = pf.taxonomy(run); print(tax)        # Table 6
>>> with pf.paper_style():                    # large LaTeX labels (override freely)
...     fig = pf.plot_scorecard(run)
...     fig.savefig("scorecard.pdf")
"""
from __future__ import annotations

import json
import warnings
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from priya_forecast.grad_faith_io import knee_row, read_grad_faith_sidecar
from priya_forecast.parameters import PARAM_NAMES

# The committed production run, as a repo-relative path (NOT absolute).
DEFAULT_RUN_DIR = "results/paper_production_20260630_perz_sobolev_z2.6-4.2"
GATE = 0.25

# LaTeX labels for the 11 PRIYA parameters (used on axes/tables). Override by
# passing your own {name: label} to the plot/table functions via `pretty=`.
PRETTY = {
    "dtau0": r"$d\tau_0$", "tau0": r"$\tau_0$", "ns": r"$n_S$", "Ap": r"$A_P$",
    "herei": r"$z_{\mathrm{HeI}}$", "heref": r"$z_{\mathrm{HeF}}$",
    "alphaq": r"$\alpha_q$", "hub": r"$h$", "omegamh2": r"$\Omega_M h^2$",
    "hireionz": r"$z_{\mathrm{Hi}}$", "bhfeedback": r"$\epsilon_{\mathrm{AGN}}$",
}

# Consistent series colours (override per-call with color kwargs).
C_VALUE = "#d6604d"    # value-loss (red)
C_SOBOLEV = "#1a9850"  # Sobolev (green)


# --------------------------------------------------------------------------- #
# Data loading — the ONE place that knows the on-disk layout / schema.
# --------------------------------------------------------------------------- #
@dataclass
class PaperRun:
    """All emulator-free artifacts of one run, loaded into memory.

    Attributes are plain pandas/dict objects, so figures never touch the disk.
    """
    data_dir: Path
    z: float
    # {(loss, param): sidecar DataFrame};  loss in {"value","sobolev"}
    sidecars: dict = field(default_factory=dict)
    # budget-control ns sidecar (value @ maxsize=35), or None
    budget_ns: pd.DataFrame | None = None
    seed_band: dict | None = None          # seed_band_summary.json
    maxsize: pd.DataFrame | None = None     # maxsize_sensitivity.csv
    multid: pd.DataFrame | None = None      # multid_bestworst.csv
    crossz: dict = field(default_factory=dict)  # {z: {param: sidecar DataFrame}}

    def sidecar(self, loss: str, param: str) -> pd.DataFrame | None:
        return self.sidecars.get((loss, param))


def _zdir(base: Path, sub: str, z: float) -> Path:
    return base / sub / "refit" / f"z{z}"


def load_run(
    data_dir: str | Path = DEFAULT_RUN_DIR,
    z: float = 3.6,
    *,
    value_sub: str = "value",
    sobolev_sub: str = "sobolev",
    budget_sub: str = "budget35_value",
    crossz: tuple[float, ...] = (2.6, 3.6, 4.2),
    figures_sub: str = "figures",
) -> PaperRun:
    """Load a run's emulator-free artifacts. All sub-paths are overridable.

    Missing files are skipped with a warning (the run object just has ``None``
    for that piece), so a partial run still loads.
    """
    base = Path(data_dir)
    run = PaperRun(data_dir=base, z=z)

    for loss, sub in (("value", value_sub), ("sobolev", sobolev_sub)):
        d = _zdir(base, sub, z)
        for p in PARAM_NAMES:
            f = d / f"grad_faith_{p}.csv"
            if f.exists():
                run.sidecars[(loss, p)] = read_grad_faith_sidecar(f)

    bud = _zdir(base, budget_sub, z) / "grad_faith_ns.csv"
    if bud.exists():
        run.budget_ns = read_grad_faith_sidecar(bud)

    sb = base / "seed_band" / "seed_band_summary.json"
    if sb.exists():
        run.seed_band = json.loads(sb.read_text())

    ms = base / figures_sub / "maxsize_sensitivity.csv"
    if ms.exists():
        run.maxsize = pd.read_csv(ms)

    md = base / figures_sub / f"multid_z{z}" / "multid_bestworst.csv"
    if md.exists():
        run.multid = pd.read_csv(md)

    for cz in crossz:
        d = _zdir(base, sobolev_sub, cz)
        got = {p: read_grad_faith_sidecar(d / f"grad_faith_{p}.csv")
               for p in PARAM_NAMES if (d / f"grad_faith_{p}.csv").exists()}
        if got:
            run.crossz[cz] = got

    if not run.sidecars:
        warnings.warn(f"No grad_faith sidecars found under {base} (z={z}); "
                      "figures/tables will be empty. Check data_dir / z.")
    return run


# --------------------------------------------------------------------------- #
# Style — the paper look, fully overridable.
# --------------------------------------------------------------------------- #
def _usetex_ok() -> bool:
    import shutil
    return shutil.which("latex") is not None and shutil.which("dvipng") is not None


@contextmanager
def paper_style(usetex: bool | None = None, scale: float = 1.0, **overrides):
    """Context manager applying the paper's large-label style.

    usetex=None auto-detects a LaTeX install (falls back to mathtext). ``scale``
    multiplies every font size. Any matplotlib rcParam can be overridden via
    kwargs, e.g. ``paper_style(scale=1.2, **{'axes.grid': False})``.
    """
    import matplotlib.pyplot as plt
    if usetex is None:
        usetex = _usetex_ok()
    rc = {
        "text.usetex": usetex,
        "font.family": "serif",
        "font.size": 18 * scale,
        "axes.titlesize": 20 * scale,
        "axes.labelsize": 22 * scale,
        "xtick.labelsize": 17 * scale,
        "ytick.labelsize": 17 * scale,
        "legend.fontsize": 16 * scale,
        "savefig.dpi": 150,
    }
    if usetex:
        rc["text.latex.preamble"] = r"\usepackage{amsmath}\usepackage{amssymb}"
    rc.update(overrides)
    with plt.rc_context(rc):
        yield


# --------------------------------------------------------------------------- #
# Tables.
# --------------------------------------------------------------------------- #
def _knee_grad_err(run: PaperRun, loss: str, param: str) -> float:
    df = run.sidecar(loss, param)
    if df is None or df.empty:
        return np.nan
    return float(knee_row(df)["grad_err"])


def classify(sobolev_grad_err: float, gate: float = GATE) -> str:
    """Faithful / above-gate / resistant from a Sobolev knee grad_err."""
    if not np.isfinite(sobolev_grad_err):
        return "n/a"
    if sobolev_grad_err <= gate:
        return "faithful"
    return "resistant" if sobolev_grad_err > 0.6 else "above-gate"


def taxonomy(run: PaperRun, gate: float = GATE) -> pd.DataFrame:
    """Table 6: value vs Sobolev knee grad_err per parameter + class."""
    rows = []
    for p in PARAM_NAMES:
        s = _knee_grad_err(run, "sobolev", p)
        rows.append({
            "param": p,
            "value_grad_err": _knee_grad_err(run, "value", p),
            "sobolev_grad_err": s,
            "class": classify(s, gate),
        })
    return pd.DataFrame(rows)


def equations(run: PaperRun, loss: str = "sobolev") -> pd.DataFrame:
    """Table 7: the knee-selected candidate per parameter (complexity/loss/grad_err).

    The equation string itself lives in the pareto_<param>.csv next to the
    sidecar; this returns the knee row's metrics (join on the pareto CSV for the
    Equation text — see load path in load_run's docstring).
    """
    rows = []
    for p in PARAM_NAMES:
        df = run.sidecar(loss, p)
        if df is None or df.empty:
            continue
        r = knee_row(df)
        rows.append({"param": p, "complexity": int(r["Complexity"]),
                     "loss": float(r["Loss"]), "grad_err": float(r["grad_err"])})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# Figures.  Single-panel ones return a Figure; the 11-panel grid writes a file.
# --------------------------------------------------------------------------- #
def plot_scorecard(run: PaperRun, *, gate: float = GATE, pretty: dict | None = None,
                   figsize=(11, 4.4)):
    """Value vs Sobolev knee grad_err, one point per parameter (sorted). Returns Figure."""
    import matplotlib.pyplot as plt
    pretty = pretty or PRETTY
    tax = taxonomy(run, gate).dropna(subset=["sobolev_grad_err"]).sort_values("sobolev_grad_err")
    x = np.arange(len(tax))
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for xi, v, s in zip(x, tax["value_grad_err"], tax["sobolev_grad_err"]):
        ax.plot([xi, xi], [min(v, 1.05), min(s, 1.05)], color="0.8", lw=1, zorder=1)
    ax.scatter(x, np.clip(tax["value_grad_err"], 0, 1.05), s=90, marker="o",
               facecolor=C_VALUE, edgecolor="k", lw=.5, zorder=3, label="value")
    ax.scatter(x, np.clip(tax["sobolev_grad_err"], 0, 1.05), s=90, marker="s",
               facecolor=C_SOBOLEV, edgecolor="k", lw=.5, zorder=3, label="Sobolev")
    ax.axhline(gate, color="k", ls="--", lw=1.2)
    ax.text(0.2, gate + 0.02, f"gate {gate}")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(p, p) for p in tax["param"]], rotation=45, ha="right")
    ax.set_ylabel(r"knee $\mathrm{grad\_err}$ (clipped 1.05)")
    ax.legend(loc="upper left")
    return fig


def plot_maxsize_sensitivity(run: PaperRun, *, highlight=("ns", "omegamh2"),
                             gate: float = GATE, pretty: dict | None = None,
                             figsize=(13, 5.2)):
    """grad_err vs maxsize budget, value (left) vs Sobolev (right). Returns Figure."""
    import matplotlib.pyplot as plt
    if run.maxsize is None:
        raise ValueError("run.maxsize is None (maxsize_sensitivity.csv not found).")
    pretty = pretty or PRETTY
    tab = run.maxsize
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=True, layout="constrained")
    for ax, loss in zip(axes, ("value", "sobolev")):
        sub = tab[tab["loss"] == loss]
        for p in PARAM_NAMES:
            s = sub[sub["param"] == p].dropna(subset=["grad_err"]).sort_values("maxsize")
            if s.empty:
                continue
            hot = p in highlight
            ax.plot(s["maxsize"], np.clip(s["grad_err"], 0, 1.2),
                    marker="o" if hot else ".", lw=3 if hot else 1,
                    alpha=1.0 if hot else 0.4, label=pretty.get(p, p) if hot else None,
                    zorder=5 if hot else 2)
        ax.axhline(gate, color="k", ls="--", lw=1.3)
        ax.set_xlabel(r"$\mathrm{maxsize}$ (complexity budget)")
        ax.set_title(f"{loss} loss")
        ax.set_yscale("log")
        ax.grid(alpha=.25, which="both")
    axes[0].set_ylabel(r"knee $\mathrm{grad\_err}$")
    axes[1].legend(loc="upper right", title="highlighted")
    return fig


def plot_seed_band(run: PaperRun, *, params=None, gate: float = GATE,
                   pretty: dict | None = None, figsize=(12, 5)):
    """Across-seed median grad_err with [min,max] whiskers, value vs Sobolev. Returns Figure."""
    import matplotlib.pyplot as plt
    if run.seed_band is None:
        raise ValueError("run.seed_band is None (seed_band_summary.json not found).")
    pretty = pretty or PRETTY
    P = run.seed_band["params"]
    params = params or [p for p in PARAM_NAMES if p in P]
    x = np.arange(len(params))
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for off, key, c, mk in ((-0.15, "value", C_VALUE, "o"), (0.15, "sobolev", C_SOBOLEV, "s")):
        med = np.array([P[p][key][0] for p in params], float)
        lo = np.array([P[p][key][1] for p in params], float)
        hi = np.array([P[p][key][2] for p in params], float)
        ax.errorbar(x + off, np.clip(med, 0, 1.2),
                    yerr=[np.clip(med, 0, 1.2) - np.clip(lo, 0, 1.2),
                          np.clip(hi, 0, 1.2) - np.clip(med, 0, 1.2)],
                    fmt=mk, color=c, ms=8, capsize=3, lw=1.2, label=key)
    ax.axhline(gate, color="k", ls="--", lw=1.2)
    ax.text(0, gate + 0.02, f"gate {gate}")
    ax.set_xticks(x)
    ax.set_xticklabels([pretty.get(p, p) for p in params], rotation=45, ha="right")
    ax.set_ylabel(r"knee $\mathrm{grad\_err}$ (median, [min,max])")
    ax.legend(loc="upper left")
    return fig


def plot_ns_budget(run: PaperRun, *, gate: float = GATE, figsize=(8, 5.4)):
    """The n_S panel: value_mse vs complexity, coloured by grad_err, for the
    value/Sobolev/deep-budget fronts. Shows the Mirage + the budget-vs-objective
    story for n_S. Returns Figure. Needs the pareto+sidecar CSVs + the budget arm."""
    import matplotlib.colors as mcolors
    import matplotlib.pyplot as plt
    from priya_forecast.pareto_diag import load_front
    d_val, d_sob = _zdir(run.data_dir, "value", run.z), _zdir(run.data_dir, "sobolev", run.z)
    series = [("value", load_front(d_val / "pareto_ns.csv", d_val / "grad_faith_ns.csv"), "o"),
              ("Sobolev", load_front(d_sob / "pareto_ns.csv", d_sob / "grad_faith_ns.csv"), "s")]
    if run.budget_ns is not None:
        bp = _zdir(run.data_dir, "budget35_value", run.z) / "pareto_ns.csv"
        if bp.exists():
            series.append(("value@budget", load_front(bp, None).merge(
                run.budget_ns[["Complexity", "grad_err", "value_mse"]], on="Complexity",
                how="left", suffixes=("", "_y")), "^"))
    cmap = mcolors.ListedColormap(["#1a9850", "#d6604d"])
    norm = mcolors.BoundaryNorm([0.0, gate + 1e-12, 1.0], cmap.N)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for lab, df, mk in series:
        ax.scatter(df["Complexity"], df["value_mse"], c=np.clip(df["grad_err"], 0, 1),
                   cmap=cmap, norm=norm, marker=mk, s=72, edgecolor="k", lw=.5, label=lab)
    ax.set_yscale("log")
    ax.set_xlabel("complexity")
    ax.set_ylabel(r"value MSE vs GP ($\log P$, HF)")
    ax.legend(loc="upper right")
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), ax=ax, ticks=[gate])
    cb.set_label(rf"$\mathrm{{grad\_err}}$: green $\leq {gate}$, red $> {gate}$")
    return fig


def plot_crossz(run: PaperRun, *, params=None, gate: float = GATE,
                pretty: dict | None = None, figsize=(9, 5.4)):
    """Sobolev knee grad_err vs redshift, one line per parameter. Returns Figure."""
    import matplotlib.pyplot as plt
    if not run.crossz:
        raise ValueError("run.crossz empty (cross-z sidecar dirs not found).")
    pretty = pretty or PRETTY
    params = params or PARAM_NAMES
    zs = sorted(run.crossz)
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    for p in params:
        ge = [float(knee_row(run.crossz[z][p])["grad_err"])
              if p in run.crossz.get(z, {}) else np.nan for z in zs]
        if np.all(np.isnan(ge)):
            continue
        ax.plot(zs, np.clip(ge, 0, 1.2), marker="o", label=pretty.get(p, p))
    ax.axhline(gate, color="k", ls="--", lw=1.2)
    ax.set_xlabel(r"redshift $z$")
    ax.set_ylabel(r"Sobolev knee $\mathrm{grad\_err}$")
    ax.set_yscale("log")
    ax.legend(ncol=2, fontsize="small")
    return fig


def plot_multid(run: PaperRun, *, figsize=(10, 5)):
    """Multi-D combine mean/p90/max relative error per combo (2D + 3D). Returns Figure."""
    import matplotlib.pyplot as plt
    if run.multid is None:
        raise ValueError("run.multid is None (multid_bestworst.csv not found).")
    df = run.multid.sort_values(["dim", "mean_pct"])
    x = np.arange(len(df))
    fig, ax = plt.subplots(figsize=figsize, layout="constrained")
    ax.bar(x, df["mean_pct"], color=["#4575b4" if d == "2D" else "#d73027" for d in df["dim"]])
    ax.errorbar(x, df["mean_pct"], yerr=[np.zeros(len(df)), df["max_pct"] - df["mean_pct"]],
                fmt="none", ecolor="0.3", capsize=3, lw=1)
    ax.set_xticks(x)
    ax.set_xticklabels(df["params"], rotation=45, ha="right", fontsize="small")
    ax.set_ylabel(r"combine vs GP rel.\ error [\%] (mean, max)")
    ax.set_title("Multi-D combine accuracy (blue=2D, red=3D)")
    return fig


def plot_pareto_faithfulness(run: PaperRun, out_path, *, gate: float = GATE, **kw):
    """The 11-panel Pareto-faithfulness grid. Delegates to the shared renderer,
    which WRITES ``out_path`` (PDF or PNG) at dpi 150 (it does not return a Figure).
    Pass ``annotate=`` / ``y_col=`` through to ``render_grid`` to tweak."""
    from priya_forecast.pareto_diag import load_front, render_grid
    base = run.data_dir
    fronts = {}
    for p in PARAM_NAMES:
        rows = []
        for loss, sub, lab, mk in (("value", "value", "value", "o"),
                                   ("sobolev", "sobolev", "Sobolev", "s")):
            d = _zdir(base, sub, run.z)
            fp, gp = d / f"pareto_{p}.csv", d / f"grad_faith_{p}.csv"
            if fp.exists() and gp.exists():
                rows.append({"front": load_front(fp, gp), "label": lab, "marker": mk})
        if rows:
            fronts[p] = rows
    render_grid(fronts, out_path, gate_tol=gate, param_order=list(PARAM_NAMES),
                y_col="value_mse",
                y_label=r"value MSE vs GP ($\log P$, HF) -- lower is better", **kw)
    return out_path
