#!/usr/bin/env python
"""Draw the paper's figures, exactly as they appear in the manuscript.

This is the script that builds the manuscript's figures. Every input is a
committed file of this repo, and each one is checked against tag v1.2-paper
(commit ce8ee6a) before it is used: if the working copy differs from the tagged
file, the script stops. No GP emulator and no PySR run is needed, so the plotted
values are the published ones.

    fig:tau0_ap_pred       pysr_pred_tau0_Ap.pdf     sobolev/refit/z3.6/{refits,payloads}/*.pkl
    fig:param_error_bars   param_error_bars.pdf      figures/table2_stats.tex
    fig:seed_band          seed_band.pdf             seed_band/seed_band_summary.json
    fig:pareto_faith       pareto_faithfulness.pdf   {value,sobolev}/refit/z3.6/grad_faith_*.csv
                                                     + seed_band/z3.6_seed0_budget (n_P)
    fig:ns_budget          ns_budget_panel.pdf       the three n_P files above

Which equation each figure shows (Sec. 2.2.2 of the paper):
  * fig:tau0_ap_pred and fig:param_error_bars use the lowest-loss equation of each fit;
  * fig:seed_band uses the Pareto-knee equation (the simplest valid equation whose loss
    is within 10% of the best valid loss; priya_forecast.grad_faith_io.knee_row);
  * fig:pareto_faith and fig:ns_budget show every valid candidate, with no selection.
The derivative-error threshold drawn in the last three is 0.25 (figstyle.THRESHOLD).

Usage (needs requirements-figures.txt, git, and a TeX installation with `latex` on PATH):
    python scripts/make_figures.py [--only seed_band ...] [--out DIR] [--dump DIR]

--dump DIR writes the exact values each figure plots to DIR/<figure>.json.
"""
from __future__ import annotations

import argparse
import json
import pickle
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import figstyle as S  # noqa: E402

CODE = HERE.parent          # this repo
ANCHOR = "v1.2-paper"
PROD_REL = "results/paper_production_20260630_perz_sobolev_z2.6-4.2"
PROD = CODE / PROD_REL
Z = 3.6
sys.path.insert(0, str(CODE / "src"))


def at_anchor(path: Path) -> Path:
    """Return `path` after checking it is byte-identical to the tagged blob."""
    rel = path.relative_to(CODE).as_posix()
    tagged = subprocess.run(["git", "-C", str(CODE), "rev-parse", f"{ANCHOR}:{rel}"],
                            capture_output=True, text=True)
    if tagged.returncode != 0:
        raise SystemExit(f"{rel} is not in {ANCHOR}; refusing to plot it.")
    here = subprocess.run(["git", "-C", str(CODE), "hash-object", str(path)],
                          capture_output=True, text=True, check=True).stdout.strip()
    if here != tagged.stdout.strip():
        raise SystemExit(f"{rel} differs from {ANCHOR}; refusing to plot it.")
    return path


def parse_stats(path: Path) -> dict:
    """Parse table2_stats.tex: {param: (lf_mean, lf_max, hf_mean, hf_max)} in percent."""
    import re
    row = re.compile(r'^(\w+)\s+LF:\s*([\d.]+)%\s*\(max\s*([\d.]+)%\)\s*'
                     r'HF:\s*([\d.]+)%\s*\(max\s*([\d.]+)%\)')
    out = {}
    for line in path.read_text().splitlines():
        if line.startswith('--- LaTeX'):
            break
        m = row.match(line.strip())
        if m:
            out[m.group(1)] = tuple(float(m.group(i)) for i in (2, 3, 4, 5))
    if len(out) != 11:
        raise SystemExit(f"expected 11 parameters in {path}, parsed {len(out)}")
    return out


def zdir(sub: str) -> Path:
    return PROD / sub / "refit" / f"z{Z}"


def load_pickle(path: Path):
    with open(at_anchor(path), "rb") as fh:
        return pickle.load(fh)


def sidecar(path: Path):
    from priya_forecast.grad_faith_io import read_grad_faith_sidecar
    return read_grad_faith_sidecar(at_anchor(path))


# --------------------------------------------------------------------------- #
# fig:tau0_ap_pred: equation against GP for tau0 and A_P
# --------------------------------------------------------------------------- #
def fig_prediction(out: Path, dump: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    LF_RES, HF_RES = 0.4, 0.8
    col = {"tau0": 1, "Ap": 3}
    cases = [("tau0", 1.0, 1.09, "$\\tau_0 = {:.3f}$\n(fiducial 1.090)", 1.0),
             ("Ap", 1.8, 1.46,
              "$A_P = {:.2f}\\times10^{{-9}}$\n(fiducial $1.46\\times10^{{-9}}$)", 1.0)]
    fig = plt.figure(figsize=(S.TEXT_IN, 3.25))
    gs = fig.add_gridspec(2, 2, height_ratios=[2.3, 1], hspace=0.06, wspace=0.24)
    for j, (p, target, _fid, fmt, _) in enumerate(cases):
        pld = load_pickle(zdir("sobolev") / "payloads" / f"{p}.pkl")
        rfit = load_pickle(zdir("sobolev") / "refits" / f"{p}.pkl")
        pl = pld["payload"] if "payload" in pld else pld
        i = int(np.argmin(np.abs(pl["params_lf"][:, col[p]] - target)))
        theta = float(pl["params_lf"][i, col[p]])
        k_lf, k_hf = pl["kfkms_lf_z"][i], pl["kfkms_hf_z"][i]
        gp_lf, gp_hf = pl["flux_lf_z"][i], pl["flux_hf_z"][i]
        eq_lf = rfit.predict(theta, k_lf, resolution=LF_RES, z=Z)
        eq_hf = rfit.predict(theta, k_hf, resolution=HF_RES, z=Z)

        top = fig.add_subplot(gs[0, j])
        bot = fig.add_subplot(gs[1, j], sharex=top)
        for k, gp, c in ((k_lf, gp_lf, S.LF), (k_hf, gp_hf, S.HF)):
            top.plot(k, k * gp / np.pi, color=c, lw=2.2, alpha=0.45, zorder=1)
        for k, gp, eq, c, cr in ((k_lf, gp_lf, eq_lf, S.LF, S.LF_RESID),
                                 (k_hf, gp_hf, eq_hf, S.HF, S.HF_RESID)):
            top.plot(k, k * eq / np.pi, color=c, lw=1.1, ls=(0, (3.5, 2)), zorder=2)
            bot.plot(k, 100 * (1 - eq / gp), color=cr, lw=1.1)
        bot.axhline(0, color=S.GREY, lw=0.6, ls=":")
        top.set_xscale("log")
        top.tick_params(labelbottom=False)
        top.text(0.03, 0.95, fmt.format(theta), transform=top.transAxes,
                 va="top", ha="left", linespacing=1.3)
        top.set_ylabel(r"$k\,P_F(k)/\pi$")
        bot.set_ylabel(r"residual [\%]")
        bot.set_xlabel(S.K_LABEL)
        bot.set_xlim(k_lf.min() * 0.95, k_lf.max() * 1.05)
        bot.set_ylim(-1.5, 3.0)        # same range in both panels, so they compare at a glance
        dump[p] = dict(theta=theta, k_lf=k_lf.tolist(), k_hf=k_hf.tolist(),
                       gp_lf=gp_lf.tolist(), gp_hf=gp_hf.tolist(),
                       eq_lf=np.asarray(eq_lf).tolist(), eq_hf=np.asarray(eq_hf).tolist())
        if j == 0:
            handles = [Line2D([], [], color=S.INK, lw=2.2, alpha=0.45,
                              label="GP emulator (wide, pale)"),
                       Line2D([], [], color=S.INK, lw=1.1, ls=(0, (3.5, 2)),
                              label="symbolic equation (dashed)"),
                       Line2D([], [], color=S.LF, lw=1.3, label="low fidelity (LF)"),
                       Line2D([], [], color=S.HF, lw=1.3, label="high fidelity (HF)")]
            top.legend(handles=handles, loc="lower right")
            bot.legend(handles=[Line2D([], [], color=S.LF_RESID, lw=1.3, label="LF residual"),
                                Line2D([], [], color=S.HF_RESID, lw=1.3, label="HF residual")],
                       loc="upper left", ncol=2, fontsize=6.5, handlelength=1.4,
                       columnspacing=0.8, borderaxespad=0.2)
    S.save(fig, out / "pysr_pred_tau0_Ap.pdf", width=0.95 * S.TEXT_IN)


# --------------------------------------------------------------------------- #
# fig:param_error_bars: per-parameter accuracy of the 1D equations (tab:stats_table)
# --------------------------------------------------------------------------- #
def fig_param_errors(out: Path, dump: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    stats = parse_stats(at_anchor(PROD / "figures" / "table2_stats.tex"))
    # As the caption says: bars give the mean, whiskers the maximum, LF orange and HF
    # green, rows by descending HF mean error from the top.
    from matplotlib.patches import Patch
    order = sorted(stats, key=lambda p: stats[p][2])          # HF mean, ascending up the axis
    fig, ax = plt.subplots(figsize=(S.COLUMN_IN, 3.3))
    # bars start at the axis edge
    x0 = 10.0 ** np.floor(np.log10(min(min(v) for v in stats.values())))
    for y, p in enumerate(order):
        lf_mean, lf_max, hf_mean, hf_max = stats[p]
        for dy, mean, mx, c in ((0.18, lf_mean, lf_max, S.LF), (-0.18, hf_mean, hf_max, S.HF)):
            ax.barh(y + dy, mean - x0, left=x0, height=0.34, color=c, lw=0)
            ax.plot([mean, mx], [y + dy] * 2, color=S.INK, lw=0.8)
            ax.plot([mx], [y + dy], marker="|", color=S.INK, ms=4.5, mew=0.9)
        dump[p] = dict(lf_mean=lf_mean, lf_max=lf_max, hf_mean=hf_mean, hf_max=hf_max)
    ax.axvline(1.0, color=S.GREY, lw=0.7, ls="--")
    ax.text(1.12, -0.75, r"1\%", color=S.GREY, va="center", fontsize=7)
    ax.set_xscale("log")
    ax.set_xlim(x0, None)
    ax.set_yticks(range(len(order)))
    ax.set_yticklabels([S.SYMBOL[p] for p in order])
    ax.tick_params(axis="y", which="both", right=False, left=False)
    ax.set_ylim(-1.05, len(order) - 0.2)
    ax.set_xlabel(r"relative error of the equation [\%]")
    handles = [Patch(color=S.LF, label="low fidelity (LF), mean"),
               Patch(color=S.HF, label="high fidelity (HF), mean"),
               Line2D([], [], color=S.INK, marker="|", lw=0.8, ms=4.5, mew=0.9,
                      label="whisker: maximum")]
    ax.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2,
              columnspacing=1.0, handletextpad=0.4)
    S.save(fig, out / "param_error_bars.pdf", width=S.COLUMN_IN)


# --------------------------------------------------------------------------- #
# fig:seed_band: derivative error across five random seeds
# --------------------------------------------------------------------------- #
def fig_seed_band(out: Path, dump: dict) -> None:
    import matplotlib.pyplot as plt

    sb = json.loads(at_anchor(PROD / "seed_band" / "seed_band_summary.json").read_text())
    P = sb["params"]
    from matplotlib.lines import Line2D
    params = [p for p in S.TABLE4_ORDER if p in P]
    x = np.arange(len(params))
    fig, ax = plt.subplots(figsize=(S.TEXT_IN, 2.45))
    ax.axhspan(-1, S.THRESHOLD, color=S.LIGHT, alpha=0.3, lw=0)
    ax.axhline(S.THRESHOLD, color=S.INK, lw=0.8, ls="--")
    ax.text(len(params) - 0.5, S.THRESHOLD + 0.025, r"threshold 0.25", ha="right",
            va="bottom", fontsize=7.5)
    ax.text(len(params) - 0.5, S.THRESHOLD - 0.025, "passes", ha="right", va="top",
            fontsize=7, color=S.GREY)
    for off, key, c, m in ((-0.14, "value", S.VALUE, S.M_VALUE),
                           (0.14, "sobolev", S.SOBOLEV, S.M_SOBOLEV)):
        med, lo, hi = (np.array([P[p][key][i] for p in params], float) for i in range(3))
        ax.errorbar(x + off, med, yerr=[med - lo, hi - med], fmt="none", ecolor=c,
                    capsize=2, elinewidth=0.9, zorder=2)
        ax.scatter(x + off, med, marker=m, s=26, zorder=3, color=c, lw=0)
        dump[key] = {p: P[p][key] for p in params}
    ax.set_xticks(x)
    ax.set_xticklabels([S.SYMBOL[p] for p in params], fontsize=8.5)
    ax.tick_params(axis="x", which="both", top=False)
    ax.set_xlim(-0.6, len(params) - 0.4)
    top = max(max(P[p][k][2] for p in params) for k in ("value", "sobolev"))
    ax.set_ylim(-0.04, max(1.05, 1.08 * top))
    ax.set_ylabel(f"derivative error {S.DERIV_ERR}")

    def h(color, marker, filled, label):
        return Line2D([], [], lw=0, marker=marker, ms=5, mec=color, mew=0.8,
                      mfc=color if filled else "white", label=label)
    ax.legend(handles=[h(S.VALUE, S.M_VALUE, True, "value-only loss"),
                       h(S.SOBOLEV, S.M_SOBOLEV, True, "Sobolev loss")],
              loc="upper left", ncol=2, columnspacing=1.2,
              title=r"median and range over five random seeds", title_fontsize=7.5)
    S.save(fig, out / "seed_band.pdf", width=S.TEXT_IN)


# --------------------------------------------------------------------------- #
# fig:pareto_faith and fig:ns_budget: candidate equations by complexity and value error
# --------------------------------------------------------------------------- #
# The encoding the captions and Sec. 4.2 describe: colour and the bold ring give the
# derivative check, marker shape gives the search. No selection is marked: every
# candidate is drawn the same way.
def _scatter_front(ax, df, marker, size=16):
    ge = df["grad_err"].to_numpy(float)
    for passes in (False, True):
        sel = (ge <= S.THRESHOLD) if passes else (ge > S.THRESHOLD)
        if sel.any():
            ax.scatter(df["Complexity"].to_numpy(float)[sel], df["value_mse"].to_numpy(float)[sel],
                       marker=marker, s=size, zorder=3 if passes else 2, **S.pass_style(passes))


def _front_series(p):
    series = [("value", sidecar(zdir("value") / f"grad_faith_{p}.csv"), S.M_VALUE),
              ("sobolev", sidecar(zdir("sobolev") / f"grad_faith_{p}.csv"), S.M_SOBOLEV)]
    if p == "ns":
        series.append(("value_maxsize35",
                       sidecar(zdir("seed_band/z3.6_seed0_budget") / "grad_faith_ns.csv"),
                       S.M_BUDGET))
    return series


def _legend_handles(with_budget: bool, short: bool = False):
    from matplotlib.lines import Line2D

    def h(marker, label, fc=S.SEARCH, ec=S.SEARCH, mew=0.5):
        return Line2D([], [], lw=0, marker=marker, ms=5, mfc=fc, mec=ec, mew=mew, label=label)
    sep = ", " if short else ",\n"
    hs = [h(S.M_VALUE, f"value-only loss{sep}complexity limit 20"),
          h(S.M_SOBOLEV, f"Sobolev loss{sep}complexity limit 20")]
    if with_budget:
        hs.append(h(S.M_BUDGET, f"value-only loss{sep}complexity limit 35"
                    + ("" if short else f" ({S.SYMBOL['ns']} only)")))
    br = " " if short else "\n"
    hs += [h("o", f"green, bold ring:{br}derivative error $\\le 0.25$", S.PASS, S.INK, 1.1),
           h("o", f"red:{br}derivative error $> 0.25$", S.FAIL, S.FAIL)]
    return hs


VALUE_ERR_LABEL = r"value error (MSE of $\ln P_F$ vs GP, HF)"


def _log_y(ax):
    """Log y-axis with two or three labelled ticks on every panel.

    matplotlib's LogLocator thins its ticks on small panels, leaving some with one
    label and narrow-range ones crowded, so ticks are picked explicitly: the
    decades in range, topped up from 1, 2, 3 and 5 times a decade.
    """
    from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator
    ax.set_yscale("log")
    lo, hi = ax.get_ylim()
    e0, e1 = int(np.floor(np.log10(lo))), int(np.ceil(np.log10(hi)))
    decades = [10.0 ** e for e in range(e0, e1 + 1) if lo <= 10.0 ** e <= hi]
    ticks = decades
    if len(ticks) < 2:
        nice = sorted(m * 10.0 ** e for e in range(e0, e1 + 1) for m in (1, 2, 3, 5)
                      if lo <= m * 10.0 ** e <= hi)
        pick = np.unique(np.linspace(0, len(nice) - 1, min(3, len(nice))).round().astype(int))
        ticks = [nice[i] for i in pick]
    elif len(ticks) > 4:
        ticks = ticks[::2]

    def label(v, _pos):
        e = int(np.floor(np.log10(v) + 1e-9))
        m = v / 10.0 ** e
        return rf"$10^{{{e}}}$" if abs(m - 1) < 1e-6 else rf"${m:g}\times10^{{{e}}}$"
    ax.yaxis.set_major_locator(FixedLocator(ticks))
    ax.yaxis.set_major_formatter(FuncFormatter(label))
    ax.yaxis.set_minor_formatter(NullFormatter())
    if len(decades) < 2:
        ax.yaxis.set_minor_locator(NullLocator())


def fig_pareto(out: Path, dump: dict) -> None:
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(3, 4, figsize=(S.TEXT_IN, 4.6))
    n = 0
    for ax, p in zip(axes.flat, S.PARAM_ORDER):
        dump[p] = {}
        for name, df, m in _front_series(p):
            _scatter_front(ax, df, m)
            dump[p][name] = df[["Complexity", "value_mse", "grad_err"]].to_dict("list")
            n += len(df)
        _log_y(ax)
        ax.set_title(S.SYMBOL[p], loc="left", fontsize=9.5, pad=3)
        ax.tick_params(labelsize=7)
    dump["_n_plotted"] = n
    legend_ax = axes.flat[-1]
    legend_ax.axis("off")
    legend_ax.legend(handles=_legend_handles(True), loc="center left", bbox_to_anchor=(-0.05, 0.45),
                     fontsize=6.5, labelspacing=0.6, frameon=False,
                     handletextpad=0.4)
    fig.supxlabel("equation complexity (number of nodes)", fontsize=8.5, y=0.015)
    fig.supylabel(VALUE_ERR_LABEL, fontsize=8.5, x=0.012)
    fig.subplots_adjust(wspace=0.55, hspace=0.42, left=0.085, bottom=0.085)
    S.save(fig, out / "pareto_faithfulness.pdf", width=0.95 * S.TEXT_IN)


def fig_ns_budget(out: Path, dump: dict) -> None:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(S.COLUMN_IN, 2.5))
    n = 0
    for name, df, m in _front_series("ns"):
        _scatter_front(ax, df, m, size=20)
        dump[name] = df[["Complexity", "value_mse", "grad_err"]].to_dict("list")
        n += len(df)
    dump["_n_plotted"] = n
    _log_y(ax)
    ax.set_xlabel("equation complexity (number of nodes)")
    ax.set_ylabel(VALUE_ERR_LABEL, fontsize=7.5)
    ax.text(0.04, 0.06, S.SYMBOL["ns"], transform=ax.transAxes, fontsize=9.5)
    ax.legend(handles=_legend_handles(True, short=True), loc="upper center",
              bbox_to_anchor=(0.45, -0.2), ncol=1, fontsize=7, handletextpad=0.3,
              labelspacing=0.35)
    S.save(fig, out / "ns_budget_panel.pdf", width=S.COLUMN_IN)


FIGURES = {
    "prediction": fig_prediction, "param_errors": fig_param_errors,
    "seed_band": fig_seed_band, "pareto": fig_pareto, "ns_budget": fig_ns_budget,
}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", type=Path,
                    default=CODE / "results" / "_repro_scratch" / "paper_figures",
                    help="output directory (default: a git-ignored scratch dir)")
    ap.add_argument("--only", nargs="*", choices=sorted(FIGURES), default=None)
    ap.add_argument("--dump", type=Path, default=None,
                    help="write each figure's plotted values to DIR/<figure>.json")
    ap.add_argument("--png", action="store_true",
                    help="also write a PNG preview next to each PDF")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    if args.dump:
        args.dump.mkdir(parents=True, exist_ok=True)
    S.use()
    S.PNG_PREVIEW = args.png
    for name in args.only or FIGURES:
        dump: dict = {}
        FIGURES[name](args.out, dump)
        if args.dump:
            (args.dump / f"{name}.json").write_text(json.dumps(dump, default=float))
        print(f"{name}: done")


if __name__ == "__main__":
    main()
