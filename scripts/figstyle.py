"""One figure style for the whole paper (used by scripts/make_figures.py).

Every figure is saved at exactly the width it is printed at (`save(..., width=)`
fits the cropped PDF to that width), so the fonts below are the fonts on the
page. The rules:

* every axis carries its label, with units;
* the same thing gets the same colour and marker in every figure;
* one accent colour for the paper's claim (the Sobolev loss), greys otherwise;
* no in-figure titles and no code names: parameters use the tables' symbols;
* reference lines are named where they are drawn.

Colours are from the Okabe-Ito palette, which stays distinguishable under the
common forms of colour blindness.
"""
from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# openjournal [twocolumn], measured from the class: \columnwidth = 245.27pt,
# \textwidth = 513.12pt, body text 10pt.
COLUMN_IN = 245.26653 / 72.27
TEXT_IN = 513.11743 / 72.27

INK = "#222222"
GREY = "#7a7a7a"
LIGHT = "#d0d0d0"

# Colours follow the paper's captions and prose. Okabe-Ito shades of the named colours.
GREEN = "#009E73"
RED = "#D6604D"     # main's red: Okabe-Ito vermilion read as orange next to the LF orange

# Fidelity (fig:tau0_ap_pred, fig:param_error_bars): "low-fidelity in orange and
# high-fidelity in green"; fig:tau0_ap_pred's residuals "in brown" and "in pink".
LF = "#E69F00"   # orange: low fidelity
HF = GREEN       # green: high fidelity
LF_RESID = "#8C510A"   # brown
HF_RESID = "#CC79A7"   # pink
M_LF, M_HF = "o", "D"

# Training loss (fig:seed_band): "value loss (red dots)", "Sobolev loss (green squares)".
VALUE = RED
SOBOLEV = GREEN
M_VALUE, M_SOBOLEV, M_BUDGET = "o", "s", "^"

# Derivative check (fig:pareto_faith, fig:ns_budget): "green faithful, red unfaithful",
# and "a bold ring marks equations clearing the 0.25 gate". Marker shape gives the search.
PASS, FAIL = GREEN, RED
SEARCH = "#9a9a9a"   # neutral fill for the marker-shape legend entries

THRESHOLD = 0.25   # the derivative-error threshold of Eq. (9)
# The derivative error's symbol; the paper defines the same one as \graderr.
DERIV_ERR = r"$E_\partial$"

# Parameter symbols, matching Tables 1-5.
SYMBOL = {
    "dtau0": r"$d\tau_0$", "tau0": r"$\tau_0$", "ns": r"$n_P$", "Ap": r"$A_P$",
    "herei": r"$z_{Hei}$", "heref": r"$z_{Hef}$", "alphaq": r"$\alpha_q$",
    "hub": r"$h$", "omegamh2": r"$\Omega_M h^2$", "hireionz": r"$z_{Hi}$",
    "bhfeedback": r"$\epsilon_{AGN}$",
}
PARAM_ORDER = ["dtau0", "tau0", "ns", "Ap", "herei", "heref", "alphaq",
               "hub", "omegamh2", "hireionz", "bhfeedback"]
# tab:faith_taxonomy's row order (the nine that pass, then the two that fail);
# fig:seed_band follows it.
TABLE4_ORDER = ["dtau0", "tau0", "ns", "Ap", "herei", "heref", "alphaq",
                "omegamh2", "hireionz", "hub", "bhfeedback"]

K_LABEL = r"$k$ [s\,km$^{-1}$]"


def use() -> None:
    plt.rcParams.update({
        "text.usetex": True,
        "text.latex.preamble": r"\usepackage{amsmath}",
        "font.family": "serif",
        "font.size": 8,
        "axes.labelsize": 8.5,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "legend.frameon": False,
        "legend.handlelength": 1.6,
        "legend.borderaxespad": 0.3,
        "axes.linewidth": 0.6,
        "axes.edgecolor": INK,
        "axes.labelcolor": INK,
        "text.color": INK,
        "xtick.color": INK,
        "ytick.color": INK,
        "xtick.major.width": 0.6,
        "ytick.major.width": 0.6,
        "xtick.minor.width": 0.4,
        "ytick.minor.width": 0.4,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "lines.linewidth": 1.2,
        "lines.markersize": 4.5,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
    })


PAD_IN = 0.02
PNG_PREVIEW = False   # also write <name>.png next to each PDF (for notebooks and READMEs)


def save(fig, path, width: float) -> None:
    """Save cropped to the content, with the cropped width equal to `width`
    inches (the printed width), so nothing is rescaled when LaTeX includes it.

    Fonts are fixed in points, so only the plotting area changes while the
    figure width is adjusted. No CreationDate: a rebuild is byte-identical.
    """
    for _ in range(4):
        fig.canvas.draw()
        bbox = fig.get_tightbbox(fig.canvas.get_renderer())
        delta = width - (bbox.width + 2 * PAD_IN)
        if abs(delta) < 0.005:
            break
        w, h = fig.get_size_inches()
        fig.set_size_inches(w + delta, h)
    fig.savefig(path, bbox_inches="tight", pad_inches=PAD_IN,
                metadata={"CreationDate": None})
    if PNG_PREVIEW:
        fig.savefig(str(path)[:-4] + ".png", dpi=200, bbox_inches="tight", pad_inches=PAD_IN)
    plt.close(fig)


def pass_style(passes: bool) -> dict:
    """Green with a bold dark ring when the derivative error passes the threshold, red otherwise."""
    if passes:
        return dict(facecolors=PASS, edgecolors=INK, linewidths=1.1)
    return dict(facecolors=FAIL, edgecolors=FAIL, linewidths=0.5)
