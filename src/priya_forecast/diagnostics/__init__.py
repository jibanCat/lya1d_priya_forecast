"""Diagnostic plots produced during forecast and HPO runs.

Each function takes already-computed data + an output directory and writes
PNG/PDF figures. Pure matplotlib; no library-internal state. Used by:

- the forecast CLI (`priya-forecast run/compare/diagnose/hpo`)
- the test suite (a small sample is regenerated under `docs/figures/` on
  every run, so the README walkthrough stays in sync with the code)
"""

from priya_forecast.diagnostics.forecast_plots import (
    plot_pysr_vs_gp,
    plot_per_parameter_sensitivity,
    plot_fisher_corner,
    plot_fisher_sigma_table,
    plot_residuals_at_fiducial,
)

__all__ = [
    "plot_pysr_vs_gp",
    "plot_per_parameter_sensitivity",
    "plot_fisher_corner",
    "plot_fisher_sigma_table",
    "plot_residuals_at_fiducial",
]
