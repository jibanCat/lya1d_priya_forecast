"""Single-z student-facing forecast pipeline.

Three modes via YAML `mode:` field:

- ``gp_only`` — GP-vs-data Fisher only (no PySR). Useful as a sanity check
  and as the upper-bound reference σ_GP.
- ``forecast_only`` — student supplies (or reuses bundled) PySR Pareto CSVs;
  pipeline picks rows, combines per the YAML, scores against GP.
- ``refit_and_forecast`` — pipeline runs single-z PySR refits per parameter
  with smart-kwargs + ANOVA-loss-OFF defaults, emits CSVs, then scores.

Designed to be controlled entirely from YAML — no code edits, no SLURM,
no payload precompute. See ``scripts/run_pipeline.py``.
"""
