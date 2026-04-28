"""PySR-equation P1D model.

Phase 3 will fill this out. Reads a YAML pointing at PySR `hall_of_fame*.csv`
Pareto-front files (one per parameter), picks an equation per a `pick` rule
(`best_loss` | `complexity_le:N` | `accuracy_at:tol` | `row:I`), and combines
them via `multiplicative` | `additive` | `joint`.
"""

# TODO(phase3): PySRModel + sympy-whitelist parser + Pareto CSV loader
