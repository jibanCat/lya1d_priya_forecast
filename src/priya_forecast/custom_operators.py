"""Custom PySR operators and the sympy/lambdify mappings to evaluate them.

Single source of truth so adding an operator is one entry. `aq` is the
analytic quotient x/sqrt(1+y^2) — a bounded, pole-free replacement for raw
division (raw `/` creates poles whose derivatives wreck the Fisher matrix;
see docs/SR_EMULATOR_LITERATURE_NOTES.md). `inv` is the pre-existing 1/x op.
"""
from __future__ import annotations

import numpy as np
import sympy as sp

# Julia definition for PySR `binary_operators`.
AQ_JULIA = "aq(x, y) = x / sqrt(1 + y^2)"

# sympy-backed mappings for PySRRegressor(extra_sympy_mappings=...) — used by
# PySR's own .sympy() expansion.
EXTRA_SYMPY_MAPPINGS = {
    "inv": lambda x: 1 / x,
    "aq": lambda x, y: x / sp.sqrt(1 + y**2),
}

# numpy-backed mappings for sympy.lambdify(..., modules=[LAMBDIFY_MODULES, "numpy"]).
# Threaded into every equation-evaluation site so a raw equation string
# containing inv(...) or aq(...) evaluates and differentiates numerically.
LAMBDIFY_MODULES = {
    "inv": lambda x: 1.0 / x,
    "aq": lambda x, y: x / np.sqrt(1.0 + y**2),
}
