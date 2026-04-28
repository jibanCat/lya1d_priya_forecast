"""Abstract base for P1D forward models.

Both `GPModel` (PRIYA emulator) and `PySRModel` (analytic equations) implement
this. The forecast pipeline only depends on the ABC: swap a model in
`configs/eqns/*.yaml` and the rest of the code is unchanged.

Units: theta = 11D physical parameter vector in `parameters.PARAM_NAMES` order;
k in s/km; z dimensionless; output P_F dimensionless.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class P1DModel(ABC):
    """Forward model for the 1D Lyman-alpha flux power spectrum."""

    @abstractmethod
    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        """Return P_F evaluated at the requested (theta, k, z).

        Parameters
        ----------
        theta : ndarray, shape (11,)
            Physical parameter vector in canonical order
            (`priya_forecast.parameters.PARAM_NAMES`).
        k : ndarray, shape (Nk,)
            Wavenumber grid (s/km), strictly increasing.
        z : float
            Redshift bin (dimensionless).

        Returns
        -------
        ndarray, shape (Nk,)
            Predicted P_F values (dimensionless), one per `k` entry.
        """
