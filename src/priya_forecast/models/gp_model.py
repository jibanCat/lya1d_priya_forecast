"""Adapter around `lyaemu.priya_explorer.PRIYAEmulatorExplorer`.

Two implementations live here:

- ``MockGPModel``: pure numpy. Cheap, deterministic, used in tests so the
  rest of the forecast pipeline (likelihood, Fisher, MCMC) can run without
  installing GPy/emukit or loading real emulator pickles.

- ``GPModel``: the real adapter. Constructs a ``PRIYAEmulatorExplorer`` once
  per forecast run, holds the trained MF emulator in memory, and produces
  P_F predictions at requested ``(theta, k, z)`` by

    1. Padding the 11D physics vector to 15D by appending the four DLA
       template amplitudes (held at zero unless ``a_template`` is passed).
    2. Calling ``explorer.emulator_wrap.get_predicted(theta_15d[:11])``.
    3. Selecting the right z-slice (the explorer returns one prediction per
       redshift) and binning/interpolating onto ``k``.

The default basedir is ``/home/mfho/student_projects/InferenceLyaData/
Emulator_Files`` per the project setup. Override with ``basedir=...``.

We deliberately keep imports of ``priya_explorer`` lazy: tests that don't
need the real emulator never trigger the GPy/emukit dependency.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from priya_forecast.data import bin_model_to_data
from priya_forecast.models.base import P1DModel
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector

DEFAULT_GP_BASEDIR = Path("/home/mfho/student_projects/InferenceLyaData/Emulator_Files")


# ---------------------------------------------------------------------------
# Mock GP (used in tests + as a fallback when GPy is unavailable)
# ---------------------------------------------------------------------------


@dataclass
class MockGPModel(P1DModel):
    """Deterministic synthetic P_F(theta, k, z) for tests.

    Form: ``P_F(theta, k, z) = A(theta) * k**alpha(theta) * exp(-k * scale(z))``
    where A and alpha are smooth linear-in-theta perturbations around 1 and -1
    respectively. Captures the qualitative shape of a real Lya P1D (power-law
    × small-scale damping) without requiring a real emulator.

    Useful as ``gp_model`` for normalization derivation and as a known-truth
    reference in likelihood/Fisher tests.
    """

    base_amplitude: float = 8.0e-3
    z_pivot: float = 3.6
    z_scale: float = 80.0  # damping scale at z_pivot
    fiducial: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.fiducial is None:
            self.fiducial = np.asarray(fiducial_vector(), dtype=float)

    def _A(self, theta: np.ndarray) -> float:
        # Sensitivity to a small subset for testability.
        # Coefficients picked so that 5% changes in `ns/Ap/hub/omegamh2`
        # give percent-level changes in P_F, mimicking real sensitivities.
        idx = {n: PARAM_NAMES.index(n) for n in ("ns", "Ap", "hub", "omegamh2")}
        d_ns = (theta[idx["ns"]] - self.fiducial[idx["ns"]]) / 0.25
        d_Ap = (theta[idx["Ap"]] - self.fiducial[idx["Ap"]]) / 1.4e-9
        d_hub = (theta[idx["hub"]] - self.fiducial[idx["hub"]]) / 0.10
        d_omh2 = (theta[idx["omegamh2"]] - self.fiducial[idx["omegamh2"]]) / 0.006
        return self.base_amplitude * (1.0 + 0.10 * d_ns + 0.20 * d_Ap + 0.05 * d_hub + 0.05 * d_omh2)

    def _alpha(self, theta: np.ndarray) -> float:
        idx = PARAM_NAMES.index("ns")
        d_ns = (theta[idx] - self.fiducial[idx]) / 0.25
        return -1.0 + 0.05 * d_ns

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        k = np.asarray(k, dtype=float)
        if theta.shape != (11,):
            raise ValueError(f"theta must be shape (11,), got {theta.shape}.")
        if not np.all(k > 0):
            raise ValueError("k must be strictly positive.")
        scale = self.z_scale * (z / self.z_pivot)
        amp = self._A(theta)
        alpha = self._alpha(theta)
        return amp * np.power(k, alpha) * np.exp(-k * scale)


# ---------------------------------------------------------------------------
# Real GP adapter
# ---------------------------------------------------------------------------


class GPModel(P1DModel):
    """Adapter around `PRIYAEmulatorExplorer` for forecast use.

    The first call constructs the explorer and loads the multi-fidelity
    emulator pickles (slow, seconds). Subsequent calls reuse the cached
    explorer. Predictions for one (theta, z) hit the GP once; the result is
    binned onto the requested k-grid via `bin_model_to_data`.

    This class requires GPy + emukit + the trained pickles. Imports are lazy
    so unit tests that don't touch this class don't need the dependency.

    Parameters
    ----------
    basedir : str | Path
        Path containing `emulator_params.json` and `trained_mf/zbin*/`.
    hires_subdir : str
        Sub-directory holding the high-fidelity sims, per upstream API.
    tau_thresh : float
        DLA flux threshold, per upstream API.
    a_template : ndarray, shape (4,) | None
        DLA template amplitudes [a_lls, a_sub, a_sdla, a_ldla]. Default zeros.
    """

    def __init__(
        self,
        basedir: str | Path = DEFAULT_GP_BASEDIR,
        *,
        hires_subdir: str = "hires",
        tau_thresh: float = 1e6,
        a_template: np.ndarray | None = None,
    ) -> None:
        self.basedir = Path(basedir)
        if not self.basedir.exists():
            raise FileNotFoundError(
                f"GP emulator basedir does not exist: {self.basedir}. "
                f"Pass `basedir=...` or set it in configs/default.yaml."
            )
        params_json = self.basedir / "emulator_params.json"
        if not params_json.exists():
            raise FileNotFoundError(
                f"`emulator_params.json` not found under {self.basedir}. "
                f"Is this the correct PRIYA Emulator_Files dir?"
            )
        self._hires_subdir = hires_subdir
        self._tau_thresh = tau_thresh
        self._explorer = None  # lazy
        self.a_template = (
            np.zeros(4, dtype=float) if a_template is None else np.asarray(a_template, dtype=float)
        )
        if self.a_template.shape != (4,):
            raise ValueError(f"a_template must have shape (4,), got {self.a_template.shape}.")

    def _ensure_loaded(self):
        if self._explorer is None:
            try:
                from lyaemu.priya_explorer import PRIYAEmulatorExplorer  # type: ignore[import-not-found]
            except ImportError as e:
                raise ImportError(
                    "GPModel requires `lyaemu` (sbird/lya_emulator). Install GPy+emukit "
                    "and add the lyaemu repo to PYTHONPATH, or use MockGPModel for tests."
                ) from e
            self._explorer = PRIYAEmulatorExplorer(
                basedir=str(self.basedir),
                hires_subdir=self._hires_subdir,
                tau_thresh=self._tau_thresh,
            )
        return self._explorer

    def _theta_15d(self, theta_11d: np.ndarray) -> np.ndarray:
        """Pad an 11D physics vector to 15D with the configured DLA amplitudes."""
        theta_11d = np.asarray(theta_11d, dtype=float)
        if theta_11d.shape != (11,):
            raise ValueError(f"theta must be shape (11,), got {theta_11d.shape}.")
        return np.concatenate([theta_11d, self.a_template])

    def _z_index(self, z: float) -> int:
        """Locate the z-slice index in the explorer's prediction list.

        The upstream explorer returns predictions ordered by `redshifts`
        (decreasing — see `priya_explorer.py`'s `set_data_corr`). We snap to
        within 1e-3.
        """
        explorer = self._ensure_loaded()
        zs = np.asarray(explorer.zout if hasattr(explorer, "zout") else explorer.redshifts)
        i = int(np.argmin(np.abs(zs - z)))
        if abs(zs[i] - z) > 1e-3:
            raise ValueError(
                f"z={z} not in emulator's z-grid: {zs}. Add the missing z-bin or pick another."
            )
        return i

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        explorer = self._ensure_loaded()
        theta_15d = self._theta_15d(theta)
        # Upstream signature: get_predicted takes the first 11 entries.
        okf_list, pred_list, _std_list = explorer.emulator_wrap.get_predicted(theta_15d[:11])
        zi = self._z_index(z)
        okf = np.asarray(okf_list[zi], dtype=float)
        pred = np.asarray(pred_list[zi], dtype=float)
        # Bin to the requested k-grid (top-hat).
        k = np.asarray(k, dtype=float)
        if np.array_equal(k, okf):
            return pred
        return bin_model_to_data(okf, pred, k)
