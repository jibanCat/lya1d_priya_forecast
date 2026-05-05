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
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector, to_physical

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

    # Tuned so MockGP P_F(z=3.6) at the eBOSS k-grid sits in a realistic range
    # (~70 at k≈0.001, ~25 at k≈0.02). Keeps Fisher chi^2 magnitudes sane when
    # combined with the real DR14 covariance.
    base_amplitude: float = 2.5
    z_pivot: float = 3.6
    z_scale: float = 8.0  # damping scale at z_pivot (≈ realistic low-k turnover)
    fiducial: np.ndarray | None = None

    def __post_init__(self) -> None:
        if self.fiducial is None:
            self.fiducial = np.asarray(fiducial_vector(), dtype=float)

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        theta = np.asarray(theta, dtype=float)
        k = np.asarray(k, dtype=float)
        if theta.shape != (11,):
            raise ValueError(f"theta must be shape (11,), got {theta.shape}.")
        if not np.all(k > 0):
            raise ValueError("k must be strictly positive.")

        # Each parameter perturbs a *distinct* k-shape so the Fisher matrix is
        # well-conditioned on the parameter subset that actually varies.
        # Inputs are in internal units (Ap is 1.46, not 1.46e-9).
        idx = {n: PARAM_NAMES.index(n) for n in PARAM_NAMES}
        d = {n: 0.0 for n in PARAM_NAMES}
        d["ns"]       = (theta[idx["ns"]]       - self.fiducial[idx["ns"]])       / 0.25
        d["Ap"]       = (theta[idx["Ap"]]       - self.fiducial[idx["Ap"]])       / 1.4
        d["hub"]      = (theta[idx["hub"]]      - self.fiducial[idx["hub"]])      / 0.10
        d["omegamh2"] = (theta[idx["omegamh2"]] - self.fiducial[idx["omegamh2"]]) / 0.006

        scale = self.z_scale * (z / self.z_pivot)
        # Base shape: power-law × exponential damping.
        base = self.base_amplitude * np.power(k, -0.5) * np.exp(-k * scale)
        # Distinct k-shape modulations per parameter:
        amp_shift   = 0.30 * d["ns"]              # broad-in-k amplitude
        tilt_shift  = 0.50 * d["Ap"] * np.log(k / 0.005)  # k-tilt (negative at low k)
        damp_shift  = 0.20 * d["hub"] * (k * scale)       # extra small-k power / large-k loss
        lowk_shift  = 0.20 * d["omegamh2"] * np.exp(-k * scale * 4.0)  # low-k boost
        return base * (1.0 + amp_shift + tilt_shift + damp_shift + lowk_shift)


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
        hires_subdir: str | None = "hires",
        fidelity: str = "hf",
        tau_thresh: float = 1e6,
        a_template: np.ndarray | None = None,
        kf: np.ndarray | None = None,
    ) -> None:
        """Construct a single-fidelity (LF) or multi-fidelity (HF) PRIYA emulator.

        Parameters
        ----------
        fidelity : {"hf", "lf"}
            "hf" → multi-fidelity emulator using `hires_subdir` (default).
            "lf" → low-fidelity (single-fidelity) emulator; equivalent to
            the notebook's `PRIYAEmulatorExplorer(..., hires_subdir=None)`.
            Used by the student's `pysr_mf_given.py` to produce LF
            training data for the resolution-feature MF PySR.
        """
        if fidelity not in ("hf", "lf"):
            raise ValueError(f"fidelity must be 'hf' or 'lf', got {fidelity!r}.")
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
        self._fidelity = fidelity
        self._hires_subdir = hires_subdir if fidelity == "hf" else None
        self._tau_thresh = tau_thresh
        self._explorer = None  # lazy
        self._kf_override: np.ndarray | None = (
            None if kf is None else np.asarray(kf, dtype=float)
        )
        self.a_template = (
            np.zeros(4, dtype=float) if a_template is None else np.asarray(a_template, dtype=float)
        )
        if self.a_template.shape != (4,):
            raise ValueError(f"a_template must have shape (4,), got {self.a_template.shape}.")

    def _ensure_loaded(self):
        """Build a `GPWrap` directly. We skip `PRIYAEmulatorExplorer` because
        its constructor instantiates `KSData()` (KODIAQ data) which has an
        upstream read-only-pf bug; we don't need the KODIAQ side anyway since
        the forecast uses eBOSS DR14 covariance from our own data loader."""
        if self._explorer is None:
            try:
                from lyaemu.gp_wrap import GPWrap  # type: ignore[import-not-found]
            except ImportError as e:
                raise ImportError(
                    "GPModel requires `lyaemu` (sbird/lya_emulator). Install GPy+emukit "
                    "and add the lyaemu repo to PYTHONPATH, or use MockGPModel for tests."
                ) from e

            # Default to the eBOSS DR14 k-grid; pass `kf=...` at construction
            # to override (e.g., kodiaq production uses k up to 0.064 s/km).
            if self._kf_override is None:
                from priya_forecast.data import load_eboss
                kf_use, _, _ = load_eboss(z=3.6)
            else:
                kf_use = self._kf_override
            gp = GPWrap(
                basedir=str(self.basedir),
                emulator_json_file="emulator_params.json",
                kf=kf_use,
                tau_thresh=self._tau_thresh,
                use_res_corr=False,
            )
            traindir = str(self.basedir / "trained_mf")
            if self._fidelity == "hf":
                if self._hires_subdir is None:
                    raise ValueError(
                        "GPModel(fidelity='hf') requires hires_subdir to "
                        "be set (got None). For LF-only emulation pass "
                        "fidelity='lf' instead."
                    )
                hires_basedir = str(self.basedir / self._hires_subdir)
                gp.set_emulator(
                    HRbasedir=hires_basedir, max_z=4.6, min_z=2.2, traindir=traindir,
                )
            else:
                # LF: single-fidelity — HRbasedir=None matches
                # PRIYAEmulatorExplorer(..., hires_subdir=None) in the
                # student's notebook (cell 6 of 14b_fernandez_explorer.ipynb).
                gp.set_emulator(
                    HRbasedir=None, max_z=4.6, min_z=2.2, traindir=traindir,
                )
            gp.set_mf_param_limits(basedir=str(self.basedir))
            self._explorer = gp
        return self._explorer

    def _theta_15d(self, theta_11d_internal: np.ndarray) -> np.ndarray:
        """Convert an 11D internal-units theta to a 15D physical-units vector
        for the upstream GP. Internal units carry Ap as 1.46 (× 10^-9);
        upstream wants 1.46e-9. The DLA template amplitudes are appended."""
        theta_11d_internal = np.asarray(theta_11d_internal, dtype=float)
        if theta_11d_internal.shape != (11,):
            raise ValueError(f"theta must be shape (11,), got {theta_11d_internal.shape}.")
        theta_11d_physical = to_physical(theta_11d_internal)
        return np.concatenate([theta_11d_physical, self.a_template])

    def _z_index(self, z: float) -> int:
        gp = self._ensure_loaded()
        zs = np.asarray(gp.zout if hasattr(gp, "zout") else gp.redshifts)
        i = int(np.argmin(np.abs(zs - z)))
        if abs(zs[i] - z) > 1e-3:
            raise ValueError(
                f"z={z} not in emulator's z-grid: {zs}. Add the missing z-bin or pick another."
            )
        return i

    def predict(self, theta: np.ndarray, k: np.ndarray, z: float) -> np.ndarray:
        gp = self._ensure_loaded()
        theta_15d = self._theta_15d(theta)
        okf_list, pred_list, _std_list = gp.get_predicted(theta_15d[:11])
        zi = self._z_index(z)
        okf = np.asarray(okf_list[zi], dtype=float)
        pred = np.asarray(pred_list[zi], dtype=float)
        k = np.asarray(k, dtype=float)
        if np.array_equal(k, okf):
            return pred
        return bin_model_to_data(okf, pred, k)
