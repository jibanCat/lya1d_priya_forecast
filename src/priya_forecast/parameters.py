"""The 11 PRIYA forecast parameters: fiducial values, priors, LaTeX labels.

Single source of truth for parameter metadata used across the forecast,
multi-D diagnostic, and HPO. Fiducial values come from the upstream
`PRIYAEmulatorExplorer.best_par` (15D vector, first 11 entries).

Cosmology + IGM-thermal-history priors come from
`InferenceLyaData/Emulator_Files/emulator_params.json` (`param_limits`).
Mean-flux priors (`dtau0`, `tau0`) come from the PRIYA paper's MCMC setup.

Units: P_F dimensionless, k in s/km, redshift dimensionless.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class Param:
    """A single forecast parameter.

    Attributes
    ----------
    name : str
        Symbol used in YAML configs and as the sympy variable in PySR equations.
    fid : float
        Fiducial value used as the linearization point for Fisher and as the
        starting point for MCMC walkers.
    prior : tuple[float, float]
        Uniform prior `(lo, hi)`. `lo < fid < hi` is enforced at load time.
    latex : str
        LaTeX label, used by plotting helpers (no surrounding ``$``).
    """

    name: str
    fid: float
    prior: tuple[float, float]
    latex: str

    def width(self) -> float:
        """Prior width `hi - lo`. Used as the natural step scale for Fisher."""
        return self.prior[1] - self.prior[0]


PARAMS_11D: tuple[Param, ...] = (
    # mean flux — confirmed priors from PRIYA paper / upstream MCMC setup
    Param("dtau0", fid=-0.009, prior=(-0.4, 0.25), latex=r"d\tau_0"),
    Param("tau0", fid=1.090, prior=(0.75, 1.25), latex=r"\tau_0"),
    # cosmology — from emulator_params.json
    Param("ns", fid=0.983, prior=(0.8, 1.05), latex=r"n_s"),
    Param("Ap", fid=1.46e-9, prior=(1.2e-9, 2.6e-9), latex=r"A_P"),
    # IGM thermal history
    Param("herei", fid=4.0, prior=(3.5, 4.5), latex=r"z^{HeII}_i"),
    Param("heref", fid=2.765, prior=(2.2, 3.2), latex=r"z^{HeII}_f"),
    Param("alphaq", fid=1.74, prior=(1.3, 3.0), latex=r"\alpha_q"),
    # cosmology cont'd
    Param("hub", fid=0.688, prior=(0.65, 0.75), latex=r"h"),
    Param("omegamh2", fid=0.1439, prior=(0.14, 0.146), latex=r"\Omega_0 h^2"),
    Param("hireionz", fid=7.24, prior=(6.5, 8.0), latex=r"z_{HI}"),
    Param("bhfeedback", fid=0.050, prior=(0.03, 0.07), latex=r"\epsilon_{AGN}"),
)


PARAM_NAMES: tuple[str, ...] = tuple(p.name for p in PARAMS_11D)
"""Canonical name order. `theta_11d[i]` corresponds to `PARAMS_11D[i]`."""


def get_param(name: str) -> Param:
    """Return the `Param` with the given name. Raises KeyError if absent."""
    for p in PARAMS_11D:
        if p.name == name:
            return p
    raise KeyError(f"Unknown PRIYA parameter: {name!r}. Known: {PARAM_NAMES}")


def fiducial_vector() -> tuple[float, ...]:
    """Return the 11D fiducial point in canonical order."""
    return tuple(p.fid for p in PARAMS_11D)


def prior_bounds() -> tuple[tuple[float, float], ...]:
    """Return the 11 prior tuples in canonical order."""
    return tuple(p.prior for p in PARAMS_11D)


def validate_priors(params: tuple[Param, ...] = PARAMS_11D) -> None:
    """Sanity-check every Param: prior is non-degenerate, fid is strictly inside.

    Raises ValueError if any parameter has `lo >= hi`, `fid <= lo`, or
    `fid >= hi`. Called by the CLI before any forecast run.
    """
    for p in params:
        lo, hi = p.prior
        if not (lo < hi):
            raise ValueError(f"Param {p.name!r}: prior lo={lo} not < hi={hi}.")
        if not (lo < p.fid < hi):
            raise ValueError(
                f"Param {p.name!r}: fid={p.fid} not strictly inside prior ({lo}, {hi})."
            )
