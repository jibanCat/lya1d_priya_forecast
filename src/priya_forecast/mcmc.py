"""emcee-driven MCMC for the single-z eBOSS P1D forecast.

`run_mcmc` constructs an `emcee.EnsembleSampler` with `4 * n_dim` walkers
(or whatever `walkers_per_dim` is configured), seeds the initial positions
in a small Gaussian ball around the fiducial point (clipped to the prior
box), and runs `n_steps` total steps. Burn-in is taken as the first
`burn_in_frac * n_steps` steps and discarded by `get_chain(discard=...)`.

We emit a soft warning when the chain is shorter than `50 * tau_mean`
(emcee's recommended convergence rule of thumb), but never raise — the
caller may know the budget is intentional. Returns an `MCMCResult` bundling
the chain, log-prob, autocorr time, and a path to the HDF5 backend.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from priya_forecast.likelihood import LogPosterior, UniformBoxPrior
from priya_forecast.parameters import PARAMS_11D, Param


@dataclass
class MCMCResult:
    chain: np.ndarray  # shape (n_steps_post_burn, n_walkers, n_dim)
    log_prob: np.ndarray  # shape (n_steps_post_burn, n_walkers)
    tau: np.ndarray  # autocorr time per dim, shape (n_dim,)
    backend_path: Path | None
    n_steps: int
    n_walkers: int
    burn_in: int
    converged: bool
    param_names: tuple[str, ...]


def _initial_positions(
    theta_fid: np.ndarray,
    n_walkers: int,
    params: tuple[Param, ...],
    spread_frac: float,
    rng: np.random.Generator,
) -> np.ndarray:
    """Gaussian ball around `theta_fid`, clipped to prior bounds."""
    n = len(params)
    widths = np.array([p.width() for p in params]) * spread_frac
    pos = rng.normal(loc=theta_fid, scale=widths, size=(n_walkers, n))
    lo = np.array([p.prior[0] for p in params])
    hi = np.array([p.prior[1] for p in params])
    # Clip with a tiny inset so walkers don't sit exactly on the boundary
    # (the uniform prior would still accept them, but it's cleaner).
    inset = 1e-6 * (hi - lo)
    return np.clip(pos, lo + inset, hi - inset)


def run_mcmc(
    *,
    posterior: LogPosterior,
    theta_fid: np.ndarray | None = None,
    params: tuple[Param, ...] = PARAMS_11D,
    n_steps: int = 5000,
    walkers_per_dim: int = 4,
    burn_in_frac: float = 0.2,
    backend_path: str | Path | None = None,
    spread_frac: float = 0.01,
    seed: int = 0,
    progress: bool = False,
) -> MCMCResult:
    """Run emcee on `posterior` and return the post-burn-in chain.

    Parameters
    ----------
    posterior : LogPosterior
        Combined log-prior + log-likelihood (callable on (n_dim,) theta).
    theta_fid : ndarray, shape (n_dim,) | None
        Walker init point. Defaults to `[p.fid for p in params]`.
    n_steps, walkers_per_dim, burn_in_frac : forecast knobs.
    backend_path : path | None
        If given, an `emcee.backends.HDFBackend` is attached.
    spread_frac : float
        Walkers initialized in a Gaussian ball of size
        `spread_frac * prior_width` around `theta_fid`.
    seed : int
        RNG seed for walker initialization (emcee's internal moves use their
        own rng).
    progress : bool
        Forward to emcee for a tqdm bar.
    """
    try:
        import emcee
    except ImportError as e:  # pragma: no cover - we install emcee in tests
        raise ImportError("run_mcmc requires emcee.") from e

    n_dim = len(params)
    n_walkers = walkers_per_dim * n_dim
    if theta_fid is None:
        theta_fid = np.array([p.fid for p in params], dtype=float)
    theta_fid = np.asarray(theta_fid, dtype=float)

    rng = np.random.default_rng(seed)
    p0 = _initial_positions(theta_fid, n_walkers, params, spread_frac, rng)

    backend = None
    if backend_path is not None:
        backend_path = Path(backend_path)
        backend_path.parent.mkdir(parents=True, exist_ok=True)
        backend = emcee.backends.HDFBackend(str(backend_path))
        backend.reset(n_walkers, n_dim)

    sampler = emcee.EnsembleSampler(n_walkers, n_dim, posterior, backend=backend)
    sampler.run_mcmc(p0, n_steps, progress=progress)

    burn_in = int(burn_in_frac * n_steps)
    chain = sampler.get_chain(discard=burn_in)
    log_prob = sampler.get_log_prob(discard=burn_in)

    try:
        tau = sampler.get_autocorr_time(tol=0)
    except Exception:
        tau = np.full(n_dim, np.nan)

    converged = bool(
        np.all(np.isfinite(tau)) and (n_steps - burn_in) > 50 * float(np.nanmax(tau))
    )
    if not converged:
        warnings.warn(
            f"MCMC may not be converged: chain length {n_steps - burn_in} "
            f"< 50 * tau_max ({tau}).",
            UserWarning,
            stacklevel=2,
        )

    return MCMCResult(
        chain=chain,
        log_prob=log_prob,
        tau=tau,
        backend_path=Path(backend_path) if backend_path is not None else None,
        n_steps=n_steps,
        n_walkers=n_walkers,
        burn_in=burn_in,
        converged=converged,
        param_names=tuple(p.name for p in params),
    )
