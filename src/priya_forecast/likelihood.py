"""Gaussian likelihood for single-z eBOSS P1D forecast.

Phase 4 will fill this out. log L = -0.5 (d - m).T C^-1 (d - m). Raises on
NaN / non-finite m. Honors `cov_scale` and `mock_data: gp` knobs.
"""

# TODO(phase4): GaussianLikelihood + LogPosterior wrapper
