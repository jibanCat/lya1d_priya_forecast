"""Fisher forecast via 5-point-stencil derivatives with adaptive halving.

Phase 4 will fill this out. Step starts at 1% of prior width, halves until
relative change in F_ii is < 1%. Outputs Fisher, covariance, 1sigma marg
errors, correlation matrix.
"""

# TODO(phase4): fisher_matrix(model, likelihood, theta_fid, params)
