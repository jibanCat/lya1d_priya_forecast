"""eBOSS DR14 P1D data + covariance loader.

Phase 2 will fill this out. Wraps the vendored BOSSData class and exposes:
- load_eboss(z=3.6) -> (k_eboss, pf_eboss, cov_eboss)
- bin_model_to_data(k_model, pf_model, k_eboss) -> pf_binned

Units: k in s/km, P_F dimensionless, redshift dimensionless.
"""

# TODO(phase2): implement load_eboss, bin_model_to_data
