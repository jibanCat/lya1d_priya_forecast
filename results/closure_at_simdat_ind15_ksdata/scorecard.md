# Off-fid closure at θ_target_simdat (Data Index 15)
emulator: /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
θ_target_simdat (with dtau0 → 0): see fisher_at_target.npz
z range: [2.6, 4.2]; KSData k-grid: 11 bins
hybrid-vs-GP at θ_target (mid-z): max |Δ/P_F| = 4.57%
priors: {'hub': 0.015, 'omegamh2': 0.001, 'bhfeedback': 0.005, 'tau0': 0.33136000000000004}

| param | route | σ_GP | σ_PySR | σ_MCMC | PySR/GP | PySR/MCMC | GP/MCMC |
|---|---|---|---|---|---|---|---|
| tau0 | PySR | 0.0228 | 0.0349 | 0.0271 | **1.53×** | **1.29×** | 0.84× |
| ns | PySR | 0.0422 | 0.0416 | 0.0576 | **0.99×** | **0.72×** | 0.73× |
| Ap | PySR | 0.229 | 0.375 | 0.253 | **1.64×** | **1.48×** | 0.91× |
| herei | PySR | 0.184 | 0.492 | 0.147 | **2.67×** | **3.34×** | 1.25× |
| heref | PySR | 0.379 | 2.43 | 0.162 | **6.40×** | **14.97×** | 2.34× |
| alphaq | PySR | 0.643 | 0.127 | 0.395 | **0.20×** | **0.32×** | 1.63× |
| hub | PySR | 0.0139 | 0.0149 | 0.0106 | **1.07×** | **1.40×** | 1.31× |
| omegamh2 | GP-slice (gated) | 0.000983 | 0.000971 | 0.0017 | **0.99×** | **0.57×** | 0.58× |
| hireionz | GP-slice (gated) | 1.92 | 1.02 | 0.429 | **0.53×** | **2.37×** | 4.47× |
| bhfeedback | GP-slice (gated) | 0.00493 | 0.00495 | 0.00354 | **1.00×** | **1.40×** | 1.39× |

## Interpretation
- `PySR/GP` ≈ 1: hybrid Fisher matches the GP Fisher at θ_target (faithful off-fid).
- `PySR/MCMC` ≈ 1: hybrid σ matches the truth (final closure target).
- `GP/MCMC` ≠ 1 → Gaussianity / boundary effects in the simdat MCMC,   not a pipeline failure. Useful sanity check for the Cramer-Rao bound.

Diverging `PySR/GP` flags params where the per-1D + additive-Taylor extrapolation has wrong fid-curvature. Candidates for Phase 1.5 smart refit (ANOVA loss + restricted operators) or Phase 2 pair coupling.
