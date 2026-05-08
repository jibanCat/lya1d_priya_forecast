# Off-fid closure at θ_target_simdat (Data Index 15)
emulator: /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
θ_target_simdat (with dtau0 → 0): see fisher_at_target.npz
z range: [2.6, 4.2]; KSData k-grid: 11 bins
hybrid-vs-GP at θ_target (mid-z): max |Δ/P_F| = 2.43%
priors: {'hub': 0.015, 'omegamh2': 0.001, 'bhfeedback': 0.005, 'tau0': 0.33136000000000004}

| param | route | σ_GP | σ_PySR | σ_MCMC | PySR/GP | PySR/MCMC | GP/MCMC |
|---|---|---|---|---|---|---|---|
| tau0 | PySR | 0.0228 | 0.0246 | 0.0271 | **1.08×** | **0.91×** | 0.84× |
| ns | PySR | 0.0422 | 0.0399 | 0.0576 | **0.95×** | **0.69×** | 0.73× |
| Ap | PySR | 0.229 | 0.371 | 0.253 | **1.62×** | **1.47×** | 0.91× |
| herei | PySR | 0.184 | 0.733 | 0.147 | **3.97×** | **4.97×** | 1.25× |
| heref | GP-slice (gated) | 0.379 | 0.29 | 0.162 | **0.77×** | **1.79×** | 2.34× |
| alphaq | PySR | 0.643 | 0.248 | 0.395 | **0.39×** | **0.63×** | 1.63× |
| hub | PySR | 0.0139 | 0.0148 | 0.0106 | **1.07×** | **1.40×** | 1.31× |
| omegamh2 | GP-slice (gated) | 0.000983 | 0.000973 | 0.0017 | **0.99×** | **0.57×** | 0.58× |
| hireionz | GP-slice (gated) | 1.92 | 1.02 | 0.429 | **0.53×** | **2.37×** | 4.47× |
| bhfeedback | GP-slice (gated) | 0.00493 | 0.00494 | 0.00354 | **1.00×** | **1.39×** | 1.39× |

## Interpretation
- `PySR/GP` ≈ 1: hybrid Fisher matches the GP Fisher at θ_target (faithful off-fid).
- `PySR/MCMC` ≈ 1: hybrid σ matches the truth (final closure target).
- `GP/MCMC` ≠ 1 → Gaussianity / boundary effects in the simdat MCMC,   not a pipeline failure. Useful sanity check for the Cramer-Rao bound.

Diverging `PySR/GP` flags params where the per-1D + additive-Taylor extrapolation has wrong fid-curvature. Candidates for Phase 1.5 smart refit (ANOVA loss + restricted operators) or Phase 2 pair coupling.
