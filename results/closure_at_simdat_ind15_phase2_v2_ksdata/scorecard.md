# Off-fid closure at θ_target_simdat (Data Index 15)
emulator: /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
θ_target_simdat (with dtau0 → 0): see fisher_at_target.npz
z range: [2.6, 4.2]; KSData k-grid: 11 bins
hybrid-vs-GP at θ_target (mid-z): max |Δ/P_F| = 3.27%
priors: {'hub': 0.015, 'omegamh2': 0.001, 'bhfeedback': 0.005}

| param | route | σ_GP | σ_PySR | σ_MCMC | PySR/GP | PySR/MCMC | GP/MCMC |
|---|---|---|---|---|---|---|---|
| tau0 | GP-slice (gated) | 0.0229 | 0.0341 | 0.0271 | **1.49×** | **1.26×** | 0.84× |
| ns | PySR | 0.0423 | 0.0542 | 0.0576 | **1.28×** | **0.94×** | 0.73× |
| Ap | PySR | 0.229 | 0.301 | 0.253 | **1.31×** | **1.19×** | 0.91× |
| herei | PySR | 0.184 | 0.604 | 0.147 | **3.28×** | **4.10×** | 1.25× |
| heref | PySR | 0.379 | 1.1 | 0.162 | **2.91×** | **6.81×** | 2.34× |
| alphaq | PySR | 0.643 | 1.38 | 0.395 | **2.14×** | **3.48×** | 1.63× |
| hub | PySR | 0.0139 | 0.015 | 0.0106 | **1.08×** | **1.41×** | 1.31× |
| omegamh2 | GP-slice (gated) | 0.000983 | 0.000979 | 0.0017 | **1.00×** | **0.57×** | 0.58× |
| hireionz | GP-slice (gated) | 1.92 | 0.979 | 0.429 | **0.51×** | **2.28×** | 4.47× |
| bhfeedback | PySR | 0.00493 | 0.005 | 0.00354 | **1.01×** | **1.41×** | 1.39× |

## Interpretation
- `PySR/GP` ≈ 1: hybrid Fisher matches the GP Fisher at θ_target (faithful off-fid).
- `PySR/MCMC` ≈ 1: hybrid σ matches the truth (final closure target).
- `GP/MCMC` ≠ 1 → Gaussianity / boundary effects in the simdat MCMC,   not a pipeline failure. Useful sanity check for the Cramer-Rao bound.

Diverging `PySR/GP` flags params where the per-1D + additive-Taylor extrapolation has wrong fid-curvature. Candidates for Phase 1.5 smart refit (ANOVA loss + restricted operators) or Phase 2 pair coupling.
