| Parameter | GP σ | GP_reference σ | GP_reference / GP | perfect_1D_slices σ | perfect_1D_slices / GP | multid_pysr_2D σ | multid_pysr_2D / GP |
|---|---|---|---|---|---|---|---|
| dtau0 | 0.298 | 0.298 | 1.000 | 0.298 | 1.000 | 1.15 | 3.875 |
| Ap | 0.314 | 0.314 | 1.000 | 0.314 | 1.000 | 1.28 | 4.090 |


### Reward gauges (lower = better)

| Gauge | What it measures | Per-param | Geomean |
|---|---|---|---|
| σ_student / σ_perfect_1D | distance from 1D-product upper bound | 3.9, 4.1 | **4** |
| σ_perfect_1D / σ_gp | 1D-factorization tax at Fisher level (always ≈ 1 — see below) | 1, 1 | 1 |

### Off-fiducial residual MSE (eBOSS-σ² units, mean over 16 Sobol points)

- perfect_1D vs GP : **3.14e-05**
- student   vs GP : **0.0278**
- ratio (student / perfect_1D) : **8.9e+02**

### Why σ_perfect_1D / σ_gp = 1 here

At the linearization point (fid), the 1D-product Fisher gradient 
∂P/∂θ_i = P_fid · (1/f_i_fid) · df_i/dθ_i equals the joint gradient 
∂GP/∂θ_i for any equation set whose per-param 1D slices match the GP. 
So Fisher *cannot* see the 1D-factorization tax — only off-fid points or 
MCMC curvature can. Use Phase 5's coupling-matrix diagnostic to quantify 
the joint-vs-product tax across off-fid Sobol space.

### Targets to chase

1. σ_student / σ_perfect_1D < 1.5 (geomean) → 1D PySR is converged.
2. off-fid MSE ratio < 2 → the equations track the GP off-fid too.
3. Once 1 and 2 are met, run the multi-D PySR diagnostic (Phase 5).