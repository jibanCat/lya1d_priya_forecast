| Parameter | GP σ | GP_reference σ | GP_reference / GP | perfect_1D_slices σ | perfect_1D_slices / GP | multid_pysr_4D σ | multid_pysr_4D / GP |
|---|---|---|---|---|---|---|---|
| dtau0 | 0.428 | 0.428 | 1.000 | 0.428 | 1.000 | 3.76e+06 | 8770214.099 |
| ns | 0.0618 | 0.0618 | 1.000 | 0.0618 | 1.000 | 1.74e+06 | 28130410.009 |
| Ap | 0.745 | 0.745 | 1.000 | 0.745 | 1.000 | 8.09e+06 | 10866648.368 |
| alphaq | 5.42 | 5.42 | 1.000 | 5.42 | 1.000 | 2.81e+12 | 517968436269.981 |


### Reward gauges (lower = better)

| Gauge | What it measures | Per-param | Geomean |
|---|---|---|---|
| σ_student / σ_perfect_1D | distance from 1D-product upper bound | 8.8e+06, 2.8e+07, 1.1e+07, 5.2e+11 | **1.9e+08** |
| σ_perfect_1D / σ_gp | 1D-factorization tax at Fisher level (always ≈ 1 — see below) | 1, 1, 1, 1 | 1 |

### Off-fiducial residual MSE (eBOSS-σ² units, mean over 16 Sobol points)

- perfect_1D vs GP : **0.00117**
- student   vs GP : **0.503**
- ratio (student / perfect_1D) : **4.3e+02**

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