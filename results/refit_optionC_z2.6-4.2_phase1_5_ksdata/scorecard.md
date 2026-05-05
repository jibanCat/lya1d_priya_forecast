# Multi-z Fisher forecast (PySR additive-Taylor combine, mode=local_anchored)
emulator: /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
z range: [2.6, 4.2] (z_grid=[2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
k-grid: linspace(0.005, 0.064, 32) s/km
cov: KSData(conservative=True), filtered to z=[2.6,4.2], k≤0.064
priors: {'hub': 0.015, 'omegamh2': 0.001, 'bhfeedback': 0.005, 'tau0': 0.33136000000000004}
hybrid vs HF GP at fid (max over z): 0.0000%
refits dropped by quality gate (routed via GP-slice): dtau0 (LF rel-err=2.5e+12% >= 5%; HF rel-err=1e+13% >= 5%), heref (no x0 term), omegamh2 (no x0 term), hireionz (no x0 term), bhfeedback (no x0 term)

| param | GP σ | hybrid σ | hybrid/GP ratio | LF rel-err | HF rel-err | x0? | complexity | route |
|---|---|---|---|---|---|---|---|---|
| tau0 | 0.0289 | 0.0364 | **1.26×** | 0.72% | 1.32% | ✓ | 19 | PySR |
| ns | 0.0441 | 0.0617 | **1.40×** | 0.64% | 1.29% | ✓ | 17 | PySR |
| Ap | 0.174 | 0.138 | **0.79×** | 1.68% | 2.06% | ✓ | 20 | PySR |
| herei | 0.12 | 0.705 | **5.90×** | 1.00% | 2.60% | ✓ | 19 | PySR |
| heref | 0.386 | 0.341 | **0.88×** | 0.92% | 1.83% | ✗ | 19 | GP-slice (gated) |
| alphaq | 0.395 | 0.512 | **1.30×** | 0.78% | 2.59% | ✓ | 20 | PySR |
| hub | 0.0118 | 0.0149 | **1.26×** | 0.48% | 1.38% | ✓ | 20 | PySR |
| omegamh2 | 0.000986 | 0.00098 | **0.99×** | 0.34% | 1.24% | ✗ | 19 | GP-slice (gated) |
| hireionz | 1.56 | 1.58 | **1.01×** | 0.31% | 1.04% | ✗ | 20 | GP-slice (gated) |
| bhfeedback | 0.00492 | 0.00495 | **1.00×** | 0.22% | 0.92% | ✗ | 19 | GP-slice (gated) |

## Target subset ('Ap', 'ns', 'tau0', 'dtau0')
  - **Ap**: ratio = 0.79×
  - **ns**: ratio = 1.40×
  - **tau0**: ratio = 1.26×
  - **dtau0**: fixed at fid=-0.009
