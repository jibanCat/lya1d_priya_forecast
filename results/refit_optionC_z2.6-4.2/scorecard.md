# Multi-z Fisher forecast (PySR additive-Taylor combine, mode=local_anchored)
emulator: /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
z range: [2.6, 4.2] (z_grid=[2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
k-grid: linspace(0.005, 0.064, 32) s/km
cov: synthetic diagonal, σ_k = 5.0%·P_F(fid, k)
priors: {'hub': 0.015, 'omegamh2': 0.001, 'bhfeedback': 0.005}
hybrid vs HF GP at fid (max over z): 0.0000%

| param | GP σ | hybrid σ | hybrid/GP ratio | LF rel-err | HF rel-err | x0? | complexity |
|---|---|---|---|---|---|---|---|
| tau0 | 0.0159 | 0.0196 | **1.23×** | 0.72% | 1.32% | ✓ | 19 |
| ns | 0.0258 | 0.0228 | **0.88×** | 0.64% | 1.29% | ✓ | 17 |
| Ap | 0.11 | 0.0479 | **0.44×** | 1.68% | 2.06% | ✓ | 20 |
| herei | 0.0677 | 0.276 | **4.08×** | 0.49% | 2.15% | ✓ | 18 |
| heref | 0.206 | 1.3 | **6.28×** | 0.88% | 1.98% | ✓ | 20 |
| alphaq | 0.241 | 0.172 | **0.71×** | 0.77% | 2.49% | ✓ | 20 |
| hub | 0.00591 | 0.0146 | **2.47×** | 0.48% | 1.38% | ✓ | 20 |
| omegamh2 | 0.000956 | 0.001 | **1.05×** | 0.34% | 1.24% | ✗ | 19 |
| hireionz | 0.676 | 7.34e+11 | **1086425773160.50×** | 0.31% | 1.04% | ✗ | 20 |
| bhfeedback | 0.00468 | 0.005 | **1.07×** | 0.22% | 0.92% | ✗ | 19 |

## Target subset ('Ap', 'ns', 'tau0', 'dtau0')
  - **Ap**: ratio = 0.44×
  - **ns**: ratio = 0.88×
  - **tau0**: ratio = 1.23×
  - **dtau0**: fixed at fid=-0.009
