# Multi-z Fisher forecast (PySR additive-Taylor combine, mode=local_anchored)
emulator: /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
z range: [2.6, 4.2] (z_grid=[2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
k-grid: linspace(0.005, 0.064, 32) s/km
cov: KSData(conservative=True), filtered to z=[2.6,4.2], k≤0.064
priors: {'hub': 0.015, 'omegamh2': 0.001, 'bhfeedback': 0.005, 'tau0': 0.33136000000000004}
hybrid vs HF GP at fid (max over z): 0.0000%
refits dropped by quality gate (routed via GP-slice): dtau0 (LF rel-err=2.5e+12% >= 5%; HF rel-err=1e+13% >= 5%), tau0 (LF rel-err=5.1% >= 5%; HF rel-err=5.4% >= 5%), omegamh2 (no x0 term), hireionz (no x0 term)

| param | GP σ | hybrid σ | hybrid/GP ratio | LF rel-err | HF rel-err | x0? | complexity | route |
|---|---|---|---|---|---|---|---|---|
| tau0 | 0.0289 | 0.0383 | **1.33×** | 5.14% | 5.41% | ✓ | 20 | GP-slice (gated) |
| ns | 0.0441 | 0.0612 | **1.39×** | 0.74% | 1.50% | ✓ | 19 | PySR |
| Ap | 0.174 | 0.454 | **2.62×** | 1.87% | 1.93% | ✓ | 20 | PySR |
| herei | 0.12 | 0.649 | **5.43×** | 1.06% | 3.16% | ✓ | 15 | PySR |
| heref | 0.386 | 1.25 | **3.25×** | 0.62% | 2.11% | ✓ | 20 | PySR |
| alphaq | 0.395 | 1.4 | **3.55×** | 0.98% | 2.85% | ✓ | 17 | PySR |
| hub | 0.0118 | 0.015 | **1.27×** | 0.53% | 1.41% | ✓ | 19 | PySR |
| omegamh2 | 0.000986 | 0.000974 | **0.99×** | 0.35% | 1.13% | ✗ | 19 | GP-slice (gated) |
| hireionz | 1.56 | 1.71 | **1.10×** | 0.37% | 0.91% | ✗ | 20 | GP-slice (gated) |
| bhfeedback | 0.00492 | 0.005 | **1.02×** | 0.28% | 1.49% | ✓ | 10 | PySR |

## Target subset ('Ap', 'ns', 'tau0', 'dtau0')
  - **Ap**: ratio = 2.62×
  - **ns**: ratio = 1.39×
  - **tau0**: ratio = 1.33×
  - **dtau0**: fixed at fid=-0.009
