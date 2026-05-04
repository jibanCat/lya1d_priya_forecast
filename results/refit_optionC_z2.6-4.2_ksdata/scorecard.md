# Multi-z Fisher forecast (PySR additive-Taylor combine, mode=local_anchored)
emulator: /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
z range: [2.6, 4.2] (z_grid=[2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
k-grid: linspace(0.005, 0.064, 32) s/km
cov: KSData(conservative=True), filtered to z=[2.6,4.2], k≤0.064
priors: {'hub': 0.015, 'omegamh2': 0.001, 'bhfeedback': 0.005}
hybrid vs HF GP at fid (max over z): 0.0000%

| param | GP σ | hybrid σ | hybrid/GP ratio | LF rel-err | HF rel-err | x0? | complexity |
|---|---|---|---|---|---|---|---|
| tau0 | 0.029 | 0.0406 | **1.40×** | 0.72% | 1.32% | ✓ | 19 |
| ns | 0.0442 | 0.0563 | **1.27×** | 0.64% | 1.29% | ✓ | 17 |
| Ap | 0.174 | 0.115 | **0.66×** | 1.68% | 2.06% | ✓ | 20 |
| herei | 0.12 | 0.5 | **4.18×** | 0.49% | 2.15% | ✓ | 18 |
| heref | 0.386 | 2.3 | **5.96×** | 0.88% | 1.98% | ✓ | 20 |
| alphaq | 0.395 | 0.249 | **0.63×** | 0.77% | 2.49% | ✓ | 20 |
| hub | 0.0118 | 0.0149 | **1.26×** | 0.48% | 1.38% | ✓ | 20 |
| omegamh2 | 0.000986 | 0.001 | **1.01×** | 0.56% | 1.16% | ✗ | 20 |
| hireionz | 1.56 | 1.32e+12 | **846689253689.13×** | 0.28% | 1.15% | ✗ | 17 |
| bhfeedback | 0.00492 | 0.005 | **1.02×** | 0.29% | 1.02% | ✗ | 20 |

## Target subset ('Ap', 'ns', 'tau0', 'dtau0')
  - **Ap**: ratio = 0.66×
  - **ns**: ratio = 1.27×
  - **tau0**: ratio = 1.40×
  - **dtau0**: fixed at fid=-0.009
