# Multi-D cross-coupled forecast (single PySR eq + GP-slice fallback)
emulator: /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
subset: ['ns', 'Ap', 'herei', 'heref', 'alphaq', 'hireionz']  (handled by joint multi-D PySR eq)
fixed:  ['dtau0']
z range: [2.6, 4.2] (z_grid=[2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
k-grid: linspace(0.005, 0.064, 32) s/km, cov: 5.0%·P_F(fid, k) per z.
priors: {'hub': 0.015, 'omegamh2': 0.001, 'bhfeedback': 0.005}
hybrid vs HF GP at fid: 0.0000%
multi-D eq complexity: 20, flux_norm loss: 0.531

| param | in subset? | GP σ | hybrid σ | hybrid/GP ratio |
|---|---|---|---|---|
| tau0 | GP-slice | 0.0159 | 0.0162 | **1.02×** |
| ns | ✓ multi-D | 0.0258 | nan | **nan×** |
| Ap | ✓ multi-D | 0.11 | nan | **nan×** |
| herei | ✓ multi-D | 0.0677 | nan | **nan×** |
| heref | ✓ multi-D | 0.206 | nan | **nan×** |
| alphaq | ✓ multi-D | 0.241 | nan | **nan×** |
| hub | GP-slice | 0.00591 | 0.00587 | **0.99×** |
| omegamh2 | GP-slice | 0.000956 | 0.00092 | **0.96×** |
| hireionz | ✓ multi-D | 0.676 | nan | **nan×** |
| bhfeedback | GP-slice | 0.00468 | 0.00455 | **0.97×** |

## Target subset ('Ap', 'ns', 'tau0', 'dtau0')
  - **Ap**: ratio = nan× (multi-D)
  - **ns**: ratio = nan× (multi-D)
  - **tau0**: ratio = 1.02× (GP-slice)
  - **dtau0**: fixed at fid=-0.009
