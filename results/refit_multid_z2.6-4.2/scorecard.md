# Multi-D cross-coupled forecast (single PySR eq + GP-slice fallback)
emulator: /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
subset: ['ns', 'Ap', 'herei', 'heref', 'alphaq', 'hireionz']  (handled by joint multi-D PySR eq)
fixed:  ['dtau0']
z range: [2.6, 4.2] (z_grid=[2.6, 2.8, 3.0, 3.2, 3.4, 3.6, 3.8, 4.0, 4.2])
k-grid: linspace(0.005, 0.064, 32) s/km, cov: 5.0%·P_F(fid, k) per z.
priors: {'hub': 0.015, 'omegamh2': 0.001, 'bhfeedback': 0.005}
hybrid vs HF GP at fid: 0.0000%
multi-D eq complexity: 25, flux_norm loss: 0.585

| param | in subset? | GP σ | hybrid σ | hybrid/GP ratio |
|---|---|---|---|---|
| tau0 | GP-slice | 0.0159 | 0.0181 | **1.14×** |
| ns | ✓ multi-D | 0.0258 | nan | **nan×** |
| Ap | ✓ multi-D | 0.11 | 9.58e+06 | **87021178.94×** |
| herei | ✓ multi-D | 0.0677 | nan | **nan×** |
| heref | ✓ multi-D | 0.206 | nan | **nan×** |
| alphaq | ✓ multi-D | 0.241 | 6.28e+19 | **260791642155434770432.00×** |
| hub | GP-slice | 0.00591 | 0.00589 | **1.00×** |
| omegamh2 | GP-slice | 0.000956 | 0.000922 | **0.96×** |
| hireionz | ✓ multi-D | 0.675 | 5.54e+19 | **82096044201251127296.00×** |
| bhfeedback | GP-slice | 0.00468 | 0.00451 | **0.96×** |

## Target subset ('Ap', 'ns', 'tau0', 'dtau0')
  - **Ap**: ratio = 87021178.94× (multi-D)
  - **ns**: ratio = nan× (multi-D)
  - **tau0**: ratio = 1.14× (GP-slice)
  - **dtau0**: fixed at fid=0
