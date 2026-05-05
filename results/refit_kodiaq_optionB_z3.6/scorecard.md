# Forecast: refit 1D PySR × additive-Taylor combine (mode=local_anchored)
emulator: /nfs/turbo/umor-yueyingn/mfho/birdgroup/lya_xq100/kodiaq_2_2_4_6-48-48
z = 3.6, niter = 20, maxsize = 20, resolution feature = (LF=0.4, HF=0.8).
k-grid: linspace(0.005, 0.064, 64) s/km, cov: 5%·P_F(fid, k) diagonal.
fixed params: ['dtau0'].

| param | GP σ | hybrid σ | hybrid/GP ratio | LF rel-err | HF rel-err | complexity |
|---|---|---|---|---|---|---|
| tau0 | 0.124 | 0.203 | **1.64×** | 0.51% | 0.67% | 19 |
| ns | 0.144 | 0.338 | **2.36×** | 0.89% | 1.03% | 18 |
| Ap | 0.747 | 0.77 | **1.03×** | 0.77% | 1.23% | 19 |
| herei | 2.22 | 0.606 | **0.27×** | 0.75% | 1.62% | 20 |
| heref | 2.21 | 2.33 | **1.05×** | 0.86% | 1.24% | 20 |
| alphaq | 3.58 | 9.26 | **2.58×** | 0.64% | 0.99% | 20 |
| hub | 0.0113 | 0.0149 | **1.32×** | 0.33% | 0.74% | 20 |
| omegamh2 | 0.000993 | 0.001 | **1.01×** | 0.29% | 0.67% | 16 |
| hireionz | 6.11 | 16 | **2.62×** | 0.38% | 0.59% | 20 |
| bhfeedback | 0.005 | 0.005 | **1.00×** | 0.19% | 0.44% | 18 |

## Target subset ('Ap', 'ns', 'tau0', 'dtau0')
  - **Ap**: ratio = 1.03×
  - **ns**: ratio = 2.36×
  - **tau0**: ratio = 1.64×
  - **dtau0**: fixed at fid=-0.009
