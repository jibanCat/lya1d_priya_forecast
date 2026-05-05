# Original Design 11D forecast scorecard

z = 3.6, mock_data = 'gp' (GP at fid as data), Fisher.

4 published 1D PySR equations (dtau0/Ap/ns/alphaq) × multiplicative combine + 
GP-slice fallback for the other 7 params.


| param | GP σ | hybrid σ | hybrid/GP ratio | has PySR eq? |
|---|---|---|---|---|
| dtau0 | nan | 7.22 | **inf×** | ✓ published |
| tau0 | nan | 0.685 | **inf×** | GP-slice fallback |
| ns | 0.156 | 0.748 | **4.79×** | ✓ published |
| Ap | 1.63 | 2.94 | **1.80×** | ✓ published |
| herei | 3.01 | 2.14 | **0.71×** | GP-slice fallback |
| heref | 4.36 | 2.83 | **0.65×** | GP-slice fallback |
| alphaq | 6.41 | 3.36e+12 | **523964323811.22×** | ✓ published |
| hub | 0.098 | 0.12 | **1.22×** | GP-slice fallback |
| omegamh2 | 0.0395 | 0.0496 | **1.26×** | GP-slice fallback |
| hireionz | 6.27 | 5.56 | **0.89×** | GP-slice fallback |
| bhfeedback | 0.327 | 0.403 | **1.23×** | GP-slice fallback |

## Target subset ('Ap', 'ns', 'tau0', 'dtau0') (from user's Comment 1):
  - **Ap**: ratio = 1.80×  (PySR)
  - **ns**: ratio = 4.79×  (PySR)
  - **tau0**: ratio = nan×  (GP-slice)
  - **dtau0**: ratio = nan×  (PySR)
