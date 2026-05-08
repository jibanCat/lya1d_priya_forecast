# Resolution correction per dimension

Per-param LF→HF lift evaluated at each parameter's physical fiducial value, with all other params held at fid.

z_eval = 3.4000000000000004

**Paper form (multiplicative)**:

    R_i(k) = P_F^HF(fid_i, k) / P_F^LF(fid_i, k)

R is the multiplicative correction that lifts the LF emulator's
prediction to the HF emulator's prediction at θ = fid_i_phys.

| param | R_min | R_max | R_mean | Δ_PF abs-max | Δ_PF mean (signed) |
|---|---|---|---|---|---|
| tau0 | 0.9534 | 0.9933 | 0.9800 | 2.01 | -0.405 |
| ns | 0.9514 | 0.9933 | 0.9815 | 1.87 | -0.387 |
| Ap | 0.9232 | 0.9957 | 0.9736 | 3.86 | -0.549 |
| herei | 0.8887 | 0.9937 | 0.9545 | 4.82 | -0.998 |
| alphaq | 0.9475 | 0.9989 | 0.9781 | 1.28 | -0.31 |
| hub | 0.9430 | 0.9895 | 0.9726 | 2.43 | -0.424 |
