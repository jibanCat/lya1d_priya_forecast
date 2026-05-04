# Resolution correction per dimension

Per-param LF→HF lift evaluated at each parameter's physical fiducial value, with all other params held at fid.

z_eval = 3.4000000000000004

**Paper form (multiplicative)**:

    R_i(k) = P_F^HF(fid_i, k) / P_F^LF(fid_i, k)

R is the multiplicative correction that lifts the LF emulator's
prediction to the HF emulator's prediction at θ = fid_i_phys.

| param | R_min | R_max | R_mean | Δ_PF abs-max | Δ_PF mean (signed) |
|---|---|---|---|---|---|
| dtau0 | 4.0000 | 4.0000 | 4.0000 | 1.15e+13 | -2.36e+12 |
| tau0 | 0.9534 | 0.9933 | 0.9800 | 2.01 | -0.405 |
| ns | 0.9514 | 0.9933 | 0.9815 | 1.87 | -0.387 |
| Ap | 0.9232 | 0.9957 | 0.9736 | 3.86 | -0.549 |
| herei | 0.8984 | 0.9968 | 0.9652 | 4.35 | -0.825 |
| heref | 0.9213 | 0.9888 | 0.9589 | 3.03 | -0.684 |
| alphaq | 0.8901 | 0.9977 | 0.9529 | 2.82 | -0.677 |
| hub | 0.9430 | 0.9895 | 0.9726 | 2.43 | -0.424 |
| omegamh2 | 0.9124 | 0.9929 | 0.9636 | 3.78 | -0.734 |
| hireionz | 0.9365 | 0.9944 | 0.9696 | 2.48 | -0.629 |
| bhfeedback | 0.9019 | 0.9990 | 0.9713 | 4.24 | -0.691 |
