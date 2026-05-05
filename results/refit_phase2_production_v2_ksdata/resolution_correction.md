# Resolution correction per dimension

Per-param LF→HF lift evaluated at each parameter's physical fiducial value, with all other params held at fid.

z_eval = 3.4000000000000004

**Paper form (multiplicative)**:

    R_i(k) = P_F^HF(fid_i, k) / P_F^LF(fid_i, k)

R is the multiplicative correction that lifts the LF emulator's
prediction to the HF emulator's prediction at θ = fid_i_phys.

| param | R_min | R_max | R_mean | Δ_PF abs-max | Δ_PF mean (signed) |
|---|---|---|---|---|---|
| ns | 0.9444 | 1.0116 | 0.9806 | 1.83 | -0.466 |
| Ap | 0.9059 | 0.9952 | 0.9699 | 4.38 | -0.623 |
| herei | 0.8469 | 0.9931 | 0.9402 | 6.68 | -1.36 |
| heref | 0.9418 | 0.9906 | 0.9709 | 2.24 | -0.511 |
| alphaq | 0.9515 | 0.9990 | 0.9799 | 1.17 | -0.284 |
| hub | 0.9368 | 0.9891 | 0.9707 | 2.71 | -0.459 |
| bhfeedback | 0.9048 | 0.9988 | 0.9653 | 4.08 | -0.779 |
