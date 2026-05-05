# Resolution correction per dimension

Per-param LF→HF lift evaluated at each parameter's physical fiducial value, with all other params held at fid.

single-z refit

**Paper form (multiplicative)**:

    R_i(k) = P_F^HF(fid_i, k) / P_F^LF(fid_i, k)

R is the multiplicative correction that lifts the LF emulator's
prediction to the HF emulator's prediction at θ = fid_i_phys.

| param | R_min | R_max | R_mean | Δ_PF abs-max | Δ_PF mean (signed) |
|---|---|---|---|---|---|
| dtau0 | 0.9752 | 0.9781 | 0.9772 | 1.15 | -0.374 |
| tau0 | 0.9848 | 0.9862 | 0.9852 | 0.691 | -0.242 |
| ns | 0.9737 | 0.9932 | 0.9794 | 0.798 | -0.344 |
| Ap | 0.8599 | 0.9944 | 0.9565 | 7.46 | -1.06 |
| herei | 0.9380 | 0.9915 | 0.9817 | 0.491 | -0.242 |
| heref | 0.9489 | 1.0048 | 0.9795 | 2.32 | -0.472 |
| alphaq | 0.9764 | 1.0007 | 0.9841 | 1 | -0.31 |
| hub | 0.8992 | 0.9875 | 0.9741 | 5.13 | -0.551 |
| omegamh2 | 0.9522 | 0.9926 | 0.9744 | 2.22 | -0.478 |
| hireionz | 0.9539 | 0.9791 | 0.9734 | 2.31 | -0.47 |
| bhfeedback | 0.9618 | 0.9869 | 0.9764 | 1.95 | -0.435 |
