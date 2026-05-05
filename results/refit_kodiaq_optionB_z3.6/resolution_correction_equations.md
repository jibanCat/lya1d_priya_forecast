# Resolution correction equations per dimension

For each parameter $i$, the LF→HF resolution correction is the
multiplicative ratio of the per-param emulator's HF and LF
predictions at the same $(\theta, k, z)$:

$$R_i(\theta, k, z) \;=\; \frac{P_F^{HF}(\theta_i, k, z)}{P_F^{LF}(\theta_i, k, z)} \;=\; \frac{f_i(x_0, x_1, 0.8, x_3)\,\sigma_F(k, z) + \mu_F(k, z)}{f_i(x_0, x_1, 0.4, x_3)\,\sigma_F(k, z) + \mu_F(k, z)}$$

where $f_i$ is the trained PySR equation, $(\mu_F, \sigma_F)$ is the
per-(k, z) anchor / std from the LF emulator. Below we report
the equation evaluated at $x_2=0.8$ (HF) and $x_2=0.4$ (LF), and
the simplified `HF − LF` in flux_norm space.

### `dtau0` (θ_fid = -0.009)

Trained equation (variables: θ, k, r=resolution, z):
```
((((θ * 3.3510156) + k) + -1.3054118) - ((k - 0.2887231) * k)) - (r * 1.6961963)
```

At HF (r = 0.8):
```
3.3510156*θ - k*(k - 0.2887231) + k - 2.66236884
```
At LF (r = 0.4):
```
3.3510156*θ - k*(k - 0.2887231) + k - 1.98389032
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-0.678478520000000
```

### `tau0` (θ_fid = 1.09)

Trained equation (variables: θ, k, r=resolution, z):
```
((θ ^ 1.0784132) * 3.317922) - sqrt(((1.0789349 / sqrt(r + k)) * r) + 2.1407413)
```

At HF (r = 0.8):
```
3.317922*θ**1.0784132 - 1.46312723301837*sqrt(1 + 0.403200480132746/sqrt(k + 0.8))
```
At LF (r = 0.4):
```
3.317922*θ**1.0784132 - 1.46312723301837*sqrt(1 + 0.201600240066373/sqrt(k + 0.4))
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
1.46312723301837*sqrt(1 + 0.201600240066373/sqrt(k + 0.4)) - 1.46312723301837*sqrt(1 + 0.403200480132746/sqrt(k + 0.8))
```

### `ns` (θ_fid = 0.983)

Trained equation (variables: θ, k, r=resolution, z):
```
θ + (((r + 0.17106605) * sqrt(k)) - (1.2299212 - ((θ + θ) - (r / 0.6163377))))
```

At HF (r = 0.8):
```
3*θ + 0.97106605*sqrt(k) - 2.52791092219288
```
At LF (r = 0.4):
```
3*θ + 0.57106605*sqrt(k) - 1.87891606109644
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
0.4*sqrt(k) - 0.648994861096441
```

### `Ap` (θ_fid = 1.46)

Trained equation (variables: θ, k, r=resolution, z):
```
(((r - k) * (θ - 0.79541785)) * 7.7039657) - (k + (r - (0.46845996 / (k + 0.24289803))))
```

At HF (r = 0.8):
```
(-(k + 0.24289803)*(k + 7.7039657*(θ - 0.79541785)*(k - 0.8) + 0.8) + 0.46845996)/(k + 0.24289803)
```
At LF (r = 0.4):
```
(-(k + 0.24289803)*(k + 7.7039657*(θ - 0.79541785)*(k - 0.4) + 0.4) + 0.46845996)/(k + 0.24289803)
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
3.08158628*θ - 2.8511487334271
```

### `herei` (θ_fid = 4)

Trained equation (variables: θ, k, r=resolution, z):
```
(((((square(k) * 0.71120685) + r) * -10.48077) - -9.03554) * ((θ + r) - 0.86003613)) - (r * 0.82994306)
```

At HF (r = 0.8):
```
-(θ - 0.06003613)*(7.4539954172745*square(k) - 0.650924) - 0.663954448
```
At LF (r = 0.4):
```
-(θ - 0.46003613)*(7.4539954172745*square(k) - 4.843232) - 0.331977224
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-4.192308*θ - 2.9815981669098*square(k) + 1.85700552408804
```

### `heref` (θ_fid = 2.765)

Trained equation (variables: θ, k, r=resolution, z):
```
(r * (k + -0.93449724)) * (square(square((r * (4.8761096 - θ)) - (k * square(k)))) * 0.09019537)
```

At HF (r = 0.8):
```
0.072156296*(k - 0.93449724)*square(square(-0.8*θ - k*square(k) + 3.90088768))
```
At LF (r = 0.4):
```
0.036078148*(k - 0.93449724)*square(square(-0.4*θ - k*square(k) + 1.95044384))
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
(k - 0.93449724)*(0.072156296*square(square(-0.8*θ - k*square(k) + 3.90088768)) - 0.036078148*square(square(-0.4*θ - k*square(k) + 1.95044384)))
```

### `alphaq` (θ_fid = 1.74)

Trained equation (variables: θ, k, r=resolution, z):
```
(square(k) * exp(θ)) + log(sqrt(square(square(log((r ^ -0.8880627) - (k + -0.38398525))))))
```

At HF (r = 0.8):
```
square(k)*exp(θ) + log(square(square(log(1.60314935571669 - k))))/2
```
At LF (r = 0.4):
```
square(k)*exp(θ) + log(square(square(log(2.64027919819367 - k))))/2
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
log(square(square(log(1.60314935571669 - k))))/2 - log(square(square(log(2.64027919819367 - k))))/2
```

### `hub` (θ_fid = 0.688)

Trained equation (variables: θ, k, r=resolution, z):
```
r * ((((r * r) * -3.9638143) / log(square(k * (θ + 0.29082415)) + inv(r))) + θ)
```

At HF (r = 0.8):
```
0.8*θ - 2.0294729216/log(inv(0.8) + square(θ*k + 0.29082415*k))
```
At LF (r = 0.4):
```
0.4*θ - 0.2536841152/log(inv(0.4) + square(θ*k + 0.29082415*k))
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
0.4*θ - 2.0294729216/log(inv(0.8) + square(θ*k + 0.29082415*k)) + 0.2536841152/log(inv(0.4) + square(θ*k + 0.29082415*k))
```

### `omegamh2` (θ_fid = 0.1439)

Trained equation (variables: θ, k, r=resolution, z):
```
((((k * (k * 10.069986)) + 3.205511) * (inv(r) + -2.4996257)) + θ) + -0.5029485
```

At HF (r = 0.8):
```
θ + (10.069986*k**2 + 3.205511)*(inv(0.8) - 2.4996257) - 0.5029485
```
At LF (r = 0.4):
```
θ + (10.069986*k**2 + 3.205511)*(inv(0.4) - 2.4996257) - 0.5029485
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
(10.069986*k**2 + 3.205511)*(-inv(0.4) + inv(0.8))
```

### `hireionz` (θ_fid = 7.24)

Trained equation (variables: θ, k, r=resolution, z):
```
exp(k ^ θ) + ((k - -4.4723105) + ((r * (k ^ k)) / -0.045948666))
```

At HF (r = 0.8):
```
k - 17.4107339699481*k**k + exp(k**θ) + 4.4723105
```
At LF (r = 0.4):
```
k - 8.70536698497406*k**k + exp(k**θ) + 4.4723105
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-8.70536698497406*k**k
```

### `bhfeedback` (θ_fid = 0.05)

Trained equation (variables: θ, k, r=resolution, z):
```
(k - θ) + ((((k ^ k) / -0.040693462) + exp(θ)) * (r + -0.3926082))
```

At HF (r = 0.8):
```
-θ + k - 10.0112347285665*k**k + 0.4073918*exp(θ)
```
At LF (r = 0.4):
```
-θ + k - 0.181645886997769*k**k + 0.0073918*exp(θ)
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-9.8295888415687*k**k + 0.4*exp(θ)
```

