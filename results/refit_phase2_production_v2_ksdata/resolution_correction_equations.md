# Resolution correction equations per dimension

For each parameter $i$, the LF→HF resolution correction is the
multiplicative ratio of the per-param emulator's HF and LF
predictions at the same $(\theta, k, z)$:

$$R_i(\theta, k, z) \;=\; \frac{P_F^{HF}(\theta_i, k, z)}{P_F^{LF}(\theta_i, k, z)} \;=\; \frac{f_i(x_0, x_1, 0.8, x_3)\,\sigma_F(k, z) + \mu_F(k, z)}{f_i(x_0, x_1, 0.4, x_3)\,\sigma_F(k, z) + \mu_F(k, z)}$$

where $f_i$ is the trained PySR equation, $(\mu_F, \sigma_F)$ is the
per-(k, z) anchor / std from the LF emulator. Below we report
the equation evaluated at $x_2=0.8$ (HF) and $x_2=0.4$ (LF), and
the simplified `HF − LF` in flux_norm space.

### `ns` (θ_fid = 0.983)

Trained equation (variables: θ, k, r=resolution, z):
```
(k * -3.2194026) + ((θ * 3.0578291) + ((log(square(k + 0.020076722)) + 0.5130467) * (r + k)))
```

At HF (r = 0.8):
```
3.0578291*θ - 3.2194026*k + (k + 0.8)*(log(square(k + 0.020076722)) + 0.5130467)
```
At LF (r = 0.4):
```
3.0578291*θ - 3.2194026*k + (k + 0.4)*(log(square(k + 0.020076722)) + 0.5130467)
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
0.4*log(square(k + 0.020076722)) + 0.20521868
```

### `Ap` (θ_fid = 1.46)

Trained equation (variables: θ, k, r=resolution, z):
```
exp(square(z - -0.24380434)) * ((exp((θ / 1.1384475) + (k * -3.7485268)) + (r / -0.44109055)) - -0.44676068)
```

At HF (r = 0.8):
```
(exp(0.878389209866946*θ - 3.7485268*k) - 1.36692587483551)*exp(square(z + 0.24380434))
```
At LF (r = 0.4):
```
(exp(0.878389209866946*θ - 3.7485268*k) - 0.460082597417755)*exp(square(z + 0.24380434))
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-0.906843277417755*exp(square(z + 0.24380434))
```

### `herei` (θ_fid = 4)

Trained equation (variables: θ, k, r=resolution, z):
```
(((r * -5.1571126) + 2.0720875) / (square(k) + (square(z) - -0.07234178))) + θ
```

At HF (r = 0.8):
```
θ - 2.05360258/(square(k) + square(z) + 0.07234178)
```
At LF (r = 0.4):
```
θ + 0.00924245999999984/(square(k) + square(z) + 0.07234178)
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-2.06284504/(square(k) + square(z) + 0.07234178)
```

### `heref` (θ_fid = 2.765)

Trained equation (variables: θ, k, r=resolution, z):
```
((r + 0.42855084) + (((5.8619432 - square(z)) * square(square(r))) / (r - exp(square(k))))) - θ
```

At HF (r = 0.8):
```
((1.22855084 - θ)*(exp(square(k)) - 0.8) + (square(z) - 5.8619432)*square(square(0.8)))/(exp(square(k)) - 0.8)
```
At LF (r = 0.4):
```
((0.82855084 - θ)*(exp(square(k)) - 0.4) + (square(z) - 5.8619432)*square(square(0.4)))/(exp(square(k)) - 0.4)
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
((-(θ - 1.22855084)*(exp(square(k)) - 0.8) + (square(z) - 5.8619432)*square(square(0.8)))*(exp(square(k)) - 0.4) + ((θ - 0.82855084)*(exp(square(k)) - 0.4) - (square(z) - 5.8619432)*square(square(0.4)))*(exp(square(k)) - 0.8))/((exp(square(k)) - 0.8)*(exp(square(k)) - 0.4))
```

### `alphaq` (θ_fid = 1.74)

Trained equation (variables: θ, k, r=resolution, z):
```
((θ + k) - exp(z)) + ((square(z + -0.3296705) / (r - 0.77130294)) * -1.5089505)
```

At HF (r = 0.8):
```
θ + k - 52.5820589286846*square(z - 0.3296705) - exp(z)
```
At LF (r = 0.4):
```
θ + k + 4.06393361711599*square(z - 0.3296705) - exp(z)
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-56.6459925458006*square(z - 0.3296705)
```

### `hub` (θ_fid = 0.688)

Trained equation (variables: θ, k, r=resolution, z):
```
(θ + ((((k * 4.891668) + ((r * exp(square(z))) / -0.10375503)) * r) + z)) + z
```

At HF (r = 0.8):
```
θ + 3.9133344*k + 2*z - 6.16837564405311*exp(square(z))
```
At LF (r = 0.4):
```
θ + 1.9566672*k + 2*z - 1.54209391101328*exp(square(z))
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
1.9566672*k - 4.62628173303983*exp(square(z))
```

### `bhfeedback` (θ_fid = 0.05)

Trained equation (variables: θ, k, r=resolution, z):
```
(4.1324344 - exp((r * 3.8957348) - k)) - θ
```

At HF (r = 0.8):
```
-θ + 4.1324344 - 22.5692382566343*exp(-k)
```
At LF (r = 0.4):
```
-θ + 4.1324344 - 4.75070923722282*exp(-k)
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-17.8185290194114*exp(-k)
```

