## Per-param fit statistics

| param | complexity | flux_norm loss | LF rel-err | HF rel-err | LF max | HF max | x0? | x3? |
|---|---|---|---|---|---|---|---|---|
| dtau0 | 17 | 3.09e+26 | 2528965306848.69% | 10402882195243.74% | 35882809667639.30% | 149034901954984.16% | ✓ | ✓ |
| tau0 | 20 | 0.139 | 5.14% | 5.41% | 21.95% | 22.95% | ✓ | — |
| ns | 19 | 0.423 | 0.74% | 1.50% | 3.31% | 9.87% | ✓ | — |
| Ap | 20 | 4.31 | 1.87% | 1.93% | 23.83% | 10.59% | ✓ | ✓ |
| herei | 15 | 14.4 | 1.06% | 3.16% | 8.37% | 15.05% | ✓ | ✓ |
| heref | 20 | 6.68 | 0.62% | 2.11% | 4.48% | 8.07% | ✓ | ✓ |
| alphaq | 17 | 17.1 | 0.98% | 2.85% | 6.09% | 25.63% | ✓ | ✓ |
| hub | 19 | 7.07 | 0.53% | 1.41% | 4.54% | 16.02% | ✓ | ✓ |
| omegamh2 | 19 | 25 | 0.35% | 1.13% | 2.37% | 10.83% | ✗ | ✓ |
| hireionz | 20 | 13.7 | 0.37% | 0.91% | 2.22% | 4.69% | ✗ | ✓ |
| bhfeedback | 10 | 29.2 | 0.28% | 1.49% | 2.02% | 6.48% | ✓ | — |

## Full equations

Inputs: `x0 = (theta - prior_lo)/(prior_hi - prior_lo)`, `x1 = (k - k_min)/(k_max - k_min)`, `x2 = resolution` (LF=0.4, HF=0.8), `x3 = (z - z_min)/(z_max - z_min)` (multi-z fits only).

### `dtau0`  (θ_fid = -0.009, complexity = 17, flux_norm loss = 3.09e+26)

```
(square(square((sqrt(z) ^ k) - square(z)) * r) * -8.147095e13) - θ
```

### `tau0`  (θ_fid = 1.09, complexity = 20, flux_norm loss = 0.139)

```
((((θ / 0.52568734) + -0.8739494) + (k / (exp((θ / 0.25306776) + -0.8948232) / 0.09724782))) / r) + -0.8736652
```

### `ns`  (θ_fid = 0.983, complexity = 19, flux_norm loss = 0.423)

```
(k * -3.2194026) + ((θ * 3.0578291) + ((log(square(k + 0.020076722)) + 0.5130467) * (r + k)))
```

### `Ap`  (θ_fid = 1.46, complexity = 20, flux_norm loss = 4.31)

```
exp(square(z - -0.24380434)) * ((exp((θ / 1.1384475) + (k * -3.7485268)) + (r / -0.44109055)) - -0.44676068)
```

### `herei`  (θ_fid = 4, complexity = 15, flux_norm loss = 14.4)

```
(((r * -5.1571126) + 2.0720875) / (square(k) + (square(z) - -0.07234178))) + θ
```

### `heref`  (θ_fid = 2.765, complexity = 20, flux_norm loss = 6.68)

```
((r + 0.42855084) + (((5.8619432 - square(z)) * square(square(r))) / (r - exp(square(k))))) - θ
```

### `alphaq`  (θ_fid = 1.74, complexity = 17, flux_norm loss = 17.1)

```
((θ + k) - exp(z)) + ((square(z + -0.3296705) / (r - 0.77130294)) * -1.5089505)
```

### `hub`  (θ_fid = 0.688, complexity = 19, flux_norm loss = 7.07)

```
(θ + ((((k * 4.891668) + ((r * exp(square(z))) / -0.10375503)) * r) + z)) + z
```

### `omegamh2`  (θ_fid = 0.1439, complexity = 19, flux_norm loss = 25)

```
((((k + (z - 0.29959577)) * ((z - 0.48878017) * k)) + 0.17089613) / (r - 0.8218645)) + z
```

### `hireionz`  (θ_fid = 7.24, complexity = 20, flux_norm loss = 13.7)

```
4.17097 - exp((r * 2.6200302) + square(square((z * -1.5229069) * r) - ((r + 0.45160598) - k)))
```

### `bhfeedback`  (θ_fid = 0.05, complexity = 10, flux_norm loss = 29.2)

```
(4.1324344 - exp((r * 3.8957348) - k)) - θ
```

