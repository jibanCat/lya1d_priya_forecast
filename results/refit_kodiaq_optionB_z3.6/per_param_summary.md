## Per-param 1D PySR fits at z = 3.6

Target: mean rel-err < 1.0% on each fidelity.
Pass: 7 / 11 params.

## Per-param fit statistics

| param | complexity | flux_norm loss | LF rel-err | HF rel-err | LF max | HF max | x0? | x3? |
|---|---|---|---|---|---|---|---|---|
| dtau0 | 17 | 0.0204 | 0.31% | 0.51% | 1.17% | 1.17% | ✓ | — |
| tau0 | 19 | 0.00164 | 0.51% | 0.67% | 1.77% | 2.62% | ✓ | — |
| ns | 18 | 0.0963 | 0.89% | 1.03% | 2.44% | 3.90% | ✓ | — |
| Ap | 19 | 1.01 | 0.77% | 1.23% | 5.57% | 9.33% | ✓ | — |
| herei | 20 | 0.838 | 0.75% | 1.62% | 3.27% | 8.38% | ✓ | — |
| heref | 20 | 1.93 | 0.86% | 1.24% | 5.05% | 8.87% | ✓ | — |
| alphaq | 20 | 2.92 | 0.64% | 0.99% | 2.75% | 6.56% | ✓ | — |
| hub | 20 | 3.38 | 0.33% | 0.74% | 2.98% | 8.57% | ✓ | — |
| omegamh2 | 16 | 7.49 | 0.29% | 0.67% | 1.56% | 3.18% | ✓ | — |
| hireionz | 20 | 2.3 | 0.38% | 0.59% | 2.96% | 3.40% | ✓ | — |
| bhfeedback | 18 | 1.52 | 0.19% | 0.44% | 1.33% | 1.51% | ✓ | — |

## Full equations

Inputs: `x0 = (theta - prior_lo)/(prior_hi - prior_lo)`, `x1 = (k - k_min)/(k_max - k_min)`, `x2 = resolution` (LF=0.4, HF=0.8), `x3 = (z - z_min)/(z_max - z_min)` (multi-z fits only).

### `dtau0`  (θ_fid = -0.009, complexity = 17, flux_norm loss = 0.0204)

```
((((θ * 3.3510156) + k) + -1.3054118) - ((k - 0.2887231) * k)) - (r * 1.6961963)
```

### `tau0`  (θ_fid = 1.09, complexity = 19, flux_norm loss = 0.00164)

```
((θ ^ 1.0784132) * 3.317922) - sqrt(((1.0789349 / sqrt(r + k)) * r) + 2.1407413)
```

### `ns`  (θ_fid = 0.983, complexity = 18, flux_norm loss = 0.0963)

```
θ + (((r + 0.17106605) * sqrt(k)) - (1.2299212 - ((θ + θ) - (r / 0.6163377))))
```

### `Ap`  (θ_fid = 1.46, complexity = 19, flux_norm loss = 1.01)

```
(((r - k) * (θ - 0.79541785)) * 7.7039657) - (k + (r - (0.46845996 / (k + 0.24289803))))
```

### `herei`  (θ_fid = 4, complexity = 20, flux_norm loss = 0.838)

```
(((((square(k) * 0.71120685) + r) * -10.48077) - -9.03554) * ((θ + r) - 0.86003613)) - (r * 0.82994306)
```

### `heref`  (θ_fid = 2.765, complexity = 20, flux_norm loss = 1.93)

```
(r * (k + -0.93449724)) * (square(square((r * (4.8761096 - θ)) - (k * square(k)))) * 0.09019537)
```

### `alphaq`  (θ_fid = 1.74, complexity = 20, flux_norm loss = 2.92)

```
(square(k) * exp(θ)) + log(sqrt(square(square(log((r ^ -0.8880627) - (k + -0.38398525))))))
```

### `hub`  (θ_fid = 0.688, complexity = 20, flux_norm loss = 3.38)

```
r * ((((r * r) * -3.9638143) / log(square(k * (θ + 0.29082415)) + inv(r))) + θ)
```

### `omegamh2`  (θ_fid = 0.1439, complexity = 16, flux_norm loss = 7.49)

```
((((k * (k * 10.069986)) + 3.205511) * (inv(r) + -2.4996257)) + θ) + -0.5029485
```

### `hireionz`  (θ_fid = 7.24, complexity = 20, flux_norm loss = 2.3)

```
exp(k ^ θ) + ((k - -4.4723105) + ((r * (k ^ k)) / -0.045948666))
```

### `bhfeedback`  (θ_fid = 0.05, complexity = 18, flux_norm loss = 1.52)

```
(k - θ) + ((((k ^ k) / -0.040693462) + exp(θ)) * (r + -0.3926082))
```


