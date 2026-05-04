## Per-param 1D PySR fits at z = 3.4000000000000004

## Per-param fit statistics

| param | complexity | flux_norm loss | LF rel-err | HF rel-err | LF max | HF max | x0? | x3? |
|---|---|---|---|---|---|---|---|---|
| dtau0 | 17 | 3.09e+26 | 2528965306848.69% | 10402882195243.74% | 35882809667639.30% | 149034901954984.16% | ✓ | ✓ |
| tau0 | 19 | 0.0051 | 0.72% | 1.32% | 5.52% | 8.17% | ✓ | — |
| ns | 17 | 0.217 | 0.64% | 1.29% | 2.46% | 8.49% | ✓ | — |
| Ap | 20 | 3.65 | 1.68% | 2.06% | 39.74% | 24.21% | ✓ | ✓ |
| herei | 18 | 12.1 | 0.49% | 2.15% | 7.01% | 15.41% | ✓ | ✓ |
| heref | 20 | 4.57 | 0.88% | 1.98% | 7.30% | 9.20% | ✓ | ✓ |
| alphaq | 20 | 17.2 | 0.77% | 2.49% | 3.39% | 12.87% | ✓ | ✓ |
| hub | 20 | 6.64 | 0.48% | 1.38% | 4.32% | 14.71% | ✓ | ✓ |
| omegamh2 | 19 | 26.3 | 0.34% | 1.24% | 2.32% | 11.82% | ✗ | ✓ |
| hireionz | 20 | 8.97 | 0.31% | 1.04% | 1.76% | 5.56% | ✗ | ✓ |
| bhfeedback | 19 | 14.3 | 0.22% | 0.92% | 1.43% | 5.46% | ✗ | ✓ |

## Full equations

Inputs: `x0 = (theta - prior_lo)/(prior_hi - prior_lo)`, `x1 = (k - k_min)/(k_max - k_min)`, `x2 = resolution` (LF=0.4, HF=0.8), `x3 = (z - z_min)/(z_max - z_min)` (multi-z fits only).

### `dtau0`  (θ_fid = -0.009, complexity = 17, flux_norm loss = 3.09e+26)

```
(square(square((sqrt(z) ^ k) - square(z)) * r) * -8.147095e13) - θ
```

### `tau0`  (θ_fid = 1.09, complexity = 19, flux_norm loss = 0.0051)

```
inv(inv(((θ * 3.3952036) - ((((r / 1.3454579) + -0.30785945) / exp(k)) / exp(k))) + -2.3109946))
```

### `ns`  (θ_fid = 0.983, complexity = 17, flux_norm loss = 0.217)

```
((θ - 0.6704223) / 0.32712656) - ((((r * 0.3299933) + -0.13978459) / (k + 0.02500174)) + 0.2567563)
```

### `Ap`  (θ_fid = 1.46, complexity = 20, flux_norm loss = 3.65)

```
(((k ^ θ) * -3.1838768) - (θ + (-1.0301452 / r))) * exp(square(square(z) * 1.1001283))
```

### `herei`  (θ_fid = 4, complexity = 18, flux_norm loss = 12.1)

```
(θ / r) - exp(r / ((square(square(k / r) + z) + 1.0407312) - r))
```

### `heref`  (θ_fid = 2.765, complexity = 20, flux_norm loss = 4.57)

```
((θ + -5.3854647) * ((r + -0.39485148) / ((z ^ z) - (0.57644606 - square(k))))) - -0.5970319
```

### `alphaq`  (θ_fid = 1.74, complexity = 20, flux_norm loss = 17.2)

```
z + ((((square(k) * square(θ / 0.31957385)) + -0.7407582) - exp((z * 4.386378) * r)) * r)
```

### `hub`  (θ_fid = 0.688, complexity = 20, flux_norm loss = 6.64)

```
r * ((exp(k * (θ - -1.3422002)) - (r * exp((r * exp(z)) + r))) - 2.1821194)
```

### `omegamh2`  (θ_fid = 0.1439, complexity = 19, flux_norm loss = 26.3)

```
(((exp((z * k) * 1.868859) - square(k / 0.80250233)) * -27.006193) - -9.462707) * (r - 0.40290532)
```

### `hireionz`  (θ_fid = 7.24, complexity = 20, flux_norm loss = 8.97)

```
(((((0.69243765 - z) * (k + -0.80715674)) * 3.9594388) - square(z)) * ((r * 39.495735) + -15.472661)) - -0.86672443
```

### `bhfeedback`  (θ_fid = 0.05, complexity = 19, flux_norm loss = 14.3)

```
((r * -29.825314) - -12.571616) * (square(exp(sqrt(r) + (r - exp(z + k)))) + z)
```

