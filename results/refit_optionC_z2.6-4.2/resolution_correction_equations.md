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
(square(square((sqrt(z) ^ k) - square(z)) * r) * -8.147095e13) - θ
```

At HF (r = 0.8):
```
-θ - 81470950000000.0*square(0.8*square(z**(k/2) - square(z)))
```
At LF (r = 0.4):
```
-θ - 81470950000000.0*square(0.4*square(z**(k/2) - square(z)))
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
81470950000000.0*square(0.4*square(z**(k/2) - square(z))) - 81470950000000.0*square(0.8*square(z**(k/2) - square(z)))
```

### `tau0` (θ_fid = 1.09)

Trained equation (variables: θ, k, r=resolution, z):
```
inv(inv(((θ * 3.3952036) - ((((r / 1.3454579) + -0.30785945) / exp(k)) / exp(k))) + -2.3109946))
```

At HF (r = 0.8):
```
inv(inv(3.3952036*θ - 2.3109946 - 0.286733662129335*exp(-2*k)))
```
At LF (r = 0.4):
```
inv(inv(3.3952036*θ - 2.3109946 + 0.0105628939353323*exp(-2*k)))
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
inv(inv(3.3952036*θ - 2.3109946 - 0.286733662129335*exp(-2*k))) - inv(inv(3.3952036*θ - 2.3109946 + 0.0105628939353323*exp(-2*k)))
```

### `ns` (θ_fid = 0.983)

Trained equation (variables: θ, k, r=resolution, z):
```
((θ - 0.6704223) / 0.32712656) - ((((r * 0.3299933) + -0.13978459) / (k + 0.02500174)) + 0.2567563)
```

At HF (r = 0.8):
```
((3.05692084433621*θ - 2.30618420337782)*(k + 0.02500174) - 0.12421005)/(k + 0.02500174)
```
At LF (r = 0.4):
```
((3.05692084433621*θ - 2.30618420337782)*(k + 0.02500174) + 0.00778726999999999)/(k + 0.02500174)
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-0.13199732/(k + 0.02500174)
```

### `Ap` (θ_fid = 1.46)

Trained equation (variables: θ, k, r=resolution, z):
```
(((k ^ θ) * -3.1838768) - (θ + (-1.0301452 / r))) * exp(square(square(z) * 1.1001283))
```

At HF (r = 0.8):
```
(-θ - 3.1838768*k**θ + 1.2876815)*exp(square(1.1001283*square(z)))
```
At LF (r = 0.4):
```
(-θ - 3.1838768*k**θ + 2.575363)*exp(square(1.1001283*square(z)))
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-1.2876815*exp(square(1.1001283*square(z)))
```

### `herei` (θ_fid = 4)

Trained equation (variables: θ, k, r=resolution, z):
```
(θ / r) - exp(r / ((square(square(k / r) + z) + 1.0407312) - r))
```

At HF (r = 0.8):
```
1.25*θ - exp(0.8/(square(z + square(1.25*k)) + 0.2407312))
```
At LF (r = 0.4):
```
2.5*θ - exp(0.4/(square(z + square(2.5*k)) + 0.6407312))
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-1.25*θ - exp(0.8/(square(z + square(1.25*k)) + 0.2407312)) + exp(0.4/(square(z + square(2.5*k)) + 0.6407312))
```

### `heref` (θ_fid = 2.765)

Trained equation (variables: θ, k, r=resolution, z):
```
((θ + -5.3854647) * ((r + -0.39485148) / ((z ^ z) - (0.57644606 - square(k))))) - -0.5970319
```

At HF (r = 0.8):
```
(0.40514852*θ + 0.5970319*z**z + 0.5970319*square(k) - 2.52606973916656)/(z**z + square(k) - 0.57644606)
```
At LF (r = 0.4):
```
(0.00514852000000005*θ + 0.5970319*z**z + 0.5970319*square(k) - 0.371883859166558)/(z**z + square(k) - 0.57644606)
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
(0.4*θ - 2.15418588)/(z**z + square(k) - 0.57644606)
```

### `alphaq` (θ_fid = 1.74)

Trained equation (variables: θ, k, r=resolution, z):
```
z + ((((square(k) * square(θ / 0.31957385)) + -0.7407582) - exp((z * 4.386378) * r)) * r)
```

At HF (r = 0.8):
```
z + 0.8*square(3.12916717059296*θ)*square(k) - 0.8*exp(3.5091024*z) - 0.59260656
```
At LF (r = 0.4):
```
z + 0.4*square(3.12916717059296*θ)*square(k) - 0.4*exp(1.7545512*z) - 0.29630328
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
0.4*square(3.12916717059296*θ)*square(k) + 0.4*exp(1.7545512*z) - 0.8*exp(3.5091024*z) - 0.29630328
```

### `hub` (θ_fid = 0.688)

Trained equation (variables: θ, k, r=resolution, z):
```
r * ((exp(k * (θ - -1.3422002)) - (r * exp((r * exp(z)) + r))) - 2.1821194)
```

At HF (r = 0.8):
```
0.8*exp(k*(θ + 1.3422002)) - 1.42434619423518*exp(0.8*exp(z)) - 1.74569552
```
At LF (r = 0.4):
```
0.4*exp(k*(θ + 1.3422002)) - 0.238691951622603*exp(0.4*exp(z)) - 0.87284776
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
0.4*exp(k*(θ + 1.3422002)) + 0.238691951622603*exp(0.4*exp(z)) - 1.42434619423518*exp(0.8*exp(z)) - 0.87284776
```

### `omegamh2` (θ_fid = 0.1439)

Trained equation (variables: θ, k, r=resolution, z):
```
(((exp((z * k) * 1.868859) - square(k / 0.80250233)) * -27.006193) - -9.462707) * (r - 0.40290532)
```

At HF (r = 0.8):
```
10.7240155673532*square(1.24610230103631*k) - 10.7240155673532*exp(1.868859*k*z) + 3.75759060809876
```
At LF (r = 0.4):
```
-0.0784616326467597*square(1.24610230103631*k) + 0.0784616326467597*exp(1.868859*k*z) - 0.0274921919012399
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
10.8024772*square(1.24610230103631*k) - 10.8024772*exp(1.868859*k*z) + 3.7850828
```

### `hireionz` (θ_fid = 7.24)

Trained equation (variables: θ, k, r=resolution, z):
```
(((((0.69243765 - z) * (k + -0.80715674)) * 3.9594388) - square(z)) * ((r * 39.495735) + -15.472661)) - -0.86672443
```

At HF (r = 0.8):
```
-63.8417021721676*(k - 0.80715674)*(z - 0.69243765) - 16.123927*square(z) + 0.86672443
```
At LF (r = 0.4):
```
-1.28932393476041*(k - 0.80715674)*(z - 0.69243765) - 0.325633000000002*square(z) + 0.86672443
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-62.5523782374072*(k - 0.80715674)*(z - 0.69243765) - 15.798294*square(z)
```

### `bhfeedback` (θ_fid = 0.05)

Trained equation (variables: θ, k, r=resolution, z):
```
((r * -29.825314) - -12.571616) * (square(exp(sqrt(r) + (r - exp(z + k)))) + z)
```

At HF (r = 0.8):
```
-11.2886352*z - 11.2886352*square(5.44352697076298*exp(-exp(k + z)))
```
At LF (r = 0.4):
```
0.6414904*z + 0.6414904*square(2.80795239320684*exp(-exp(k + z)))
```
Resolution correction in flux_norm space (HF − LF, simplified):
```
-11.9301256*z - 0.6414904*square(2.80795239320684*exp(-exp(k + z))) - 11.2886352*square(5.44352697076298*exp(-exp(k + z)))
```

