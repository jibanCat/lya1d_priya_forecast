# Multi-D cross-coupled PySR equation

**Subset**: ['ns', 'Ap', 'herei', 'heref', 'alphaq', 'hireionz']
**Inputs** (in order): x0=θ_ns, x1=θ_Ap, x2=θ_herei, x3=θ_heref, x4=θ_alphaq, x5=θ_hireionz, x6=k, x7=r, x8=z

**Trained equation** (raw):
```
(exp(((x2 + (x3 / (x1 * -3.936307))) + (x6 * -5.1878595)) + 0.59731084) - 2.2224786) + (((x0 * 2.5508106) - x7) + x6)
```

**Trained equation (variables prettified)**:
```
(exp(((θ_herei + (θ_heref / (θ_Ap * -3.936307))) + (k * -5.1878595)) + 0.59731084) - 2.2224786) + (((θ_ns * 2.5508106) - r) + k)
```

complexity = 24
flux_norm loss = 0.554
LF rel-err = 2.69%, HF rel-err = 2.85%
