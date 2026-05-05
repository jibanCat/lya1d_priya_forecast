# Multi-D cross-coupled PySR equation

**Subset**: ['ns', 'Ap', 'herei', 'heref', 'alphaq', 'hireionz']
**Inputs** (in order): x0=θ_ns, x1=θ_Ap, x2=θ_herei, x3=θ_heref, x4=θ_alphaq, x5=θ_hireionz, x6=k, x7=r, x8=z

**Trained equation** (raw):
```
(sqrt(x2 * x1) * (1.2668293 - sqrt(x3 * sqrt(x6)))) + ((square(x6) - x7) - ((sqrt(sqrt(x6)) - x0) / 0.3931706))
```

**Trained equation (variables prettified)**:
```
(sqrt(θ_herei * θ_Ap) * (1.2668293 - sqrt(θ_heref * sqrt(k)))) + ((square(k) - r) - ((sqrt(sqrt(k)) - θ_ns) / 0.3931706))
```

complexity = 25
flux_norm loss = 0.585
LF rel-err = 2.79%, HF rel-err = 3.00%
