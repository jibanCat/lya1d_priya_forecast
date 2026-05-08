# Multi-D cross-coupled PySR equation

**Subset**: ['ns', 'Ap', 'herei', 'heref', 'alphaq', 'hireionz']
**Inputs** (in order): x0=θ_ns, x1=θ_Ap, x2=θ_herei, x3=θ_heref, x4=θ_alphaq, x5=θ_hireionz, x6=k, x7=r, x8=z

**Trained equation** (raw):
```
((sqrt(x3) * -0.34080976) + ((x0 + (x0 - sqrt(sqrt(x6 / (x2 + x1))))) * 1.2618897)) - x7
```

**Trained equation (variables prettified)**:
```
((sqrt(θ_heref) * -0.34080976) + ((θ_ns + (θ_ns - sqrt(sqrt(k / (θ_herei + θ_Ap))))) * 1.2618897)) - r
```

complexity = 20
flux_norm loss = 0.531
LF rel-err = 2.93%, HF rel-err = 3.03%
