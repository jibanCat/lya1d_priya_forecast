| Parameter | GP σ | per_param_1d_o4 σ | per_param_1d_o4 / GP |
|---|---|---|---|
| dtau0 | 0.428 | 3.8 | 8.864 |
| ns | 0.0618 | 1.73 | 27.915 |
| Ap | 0.745 | 13.3 | 17.831 |
| alphaq | 5.42 | 109 | 20.154 |


### Reward (lower = better)
- worst σ_pysr / σ_gp across params: **27.9**
- geometric-mean σ_pysr / σ_gp:      **17.3**

Targets to chase: keep retraining PySR (more iterations, larger maxsize, different operators) until the geometric mean drops below ~2.0 — that's where the equations start preserving most of the GP's information.