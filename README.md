# priya-forecast

PRIYA P1D forecast at a single redshift, with model swappable between the
PRIYA GP emulator and PySR-derived analytic equations. Includes a multi-D
PySR diagnostic and a reusable PySR HPO driver.

## Quick start (student-facing)

You trained PySR per parameter and have a directory full of
`hall_of_fame_*.csv` Pareto-front files. Wire them up like this:

```yaml
# configs/eqns/my_pysr.yaml
name: my_pysr
redshift: 3.6
combine: multiplicative
fiducial_p1d: data/priya_fiducial/p1d_z3.6.npz
parameters:
  ns:
    pareto_csv: pysr_outputs/hall_of_fame_ns_z3.6.csv
    pick: best_loss
    fiducial: 0.97
  hub:
    pareto_csv: pysr_outputs/hall_of_fame_hub_z3.6.csv
    pick: complexity_le:15
    fiducial: 0.6726
  # ... 11 entries total
```

Then forecast:

```
priya-forecast run --config configs/default.yaml --eqn configs/eqns/my_pysr.yaml --mode fisher
priya-forecast run --config configs/default.yaml --eqn configs/eqns/my_pysr.yaml --mode mcmc
```

To compare every YAML in `configs/eqns/` against the GP baseline:

```
priya-forecast compare --eqn-dir configs/eqns/ --output results/compare/
```

## Install

```
pip install -e ".[forecast,pysr,dev]"
```

## What's in here

- `src/priya_forecast/` — the library.
- `configs/` — YAML configs.
- `data/eboss_dr14/` — vendored eBOSS DR14 P1D data + covariance.
- `scripts/` — one-shot utilities (port PySR → forecast YAML, compare equation sets).
- `tests/` — unit tests + hypothesis property-based tests.

See `CLAUDE_CODE_INSTRUCTIONS.md` (in the parent repo) for the full design.
