# PySR performance analysis (for the paper methods section)

## Wall-time benchmark (kodiaq emulator, this repo's pipeline)

| Run | Inputs | Rows / param | niter | parallelism | Per-fit wall | All-11 wall |
|---|---|---|---|---|---|---|
| Single-z (this paper § 3.1) | (θ, k, res) | 6,400 | 20 | serial | ~100 s | ~25 min |
| Single-z, niter=50 | (θ, k, res) | 6,400 | 50 | multithread, procs=4 | ~25 s | ~5 min |
| Multi-z (§ 3.2) | (θ, k, res, z) | 14,400 | 50 | multithread, procs=4 | ~50 s | ~10 min |

Numbers measured on a Greatlakes login node; all the fits happen
genetic-evolution-style under `pysr.PySRRegressor`.

## Why PySR is the rate-limiting step

For each of the 11 PRIYA parameters we run one PySR `model.fit()`. Each
call has the following cost components, in order of impact:

1. **Julia startup overhead per call** — PySR is a Python wrapper around
   a Julia genetic-programming kernel. Each `PySRRegressor()` instance
   spins up a fresh Julia subprocess (≈ 5–15 s cold start). With one fit
   per parameter and 11 parameters, that's ~1–3 minutes of pure init
   overhead per pipeline run, before any evolutionary search.

2. **`parallelism="serial"` + `deterministic=True`** — for unit-test
   stability we default to single-threaded evolution. PySR's native
   default is multithreaded (uses all cores). We toggle this via the
   `--multi-z` / production code paths (see
   `src/priya_forecast/refit_1d_pysr.py::DEFAULT_PYSR_KWARGS`).

3. **Genetic search size** — defaults `populations=15`, `population_size=33`,
   `niterations=N`. Each iteration evaluates ~10⁵ candidate equations on
   the full training set. At our typical N=50 and with a 14k-row training
   set, the per-fit cost dominates over Julia startup.

4. **Operator cost** — `^`, `exp`, `log` are ~3× slower per evaluation
   than `+ * −`. We constrain `^` (`constraints={"^": (-1, 1)}`) and
   penalize it lightly (`complexity_of_operators={"^": 3}`) to make
   PySR prefer cheaper ops where possible.

5. **Datapoint count** — PySR explicitly warns above 10k datapoints
   ("you should also reconsider if you need that many datapoints"). The
   multi-z run with 14,400 rows per fit triggers this warning. We
   compensate with `batching` (see Optimizations below).

## Speed optimizations applied in this repo

Each is a single-line change in `DEFAULT_PYSR_KWARGS`
(`src/priya_forecast/refit_1d_pysr.py`).

| Optimization | Speedup observed | Side effect |
|---|---|---|
| `parallelism="multithreading"`, `procs=4` | 4–5× | non-reproducible across runs (results stable up to genetic-algorithm noise; `random_state` still controls seed) |
| `constraints={"^": (-1, 1)}` | ~1.2× (tames runaway `^`) | tames Ap HF max rel-err from 19% → 8% |
| `complexity_of_operators={"^": 3}` | none directly | biases Pareto toward smooth alternatives |
| `niterations=50` (vs 20) | 2.5× slower per fit | mean rel-err improves ~0.3% absolute |

For the multi-z run (>10k rows), we recommend additionally enabling
`batching=True, batch_size=1000` per the PySR docs — currently NOT on
in the repo; potential 2–3× speedup at the cost of slightly higher
training-loss variance (the genetic search sees a random batch each
generation).

## Speedups not yet applied (defer to a follow-up)

- **Reuse Julia process across fits**: call PySR once over a stacked
  dataset for all 11 parameters with a categorical "param ID" column,
  removing the per-param Julia cold-start. Estimated savings: 1–3 min /
  pipeline run. Cost: 2–3 days of wrapper code; the Pareto-by-param
  selection becomes more involved.

- **Pre-train on a polynomial surrogate**: fit a 6th-order polynomial
  to seed PySR's initial population. Not exposed in the PySR API.

- **Dedicated Julia HPC node**: at scale the genetic search benefits
  from `procs=os.cpu_count()` on a dedicated node. We didn't tune for
  this.

## Comparison to the baseline GP emulator training

For context: training the multi-fidelity GP emulator over the full 9D
parameter cube takes ~30 s once, then per-prediction is ~ms. PySR's
genetic search is the right tool for **interpretable** equation
discovery, but on per-prediction wall time the GP wins by 4–5 orders of
magnitude. The takeaway for the paper: PySR provides a
*compressed-form* surrogate of the GP, useful for scientific
interpretation and posterior visualization, not as a runtime
replacement.

## Recommendation for the methods section

> "We use PySR (Cranmer et al., XX) to fit each parameter's effect on
> the per-z-normalized flux power independently, with PySR genetic
> search at `niter=50, maxsize=20`. Each per-parameter fit uses 4
> input features (θ, k, fidelity, redshift) and a stacked LF+HF
> training set of ~14,400 rows over the kodiaq production k-grid (k ∈
> [0.005, 0.064] s/km) and z ∈ [2.6, 4.2]. Wall time is ~50 s per
> parameter fit on a multi-core node, dominated by the genetic
> search. The 11-parameter refit completes in ~10 minutes."

## Reproducibility footnote

For the *production-paper-final* fits, we re-run with
`deterministic=True, parallelism="serial"` to lock in
bit-reproducible equations, accepting the 4–5× wall-time penalty. The
science conclusions (σ ratios, Pareto choices) are stable across
reproducibility modes within genetic-algorithm noise.
