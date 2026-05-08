# Smoke: log-target vs linear-target for Ap
emulator: kodiaq_2_2_4_6-48-48; Ap payload: results/refit_phase2_production/payloads/Ap.pkl
fit time: 365.6s with niter=50, smart kwargs (option B).

## log-target eq
```
square((square(square(x3) + square(x2 - x0)) + 2.0314214) - square(x1)) * ((0.76134586 - x2) - x1)
```
complexity=20, normalized loss=4.272

## rel-err (linear P_F space, after exp())
- LF: mean=1.87%, max=19.42%
- HF: mean=2.45%, max=12.64%

## slope at fid (θ_Ap_norm direction)
- log-eq:    3.638
- linear-eq: 0.2756

## reference linear-target eq (from results/refit_phase2_production/refits/Ap.pkl)
```
exp(square(x3 - -0.24380434)) * ((exp((x0 / 1.1384475) + (x1 * -3.7485268)) + (x2 / -0.44109055)) - -0.44676068)
```
- LF rel-err mean/max: 1.87% / 23.83%
- HF rel-err mean/max: 1.93% / 10.59%

## interpretation
- If log-eq has lower **max** rel-err → log-target is more Lipschitz off-fid.
- If |log-eq slope| ≈ |linear-eq slope| × |∂P_F/∂log P_F| → log-target gradient
  matches linear's at-fid Fisher contribution (to first order).
- Gradient mismatch in the *current* linear-eq drives the σ_PySR/σ_GP=2.62×
  Ap regression at fid; if log-eq has comparable shape but better Lipschitz
  off-fid, switching to log-target is the cleaner production target.
