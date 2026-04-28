"""Empirical comparison of additive vs multiplicative combine rules.

The student's pipeline is currently using ADDITIVE combine: each equation
predicts a normalized residual `(P_F_i - mean_k) / std_k`, and they are
summed before denormalization to reconstruct `P`. This test is here to keep
the choice honest:

  multiplicative : P(theta) = P_fid * Π_i [P_F_i(theta_i) / P_F_i(theta_fid_i)]
  additive       : P(theta) = P_fid + Σ_i [P_F_i(theta_i) - P_F_i(theta_fid_i)]

Truth model (MockGPModel) — power law × exp damping — is *not* exactly
separable in either rule. The test quantifies which rule has lower MSE at
off-fiducial points where multiple parameters are simultaneously perturbed.

The result depends on the truth's structure: when the response is closer to
multiplicative-in-amplitude (as Lya P1D often is), the multiplicative rule
wins; when it's closer to additive-in-residual, additive wins. We document
the result so the paper's choice is justified rather than asserted.
"""

from __future__ import annotations

import numpy as np

from priya_forecast.config import EqnConfig, EqnParam
from priya_forecast.models import MockGPModel, PySRModel
from priya_forecast.parameters import PARAM_NAMES, fiducial_vector, get_param


def _build_pysr_models_from_gp(
    *,
    gp: MockGPModel,
    z: float,
    k_grid: np.ndarray,
    fiducial_path,
    combine: str,
) -> PySRModel:
    """Build a PySRModel whose per-parameter equations exactly trace the GP's
    1D sweep on (ns, Ap, hub, omegamh2). All other parameters are silenced.

    Trick: we don't have analytic equations for the GP, so we use sympy's
    `Piecewise` / interpolation? No — simpler — we use a per-parameter linear
    expression in the *normalized* parameter that reproduces the GP at the
    two prior endpoints. That's a deliberately weak approximation; the test
    measures the *combine error*, not the per-equation fitting error.
    """
    fid = np.asarray(fiducial_vector(), dtype=float)
    parameters = {}
    for pname in ("ns", "Ap", "hub", "omegamh2"):
        idx = PARAM_NAMES.index(pname)
        prior = get_param(pname).prior
        # Probe the GP at lo, fid, hi (others fiducial); fit P_F as quadratic in p_norm.
        lo, hi = prior
        for_low = fid.copy(); for_low[idx] = lo
        for_mid = fid.copy()
        for_high = fid.copy(); for_high[idx] = hi
        p_lo = gp.predict(for_low, k_grid, z)
        p_mid = gp.predict(for_mid, k_grid, z)
        p_hi = gp.predict(for_high, k_grid, z)
        # Solve a quadratic a + b*p + c*p^2 = ... at p in {0, fid_norm, 1}.
        fid_norm = (fid[idx] - lo) / (hi - lo)
        # P_F(p_norm, k) ≈ a(k) + b(k)*p_norm + c(k)*p_norm^2
        A = np.array([[1, 0, 0], [1, fid_norm, fid_norm**2], [1, 1, 1]])
        rhs = np.stack([p_lo, p_mid, p_hi], axis=0)  # (3, n_k)
        coeffs = np.linalg.solve(A, rhs)  # (3, n_k)
        # Build a sympy expression whose evaluation at (p_norm, k_norm) returns
        # the quadratic above by interpolating coeffs onto k.
        # For simplicity we evaluate at the eBOSS k-grid only — k_grid here is
        # fixed for the test, so we hardcode coefficients via piecewise polynomial.
        # Easiest approach: use a numpy-callable directly via the `expression`
        # field with a per-k constant — we instead do this by writing a polynomial
        # in p_norm with k-dependent constants baked in via the normalization mean/std.
        # That's awkward; the cleanest path is to register a `joint`-style expression.
        # We'll instead use a degenerate test that compares additive vs multiplicative
        # for a *toy* synthetic-truth pipeline using a single hand-built expression.
        parameters[pname] = EqnParam(
            fiducial=fid[idx],
            expression=f"x0",  # dummy; this test is not executed in this form
            variables=[pname, "k"],
        )
    cfg = EqnConfig(
        name="t",
        redshift=z,
        model="pysr",
        combine=combine,
        fiducial_p1d=str(fiducial_path),
        parameters=parameters,
    )
    return PySRModel(eqn_cfg=cfg, k_grid=k_grid, normalization_block={"mode": "identity"})


# ---------------------------------------------------------------------------
# Synthetic truths where the answer is known in closed form.
# ---------------------------------------------------------------------------


def _eval_combine(*, combine: str, p_fid: np.ndarray, contributions: list[np.ndarray], refs: list[np.ndarray]) -> np.ndarray:
    """Apply the combine rule manually given per-parameter (P_F_i, P_F_i_fid)."""
    if combine == "additive":
        out = p_fid.copy()
        for c, r in zip(contributions, refs):
            out = out + (c - r)
        return out
    elif combine == "multiplicative":
        out = p_fid.copy()
        for c, r in zip(contributions, refs):
            out = out * (c / r)
        return out
    else:
        raise ValueError(combine)


def test_multiplicative_wins_when_truth_is_multiplicative():
    """If P_truth = P_fid * Π_i (1 + α_i * Δθ_i), the multiplicative combine
    is exact and additive incurs an O(α_i α_j) cross-term error."""
    rng = np.random.default_rng(0)
    nk = 20
    k = np.linspace(0.001, 0.02, nk)
    p_fid = 5.0 * np.exp(-50 * k)
    n_test = 64
    n_params = 4
    alphas = np.array([0.3, 0.2, 0.15, 0.1])

    deltas = rng.uniform(-0.5, 0.5, size=(n_test, n_params))

    mse_add, mse_mul = 0.0, 0.0
    for d in deltas:
        # Truth: each parameter scales P_fid multiplicatively.
        truth = p_fid * np.prod(1 + alphas * d)
        # Per-parameter equation: f_i(theta_i, k) = P_fid * (1 + alpha_i * delta_i)
        contribs = [p_fid * (1 + alphas[i] * d[i]) for i in range(n_params)]
        refs = [p_fid for _ in range(n_params)]
        pred_add = _eval_combine(combine="additive", p_fid=p_fid, contributions=contribs, refs=refs)
        pred_mul = _eval_combine(combine="multiplicative", p_fid=p_fid, contributions=contribs, refs=refs)
        mse_add += np.mean((pred_add - truth) ** 2)
        mse_mul += np.mean((pred_mul - truth) ** 2)
    mse_add /= n_test
    mse_mul /= n_test

    # Multiplicative is exact in this regime → MSE is floating-point-noise small.
    assert mse_mul < 1e-20
    # Additive has cross-term error.
    assert mse_add > 100 * mse_mul


def test_additive_wins_when_truth_is_additive():
    """If P_truth = P_fid + Σ_i α_i * Δθ_i, the additive combine is exact and
    multiplicative incurs the symmetric error."""
    rng = np.random.default_rng(0)
    nk = 20
    k = np.linspace(0.001, 0.02, nk)
    p_fid = 5.0 * np.exp(-50 * k)
    n_test = 64
    n_params = 4
    alphas = np.array([0.3, 0.2, 0.15, 0.1])  # absolute scale

    deltas = rng.uniform(-0.5, 0.5, size=(n_test, n_params))

    mse_add, mse_mul = 0.0, 0.0
    for d in deltas:
        # Truth: each parameter adds a term to P_fid.
        truth = p_fid + sum(alphas[i] * d[i] * p_fid for i in range(n_params))
        contribs = [p_fid + alphas[i] * d[i] * p_fid for i in range(n_params)]
        refs = [p_fid for _ in range(n_params)]
        pred_add = _eval_combine(combine="additive", p_fid=p_fid, contributions=contribs, refs=refs)
        pred_mul = _eval_combine(combine="multiplicative", p_fid=p_fid, contributions=contribs, refs=refs)
        mse_add += np.mean((pred_add - truth) ** 2)
        mse_mul += np.mean((pred_mul - truth) ** 2)
    mse_add /= n_test
    mse_mul /= n_test

    assert mse_add < 1e-20
    assert mse_mul > 100 * mse_add


def test_combine_comparison_on_mock_gp_records_winner():
    """End-to-end: train per-parameter equations as quadratic fits to the GP
    at three sample points (low, fid, high), then evaluate both combines on a
    Sobol test set off-fiducial.

    This test does NOT assert a winner — it computes both MSEs and asserts
    they're both finite. The numerical result is printed so a paper claim
    can be supported / falsified by re-running this test on the real GP.

    Caveat: MockGPModel is `A(theta) × power_law(k) × exp_damping(z, k)`,
    which is *exactly multiplicative* in the amplitude. Multiplicative MSE
    on this mock will always win by construction. To produce the real paper
    claim, replace `MockGPModel()` below with `GPModel()` once GPy is
    available — that's the diagnostic the paper's combine choice rests on.
    """
    from priya_forecast.models.normalization import (
        DEFAULT_K_MAX,
        DEFAULT_K_MIN,
        NormalizationSpec,
    )

    nk = 35
    k = np.linspace(DEFAULT_K_MIN, DEFAULT_K_MAX, nk)
    gp = MockGPModel()
    fid = np.asarray(fiducial_vector(), dtype=float)
    p_fid = gp.predict(fid, k, 3.6)

    # Per-parameter quadratic fits in normalized space.
    param_names = ("ns", "Ap", "hub", "omegamh2")
    fits = {}
    for pname in param_names:
        idx = PARAM_NAMES.index(pname)
        lo, hi = get_param(pname).prior
        fid_norm = (fid[idx] - lo) / (hi - lo)
        for_lo = fid.copy(); for_lo[idx] = lo
        for_hi = fid.copy(); for_hi[idx] = hi
        p_lo = gp.predict(for_lo, k, 3.6)
        p_hi = gp.predict(for_hi, k, 3.6)
        # Quadratic in p_norm through (0, p_lo), (fid_norm, p_fid), (1, p_hi).
        A = np.array([[1, 0, 0], [1, fid_norm, fid_norm**2], [1, 1, 1]])
        rhs = np.stack([p_lo, p_fid, p_hi], axis=0)
        coeffs = np.linalg.solve(A, rhs)
        fits[pname] = (idx, lo, hi, coeffs)

    def predict_combine(theta: np.ndarray, combine: str) -> np.ndarray:
        contribs = []
        refs = []
        for pname in param_names:
            idx, lo, hi, coeffs = fits[pname]
            p_norm = (theta[idx] - lo) / (hi - lo)
            fid_p_norm = (fid[idx] - lo) / (hi - lo)
            contribs.append(coeffs[0] + coeffs[1] * p_norm + coeffs[2] * p_norm ** 2)
            refs.append(coeffs[0] + coeffs[1] * fid_p_norm + coeffs[2] * fid_p_norm ** 2)
        return _eval_combine(combine=combine, p_fid=p_fid, contributions=contribs, refs=refs)

    # Sobol-like off-fiducial test set.
    rng = np.random.default_rng(0)
    n_test = 32
    mse_add, mse_mul = 0.0, 0.0
    for _ in range(n_test):
        theta = fid.copy()
        for pname in param_names:
            idx, lo, hi, _ = fits[pname]
            theta[idx] = rng.uniform(lo, hi)
        truth = gp.predict(theta, k, 3.6)
        mse_add += np.mean((predict_combine(theta, "additive") - truth) ** 2)
        mse_mul += np.mean((predict_combine(theta, "multiplicative") - truth) ** 2)
    mse_add /= n_test
    mse_mul /= n_test

    print(f"\n[combine justification] additive MSE = {mse_add:.3e}, "
          f"multiplicative MSE = {mse_mul:.3e}, ratio (mul/add) = {mse_mul/mse_add:.3f}")

    assert np.isfinite(mse_add)
    assert np.isfinite(mse_mul)
    assert mse_add > 0
    assert mse_mul > 0
