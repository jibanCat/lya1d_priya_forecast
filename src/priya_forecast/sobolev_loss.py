"""Sobolev derivative-matching loss for PySR refits.

Adds  λ·MSE( ∂eq/∂θ_norm , target_grad )  to the value MSE, where ∂eq/∂θ_norm
is finite-differenced INSIDE the loss (eval the tree at X and at X shifted by
+h in the θ-feature row) and `target_grad` is the GP's gradient delivered via
PySR's per-point `weights` channel. Spike-confirmed to run in PySR 1.5.10.
"""
from __future__ import annotations

import numpy as np


def make_sobolev_loss(lam: float, h: float = 1e-4) -> str:
    """Return a Julia `loss_function` string with λ and h injected as literals."""
    return (
        "function loss_function(tree, dataset::Dataset{T,L}, options) where {T,L}\n"
        "    prediction, complete = eval_tree_array(tree, dataset.X, options)\n"
        "    if !complete || any(isnan, prediction) || any(isinf, prediction)\n"
        "        return L(Inf)\n"
        "    end\n"
        "    n = length(prediction)\n"
        "    residual = prediction .- dataset.y\n"
        "    mse = sum(residual .^ 2) / n\n"
        f"    h = T({h!r})\n"
        "    X2 = copy(dataset.X)\n"
        "    @inbounds X2[1, :] .+= h\n"
        "    pred2, complete2 = eval_tree_array(tree, X2, options)\n"
        "    if !complete2 || any(isnan, pred2) || any(isinf, pred2)\n"
        "        return L(Inf)\n"
        "    end\n"
        "    grad = (pred2 .- prediction) ./ h\n"
        "    gdiff = grad .- dataset.weights\n"
        "    gmse = sum(gdiff .^ 2) / n\n"
        f"    return mse + L({float(lam)!r}) * gmse\n"
        "end\n"
    )


def _fidelity_grad_weights(*, params, kfkms, gp, param_idx, z,
                           x_param_min, x_param_max, std_on_k, norm_k_grid, h):
    """Per-row normalized target gradient for one fidelity, point-major/k-minor.

    weight = (∂logP/∂θ_phys) · width / std_k   (width = x_param_max − x_param_min)
    Rows ordered point-major (k varies fastest), matching _build_training_matrix.

    `gp.predict(theta, k, z)` must return linear P_F (not log); this routine
    takes the log internally.

    Perturbations are clamped to [x_param_min, x_param_max] so that sweep
    boundary points never push theta outside the emulator's valid range
    (which would trigger an AssertionError in map_to_unit_cube).  The actual
    (possibly asymmetric) span is used as the finite-difference denominator.
    """
    x_min = float(x_param_min)
    x_max = float(x_param_max)
    width = x_max - x_min
    n_points = params.shape[0]
    rows = []
    for j in range(n_points):
        k_j = np.asarray(kfkms[j], dtype=float)
        theta = np.asarray(params[j], dtype=float)
        ti = float(theta[param_idx])
        step = h * max(abs(ti), 1.0)
        tp_val = min(ti + step, x_max)
        tm_val = max(ti - step, x_min)
        denom = tp_val - tm_val
        if denom <= 0.0:        # degenerate (zero-width sweep) -> zero gradient
            rows.append(np.zeros_like(k_j))
            continue
        tp = theta.copy(); tp[param_idx] = tp_val
        tm = theta.copy(); tm[param_idx] = tm_val
        lp_p = np.log(np.asarray(gp.predict(tp, k_j, z), dtype=float))
        lp_m = np.log(np.asarray(gp.predict(tm, k_j, z), dtype=float))
        grad_phys = (lp_p - lp_m) / denom                    # ∂logP/∂θ_phys, per k (clamped step)
        std_k = np.interp(k_j, np.asarray(norm_k_grid, float), np.asarray(std_on_k, float))
        rows.append(grad_phys * width / std_k)                # normalized to (x0, std)
    return np.concatenate(rows)


def sobolev_target_weights(*, payload, param_idx, gp_lf, gp_hf, z,
                           x_param_min, x_param_max, std_flux, norm_k_grid, h=1e-3):
    """Per-row Sobolev target gradient matching X_act row order (LF rows then HF).

    `std_flux` is the SINGLE global per-k std from the refit's NormalizationSpec
    (`norm.std_flux` on `norm.k_grid`) — the SAME one `_build_training_matrix`
    interpolates onto BOTH the LF and HF k-grids. Do NOT pass separate LF/HF
    stds; that would diverge from the training-matrix normalization.
    """
    w_lf = _fidelity_grad_weights(
        params=np.asarray(payload["params_lf"], float), kfkms=payload["kfkms_lf_z"],
        gp=gp_lf, param_idx=param_idx, z=z, x_param_min=x_param_min, x_param_max=x_param_max,
        std_on_k=std_flux, norm_k_grid=norm_k_grid, h=h)
    w_hf = _fidelity_grad_weights(
        params=np.asarray(payload["params_hf"], float), kfkms=payload["kfkms_hf_z"],
        gp=gp_hf, param_idx=param_idx, z=z, x_param_min=x_param_min, x_param_max=x_param_max,
        std_on_k=std_flux, norm_k_grid=norm_k_grid, h=h)
    return np.concatenate([w_lf, w_hf])
