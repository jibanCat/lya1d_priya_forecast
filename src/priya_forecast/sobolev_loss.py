"""Sobolev derivative-matching loss for PySR refits.

Adds  λ·MSE( ∂eq/∂θ_norm , target_grad )  to the value MSE, where ∂eq/∂θ_norm
is finite-differenced INSIDE the loss (eval the tree at X and at X shifted by
+h in the θ-feature row) and `target_grad` is the GP's gradient delivered via
PySR's per-point `weights` channel. Spike-confirmed to run in PySR 1.5.10.
"""
from __future__ import annotations


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
