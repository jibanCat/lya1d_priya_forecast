# tests/test_stage9_loss_string.py
from priya_forecast.sobolev_loss import make_sobolev_loss


def test_loss_string_has_terms_and_constants():
    s = make_sobolev_loss(lam=2.5, h=1e-4)
    assert "eval_tree_array(tree, dataset.X, options)" in s
    assert "dataset.y" in s and "dataset.weights" in s
    assert "X2[1, :]" in s
    assert "2.5" in s
    assert "0.0001" in s or "1.0e-4" in s or "1e-4" in s
    assert s.strip().startswith("function loss_function")


def test_lambda_changes_string():
    assert make_sobolev_loss(lam=1.0, h=1e-4) != make_sobolev_loss(lam=9.0, h=1e-4)
