# tests/test_stage8_gate.py
import numpy as np
from priya_forecast.derivative_gate import derivative_faithful


def test_faithful_passes():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    cand = target * 1.05               # 5% off -> median rel err 0.05 < 0.25
    assert derivative_faithful(cand_grad=cand, target_grad=target, tol=0.25) is True


def test_unfaithful_rejected():
    target = np.array([1.0, 2.0, 3.0, 4.0])
    cand = target * -0.5               # wrong sign + magnitude -> rel err ~1.5
    assert derivative_faithful(cand_grad=cand, target_grad=target, tol=0.25) is False


def test_near_zero_target_bins_masked():
    # bins where the GP gradient is ~0 must not blow up the ratio
    target = np.array([1.0, 1e-12, 1e-12, 1.0])
    cand = np.array([1.05, 5.0, -3.0, 1.05])   # huge rel err only in masked bins
    assert derivative_faithful(cand_grad=cand, target_grad=target, tol=0.25) is True
