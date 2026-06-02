# tests/test_multi_z_shared_grid.py
import numpy as np
import pytest
from priya_forecast.multi_z.forecast import shared_k_and_z_grid


class _StubLike:
    def __init__(self, kept_k, z_blocks):
        self.kept_k = np.asarray(kept_k, float)
        self.z_blocks = z_blocks


def test_uniform_k_per_z_ok():
    k = np.array([0.01, 0.02, 0.03, 0.01, 0.02, 0.03])
    like = _StubLike(k, [(3.4, slice(0, 3)), (3.6, slice(3, 6))])
    k0, zgrid = shared_k_and_z_grid(like)
    np.testing.assert_allclose(k0, [0.01, 0.02, 0.03])
    np.testing.assert_allclose(zgrid, [3.4, 3.6])


def test_nonuniform_k_per_z_raises():
    k = np.array([0.01, 0.02, 0.03, 0.01, 0.02])   # second block shorter
    like = _StubLike(k, [(3.4, slice(0, 3)), (3.6, slice(3, 5))])
    with pytest.raises(ValueError, match="differs across z-blocks"):
        shared_k_and_z_grid(like)
