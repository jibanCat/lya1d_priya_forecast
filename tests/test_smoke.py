"""Phase 0 smoke test: package imports cleanly."""

import priya_forecast


def test_package_imports():
    assert priya_forecast.__version__ == "0.1.0"
