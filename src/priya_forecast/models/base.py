"""Abstract base for P1D models. Both PySR and GP models conform to this."""

# TODO(phase3): P1DModel ABC with predict(theta_11d, k, z) -> p1d


class P1DModel:
    """Placeholder until phase 3 lands."""

    def predict(self, theta, k, z):  # pragma: no cover - placeholder
        raise NotImplementedError
