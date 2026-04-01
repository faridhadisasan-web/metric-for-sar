import numpy as np


def generate_synthetic_targets(num_samples: int, seed: int = 42) -> np.ndarray:
    """Generates synthetic severity targets for testing monotonicity."""
    np.random.seed(seed)
    return np.linspace(0, 1, num_samples)
