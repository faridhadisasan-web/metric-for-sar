import numpy as np


def calculate_divergence(ratio: np.ndarray, theoretical_mean: float = 1.0) -> float:
    """Calculates a simple KL-like divergence or MSE from the theoretical mean (1.0)."""
    # Assuming fully developed speckle where the ratio image should have mean 1
    mse = np.mean((ratio - theoretical_mean) ** 2)
    return float(mse)
