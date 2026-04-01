import numpy as np
from scipy.ndimage import uniform_filter


def get_local_cv(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Calculates the local Coefficient of Variation."""
    mean_val = uniform_filter(image, (size, size))
    mean_sq = uniform_filter(image**2, (size, size))
    variance = mean_sq - mean_val**2
    variance[variance < 0] = 0

    cv = np.sqrt(variance) / (mean_val + 1e-8)
    return cv


def build_homogeneous_mask(
    image: np.ndarray, threshold: float = 0.15, size: int = 5
) -> np.ndarray:
    """
    Creates a mask for homogeneous regions based on low local coefficient of variation.
    Homogeneous regions are typical of flat surfaces like water or calm ground.
    """
    cv = get_local_cv(image, size)
    return (cv < threshold).astype(np.uint8)
