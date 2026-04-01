import numpy as np
from skimage.filters import sobel


def compute_rgpi(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Computes Radiometric Gradient Preservation Index (RGPI).
    RGPI = sum(|gradient(filtered)|) / sum(|gradient(original)|).
    Values closer to 1 indicate perfect gradient/edge preservation.
    """
    grad_o = sobel(original)
    grad_f = sobel(filtered)

    sum_o = np.sum(np.abs(grad_o))
    sum_f = np.sum(np.abs(grad_f))

    if sum_o == 0:
        return np.nan

    return float(sum_f / sum_o)
