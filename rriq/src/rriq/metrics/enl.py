import numpy as np


def compute_enl(image: np.ndarray, homogeneous_mask: np.ndarray) -> float:
    """
    Computes Equivalent Number of Looks (ENL) over homogeneous regions.
    ENL = (mean^2) / variance.
    Higher ENL indicates better speckle suppression.
    """
    valid_pixels = image[homogeneous_mask > 0]
    if len(valid_pixels) == 0:
        return np.nan

    mean_val = np.mean(valid_pixels)
    var_val = np.var(valid_pixels)

    if var_val == 0:
        return np.inf

    return float((mean_val**2) / var_val)
