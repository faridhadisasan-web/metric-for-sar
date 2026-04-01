import numpy as np


def apply_normalization(img: np.ndarray, method: str) -> np.ndarray:
    """Normalizes an image based on the chosen strategy ('none', 'min-max', 'z-score', 'log')."""
    if method == "none":
        return img
    elif method == "min-max":
        min_v, max_v = img.min(), img.max()
        if max_v > min_v:
            return (img - min_v) / (max_v - min_v)
        return np.zeros_like(img)
    elif method == "z-score":
        std = img.std()
        if std > 0:
            return (img - img.mean()) / std
        return np.zeros_like(img)
    elif method == "log":
        # Log-transform handling zeros/negatives safely
        shifted = img - img.min() + 1e-6 if img.min() <= 0 else img
        return np.log(shifted)
    else:
        raise ValueError(f"Unknown normalization method: {method}")


def clip_percentiles(
    img: np.ndarray, p_min: float = 1.0, p_max: float = 99.0
) -> np.ndarray:
    """Clips intensity values to the specified percentiles."""
    v_min, v_max = np.percentile(img, [p_min, p_max])
    return np.clip(img, v_min, v_max)


def ensure_positive(img: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Shifts image intensities to ensure all values are strictly positive for ratio computation."""
    min_val = img.min()
    if min_val <= 0:
        return img - min_val + eps
    return img
