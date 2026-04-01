import numpy as np
from rriq.preprocessing.normalize import clip_percentiles


def scale_to_255(
    img: np.ndarray, p_min: float = 1.0, p_max: float = 99.0
) -> np.ndarray:
    """Clips an image to percentiles and scales it to [0, 255] float32."""
    clipped = clip_percentiles(img, p_min, p_max)
    min_v, max_v = clipped.min(), clipped.max()
    if max_v > min_v:
        scaled = (clipped - min_v) / (max_v - min_v) * 255.0
    else:
        scaled = np.zeros_like(clipped)
    return scaled.astype(np.float32)
