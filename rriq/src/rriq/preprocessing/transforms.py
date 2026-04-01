import numpy as np


def apply_transforms(img: np.ndarray, config: dict) -> np.ndarray:
    """Applies a sequence of preprocessing transformations from config."""
    from rriq.preprocessing.normalize import (
        apply_normalization,
        clip_percentiles,
        ensure_positive,
    )
    from rriq.preprocessing.scaling import scale_to_255

    # Ensure all values are strictly positive first for safe ratios
    processed = ensure_positive(img)

    method = config.get("normalize", "none")
    processed = apply_normalization(processed, method)

    p_min, p_max = config.get("clip_percentiles", [1.0, 99.0])

    if config.get("scale_to_255", True):
        processed = scale_to_255(processed, p_min, p_max)
    else:
        processed = clip_percentiles(processed, p_min, p_max)

    return processed.astype(np.float32)
