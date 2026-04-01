import numpy as np
from typing import Dict, Any, List

def add_speckle(image: np.ndarray, looks: int) -> np.ndarray:
    """Adds multiplicative Gamma-distributed speckle noise to a clean image."""
    shape = image.shape
    # Gamma distribution with mean=1 and var=1/looks
    noise = np.random.gamma(shape=looks, scale=1.0/looks, size=shape)
    return image * noise

def generate_synthetic_degradations(
    clean_image: np.ndarray,
    look_ladder: List[int],
    seed: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Generates a ladder of degraded images from a clean synthetic reference.
    Lower looks = higher severity (more noise).
    Severity score is normalized to [0, 1] where 1 is the cleanest (highest looks).
    """
    np.random.seed(seed)

    # Sort looks to compute monotonic severity
    sorted_looks = sorted(look_ladder)
    max_looks = max(look_ladder)
    min_looks = min(look_ladder)

    results = {}

    for l in look_ladder:
        degraded = add_speckle(clean_image, l)

        # severity score: 1.0 is best (cleanest), 0.0 is worst (noisiest)
        if max_looks > min_looks:
            severity = (l - min_looks) / (max_looks - min_looks)
        else:
            severity = 1.0

        results[f"looks_{l}"] = {
            "image": degraded,
            "looks": l,
            "severity_target": float(severity)
        }

    return results
