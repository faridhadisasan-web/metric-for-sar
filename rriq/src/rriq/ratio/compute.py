import numpy as np
from typing import Dict


def compute_ratio(
    original: np.ndarray, filtered: np.ndarray, eps: float = 1e-8
) -> np.ndarray:
    """Computes the ratio image: original / (filtered + eps)."""
    return original / (filtered + eps)


def compute_log_ratio(ratio: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """Computes the natural log of the ratio image."""
    return np.log(ratio + eps)


def get_ratio_stats(ratio: np.ndarray) -> Dict[str, float]:
    """Calculates summary statistics of the ratio image."""
    return {
        "mean": float(np.mean(ratio)),
        "std": float(np.std(ratio)),
        "min": float(np.min(ratio)),
        "max": float(np.max(ratio)),
        "median": float(np.median(ratio)),
    }
