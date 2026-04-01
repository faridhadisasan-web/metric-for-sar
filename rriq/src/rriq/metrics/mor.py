import numpy as np


def compute_mor(ratio: np.ndarray) -> float:
    """
    Computes Measure of Ratio (MOR).
    MOR = mean(ratio) / (1 + std(ratio)).
    Values closer to 1 indicate perfect speckle preservation (assuming fully developed speckle).
    Sometimes reported as abs(1 - MOR) or similar.
    We report the raw ratio mean/std combination.
    """
    mean_r = np.mean(ratio)
    std_r = np.std(ratio)

    return float(mean_r / (1.0 + std_r))
