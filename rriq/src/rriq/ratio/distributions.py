import numpy as np
import scipy.stats


def fit_gamma(ratio: np.ndarray) -> tuple:
    """Fits a Gamma distribution to the ratio image data."""
    # This is a common theoretical distribution for SAR ratio images under speckle
    shape, loc, scale = scipy.stats.gamma.fit(ratio.flatten(), floc=0)
    return shape, loc, scale
