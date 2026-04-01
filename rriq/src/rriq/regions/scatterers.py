import numpy as np
import warnings
from skimage.morphology import remove_small_objects


def build_scatterer_mask(
    image: np.ndarray, percentile: float = 99.5, min_size: int = 3
) -> np.ndarray:
    """
    Creates a mask for bright scatterers using high-intensity percentile thresholding.
    Scatterers correspond to strong targets like buildings or metallic structures.
    """
    threshold = np.percentile(image, percentile)
    mask = image > threshold

    # Remove small noise artifacts
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return remove_small_objects(mask, min_size=min_size).astype(np.uint8)
