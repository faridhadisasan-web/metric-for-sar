import numpy as np
from skimage.filters import sobel
from skimage.feature import canny
from skimage.morphology import dilation, disk


def build_edge_mask(
    image: np.ndarray, method: str = "sobel", threshold: float = 0.05
) -> np.ndarray:
    """
    Creates an edge mask using Sobel magnitude or Canny edge detection.
    Edges represent boundaries and structure in the image.
    """
    if method == "sobel":
        edges = sobel(image)
        mask = (edges > threshold).astype(np.uint8)
    elif method == "canny":
        # Normalize for Canny
        norm = (image - np.min(image)) / (np.max(image) - np.min(image) + 1e-8)
        mask = canny(norm, sigma=1.0, low_threshold=0.1, high_threshold=0.3).astype(
            np.uint8
        )
    else:
        raise ValueError(f"Unknown edge detection method: {method}")

    # Dilate edges slightly to capture the transition zone better
    return dilation(mask, disk(1))
