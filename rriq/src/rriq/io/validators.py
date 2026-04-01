import numpy as np
from rriq.utils.logging_utils import logger


def validate_sar_image(
    image: np.ndarray, coerce_complex: bool = False, reduce_multichannel: str = "error"
) -> np.ndarray:
    """
    Validates that the input image is a 2D single-channel real-valued array.

    Args:
        image (np.ndarray): The input image array.
        coerce_complex (bool): If true, take the absolute value of complex arrays.
        reduce_multichannel (str): 'error', 'mean', or 'first' to handle multi-channel images.

    Returns:
        np.ndarray: A 2D single-channel float32 array.

    Raises:
        ValueError: If constraints are violated and coercion is disallowed.
    """
    img = np.array(image)

    # 1. Complex check
    if np.iscomplexobj(img):
        if not coerce_complex:
            raise ValueError(
                "Input image is complex-valued. Expected 2D real-valued intensity/amplitude. "
                "Set coerce_complex_to_magnitude=True in config to force magnitude conversion."
            )
        logger.warning("Coercing complex input to absolute magnitude.")
        img = np.abs(img)

    # 2. Dimensions check
    if img.ndim > 2:
        if reduce_multichannel == "error":
            raise ValueError(
                f"Input image has {img.ndim} dimensions {img.shape}. Expected 2D. "
                "Set reduce_multichannel='mean' or 'first' to force reduction."
            )
        elif reduce_multichannel == "first":
            logger.warning(f"Taking first channel of {img.ndim}D input.")
            # Assume (H, W, C) or similar, take the first slice along the last axis
            while img.ndim > 2:
                img = img[..., 0]
        elif reduce_multichannel == "mean":
            logger.warning(f"Averaging across channels of {img.ndim}D input.")
            while img.ndim > 2:
                img = img.mean(axis=-1)
        else:
            raise ValueError(f"Unknown reduce_multichannel mode: {reduce_multichannel}")

    if img.ndim < 2:
        raise ValueError(f"Input image must be 2D, but got shape {img.shape}.")

    return img.astype(np.float32)
