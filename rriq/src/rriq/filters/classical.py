import numpy as np
from scipy.ndimage import uniform_filter
from skimage.filters import median, gaussian
from skimage.morphology import disk
from skimage.restoration import denoise_bilateral, denoise_nl_means, estimate_sigma
from rriq.filters.registry import register_filter


@register_filter("median")
def filter_median(image: np.ndarray, size: int = 3) -> np.ndarray:
    """Standard Median filter using scikit-image."""
    return median(image, disk(size))


@register_filter("gaussian")
def filter_gaussian(image: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Standard Gaussian filter using scikit-image."""
    return gaussian(image, sigma=sigma)


@register_filter("lee")
def filter_lee(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Custom CPU-based Lee filter for SAR speckle reduction."""
    mean_val = uniform_filter(image, (size, size))
    mean_sq = uniform_filter(image**2, (size, size))
    variance = mean_sq - mean_val**2

    # Avoid division by zero
    variance[variance < 0] = 0
    img_variance = np.var(image)

    if img_variance == 0:
        return image.copy()

    weights = variance / (variance + img_variance + 1e-8)
    filtered = mean_val + weights * (image - mean_val)
    return filtered


@register_filter("kuan")
def filter_kuan(image: np.ndarray, size: int = 5) -> np.ndarray:
    """Custom CPU-based Kuan filter for SAR speckle reduction."""
    mean_val = uniform_filter(image, (size, size))
    mean_sq = uniform_filter(image**2, (size, size))
    variance = mean_sq - mean_val**2

    # Avoid div zero
    variance[variance < 0] = 0

    img_mean = np.mean(image)
    if img_mean == 0:
        return image.copy()

    cu = np.sqrt(np.var(image)) / img_mean
    ci = np.sqrt(variance) / (mean_val + 1e-8)

    w = (1 - (cu**2 / (ci**2 + 1e-8))) / (1 + cu**2)
    w = np.clip(w, 0, 1)

    filtered = mean_val + w * (image - mean_val)
    return filtered


@register_filter("frost")
def filter_frost(image: np.ndarray, size: int = 5, damp: float = 1.0) -> np.ndarray:
    """Custom CPU-based Frost filter for SAR speckle reduction."""
    # This is a simplified efficient spatial frost filter approximation
    # Calculate local mean and variance
    mean_val = uniform_filter(image, (size, size))
    mean_sq = uniform_filter(image**2, (size, size))
    variance = mean_sq - mean_val**2
    variance[variance < 0] = 0

    # Calculate local coefficient of variation (ci)
    ci = np.sqrt(variance) / (mean_val + 1e-8)

    # Calculate distances from center
    center = size // 2
    y, x = np.ogrid[-center : center + 1, -center : center + 1]
    distance = np.sqrt(x**2 + y**2)

    h, w = image.shape
    filtered = np.zeros_like(image)

    # Pad image to handle borders
    padded = np.pad(image, center, mode="reflect")
    ci_padded = np.pad(ci, center, mode="reflect")

    # Note: A full sliding window with spatial weighting is slow in pure Python.
    # For Version 1, we use an approximation or a simpler loop. We'll use a loop.
    for i in range(h):
        for j in range(w):
            patch = padded[i : i + size, j : j + size]
            k = damp * ci_padded[i + center, j + center]
            weights = np.exp(-k * distance)
            weights /= np.sum(weights)
            filtered[i, j] = np.sum(patch * weights)

    return filtered


@register_filter("bilateral")
def filter_bilateral(
    image: np.ndarray, sigma_color: float = 0.05, sigma_spatial: float = 15.0
) -> np.ndarray:
    """Standard Bilateral filter using scikit-image."""
    return denoise_bilateral(
        image, sigma_color=sigma_color, sigma_spatial=sigma_spatial, channel_axis=None
    )


@register_filter("nlm")
def filter_nlm(
    image: np.ndarray, patch_size: int = 5, patch_distance: int = 6
) -> np.ndarray:
    """Non-local means filter using scikit-image."""
    sigma_est = np.mean(estimate_sigma(image, channel_axis=None))
    return denoise_nl_means(
        image,
        h=1.15 * sigma_est,
        fast_mode=True,
        patch_size=patch_size,
        patch_distance=patch_distance,
        channel_axis=None,
    )


@register_filter("srad")
def filter_srad(image: np.ndarray) -> np.ndarray:
    """Speckle Reducing Anisotropic Diffusion (SRAD) - Optional/Experimental."""
    raise NotImplementedError("SRAD is experimental and not yet implemented in V1.")
