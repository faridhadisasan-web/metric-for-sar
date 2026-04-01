import numpy as np


def compute_tcr(
    image: np.ndarray, scatterer_mask: np.ndarray, background_mask: np.ndarray
) -> float:
    """
    Computes Target Contrast Ratio (TCR).
    TCR = |mean(target) - mean(background)| / sqrt(var(target) + var(background)).
    Higher TCR indicates better contrast preservation of strong scatterers.
    """
    target_pixels = image[scatterer_mask > 0]
    bg_pixels = image[background_mask > 0]

    if len(target_pixels) == 0 or len(bg_pixels) == 0:
        return np.nan

    m_t = np.mean(target_pixels)
    v_t = np.var(target_pixels)

    m_b = np.mean(bg_pixels)
    v_b = np.var(bg_pixels)

    denom = np.sqrt(v_t + v_b)
    if denom == 0:
        return np.nan

    return float(np.abs(m_t - m_b) / denom)
