import numpy as np


def compute_epd_roa(original: np.ndarray, filtered: np.ndarray) -> float:
    """
    Computes Edge Preservation Degree based on Ratio of Average (EPD-ROA).
    Based on the standard horizontal and vertical ROA formulation.
    Values closer to 1 indicate better edge preservation.
    """
    # Simplified vertical/horizontal ROA
    # Add a small epsilon to avoid division by zero
    eps = 1e-8

    # Original ROA
    v_roa_o = np.abs(original[1:, :] / (original[:-1, :] + eps))
    h_roa_o = np.abs(original[:, 1:] / (original[:, :-1] + eps))

    # Filtered ROA
    v_roa_f = np.abs(filtered[1:, :] / (filtered[:-1, :] + eps))
    h_roa_f = np.abs(filtered[:, 1:] / (filtered[:, :-1] + eps))

    # Mean values
    epd_v = np.mean(v_roa_f) / (np.mean(v_roa_o) + eps)
    epd_h = np.mean(h_roa_f) / (np.mean(h_roa_o) + eps)

    # Combined average
    return float(0.5 * (epd_v + epd_h))
