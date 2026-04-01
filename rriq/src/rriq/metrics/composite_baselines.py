from typing import Dict
import numpy as np


def compute_all_baselines(
    original: np.ndarray,
    filtered: np.ndarray,
    ratio: np.ndarray,
    masks: Dict[str, np.ndarray],
) -> Dict[str, float]:
    """Computes all canonical baseline metrics."""
    from rriq.metrics.enl import compute_enl
    from rriq.metrics.epd_roa import compute_epd_roa
    from rriq.metrics.mor import compute_mor
    from rriq.metrics.tcr import compute_tcr
    from rriq.metrics.rgpi import compute_rgpi

    metrics = {}

    if "homogeneous" in masks:
        metrics["enl"] = compute_enl(filtered, masks["homogeneous"])
        metrics["enl_ratio"] = compute_enl(ratio, masks["homogeneous"])

    metrics["epd_roa"] = compute_epd_roa(original, filtered)
    metrics["mor"] = compute_mor(ratio)
    metrics["rgpi"] = compute_rgpi(original, filtered)

    if "scatterers" in masks and "homogeneous" in masks:
        metrics["tcr"] = compute_tcr(
            filtered, masks["scatterers"], masks["homogeneous"]
        )

    return metrics
