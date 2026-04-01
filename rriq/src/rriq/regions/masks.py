import numpy as np
from typing import Dict
from rriq.regions.homogeneous import build_homogeneous_mask
from rriq.regions.edges import build_edge_mask
from rriq.regions.scatterers import build_scatterer_mask


def generate_all_masks(image: np.ndarray, config: dict) -> Dict[str, np.ndarray]:
    """Generates homogeneous, edge, and scatterer masks."""
    masks = {}

    masks["homogeneous"] = build_homogeneous_mask(
        image, threshold=config.get("homogeneous", {}).get("threshold", 0.15)
    )

    masks["edges"] = build_edge_mask(
        image,
        method=config.get("edges", {}).get("method", "sobel"),
        threshold=config.get("edges", {}).get("threshold", 0.05),
    )

    masks["scatterers"] = build_scatterer_mask(
        image, percentile=config.get("scatterers", {}).get("percentile", 99.5)
    )

    # Optional: ensure masks are mutually exclusive or prioritize them
    # For independent ROI extraction, overlapping is fine.
    return masks
