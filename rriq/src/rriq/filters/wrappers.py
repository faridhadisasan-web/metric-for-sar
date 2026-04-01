from rriq.filters.registry import register_filter
import numpy as np


@register_filter("merlin")
def filter_merlin(image: np.ndarray) -> np.ndarray:
    """Optional wrapper for MERLIN deep filter."""
    try:
        import torch  # noqa: F401

        # Placeholder wrapper for actual MERLIN implementation
        raise NotImplementedError(
            "MERLIN wrapper requires the deepdespeckling package and model weights."
        )
    except ImportError:
        raise ImportError(
            "MERLIN requires 'torch' and 'deepdespeckling'. Install with pip install rriq[deep]"
        )


@register_filter("sar2sar")
def filter_sar2sar(image: np.ndarray) -> np.ndarray:
    """Optional wrapper for SAR2SAR deep filter."""
    raise NotImplementedError(
        "SAR2SAR wrapper is not fully implemented in V1. Requires deep dependencies."
    )
