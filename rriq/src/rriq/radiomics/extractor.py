import logging
import numpy as np
import SimpleITK as sitk
from typing import Dict, Any
from pathlib import Path
from rriq.utils.logging_utils import logger

# Prevent pyradiomics from spamming the console
import radiomics

radiomics.logger.setLevel(logging.ERROR)
from radiomics.featureextractor import RadiomicsFeatureExtractor


def array_to_sitk(img: np.ndarray) -> sitk.Image:
    """Converts a 2D numpy array to a SimpleITK image."""
    sitk_img = sitk.GetImageFromArray(img)
    # Give dummy spacing so radiomics won't complain
    sitk_img.SetSpacing([1.0, 1.0])
    return sitk_img


def extract_features(
    image: np.ndarray, mask: np.ndarray, config_path: str | Path, label: int = 1
) -> Dict[str, Any]:
    """
    Extracts PyRadiomics features for a given image and mask.
    Image must be pre-normalized to [0, 255].
    """
    sitk_img = array_to_sitk(image)
    sitk_mask = array_to_sitk(mask.astype(np.uint8))

    extractor = RadiomicsFeatureExtractor(str(config_path))
    # We enforce 2D feature extraction
    extractor.settings["force2D"] = True

    # Run extractor
    try:
        features = extractor.execute(sitk_img, sitk_mask, label=label)

        # Filter only diagnostic/feature keys (ignore general config metadata)
        clean_features = {}
        for k, v in features.items():
            if k.startswith("original_"):
                clean_features[k] = float(v)

        return clean_features
    except Exception as e:
        logger.error(f"Failed to extract features: {e}")
        return {}
