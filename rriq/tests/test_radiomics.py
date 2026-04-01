import numpy as np
from rriq.radiomics.extractor import array_to_sitk, extract_features


def test_array_to_sitk(tiny_synthetic_image):
    sitk_img = array_to_sitk(tiny_synthetic_image)
    assert sitk_img.GetSize() == (64, 64)


def test_extract_features(tiny_synthetic_image, tmp_path):
    # Create a tiny config
    config_path = tmp_path / "radiomics.yaml"
    with open(config_path, "w") as f:
        f.write("""
imageType:
  Original: {}
setting:
  normalize: false
  binWidth: 5.0
  label: 1
featureClass:
  firstorder:
""")

    # Needs a mask with at least one target pixel
    mask = np.zeros_like(tiny_synthetic_image)
    mask[20:44, 20:44] = 1

    features = extract_features(tiny_synthetic_image, mask, config_path)
    assert isinstance(features, dict)
    # PyRadiomics should have extracted some original_firstorder features
    assert any(k.startswith("original_firstorder") for k in features.keys())
