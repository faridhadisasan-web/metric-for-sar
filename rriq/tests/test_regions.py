import numpy as np
from rriq.regions.homogeneous import build_homogeneous_mask
from rriq.regions.edges import build_edge_mask
from rriq.regions.scatterers import build_scatterer_mask
from rriq.regions.masks import generate_all_masks


def test_build_homogeneous_mask(tiny_synthetic_image):
    mask = build_homogeneous_mask(tiny_synthetic_image)
    assert mask.shape == tiny_synthetic_image.shape
    assert set(np.unique(mask)).issubset({0, 1})


def test_build_edge_mask(tiny_synthetic_image):
    mask = build_edge_mask(tiny_synthetic_image)
    assert mask.shape == tiny_synthetic_image.shape
    assert set(np.unique(mask)).issubset({0, 1})


def test_build_scatterer_mask(tiny_synthetic_image):
    mask = build_scatterer_mask(tiny_synthetic_image, percentile=95)
    assert mask.shape == tiny_synthetic_image.shape
    assert set(np.unique(mask)).issubset({0, 1})


def test_generate_all_masks(tiny_synthetic_image):
    config = {
        "homogeneous": {"threshold": 0.2},
        "edges": {"method": "sobel", "threshold": 0.05},
        "scatterers": {"percentile": 99},
    }
    masks = generate_all_masks(tiny_synthetic_image, config)
    assert "homogeneous" in masks
    assert "edges" in masks
    assert "scatterers" in masks
