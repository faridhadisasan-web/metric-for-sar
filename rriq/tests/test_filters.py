import numpy as np
from rriq.filters.classical import filter_median, filter_lee, filter_kuan, filter_frost
from rriq.filters.registry import get_filter


def test_median_filter(tiny_synthetic_image):
    filtered = filter_median(tiny_synthetic_image, size=3)
    assert filtered.shape == tiny_synthetic_image.shape


def test_lee_filter(tiny_synthetic_image):
    filtered = filter_lee(tiny_synthetic_image, size=5)
    assert filtered.shape == tiny_synthetic_image.shape
    assert not np.any(np.isnan(filtered))


def test_kuan_filter(tiny_synthetic_image):
    filtered = filter_kuan(tiny_synthetic_image, size=5)
    assert filtered.shape == tiny_synthetic_image.shape
    assert not np.any(np.isnan(filtered))


def test_frost_filter(tiny_synthetic_image):
    filtered = filter_frost(tiny_synthetic_image, size=5, damp=1.0)
    assert filtered.shape == tiny_synthetic_image.shape
    assert not np.any(np.isnan(filtered))


def test_filter_registry():
    func = get_filter("lee")
    assert callable(func)
