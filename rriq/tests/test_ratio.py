import numpy as np
from rriq.ratio.compute import compute_ratio, compute_log_ratio, get_ratio_stats
from rriq.ratio.stats import calculate_divergence
from rriq.ratio.divergence import kl_divergence_gamma


def test_compute_ratio(tiny_synthetic_image):
    filtered = tiny_synthetic_image + 1.0  # Dummy filter
    ratio = compute_ratio(tiny_synthetic_image, filtered)
    assert ratio.shape == tiny_synthetic_image.shape
    assert np.all(ratio >= 0)


def test_compute_log_ratio(tiny_synthetic_image):
    filtered = tiny_synthetic_image + 1.0
    ratio = compute_ratio(tiny_synthetic_image, filtered)
    log_ratio = compute_log_ratio(ratio)
    assert log_ratio.shape == tiny_synthetic_image.shape


def test_get_ratio_stats(tiny_synthetic_image):
    filtered = tiny_synthetic_image + 1.0
    ratio = compute_ratio(tiny_synthetic_image, filtered)
    stats = get_ratio_stats(ratio)
    assert "mean" in stats
    assert "std" in stats
    assert "min" in stats
    assert "max" in stats
    assert "median" in stats


def test_calculate_divergence(tiny_synthetic_image):
    filtered = tiny_synthetic_image + 1.0
    ratio = compute_ratio(tiny_synthetic_image, filtered)
    div = calculate_divergence(ratio, theoretical_mean=1.0)
    assert isinstance(div, float)


def test_kl_divergence_gamma():
    # Generate data from a known Gamma distribution
    np.random.seed(42)
    gamma_data = np.random.gamma(shape=1.0, scale=1.0, size=(100, 100))
    kl = kl_divergence_gamma(gamma_data, looks=1)
    # The divergence should be close to 0 since the empirical distribution matches the theoretical one
    assert kl < 0.1
