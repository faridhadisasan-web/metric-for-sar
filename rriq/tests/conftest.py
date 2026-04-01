import pytest
import numpy as np


@pytest.fixture
def tiny_synthetic_image():
    """Generates a tiny 64x64 synthetic SAR-like image."""
    np.random.seed(42)
    # Background
    img = np.ones((64, 64), dtype=np.float32)
    # Target
    img[20:44, 20:44] = 5.0
    # Speckle noise (Gamma)
    speckle = np.random.gamma(shape=1.0, scale=1.0, size=(64, 64))
    return img * speckle


@pytest.fixture
def temp_run_dir(tmp_path):
    run_dir = tmp_path / "test_run"
    run_dir.mkdir()
    return run_dir
