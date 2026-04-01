import numpy as np
import pytest
from rriq.io.validators import validate_sar_image
from rriq.io.writers import save_image
from rriq.io.readers import read_image


def test_validate_sar_image_valid():
    img = np.random.rand(100, 100)
    validated = validate_sar_image(img)
    assert validated.shape == (100, 100)
    assert validated.dtype == np.float32


def test_validate_sar_image_complex_error():
    img = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    with pytest.raises(ValueError, match="complex"):
        validate_sar_image(img, coerce_complex=False)


def test_validate_sar_image_complex_coerce():
    img = np.random.rand(10, 10) + 1j * np.random.rand(10, 10)
    validated = validate_sar_image(img, coerce_complex=True)
    assert validated.dtype == np.float32
    np.testing.assert_allclose(validated, np.abs(img), rtol=1e-5)


def test_validate_sar_image_multichannel_error():
    img = np.random.rand(10, 10, 3)
    with pytest.raises(ValueError, match="dimensions"):
        validate_sar_image(img, reduce_multichannel="error")


def test_save_and_read_image(tmp_path, tiny_synthetic_image):
    file_path = tmp_path / "test_img.tif"
    save_image(tiny_synthetic_image, file_path)

    loaded = read_image(file_path)
    assert loaded.shape == tiny_synthetic_image.shape
    np.testing.assert_allclose(
        loaded, tiny_synthetic_image.astype(np.float32), atol=1e-4
    )
