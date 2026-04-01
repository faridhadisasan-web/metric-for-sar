import numpy as np
from rriq.metrics.enl import compute_enl
from rriq.metrics.epd_roa import compute_epd_roa
from rriq.metrics.mor import compute_mor
from rriq.metrics.tcr import compute_tcr
from rriq.metrics.rgpi import compute_rgpi
from rriq.metrics.composite_baselines import compute_all_baselines


def test_compute_enl(tiny_synthetic_image):
    mask = np.ones_like(tiny_synthetic_image)
    enl = compute_enl(tiny_synthetic_image, mask)
    assert isinstance(enl, float)
    assert enl > 0


def test_compute_epd_roa(tiny_synthetic_image):
    filtered = tiny_synthetic_image * 0.9
    epd = compute_epd_roa(tiny_synthetic_image, filtered)
    assert isinstance(epd, float)
    assert epd > 0


def test_compute_mor(tiny_synthetic_image):
    ratio = tiny_synthetic_image / (tiny_synthetic_image * 0.9 + 1e-8)
    mor = compute_mor(ratio)
    assert isinstance(mor, float)
    assert mor > 0


def test_compute_tcr(tiny_synthetic_image):
    scatterer_mask = np.zeros_like(tiny_synthetic_image)
    scatterer_mask[20:44, 20:44] = 1
    background_mask = np.ones_like(tiny_synthetic_image) - scatterer_mask

    tcr = compute_tcr(tiny_synthetic_image, scatterer_mask, background_mask)
    assert isinstance(tcr, float)
    assert tcr >= 0


def test_compute_rgpi(tiny_synthetic_image):
    filtered = tiny_synthetic_image * 0.9
    rgpi = compute_rgpi(tiny_synthetic_image, filtered)
    assert isinstance(rgpi, float)
    assert rgpi > 0


def test_compute_all_baselines(tiny_synthetic_image):
    filtered = tiny_synthetic_image * 0.9
    ratio = tiny_synthetic_image / (filtered + 1e-8)

    masks = {
        "homogeneous": np.ones_like(tiny_synthetic_image),
        "scatterers": np.zeros_like(tiny_synthetic_image),
    }
    masks["scatterers"][20:44, 20:44] = 1

    metrics = compute_all_baselines(tiny_synthetic_image, filtered, ratio, masks)
    assert "enl" in metrics
    assert "epd_roa" in metrics
    assert "mor" in metrics
    assert "tcr" in metrics
    assert "rgpi" in metrics
