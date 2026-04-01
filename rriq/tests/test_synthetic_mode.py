import pandas as pd
from rriq.modeling.synthetic_mode import generate_synthetic_degradations
from rriq.modeling.monotonicity import screen_monotonic_features

def test_generate_synthetic_degradations(tiny_synthetic_image):
    ladder = [1, 2, 4]
    degs = generate_synthetic_degradations(tiny_synthetic_image, ladder)

    assert "looks_1" in degs
    assert "looks_4" in degs

    # Severity should be increasing with looks
    assert degs["looks_1"]["severity_target"] < degs["looks_4"]["severity_target"]
    assert degs["looks_1"]["image"].shape == tiny_synthetic_image.shape

def test_screen_monotonic_features():
    df = pd.DataFrame({
        "severity_target": [0.0, 0.5, 1.0],
        "good_feat": [10, 20, 30], # perfectly correlated
        "bad_feat": [10, 5, 10]    # not monotonic
    })

    res = screen_monotonic_features(df, ["good_feat", "bad_feat"], "severity_target", threshold=0.9)
    assert "good_feat" in res
    assert "bad_feat" not in res
