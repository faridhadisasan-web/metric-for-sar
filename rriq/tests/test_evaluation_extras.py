import pandas as pd
from rriq.evaluation.ablation import compute_ablation
from rriq.evaluation.robustness import compute_robustness


def test_compute_robustness():
    df = pd.DataFrame(
        {
            "image_id": [1, 1, 2, 2],
            "filter_name": ["A", "B", "A", "B"],
            "RRIQ": [100, 50, 90, 60],  # A is always better
        }
    )
    rob = compute_robustness(df)
    assert isinstance(rob, float)
    # std deviation of rank should be 0 because A is always rank 1 and B is rank 2
    assert rob == 0.0


def test_compute_ablation():
    df = pd.DataFrame(
        {"RRIQ": [10, 20, 30], "target": [1, 2, 3], "f1": [1, 2, 3], "f2": [3, 2, 1]}
    )
    res = compute_ablation(df, ["f1", "f2"], "target")
    assert "baseline" in res
    assert "ablated_f1" in res
    assert "ablated_f2" in res
