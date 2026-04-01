import pandas as pd
from rriq.modeling.calibrate import build_unsupervised_rriq
from rriq.modeling.export import export_rriq_formula
from rriq.evaluation.ranking import compute_rank_correlation, rank_filters


def test_build_unsupervised_rriq():
    df = pd.DataFrame(
        {
            "filter_name": ["A", "B"],
            "mor": [0.9, 0.5],
            "epd_roa": [0.8, 0.4],
            "rgpi": [0.95, 0.6],
            "tcr": [2.0, 1.0],
        }
    )
    weights = {
        "residual_purity": 0.35,
        "structure_preservation": 0.30,
        "radiometric_preservation": 0.20,
        "scatterer_preservation": 0.15,
    }
    res = build_unsupervised_rriq(df, weights)
    assert "RRIQ" in res.columns
    # Filter A should be strictly better than B based on these stats
    assert res.loc[0, "RRIQ"] > res.loc[1, "RRIQ"]


def test_export_rriq_formula(tmp_path):
    weights = {"w1": 0.5, "w2": 0.5}
    out_path = tmp_path / "formula.json"
    export_rriq_formula(weights, out_path)
    assert out_path.exists()
    assert "w1" in out_path.read_text()


def test_compute_rank_correlation():
    df = pd.DataFrame({"score": [1, 2, 3, 4], "target": [2, 1, 4, 3]})
    res = compute_rank_correlation(df, "score", "target")
    assert "spearman" in res
    assert "kendall" in res


def test_rank_filters():
    df = pd.DataFrame({"filter_name": ["A", "B", "A", "B"], "RRIQ": [10, 5, 12, 6]})
    ranked = rank_filters(df)
    assert ranked.iloc[0]["filter_name"] == "A"
