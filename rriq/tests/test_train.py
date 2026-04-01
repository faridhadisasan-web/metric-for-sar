import pandas as pd
import numpy as np
from rriq.modeling.train import train_supervised_rriq
from rriq.modeling.infer import infer_rriq

def test_train_and_infer_rriq(tmp_path):
    # Dummy data
    np.random.seed(42)
    df = pd.DataFrame({
        "original_firstorder_Energy": np.random.rand(20),
        "original_glcm_Contrast": np.random.rand(20),
        "enl": np.random.rand(20),
        "target_score": np.random.rand(20)
    })

    model_path = tmp_path / "model.pkl"
    scaler_path = tmp_path / "scaler.pkl"

    model, scaler, features, weights = train_supervised_rriq(df, "target_score", model_path, scaler_path)

    assert model is not None
    assert scaler is not None
    assert len(features) > 0
    assert isinstance(weights, dict)

    # Test inference
    test_df = pd.DataFrame({
        "original_firstorder_Energy": np.random.rand(5),
        "original_glcm_Contrast": np.random.rand(5),
        "enl": np.random.rand(5)
    })

    res = infer_rriq(test_df, model_path, scaler_path, features)
    assert "RRIQ" in res.columns
    assert len(res) == 5
    assert res["RRIQ"].min() >= 0
    assert res["RRIQ"].max() <= 100
