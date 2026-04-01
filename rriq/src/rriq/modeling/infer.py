import pandas as pd
import joblib
from pathlib import Path

def infer_rriq(df: pd.DataFrame, model_path: Path | str, scaler_path: Path | str, features: list) -> pd.DataFrame:
    """Applies a trained supervised RRIQ model to new data."""
    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)

    # Fill missing features with 0 if any
    X = df[features].fillna(0)
    X_scaled = scaler.transform(X)

    preds = model.predict(X_scaled)
    df = df.copy()

    # Min-Max normalize predictions to [0, 100] scale
    min_v, max_v = preds.min(), preds.max()
    if max_v > min_v:
        df["RRIQ"] = (preds - min_v) / (max_v - min_v) * 100.0
    else:
        df["RRIQ"] = 50.0 # Default fallback

    return df
