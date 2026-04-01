import pandas as pd
from sklearn.linear_model import ElasticNet
from sklearn.preprocessing import StandardScaler
from typing import Dict, Any, Tuple
import joblib
from pathlib import Path
from rriq.utils.logging_utils import logger
from rriq.modeling.feature_screening import screen_features, remove_highly_correlated

def train_supervised_rriq(
    df: pd.DataFrame,
    target_col: str,
    model_path: Path | str,
    scaler_path: Path | str
) -> Tuple[Any, Any, list, Dict[str, float]]:
    """
    Trains a sparse interpretable model (ElasticNet) to predict the target score.
    Uses all numeric columns that start with 'original_' (radiomics) or standard baselines.
    """
    # 1. Identify features
    candidates = [c for c in df.columns if c.startswith("original_") or c in ["enl", "epd_roa", "mor", "tcr", "rgpi"]]
    candidates = [c for c in candidates if pd.api.types.is_numeric_dtype(df[c])]

    # 2. Screening
    candidates = screen_features(df, candidates)

    # Drop NaNs
    train_df = df.dropna(subset=candidates + [target_col]).copy()
    if len(train_df) < 5:
        logger.warning("Not enough samples to train supervised model. Returning naive model.")
        return None, None, [], {}

    # 3. Correlation Pruning
    features = remove_highly_correlated(train_df, candidates, threshold=0.9)
    logger.info(f"Selected {len(features)} features for training after pruning.")

    X = train_df[features]
    y = train_df[target_col]

    # 4. Scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # 5. Modeling (Sparse linear model)
    model = ElasticNet(alpha=0.1, l1_ratio=0.5, random_state=42, positive=False)
    # Using simple ElasticNet. For strictly monotonic GAM, we'd use PyGAM or similar,
    # but ElasticNet fits the sparse interpretable constraint nicely without extra heavy dependencies.
    model.fit(X_scaled, y)

    # 6. Extract weights
    weights = {f: float(coef) for f, coef in zip(features, model.coef_) if abs(coef) > 1e-4}

    # 7. Save artifacts
    joblib.dump(model, model_path)
    joblib.dump(scaler, scaler_path)

    return model, scaler, features, weights
