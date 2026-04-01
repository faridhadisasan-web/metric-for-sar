import pandas as pd
import numpy as np
from typing import List

def screen_features(df: pd.DataFrame, feature_cols: List[str], variance_threshold: float = 1e-4) -> List[str]:
    """Filters out low-variance and highly missing features."""
    valid_features = []
    for f in feature_cols:
        if df[f].isnull().mean() > 0.1:
            continue
        if df[f].var() < variance_threshold:
            continue
        valid_features.append(f)
    return valid_features

def remove_highly_correlated(df: pd.DataFrame, feature_cols: List[str], threshold: float = 0.9) -> List[str]:
    """Removes highly correlated features to prevent multicollinearity."""
    corr_matrix = df[feature_cols].corr().abs()
    upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
    return [f for f in feature_cols if f not in to_drop]
