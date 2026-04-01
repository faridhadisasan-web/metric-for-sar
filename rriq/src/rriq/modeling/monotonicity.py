import pandas as pd
import scipy.stats
from typing import List

def screen_monotonic_features(df: pd.DataFrame, feature_cols: List[str], target_col: str, threshold: float = 0.5) -> List[str]:
    """
    Selects features that have a high absolute Spearman correlation with the severity target.
    """
    selected = []
    for f in feature_cols:
        corr, _ = scipy.stats.spearmanr(df[f], df[target_col])
        if abs(corr) >= threshold:
            selected.append(f)
    return selected
