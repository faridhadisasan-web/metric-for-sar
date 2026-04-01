import pandas as pd
import numpy as np
from typing import Dict


def normalize_series(series: pd.Series) -> pd.Series:
    """Min-max normalizes a pandas Series to [0, 100]."""
    min_v, max_v = series.min(), series.max()
    if max_v > min_v:
        return (series - min_v) / (max_v - min_v) * 100.0
    return pd.Series(np.zeros_like(series), index=series.index)


def build_unsupervised_rriq(
    df: pd.DataFrame, weights: Dict[str, float]
) -> pd.DataFrame:
    """
    Constructs the default unsupervised interpretable RRIQ metric.
    Assumes `df` contains the following normalized baseline proxies:
    - `residual_purity`: mapped from MOR or divergence
    - `structure_preservation`: mapped from EPD-ROA
    - `radiometric_preservation`: mapped from RGPI
    - `scatterer_preservation`: mapped from TCR
    """
    df = df.copy()

    # We will compute pseudo-subscores based on typical canonical metrics.
    # Higher is better for all of these in our formulation.
    if "mor" in df.columns:
        # MOR ~ 1 is best, so we use negative absolute deviation from 1
        df["residual_purity"] = -np.abs(1.0 - df["mor"])
    else:
        df["residual_purity"] = 0.0

    if "epd_roa" in df.columns:
        df["structure_preservation"] = df["epd_roa"]
    else:
        df["structure_preservation"] = 0.0

    if "rgpi" in df.columns:
        df["radiometric_preservation"] = df["rgpi"]
    else:
        df["radiometric_preservation"] = 0.0

    if "tcr" in df.columns:
        df["scatterer_preservation"] = df["tcr"]
    else:
        df["scatterer_preservation"] = 0.0

    # Normalize subscores
    for subscore in [
        "residual_purity",
        "structure_preservation",
        "radiometric_preservation",
        "scatterer_preservation",
    ]:
        df[f"{subscore}_norm"] = normalize_series(df[subscore])

    # Compute RRIQ
    rriq = (
        df["residual_purity_norm"] * weights.get("residual_purity", 0.35)
        + df["structure_preservation_norm"]
        * weights.get("structure_preservation", 0.30)
        + df["radiometric_preservation_norm"]
        * weights.get("radiometric_preservation", 0.20)
        + df["scatterer_preservation_norm"]
        * weights.get("scatterer_preservation", 0.15)
    )

    # Final normalization to ensure exact [0, 100] range
    df["RRIQ"] = normalize_series(rriq)
    return df
