import pandas as pd


def compute_robustness(df: pd.DataFrame, score_col: str = "RRIQ") -> float:
    """
    Evaluates metric robustness across different scenes/datasets.
    Computes the average standard deviation of filter ranks across different images.
    Lower standard deviation implies higher robustness.
    """
    if "image_id" not in df.columns or "filter_name" not in df.columns:
        return 0.0

    df["rank"] = df.groupby("image_id")[score_col].rank(ascending=False)
    # Variance of the rank for each filter across images
    rank_stds = df.groupby("filter_name")["rank"].std()

    # Average standard deviation
    return float(rank_stds.mean())
