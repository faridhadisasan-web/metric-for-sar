import pandas as pd
from typing import List, Dict


def compute_ablation(
    df: pd.DataFrame, feature_cols: List[str], target_col: str
) -> Dict[str, float]:
    """
    Computes a simple ablation study by checking correlation
    when one feature/subscore is removed from the aggregation.
    """
    results = {}
    from rriq.evaluation.ranking import compute_rank_correlation

    # Baseline performance
    baseline_corr = compute_rank_correlation(df, "RRIQ", target_col)["spearman"]
    results["baseline"] = baseline_corr

    for f in feature_cols:
        # Create an ablated score (simple average of remaining)
        remaining = [c for c in feature_cols if c != f]
        if remaining:
            ablated_score = df[remaining].mean(axis=1)
            temp_df = df.copy()
            temp_df["ablated"] = ablated_score
            ablated_corr = compute_rank_correlation(temp_df, "ablated", target_col)[
                "spearman"
            ]
            results[f"ablated_{f}"] = ablated_corr

    return results
