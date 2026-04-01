import pandas as pd
import scipy.stats


def compute_rank_correlation(df: pd.DataFrame, score_col: str, target_col: str) -> dict:
    """Computes Spearman and Kendall rank correlation."""
    spearman_corr, spearman_p = scipy.stats.spearmanr(df[score_col], df[target_col])
    kendall_corr, kendall_p = scipy.stats.kendalltau(df[score_col], df[target_col])

    return {
        "spearman": float(spearman_corr),
        "spearman_p": float(spearman_p),
        "kendall": float(kendall_corr),
        "kendall_p": float(kendall_p),
    }


def rank_filters(df: pd.DataFrame, score_col: str = "RRIQ") -> pd.DataFrame:
    """Ranks filters based on the aggregated score."""
    avg_scores = df.groupby("filter_name")[score_col].mean().reset_index()
    return avg_scores.sort_values(by=score_col, ascending=False).reset_index(drop=True)
