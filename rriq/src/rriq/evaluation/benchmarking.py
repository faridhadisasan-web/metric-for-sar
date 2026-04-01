import pandas as pd
from pathlib import Path


def export_benchmark_report(df: pd.DataFrame, output_path: Path | str) -> None:
    """Generates a benchmarking report CSV."""
    from rriq.evaluation.ranking import rank_filters
    from rriq.evaluation.robustness import compute_robustness

    ranked = rank_filters(df, "RRIQ")
    rob = compute_robustness(df, "RRIQ")

    # Save ranked table
    ranked.to_csv(output_path, index=False)
