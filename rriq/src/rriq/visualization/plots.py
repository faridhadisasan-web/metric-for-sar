import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path


def plot_boxplot_by_filter(df, metric: str, output_path: Path) -> None:
    """Creates a boxplot comparing a single metric across different filters."""
    plt.figure(figsize=(10, 6))
    sns.boxplot(x="filter_name", y=metric, data=df)
    plt.title(f"Comparison of {metric.upper()} by Filter")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
