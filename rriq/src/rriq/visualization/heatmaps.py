import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
from pathlib import Path


def plot_correlation_heatmap(df: pd.DataFrame, output_path: Path) -> None:
    """Plots a feature correlation heatmap."""
    # Ensure only numeric cols are plotted
    numeric_df = df.select_dtypes(include=["number"]).dropna(axis=1)
    corr = numeric_df.corr()

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False)
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig(output_path, dpi=300)
    plt.close()
