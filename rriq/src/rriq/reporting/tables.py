import pandas as pd
from pathlib import Path


def save_dataframe_to_markdown(
    df: pd.DataFrame, path: Path | str, title: str = "Table"
) -> None:
    """Saves a pandas DataFrame to a Markdown table."""
    with open(path, "w") as f:
        f.write(f"### {title}\n\n")
        f.write(df.to_markdown(index=False))
        f.write("\n\n")
