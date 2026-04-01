from pathlib import Path


def generate_markdown_report(run_dir: Path, config: dict, metrics_df) -> None:
    """Generates a summary report of the run."""
    report_path = run_dir / "report.md"

    with open(report_path, "w") as f:
        f.write("# RRIQ Run Report\n\n")

        f.write("## Configuration Summary\n")
        f.write(f"- **Input Dir:** `{config.get('io', {}).get('input_dir')}`\n")
        f.write(f"- **Output Dir:** `{run_dir}`\n")

        f.write("\n## Run Metrics\n")
        if metrics_df is not None:
            # Averages by filter
            avg_metrics = metrics_df.groupby("filter_name").mean(numeric_only=True)
            f.write(avg_metrics.to_markdown())
            f.write("\n\n")

        f.write("## Key Figures\n")
        f.write("See `visualizations/` folder for complete plots.\n")
