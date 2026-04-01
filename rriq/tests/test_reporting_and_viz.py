import numpy as np
import pandas as pd
from rriq.visualization.plots import plot_boxplot_by_filter
from rriq.visualization.qc_panels import plot_triplet
from rriq.visualization.heatmaps import plot_correlation_heatmap
from rriq.reporting.tables import save_dataframe_to_markdown
from rriq.reporting.markdown_report import generate_markdown_report


def test_plot_boxplot_by_filter(tmp_path):
    df = pd.DataFrame(
        {
            "filter_name": ["Lee", "Lee", "Frost", "Frost"],
            "enl": [10.5, 11.2, 9.8, 10.1],
        }
    )
    output_path = tmp_path / "boxplot.png"
    plot_boxplot_by_filter(df, "enl", output_path)
    assert output_path.exists()


def test_plot_triplet(tiny_synthetic_image, tmp_path):
    filtered = tiny_synthetic_image * 0.9
    ratio = tiny_synthetic_image / (filtered + 1e-8)
    output_path = tmp_path / "triplet.png"
    plot_triplet(tiny_synthetic_image, filtered, ratio, output_path)
    assert output_path.exists()


def test_plot_correlation_heatmap(tmp_path):
    df = pd.DataFrame({"f1": np.random.rand(10), "f2": np.random.rand(10)})
    output_path = tmp_path / "heatmap.png"
    plot_correlation_heatmap(df, output_path)
    assert output_path.exists()


def test_save_dataframe_to_markdown(tmp_path):
    df = pd.DataFrame({"A": [1, 2], "B": [3, 4]})
    output_path = tmp_path / "table.md"
    save_dataframe_to_markdown(df, output_path, title="Test Table")
    assert output_path.exists()
    content = output_path.read_text()
    assert "Test Table" in content
    assert "A" in content


def test_generate_markdown_report(tmp_path):
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    config = {"io": {"input_dir": "data/raw"}}
    df = pd.DataFrame({"filter_name": ["Lee", "Frost"], "enl": [10.5, 9.8]})
    generate_markdown_report(run_dir, config, df)
    report_path = run_dir / "report.md"
    assert report_path.exists()
    assert "RRIQ Run Report" in report_path.read_text()
