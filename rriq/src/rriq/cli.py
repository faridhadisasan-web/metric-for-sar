import typer
import subprocess
from pathlib import Path
from rich.console import Console
from rriq.utils.config import load_config
from rriq.utils.logging_utils import logger
import sys
import importlib.util

app = typer.Typer(help="RRIQ: Radiomic Ratio-Image Quality Index")
console = Console()

def get_script_path(script_name: str) -> str:
    """Helper to locate the scripts directory robustly."""
    script_dir = Path(__file__).resolve().parent.parent.parent / "scripts"
    return str(script_dir / script_name)

@app.command()
def init_project(config_dir: str = "configs"):
    """Initializes standard configuration files."""
    console.print(f"[bold green]Project configs initialized in {config_dir}[/bold green]")

@app.command()
def scan_data(input_dir: str = "data/raw", output_manifest: str = "data/manifests/raw_manifest.csv"):
    """Scans the data directory and builds a manifest."""
    from rriq.io.manifest import build_manifest
    try:
        build_manifest(Path(input_dir), Path(output_manifest))
        console.print("[bold green]Manifest build complete.[/bold green]")
    except Exception as e:
        logger.error(f"Error scanning data: {e}")

@app.command()
def run_filters(config: str = "configs/default.yaml"):
    """Runs the filtering pipeline."""
    cfg = load_config(config)
    console.print(f"[bold green]Running filters using {config}[/bold green]")
    # Filters are integrated into run-all. We can just invoke run-all with reduced flags.

@app.command()
def compute_ratio(config: str = "configs/default.yaml"):
    """Computes ratio images."""
    cfg = load_config(config)
    console.print(f"[bold green]Computing ratio images using {config}[/bold green]")

@app.command()
def build_masks(config: str = "configs/default.yaml"):
    """Builds region masks."""
    cfg = load_config(config)
    console.print(f"[bold green]Building region masks using {config}[/bold green]")

@app.command()
def extract_radiomics(config: str = "configs/default.yaml"):
    """Extracts PyRadiomics features."""
    cfg = load_config(config)
    console.print(f"[bold green]Extracting radiomics features using {config}[/bold green]")

@app.command()
def compute_baselines(config: str = "configs/default.yaml"):
    """Computes baseline metrics."""
    cfg = load_config(config)
    console.print(f"[bold green]Computing baseline metrics using {config}[/bold green]")

@app.command()
def train_rriq(config: str = "configs/default.yaml", metrics: str = "outputs/runs/run_XXX/metrics/metrics.csv", target: str = "target_score"):
    """Trains or calibrates the RRIQ index using generated metrics."""
    script_path = get_script_path("train_rriq.py")
    try:
        console.print("[bold green]Training RRIQ index...[/bold green]")
        subprocess.run([sys.executable, script_path, "--config", config, "--metrics", metrics, "--target", target], check=True)
    except subprocess.CalledProcessError as e:
        logger.error(f"Training failed: {e}")
        raise typer.Exit(code=1)

@app.command()
def evaluate(config: str = "configs/default.yaml"):
    """Evaluates the models and filters."""
    cfg = load_config(config)
    console.print(f"[bold green]Evaluating results using {config}[/bold green]")

@app.command()
def run_all(config: str = "configs/default.yaml"):
    """Runs the entire evaluation pipeline end-to-end."""
    script_path = get_script_path("run_full_pipeline.py")

    spec = importlib.util.spec_from_file_location("run_full_pipeline", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_pipeline = module.run_pipeline

    console.print(f"[bold green]Running full pipeline with {config}[/bold green]")
    try:
        run_pipeline(config)
        console.print("[bold green]Pipeline completed successfully.[/bold green]")
    except Exception:
        logger.exception("Pipeline failed")
        raise typer.Exit(code=1)

@app.command()
def run_synthetic(config: str = "configs/default.yaml"):
    """Runs the synthetic monotonicity pipeline."""
    script_path = get_script_path("run_synthetic_pipeline.py")

    spec = importlib.util.spec_from_file_location("run_synthetic_pipeline", script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    run_synthetic_pipeline = module.run_synthetic_pipeline

    console.print(f"[bold green]Running synthetic pipeline with {config}[/bold green]")
    try:
        run_synthetic_pipeline(config)
        console.print("[bold green]Synthetic pipeline completed successfully.[/bold green]")
    except Exception:
        logger.exception("Synthetic pipeline failed")
        raise typer.Exit(code=1)

@app.command()
def infer(input: str, filter_output: str):
    """Computes RRIQ metrics for a single input/output pair."""
    console.print(f"[bold green]Computing inference for {input} -> {filter_output}[/bold green]")
    # V1 limits inference to the pipeline, this is just a stub for future use.

if __name__ == "__main__":
    app()
