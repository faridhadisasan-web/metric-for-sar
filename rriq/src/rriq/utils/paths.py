from pathlib import Path
from datetime import datetime


def get_run_dir(base_dir: str | Path, name: str = "run") -> Path:
    """Creates and returns a new run directory stamped with the current time."""
    base = Path(base_dir)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = base / f"{name}_{timestamp}"
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def ensure_dir(path: Path) -> Path:
    """Ensures a directory exists, creating it if necessary."""
    path.mkdir(parents=True, exist_ok=True)
    return path
