import yaml
from pathlib import Path
from typing import Dict, Any


def load_config(path: str | Path) -> Dict[str, Any]:
    """Loads a YAML configuration file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_config(config: Dict[str, Any], path: str | Path) -> None:
    """Saves a configuration dictionary to a YAML file."""
    with open(path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)
