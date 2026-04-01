import json
from pathlib import Path
from typing import Dict


def export_rriq_formula(weights: Dict[str, float], path: Path | str) -> None:
    """Exports the human-readable RRIQ formula and weights."""
    formula = "RRIQ = " + " + ".join([f"({w} * {k})" for k, w in weights.items()])

    data = {
        "formula": formula,
        "weights": weights,
        "normalization": "min-max scaling to [0, 100]",
    }

    with open(path, "w") as f:
        json.dump(data, f, indent=2)
