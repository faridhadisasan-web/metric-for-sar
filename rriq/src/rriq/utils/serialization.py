import json
import numpy as np
from pathlib import Path


class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):
            return obj.item()
        return super().default(obj)


def save_json(data: dict, path: Path) -> None:
    with open(path, "w") as f:
        json.dump(data, f, cls=NumpyEncoder, indent=2)
