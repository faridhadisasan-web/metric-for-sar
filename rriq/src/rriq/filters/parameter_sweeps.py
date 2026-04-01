import numpy as np
from typing import Dict, Any, List
from itertools import product
from rriq.filters.registry import get_filter

def run_filter_sweep(
    image: np.ndarray, filter_name: str, params: Dict[str, Any]
) -> List[tuple]:
    """
    Runs a filter with multiple parameter combinations.
    params should be a dict where values are lists of options to sweep over.
    Returns a list of tuples: (param_dict, filtered_image)
    """
    func = get_filter(filter_name)

    # Find combinations
    keys, values = zip(*params.items())
    combinations = [
        dict(zip(keys, v))
        for v in product(*[v if isinstance(v, list) else [v] for v in values])
    ]

    results = []
    for kwargs in combinations:
        res = func(image, **kwargs)
        results.append((kwargs, res))

    return results
