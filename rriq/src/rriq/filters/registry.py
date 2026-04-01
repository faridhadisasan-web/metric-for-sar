from typing import Callable, Dict

FILTER_REGISTRY: Dict[str, Callable] = {}


def register_filter(name: str):
    """Decorator to register a filtering function in the registry."""

    def decorator(func: Callable):
        FILTER_REGISTRY[name.lower()] = func
        return func

    return decorator


def get_filter(name: str) -> Callable:
    """Retrieves a filter from the registry by its name."""
    if name.lower() not in FILTER_REGISTRY:
        raise KeyError(
            f"Filter '{name}' not found. Available: {list(FILTER_REGISTRY.keys())}"
        )
    return FILTER_REGISTRY[name.lower()]
