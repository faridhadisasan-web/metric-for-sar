from typing import Dict

def flatten_features(features: Dict[str, float], prefix: str = "") -> Dict[str, float]:
    """Prefixes feature names for flat dictionary storage."""
    return {f"{prefix}_{k}" if prefix else k: v for k, v in features.items()}
