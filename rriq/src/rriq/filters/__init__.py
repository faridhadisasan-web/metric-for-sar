"""Registry and implementations of classical and optional deep filters."""

# Import all filters to trigger the registration decorators
from rriq.filters import classical
from rriq.filters import wrappers

__all__ = ["classical", "wrappers"]
