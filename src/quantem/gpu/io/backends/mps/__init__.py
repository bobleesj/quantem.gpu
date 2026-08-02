"""MPS implementation details for QuantEM I/O.

The Metal decoder stays lazy so platform-independent modules such as
``mps.series`` remain importable on Linux and Windows.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "MPSChunked4DSTEM",
    "clear_mps_cache",
    "load_mps_4dstem",
    "load_prepared_frames",
]


def __getattr__(name: str) -> Any:
    """Load public Metal decoder symbols only when they are requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    value = getattr(import_module(".decoder", __name__), name)
    globals()[name] = value
    return value
