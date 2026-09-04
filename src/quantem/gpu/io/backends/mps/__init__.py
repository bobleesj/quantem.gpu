"""MPS implementation details for QuantEM I/O.

The Metal decoder stays lazy so platform-independent modules such as
``mps.series`` remain importable on Linux and Windows.

The accepted schema-specific QGIX v3 backend remains a real explicit submodule
at ``mps.compact_v3``. It is not re-exported here because scientist-facing
source selection belongs to the canonical QuantEM I/O loader rather than a
second backend-specific load verb.
"""

from importlib import import_module
from typing import Any

__all__ = [
    "MPSChunked4DSTEM",
    "MPSDPCConfiguration",
    "MPSDPCMetrics",
    "MPSDPCProcessor",
    "MPSDPCResidentResult",
    "MPSPublicationRecorder",
    "MPSResidentCapabilities",
    "MPSTimingBoundary",
    "MPSTimingSummary",
    "clear_mps_cache",
    "describe_resident",
    "load_mps_4dstem",
    "load_prepared_frames",
]

_CONSUMER_NAMES = {
    "MPSPublicationRecorder",
    "MPSResidentCapabilities",
    "MPSTimingBoundary",
    "MPSTimingSummary",
    "describe_resident",
}

_DPC_NAMES = {
    "MPSDPCConfiguration",
    "MPSDPCMetrics",
    "MPSDPCProcessor",
    "MPSDPCResidentResult",
}


def __getattr__(name: str) -> Any:
    """Load public Metal decoder symbols only when they are requested."""
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    if name in _CONSUMER_NAMES:
        module = ".consumer"
    elif name in _DPC_NAMES:
        module = ".resident_dpc"
    else:
        module = ".decoder"
    value = getattr(import_module(module, __name__), name)
    globals()[name] = value
    return value
