"""Private-loopback remote viewing for native 4D-STEM clients."""

from .maped_api import MAPEDProtocolError, MAPEDProtocolService
from .prepare import prepare_browse_source

__all__ = [
    "BrowseService",
    "CompactBrowseSource",
    "MAPEDProtocolError",
    "MAPEDProtocolService",
    "create_app",
    "load_compact_browse_sources",
    "prepare_browse_source",
]


def __getattr__(name: str) -> object:
    if name in {
        "BrowseService",
        "CompactBrowseSource",
        "create_app",
        "load_compact_browse_sources",
    }:
        from .server import (
            BrowseService,
            CompactBrowseSource,
            create_app,
            load_compact_browse_sources,
        )

        return {
            "BrowseService": BrowseService,
            "CompactBrowseSource": CompactBrowseSource,
            "create_app": create_app,
            "load_compact_browse_sources": load_compact_browse_sources,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
