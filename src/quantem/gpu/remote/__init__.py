"""Private-loopback remote viewing for native 4D-STEM clients."""

from .maped_api import MAPEDProtocolError, MAPEDProtocolService

__all__ = [
    "BrowseService",
    "MAPEDProtocolError",
    "MAPEDProtocolService",
    "create_app",
]


def __getattr__(name: str) -> object:
    if name in {"BrowseService", "create_app"}:
        from .server import BrowseService, create_app

        return {"BrowseService": BrowseService, "create_app": create_app}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
