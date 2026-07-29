"""Backend selection contract for accelerated STEM I/O."""
from __future__ import annotations

from typing import Literal, TypeAlias

from quantem.gpu.device import detect, resolve

BackendName: TypeAlias = Literal["cuda", "mps", "cpu"]
_VALID: tuple[BackendName, ...] = ("cuda", "mps", "cpu")


def detect_backend() -> Literal["cuda", "mps"]:
    """Select an accelerated backend without silently falling back to CPU."""
    return detect()


def resolve_backend(backend: str | None) -> BackendName:
    """Validate an explicit backend or resolve ``None``/``auto``."""
    if backend in (None, "auto"):
        return detect_backend()
    if backend not in _VALID:
        allowed = ", ".join(repr(name) for name in _VALID)
        raise ValueError(
            f"Unknown I/O backend {backend!r}. Use 'auto' or one of {allowed}."
        )
    if backend == "cpu":
        return backend
    return resolve(backend)
