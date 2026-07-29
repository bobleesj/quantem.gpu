"""Private device backends for :mod:`quantem.gpu.io`.

Scientists select a backend through :func:`quantem.gpu.io.load` or
:func:`quantem.gpu.io.save`; backend modules are implementation details.
"""

from .protocol import BackendName, detect_backend, resolve_backend

__all__ = ["BackendName", "detect_backend", "resolve_backend"]
