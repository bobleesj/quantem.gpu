"""Optional CuPy access without importing an accelerator during discovery."""

from importlib import import_module
from importlib.util import find_spec


class _LazyCuPy:
    """Resolve and cache runtime attributes only when a CUDA path uses them."""

    def __getattr__(self, name: str):
        value = getattr(import_module("cupy"), name)
        setattr(self, name, value)
        return value


# Missing CuPy remains distinguishable from an installed but broken runtime.
# Runtime import failures propagate on use; they must not select a CPU fallback.
try:
    _available = find_spec("cupy") is not None
except ModuleNotFoundError:
    _available = False
cp = _LazyCuPy() if _available else None
