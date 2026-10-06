"""Arrays held in one shared Metal buffer: query results and staged blocks on the Apple GPU."""

import math

import numpy as np

from quantem.gpu.device.metal_runtime import (
    allocate_shared,
    buffer_view,
    copy_to_torch,
    release_buffer,
)


class MetalArray:
    """Own one array in a shared Metal buffer until ``release``.

    Kernels bind the buffer ``_mtl`` directly, the name ``SharedArray`` also
    uses. ``get`` (as on a CuPy array, so CUDA and MPS results read back the
    same way) and ``to_torch`` copy the array out; neither borrows a view that
    outlives the buffer. PyObjC never frees a Metal buffer when its wrapper is
    collected, so callers release a result as soon as its consumer finishes;
    collection releases whatever is left.
    """

    def __init__(self, shape, dtype, buffer=None):
        self.shape = tuple(shape)
        self.dtype = np.dtype(dtype)
        # Metal rejects zero-length buffers.
        self._mtl = buffer if buffer is not None else allocate_shared(max(4, self.nbytes), "Metal array")

    @property
    def ndim(self) -> int:
        return len(self.shape)

    @property
    def size(self) -> int:
        return math.prod(self.shape)

    @property
    def nbytes(self) -> int:
        return self.size * self.dtype.itemsize

    @property
    def is_released(self) -> bool:
        return self._mtl is None

    def get(self) -> np.ndarray:
        """Copy the array to a host array of the same shape and dtype, as ``cupy.ndarray.get`` does."""
        if self._mtl is None:
            raise RuntimeError("This Metal result was released; request it again.")
        return np.frombuffer(buffer_view(self._mtl), self.dtype, count=self.size).reshape(self.shape).copy()

    def to_torch(self):
        """Copy the array into a Torch MPS tensor."""
        if self._mtl is None:
            raise RuntimeError("This Metal result was released; request it again.")
        return copy_to_torch(self._mtl, self.shape, self.dtype, "Metal array Torch transfer")

    def __array__(self, dtype=None, copy=None):
        return np.asarray(self.get(), dtype=dtype)

    def release(self) -> None:
        """Return the buffer now; a second call does nothing."""
        buffer, self._mtl = self._mtl, None
        release_buffer(buffer)

    def __del__(self):
        # A failed __init__ leaves no _mtl (AttributeError), and at interpreter
        # shutdown the module globals may already be None (TypeError).
        try:
            self.release()
        except (AttributeError, TypeError):
            pass
