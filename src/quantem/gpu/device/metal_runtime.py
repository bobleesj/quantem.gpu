"""Metal device, pipelines, buffers and command completion shared by every MPS module.

Every Metal path in the package compiles kernels, allocates unified-memory
buffers, reads them from the host, frees them and waits on command buffers the
same way. One copy means a fix to compilation or buffer ownership reaches every
codec. PyObjC Metal is imported on first use, so these modules import on Linux.
"""

import ctypes
import os
from functools import cache

import numpy as np


def metal_module():
    """Import PyObjC Metal on first use so Linux imports of the codecs succeed."""
    try:
        import Metal
    except ImportError as error:
        raise RuntimeError("Metal acceleration requires pyobjc-framework-Metal") from error
    return Metal


@cache
def metal_device():
    """Return the system GPU; every module allocates on and compiles for this one device."""
    device = metal_module().MTLCreateSystemDefaultDevice()
    if device is None:
        raise RuntimeError("Metal acceleration needs an available Apple GPU.")
    return device


@cache
def metal_queue():
    """Return the one command queue; every caller waits for its commands, so they never interleave."""
    return metal_device().newCommandQueue()


@cache
def metal_pipelines(source: str, names: tuple[str, ...], *, fast_math: bool = True) -> dict:
    """Compile Metal ``source`` once and return a compute pipeline for each function in ``names``.

    Compiling takes seconds, so each library is built once per process.
    ``fast_math=False`` keeps IEEE float semantics for kernels whose results
    must match CUDA bit for bit; integer codecs keep Metal's default options.
    """
    metal = metal_module()
    device = metal_device()
    options = metal.MTLCompileOptions.alloc().init()
    if not fast_math:
        options.setFastMathEnabled_(False)
    library, error = device.newLibraryWithSource_options_error_(source, options, None)
    if library is None or error:
        raise RuntimeError(f"Metal kernel compilation failed: {error}")
    pipelines = {}
    for name in names:
        function = library.newFunctionWithName_(name)
        if function is None:
            available = ", ".join(str(item) for item in library.functionNames())
            raise RuntimeError(
                f"Metal function {name!r} is missing from the compiled library. "
                f"Available functions: {available}"
            )
        pipeline, error = device.newComputePipelineStateWithFunction_error_(function, None)
        if pipeline is None or error:
            raise RuntimeError(f"Metal compute pipeline {name!r} failed: {error}")
        pipelines[name] = pipeline
    return pipelines


def allocate_shared(nbytes: int, label: str = "Metal data"):
    """Allocate unified memory that both the host and the GPU address directly."""
    buffer = metal_device().newBufferWithLength_options_(
        int(nbytes), metal_module().MTLResourceStorageModeShared
    )
    if buffer is None:
        raise MemoryError(f"Metal could not allocate {nbytes} bytes for {label}")
    return buffer


def upload_shared(values: np.ndarray, label: str):
    """Copy a small host array (a mask, table or index list) into a new shared Metal buffer.

    Metal rejects zero-length buffers, so an empty array still gets one byte.
    """
    values = np.ascontiguousarray(values)
    buffer = allocate_shared(max(1, values.nbytes), label)
    if values.nbytes:
        buffer_view(buffer, values.nbytes)[:] = memoryview(values).cast("B")
    return buffer


def buffer_view(buffer, nbytes: int | None = None) -> memoryview:
    """Writable host view of the first ``nbytes`` of a shared Metal buffer."""
    count = int(buffer.length()) if nbytes is None else int(nbytes)
    return memoryview(_contents(buffer).as_buffer(count))


def numpy_view(buffer, dtype, count: int) -> np.ndarray:
    """Writable NumPy view of the first ``count`` values of a shared Metal buffer, without a copy."""
    return np.frombuffer(_contents(buffer).as_buffer(buffer.length()), dtype=dtype, count=count)


def _contents(buffer):
    """The buffer's host memory as a PyObjC variable-length array.

    Until PyObjC loads the Metal framework's metadata, ``contents()`` returns
    the bare address as an int. A Torch tensor's buffer borrowed through
    ``tensor_buffer`` before anything imported Metal hits that (SSB on an MPS
    tensor run on its own did), so Metal is imported before every read.
    """
    metal_module()
    return buffer.contents()


def read_exact(fd: int, buffer: memoryview, offset: int, label: str) -> None:
    """Fill ``buffer`` from ``offset``, retrying the short reads a large pread returns."""
    position = 0
    while position < len(buffer):
        count = os.preadv(fd, [buffer[position:]], int(offset) + position)
        if count <= 0:
            raise ValueError(f"short read while loading {label}")
        position += count


def release_buffer(buffer) -> None:
    """Hand a +1-retained Metal buffer's memory back to the system.

    PyObjC does not release buffers created by ``newBufferWithLength_options_``
    when the Python wrapper is collected: ``del``, ``gc.collect()``, an
    autorelease pool and ``setPurgeableState_`` all leave the allocation in
    place, so an explicit ``release()`` is the only thing that frees it. Each
    buffer must reach this function exactly once; a second raw Objective-C
    ``release`` can crash.
    """
    if buffer is None:
        return
    try:
        buffer.release()
    except (AttributeError, ValueError):
        pass


def complete_command(command, operation: str) -> float:
    """Run one command buffer to completion and return its GPU time in ms."""
    command.commit()
    command.waitUntilCompleted()
    if command.error() is not None:
        raise RuntimeError(f"Metal {operation} failed: {command.error()}")
    return max(0.0, float(command.GPUEndTime()) - float(command.GPUStartTime())) * 1_000


def tensor_buffer(tensor):
    """Borrow the Metal buffer behind a Torch MPS tensor's storage, without a copy.

    Torch storage holds the MTLBuffer object itself (as in ATen's
    ``getMTLBufferStorage``), so a native kernel can read or write the tensor
    in place. The wrapper only borrows the buffer: never release it. Call
    ``torch.mps.synchronize()`` first, or the native queue can touch storage
    that pending Torch work still owns.
    """
    import objc

    return objc.objc_object(c_void_p=ctypes.c_void_p(tensor.untyped_storage().data_ptr()))


def copy_to_torch(buffer, shape: tuple[int, ...], dtype, label: str):
    """Copy the leading bytes of a shared Metal buffer into a new Torch MPS tensor.

    A tensor that viewed the buffer would outlive it once the owner released
    it, so callers that hand results to Torch get an independent copy.
    """
    import torch

    output = torch.empty(shape, dtype=getattr(torch, np.dtype(dtype).name), device="mps")
    if not output.numel():
        return output
    torch.mps.synchronize()
    command = metal_queue().commandBuffer()
    encoder = command.blitCommandEncoder()
    encoder.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size_(
        buffer, 0, tensor_buffer(output), 0, output.numel() * output.element_size()
    )
    encoder.endEncoding()
    complete_command(command, label)
    return output


class SharedArray(np.ndarray):
    """NumPy array over a shared Metal buffer that keeps the buffer handle as ``_mtl``.

    Decoded frames are a zero-copy view into a unified-memory buffer; kernels
    read the buffer through ``_mtl`` and the owning caller releases it exactly
    once. ``_owner`` keeps an object that owns borrowed storage alive (for
    example a Torch tensor viewed in place).

    ``__array_finalize__`` carries both references onto every slice and
    reshape. Without it a view kept the memory readable but lost the
    release/lifetime handle, so the buffer could be released out from under a
    live view.
    """

    _mtl = None
    _owner = None

    def __array_finalize__(self, obj):
        if isinstance(obj, SharedArray):
            self._mtl = obj._mtl
            self._owner = obj._owner


def shared_array(buffer, dtype, shape: tuple[int, ...]) -> SharedArray:
    """Wrap the first ``prod(shape)`` values of a shared Metal buffer as a ``SharedArray``."""
    count = int(np.prod(shape, dtype=np.int64))
    view = numpy_view(buffer, dtype, count).reshape(shape).view(SharedArray)
    view._mtl = buffer
    return view
