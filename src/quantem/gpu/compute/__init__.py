"""compute kernels: masked-sum / prefix / bin / reduce (BF/DF/DPC callers).

Pick via ``quantem.gpu.compute_backend()``. CHURNS — new virtual
detectors / derived properties land here. The MPS reductions live in
``compute.mps`` (MetalVirtualImage / ChunkedFrames); CUDA resident virtual-image
dragging lives in ``compute.cuda``.
"""
from __future__ import annotations

from .backends import (
    CudaKernelCompute,
    MetalCompute,
    MetalRawBackend,
    TorchBackend,
    TorchCompute,
    compute_backend,
)
from .virtual_image_support import (
    VirtualImageKernelSupport,
    virtual_image_kernel_support,
)

__all__ = [
    "CudaKernelCompute",
    "MetalCompute",
    "MetalRawBackend",
    "TorchBackend",
    "TorchCompute",
    "VirtualImageKernelSupport",
    "compute_backend",
    "virtual_image_kernel_support",
]
