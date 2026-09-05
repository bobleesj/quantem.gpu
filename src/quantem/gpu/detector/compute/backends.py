"""Compatibility imports; canonical implementation: ``quantem.gpu.detector.backends.dispatch``."""

from quantem.gpu.detector.backends.dispatch import (
    CudaKernelCompute as CudaKernelCompute,
    CudaPackedUInt4Compute as CudaPackedUInt4Compute,
    MetalRawBackend as MetalRawBackend,
    TorchBackend as TorchBackend,
    _CHUNK_BYTE_BUDGET as _CHUNK_BYTE_BUDGET,
    _CUDA_MASK_INDEX_CACHE_SIZE as _CUDA_MASK_INDEX_CACHE_SIZE,
    _SPARSE_MASK_CHUNK_BYTE_BUDGET as _SPARSE_MASK_CHUNK_BYTE_BUDGET,
    compute_backend as compute_backend,
)
