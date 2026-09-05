"""Compatibility imports; canonical implementation: ``quantem.gpu.parallax.backends.cuda.backend``."""

from quantem.gpu.parallax.backends.cuda.backend import (
    CudaParallaxBackend as CudaParallaxBackend,
    _validate_scan_shape as _validate_scan_shape,
    circular_mask as circular_mask,
    run_cuda as run_cuda,
)
