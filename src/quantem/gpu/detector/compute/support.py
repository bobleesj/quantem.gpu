"""Compatibility imports; canonical implementation: ``quantem.gpu.detector.backends.support``."""

from quantem.gpu.detector.backends.support import (
    VirtualImageBackend as VirtualImageBackend,
    VirtualImageKernelSupport as VirtualImageKernelSupport,
    _INTEGER_DTYPES as _INTEGER_DTYPES,
    _cuda_mask_paths as _cuda_mask_paths,
    _detector_mask_pixels as _detector_mask_pixels,
    _dtype_from_data as _dtype_from_data,
    _infer_backend as _infer_backend,
    _module_root as _module_root,
    _resident_gib as _resident_gib,
    _scan_det_shape as _scan_det_shape,
    _shape_from_data as _shape_from_data,
    _uint32_accum_safe as _uint32_accum_safe,
    virtual_image_kernel_support as virtual_image_kernel_support,
)
