"""Compatibility imports; canonical implementation: ``quantem.gpu.detector.backends.mps.kernels``."""

from quantem.gpu.detector.backends.mps.kernels import (
    ChunkedFrames as ChunkedFrames,
    MetalVirtualImage as MetalVirtualImage,
    MultiChunkedFrames as MultiChunkedFrames,
    _CHUNKS_PER_CMDBUF as _CHUNKS_PER_CMDBUF,
    _COMMAND_BUFFER_BYTES as _COMMAND_BUFFER_BYTES,
    _DEFAULT_COMPACT_TARGET_BYTES as _DEFAULT_COMPACT_TARGET_BYTES,
    _MASKED_SUM_MSL as _MASKED_SUM_MSL,
    _RADIAL_INTERACTION_IDLE_DELAY as _RADIAL_INTERACTION_IDLE_DELAY,
    _bin_mask as _bin_mask,
    _chunk_groups as _chunk_groups,
    _chunk_nbytes as _chunk_nbytes,
    _column_gather_workers as _column_gather_workers,
    _format_seconds as _format_seconds,
    _roi_accumulator_dtype as _roi_accumulator_dtype,
    _torch_dtype as _torch_dtype,
    default_fast_bin as default_fast_bin,
)
