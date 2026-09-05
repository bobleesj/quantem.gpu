"""Compatibility imports for the canonical ``packed`` backend.

New internal callers use the representation-named module. These aliases
retain the same functions/classes without a second implementation.
"""

from .packed import (
    CudaCompactH5ColumnMetrics as CudaCompactH5ColumnMetrics,
    CudaCompactH5DetectorMetrics as CudaCompactH5DetectorMetrics,
    CudaCompactH5LoadMetrics as CudaCompactH5LoadMetrics,
    CudaCompactH5ResidentSource as CudaCompactH5ResidentSource,
    _CudaCompactShard as _CudaCompactShard,
    _cuda_kernels as _cuda_kernels,
    _header_words_per_pixel as _header_words_per_pixel,
    _load_compact_h5_cuda_v3 as _load_compact_h5_cuda_v3,
    _pread_exact as _pread_exact,
    _read_prepared_center_of_mass as _read_prepared_center_of_mass,
    _read_shard_envelope as _read_shard_envelope,
    _sha256_file as _sha256_file,
    _timed_chunk_sha256_file as _timed_chunk_sha256_file,
    _timed_sha256_file as _timed_sha256_file,
    _update_v3_maximum_widths as _update_v3_maximum_widths,
    _upload_mapped_array as _upload_mapped_array,
    _validate_widths as _validate_widths,
    load_compact_h5_cuda as load_compact_h5_cuda,
    warm_compact_h5_cuda_kernels as warm_compact_h5_cuda_kernels,
)
