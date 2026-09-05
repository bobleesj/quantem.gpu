"""Compatibility imports for the canonical ``packed`` backend.

New internal callers use the representation-named module. These aliases
retain the same functions/classes without a second implementation.
"""

from .packed import (
    MPSCompactV3Cancelled as MPSCompactV3Cancelled,
    MPSCompactV3DetectorMetrics as MPSCompactV3DetectorMetrics,
    MPSCompactV3Error as MPSCompactV3Error,
    MPSCompactV3ExactDPCMoments as MPSCompactV3ExactDPCMoments,
    MPSCompactV3Index as MPSCompactV3Index,
    MPSCompactV3LoadMetrics as MPSCompactV3LoadMetrics,
    MPSCompactV3LogicalHashMetrics as MPSCompactV3LogicalHashMetrics,
    MPSCompactV3MeanDiffraction as MPSCompactV3MeanDiffraction,
    MPSCompactV3PreparedDPCMoments as MPSCompactV3PreparedDPCMoments,
    MPSCompactV3PreparedDetectorProduct as MPSCompactV3PreparedDetectorProduct,
    MPSCompactV3PreparedDetectorProducts as MPSCompactV3PreparedDetectorProducts,
    MPSCompactV3Resident as MPSCompactV3Resident,
    MPSCompactV3Shard as MPSCompactV3Shard,
    _MPSPreparedDetectorResident as _MPSPreparedDetectorResident,
    __all__ as __all__,
    _allocate_shared as _allocate_shared,
    _buffer_view as _buffer_view,
    _calibration_digest as _calibration_digest,
    _complete as _complete,
    _load_prepared_detector_products as _load_prepared_detector_products,
    _load_prepared_dpc as _load_prepared_dpc,
    _make_pipelines as _make_pipelines,
    _metal_module as _metal_module,
    _parallel_authenticate as _parallel_authenticate,
    _parse_prepared_detector_products as _parse_prepared_detector_products,
    _parse_prepared_dpc_moments as _parse_prepared_dpc_moments,
    _pread_exact as _pread_exact,
    _prepared_dpc_display_arrays as _prepared_dpc_display_arrays,
    _release as _release,
    _require_sha256 as _require_sha256,
    load_compact_v3_mps as load_compact_v3_mps,
    read_compact_v3_index as read_compact_v3_index,
)
