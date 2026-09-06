"""UI-free resident capability and publication contracts for Apple backends.

The module does not choose files, cache names, device admission, cancellation
policy, or presentation. It describes an already-published resident result and
records generation-safe milestones supplied by a consumer.
"""

from __future__ import annotations

import math
import threading
import time
from collections.abc import Iterable
from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np

from ...resident_contract import (
    ResidentGenerationReceipt,
    metadata_sha256,
)
from ...representation import DataRepresentation

__all__ = [
    "MPSProductAvailability",
    "MPSProductNumerics",
    "MPSPublicationCounters",
    "MPSPublicationEvent",
    "MPSPublicationMilestone",
    "MPSPublicationRecorder",
    "MPSResidentCapabilities",
    "MPSResidentProduct",
    "MPSResidentProductCapability",
    "MPSTimingBoundary",
    "MPSTimingSummary",
    "describe_resident",
]


class MPSResidentProduct(str, Enum):
    """Product roles required by a complete interactive 4D-STEM consumer."""

    DIFFRACTION_PATTERN = "diffraction-pattern"
    BRIGHT_FIELD = "bright-field"
    ANNULAR_BRIGHT_FIELD = "annular-bright-field"
    ANNULAR_DARK_FIELD = "annular-dark-field"
    MEAN_DIFFRACTION_PATTERN = "mean-diffraction-pattern"
    TOTAL = "total"
    CENTER_OF_MASS = "center-of-mass"
    DPC = "dpc"
    IDPC = "idpc"
    FFT = "fft"


class MPSProductAvailability(str, Enum):
    """When a product can be consumed after resident-ready publication."""

    IMMEDIATE = "immediate"
    RESIDENT_ON_DEMAND = "resident-on-demand"
    UNAVAILABLE = "unavailable"


class MPSProductNumerics(str, Enum):
    """Numerical boundary advertised for one resident product."""

    EXACT_INTEGER = "exact-integer"
    EXACT_INTEGER_THEN_FLOAT32 = "exact-integer-then-float32"
    FROZEN_FLOAT32 = "frozen-float32"


@dataclass(frozen=True)
class MPSResidentProductCapability:
    """Availability and numerics for one product role."""

    product: MPSResidentProduct
    availability: MPSProductAvailability
    numerics: MPSProductNumerics

    def to_dict(self) -> dict[str, str]:
        """Return the stable cross-language JSON spelling."""

        return {
            "product": self.product.value,
            "availability": self.availability.value,
            "numerics": self.numerics.value,
        }


@dataclass(frozen=True)
class MPSResidentCapabilities:
    """Capability receipt for one fully published resident generation."""

    representation: DataRepresentation
    source_identity_sha256: str
    scan_shape: tuple[int, int]
    detector_shape: tuple[int, int]
    storage_schema: str
    working_dtype: str
    exact_integer_bits: int
    logical_tensor_bytes: int
    complete_source_resident: bool
    resident_bytes: int
    lossless: bool
    products: tuple[MPSResidentProductCapability, ...]
    resident_receipt: ResidentGenerationReceipt

    SCHEMA = "quantem.gpu.apple-4dstem-resident-capabilities/v4"

    @property
    def full_interactive_resident(self) -> bool:
        """Whether every required role has an immediate or on-demand seam."""

        try:
            self.resident_receipt.validate()
        except (TypeError, ValueError):
            return False

        return (
            self.complete_source_resident
            and self.lossless
            and self.logical_tensor_bytes > 0
            and self.resident_bytes > 0
            and self.resident_receipt.lossless_exact
            and self.resident_receipt.source_identity_sha256
            == self.source_identity_sha256
            and self.resident_receipt.working_shape
            == (*self.scan_shape, *self.detector_shape)
            and self.resident_receipt.working_dtype == self.working_dtype
            and self.resident_receipt.working_logical_tensor_bytes
            == self.logical_tensor_bytes
            and self.resident_receipt.physical_resident_bytes
            == self.resident_bytes
            and self.resident_receipt.storage_schema == self.storage_schema
            and {item.product for item in self.products} == set(MPSResidentProduct)
            and all(
                item.availability is not MPSProductAvailability.UNAVAILABLE
                for item in self.products
            )
        )

    def to_dict(self) -> dict[str, Any]:
        """Return the stable cross-language JSON spelling."""

        return {
            "schema": self.SCHEMA,
            "representation": self.representation.value,
            "sourceIdentitySHA256": self.source_identity_sha256,
            "scanRows": self.scan_shape[0],
            "scanColumns": self.scan_shape[1],
            "detectorRows": self.detector_shape[0],
            "detectorColumns": self.detector_shape[1],
            "storageSchema": self.storage_schema,
            "workingDtype": self.working_dtype,
            "exactIntegerBits": self.exact_integer_bits,
            "logicalTensorBytes": self.logical_tensor_bytes,
            "completeSourceResident": self.complete_source_resident,
            "residentBytes": self.resident_bytes,
            "residentStorageBytes": self.resident_bytes,
            "lossless": self.lossless,
            "residentReceipt": self.resident_receipt.to_camel_case_dict(),
            "products": [item.to_dict() for item in self.products],
        }


def _capability(
    product: MPSResidentProduct,
    availability: MPSProductAvailability,
    numerics: MPSProductNumerics,
) -> MPSResidentProductCapability:
    return MPSResidentProductCapability(product, availability, numerics)


def _source_identity(metadata: dict[str, Any]) -> str:
    for key in (
        "source_identity_sha256",
        "sourceIdentitySHA256",
        "source_sha256",
    ):
        value = metadata.get(key)
        if isinstance(value, str) and _is_sha256(value):
            return value
    return ""


def _calibration_identity(
    value: object,
) -> tuple[str | None, str | None]:
    if value is None:
        return None, None
    if not isinstance(value, dict):
        raise TypeError("Detector calibration metadata must be an object or null.")
    schema = value.get("schema")
    if not isinstance(schema, str) or not schema.strip():
        raise ValueError("Detector calibration metadata requires a versioned schema.")
    return schema, metadata_sha256(value)


def _chunked_capabilities(source: Any) -> MPSResidentCapabilities:
    dtype = np.dtype(source.dtype)
    if dtype.kind != "u" or dtype.itemsize not in (1, 2, 4):
        raise TypeError(
            "An exact MPS resident source requires uint8, uint16, or uint32 chunks; "
            f"got {dtype}."
        )
    scan_shape = tuple(int(value) for value in (source.scan_shape or ()))
    if len(scan_shape) != 2 or math.prod(scan_shape) != int(source.n_frames):
        raise ValueError(
            "An exact MPS resident source requires a complete (row, column) scan shape."
        )
    metadata = dict(source.metadata or {})
    source_identity = _source_identity(metadata)
    if not source_identity:
        raise ValueError(
            "An exact MPS resident capability receipt requires a source SHA-256 identity."
        )
    raw_detector_shape = tuple(
        int(value)
        for value in metadata.get("raw_detector_shape", source.detector_shape)
    )
    source_dtype = np.dtype(metadata.get("source_dtype", dtype.name)).name
    scan_bin = int(metadata.get("scan_bin", 1))
    detector_bin = int(metadata.get("det_bin", 1))
    source_shape = (
        scan_shape[0] * scan_bin,
        scan_shape[1] * scan_bin,
        raw_detector_shape[0],
        raw_detector_shape[1],
    )
    working_shape = (*scan_shape, *tuple(int(value) for value in source.detector_shape))
    calibration_schema, calibration_sha256 = _calibration_identity(
        metadata.get("detector_calibration")
    )
    detector_mask_count = int(metadata.get("detector_mask_count", 0))
    detector_mask_sha256 = metadata.get("detector_mask_sha256")
    receipt = ResidentGenerationReceipt(
        representation=DataRepresentation.DENSE,
        source_identity_sha256=source_identity,
        source_shape=source_shape,
        working_shape=working_shape,
        source_dtype=source_dtype,
        working_dtype=dtype.name,
        source_logical_tensor_bytes=math.prod(source_shape)
        * np.dtype(source_dtype).itemsize,
        working_logical_tensor_bytes=int(source.nbytes),
        physical_resident_bytes=int(source.nbytes),
        storage_schema="quantem.gpu.indexed-resident-integer/v1",
        scan_bin=scan_bin,
        detector_bin=detector_bin,
        crop=None,
        detector_mask_count=detector_mask_count,
        detector_mask_sha256=(
            str(detector_mask_sha256) if detector_mask_sha256 is not None else None
        ),
        detector_mask_schema=(
            "quantem.gpu.detector-mask-identity/opaque-v1"
            if detector_mask_sha256 is not None
            else None
        ),
        calibration_schema=calibration_schema,
        calibration_sha256=calibration_sha256,
        provenance_schema="quantem.gpu.mps-chunked-metadata/v1",
        provenance_sha256=metadata_sha256(metadata),
        source_raw_logical_sha256=metadata.get("source_raw_logical_sha256"),
        working_logical_sha256=metadata.get("working_logical_sha256"),
    )
    receipt.validate()
    products = (
        _capability(
            MPSResidentProduct.DIFFRACTION_PATTERN,
            MPSProductAvailability.RESIDENT_ON_DEMAND,
            MPSProductNumerics.EXACT_INTEGER,
        ),
        _capability(
            MPSResidentProduct.BRIGHT_FIELD,
            MPSProductAvailability.RESIDENT_ON_DEMAND,
            MPSProductNumerics.EXACT_INTEGER,
        ),
        _capability(
            MPSResidentProduct.ANNULAR_BRIGHT_FIELD,
            MPSProductAvailability.RESIDENT_ON_DEMAND,
            MPSProductNumerics.EXACT_INTEGER,
        ),
        _capability(
            MPSResidentProduct.ANNULAR_DARK_FIELD,
            MPSProductAvailability.RESIDENT_ON_DEMAND,
            MPSProductNumerics.EXACT_INTEGER,
        ),
        _capability(
            MPSResidentProduct.MEAN_DIFFRACTION_PATTERN,
            MPSProductAvailability.RESIDENT_ON_DEMAND,
            MPSProductNumerics.EXACT_INTEGER_THEN_FLOAT32,
        ),
        _capability(
            MPSResidentProduct.TOTAL,
            MPSProductAvailability.RESIDENT_ON_DEMAND,
            MPSProductNumerics.EXACT_INTEGER,
        ),
        _capability(
            MPSResidentProduct.CENTER_OF_MASS,
            MPSProductAvailability.RESIDENT_ON_DEMAND,
            MPSProductNumerics.EXACT_INTEGER_THEN_FLOAT32,
        ),
        _capability(
            MPSResidentProduct.DPC,
            MPSProductAvailability.RESIDENT_ON_DEMAND,
            MPSProductNumerics.EXACT_INTEGER_THEN_FLOAT32,
        ),
        _capability(
            MPSResidentProduct.IDPC,
            MPSProductAvailability.RESIDENT_ON_DEMAND,
            MPSProductNumerics.FROZEN_FLOAT32,
        ),
        _capability(
            MPSResidentProduct.FFT,
            MPSProductAvailability.RESIDENT_ON_DEMAND,
            MPSProductNumerics.FROZEN_FLOAT32,
        ),
    )
    return MPSResidentCapabilities(
        representation=DataRepresentation.DENSE,
        source_identity_sha256=source_identity,
        scan_shape=scan_shape,
        detector_shape=tuple(int(value) for value in source.detector_shape),
        storage_schema="quantem.gpu.indexed-resident-integer/v1",
        working_dtype=dtype.name,
        exact_integer_bits=dtype.itemsize * 8,
        logical_tensor_bytes=int(source.nbytes),
        complete_source_resident=True,
        resident_bytes=int(source.nbytes),
        lossless=True,
        products=products,
        resident_receipt=receipt,
    )


def _compact_capabilities(source: Any) -> MPSResidentCapabilities:
    index = source.index
    if (
        not isinstance(getattr(index, "manifest", None), dict)
        or index.manifest.get("schema") != "quantem.gpu.packed-detector-h5/v3"
    ):
        raise ValueError(
            "Compact resident capabilities require exact QGIX v3 uint8 semantics."
        )
    if getattr(index, "raw_access_mode", None) not in {
        "exact_no_exclusions",
        "exact_exclusion_constants",
    }:
        raise ValueError(
            "Compact resident capabilities require lossless access to every source count."
        )
    prepared_names = {
        product.name
        for product in (
            index.prepared_detector_products.products
            if index.prepared_detector_products
            else ()
        )
    }
    has_dpc = index.prepared_dpc_moments is not None
    manifest = index.manifest
    source_dtype = str(manifest.get("source_dtype", ""))
    working_dtype = str(manifest.get("working_dtype", ""))
    calibration_schema, calibration_sha256 = _calibration_identity(
        manifest.get("detector_calibration")
    )
    source_shape = tuple(int(value) for value in index.shape)
    receipt = ResidentGenerationReceipt(
        representation=DataRepresentation.PACKED,
        source_identity_sha256=index.source_identity_sha256,
        source_shape=source_shape,
        working_shape=source_shape,
        source_dtype=source_dtype,
        working_dtype=working_dtype,
        source_logical_tensor_bytes=math.prod(source_shape)
        * np.dtype(source_dtype).itemsize,
        working_logical_tensor_bytes=math.prod(source_shape)
        * np.dtype(working_dtype).itemsize,
        physical_resident_bytes=int(source.load_metrics.total_resident_bytes),
        container_bytes=int(index.file_bytes),
        storage_schema=str(manifest["schema"]),
        scan_bin=int(manifest["scan_bin"]),
        detector_bin=int(manifest["detector_bin"]),
        crop=None,
        detector_mask_count=len(index.excluded_detector_pixels),
        detector_mask_sha256=manifest.get("detector_mask_sha256"),
        detector_mask_schema="quantem.gpu.detector-mask-identity/opaque-v1",
        calibration_schema=calibration_schema,
        calibration_sha256=calibration_sha256,
        provenance_schema="quantem.gpu.packed-detector-h5-manifest/v1",
        provenance_sha256=metadata_sha256(manifest),
        source_raw_logical_sha256=manifest.get("source_raw_logical_sha256"),
        working_logical_sha256=manifest.get("prepared_uint8_sha256"),
    )
    receipt.validate()
    prepared = MPSProductAvailability.IMMEDIATE
    on_demand = MPSProductAvailability.RESIDENT_ON_DEMAND
    unavailable = MPSProductAvailability.UNAVAILABLE
    products = (
        _capability(
            MPSResidentProduct.DIFFRACTION_PATTERN,
            on_demand,
            MPSProductNumerics.EXACT_INTEGER,
        ),
        _capability(
            MPSResidentProduct.BRIGHT_FIELD,
            prepared if "bf" in prepared_names else on_demand,
            MPSProductNumerics.EXACT_INTEGER,
        ),
        _capability(
            MPSResidentProduct.ANNULAR_BRIGHT_FIELD,
            prepared if "abf" in prepared_names else on_demand,
            MPSProductNumerics.EXACT_INTEGER,
        ),
        _capability(
            MPSResidentProduct.ANNULAR_DARK_FIELD,
            prepared if "adf" in prepared_names else on_demand,
            MPSProductNumerics.EXACT_INTEGER,
        ),
        _capability(
            MPSResidentProduct.MEAN_DIFFRACTION_PATTERN,
            on_demand,
            MPSProductNumerics.EXACT_INTEGER_THEN_FLOAT32,
        ),
        _capability(
            MPSResidentProduct.TOTAL,
            prepared if has_dpc else unavailable,
            MPSProductNumerics.EXACT_INTEGER,
        ),
        _capability(
            MPSResidentProduct.CENTER_OF_MASS,
            prepared if has_dpc else unavailable,
            MPSProductNumerics.EXACT_INTEGER_THEN_FLOAT32,
        ),
        _capability(
            MPSResidentProduct.DPC,
            prepared if has_dpc else unavailable,
            MPSProductNumerics.EXACT_INTEGER_THEN_FLOAT32,
        ),
        _capability(
            MPSResidentProduct.IDPC,
            on_demand if has_dpc else unavailable,
            MPSProductNumerics.FROZEN_FLOAT32,
        ),
        _capability(
            MPSResidentProduct.FFT,
            on_demand if has_dpc else unavailable,
            MPSProductNumerics.FROZEN_FLOAT32,
        ),
    )
    return MPSResidentCapabilities(
        representation=DataRepresentation.PACKED,
        source_identity_sha256=index.source_identity_sha256,
        scan_shape=tuple(int(value) for value in index.shape[:2]),
        detector_shape=tuple(int(value) for value in index.shape[2:]),
        storage_schema="quantem.gpu.packed-detector-h5/v3",
        working_dtype=working_dtype,
        exact_integer_bits=8,
        logical_tensor_bytes=math.prod(int(value) for value in index.shape),
        complete_source_resident=not bool(source.is_released),
        resident_bytes=int(source.load_metrics.total_resident_bytes),
        lossless=True,
        products=products,
        resident_receipt=receipt,
    )


def describe_resident(source: Any) -> MPSResidentCapabilities:
    """Describe a complete MPS chunked or compact-v3 resident source.

    The function intentionally does not accept an arbitrary NumPy array. A
    capability receipt is reserved for sources that retain explicit Metal
    residency and lifecycle ownership.
    """

    if hasattr(source, "index") and hasattr(source, "load_metrics"):
        return _compact_capabilities(source)
    if all(
        hasattr(source, name)
        for name in ("chunks", "metadata", "scan_shape", "detector_shape", "dtype")
    ):
        return _chunked_capabilities(source)
    raise TypeError(
        "Expected an MPSChunked4DSTEM or MPSCompactV3Resident source with explicit "
        "resident-lifecycle ownership."
    )


class MPSPublicationMilestone(str, Enum):
    """Stable milestones for latest-wins publication telemetry."""

    REQUESTED = "requested"
    SOURCE_ADMITTED = "source-admitted"
    RESIDENT_READY = "resident-ready"
    FIRST_RESIDENT_PRESENT = "first-resident-present"
    SUPERSEDED_REJECTED = "superseded-rejected"
    CANCELLED = "cancelled"
    FAILED = "failed"
    DEVICE_LOST = "device-lost"
    RECOVERY_READY = "recovery-ready"


class MPSTimingBoundary(str, Enum):
    """Non-interchangeable load, reopen, switching, and presentation timings."""

    COLD_ARBITRARY_TO_RESIDENT_READY = "cold-arbitrary-to-resident-ready"
    COLD_ARBITRARY_TO_FIRST_PRESENT = "cold-arbitrary-to-first-resident-present"
    PREPARED_CREATION = "prepared-creation"
    WARM_REOPEN_TO_RESIDENT_READY = "warm-reopen-to-resident-ready"
    WARM_REOPEN_TO_FIRST_PRESENT = "warm-reopen-to-first-resident-present"
    PREPARED_REOPEN_TO_RESIDENT_READY = "prepared-reopen-to-resident-ready"
    PREPARED_REOPEN_TO_FIRST_PRESENT = "prepared-reopen-to-first-resident-present"
    CACHE_REOPEN_TO_RESIDENT_READY = "cache-reopen-to-resident-ready"
    CACHE_REOPEN_TO_FIRST_PRESENT = "cache-reopen-to-first-resident-present"
    EXACT_SWITCH_TO_RESIDENT_READY = "exact-switch-to-resident-ready"
    EXACT_SWITCH_TO_FIRST_PRESENT = "exact-switch-to-first-resident-present"


@dataclass(frozen=True)
class MPSPublicationCounters:
    """Counter snapshot attached to a publication event."""

    source_bytes: int | None = None
    resident_bytes: int | None = None
    process_rss_bytes: int | None = None
    peak_process_rss_bytes: int | None = None
    compressed_memory_bytes: int | None = None
    swap_bytes: int | None = None
    device_allocated_bytes: int | None = None
    peak_device_allocated_bytes: int | None = None
    storage_read_bytes: int | None = None
    upload_bytes: int | None = None
    readback_bytes: int | None = None
    synchronization_count: int | None = None

    def to_dict(self) -> dict[str, int | None]:
        """Return the stable cross-language JSON spelling."""

        values = {
            "sourceBytes": self.source_bytes,
            "residentBytes": self.resident_bytes,
            "processRSSBytes": self.process_rss_bytes,
            "peakProcessRSSBytes": self.peak_process_rss_bytes,
            "compressedMemoryBytes": self.compressed_memory_bytes,
            "swapBytes": self.swap_bytes,
            "deviceAllocatedBytes": self.device_allocated_bytes,
            "peakDeviceAllocatedBytes": self.peak_device_allocated_bytes,
            "storageReadBytes": self.storage_read_bytes,
            "uploadBytes": self.upload_bytes,
            "readbackBytes": self.readback_bytes,
            "synchronizationCount": self.synchronization_count,
        }
        return {key: value for key, value in values.items() if value is not None}


@dataclass(frozen=True)
class MPSPublicationEvent:
    """One immutable generation-bound telemetry event."""

    generation: int
    source_identity_sha256: str
    representation: DataRepresentation
    milestone: MPSPublicationMilestone
    monotonic_nanoseconds: int
    counters: MPSPublicationCounters
    detail: str | None = None

    SCHEMA = "quantem.gpu.apple-4dstem-publication/v2"

    def to_dict(self) -> dict[str, Any]:
        """Return the stable cross-language JSON spelling."""

        values = {
            "schema": self.SCHEMA,
            "generation": self.generation,
            "sourceIdentitySHA256": self.source_identity_sha256,
            "representation": self.representation.value,
            "milestone": self.milestone.value,
            "monotonicNanoseconds": self.monotonic_nanoseconds,
            "counters": self.counters.to_dict(),
            "detail": self.detail,
        }
        return {key: value for key, value in values.items() if value is not None}


class MPSPublicationRecorder:
    """Thread-safe latest-wins recorder for resident and presentation events."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._latest_generation: int | None = None
        self._source_identity_sha256 = ""
        self._representation = DataRepresentation.DENSE
        self._state = "empty"
        self._events: list[MPSPublicationEvent] = []
        self._source_by_generation: dict[
            int, tuple[str, DataRepresentation]
        ] = {}

    def begin(
        self,
        generation: int,
        source_identity_sha256: str,
        representation: DataRepresentation,
        counters: MPSPublicationCounters | None = None,
    ) -> bool:
        """Begin a strictly newer A-B-A-safe generation."""

        generation = int(generation)
        if generation < 0:
            raise ValueError("publication generation must be nonnegative")
        if not _is_sha256(source_identity_sha256):
            raise ValueError("publication requires a lowercase source SHA-256 identity")
        with self._lock:
            counters = counters or MPSPublicationCounters()
            if (
                self._latest_generation is not None
                and generation <= self._latest_generation
            ):
                self._append(
                    generation,
                    source_identity_sha256,
                    representation,
                    MPSPublicationMilestone.SUPERSEDED_REJECTED,
                    counters,
                    "generation is not newer than the active request",
                )
                return False
            self._latest_generation = generation
            self._source_identity_sha256 = source_identity_sha256
            self._representation = representation
            self._state = "active"
            self._source_by_generation[generation] = (
                source_identity_sha256,
                representation,
            )
            self._append(
                generation,
                source_identity_sha256,
                representation,
                MPSPublicationMilestone.REQUESTED,
                counters,
                None,
            )
            return True

    def record(
        self,
        generation: int,
        milestone: MPSPublicationMilestone,
        counters: MPSPublicationCounters | None = None,
        detail: str | None = None,
    ) -> bool:
        """Record a milestone only when the generation remains current."""

        generation = int(generation)
        with self._lock:
            counters = counters or MPSPublicationCounters()
            if generation != self._latest_generation:
                if self._latest_generation is not None:
                    rejected_source, rejected_representation = (
                        self._source_by_generation.get(
                            generation,
                            (self._source_identity_sha256, self._representation),
                        )
                    )
                    self._append(
                        generation,
                        rejected_source,
                        rejected_representation,
                        MPSPublicationMilestone.SUPERSEDED_REJECTED,
                        counters,
                        detail or "generation was superseded before publication",
                    )
                return False
            if milestone in {
                MPSPublicationMilestone.REQUESTED,
                MPSPublicationMilestone.SUPERSEDED_REJECTED,
            }:
                raise ValueError(
                    "use begin for requested generations; stale rejection is recorder-owned"
                )
            allowed = False
            if milestone is MPSPublicationMilestone.SOURCE_ADMITTED:
                allowed = self._state == "active"
            elif milestone is MPSPublicationMilestone.RESIDENT_READY:
                allowed = self._state == "active"
                if allowed:
                    self._state = "ready"
            elif milestone is MPSPublicationMilestone.FIRST_RESIDENT_PRESENT:
                allowed = self._state in {"ready", "presented"}
                if allowed:
                    self._state = "presented"
            elif milestone is MPSPublicationMilestone.DEVICE_LOST:
                allowed = self._state != "terminal"
                if allowed:
                    self._state = "lost"
            elif milestone is MPSPublicationMilestone.RECOVERY_READY:
                allowed = self._state == "lost"
                if allowed:
                    self._state = "ready"
            elif milestone in {
                MPSPublicationMilestone.CANCELLED,
                MPSPublicationMilestone.FAILED,
            }:
                allowed = self._state != "terminal"
                if allowed:
                    self._state = "terminal"
            if not allowed:
                raise ValueError(
                    f"publication milestone {milestone.value} is invalid for "
                    "the current generation state"
                )
            self._append(
                generation,
                self._source_identity_sha256,
                self._representation,
                milestone,
                counters,
                detail,
            )
            return True

    def events(self) -> tuple[MPSPublicationEvent, ...]:
        """Return an immutable snapshot in monotonic record order."""

        with self._lock:
            return tuple(self._events)

    def _append(
        self,
        generation: int,
        source_identity_sha256: str,
        representation: DataRepresentation,
        milestone: MPSPublicationMilestone,
        counters: MPSPublicationCounters,
        detail: str | None,
    ) -> None:
        self._events.append(
            MPSPublicationEvent(
                generation=generation,
                source_identity_sha256=source_identity_sha256,
                representation=representation,
                milestone=milestone,
                monotonic_nanoseconds=time.monotonic_ns(),
                counters=counters,
                detail=detail,
            )
        )


@dataclass(frozen=True)
class MPSTimingSummary:
    """Nearest-rank summary for one explicitly named timing boundary."""

    sample_count: int
    p50_seconds: float
    p95_seconds: float
    maximum_seconds: float

    @classmethod
    def from_samples(cls, samples_seconds: Iterable[float]) -> MPSTimingSummary:
        """Summarize finite nonnegative samples without changing their boundary."""

        samples = sorted(float(value) for value in samples_seconds)
        if not samples or any(
            not math.isfinite(value) or value < 0 for value in samples
        ):
            raise ValueError(
                "a timing summary requires one or more finite nonnegative samples"
            )

        def percentile(probability: float) -> float:
            rank = max(1, math.ceil(probability * len(samples)))
            return samples[min(rank - 1, len(samples) - 1)]

        return cls(
            sample_count=len(samples),
            p50_seconds=percentile(0.50),
            p95_seconds=percentile(0.95),
            maximum_seconds=samples[-1],
        )


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(
        character in "0123456789abcdef" for character in value
    )
