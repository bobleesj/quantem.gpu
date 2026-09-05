"""Pure contract tests for Apple resident capabilities and publication."""

from __future__ import annotations

from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

from quantem.gpu.io.backends.mps.consumer import (
    MPSProductAvailability,
    MPSPublicationCounters,
    MPSPublicationMilestone,
    MPSPublicationRecorder,
    MPSResidentProduct,
    MPSResidentRepresentation,
    MPSTimingBoundary,
    MPSTimingSummary,
    describe_resident,
)

SOURCE_A = "a" * 64
SOURCE_B = "b" * 64


class _ChunkedResident:
    def __init__(self, dtype=np.uint16) -> None:
        self.chunks = [np.zeros((2, 3, 4), dtype=dtype)]
        self.metadata = {"source_identity_sha256": SOURCE_A}
        self.scan_shape = (1, 2)

    @property
    def detector_shape(self) -> tuple[int, int]:
        return (3, 4)

    @property
    def dtype(self) -> np.dtype:
        return self.chunks[0].dtype

    @property
    def n_frames(self) -> int:
        return 2

    @property
    def nbytes(self) -> int:
        return self.chunks[0].nbytes


def _compact_resident(*, released: bool = False):
    products = SimpleNamespace(
        products=tuple(SimpleNamespace(name=name) for name in ("bf", "abf", "adf"))
    )
    index = SimpleNamespace(
        shape=(2, 2, 3, 4),
        file_bytes=600,
        source_identity_sha256=SOURCE_B,
        excluded_detector_pixels=(1,),
        raw_access_mode="exact_exclusion_constants",
        manifest={
            "schema": "quantem.gpu.packed-detector-h5/v3",
            "source_dtype": "uint16",
            "working_dtype": "uint8",
            "scan_bin": 1,
            "detector_bin": 1,
            "crop": None,
            "detector_mask_sha256": "c" * 64,
            "source_raw_logical_sha256": "d" * 64,
            "prepared_uint8_sha256": "e" * 64,
        },
        prepared_detector_products=products,
        prepared_dpc_moments=object(),
    )
    return SimpleNamespace(
        index=index,
        load_metrics=SimpleNamespace(total_resident_bytes=1234),
        is_released=released,
    )


def test_chunked_capabilities_preserve_dynamic_exact_integer_width() -> None:
    capabilities = describe_resident(_ChunkedResident(np.uint16))

    assert (
        capabilities.representation
        is MPSResidentRepresentation.DENSE
    )
    assert capabilities.working_dtype == "uint16"
    assert capabilities.exact_integer_bits == 16
    assert capabilities.scan_shape == (1, 2)
    assert capabilities.detector_shape == (3, 4)
    assert capabilities.storage_schema == "quantem.gpu.indexed-resident-integer/v1"
    assert capabilities.logical_tensor_bytes == 48
    assert capabilities.resident_bytes == 48
    assert capabilities.lossless
    assert capabilities.full_interactive_resident
    assert not replace(capabilities, lossless=False).full_interactive_resident
    assert capabilities.to_dict()["sourceIdentitySHA256"] == SOURCE_A


def test_compact_v3_receipt_does_not_invent_uint16() -> None:
    capabilities = describe_resident(_compact_resident())
    by_product = {item.product: item for item in capabilities.products}

    assert (
        capabilities.representation is MPSResidentRepresentation.LOSSLESS_PACKED
    )
    assert capabilities.working_dtype == "uint8"
    assert capabilities.exact_integer_bits == 8
    assert capabilities.storage_schema == "quantem.gpu.packed-detector-h5/v3"
    assert capabilities.logical_tensor_bytes == 48
    assert capabilities.to_dict()["residentStorageBytes"] == 1234
    receipt = capabilities.resident_receipt
    assert receipt.source_logical_tensor_bytes == 96
    assert receipt.working_logical_tensor_bytes == 48
    assert receipt.physical_resident_bytes == 1234
    assert receipt.detector_mask_count == 1
    assert capabilities.to_dict()["residentReceipt"]["sourceDtype"] == "uint16"
    assert capabilities.lossless
    assert capabilities.full_interactive_resident
    assert not replace(
        capabilities,
        resident_receipt=replace(
            receipt,
            source_identity_sha256=SOURCE_A,
        ),
    ).full_interactive_resident
    assert (
        by_product[MPSResidentProduct.MEAN_DIFFRACTION_PATTERN].availability
        is MPSProductAvailability.RESIDENT_ON_DEMAND
    )
    assert (
        by_product[MPSResidentProduct.BRIGHT_FIELD].availability
        is MPSProductAvailability.IMMEDIATE
    )


def test_released_compact_source_is_not_complete_resident() -> None:
    capabilities = describe_resident(_compact_resident(released=True))

    assert not capabilities.complete_source_resident
    assert not capabilities.full_interactive_resident


def test_compact_receipt_rejects_non_v3_schema() -> None:
    source = _compact_resident()
    source.index.manifest["schema"] = "quantem.gpu.packed-detector-h5/v1"

    with pytest.raises(ValueError, match="QGIX v3 uint8"):
        describe_resident(source)


def test_publication_recorder_rejects_stale_a_b_a_generations() -> None:
    recorder = MPSPublicationRecorder()
    counters = MPSPublicationCounters(
        source_bytes=10,
        resident_bytes=20,
        peak_process_rss_bytes=30,
        peak_device_allocated_bytes=40,
    )

    assert recorder.begin(
        1,
        SOURCE_A,
        MPSResidentRepresentation.DENSE,
        counters,
    )
    assert recorder.record(1, MPSPublicationMilestone.SOURCE_ADMITTED)
    assert recorder.record(1, MPSPublicationMilestone.RESIDENT_READY)
    assert recorder.begin(
        2,
        SOURCE_B,
        MPSResidentRepresentation.LOSSLESS_PACKED,
    )
    assert not recorder.record(1, MPSPublicationMilestone.FIRST_RESIDENT_PRESENT)
    assert recorder.begin(
        3,
        SOURCE_A,
        MPSResidentRepresentation.DENSE,
    )
    assert recorder.record(3, MPSPublicationMilestone.RESIDENT_READY)
    assert recorder.record(3, MPSPublicationMilestone.FIRST_RESIDENT_PRESENT)

    events = recorder.events()
    assert [event.milestone for event in events] == [
        MPSPublicationMilestone.REQUESTED,
        MPSPublicationMilestone.SOURCE_ADMITTED,
        MPSPublicationMilestone.RESIDENT_READY,
        MPSPublicationMilestone.REQUESTED,
        MPSPublicationMilestone.SUPERSEDED_REJECTED,
        MPSPublicationMilestone.REQUESTED,
        MPSPublicationMilestone.RESIDENT_READY,
        MPSPublicationMilestone.FIRST_RESIDENT_PRESENT,
    ]
    assert [event.monotonic_nanoseconds for event in events] == sorted(
        event.monotonic_nanoseconds for event in events
    )
    assert events[0].to_dict()["counters"]["residentBytes"] == 20
    assert events[0].to_dict()["counters"]["peakProcessRSSBytes"] == 30
    assert events[0].to_dict()["counters"]["peakDeviceAllocatedBytes"] == 40
    assert events[4].source_identity_sha256 == SOURCE_A


def test_present_requires_resident_ready_and_recovery_requires_loss() -> None:
    recorder = MPSPublicationRecorder()
    recorder.begin(
        7,
        SOURCE_A,
        MPSResidentRepresentation.DENSE,
    )

    with pytest.raises(ValueError, match="invalid for the current generation"):
        recorder.record(7, MPSPublicationMilestone.FIRST_RESIDENT_PRESENT)
    with pytest.raises(ValueError, match="invalid for the current generation"):
        recorder.record(7, MPSPublicationMilestone.RECOVERY_READY)

    assert recorder.record(7, MPSPublicationMilestone.DEVICE_LOST)
    assert recorder.record(7, MPSPublicationMilestone.RECOVERY_READY)
    assert recorder.record(7, MPSPublicationMilestone.FIRST_RESIDENT_PRESENT)


def test_timing_summary_uses_nearest_rank_percentiles() -> None:
    summary = MPSTimingSummary.from_samples([0.4, 0.1, 0.3, 0.2, 0.5])

    assert summary.sample_count == 5
    assert summary.p50_seconds == 0.3
    assert summary.p95_seconds == 0.5
    assert summary.maximum_seconds == 0.5
    assert (
        MPSTimingBoundary.COLD_ARBITRARY_TO_RESIDENT_READY.value
        != MPSTimingBoundary.PREPARED_REOPEN_TO_RESIDENT_READY.value
    )

    with pytest.raises(ValueError, match="finite nonnegative"):
        MPSTimingSummary.from_samples([float("nan")])


def test_consumer_symbols_are_lazy_backend_exports() -> None:
    from quantem.gpu.io.backends import mps
    from quantem.gpu.io.backends.mps.resident_dpc import MPSDPCProcessor

    assert mps.describe_resident is describe_resident
    assert mps.MPSPublicationRecorder is MPSPublicationRecorder
    assert mps.MPSDPCProcessor is MPSDPCProcessor
