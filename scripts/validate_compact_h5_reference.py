#!/usr/bin/env python3
"""Validate compact random access against an original 4D-STEM HDF5 family."""

from __future__ import annotations

import argparse
import hashlib
import json
from bisect import bisect_right
from pathlib import Path

import h5py
import hdf5plugin  # noqa: F401  Register bitshuffle/LZ4 with h5py.
import numpy as np

from quantem.gpu.io._compact_h5 import CompactH5Index, CompactH5ReferenceDecoder


def sha256_file(path: Path, block_bytes: int = 8 << 20) -> str:
    """Return a streaming SHA-256 digest for one file."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(block_bytes):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Compare deterministic source samples and write one JSON receipt."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--compact", required=True, type=Path)
    parser.add_argument("--master", required=True, type=Path)
    parser.add_argument("--source-audit", type=Path)
    parser.add_argument("--expected-compact-sha256")
    parser.add_argument("--expected-prepared-sha256")
    parser.add_argument(
        "--skip-v3-payload-sha256",
        action="store_true",
        help="Skip expensive per-shard QGIX v3 payload authentication.",
    )
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260903)
    parser.add_argument("--output", type=Path)
    arguments = parser.parse_args()

    index = CompactH5Index.from_file(arguments.compact)
    decoder = CompactH5ReferenceDecoder(index)
    if index.schema_version == 3:
        index.require_raw_reconstruction()
    for shard_index in range(len(index.shards)):
        decoder.validate_shard_metadata(shard_index)

    compact_sha256 = sha256_file(arguments.compact)
    if index.schema_version == 3 and arguments.expected_compact_sha256 is None:
        raise ValueError(
            "QGIX v3 acceptance requires --expected-compact-sha256 because its "
            "binary index authenticates payloads but not compact headers."
        )
    if (
        arguments.expected_compact_sha256 is not None
        and compact_sha256 != arguments.expected_compact_sha256
    ):
        raise ValueError(
            f"Compact file SHA-256 is {compact_sha256}, expected "
            f"{arguments.expected_compact_sha256}."
        )
    prepared_sha256 = index.manifest.get("prepared_uint8_sha256")
    if (
        arguments.expected_prepared_sha256 is not None
        and prepared_sha256 != arguments.expected_prepared_sha256
    ):
        raise ValueError(
            "Compact prepared logical SHA-256 does not match the expected "
            "mask-applied uint8 identity."
        )

    payload_sha256_checks = 0
    if index.schema_version == 3 and not arguments.skip_v3_payload_sha256:
        for shard_index in range(len(index.shards)):
            decoder.validate_shard_payload(shard_index)
            payload_sha256_checks += 1

    audit = (
        json.loads(arguments.source_audit.read_text())
        if arguments.source_audit
        else None
    )
    if audit is not None:
        if audit["source_identity_sha256"] != index.source_identity_sha256:
            raise ValueError("Compact and raw-audit source identities disagree.")
        if tuple(audit["source_shape"]) != index.shape:
            raise ValueError("Compact and raw-audit logical shapes disagree.")
        if audit["source_dtype"] != "uint16":
            raise ValueError(
                f"Compact validation requires raw uint16, got {audit['source_dtype']}."
            )
        compact_raw_sha = index.manifest.get("source_raw_logical_sha256")
        audit_raw_sha = audit["range_audit"]["logical_source_sha256"]
        if compact_raw_sha != audit_raw_sha:
            raise ValueError("Compact and raw-audit logical SHA-256 values disagree.")

    scan_rows, scan_columns, detector_rows, detector_columns = index.shape
    scan_count = scan_rows * scan_columns
    detector_pixels = detector_rows * detector_columns
    rng = np.random.default_rng(arguments.seed)
    pairs = {
        (0, 0),
        (0, detector_pixels - 1),
        (scan_count - 1, 0),
        (scan_count - 1, detector_pixels - 1),
        (scan_count // 2, detector_pixels // 2),
    }
    while len(pairs) < arguments.samples:
        pairs.add(
            (
                int(rng.integers(0, scan_count)),
                int(rng.integers(0, detector_pixels)),
            )
        )
    for scan in (0, scan_count // 2, scan_count - 1):
        pairs.update((scan, pixel) for pixel in index.excluded_detector_pixels)

    mismatches = []
    raw_mismatches = []
    masked_checks = 0
    sampled_raw_maximum = 0
    sampled_compact_maximum = 0
    sample_digest = hashlib.sha256()
    raw_sample_digest = hashlib.sha256()
    with h5py.File(arguments.master, "r") as master:
        data_group = master["/entry/data"]
        names = sorted(data_group)
        datasets = [data_group[name] for name in names]
        frame_stops = np.cumsum([dataset.shape[0] for dataset in datasets]).tolist()
        if frame_stops[-1] != scan_count:
            raise ValueError(
                f"Master family contains {frame_stops[-1]} frames, expected {scan_count}."
            )
        if any(
            tuple(dataset.shape[1:]) != (detector_rows, detector_columns)
            or dataset.dtype != np.dtype("uint16")
            for dataset in datasets
        ):
            raise ValueError(
                "Every source member must be uint16 with the compact detector shape."
            )
        excluded = set(index.excluded_detector_pixels)
        for scan, pixel in sorted(pairs):
            member = bisect_right(frame_stops, scan)
            member_start = 0 if member == 0 else frame_stops[member - 1]
            detector_row, detector_column = divmod(pixel, detector_columns)
            raw = int(
                datasets[member][scan - member_start, detector_row, detector_column]
            )
            compact = decoder.value(
                scan // scan_columns,
                scan % scan_columns,
                detector_row,
                detector_column,
            )
            expected = 0 if pixel in excluded else raw
            reconstructed_raw = (
                decoder.raw_value(
                    scan // scan_columns,
                    scan % scan_columns,
                    detector_row,
                    detector_column,
                )
                if index.raw_reconstruction_available
                else None
            )
            masked_checks += int(pixel in excluded)
            sampled_raw_maximum = max(sampled_raw_maximum, raw)
            sampled_compact_maximum = max(sampled_compact_maximum, compact)
            sample_digest.update(
                scan.to_bytes(4, "little")
                + pixel.to_bytes(4, "little")
                + raw.to_bytes(2, "little")
                + compact.to_bytes(2, "little")
            )
            if reconstructed_raw is not None:
                raw_sample_digest.update(
                    scan.to_bytes(4, "little")
                    + pixel.to_bytes(4, "little")
                    + raw.to_bytes(2, "little")
                    + reconstructed_raw.to_bytes(2, "little")
                )
            if compact != expected:
                mismatches.append(
                    {
                        "scan_row": scan // scan_columns,
                        "scan_column": scan % scan_columns,
                        "detector_row": detector_row,
                        "detector_column": detector_column,
                        "raw": raw,
                        "expected_working": expected,
                        "compact": compact,
                    }
                )
            if reconstructed_raw is not None and reconstructed_raw != raw:
                raw_mismatches.append(
                    {
                        "scan_row": scan // scan_columns,
                        "scan_column": scan % scan_columns,
                        "detector_row": detector_row,
                        "detector_column": detector_column,
                        "raw": raw,
                        "reconstructed_raw": reconstructed_raw,
                    }
                )

    result = {
        "schema": "quantem.gpu.compact-h5-reference-validation/v1",
        "status": "pass" if not mismatches and not raw_mismatches else "fail",
        "compact": {
            "path": str(arguments.compact.resolve()),
            "bytes": index.file_bytes,
            "sha256": compact_sha256,
            "expected_sha256": arguments.expected_compact_sha256,
            "whole_file_identity_verified": (
                arguments.expected_compact_sha256 == compact_sha256
            ),
            "qgix_schema_version": index.schema_version,
            "source_identity_sha256": index.source_identity_sha256,
            "source_raw_logical_sha256": index.manifest.get(
                "source_raw_logical_sha256"
            ),
            "prepared_uint8_sha256": prepared_sha256,
            "logical_shape": list(index.shape),
            "logical_source_dtype": "uint16",
            "working_dtype": index.manifest.get("working_dtype"),
            "detector_mask_sha256": index.manifest.get("detector_mask_sha256"),
            "masked_detector_pixels_sha256": (index.masked_detector_pixels_sha256),
            "masked_detector_raw_values": index.masked_detector_raw_values,
            "raw_reconstruction_available": index.raw_reconstruction_available,
            "scan_bin": 1,
            "detector_bin": 1,
            "crop": None,
            "scan_tile": index.scan_tile,
            "header_encoding": index.header_encoding,
            "resident_bytes": index.resident_bytes,
            "payload_dataset_pattern": (
                "/quantem_gpu/shards/NNN/payload_u32"
                if index.schema_version == 3
                else "/quantem_gpu/shards/NNN/payload_lz4"
            ),
            "header_dataset_pattern": (
                "/quantem_gpu/shards/NNN/compact_headers_u32"
                if index.schema_version == 3
                else "/quantem_gpu/shards/NNN/descriptor_widths"
            ),
            "payload_sha256_checks": payload_sha256_checks,
            "payload_sha256_complete": (
                index.schema_version != 3 or payload_sha256_checks == len(index.shards)
            ),
            "header_integrity_policy": (
                "whole-file SHA-256 seals QGIX v3 compact headers"
                if index.schema_version == 3
                else "v1 manifest and binary-index agreement"
            ),
        },
        "source": {
            "master_path": str(arguments.master.resolve()),
            "master_bytes": arguments.master.stat().st_size,
            "master_sha256": sha256_file(arguments.master),
            "dataset_path": "/entry/data/data",
            "member_count": len(frame_stops),
            "logical_shape": list(index.shape),
            "dtype": "uint16",
        },
        "reference": {
            "seed": arguments.seed,
            "sample_count": len(pairs),
            "masked_sample_count": masked_checks,
            "mismatch_count": len(mismatches),
            "sample_receipt_sha256": sample_digest.hexdigest(),
            "sampled_raw_maximum": sampled_raw_maximum,
            "sampled_compact_maximum": sampled_compact_maximum,
            "mask_policy": "nonzero detector-mask entries are exact working zeros",
            "mismatches": mismatches[:32],
            "raw_reconstruction_mismatch_count": len(raw_mismatches),
            "raw_sample_receipt_sha256": raw_sample_digest.hexdigest(),
            "raw_mismatches": raw_mismatches[:32],
        },
    }
    encoded = json.dumps(result, indent=2, sort_keys=True) + "\n"
    if arguments.output:
        arguments.output.parent.mkdir(parents=True, exist_ok=True)
        arguments.output.write_text(encoded)
    print(encoded, end="")
    if mismatches or raw_mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
