#!/usr/bin/env python3
"""Prepare a source-preserving compact-H5 copy for fast exact reopen."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from quantem.gpu.io._compact_h5 import prepare_compact_h5_metadata_copy


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    """Attach authenticated envelope hashes and optional detector calibration."""
    parser = argparse.ArgumentParser(
        description=(
            "Create a new compact-H5 copy with source-bound prepared metadata. "
            "The source is never modified and an existing destination is refused."
        )
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--center-row", type=float)
    parser.add_argument("--center-column", type=float)
    parser.add_argument("--bright-field-radius", type=float)
    parser.add_argument("--dpc-rotation-degrees", type=float)
    parser.add_argument(
        "--dpc-component-order-exchanged",
        action=argparse.BooleanOptionalAction,
        default=None,
    )
    parser.add_argument(
        "--method",
        default="mean-diffraction-half-maximum",
    )
    parser.add_argument(
        "--no-encoded-envelope-hashes",
        action="store_true",
        help="omit the prepared-reopen hashes (normally not useful)",
    )
    parser.add_argument(
        "--masked-raw-values",
        type=int,
        nargs="*",
        help=(
            "QGIX v3 producer-proven constant uint16 values in exact ordered "
            "masked-pixel order"
        ),
    )
    parser.add_argument(
        "--expected-source-sha256",
        help="required immutable whole-file SHA-256 for a QGIX v3 source",
    )
    arguments = parser.parse_args()

    detector_values = (
        arguments.center_row,
        arguments.center_column,
        arguments.bright_field_radius,
    )
    calibration_requested = any(value is not None for value in detector_values)
    calibration = None
    if calibration_requested:
        if any(value is None for value in detector_values):
            parser.error(
                "--center-row, --center-column, and --bright-field-radius "
                "must be supplied together"
            )
        dpc_values = (
            arguments.dpc_rotation_degrees,
            arguments.dpc_component_order_exchanged,
        )
        if (dpc_values[0] is None) != (dpc_values[1] is None):
            parser.error(
                "--dpc-rotation-degrees and "
                "--dpc-component-order-exchanged/"
                "--no-dpc-component-order-exchanged must be supplied together"
            )
        calibration = {
            "detector_center_px": [
                arguments.center_row,
                arguments.center_column,
            ],
            "bright_field_radius_px": arguments.bright_field_radius,
            "method": arguments.method,
        }
        if dpc_values[0] is not None:
            calibration.update(
                {
                    "dpc_rotation_degrees": arguments.dpc_rotation_degrees,
                    "dpc_component_order_exchanged": (
                        arguments.dpc_component_order_exchanged
                    ),
                }
            )
    elif arguments.dpc_rotation_degrees is not None or (
        arguments.dpc_component_order_exchanged is not None
    ):
        parser.error("DPC calibration requires detector center and BF radius")

    prepared = prepare_compact_h5_metadata_copy(
        arguments.source,
        arguments.destination,
        detector_calibration=calibration,
        add_encoded_envelope_hashes=not arguments.no_encoded_envelope_hashes,
        masked_detector_raw_values=(
            tuple(arguments.masked_raw_values)
            if arguments.masked_raw_values is not None
            else None
        ),
        expected_source_sha256=arguments.expected_source_sha256,
    )
    summary = {
        "source": str(arguments.source.resolve()),
        "destination": str(prepared.path),
        "source_identity_sha256": prepared.source_identity_sha256,
        "shape": list(prepared.shape),
        "file_bytes": prepared.file_bytes,
        "whole_file_sha256": _sha256_file(prepared.path),
        "shard_count": len(prepared.shards),
        "encoded_envelope_sha256_count": sum(
            "encoded_envelope_sha256" in record
            for record in prepared.manifest.get("shards", [])
        ),
        "masked_detector_pixels_sha256": (prepared.masked_detector_pixels_sha256),
        "masked_detector_raw_values": prepared.masked_detector_raw_values,
        "raw_reconstruction_available": prepared.raw_reconstruction_available,
        "detector_calibration": prepared.manifest.get("detector_calibration"),
    }
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
