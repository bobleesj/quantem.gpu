"""GPU-free integrity checks for portable QuantEM data files.

Run ``python -m quantem.gpu.io.qem_validation acquisition.qem``. This checks
the saved bytes, not their decoded scientific values or source completeness.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import struct

import numpy as np

from . import _qem_metadata

_BLOCK_BYTES = 64 << 20
_INTEGER_CODEC = "runtime-column-rans-spatial-v2"
_EMPAD_CODEC = "empad-xor-row-packed-v1"
_FLOAT_ANS_CODEC = "float32-bit-lanes-rans-v1"


def _validate_float_ans_layout(handle, header: dict, start: int) -> None:
    """Validate bounded IEEE bit-lane streams without decoding measurements."""
    if (header.get("version") != 1 or header.get("dtype") != "float32"
            or not isinstance(header.get("empad"), dict)
            or not isinstance(header.get("logical_sha256"), str)
            or len(header["logical_sha256"]) != 64):
        raise ValueError("Invalid QEM floating-point ANS description.")
    frames = header["shape"][0] * header["shape"][1]
    lanes = header["shape"][2] * header["shape"][3] * 2
    frame_bytes = lanes * 2
    if frame_bytes > 32 << 20:
        raise ValueError("Float ANS frame exceeds the 32 MiB decode budget.")
    maximum = min(512, (32 << 20) // frame_bytes)
    cursor = first = 0
    for chunk in header["chunks"]:
        count = chunk["scans"]
        if (type(count) is not int or not 1 <= count <= min(maximum, frames - first)
                or chunk["first"] != first):
            raise ValueError("Invalid float ANS frame coverage.")
        arrays = {}
        for name in ("payload", "offset", "model"):
            length = chunk[name + "_bytes"]
            if (type(length) is not int or length <= 0
                    or chunk[name + "_offset"] != cursor
                    or length > header["bytes"] - cursor
                    or name == "offset" and length != (lanes + 1) * 4
                    or name == "model" and length != lanes):
                raise ValueError("Invalid float ANS array bounds.")
            if name != "payload":
                handle.seek(start + cursor)
                arrays[name] = handle.read(length)
            cursor += length
        offsets = np.frombuffer(arrays["offset"], "<u4").astype(np.int64)
        models = np.frombuffer(arrays["model"], "u1")
        lengths = np.diff(offsets)
        if (offsets[0] != 0 or offsets[-1] > chunk["payload_bytes"]
                or np.any(lengths < 0)):
            raise ValueError("Invalid float ANS stream offsets.")
        valid = (((models < 64) & (lengths >= 4) & (lengths <= count * 2))
                 | ((models == 252) & (lengths % 2 == 0) & (lengths <= count * 2))
                 | ((models == 253) & (lengths == 0))
                 | ((models == 254) & (lengths == count * 2))
                 | ((models == 255) & (lengths == 2)))
        if not valid.all():
            raise ValueError("Invalid float ANS stream model or extent.")
        first += count
    if first != frames or cursor != header["bytes"]:
        raise ValueError("Incomplete float ANS coverage.")


def _validate_float_layout(handle, header: dict, start: int) -> None:
    """Check every row descriptor before any float payload is decoded."""
    if (header.get("version") != 1 or header.get("dtype") != "float32"
            or header["shape"][2:] != [128, 128]
            or not isinstance(header.get("logical_sha256"), str)
            or len(header["logical_sha256"]) != 64
            or not isinstance(header.get("empad"), dict)):
        raise ValueError("Invalid QEM float32 codec description.")
    cursor = first = 0
    frames = header["shape"][0] * header["shape"][1]
    for chunk in header["chunks"]:
        count, length = chunk["scans"], chunk["payload_bytes"]
        if (type(count) is not int or not 1 <= count <= min(512, frames - first)
                or chunk["first"] != first or chunk["payload_offset"] != cursor
                or type(length) is not int or length < 4 or length % 4
                or chunk["descriptor_offset"] != cursor + length
                or chunk["descriptor_bytes"] != count * 128 * 16
                or cursor + length + count * 128 * 16 > header["bytes"]):
            raise ValueError("Invalid QEM float32 chunk coverage.")
        handle.seek(start + chunk["descriptor_offset"])
        blob = handle.read(count * 128 * 16)
        descriptors = np.frombuffer(blob, "<u4").reshape(-1, 4)
        words = 0
        for _, width, shift, offset in descriptors:
            width, shift, offset = int(width), int(shift), int(offset)
            if width > 32 or shift > 32 - width or offset != words:
                raise ValueError("Invalid QEM float32 row descriptor.")
            words += width * 4
        if max(4, words * 4) != length:
            raise ValueError("Invalid QEM float32 packed length.")
        first += count
        cursor += length + count * 128 * 16
    if first != frames or cursor != header["bytes"]:
        raise ValueError("Incomplete QEM float32 chunk coverage.")


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate QEM metadata key {key!r}; re-export the file.")
        result[key] = value
    return result


def _validate_declared_processing(header: dict) -> None:
    """A stored change to the counts must appear in the scientific processing list."""
    retained = header.get("metadata") if isinstance(header.get("metadata"), dict) else {}
    declared = {
        record["operation"]: record
        for record in header["scientific_metadata"]["processing"]
    }
    correction = retained.get("hot_pixel_correction")
    if isinstance(correction, dict) and correction.get("applied"):
        record = declared.get("flagged_pixel_replacement")
        if record is None or record["changes_measurements"] is not True:
            raise ValueError(
                "Flagged detector pixels were replaced but processing does not declare "
                "flagged_pixel_replacement; re-export the original acquisition."
            )
    source_dtype, stored_dtype = retained.get("source_dtype"), header.get("dtype")
    if source_dtype and stored_dtype and source_dtype != stored_dtype:
        if "exact_integer_narrowing" not in declared:
            raise ValueError(
                f"Counts were stored as {stored_dtype} from {source_dtype} but processing does "
                "not declare exact_integer_narrowing; re-export the original acquisition."
            )


def validate_qem(path: str | Path) -> dict[str, object]:
    """Verify the envelope, metadata contract and every body checksum.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing QEM file; its signature, not its suffix, identifies the format.

    Returns
    -------
    dict
        JSON-compatible integrity report. Codec-layout and decoded-parity
        status are separate: checksums never establish scientific correctness.

    Raises
    ------
    ValueError
        The envelope, metadata, file length or a checksum is invalid.
    NotImplementedError
        The codec has no qualified validation route.

    Examples
    --------
    >>> report = validate_qem("counts.qem")  # doctest: +SKIP
    >>> report["integrity"]  # doctest: +SKIP
    'verified'
    """
    path = Path(path)
    with path.open("rb") as handle:
        before = os.fstat(handle.fileno())
        prefix = handle.read(56)
        if len(prefix) != 56 or prefix[:8] != _qem_metadata.MAGIC:
            raise ValueError(
                "Not a complete QEM file; export as .qem, do not rename it."
            )
        length, start = struct.unpack("<QQ", prefix[8:24])
        if not 0 < length <= 16 << 20 or start != 56 + length:
            raise ValueError("Invalid QEM header length; recopy the complete file.")
        blob = handle.read(length)
        if len(blob) != length or hashlib.sha256(blob).digest() != prefix[24:]:
            raise ValueError("QEM header checksum mismatch; recopy the complete file.")
        header = json.loads(blob, object_pairs_hook=_unique_keys)
        _qem_metadata.validate_header(header)
        _validate_declared_processing(header)
        shape = header.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 4
            or any(type(size) is not int or not 0 < size < 1 << 24 for size in shape)
        ):
            raise ValueError("Invalid QEM shape; expected four positive integer axes.")
        codec = header.get("codec")
        if codec not in (_INTEGER_CODEC, _EMPAD_CODEC, _FLOAT_ANS_CODEC):
            raise NotImplementedError(
                f"QEM codec {codec!r} is not supported by this validator."
            )
        body_bytes = header.get("bytes")
        if (
            type(body_bytes) is not int
            or body_bytes <= 0
            or start + body_bytes != before.st_size
        ):
            raise ValueError(
                "QEM file length disagrees with metadata; recopy the file."
            )
        checksums = header.get("sha256")
        if (
            not isinstance(checksums, list)
            or len(checksums) != (body_bytes - 1) // _BLOCK_BYTES + 1
        ):
            raise ValueError("Incomplete QEM checksum table; recopy the file.")
        for number, expected in enumerate(checksums):
            block = handle.read(min(_BLOCK_BYTES, body_bytes - number * _BLOCK_BYTES))
            if hashlib.sha256(block).hexdigest() != expected:
                raise ValueError(
                    f"QEM body checksum mismatch in block {number}; recopy the file."
                )
        layout = "not_checked"
        if codec == _INTEGER_CODEC:
            from ._streamed_file import read_header

            read_header(
                path
            )  # Same geometry/span validator used by production loaders.
            layout = "verified"
        elif codec == _EMPAD_CODEC:
            _validate_float_layout(handle, header, start)
            layout = "verified"
        elif codec == _FLOAT_ANS_CODEC:
            _validate_float_ans_layout(handle, header, start)
            layout = "verified"
        after = path.stat()
        if (before.st_dev, before.st_ino, before.st_size, before.st_mtime_ns) != (
            after.st_dev,
            after.st_ino,
            after.st_size,
            after.st_mtime_ns,
        ):
            raise ValueError(
                "QEM file changed during validation; retry with a stable copy."
            )
    scientific = header["scientific_metadata"]
    return dict(
        container_version=header["container_version"],
        codec=codec,
        shape=shape,
        dtype=header.get("dtype"),
        file_bytes=before.st_size,
        integrity="verified",
        codec_layout=layout,
        decoded_parity="not_checked",
        metadata_coverage=scientific.get("source_metadata_coverage", "unknown"),
        processing=[record["operation"] for record in scientific["processing"]],
        measurements_changed_by=[
            record["operation"] for record in scientific["processing"]
            if record["changes_measurements"]
        ],
        calibrated_axes=[
            axis["name"] for axis in scientific["axes"] if "sampling" in axis
        ],
        normalized_fields=sorted(scientific.get("electron_microscope", {})),
        override_fields=sorted(scientific.get("calibration_overrides", {})),
    )


def main(argv: list[str] | None = None) -> int:
    """Validate files without loading a GPU; emit one JSON report per path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args(argv)
    failed = False
    for path in args.paths:
        try:
            print(
                json.dumps(dict(path=str(path), **validate_qem(path)), sort_keys=True)
            )
        except (OSError, ValueError, NotImplementedError, TypeError, KeyError) as error:
            failed = True
            print(
                json.dumps(dict(path=str(path), integrity="failed", error=str(error)))
            )
    return int(failed)


if __name__ == "__main__":
    raise SystemExit(main())
