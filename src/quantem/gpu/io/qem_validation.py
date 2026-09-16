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
        shape = header.get("shape")
        if (
            not isinstance(shape, list)
            or len(shape) != 4
            or any(type(size) is not int or not 0 < size < 1 << 24 for size in shape)
        ):
            raise ValueError("Invalid QEM shape; expected four positive integer axes.")
        codec = header.get("codec")
        if codec not in (_INTEGER_CODEC, _EMPAD_CODEC):
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
        calibrated_axes=[
            axis["name"] for axis in scientific["axes"] if "sampling" in axis
        ],
        normalized_fields=sorted(scientific.get("electron_microscope", {})),
        override_fields=sorted(scientific.get("calibration_overrides", {})),
    )


def main() -> int:
    """Validate files without loading a GPU; emit one JSON report per path."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("paths", nargs="+", type=Path)
    args = parser.parse_args()
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
