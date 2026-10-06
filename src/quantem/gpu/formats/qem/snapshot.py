"""The ``.qem`` envelope and the encoded-array layout of each codec.

A saved copy is ``MAGIC``, the header length and body offset, the SHA-256 of
the JSON header, the header, then the encoded arrays in 64 MiB checksummed
blocks. Readers authenticate the header and check every array span before
allocating GPU memory, and check every body block before a decoder sees it.
This module holds that format alone: no GPU, decoder or encoder code.
"""

import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from quantem.gpu.formats.precision import validate_regions
from quantem.gpu.formats.publish import publish_file
from quantem.gpu.formats.qem import metadata

INTEGER_CODEC = "runtime-column-rans-spatial-v2"
SCALED_CODEC = "scaled-uint16-column-rans-v1"
FLOAT_CODEC = "float32-bit-lanes-rans-v1"
BLOCK_BYTES = 64 << 20
# Per chunk of the integer codec: payload, stream offsets, models, then the
# spatial index's packed words, their starts and their bit widths.
ARRAY_DTYPES = ("uint8", "uint32", "uint8", "uint32", "uint64", "uint8")


def is_qem_file(path) -> bool:
    """Identify a saved ``.qem`` copy from its magic, without reading detector payloads."""
    if not isinstance(path, (str, os.PathLike)):
        return False
    try:
        with open(path, "rb") as handle:
            return handle.read(8) == metadata.MAGIC
    except (OSError, TypeError):
        return False


def read_envelope(path: str | Path) -> tuple[dict, int]:
    """Read and authenticate a ``.qem`` header; return it and the body offset.

    The header checksum is verified before the JSON is parsed, duplicate JSON
    keys are refused (``json`` would silently keep the last), and the metadata
    contract is validated. No detector data is read.
    """
    with open(path, "rb") as handle:
        prefix = handle.read(56)
        if len(prefix) != 56 or prefix[:8] != metadata.MAGIC:
            raise ValueError("Not a complete QEM file; choose a .qem file written by io.save.")
        length, start = struct.unpack("<QQ", prefix[8:24])
        size = os.fstat(handle.fileno()).st_size
        if not 0 < length <= 16 << 20 or start != 56 + length or start > size:
            raise ValueError("Invalid QEM header length; recopy the file.")
        blob = handle.read(length)
        if len(blob) != length or hashlib.sha256(blob).digest() != prefix[24:]:
            raise ValueError("QEM header checksum mismatch; recopy the file.")
    header = json.loads(blob, object_pairs_hook=_unique_keys)
    metadata.validate_header(header)
    return header, start


def read_header(path: str | Path) -> tuple[dict, int]:
    """Authenticate a ``.qem`` header and check every encoded array span it declares.

    Loaders call this before allocating, so a truncated or inconsistent file
    fails with a clear message instead of a decoder fault.
    """
    header, start = read_envelope(path)
    size = os.stat(path).st_size
    codec = header.get("codec")
    if codec == SCALED_CODEC:
        try:
            _validate_scaled_layout(header)
        except (KeyError, TypeError, IndexError) as error:
            raise ValueError("Malformed scaled .qem header; save the result again.") from error
        if (start + header["bytes"] != size or not isinstance(header.get("sha256"), list)
                or len(header["sha256"]) != math.ceil(header["bytes"] / BLOCK_BYTES)):
            raise ValueError("Incomplete scaled .qem file or checksum table; recopy the file.")
        return header, start
    if codec == FLOAT_CODEC:
        shape = header.get("shape")
        if (not isinstance(shape, list) or len(shape) != 4
                or any(type(n) is not int or not 0 < n < 1 << 24 for n in shape)
                or type(header.get("bytes")) is not int or header["bytes"] <= 0
                or start + header["bytes"] != size
                or not isinstance(header.get("sha256"), list)
                or len(header["sha256"]) != math.ceil(header["bytes"] / BLOCK_BYTES)):
            raise ValueError("Invalid float ANS shape, length or checksum table; recopy the file.")
        validate_declared_processing(header)
        with open(path, "rb") as handle:
            validate_float_layout(handle, header, start)
        if any(chunk["payload_bytes"] > 32 << 20 for chunk in header["chunks"]):
            raise ValueError("Float ANS payload padding exceeds the 32 MiB GPU window; re-export the source.")
        return header, start
    if codec != INTEGER_CODEC:
        raise NotImplementedError(
            f"This Python reader does not support QEM codec {codec!r}. "
            "Re-export the original acquisition using the current .qem writer."
        )
    try:
        shape = header["shape"]
        if (header["profile"] != INTEGER_CODEC or header["version"] != 1
                or header["interval"] != 512 or header["dtype"] not in ("uint8", "uint16")
                or len(shape) != 4 or any(type(n) is not int or n <= 0 for n in shape)):
            raise ValueError("Unsupported ANS snapshot geometry or codec version.")
        pixels, fields = math.prod(shape[2:]), field_count(shape[2:])
        if len(bytes.fromhex(header["valid"])) != (pixels + 7) // 8:
            raise ValueError("Invalid detector validity mask.")
        cursor = first = 0
        for chunk in header["chunks"]:
            scans = chunk["scans"]
            if chunk["first"] != first or type(scans) is not int or scans <= 0:
                raise ValueError("Invalid ANS chunk scan coverage.")
            blocks = math.ceil(scans / 512)
            if blocks * pixels * (2 * min(scans, 512) + 4) >= 2**32:
                raise ValueError("ANS chunk offsets exceed the codec's uint32 range.")
            first += scans
            if len(chunk["arrays"]) != 6:
                raise ValueError("Incomplete ANS chunk arrays.")
            for index, spec in enumerate(chunk["arrays"]):
                count = spec["count"]
                expected = {1: blocks * pixels + 1, 2: blocks * pixels,
                            4: blocks * fields + 1, 5: blocks * fields}
                cursor = (cursor + 7) & ~7
                if (type(count) is not int or count < 0 or spec["offset"] != cursor
                        or index in expected and count != expected[index]):
                    raise ValueError("Invalid ANS array span.")
                cursor += count * np.dtype(ARRAY_DTYPES[index]).itemsize
        if first != math.prod(shape[:2]) or header["bytes"] != cursor or start + cursor != size:
            raise ValueError("Incomplete ANS snapshot; recopy the complete file.")
        if len(header["sha256"]) != math.ceil(cursor / BLOCK_BYTES):
            raise ValueError("Incomplete ANS snapshot checksum table.")
        if not isinstance(header["metadata"], dict):
            raise ValueError("Invalid acquisition metadata.")
    except (KeyError, TypeError, IndexError, OverflowError) as error:
        raise ValueError("Malformed ANS snapshot header; recopy the file.") from error
    return header, start


def write_envelope(path: Path, header: dict, body) -> None:
    """Publish ``header`` and the encoded arrays in ``body`` as a new ``.qem`` file.

    ``body`` is an open file positioned at the end of the arrays. Its length
    and per-block checksums go into the header; the file is written beside
    ``path`` and renamed into place only when complete, so a reader never
    sees a partial copy and an existing file is never replaced.
    """
    header["bytes"] = body.tell()
    body.seek(0)
    header["sha256"] = []
    while block := body.read(BLOCK_BYTES):
        header["sha256"].append(hashlib.sha256(block).hexdigest())
    blob = json.dumps(
        header,
        default=metadata.json_metadata,
        sort_keys=True,
        allow_nan=False,
        separators=(",", ":"),
    ).encode()
    if len(blob) > 16 << 20:
        raise ValueError("QEM metadata exceeds the 16 MiB header limit.")
    descriptor, temporary = tempfile.mkstemp(prefix=".qem-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as output:
            output.write(metadata.MAGIC + struct.pack("<QQ", len(blob), 56 + len(blob)))
            output.write(hashlib.sha256(blob).digest() + blob)
            body.seek(0)
            while block := body.read(BLOCK_BYTES):
                output.write(block)
            output.flush()
            os.fsync(output.fileno())
        publish_file(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)


def verify_body(handle, header: dict, start: int) -> None:
    """Check every 64 MiB body block of an open ``.qem`` file against its header checksum."""
    handle.seek(start)
    for number, expected in enumerate(header["sha256"]):
        block = handle.read(min(BLOCK_BYTES, header["bytes"] - number * BLOCK_BYTES))
        if hashlib.sha256(block).hexdigest() != expected:
            raise ValueError(f"QEM body checksum mismatch in block {number}; recopy the file.")


def encode_valid(valid: np.ndarray) -> str:
    """Pack a boolean detector-validity mask into the header's hex string, row-major."""
    return np.packbits(np.asarray(valid, bool).ravel()).tobytes().hex()


def decode_valid(text: str, shape: tuple[int, int]) -> np.ndarray:
    """Unpack the header's hex validity string into a ``(row, col)`` boolean mask."""
    bits = np.unpackbits(np.frombuffer(bytes.fromhex(text), np.uint8))
    return bits[: math.prod(shape)].astype(bool).reshape(shape)


def field_count(shape: tuple[int, int]) -> int:
    """Number of exact spatial sums per scan: every 8x8 tile plus every 32x32 tile of the detector."""
    return sum(
        math.ceil(shape[0] / side) * math.ceil(shape[1] / side) for side in (8, 32)
    )


def _unique_keys(pairs: list[tuple[str, object]]) -> dict[str, object]:
    """Build a JSON object, refusing duplicate keys that ``json`` would silently overwrite."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate QEM metadata key {key!r}; re-export the file.")
        result[key] = value
    return result


def validate_declared_processing(header: dict) -> None:
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
        operation = (
            "exact_float_narrowing"
            if (source_dtype, stored_dtype) == ("float64", "float32")
            else "exact_integer_narrowing"
        )
        record = declared.get(operation)
        if (record is None or record.get("source_dtype") != source_dtype
                or record.get("stored_dtype") != stored_dtype
                or record.get("changes_measurements") is not False):
            raise ValueError(
                f"Counts were stored as {stored_dtype} from {source_dtype} but processing does "
                f"not declare matching {operation}; re-export the original acquisition."
            )


def _validate_scaled_layout(header: dict) -> None:
    """Check the regional calibration and every span of saved scaled uint16 codes.

    The three arrays per chunk are the integer codec's payload, offsets and
    models, without a spatial index; ``region`` names the chunk's calibration.
    """
    shape = header["shape"]
    frames, pixels = shape[0] * shape[1], shape[2] * shape[3]
    report = header.get("intensity_calibration")
    if (header.get("version") != 1 or header.get("interval") != 512
            or header.get("dtype") != "uint16" or not isinstance(report, dict)
            or report.get("storage") != "scaled_uint16" or report.get("version") != 2
            or report.get("complete") is not True or report.get("source_shape") != shape):
        raise ValueError("Invalid scaled uint16 description; save the result again.")
    validate_regions(report)
    if not all(region.get("storage") == "scaled_uint16"
               and all(isinstance(region.get(key), (int, float))
                       and math.isfinite(region[key])
                       for key in ("intensity_min", "intensity_max"))
               for region in report["regions"]):
        raise ValueError("Invalid regional intensity range; save the result again.")
    if not any(record.get("operation") == "scaled_uint16_quantization"
               and record.get("changes_measurements") is True
               for record in header["scientific_metadata"]["processing"]):
        raise ValueError(
            "Scaled .qem metadata must declare scaled_uint16_quantization; save the result again."
        )
    regions = report["regions"]
    cursor = first = 0
    for chunk in header["chunks"]:
        scans, region, arrays = chunk.get("scans"), chunk.get("region"), chunk.get("arrays")
        if (type(scans) is not int or scans <= 0 or chunk.get("first") != first
                or type(region) is not int or not 0 <= region < len(regions)
                or not regions[region]["first_frame"] <= first
                or first + scans > regions[region]["stop_frame"]
                or not isinstance(arrays, list) or len(arrays) != 3):
            raise ValueError("Invalid scaled .qem chunk coverage; save the result again.")
        streams = -(-scans // 512) * pixels
        if streams * (2 * min(scans, 512) + 4) >= 2**32:
            raise ValueError("Scaled .qem chunk offsets exceed the codec's uint32 range.")
        for index, (spec, itemsize) in enumerate(zip(arrays, (1, 4, 1))):
            cursor = (cursor + 7) & ~7
            count = spec.get("count")
            if (type(count) is not int or count < 0 or spec.get("offset") != cursor
                    or index == 1 and count != streams + 1
                    or index == 2 and count != streams):
                raise ValueError("Invalid scaled .qem array span; save the result again.")
            cursor += count * itemsize
        first += scans
    if first != frames or header.get("bytes") != cursor:
        raise ValueError("Incomplete scaled .qem coverage; save the result again.")


def validate_float_layout(handle, header: dict, start: int) -> None:
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
