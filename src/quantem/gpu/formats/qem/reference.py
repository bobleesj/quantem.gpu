"""Readable, explicit CPU reference for QEM version-1 measurement codecs.

Only NumPy and the standard library are used. No accelerated codec or entropy
table builder is imported. The frozen table and bitstream specification own the
decoding rules; this implementation favors inspectability over throughput.
"""

import hashlib
import json
import math
import tempfile
from functools import cache
from pathlib import Path

import numpy as np

from quantem.gpu.formats.qem import metadata as qem_metadata
from quantem.gpu.formats.qem.snapshot import (
    FLOAT_CODEC,
    INTEGER_CODEC,
    SCALED_CODEC,
    decode_valid,
    encode_valid,
    read_envelope,
    write_envelope,
)
from quantem.gpu.formats.qem.validation import validate_qem

_LOWER = 1 << 23


@cache
def _tables() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    record = json.loads(Path(__file__).with_name("qem-rans-tables-v1.json").read_text())
    frequencies = np.asarray(record["frequencies"], dtype=np.int64)
    starts = np.cumsum(frequencies, axis=1) - frequencies
    symbols = np.stack([np.repeat(np.arange(33), row) for row in frequencies])
    return frequencies, starts, symbols


@cache
def count_tables() -> tuple[np.ndarray, np.ndarray]:
    """Return the rANS encoding and decoding tables the CUDA and Metal codecs upload.

    ``encoding[model, symbol]`` packs ``frequency << 16 | start``; each of a
    model's 1024 decoding slots packs ``frequency << 16 | start << 6 | symbol``.
    Both expand the frequencies of ``qem-rans-tables-v1.json``, the file this
    CPU reference decoder reads, so every decoder uses the same model. The
    arrays are shared by every caller; upload them, never modify them.
    """
    frequencies, starts, symbols = (table.astype(np.uint32) for table in _tables())
    encoding = (frequencies << 16) | starts
    decoding = np.stack([
        (np.repeat(frequency, frequency) << 16) | (np.repeat(start, frequency) << 6) | symbol
        for frequency, start, symbol in zip(frequencies, starts, symbols, strict=True)
    ])
    return encoding, decoding


def _decode_stream(encoded: bytes, model: int, count: int) -> list[int]:
    """Decode one pixel's 512-scan count stream on the CPU, as the GPU decoder does.

    Model 253 is all zeros, 255 one repeated count, 254 raw uint16, 252 sparse
    (position, count) events, and 0..63 an rANS table; any inconsistency
    raises instead of returning wrong counts.
    """
    if model == 253 and not encoded:
        return [0] * count
    if model == 255 and len(encoded) == 2:
        return [int.from_bytes(encoded, "little")] * count
    if model == 254 and len(encoded) == 2 * count:
        return np.frombuffer(encoded, "<u2").tolist()
    if model == 252 and len(encoded) % 2 == 0:
        values, previous = [0] * count, -1
        for event in np.frombuffer(encoded, "<u2"):
            position, value = int(event) >> 7, (int(event) & 127) + 1
            if not previous < position < count:
                raise ValueError(
                    "Invalid QEM sparse-event position; re-export the file."
                )
            values[position], previous = value, position
        return values
    if not 0 <= model < 64 or len(encoded) < 4:
        raise ValueError("Invalid QEM count stream; re-export the file.")
    frequencies, starts, symbols = _tables()
    state, cursor = int.from_bytes(encoded[:4], "little"), 4
    if not _LOWER <= state < 1 << 31:
        raise ValueError("Invalid QEM entropy state.")
    values = []
    for _ in range(count):
        slot = state & 1023
        symbol = int(symbols[model, slot])
        state = (
            int(frequencies[model, symbol]) * (state >> 10)
            + slot
            - int(starts[model, symbol])
        )
        while state < _LOWER:
            if cursor >= len(encoded):
                raise ValueError("Truncated QEM entropy stream.")
            state = (state << 8) | encoded[cursor]
            cursor += 1
        if symbol == 32:
            if cursor + 2 > len(encoded):
                raise ValueError("Truncated QEM escaped count.")
            symbol = int.from_bytes(encoded[cursor : cursor + 2], "little")
            cursor += 2
        values.append(symbol)
    if cursor != len(encoded) or state != _LOWER:
        raise ValueError(
            "QEM count stream has trailing data or an invalid final state."
        )
    return values


def _encode_stream(values: np.ndarray) -> tuple[int, bytes]:
    """Encode one pixel's count stream with the smallest of the decoder's models."""
    if np.all(values == values[0]):
        value = int(values[0])
        return (255, value.to_bytes(2, "little")) if value else (253, b"")
    frequencies, starts, _ = _tables()
    mean = max(float(np.minimum(values, 32).mean()), 0.002)
    model = max(0, min(63, round(math.log(mean / 0.002) * 63 / math.log(16000))))
    state, emitted = _LOWER, bytearray()
    for raw in values[::-1]:
        value = int(raw)
        symbol = min(value, 32)
        if value >= 32:
            emitted.extend((value >> 8, value & 255))
        frequency, start = int(frequencies[model, symbol]), int(starts[model, symbol])
        while state >= ((_LOWER >> 10) << 8) * frequency:
            emitted.append(state & 255)
            state >>= 8
        state = (state // frequency << 10) + state % frequency + start
    encoded = state.to_bytes(4, "little") + bytes(reversed(emitted))
    if len(encoded) >= values.size * 2:
        model, encoded = 254, values.astype("<u2").tobytes()
    nonzero = np.flatnonzero(values)
    if values.max() <= 128 and nonzero.size * 2 < len(encoded):
        model = 252
        encoded = b"".join(
            ((int(i) << 7) | (int(values[i]) - 1)).to_bytes(2, "little")
            for i in nonzero
        )
    return model, encoded


def _pack(values: np.ndarray, width: int) -> bytes:
    """Bit-pack values at ``width`` bits each into whole little-endian 32-bit words."""
    reservoir = 0
    for index, value in enumerate(values):
        reservoir |= int(value) << (index * width)
    return reservoir.to_bytes(((len(values) * width + 31) // 32) * 4, "little")


def _integer_arrays(frames: np.ndarray, valid: np.ndarray) -> tuple[bytes, ...]:
    """Reference encoding of native counts: per-pixel streams plus the spatial index.

    The index stores exact 8 x 8 and 32 x 32 detector-block sums of valid
    pixels per scan, bit-packed per 512-scan block, so region sums need not
    decode every pixel.
    """
    scans, rows, columns = frames.shape
    flat = frames.reshape(scans, -1)
    payload, offsets, models = bytearray(), [0], []
    for first in range(0, scans, 512):
        for pixel in range(rows * columns):
            model, encoded = _encode_stream(flat[first : first + 512, pixel])
            models.append(model)
            payload.extend(encoded)
            offsets.append(len(payload))
    fields = []
    for side in (8, 32):
        for row in range(0, rows, side):
            for col in range(0, columns, side):
                block = frames[:, row : row + side, col : col + side]
                mask = valid[row : row + side, col : col + side]
                fields.append((block * mask).sum(axis=(1, 2), dtype=np.uint64))
    words, starts, widths = bytearray(), [0], []
    for first in range(0, scans, 512):
        for field in fields:
            values = field[first : first + 512]
            width = int(values.max()).bit_length()
            widths.append(width)
            words.extend(_pack(values, width))
            starts.append(len(words) // 4)
    return (
        bytes(payload),
        np.asarray(offsets, "<u4").tobytes(),
        bytes(models),
        bytes(words),
        np.asarray(starts, "<u8").tobytes(),
        bytes(widths),
    )


def _float_arrays(frames: np.ndarray) -> tuple[bytes, bytes, bytes]:
    """Reference ANS encoding of IEEE bits, never float-to-integer conversion."""
    lanes = frames.view("<u2").reshape(len(frames), -1)
    payload, offsets, models = bytearray(), [0], []
    for lane in lanes.T:
        model, encoded = _encode_stream(lane)
        models.append(model)
        payload.extend(encoded)
        offsets.append(len(payload))
    return (bytes(payload or b"\0"), np.asarray(offsets, "<u4").tobytes(),
            bytes(models))


def save_array(
    path: str | Path,
    data: np.ndarray,
    metadata: dict | None = None,
    *,
    chunk_scans: int = 512,
) -> None:
    """Write unchanged counts with the reference encoder; publish without overwrite."""
    path = Path(path)
    if path.suffix.lower() != ".qem" or path.exists():
        raise ValueError("Choose a new, non-existing .qem destination.")
    if (
        data.ndim != 4
        or min(data.shape) <= 0
        or data.dtype
        not in (np.dtype("uint8"), np.dtype("uint16"), np.dtype("float32"))
    ):
        raise ValueError(
            "QEM needs 4D uint8/uint16 counts or float32 measurements; do not cast measurements silently."
        )
    floating = data.dtype == np.float32
    frame_bytes = math.prod(data.shape[2:]) * data.dtype.itemsize
    if floating and frame_bytes > 32 << 20:
        raise ValueError("One float32 diffraction frame exceeds the 32 MiB decode budget.")
    maximum = min(512, (32 << 20) // frame_bytes) if floating else 1024
    if floating:
        chunk_scans = min(chunk_scans, maximum)
    if not 1 <= chunk_scans <= maximum:
        raise ValueError(
            f"Reference encoding chunk_scans must be between 1 and {maximum}."
        )
    streams = ((chunk_scans + 511) // 512) * math.prod(data.shape[2:])
    if not floating and streams * (2 * min(chunk_scans, 512) + 4) >= 2**32:
        raise ValueError("Reduce batch_size so QEM chunk offsets fit uint32.")
    metadata = {} if metadata is None else dict(metadata)
    if floating and metadata.get("background_applied") is True:
        raise NotImplementedError(
            "Already background-corrected float export is not qualified across QEM readers; "
            "keep the corrected array and its metadata, or export the original measurements."
        )
    scientific = qem_metadata.acquisition_metadata(data.shape, metadata)
    valid = np.asarray(
        metadata.get("valid_pixels", np.ones(data.shape[2:], bool)), dtype=bool
    )
    if valid.shape != data.shape[2:]:
        raise ValueError("valid_pixels must match the detector shape.")
    if floating and not valid.all():
        raise NotImplementedError(
            "The float32 QEM codec cannot retain a detector validity mask yet; "
            "keep the original acquisition instead of silently discarding the mask."
        )
    codec = FLOAT_CODEC if floating else INTEGER_CODEC
    header = dict(
        container="quantem.qem",
        container_version=1,
        version=1,
        profile=codec,
        codec=codec,
        shape=list(data.shape),
        dtype=data.dtype.name,
        scientific_metadata=scientific,
        metadata=metadata,
        chunks=[],
    )
    qem_metadata.validate_header(header)
    frames = data.reshape(-1, *data.shape[2:])
    with tempfile.TemporaryFile(dir=path.parent) as body:
        logical = hashlib.sha256()
        for first in range(0, len(frames), chunk_scans):
            block = np.ascontiguousarray(frames[first : first + chunk_scans])
            if floating:
                logical.update(block.tobytes())
                entry = dict(first=first, scans=len(block))
                for name, encoded in zip(("payload", "offset", "model"),
                                         _float_arrays(block)):
                    entry[name + "_offset"] = body.tell()
                    entry[name + "_bytes"] = len(encoded)
                    body.write(encoded)
                header["chunks"].append(entry)
            else:
                arrays = []
                for encoded, itemsize in zip(
                    _integer_arrays(block, valid), (1, 4, 1, 4, 8, 1)
                ):
                    body.write(b"\0" * ((-body.tell()) % 8))
                    arrays.append(
                        dict(offset=body.tell(), count=len(encoded) // itemsize)
                    )
                    body.write(encoded)
                header["chunks"].append(
                    dict(first=first, scans=len(block), arrays=arrays)
                )
        if floating:
            header["logical_sha256"] = logical.hexdigest()
            header["empad"] = metadata.get(
                "qem_empad",
                dict(
                    format_identifier="numpy-float32",
                    format_name="NumPy float32",
                    microscope_metadata={
                        str(k): str(v) for k, v in scientific["source_metadata"].items()
                    },
                ),
            )
        else:
            header.update(interval=512, valid=encode_valid(valid))
        write_envelope(path, header, body)


def _read_array(handle, start: int, span: dict, dtype: str) -> np.ndarray:
    """Read one declared body array, refusing a file that ends inside it."""
    dtype = np.dtype(dtype)
    handle.seek(start + span["offset"])
    encoded = handle.read(span["count"] * dtype.itemsize)
    if len(encoded) != span["count"] * dtype.itemsize:
        raise ValueError("Truncated QEM array.")
    return np.frombuffer(encoded, dtype)


def load_array(path: str | Path) -> tuple[np.ndarray, dict]:
    """Decode original measurement bits after verifying the complete file."""
    validate_qem(path)
    header, start = read_envelope(path)
    shape, dtype = tuple(header["shape"]), np.dtype(header["dtype"])
    data = np.empty(shape, dtype=dtype)
    frames = data.reshape(-1, math.prod(shape[2:]))
    with open(path, "rb") as handle:
        for chunk in header["chunks"]:
            first, count = chunk["first"], chunk["scans"]
            if header["codec"] in (INTEGER_CODEC, SCALED_CODEC):
                payload, offsets, models = [
                    _read_array(handle, start, span, kind)
                    for span, kind in zip(chunk["arrays"], ("u1", "<u4", "u1"))
                ]
                if (
                    offsets[0] != 0
                    or offsets[-1] != payload.size
                    or np.any(offsets[1:] < offsets[:-1])
                ):
                    raise ValueError("Invalid QEM count offsets.")
                for stream, model in enumerate(models):
                    block, pixel = divmod(stream, frames.shape[1])
                    length = min(512, count - block * 512)
                    values = _decode_stream(
                        payload[
                            int(offsets[stream]) : int(offsets[stream + 1])
                        ].tobytes(),
                        int(model),
                        length,
                    )
                    if max(values) > np.iinfo(dtype).max:
                        raise ValueError(
                            "Decoded QEM count exceeds declared dtype; refusing truncation."
                        )
                    frames[
                        first + block * 512 : first + block * 512 + length, pixel
                    ] = values
            elif header["codec"] == FLOAT_CODEC:
                handle.seek(start + chunk["payload_offset"])
                payload = handle.read(chunk["payload_bytes"])
                offsets = np.frombuffer(handle.read(chunk["offset_bytes"]), "<u4")
                models = handle.read(chunk["model_bytes"])
                lanes = frames[first:first + count].view("<u2")
                for lane, model in enumerate(models):
                    lanes[:, lane] = _decode_stream(
                        payload[int(offsets[lane]):int(offsets[lane + 1])], model, count)
            else:
                raise NotImplementedError(f"Unsupported QEM codec {header['codec']!r}.")
    if header["codec"] == SCALED_CODEC:
        # Regional calibration, once per value: float32(code * scale + offset).
        codes, data = frames, np.empty(shape, np.float32)
        for region in header["intensity_calibration"]["regions"]:
            first, stop = region["first_frame"], region["stop_frame"]
            data.reshape(frames.shape)[first:stop] = (
                codes[first:stop] * region["scale"] + region["offset"]
            )
    if (
        header["codec"] == FLOAT_CODEC
        and hashlib.sha256(data.tobytes()).hexdigest() != header["logical_sha256"]
    ):
        raise ValueError("QEM decoded float32 checksum mismatch.")
    metadata = qem_metadata.effective_metadata(
        header.get("metadata", {}), header["scientific_metadata"]
    )
    metadata.update(
        scientific_metadata=header["scientific_metadata"],
        backend="cpu",
        representation="dense",
        source_path=str(path),
        dtype=data.dtype.name,
        scan_shape=shape[:2],
        detector_shape=shape[2:],
        file_counts_exact=header["codec"] != SCALED_CODEC,
    )
    if header["codec"] == SCALED_CODEC:
        metadata["precision"] = header["intensity_calibration"]
    elif header["codec"] == FLOAT_CODEC:
        metadata["qem_empad"] = header["empad"]
        metadata.setdefault("background_applied", False)
        metadata["background_applied_by_reader"] = False
    else:
        metadata["valid_pixels"] = decode_valid(header["valid"], shape[2:])
    return data, metadata
