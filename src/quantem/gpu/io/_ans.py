"""Self-contained exact count-ANS files and an explicit CPU reference writer.

The file is not HDF5. Little-endian typed sections follow a bounded JSON
manifest. Independent checksums detect corruption; authentication requires an
externally trusted whole-file digest. Backend runtimes own GPU decoding.
"""

import array
import hashlib
import json
import math
import os
import struct
import tempfile
from pathlib import Path

import numpy as np

from ._ans_contract import _validate_arrays

MAGIC = b"QGANS\0\1\0"
_HEADER = struct.Struct("<8sQQ")
_DATA_START = 65536
_LOWER = 1 << 23
_SECTION_DTYPES = {
    "payload": "u1",
    "offsets": "<u8",
    "model_ids": "<u4",
    "context_offsets": "<u4",
    "symbols": "<u2",
    "cumulative": "<u2",
    "frequencies": "<u2",
    "literal": "u1",
}


def _digest_file(stream) -> str:
    stream.seek(0)
    digest = hashlib.sha256()
    while block := stream.read(8 << 20):
        digest.update(block)
    return digest.hexdigest()


def _json_metadata(value):
    """Preserve NumPy metadata without arbitrary-object serialization."""
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(
        f"ANS metadata must contain JSON values, not {type(value).__name__}."
    )


def _no_duplicate_keys(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"Duplicate ANS manifest field: {key}.")
        result[key] = value
    return result


def _validate_scientific_metadata(metadata, shape, dtype):
    """Validate declared geometry without inventing acquisition calibration."""
    working = metadata.get("working_shape", list(shape))
    source = metadata.get("source_shape", list(shape))
    for name, value in (("working_shape", working), ("source_shape", source)):
        if (
            not isinstance(value, (list, tuple))
            or len(value) != 4
            or any(type(size) is not int or size < 1 for size in value)
        ):
            raise ValueError(
                f"ANS {name} must contain four positive integer dimensions."
            )
    if tuple(working) != tuple(shape):
        raise ValueError("ANS working_shape disagrees with stored counts.")
    if metadata.get("working_dtype", dtype) != dtype:
        raise ValueError("ANS working_dtype disagrees with stored native counts.")
    source_dtype = metadata.get("source_dtype", dtype)
    if source_dtype not in {"uint8", "uint16"}:
        raise ValueError("ANS source_dtype must declare native uint8 or uint16 counts.")
    scan_bin = metadata.get("scan_bin", 1)
    detector_bin = metadata.get("detector_bin", 1)
    if type(scan_bin) is not int or scan_bin != 1 or metadata.get("crop") is not None:
        raise ValueError("Count-ANS v1 provenance requires scan_bin=1 and crop=None.")
    if type(detector_bin) is not int or detector_bin < 1:
        raise ValueError("ANS detector_bin must be a positive integer.")
    expected = (*shape[:2], shape[2] * detector_bin, shape[3] * detector_bin)
    if tuple(source) != expected:
        raise ValueError("ANS source_shape, working_shape and detector_bin disagree.")
    for name, expected_bytes in (
        (
            "source_logical_tensor_bytes",
            math.prod(source) * np.dtype(source_dtype).itemsize,
        ),
        ("working_logical_tensor_bytes", math.prod(shape) * np.dtype(dtype).itemsize),
    ):
        if name in metadata and (
            type(metadata[name]) is not int or metadata[name] != expected_bytes
        ):
            raise ValueError(f"ANS {name} disagrees with declared shape/dtype.")
    if "lossless_exact" in metadata and type(metadata["lossless_exact"]) is not bool:
        raise ValueError(
            "ANS lossless_exact must be an explicit boolean, not a truthy token."
        )


def _reject_constant(value):
    raise ValueError(f"ANS metadata cannot contain nonfinite number {value}.")


def _stat_identity(stream):
    value = os.fstat(stream.fileno())
    return (
        value.st_dev,
        value.st_ino,
        value.st_size,
        value.st_mtime_ns,
        value.st_ctime_ns,
    )


class ANSFile:
    """Validated, read-only mapped exact count source.

    Parameters
    ----------
    path : str or Path
        Immutable self-contained ANS source.
    expected_sha256 : str, optional
        Independently recorded whole-file digest. Section hashes alone check
        integrity, not source authenticity. No arbitrary-source cold claim is
        implied by opening a previously audited file.

    Notes
    -----
    Mappings remain owned by this object. Keep it open while passing arrays to
    a runtime constructor. Constructors must copy them into owned GPU storage.
    Malformed entropy streams are checked by the decoder before publication.
    """

    def __init__(self, path, *, expected_sha256=None):
        self.path = Path(path)
        self.arrays = {}
        self._stream = self.path.open("rb")
        self._identity = _stat_identity(self._stream)
        try:
            self._open(expected_sha256)
        except BaseException:
            self.close()
            raise

    def _open(self, expected_sha256):
        stream = self._stream
        self.file_bytes = os.fstat(stream.fileno()).st_size
        header = stream.read(_HEADER.size)
        if len(header) != _HEADER.size:
            raise ValueError("Truncated ANS header.")
        magic, length, start = _HEADER.unpack(header)
        if (
            magic != MAGIC
            or start != _DATA_START
            or not 0 < length <= start - _HEADER.size
        ):
            raise ValueError("Unsupported or invalid ANS header.")
        if self.file_bytes < start:
            raise ValueError("Truncated ANS manifest area.")
        if expected_sha256 is not None:
            if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
                raise ValueError(
                    "expected_sha256 must be an independently recorded SHA-256 digest."
                )
            if _digest_file(stream) != expected_sha256.lower():
                raise ValueError(
                    "ANS source does not match the expected SHA-256 digest."
                )
            stream.seek(_HEADER.size)
        document = json.loads(
            stream.read(length),
            object_pairs_hook=_no_duplicate_keys,
            parse_constant=_reject_constant,
        )
        if not isinstance(document, dict):
            raise ValueError("ANS manifest must be an object.")  # noqa: TRY004
        logical_hash = document.get("logical_sha256")
        if (
            not isinstance(logical_hash, str)
            or len(logical_hash) != 64
            or any(value not in "0123456789abcdef" for value in logical_hash)
        ):
            raise ValueError(
                "ANS manifest requires a hexadecimal logical_sha256 digest."
            )
        if document.get("schema") != "quantem.gpu.count-ans.v1":
            raise ValueError("Unsupported ANS manifest schema.")
        if document.get("codec") != "block-column-rans-byte-v1":
            raise ValueError("Unsupported ANS entropy codec.")
        if document.get("order") != "scan_row,scan_column,detector_row,detector_column":
            raise ValueError("ANS dimension order must be explicit row/column order.")
        if document.get("dtype") not in {"uint8", "uint16"}:
            raise ValueError("ANS native count dtype must be uint8 or uint16.")
        if not isinstance(document.get("metadata"), dict):
            raise ValueError("ANS metadata must be a JSON object.")  # noqa: TRY004
        sections = document.get("sections")
        if not isinstance(sections, dict) or set(sections) != set(_SECTION_DTYPES):
            raise ValueError("ANS must contain exactly the declared typed sections.")
        cursor = start
        for name, dtype in _SECTION_DTYPES.items():
            section = sections[name]
            offset, count = section.get("offset"), section.get("count")
            if any(type(value) is not int or value < 0 for value in (offset, count)):
                raise ValueError(
                    "ANS section offsets and counts must be nonnegative integers."
                )
            size = count * np.dtype(dtype).itemsize
            aligned = (cursor + 7) // 8 * 8
            if (
                section.get("dtype") != dtype
                or offset != aligned
                or offset + size > self.file_bytes
            ):
                raise ValueError(f"Invalid ANS section bounds or dtype: {name}.")
            stream.seek(offset)
            digest = hashlib.sha256()
            remaining = size
            while remaining:
                chunk = stream.read(min(8 << 20, remaining))
                if not chunk:
                    raise ValueError("Truncated ANS section.")
                digest.update(chunk)
                remaining -= len(chunk)
            if digest.hexdigest() != section.get("sha256"):
                raise ValueError(f"ANS section checksum mismatch: {name}.")
            self.arrays[name] = (
                np.memmap(stream, dtype=dtype, mode="r", offset=offset, shape=(count,))
                if count
                else np.empty(0, dtype=dtype)
            )
            cursor = offset + size
        if cursor != self.file_bytes:
            raise ValueError("ANS contains undeclared trailing bytes.")
        self.shape, _ = _validate_arrays(
            shape=document.get("shape", ()),
            block_frames=document.get("block_frames"),
            scale=document.get("scale"),
            **self.arrays,
        )
        self.dtype = np.dtype(document["dtype"])
        if np.any(self.arrays["symbols"] > np.iinfo(self.dtype).max):
            raise ValueError("ANS entropy symbols exceed the declared native dtype.")
        self.manifest = document
        self.block_frames = int(document["block_frames"])
        self.scale = int(document["scale"])
        self.metadata = document["metadata"]
        _validate_scientific_metadata(self.metadata, self.shape, self.dtype.name)
        self.logical_nbytes = math.prod(self.shape) * self.dtype.itemsize
        self.encoded_nbytes = sum(value.nbytes for value in self.arrays.values())
        self.assert_unchanged()

    def assert_unchanged(self):
        """Reject changed file identity/timestamps during audit or upload.

        This cheap stat guard assumes an immutable input. Filesystem timestamps
        can coalesce nearby writes, so it is not content authentication against
        a concurrent writer. Section checksums are verified when opening.
        """
        if self._stream is None or _stat_identity(self._stream) != self._identity:
            raise ValueError(
                "ANS source changed during audit/upload; freeze it and retry."
            )

    def runtime_arguments(self):
        """Return the one backend-neutral array contract while mappings are open."""
        if self._stream is None:
            raise RuntimeError("ANS source is closed; reopen it before loading.")
        return dict(
            shape=self.shape,
            dtype=self.dtype.name,
            scale=self.scale,
            block_frames=self.block_frames,
            **self.arrays,
        )

    def decode_block_reference(self, block):
        """Decode a bounded block on CPU for independent parity, not a GPU fallback."""
        if self._stream is None:
            raise RuntimeError("ANS source is closed.")
        scans = self.shape[0] * self.shape[1]
        count = (scans + self.block_frames - 1) // self.block_frames
        if type(block) is not int or not 0 <= block < count:
            raise IndexError("ANS block index is outside the source.")
        frames = min(self.block_frames, scans - block * self.block_frames)
        pixels = self.shape[2] * self.shape[3]
        out = np.empty((frames, pixels), dtype=self.dtype)
        arrays = self.arrays
        for pixel in range(pixels):
            index = block * pixels + pixel
            first, stop = map(int, arrays["offsets"][index : index + 2])
            payload = arrays["payload"][first:stop]
            model = int(arrays["model_ids"][index])
            if arrays["literal"][model]:
                values = np.frombuffer(payload, dtype="<u2")
                if np.any(values > np.iinfo(self.dtype).max):
                    raise ValueError(
                        "ANS literal count exceeds the declared native dtype."
                    )
                out[:, pixel] = values
                continue
            first, stop = map(int, arrays["context_offsets"][model : model + 2])
            cumulative = arrays["cumulative"][first:stop]
            frequencies = arrays["frequencies"][first:stop]
            symbols = arrays["symbols"][first:stop]
            state = int.from_bytes(payload[:4], "little")
            if not _LOWER <= state < _LOWER * 256:
                raise ValueError("Invalid initial rANS state.")
            cursor = 4
            for frame in range(frames):
                slot = state & ((1 << self.scale) - 1)
                entry = int(np.searchsorted(cumulative, slot, side="right")) - 1
                out[frame, pixel] = symbols[entry]
                state = (
                    int(frequencies[entry]) * (state >> self.scale)
                    + slot
                    - int(cumulative[entry])
                )
                while state < _LOWER:
                    if cursor == len(payload):
                        raise ValueError("Truncated rANS stream.")
                    state = (state << 8) | int(payload[cursor])
                    cursor += 1
            if state != _LOWER or cursor != len(payload):
                raise ValueError("rANS terminal state or byte consumption is invalid.")
        return out.reshape(frames, *self.shape[2:])

    def close(self):
        """Release mapped source ownership; previously borrowed arrays expire."""
        for values in self.arrays.values():
            mapping = getattr(values, "_mmap", None)
            if mapping is not None:
                mapping.close()
        self.arrays.clear()
        if self._stream is not None:
            self._stream.close()
            self._stream = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()


def _encode_column(values, scale):
    """Return an exact entropy stream and deterministic finite probability model."""
    symbols, counts = np.unique(values, return_counts=True)
    slots = 1 << scale
    if len(symbols) > slots:
        return None
    # Positive frequencies are mandatory. Distribute remaining slots by largest
    # remainder with stable symbol-order tie breaks. This changes the code, not counts.
    numerator = counts.astype(np.int64) * (slots - len(symbols))
    frequencies = 1 + numerator // len(values)
    missing = slots - int(frequencies.sum())
    if missing:
        order = np.argsort(-(numerator % len(values)), kind="stable")
        frequencies[order[:missing]] += 1
    cumulative = np.cumsum(frequencies) - frequencies
    lookup = {
        int(value): (int(f), int(c))
        for value, f, c in zip(symbols, frequencies, cumulative)
    }
    state = _LOWER
    emitted = bytearray()
    for value in values[::-1]:
        frequency, start = lookup[int(value)]
        limit = ((_LOWER >> scale) << 8) * frequency
        while state >= limit:
            emitted.append(state & 255)
            state >>= 8
        state = ((state // frequency) << scale) + state % frequency + start
    payload = state.to_bytes(4, "little") + bytes(reversed(emitted))
    model = tuple(
        np.asarray(x, dtype="<u2") for x in (symbols, cumulative, frequencies)
    )
    return payload, model


def write_ans_reference(path, data, *, metadata=None, block_frames=256, scale=15):
    """Write exact uint8/uint16 counts transactionally using bounded CPU blocks.

    This is an explicitly requested reference encoder, not an accelerated
    first-load path. Input is a NumPy array or NumPy memory map with four
    row/column dimensions. Original excluded/saturated pixels are stored too.
    No output is overwritten. The writer retains encoded tables and indexes,
    but never makes a full dense copy of the source or full compressed payload.
    """
    if not isinstance(data, np.ndarray) or data.ndim != 4:
        raise TypeError(
            "The reference ANS writer requires a four-dimensional NumPy array or memory map."
        )
    if data.dtype not in (np.dtype("uint8"), np.dtype("uint16")) or any(
        size < 1 for size in data.shape
    ):
        raise ValueError(
            "ANS preserves a nonempty uint8/uint16 array; conversion or clipping is not implicit."
        )
    if type(block_frames) is not int or not 1 <= block_frames < 2**32:
        raise ValueError("block_frames must be a positive integer smaller than 2**32.")
    if type(scale) is not int or not 1 <= scale <= 15:
        raise ValueError("scale must be an integer from 1 to 15.")
    pixels = data.shape[2] * data.shape[3]
    if pixels >= 2**32 or math.prod(data.shape) >= 2**63:
        raise ValueError("Source geometry exceeds the ANS count range.")
    if block_frames * pixels * data.dtype.itemsize > 32 << 20:
        raise ValueError(
            "Reference ANS blocks must fit within 32 MiB; reduce block_frames."
        )
    metadata = json.loads(
        json.dumps(metadata or {}, default=_json_metadata, allow_nan=False)
    )
    if not isinstance(metadata, dict):
        raise TypeError("metadata must be a dictionary.")
    _validate_scientific_metadata(metadata, data.shape, data.dtype.name)
    path = Path(path)
    if path.exists():
        raise FileExistsError(
            f"ANS output already exists: {path}. Choose a new destination."
        )
    sections = {}
    tables = {
        "symbols": array.array("H"),
        "cumulative": array.array("H"),
        "frequencies": array.array("H"),
    }
    offsets, models = array.array("Q", [0]), array.array("I")
    contexts, literals = array.array("I", [0, 0]), bytearray([1])
    model_lookup = {}
    payload_digest, logical_digest = hashlib.sha256(), hashlib.sha256()
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".partial",
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            stream.write(bytes(_DATA_START))
            scans = data.shape[0] * data.shape[1]
            for first in range(0, scans, block_frames):
                indices = np.arange(first, min(first + block_frames, scans))
                rows, columns = np.divmod(indices, data.shape[1])
                block = np.ascontiguousarray(data[rows, columns])
                logical_digest.update(memoryview(block).cast("B"))
                for pixel in range(pixels):
                    detector_row, detector_column = divmod(pixel, data.shape[3])
                    values = block[:, detector_row, detector_column]
                    encoded = _encode_column(values, scale)
                    # Incompressible streams use a declared exact literal profile.
                    # Table overhead counts toward this conservative local choice.
                    if encoded is None or len(encoded[0]) + sum(
                        x.nbytes for x in encoded[1]
                    ) >= 2 * len(values):
                        payload, model = values.astype("<u2").tobytes(), 0
                    else:
                        payload, entries = encoded
                        key = b"".join(x.tobytes() for x in entries)
                        model = model_lookup.get(key)
                        if model is None:
                            model = len(literals)
                            if len(tables["symbols"]) + len(entries[0]) >= 2**32:
                                raise ValueError(
                                    "ANS model table exceeds uint32 indexing; use smaller files."
                                )
                            model_lookup[key] = model
                            literals.append(0)
                            for name, entries_array in zip(tables, entries):
                                tables[name].extend(map(int, entries_array))
                            contexts.append(len(tables["symbols"]))
                    stream.write(payload)
                    payload_digest.update(payload)
                    offsets.append(offsets[-1] + len(payload))
                    models.append(model)
            sections["payload"] = {
                "offset": _DATA_START,
                "count": offsets[-1],
                "dtype": "u1",
                "sha256": payload_digest.hexdigest(),
            }
            arrays = dict(
                offsets=offsets,
                model_ids=models,
                context_offsets=contexts,
                **tables,
                literal=literals,
            )
            for name, value in arrays.items():
                padding = (-stream.tell()) % 8
                stream.write(bytes(padding))
                values = np.asarray(value, dtype=_SECTION_DTYPES[name])
                raw = memoryview(values).cast("B")
                sections[name] = {
                    "offset": stream.tell(),
                    "count": values.size,
                    "dtype": _SECTION_DTYPES[name],
                    "sha256": hashlib.sha256(raw).hexdigest(),
                }
                stream.write(raw)
            document = {
                "schema": "quantem.gpu.count-ans.v1",
                "codec": "block-column-rans-byte-v1",
                "order": "scan_row,scan_column,detector_row,detector_column",
                "shape": list(data.shape),
                "dtype": data.dtype.name,
                "block_frames": block_frames,
                "scale": scale,
                "metadata": metadata,
                "sections": sections,
                "logical_sha256": logical_digest.hexdigest(),
                "encoder": "cpu-reference-v1",
            }
            raw = json.dumps(
                document, allow_nan=False, separators=(",", ":"), sort_keys=True
            ).encode()
            if len(raw) > _DATA_START - _HEADER.size:
                raise ValueError("ANS metadata exceeds the v1 64 KiB manifest limit.")
            stream.seek(0)
            stream.write(_HEADER.pack(MAGIC, len(raw), _DATA_START))
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        # Link publishes atomically without overwriting an existing destination,
        # including one created by another writer after the initial check.
        os.link(temporary, path)
        return path
    finally:
        if temporary is not None:
            temporary.unlink(missing_ok=True)
