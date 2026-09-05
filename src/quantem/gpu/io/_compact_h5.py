"""Reference parser and decoder for QuantEM compact 4D-STEM HDF5 files."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import struct
import tempfile
import zlib
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import numpy as np

_CONTAINER_MAGIC = b"QGPUH5\0\1"
_INDEX_MAGIC_V1 = b"QGIX\0\0\0\1"
_INDEX_MAGIC_V3 = b"QGIX\0\0\0\3"
_PRELUDE = struct.Struct("<8sIIII")
_INDEX_HEADER_V1 = struct.Struct("<8sIIIIIII")
_INDEX_HEADER_V3 = struct.Struct("<8sIIIIIIIII")
_SHARD_RECORD = struct.Struct("<QQQQQQQII32s")
_SCAN_TILE_V1 = 128
_SCAN_TILE_V3 = 32
_COMPACT_CHECKPOINT_TILES = 32
_WIDTHS_PER_WORD = 8
_DESCRIPTOR_WIDTH_BITS = 5
_DESCRIPTOR_OFFSET_BITS = 27
_MAX_SOURCE_WIDTH = 16
_HDF5_MAGIC = b"\x89HDF\r\n\x1a\n"


@dataclass(frozen=True)
class CompactH5Shard:
    """One directly addressable compressed shard in a compact HDF5 file."""

    payload_offset: int
    payload_bytes: int
    lengths_offset: int
    lengths_bytes: int
    widths_offset: int
    widths_bytes: int
    decoded_bytes: int
    descriptor_count: int
    chunk_count: int
    decoded_sha256: str


@dataclass(frozen=True)
class CompactH5PreparedDPCMoments:
    """Authenticated exact detector moments appended to one compact source."""

    file_offset: int
    file_bytes: int
    sha256: str
    working_logical_sha256: str
    working_dtype: str
    detector_mask_sha256: str
    scan_count: int
    selected_detector_pixels: int
    detector_columns: int
    total_bound: int
    row_moment_bound: int
    column_moment_bound: int
    narrow_integer: bool
    narrow_products: bool


@dataclass(frozen=True)
class CompactH5Index:
    """Validated immutable metadata for one compact 4D-STEM HDF5 source.

    The logical array is always addressed as
    ``(scan_row, scan_column, detector_row, detector_column)``. The parser
    validates both copies of the format metadata, but it does not decode the
    multi-gigabyte resident payload.

    Parameters
    ----------
    path
        Compact HDF5 file containing the QuantEM user-block index.

    Examples
    --------
    >>> index = CompactH5Index.from_file("scan-gpu-native.h5")
    >>> index.shape
    (512, 512, 192, 192)
    """

    path: Path
    file_bytes: int
    schema_version: int
    shape: tuple[int, int, int, int]
    scans_per_shard: int
    scan_tile: int
    header_encoding: int
    payload_chunk_bytes: int
    source_identity_sha256: str
    excluded_detector_pixels: tuple[int, ...]
    masked_detector_pixels_sha256: str | None
    masked_detector_raw_values: tuple[int, ...] | None
    prepared_dpc_moments: CompactH5PreparedDPCMoments | None
    shards: tuple[CompactH5Shard, ...]
    manifest: dict[str, object]

    @classmethod
    def from_file(cls, path: str | Path) -> CompactH5Index:
        """Read and validate the compact user-block index.

        Parameters
        ----------
        path
            Compact HDF5 source path.

        Returns
        -------
        CompactH5Index
            Validated binary and JSON metadata without payload decoding.

        Raises
        ------
        ValueError
            If either metadata copy is malformed, inconsistent, or outside the
            file.

        Examples
        --------
        >>> source = CompactH5Index.from_file("scan-gpu-native.h5")
        >>> source.scan_tile
        128
        """
        source = Path(path)
        file_bytes = source.stat().st_size
        with source.open("rb") as stream:
            prelude = _read_exact(stream, _PRELUDE.size, "container prelude")
            (
                container_magic,
                header_bytes,
                header_crc32,
                binary_offset,
                binary_bytes,
            ) = _PRELUDE.unpack(prelude)
            if container_magic != _CONTAINER_MAGIC:
                raise ValueError(
                    f"{source} has no QuantEM compact HDF5 user-block index."
                )
            if header_bytes == 0 or header_bytes > binary_offset - _PRELUDE.size:
                raise ValueError(
                    f"{source} has an invalid compact JSON header length "
                    f"{header_bytes}."
                )
            header = _read_exact(stream, header_bytes, "JSON header")
            if zlib.crc32(header) != header_crc32:
                raise ValueError(
                    f"{source} compact JSON header failed its CRC-32 check."
                )
            try:
                manifest = json.loads(header)
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"{source} compact JSON header is not valid UTF-8 JSON."
                ) from error
            if (
                binary_offset < _PRELUDE.size + header_bytes
                or binary_bytes < _INDEX_HEADER_V1.size + 4 + 32
                or binary_offset > file_bytes
                or binary_bytes > file_bytes - binary_offset
            ):
                raise ValueError(
                    f"{source} compact binary index range is outside the file."
                )
            stream.seek(binary_offset)
            binary = _read_exact(stream, binary_bytes, "binary index")
        return cls._from_parts(source, file_bytes, manifest, binary)

    @classmethod
    def _from_parts(
        cls,
        path: Path,
        file_bytes: int,
        manifest: dict[str, object],
        binary: bytes,
    ) -> CompactH5Index:
        if len(binary) < 8:
            raise ValueError(f"{path} compact binary index is truncated.")
        index_magic = binary[:8]
        if index_magic == _INDEX_MAGIC_V1:
            if len(binary) < _INDEX_HEADER_V1.size:
                raise ValueError(f"{path} compact v1 binary index is truncated.")
            (
                _,
                shard_count,
                payload_chunk_bytes,
                scan_rows,
                scan_columns,
                detector_rows,
                detector_columns,
                scans_per_shard,
            ) = _INDEX_HEADER_V1.unpack_from(binary)
            schema_version = 1
            scan_tile = _SCAN_TILE_V1
            header_encoding = 0
            cursor = _INDEX_HEADER_V1.size
        elif index_magic == _INDEX_MAGIC_V3:
            if len(binary) < _INDEX_HEADER_V3.size:
                raise ValueError(f"{path} compact v3 binary index is truncated.")
            (
                _,
                shard_count,
                reserved,
                scan_rows,
                scan_columns,
                detector_rows,
                detector_columns,
                scans_per_shard,
                scan_tile,
                header_encoding,
            ) = _INDEX_HEADER_V3.unpack_from(binary)
            if reserved != 0 or scan_tile != _SCAN_TILE_V3 or header_encoding != 1:
                raise ValueError(
                    f"{path} compact v3 requires reserved=0, scan_tile=32, and "
                    "compact header encoding 1."
                )
            payload_chunk_bytes = 0
            schema_version = 3
            cursor = _INDEX_HEADER_V3.size
        else:
            raise ValueError(
                f"{path} uses an unsupported compact binary index version."
            )
        shape = (scan_rows, scan_columns, detector_rows, detector_columns)
        if any(value <= 0 for value in shape):
            raise ValueError(f"{path} has invalid compact shape {shape}.")
        if shard_count <= 0 or scans_per_shard <= 0:
            raise ValueError(f"{path} requires positive compact shard and scan counts.")
        if scan_rows * scan_columns != shard_count * scans_per_shard:
            raise ValueError(
                f"{path} compact shards do not cover shape {shape}; got "
                f"{shard_count} shards of {scans_per_shard} scans."
            )
        if schema_version == 1 and payload_chunk_bytes != 128:
            raise ValueError(
                f"{path} compact v1 requires 128-byte raw LZ4 chunks; got "
                f"{payload_chunk_bytes}."
            )
        if schema_version == 3 and scans_per_shard % scan_tile:
            raise ValueError(
                f"{path} compact v3 requires complete {scan_tile}-scan tiles."
            )
        if cursor + 4 > len(binary):
            raise ValueError(f"{path} compact detector mask is truncated.")
        (mask_count,) = struct.unpack_from("<I", binary, cursor)
        cursor += 4
        detector_pixels = detector_rows * detector_columns
        if mask_count > detector_pixels or cursor + mask_count * 4 + 32 > len(binary):
            raise ValueError(f"{path} compact detector mask is invalid.")
        excluded = struct.unpack_from(f"<{mask_count}I", binary, cursor)
        cursor += mask_count * 4
        if len(set(excluded)) != mask_count or any(
            pixel >= detector_pixels for pixel in excluded
        ):
            raise ValueError(f"{path} compact detector mask is invalid.")
        source_identity = binary[cursor : cursor + 32].hex()
        cursor += 32
        shards = []
        for shard_index in range(shard_count):
            if cursor + _SHARD_RECORD.size > len(binary):
                raise ValueError(
                    f"{path} compact shard record {shard_index} is truncated."
                )
            values = _SHARD_RECORD.unpack_from(binary, cursor)
            cursor += _SHARD_RECORD.size
            shard = CompactH5Shard(*values[:-1], values[-1].hex())
            _validate_shard_record(
                path,
                file_bytes,
                shard_index,
                shard,
                detector_pixels,
                scans_per_shard,
                scan_tile,
                payload_chunk_bytes,
                schema_version,
            )
            shards.append(shard)
        if cursor != len(binary):
            raise ValueError(f"{path} compact binary index has trailing bytes.")
        _validate_nonoverlapping_ranges(path, shards)
        (
            masked_detector_pixels_sha256,
            masked_detector_raw_values,
        ) = _validate_manifest(
            path,
            manifest,
            shape,
            shard_count,
            scans_per_shard,
            payload_chunk_bytes,
            source_identity,
            excluded,
            schema_version,
            scan_tile,
            header_encoding,
        )
        prepared_dpc_moments = _validate_prepared_dpc_moments(
            path,
            file_bytes,
            manifest,
            shape,
            source_identity,
            tuple(excluded),
            schema_version,
            tuple(shards),
        )
        _validated_manifest_shards(path, manifest, tuple(shards))
        return cls(
            path=path,
            file_bytes=file_bytes,
            schema_version=schema_version,
            shape=shape,
            scans_per_shard=scans_per_shard,
            scan_tile=scan_tile,
            header_encoding=header_encoding,
            payload_chunk_bytes=payload_chunk_bytes,
            source_identity_sha256=source_identity,
            excluded_detector_pixels=tuple(excluded),
            masked_detector_pixels_sha256=masked_detector_pixels_sha256,
            masked_detector_raw_values=masked_detector_raw_values,
            prepared_dpc_moments=prepared_dpc_moments,
            shards=tuple(shards),
            manifest=manifest,
        )

    @property
    def detector_calibration(self) -> dict[str, object] | None:
        """Return validated detector calibration embedded for this source.

        The optional calibration is bound to ``source_identity_sha256`` and
        uses detector ``[row, column]`` coordinates. Readers that do not need
        calibrated presets can ignore it without changing packed values.
        """
        calibration = self.manifest.get("detector_calibration")
        return calibration if isinstance(calibration, dict) else None

    @property
    def logical_source_bytes(self) -> int:
        """Return the byte count of the complete logical uint16 source."""
        return int(np.prod(self.shape, dtype=np.uint64)) * 2

    @property
    def raw_reconstruction_available(self) -> bool:
        """Return whether every original raw uint16 sample is recoverable.

        Earlier v3 candidates omitted authenticated detector streams without
        retaining their constant raw values. They remain usable for explicitly
        mask-applied products, but are not portable raw-lossless sources. New
        exact-uint16 v1 producers can instead retain masked detector streams in
        the packed payload and declare that policy explicitly.
        """
        if self.schema_version == 1:
            return (
                not self.excluded_detector_pixels
                or self.manifest.get("masked_detector_payload_policy")
                == "retained_exactly_in_payload"
            )
        return self.schema_version == 3 and (
            not self.excluded_detector_pixels
            or (
                self.masked_detector_pixels_sha256 is not None
                and self.masked_detector_raw_values is not None
            )
        )

    def require_raw_reconstruction(self) -> None:
        """Fail unless every original raw uint16 sample is recoverable."""
        if not self.raw_reconstruction_available:
            raise ValueError(
                "Raw reconstruction requires retained masked payloads for QGIX "
                "v1 or ordered masked_detector_pixels_sha256 and "
                "masked_detector_raw_values for QGIX v3. This source is "
                "admissible only for explicit mask-applied products."
            )

    @property
    def resident_bytes(self) -> int:
        """Return descriptor plus decoded-payload bytes for every shard."""
        return sum(
            shard.descriptor_count * 4 + shard.decoded_bytes for shard in self.shards
        )


class CompactH5ReferenceDecoder:
    """Independent random-access decoder for compact scientific parity.

    This deliberately favors readable, bounded reference behavior over
    throughput. It reads only metadata and the raw-LZ4 chunks containing a
    requested sample, so parity checks never expand the logical 4D cube.

    Parameters
    ----------
    index
        Validated compact source metadata.

    Examples
    --------
    >>> index = CompactH5Index.from_file("scan-gpu-native.h5")
    >>> decoder = CompactH5ReferenceDecoder(index)
    >>> decoder.value(0, 0, 0, 0) >= 0
    True
    """

    def __init__(self, index: CompactH5Index) -> None:
        self.index = index
        self._metadata: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
        self._chunks: dict[tuple[int, int], bytes] = {}
        self._v3_metadata: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        self._v3_payloads: dict[int, np.memmap] = {}

    def value(
        self,
        scan_row: int,
        scan_column: int,
        detector_row: int,
        detector_column: int,
    ) -> int:
        """Decode one exact integer sample from the compact source.

        Parameters
        ----------
        scan_row, scan_column
            Zero-based scan coordinates.
        detector_row, detector_column
            Zero-based detector coordinates.
        Returns
        -------
        int
            Exact mask-applied working integer. Authenticated excluded detector
            pixels read as zero even when compact v1 retains their raw payloads.

        Examples
        --------
        >>> decoder.value(12, 34, 95, 96)
        3
        """
        scan, detector_pixel = self._flat_coordinates(
            scan_row,
            scan_column,
            detector_row,
            detector_column,
        )
        if detector_pixel in self.index.excluded_detector_pixels:
            return 0
        return self._stored_value(scan, detector_pixel)

    def _flat_coordinates(
        self,
        scan_row: int,
        scan_column: int,
        detector_row: int,
        detector_column: int,
    ) -> tuple[int, int]:
        """Validate public row-column coordinates and return flat indices."""
        scan_rows, scan_columns, detector_rows, detector_columns = self.index.shape
        if not 0 <= scan_row < scan_rows or not 0 <= scan_column < scan_columns:
            raise IndexError(
                f"Scan (row: {scan_row}, column: {scan_column}) is outside "
                f"shape ({scan_rows}, {scan_columns})."
            )
        if (
            not 0 <= detector_row < detector_rows
            or not 0 <= detector_column < detector_columns
        ):
            raise IndexError(
                f"Detector (row: {detector_row}, column: {detector_column}) is "
                f"outside shape ({detector_rows}, {detector_columns})."
            )
        detector_pixel = detector_row * detector_columns + detector_column
        scan = scan_row * scan_columns + scan_column
        return scan, detector_pixel

    def _stored_value(self, scan: int, detector_pixel: int) -> int:
        """Decode one packed sample without applying the detector mask."""
        shard_index, local_scan = divmod(scan, self.index.scans_per_shard)
        if self.index.schema_version == 3:
            return self._v3_value(shard_index, local_scan, detector_pixel)
        widths, word_offsets, compressed_offsets = self._shard_metadata(shard_index)
        tile_count = (
            self.index.scans_per_shard + self.index.scan_tile - 1
        ) // self.index.scan_tile
        tile = local_scan // self.index.scan_tile
        descriptor_index = detector_pixel * tile_count + tile
        width = int(widths[descriptor_index])
        if width == 0:
            return 0
        first_word = int(word_offsets[descriptor_index])
        bit = (local_scan % self.index.scan_tile) * width
        word_index = first_word + bit // 32
        shift = bit % 32
        byte_offset = word_index * 4
        raw = self._decoded_range(
            shard_index,
            byte_offset,
            8 if shift + width > 32 else 4,
            compressed_offsets,
        )
        low = int.from_bytes(raw[:4], "little")
        value = low >> shift
        if shift + width > 32:
            value |= int.from_bytes(raw[4:8], "little") << (32 - shift)
        return value & ((1 << width) - 1)

    def raw_value(
        self,
        scan_row: int,
        scan_column: int,
        detector_row: int,
        detector_column: int,
    ) -> int:
        """Return one exact raw uint16 sample from a lossless compact source.

        Unlike :meth:`value`, this reads retained QGIX v1 masked payloads or
        restores an omitted QGIX v3 stream from its producer-proven constant.
        Scientific products continue to use :meth:`value` and therefore keep
        excluded pixels at zero.
        """
        self.index.require_raw_reconstruction()
        scan, detector_pixel = self._flat_coordinates(
            scan_row,
            scan_column,
            detector_row,
            detector_column,
        )
        if (
            self.index.schema_version == 3
            and detector_pixel in self.index.excluded_detector_pixels
        ):
            raw_values = self.index.masked_detector_raw_values
            if raw_values is None:  # Guarded above; narrows the optional type.
                raise AssertionError("raw reconstruction admitted without constants")
            raw_index = self.index.excluded_detector_pixels.index(detector_pixel)
            return raw_values[raw_index]
        return self._stored_value(scan, detector_pixel)

    def validate_shard_metadata(self, shard_index: int) -> None:
        """Validate one shard's complete descriptor and chunk coverage.

        Parameters
        ----------
        shard_index
            Zero-based compact shard index.

        Examples
        --------
        >>> decoder.validate_shard_metadata(0)
        """
        if self.index.schema_version == 3:
            self._v3_shard_metadata(shard_index)
        else:
            self._shard_metadata(shard_index)

    def validate_shard_payload(self, shard_index: int) -> None:
        """Authenticate one v3 direct payload against its binary-index digest.

        This check deliberately remains separate from metadata parsing because
        hashing every real shard reads the complete resident payload. QGIX v1
        payload integrity is checked after raw-LZ4 decode instead.
        """
        if self.index.schema_version != 3:
            raise ValueError("Direct payload authentication applies only to QGIX v3.")
        try:
            shard = self.index.shards[shard_index]
        except IndexError as error:
            raise IndexError(
                f"Compact shard index {shard_index} is outside 0 through "
                f"{len(self.index.shards) - 1}."
            ) from error
        with self.index.path.open("rb") as stream:
            observed = _sha256_stream_range(
                stream, shard.payload_offset, shard.payload_bytes
            )
        if observed != shard.decoded_sha256:
            raise ValueError(
                f"Compact v3 shard {shard_index} direct payload failed SHA-256."
            )

    def selected_diffraction(self, scan_row: int, scan_column: int) -> np.ndarray:
        """Return one exact mask-applied diffraction pattern as uint16."""
        detector_rows, detector_columns = self.index.shape[2:]
        result = np.empty((detector_rows, detector_columns), dtype=np.uint16)
        for detector_row in range(detector_rows):
            for detector_column in range(detector_columns):
                result[detector_row, detector_column] = self.value(
                    scan_row, scan_column, detector_row, detector_column
                )
        return result

    def raw_diffraction(self, scan_row: int, scan_column: int) -> np.ndarray:
        """Return one exact raw diffraction pattern from portable QGIX v3."""
        detector_rows, detector_columns = self.index.shape[2:]
        result = np.empty((detector_rows, detector_columns), dtype=np.uint16)
        for detector_row in range(detector_rows):
            for detector_column in range(detector_columns):
                result[detector_row, detector_column] = self.raw_value(
                    scan_row, scan_column, detector_row, detector_column
                )
        return result

    def prepared_dpc_moment_values(
        self,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
        """Read authenticated exact total and detector-coordinate moments.

        Returns
        -------
        tuple of numpy.ndarray or None
            Total, detector-row moment, and detector-column moment arrays in
            scan-row order, or ``None`` when the source has no prepared moment
            extension.

        Examples
        --------
        >>> moments = decoder.prepared_dpc_moment_values()
        >>> moments is None or moments[0].dtype == np.dtype("uint64")
        True
        """
        prepared = self.index.prepared_dpc_moments
        if prepared is None:
            return None
        with self.index.path.open("rb") as stream:
            stream.seek(prepared.file_offset)
            payload = _read_exact(
                stream,
                prepared.file_bytes,
                "prepared DPC moments",
            )
        observed_sha256 = hashlib.sha256(payload).hexdigest()
        if observed_sha256 != prepared.sha256:
            raise ValueError(
                f"Prepared DPC SHA-256 is {observed_sha256}, expected "
                f"{prepared.sha256}."
            )
        words = np.frombuffer(payload, dtype="<u4").reshape(prepared.scan_count, 8)
        if np.any(words[:, 6:]):
            raise ValueError("Prepared DPC padding words must be zero.")
        total = words[:, 0].astype(np.uint64) | (
            words[:, 1].astype(np.uint64) << np.uint64(32)
        )
        row = words[:, 2].astype(np.uint64) | (
            words[:, 3].astype(np.uint64) << np.uint64(32)
        )
        column = words[:, 4].astype(np.uint64) | (
            words[:, 5].astype(np.uint64) << np.uint64(32)
        )
        if (
            np.any(total > prepared.total_bound)
            or np.any(row > prepared.row_moment_bound)
            or np.any(column > prepared.column_moment_bound)
        ):
            raise ValueError("Prepared DPC moments violate exact source bounds.")
        return total, row, column

    def detector_sum(self, detector_mask: np.ndarray) -> np.ndarray:
        """Return an exact uint32 scan image for one binary detector mask.

        The mask uses detector ``(row, column)`` order. Authenticated excluded
        pixels are always cleared before summation. A sum that cannot fit
        uint32 fails closed instead of wrapping.
        """
        detector_rows, detector_columns = self.index.shape[2:]
        mask = np.asarray(detector_mask)
        if mask.shape != (detector_rows, detector_columns):
            raise ValueError(
                f"Detector mask shape {mask.shape} does not match "
                f"({detector_rows}, {detector_columns})."
            )
        if not np.all((mask == 0) | (mask == 1)):
            raise ValueError(
                "Detector mask must contain only exact zero or one values."
            )
        selected = np.flatnonzero(mask.ravel())
        if self.index.excluded_detector_pixels:
            selected = selected[~np.isin(selected, self.index.excluded_detector_pixels)]
        scan_rows, scan_columns = self.index.shape[:2]
        result = np.empty((scan_rows, scan_columns), dtype=np.uint32)
        maximum = np.iinfo(np.uint32).max
        for scan_row in range(scan_rows):
            for scan_column in range(scan_columns):
                total = 0
                for pixel in selected:
                    detector_row, detector_column = divmod(int(pixel), detector_columns)
                    total += self.value(
                        scan_row, scan_column, detector_row, detector_column
                    )
                if total > maximum:
                    raise OverflowError(
                        "Exact detector sum exceeds uint32; use a wide-count path."
                    )
                result[scan_row, scan_column] = total
        return result

    def _v3_value(self, shard_index: int, local_scan: int, detector_pixel: int) -> int:
        widths, cumulative_words = self._v3_shard_metadata(shard_index)
        tile, scan_in_tile = divmod(local_scan, self.index.scan_tile)
        width = int(widths[detector_pixel, tile])
        if width == 0:
            return 0
        word_index = int(cumulative_words[detector_pixel, tile])
        bit = scan_in_tile * width
        word_index += bit // 32
        shift = bit % 32
        payload = self._v3_payload(shard_index)
        value = int(payload[word_index]) >> shift
        if shift + width > 32:
            value |= int(payload[word_index + 1]) << (32 - shift)
        return value & ((1 << width) - 1)

    def _v3_shard_metadata(self, shard_index: int) -> tuple[np.ndarray, np.ndarray]:
        if shard_index in self._v3_metadata:
            return self._v3_metadata[shard_index]
        try:
            shard = self.index.shards[shard_index]
        except IndexError as error:
            raise IndexError(
                f"Compact shard index {shard_index} is outside 0 through "
                f"{len(self.index.shards) - 1}."
            ) from error
        detector_pixels = self.index.shape[2] * self.index.shape[3]
        tile_count = self.index.scans_per_shard // self.index.scan_tile
        checkpoint_words = (
            tile_count + _COMPACT_CHECKPOINT_TILES - 1
        ) // _COMPACT_CHECKPOINT_TILES
        width_words = (tile_count + _WIDTHS_PER_WORD - 1) // _WIDTHS_PER_WORD
        header_words_per_pixel = checkpoint_words + width_words
        with self.index.path.open("rb") as stream:
            stream.seek(shard.widths_offset)
            header_bytes = _read_exact(stream, shard.widths_bytes, "compact v3 headers")
        headers = np.frombuffer(header_bytes, dtype="<u4")
        expected_words = detector_pixels * header_words_per_pixel
        if headers.size != expected_words:
            raise ValueError(
                f"Compact v3 shard {shard_index} has {headers.size} header "
                f"words, expected {expected_words}."
            )
        headers = headers.reshape(detector_pixels, header_words_per_pixel)
        widths = np.empty((detector_pixels, tile_count), dtype=np.uint8)
        packed_widths = headers[:, checkpoint_words:]
        for tile in range(tile_count):
            widths[:, tile] = (
                packed_widths[:, tile // _WIDTHS_PER_WORD]
                >> np.uint32((tile % _WIDTHS_PER_WORD) * 4)
            ) & np.uint32(15)
        unused_nibbles = width_words * _WIDTHS_PER_WORD - tile_count
        if unused_nibbles:
            used_nibbles = _WIDTHS_PER_WORD - unused_nibbles
            if np.any(packed_widths[:, -1] >> np.uint32(used_nibbles * 4)):
                raise ValueError(
                    f"Compact v3 shard {shard_index} has nonzero tail width nibbles."
                )
        maximum_width = int(widths.max(initial=0))
        if maximum_width > 8:
            raise ValueError(
                f"Compact v3 shard {shard_index} requires width {maximum_width}; "
                "the direct uint8 contract permits at most 8."
            )
        if self.index.excluded_detector_pixels and np.any(
            widths[np.asarray(self.index.excluded_detector_pixels)] != 0
        ):
            raise ValueError(
                f"Compact v3 shard {shard_index} retains payload widths for an "
                "authenticated excluded detector pixel."
            )
        cumulative_words = np.zeros((detector_pixels, tile_count + 1), dtype=np.uint64)
        np.cumsum(widths, axis=1, dtype=np.uint64, out=cumulative_words[:, 1:])
        for checkpoint in range(1, checkpoint_words):
            expected = cumulative_words[:, checkpoint * _COMPACT_CHECKPOINT_TILES]
            if not np.array_equal(headers[:, checkpoint].astype(np.uint64), expected):
                raise ValueError(
                    f"Compact v3 shard {shard_index} checkpoint {checkpoint} "
                    "does not match its width nibbles."
                )
        pixel_words = cumulative_words[:, -1]
        expected_bases = np.zeros(detector_pixels, dtype=np.uint64)
        if detector_pixels > 1:
            np.cumsum(pixel_words[:-1], dtype=np.uint64, out=expected_bases[1:])
        bases = headers[:, 0].astype(np.uint64)
        if not np.array_equal(bases, expected_bases):
            raise ValueError(
                f"Compact v3 shard {shard_index} pixel bases do not exactly "
                "cover the direct payload."
            )
        payload_words = shard.decoded_bytes // 4
        if int(expected_bases[-1] + pixel_words[-1]) != payload_words:
            raise ValueError(
                f"Compact v3 shard {shard_index} headers do not cover its "
                "direct payload."
            )
        cumulative_words += bases[:, None]
        result = widths, cumulative_words
        self._v3_metadata[shard_index] = result
        return result

    def _v3_payload(self, shard_index: int) -> np.memmap:
        if shard_index not in self._v3_payloads:
            shard = self.index.shards[shard_index]
            self._v3_payloads[shard_index] = np.memmap(
                self.index.path,
                dtype="<u4",
                mode="r",
                offset=shard.payload_offset,
                shape=(shard.decoded_bytes // 4,),
            )
        return self._v3_payloads[shard_index]

    def _shard_metadata(
        self, shard_index: int
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if shard_index in self._metadata:
            return self._metadata[shard_index]
        try:
            shard = self.index.shards[shard_index]
        except IndexError as error:
            raise IndexError(
                f"Compact shard index {shard_index} is outside 0 through "
                f"{len(self.index.shards) - 1}."
            ) from error
        with self.index.path.open("rb") as stream:
            stream.seek(shard.widths_offset)
            widths_bytes = _read_exact(stream, shard.widths_bytes, "descriptor widths")
            stream.seek(shard.lengths_offset)
            lengths_bytes = _read_exact(stream, shard.lengths_bytes, "chunk lengths")
        widths = np.frombuffer(widths_bytes, dtype=np.uint8)
        if widths.size != shard.descriptor_count:
            raise ValueError(
                f"Compact shard {shard_index} descriptor-width count changed."
            )
        maximum_width = int(widths.max(initial=0))
        if maximum_width > _MAX_SOURCE_WIDTH:
            raise ValueError(
                f"Compact shard {shard_index} requires {maximum_width} bits per "
                f"sample, beyond exact uint16 width 16."
            )
        working_dtype = self.index.manifest.get("working_dtype")
        if working_dtype == "uint8" and maximum_width > 8:
            tile_count = (
                self.index.scans_per_shard + self.index.scan_tile - 1
            ) // self.index.scan_tile
            wide_pixels = {
                int(index) // tile_count for index in np.flatnonzero(widths > 8)
            }
            unexpected = wide_pixels.difference(self.index.excluded_detector_pixels)
            if unexpected:
                raise ValueError(
                    f"Compact shard {shard_index} requires more than eight bits "
                    f"for nonexcluded detector pixels {sorted(unexpected)[:8]}, "
                    "but its legacy manifest declares uint8 working values."
                )
        word_offsets = np.empty(widths.size + 1, dtype=np.uint64)
        word_offsets[0] = 0
        np.cumsum(widths, dtype=np.uint64, out=word_offsets[1:])
        word_offsets *= 4
        if int(word_offsets[-1]) * 4 != shard.decoded_bytes:
            raise ValueError(
                f"Compact shard {shard_index} widths do not cover its decoded payload."
            )
        lengths = np.frombuffer(lengths_bytes, dtype=np.uint8).astype(np.uint64) + 1
        if lengths.size != shard.chunk_count:
            raise ValueError(f"Compact shard {shard_index} chunk count changed.")
        compressed_offsets = np.empty(lengths.size + 1, dtype=np.uint64)
        compressed_offsets[0] = 0
        np.cumsum(lengths, dtype=np.uint64, out=compressed_offsets[1:])
        if int(compressed_offsets[-1]) != shard.payload_bytes:
            raise ValueError(
                f"Compact shard {shard_index} chunk lengths do not cover the "
                "compressed payload."
            )
        result = widths, word_offsets, compressed_offsets
        self._metadata[shard_index] = result
        return result

    def _decoded_range(
        self,
        shard_index: int,
        byte_offset: int,
        byte_count: int,
        compressed_offsets: np.ndarray,
    ) -> bytes:
        chunk_bytes = self.index.payload_chunk_bytes
        first_chunk = byte_offset // chunk_bytes
        last_chunk = (byte_offset + byte_count - 1) // chunk_bytes
        joined = b"".join(
            self._decoded_chunk(shard_index, chunk, compressed_offsets)
            for chunk in range(first_chunk, last_chunk + 1)
        )
        local_offset = byte_offset - first_chunk * chunk_bytes
        return joined[local_offset : local_offset + byte_count]

    def _decoded_chunk(
        self,
        shard_index: int,
        chunk_index: int,
        compressed_offsets: np.ndarray,
    ) -> bytes:
        key = shard_index, chunk_index
        if key in self._chunks:
            return self._chunks[key]
        shard = self.index.shards[shard_index]
        input_start = int(compressed_offsets[chunk_index])
        input_stop = int(compressed_offsets[chunk_index + 1])
        output_bytes = min(
            self.index.payload_chunk_bytes,
            shard.decoded_bytes - chunk_index * self.index.payload_chunk_bytes,
        )
        with self.index.path.open("rb") as stream:
            stream.seek(shard.payload_offset + input_start)
            encoded = _read_exact(
                stream, input_stop - input_start, "compressed payload chunk"
            )
        decoded = _decode_raw_lz4_block(encoded, output_bytes)
        self._chunks[key] = decoded
        return decoded


def _decode_raw_lz4_block(source: bytes, expected_bytes: int) -> bytes:
    """Decode one independent raw LZ4 block without native dependencies."""
    output = bytearray()
    position = 0
    while position < len(source) and len(output) < expected_bytes:
        token = source[position]
        position += 1
        literal_count = token >> 4
        if literal_count == 15:
            extension = 255
            while extension == 255:
                if position >= len(source):
                    raise ValueError("Raw LZ4 literal length is truncated.")
                extension = source[position]
                position += 1
                literal_count += extension
        literal_stop = position + literal_count
        if literal_stop > len(source) or len(output) + literal_count > expected_bytes:
            raise ValueError("Raw LZ4 literals exceed the input or output block.")
        output.extend(source[position:literal_stop])
        position = literal_stop
        if len(output) == expected_bytes or position == len(source):
            break
        if position + 2 > len(source):
            raise ValueError("Raw LZ4 match offset is truncated.")
        match_offset = int.from_bytes(source[position : position + 2], "little")
        position += 2
        if match_offset == 0 or match_offset > len(output):
            raise ValueError(f"Raw LZ4 match offset {match_offset} is invalid.")
        match_count = token & 15
        if match_count == 15:
            extension = 255
            while extension == 255:
                if position >= len(source):
                    raise ValueError("Raw LZ4 match length is truncated.")
                extension = source[position]
                position += 1
                match_count += extension
        match_count += 4
        if len(output) + match_count > expected_bytes:
            raise ValueError("Raw LZ4 match exceeds the output block.")
        for _ in range(match_count):
            output.append(output[-match_offset])
    if position != len(source) or len(output) != expected_bytes:
        raise ValueError(
            f"Raw LZ4 block decoded {len(output)} of {expected_bytes} bytes and "
            f"consumed {position} of {len(source)} bytes."
        )
    return bytes(output)


def prepare_compact_h5_metadata_copy(
    source: str | Path,
    destination: str | Path,
    *,
    detector_calibration: dict[str, object] | None = None,
    add_encoded_envelope_hashes: bool = True,
    masked_detector_raw_values: tuple[int, ...] | None = None,
    expected_source_sha256: str | None = None,
    working_logical_sha256: str | None = None,
    prepared_dpc_moments: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None,
) -> CompactH5Index:
    """Create a prepared compact-H5 copy with source-bound reusable metadata.

    The source is never modified. On filesystems that expose ``os.clonefile``,
    the initial copy is copy-on-write; other platforms use a normal file copy.
    The destination is published atomically only after its rewritten user block
    validates through :class:`CompactH5Index`.

    Parameters
    ----------
    source, destination
        Existing compact source and a new output path. The destination must not
        already exist.
    detector_calibration
        Optional detector-calibration object. Its source identity is filled
        from the authenticated compact index and cannot be overridden.
    add_encoded_envelope_hashes
        Add one SHA-256 for each shard's exact compressed-payload, length, and
        width envelope. Backends can authenticate prepared reopen input without
        reading the multi-gigabyte decoded payload back from the GPU.
    masked_detector_raw_values
        For QGIX v3 only, producer-proven constant raw uint16 values aligned
        with the ordered binary-index exclusions. Required when v3 omits any
        detector stream.
    expected_source_sha256
        Expected whole-file SHA-256 of the immutable input. It is required for
        QGIX v3 and optional for an already authenticated QGIX v1 source.
    working_logical_sha256
        For exact-uint16 QGIX v1 only, the independently verified SHA-256 of
        mask-applied working values in scan-major order. Required when adding
        prepared moments.
    prepared_dpc_moments
        For exact-uint16 QGIX v1 only, exact total, detector-row moment, and
        detector-column moment arrays in scan-row order. The copy stores and
        authenticates them inside the HDF5 artifact.

    Returns
    -------
    CompactH5Index
        The fully re-read and validated destination index.
    """
    source_path = Path(source).resolve()
    destination_path = Path(destination).resolve()
    if source_path == destination_path:
        raise ValueError("Prepared compact metadata must be written to a new file.")
    if destination_path.exists():
        raise FileExistsError(
            f"Refusing to replace existing prepared compact file {destination_path}."
        )
    index = CompactH5Index.from_file(source_path)
    manifest = json.loads(json.dumps(index.manifest))
    records = _validated_manifest_shards(source_path, manifest, index.shards)
    if expected_source_sha256 is not None:
        if (
            not isinstance(expected_source_sha256, str)
            or len(expected_source_sha256) != 64
            or any(
                character not in "0123456789abcdef"
                for character in expected_source_sha256
            )
        ):
            raise ValueError(
                "A compact metadata copy requires a lowercase expected_source_sha256."
            )
        observed_source_sha256 = _sha256_path(source_path)
        if observed_source_sha256 != expected_source_sha256:
            raise ValueError(
                f"Compact input SHA-256 is {observed_source_sha256}, expected "
                f"{expected_source_sha256}."
            )
    if index.schema_version == 3:
        if expected_source_sha256 is None:
            raise ValueError(
                "A QGIX v3 metadata copy requires the immutable input's "
                "lowercase expected_source_sha256."
            )
        if masked_detector_raw_values is None:
            if index.excluded_detector_pixels:
                raise ValueError(
                    "A portable QGIX v3 copy requires one producer-proven raw "
                    "constant for every omitted detector pixel."
                )
            masked_detector_raw_values = ()
        if len(masked_detector_raw_values) != len(
            index.excluded_detector_pixels
        ) or any(
            type(value) is not int or not 0 <= value <= 65535
            for value in masked_detector_raw_values
        ):
            raise ValueError(
                "QGIX v3 raw constants must be exact uint16 values aligned "
                "one-to-one with the ordered detector exclusions."
            )
        mask_bytes = struct.pack(
            f"<{len(index.excluded_detector_pixels)}I",
            *index.excluded_detector_pixels,
        )
        manifest["masked_detector_pixels_sha256"] = hashlib.sha256(
            mask_bytes
        ).hexdigest()
        manifest["masked_detector_raw_values"] = list(masked_detector_raw_values)
    else:
        if working_logical_sha256 is not None:
            manifest["working_logical_sha256"] = _require_sha256(
                working_logical_sha256,
                "working_logical_sha256",
            )
        if masked_detector_raw_values is not None:
            if len(masked_detector_raw_values) != len(
                index.excluded_detector_pixels
            ) or any(
                type(value) is not int or not 0 <= value <= 65535
                for value in masked_detector_raw_values
            ):
                raise ValueError(
                    "QGIX v1 masked_detector_raw_values must contain one exact "
                    "uint16 value for each ordered detector exclusion."
                )
            mask_bytes = struct.pack(
                f"<{len(index.excluded_detector_pixels)}I",
                *index.excluded_detector_pixels,
            )
            manifest["masked_detector_pixels_sha256"] = hashlib.sha256(
                mask_bytes
            ).hexdigest()
            manifest["masked_detector_raw_values"] = list(masked_detector_raw_values)
        if prepared_dpc_moments is not None:
            if index.manifest.get("working_dtype") != "uint16":
                raise ValueError(
                    "QGIX v1 prepared moments require uint16 working data."
                )
            if not index.raw_reconstruction_available:
                raise ValueError(
                    "QGIX v1 prepared moments require an exact retained raw payload."
                )
            if working_logical_sha256 is None:
                raise ValueError(
                    "QGIX v1 prepared moments require working_logical_sha256."
                )
            if index.excluded_detector_pixels and masked_detector_raw_values is None:
                raise ValueError(
                    "QGIX v1 prepared moments require recoverable masked-original "
                    "metadata for every detector exclusion."
                )
    if index.schema_version == 3 and (
        working_logical_sha256 is not None or prepared_dpc_moments is not None
    ):
        raise ValueError(
            "QGIX v1 working identity and prepared moments cannot be applied "
            "to a QGIX v3 file."
        )
    if detector_calibration is not None:
        calibration = dict(detector_calibration)
        calibration["schema"] = "quantem.gpu.detector-calibration/v1"
        calibration["source_identity_sha256"] = index.source_identity_sha256
        manifest["detector_calibration"] = calibration
    if add_encoded_envelope_hashes and index.schema_version == 1:
        with source_path.open("rb") as stream:
            for record, shard in zip(records, index.shards, strict=True):
                range_start = min(
                    shard.payload_offset,
                    shard.lengths_offset,
                    shard.widths_offset,
                )
                range_end = max(
                    shard.payload_offset + shard.payload_bytes,
                    shard.lengths_offset + shard.lengths_bytes,
                    shard.widths_offset + shard.widths_bytes,
                )
                record["encoded_envelope_sha256"] = _sha256_stream_range(
                    stream,
                    range_start,
                    range_end - range_start,
                )

    destination_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination_path.name}.",
        suffix=".part",
        dir=destination_path.parent,
    )
    os.close(descriptor)
    temporary_path = Path(temporary_name)
    temporary_path.unlink()
    try:
        clonefile = getattr(os, "clonefile", None)
        if callable(clonefile):
            clonefile(source_path, temporary_path)
        else:
            shutil.copy2(source_path, temporary_path)
        if prepared_dpc_moments is not None:
            total, row_moment, column_moment = (
                np.asarray(values) for values in prepared_dpc_moments
            )
            scan_count = index.shape[0] * index.shape[1]
            arrays = (total, row_moment, column_moment)
            if any(
                values.ndim != 1
                or values.shape[0] != scan_count
                or values.dtype.kind not in "ui"
                for values in arrays
            ):
                raise ValueError(
                    "Prepared DPC moments must be one-dimensional integer arrays "
                    f"with {scan_count} scan-row-order values."
                )
            if any(
                values.dtype.kind == "i" and np.any(values < 0) for values in arrays
            ):
                raise ValueError("Prepared DPC moments cannot contain negative values.")
            total = total.astype(np.uint64, copy=False)
            row_moment = row_moment.astype(np.uint64, copy=False)
            column_moment = column_moment.astype(np.uint64, copy=False)
            detector_pixels = index.shape[2] * index.shape[3]
            selected_detector_pixels = detector_pixels - len(
                index.excluded_detector_pixels
            )
            excluded = set(index.excluded_detector_pixels)
            maximum_value = int(np.iinfo(np.uint16).max)
            total_bound = selected_detector_pixels * maximum_value
            row_moment_bound = maximum_value * sum(
                pixel // index.shape[3]
                for pixel in range(detector_pixels)
                if pixel not in excluded
            )
            column_moment_bound = maximum_value * sum(
                pixel % index.shape[3]
                for pixel in range(detector_pixels)
                if pixel not in excluded
            )
            if (
                np.any(total > total_bound)
                or np.any(row_moment > row_moment_bound)
                or np.any(column_moment > column_moment_bound)
            ):
                raise ValueError("Prepared DPC moments violate exact uint16 bounds.")
            words = np.zeros((scan_count, 8), dtype="<u4")
            words[:, 0] = (total & np.uint64(0xFFFF_FFFF)).astype(np.uint32)
            words[:, 1] = (total >> np.uint64(32)).astype(np.uint32)
            words[:, 2] = (row_moment & np.uint64(0xFFFF_FFFF)).astype(np.uint32)
            words[:, 3] = (row_moment >> np.uint64(32)).astype(np.uint32)
            words[:, 4] = (column_moment & np.uint64(0xFFFF_FFFF)).astype(np.uint32)
            words[:, 5] = (column_moment >> np.uint64(32)).astype(np.uint32)
            try:
                import h5py
            except ImportError as error:
                raise RuntimeError(
                    "Preparing an exact moment payload requires h5py."
                ) from error
            # Compressed sources can end at any byte offset. Align newly
            # allocated datasets without changing existing payload ranges.
            with h5py.File(
                temporary_path, "r+", alignment_threshold=1, alignment_interval=4
            ) as handle:
                group = handle.require_group("quantem_gpu").require_group("prepared")
                dataset_name = "dpc_moments_u32_v2"
                if dataset_name in group:
                    raise ValueError(
                        f"{source_path} already contains {group.name}/{dataset_name}."
                    )
                dataset = group.create_dataset(
                    dataset_name,
                    data=words,
                    dtype="<u4",
                    chunks=None,
                )
                handle.flush()
                file_offset = dataset.id.get_offset()
                if file_offset is None or dataset.chunks is not None:
                    raise RuntimeError(
                        "Prepared DPC moments are not a contiguous HDF5 range."
                    )
            prepared_bytes = int(words.nbytes)
            if file_offset % np.dtype("<u4").itemsize:
                raise RuntimeError("Prepared DPC moments are not uint32 aligned.")
            prepared_payload = words.tobytes(order="C")
            manifest["prepared_dpc_moments"] = {
                "schema": "quantem.gpu.prepared-dpc-moments/v2",
                "source_identity_sha256": index.source_identity_sha256,
                "working_logical_sha256": manifest["working_logical_sha256"],
                "working_dtype": "uint16",
                "detector_mask_sha256": manifest["detector_mask_sha256"],
                "detector_selection": "all-nonexcluded-v1",
                "scan_count": scan_count,
                "selected_detector_pixels": selected_detector_pixels,
                "detector_columns": index.shape[3],
                "maximum_value": maximum_value,
                "dtype": "little-endian-u32",
                "word_order": "little-endian-u32-pairs",
                "words_per_scan": 8,
                "layout": [
                    "total_lo",
                    "total_hi",
                    "row_lo",
                    "row_hi",
                    "column_lo",
                    "column_hi",
                    "padding_0",
                    "padding_1",
                ],
                "file_offset": int(file_offset),
                "file_bytes": prepared_bytes,
                "sha256": hashlib.sha256(prepared_payload).hexdigest(),
                "total_bound": str(total_bound),
                "row_moment_bound": str(row_moment_bound),
                "column_moment_bound": str(column_moment_bound),
                "narrow_integer": total_bound <= np.iinfo(np.uint32).max,
                "narrow_products": max(row_moment_bound, column_moment_bound)
                <= np.iinfo(np.uint32).max,
            }

        header = json.dumps(
            manifest,
            separators=(",", ":"),
            sort_keys=True,
        ).encode("utf-8")
        with source_path.open("rb") as stream:
            prelude = _read_exact(stream, _PRELUDE.size, "container prelude")
            _, _, _, binary_offset, binary_bytes = _PRELUDE.unpack(prelude)
            stream.seek(binary_offset)
            binary = _read_exact(stream, binary_bytes, "binary index")
            stream.seek(0)
            prefix = stream.read(max(1 << 20, binary_offset + binary_bytes))
        hdf5_offset = prefix.find(_HDF5_MAGIC)
        user_block_bytes = (
            hdf5_offset
            if hdf5_offset >= 0
            else min(
                offset
                for shard in index.shards
                for offset, byte_count in (
                    (shard.payload_offset, shard.payload_bytes),
                    (shard.lengths_offset, shard.lengths_bytes),
                    (shard.widths_offset, shard.widths_bytes),
                )
                if byte_count
            )
        )
        new_binary_offset = (_PRELUDE.size + len(header) + 7) & ~7
        if new_binary_offset + len(binary) > user_block_bytes:
            raise ValueError(
                f"Prepared compact metadata needs {new_binary_offset + len(binary)} "
                f"user-block bytes, but {source_path} reserves {user_block_bytes}."
            )
        user_block = bytearray(user_block_bytes)
        user_block[: _PRELUDE.size] = _PRELUDE.pack(
            _CONTAINER_MAGIC,
            len(header),
            zlib.crc32(header),
            new_binary_offset,
            len(binary),
        )
        user_block[_PRELUDE.size : _PRELUDE.size + len(header)] = header
        user_block[new_binary_offset : new_binary_offset + len(binary)] = binary
        with temporary_path.open("r+b") as stream:
            stream.seek(0)
            stream.write(user_block)
            stream.flush()
            os.fsync(stream.fileno())
        prepared = CompactH5Index.from_file(temporary_path)
        if index.schema_version == 3:
            prepared.require_raw_reconstruction()
        if prepared_dpc_moments is not None:
            decoded_moments = CompactH5ReferenceDecoder(
                prepared
            ).prepared_dpc_moment_values()
            if decoded_moments is None or any(
                not np.array_equal(observed, expected)
                for observed, expected in zip(
                    decoded_moments,
                    prepared_dpc_moments,
                    strict=True,
                )
            ):
                raise ValueError(
                    "Prepared DPC moments changed before atomic publication."
                )
        os.replace(temporary_path, destination_path)
    finally:
        temporary_path.unlink(missing_ok=True)
    return CompactH5Index.from_file(destination_path)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(16 << 20):
            digest.update(block)
    return digest.hexdigest()


def _validated_manifest_shards(
    path: Path,
    manifest: dict[str, object],
    shards: tuple[CompactH5Shard, ...],
) -> list[dict[str, object]]:
    if manifest.get("schema") == "quantem.gpu.packed-detector-h5/v3":
        if "shards" in manifest:
            raise ValueError(
                f"{path} compact v3 embeds an unsupported JSON shard table."
            )
        return []
    records = manifest.get("shards")
    if not isinstance(records, list) or len(records) != len(shards):
        raise ValueError(f"{path} compact JSON shard list is incomplete.")
    validated: list[dict[str, object]] = []
    for ordinal, (record, shard) in enumerate(zip(records, shards, strict=True)):
        if not isinstance(record, dict):
            raise ValueError(  # noqa: TRY004 - malformed serialized input
                f"{path} compact JSON shard {ordinal} is not an object."
            )
        expected = {
            "ordinal": ordinal,
            "payload_file_offset": shard.payload_offset,
            "payload_file_bytes": shard.payload_bytes,
            "lengths_file_offset": shard.lengths_offset,
            "lengths_file_bytes": shard.lengths_bytes,
            "descriptor_widths_file_offset": shard.widths_offset,
            "descriptor_widths_file_bytes": shard.widths_bytes,
            "payload_decoded_bytes": shard.decoded_bytes,
            "descriptor_count": shard.descriptor_count,
            "payload_chunk_count": shard.chunk_count,
            "payload_decoded_sha256": shard.decoded_sha256,
        }
        mismatch = {
            key: (record.get(key), value)
            for key, value in expected.items()
            if record.get(key) != value
        }
        if mismatch:
            raise ValueError(
                f"{path} compact JSON shard {ordinal} disagrees with its binary "
                f"record: {mismatch}."
            )
        encoded_hash = record.get("encoded_envelope_sha256")
        if encoded_hash is not None and (
            not isinstance(encoded_hash, str)
            or len(encoded_hash) != 64
            or any(character not in "0123456789abcdef" for character in encoded_hash)
        ):
            raise ValueError(
                f"{path} compact JSON shard {ordinal} encoded-envelope SHA-256 "
                "is invalid."
            )
        validated.append(record)
    return validated


def _sha256_stream_range(stream, offset: int, byte_count: int) -> str:
    digest = hashlib.sha256()
    stream.seek(offset)
    remaining = byte_count
    while remaining:
        block = stream.read(min(8 << 20, remaining))
        if not block:
            raise ValueError(
                f"Compact HDF5 encoded envelope ended before {byte_count} bytes."
            )
        digest.update(block)
        remaining -= len(block)
    return digest.hexdigest()


def _read_exact(stream, byte_count: int, label: str) -> bytes:
    data = stream.read(byte_count)
    if len(data) != byte_count:
        raise ValueError(f"Compact HDF5 {label} ended before {byte_count} bytes.")
    return data


def _validate_shard_record(
    path: Path,
    file_bytes: int,
    shard_index: int,
    shard: CompactH5Shard,
    detector_pixels: int,
    scans_per_shard: int,
    scan_tile: int,
    payload_chunk_bytes: int,
    schema_version: int,
) -> None:
    tile_count = (scans_per_shard + scan_tile - 1) // scan_tile
    if schema_version == 3:
        checkpoint_words = (
            tile_count + _COMPACT_CHECKPOINT_TILES - 1
        ) // _COMPACT_CHECKPOINT_TILES
        width_words = (tile_count + _WIDTHS_PER_WORD - 1) // _WIDTHS_PER_WORD
        expected_headers = detector_pixels * (checkpoint_words + width_words)
        if shard.descriptor_count != expected_headers:
            raise ValueError(
                f"{path} compact v3 shard {shard_index} has "
                f"{shard.descriptor_count} header words, expected "
                f"{expected_headers}."
            )
        if (
            shard.lengths_offset != 0
            or shard.lengths_bytes != 0
            or shard.chunk_count != 0
        ):
            raise ValueError(
                f"{path} compact v3 shard {shard_index} unexpectedly contains "
                "raw-LZ4 metadata."
            )
        if shard.widths_bytes != shard.descriptor_count * 4:
            raise ValueError(
                f"{path} compact v3 shard {shard_index} header byte count "
                "does not match its header words."
            )
        if (
            shard.payload_bytes == 0
            or shard.payload_bytes != shard.decoded_bytes
            or shard.payload_bytes % 4
        ):
            raise ValueError(
                f"{path} compact v3 shard {shard_index} direct payload is not "
                "a nonempty uint32 range."
            )
        if shard.decoded_bytes // 4 > 1 << _DESCRIPTOR_OFFSET_BITS:
            raise ValueError(
                f"{path} compact v3 shard {shard_index} exceeds the "
                f"{_DESCRIPTOR_OFFSET_BITS}-bit payload-word offset."
            )
        ranges = (
            ("direct payload", shard.payload_offset, shard.payload_bytes),
            ("compact headers", shard.widths_offset, shard.widths_bytes),
        )
        for label, offset, byte_count in ranges:
            if offset > file_bytes - byte_count:
                raise ValueError(
                    f"{path} compact v3 shard {shard_index} {label} range is "
                    "outside the file."
                )
        return
    if shard.descriptor_count != detector_pixels * tile_count:
        raise ValueError(
            f"{path} compact shard {shard_index} has descriptor count "
            f"{shard.descriptor_count}, expected {detector_pixels * tile_count}."
        )
    if shard.widths_bytes != shard.descriptor_count:
        raise ValueError(
            f"{path} compact shard {shard_index} must store one width byte per "
            "descriptor."
        )
    if shard.lengths_bytes != shard.chunk_count or shard.payload_bytes == 0:
        raise ValueError(
            f"{path} compact shard {shard_index} has inconsistent LZ4 metadata."
        )
    if shard.decoded_bytes == 0 or shard.decoded_bytes % 4:
        raise ValueError(
            f"{path} compact shard {shard_index} decoded payload is not uint32-sized."
        )
    expected_chunks = (
        shard.decoded_bytes + payload_chunk_bytes - 1
    ) // payload_chunk_bytes
    if shard.chunk_count != expected_chunks:
        raise ValueError(
            f"{path} compact shard {shard_index} has {shard.chunk_count} chunks, "
            f"expected {expected_chunks}."
        )
    for label, offset, byte_count in (
        ("payload", shard.payload_offset, shard.payload_bytes),
        ("chunk lengths", shard.lengths_offset, shard.lengths_bytes),
        ("descriptor widths", shard.widths_offset, shard.widths_bytes),
    ):
        if offset < 0 or byte_count < 0 or offset > file_bytes - byte_count:
            raise ValueError(
                f"{path} compact shard {shard_index} {label} range is outside the file."
            )


def _validate_nonoverlapping_ranges(path: Path, shards: list[CompactH5Shard]) -> None:
    ranges = sorted(
        (offset, offset + byte_count, shard_index, label)
        for shard_index, shard in enumerate(shards)
        for label, offset, byte_count in (
            ("payload", shard.payload_offset, shard.payload_bytes),
            ("chunk lengths", shard.lengths_offset, shard.lengths_bytes),
            ("descriptor widths", shard.widths_offset, shard.widths_bytes),
        )
        if byte_count
    )
    for previous, current in pairwise(ranges):
        if current[0] < previous[1]:
            raise ValueError(
                f"{path} compact ranges overlap between shard {previous[2]} "
                f"{previous[3]} and shard {current[2]} {current[3]}."
            )


def _require_sha256(value: object, label: str) -> str:
    """Return one validated lowercase SHA-256 digest."""
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be one lowercase SHA-256 digest")
    return value


def _validate_prepared_dpc_moments(
    path: Path,
    file_bytes: int,
    manifest: dict[str, object],
    shape: tuple[int, int, int, int],
    source_identity: str,
    excluded: tuple[int, ...],
    schema_version: int,
    shards: tuple[CompactH5Shard, ...],
) -> CompactH5PreparedDPCMoments | None:
    """Validate an optional source-bound exact moment payload."""
    value = manifest.get("prepared_dpc_moments")
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError(  # noqa: TRY004 - malformed serialized input
            f"{path} compact prepared DPC moments are not an object."
        )
    if schema_version == 1:
        if manifest.get("working_dtype") != "uint16":
            raise ValueError(
                f"{path} compact prepared DPC moments require uint16 working data."
            )
        schema = "quantem.gpu.prepared-dpc-moments/v2"
        working_dtype = "uint16"
        maximum_value = int(np.iinfo(np.uint16).max)
        working_sha = _require_sha256(
            manifest.get("working_logical_sha256"),
            f"{path} compact working logical identity",
        )
        working_field = "working_logical_sha256"
    else:
        schema = "quantem.gpu.prepared-dpc-moments/v1"
        working_dtype = "uint8"
        maximum_value = int(np.iinfo(np.uint8).max)
        working_sha = _require_sha256(
            manifest.get("prepared_uint8_sha256"),
            f"{path} compact working uint8 identity",
        )
        working_field = "working_uint8_sha256"
    detector_mask_sha = _require_sha256(
        manifest.get("detector_mask_sha256"),
        f"{path} compact detector mask identity",
    )
    scan_count = shape[0] * shape[1]
    detector_pixels = shape[2] * shape[3]
    selected = detector_pixels - len(excluded)
    excluded_set = set(excluded)
    total_bound = selected * maximum_value
    row_moment_bound = maximum_value * sum(
        pixel // shape[3]
        for pixel in range(detector_pixels)
        if pixel not in excluded_set
    )
    column_moment_bound = maximum_value * sum(
        pixel % shape[3]
        for pixel in range(detector_pixels)
        if pixel not in excluded_set
    )
    expected = {
        "schema": schema,
        "source_identity_sha256": source_identity,
        working_field: working_sha,
        "working_dtype": working_dtype,
        "detector_mask_sha256": detector_mask_sha,
        "detector_selection": "all-nonexcluded-v1",
        "scan_count": scan_count,
        "selected_detector_pixels": selected,
        "detector_columns": shape[3],
        "maximum_value": maximum_value,
        "dtype": "little-endian-u32",
        "word_order": "little-endian-u32-pairs",
        "words_per_scan": 8,
        "layout": [
            "total_lo",
            "total_hi",
            "row_lo",
            "row_hi",
            "column_lo",
            "column_hi",
            "padding_0",
            "padding_1",
        ],
        "total_bound": str(total_bound),
        "row_moment_bound": str(row_moment_bound),
        "column_moment_bound": str(column_moment_bound),
        "narrow_integer": total_bound <= np.iinfo(np.uint32).max,
        "narrow_products": max(row_moment_bound, column_moment_bound)
        <= np.iinfo(np.uint32).max,
    }
    if schema_version == 3:
        expected.pop("working_dtype")
        expected.pop("maximum_value")
    mismatches = {
        key: (value.get(key), expected_value)
        for key, expected_value in expected.items()
        if value.get(key) != expected_value
    }
    if mismatches:
        raise ValueError(
            f"{path} compact prepared DPC moments disagree with the source: "
            f"{mismatches}."
        )
    file_offset = value.get("file_offset")
    prepared_bytes = value.get("file_bytes")
    expected_bytes = scan_count * 8 * np.dtype("<u4").itemsize
    if (
        type(file_offset) is not int
        or type(prepared_bytes) is not int
        or file_offset < 0
        or file_offset % np.dtype("<u4").itemsize
        or prepared_bytes != expected_bytes
        or file_offset > file_bytes - prepared_bytes
    ):
        raise ValueError(f"{path} compact prepared DPC byte range is invalid.")
    prepared_end = file_offset + prepared_bytes
    for shard_index, shard in enumerate(shards):
        for label, offset, byte_count in (
            ("payload", shard.payload_offset, shard.payload_bytes),
            ("lengths", shard.lengths_offset, shard.lengths_bytes),
            ("headers", shard.widths_offset, shard.widths_bytes),
        ):
            if (
                byte_count
                and file_offset < offset + byte_count
                and offset < prepared_end
            ):
                raise ValueError(
                    f"{path} compact prepared DPC range overlaps shard "
                    f"{shard_index} {label}."
                )
    return CompactH5PreparedDPCMoments(
        file_offset=file_offset,
        file_bytes=prepared_bytes,
        sha256=_require_sha256(
            value.get("sha256"), f"{path} compact prepared DPC identity"
        ),
        working_logical_sha256=working_sha,
        working_dtype=working_dtype,
        detector_mask_sha256=detector_mask_sha,
        scan_count=scan_count,
        selected_detector_pixels=selected,
        detector_columns=shape[3],
        total_bound=total_bound,
        row_moment_bound=row_moment_bound,
        column_moment_bound=column_moment_bound,
        narrow_integer=total_bound <= np.iinfo(np.uint32).max,
        narrow_products=max(row_moment_bound, column_moment_bound)
        <= np.iinfo(np.uint32).max,
    )


def _validate_manifest(
    path: Path,
    manifest: dict[str, object],
    shape: tuple[int, int, int, int],
    shard_count: int,
    scans_per_shard: int,
    payload_chunk_bytes: int,
    source_identity: str,
    excluded: tuple[int, ...],
    schema_version: int,
    scan_tile: int,
    header_encoding: int,
) -> tuple[str | None, tuple[int, ...] | None]:
    detector_rows = shape[2]
    detector_columns = shape[3]
    coordinates = manifest.get("masked_detector_pixels")
    if not isinstance(coordinates, list):
        raise ValueError(  # noqa: TRY004 - malformed serialized input
            f"{path} compact JSON detector mask is not a list."
        )
    manifest_pixels = []
    if schema_version == 1:
        for coordinate in coordinates:
            if (
                not isinstance(coordinate, list)
                or len(coordinate) != 2
                or any(type(value) is not int for value in coordinate)
                or not 0 <= coordinate[0] < detector_rows
                or not 0 <= coordinate[1] < detector_columns
            ):
                raise ValueError(
                    f"{path} compact JSON detector coordinate "
                    f"{coordinate!r} is invalid."
                )
            manifest_pixels.append(coordinate[0] * detector_columns + coordinate[1])
        expected = {
            "schema": "quantem.gpu.packed-detector-h5/v1",
            "status": "complete",
            "source_shape": list(shape),
            "source_dtype": "uint16",
            "scan_bin": 1,
            "detector_bin": 1,
            "crop": None,
            "shard_count": shard_count,
            "scans_per_shard": scans_per_shard,
            "payload_chunk_bytes": payload_chunk_bytes,
            "payload_chunk_codec": "independent raw LZ4 blocks",
            "payload_chunk_length_codec": "uint8 encoded_bytes_minus_one",
            "descriptor_codec": "uint8 five-bit widths",
            "source_identity_sha256": source_identity,
        }
    else:
        detector_pixels = detector_rows * detector_columns
        if any(
            type(pixel) is not int or not 0 <= pixel < detector_pixels
            for pixel in coordinates
        ):
            raise ValueError(f"{path} compact v3 detector mask is invalid.")
        if coordinates != sorted(coordinates):
            raise ValueError(
                f"{path} compact v3 detector mask must use ordered row-major "
                "pixel indices."
            )
        manifest_pixels.extend(coordinates)
        expected = {
            "schema": "quantem.gpu.packed-detector-h5/v3",
            "status": "complete",
            "payload_codec": "direct-bitpacked-u32",
            "source_shape": list(shape),
            "source_dtype": "uint16",
            "working_dtype": "uint8",
            "working_value_definition": (
                "all admitted source counts exactly; authenticated dead pixels "
                "set to zero"
            ),
            "scan_bin": 1,
            "detector_bin": 1,
            "crop": None,
            "shard_count": shard_count,
            "scan_tile": scan_tile,
            "source_identity_sha256": source_identity,
        }
    mismatches = {
        key: (manifest.get(key), value)
        for key, value in expected.items()
        if manifest.get(key) != value
    }
    if mismatches:
        raise ValueError(
            f"{path} compact JSON and binary contract disagree: {mismatches}."
        )
    if tuple(manifest_pixels) != excluded:
        raise ValueError(f"{path} compact JSON and binary detector masks disagree.")
    if schema_version == 1 and manifest.get("working_dtype") not in {
        "uint8",
        "uint16",
    }:
        raise ValueError(
            f"{path} compact working dtype must be uint8 or uint16; got "
            f"{manifest.get('working_dtype')!r}."
        )
    if schema_version == 3:
        for field in (
            "source_raw_logical_sha256",
            "prepared_uint8_sha256",
            "detector_mask_sha256",
        ):
            value = manifest.get(field)
            if (
                not isinstance(value, str)
                or len(value) != 64
                or any(character not in "0123456789abcdef" for character in value)
            ):
                raise ValueError(f"{path} compact v3 {field} is invalid.")
        if header_encoding != 1:
            raise ValueError(f"{path} compact v3 header encoding is unsupported.")
        pixel_sequence_sha256 = manifest.get("masked_detector_pixels_sha256")
        if pixel_sequence_sha256 is not None:
            if (
                not isinstance(pixel_sequence_sha256, str)
                or len(pixel_sequence_sha256) != 64
                or any(
                    character not in "0123456789abcdef"
                    for character in pixel_sequence_sha256
                )
            ):
                raise ValueError(
                    f"{path} compact v3 masked_detector_pixels_sha256 is invalid."
                )
            expected_pixel_sequence_sha256 = hashlib.sha256(
                struct.pack(f"<{len(excluded)}I", *excluded)
            ).hexdigest()
            if pixel_sequence_sha256 != expected_pixel_sequence_sha256:
                raise ValueError(
                    f"{path} compact v3 masked_detector_pixels_sha256 does not "
                    "match the ordered binary-index detector pixels."
                )
        masked_detector_pixels_sha256 = pixel_sequence_sha256
        raw_values = manifest.get("masked_detector_raw_values")
        if raw_values is None:
            masked_detector_raw_values = None if excluded else ()
        else:
            if (
                not isinstance(raw_values, list)
                or len(raw_values) != len(excluded)
                or any(
                    type(value) is not int or not 0 <= value <= 65535
                    for value in raw_values
                )
            ):
                raise ValueError(
                    f"{path} compact v3 masked_detector_raw_values must be exact "
                    "uint16 values aligned one-to-one with the ordered detector mask."
                )
            masked_detector_raw_values = tuple(raw_values)
    else:
        working_logical_sha256 = manifest.get("working_logical_sha256")
        if working_logical_sha256 is not None:
            _require_sha256(
                working_logical_sha256,
                f"{path} compact working logical identity",
            )
        raw_values = manifest.get("masked_detector_raw_values")
        pixel_sequence_sha256 = manifest.get("masked_detector_pixels_sha256")
        if raw_values is None and pixel_sequence_sha256 is None:
            masked_detector_pixels_sha256 = None
            masked_detector_raw_values = None
        elif raw_values is None or pixel_sequence_sha256 is None:
            raise ValueError(
                f"{path} compact v1 masked-original metadata is incomplete."
            )
        else:
            masked_detector_pixels_sha256 = _require_sha256(
                pixel_sequence_sha256,
                f"{path} compact ordered masked detector pixels",
            )
            expected_pixel_sequence_sha256 = hashlib.sha256(
                struct.pack(f"<{len(excluded)}I", *excluded)
            ).hexdigest()
            if masked_detector_pixels_sha256 != expected_pixel_sequence_sha256:
                raise ValueError(
                    f"{path} compact v1 masked_detector_pixels_sha256 does not "
                    "match the ordered binary-index detector pixels."
                )
            if (
                not isinstance(raw_values, list)
                or len(raw_values) != len(excluded)
                or any(
                    type(value) is not int or not 0 <= value <= 65535
                    for value in raw_values
                )
            ):
                raise ValueError(
                    f"{path} compact v1 masked_detector_raw_values must be exact "
                    "uint16 values aligned one-to-one with the detector mask."
                )
            masked_detector_raw_values = tuple(raw_values)
    _validate_detector_calibration(path, manifest, shape, source_identity)
    return masked_detector_pixels_sha256, masked_detector_raw_values


def _validate_detector_calibration(
    path: Path,
    manifest: dict[str, object],
    shape: tuple[int, int, int, int],
    source_identity: str,
) -> None:
    calibration = manifest.get("detector_calibration")
    if calibration is None:
        return
    if not isinstance(calibration, dict):
        raise ValueError(  # noqa: TRY004 - malformed serialized input
            f"{path} compact detector calibration is not an object."
        )
    if calibration.get("schema") != "quantem.gpu.detector-calibration/v1":
        raise ValueError(f"{path} compact detector calibration schema is unsupported.")
    if calibration.get("source_identity_sha256") != source_identity:
        raise ValueError(
            f"{path} compact detector calibration belongs to a different source."
        )
    center = calibration.get("detector_center_px")
    if (
        not isinstance(center, list)
        or len(center) != 2
        or any(type(value) not in {int, float} for value in center)
        or any(not math.isfinite(float(value)) for value in center)
        or not 0 <= float(center[0]) < shape[2]
        or not 0 <= float(center[1]) < shape[3]
    ):
        raise ValueError(
            f"{path} compact detector calibration center must be finite [row, column]."
        )
    radius = calibration.get("bright_field_radius_px")
    if (
        type(radius) not in {int, float}
        or not math.isfinite(float(radius))
        or not 0 < float(radius) <= math.hypot(shape[2], shape[3])
    ):
        raise ValueError(
            f"{path} compact detector calibration bright-field radius is invalid."
        )
    rotation = calibration.get("dpc_rotation_degrees")
    exchanged = calibration.get("dpc_component_order_exchanged")
    if (rotation is None) != (exchanged is None):
        raise ValueError(
            f"{path} compact DPC calibration requires both rotation and component order."
        )
    if rotation is not None and (
        type(rotation) not in {int, float}
        or not math.isfinite(float(rotation))
        or type(exchanged) is not bool
    ):
        raise ValueError(f"{path} compact DPC calibration is invalid.")
    method = calibration.get("method")
    if not isinstance(method, str) or not method.strip():
        raise ValueError(f"{path} compact detector calibration method is missing.")
