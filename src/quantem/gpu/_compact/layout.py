"""Portable prepared-record validation and bounded file reads."""

from __future__ import annotations

import math
import os

import numpy as np

from . import LAYOUT_FORMAT, _matches_format

COMPONENTS = (
    "dense",
    "dense_offsets",
    "sparse",
    "sparse_offsets",
    "fields",
    "field_descriptors",
    "field_word_starts",
    "total",
    "coarse",
    "coarse_widths",
    "coarse_offsets",
)
SOURCE_COMPONENTS = COMPONENTS[:4]
VARIABLE_COMPONENTS = {"dense", "sparse", "fields", "coarse"}
FIXED_SHAPES = {
    "dense_offsets": (17466 * 9,),
    "sparse_offsets": (19398 * 9 + 1,),
    "field_descriptors": (16, 1032),
    "field_word_starts": (1033,),
    "total": (16384,),
    "coarse_widths": (36,),
    "coarse_offsets": (37,),
}


def _align(value: int, alignment: int) -> int:
    return ((value + alignment - 1) // alignment) * alignment


def _spec(name: str, shape, dtype) -> dict:
    """Validate fixed scientific extents before planning any file offsets."""
    shape = tuple(int(size) for size in shape)
    expected_dtype = np.dtype("u1" if name == "coarse_widths" else "<u4")
    if np.dtype(dtype) != expected_dtype:
        raise ValueError(f"Component {name} requires {expected_dtype}; got {dtype}.")
    if name in VARIABLE_COMPONENTS:
        if len(shape) != 1 or shape[0] <= 0:
            raise ValueError(
                f"Component {name} requires a nonempty flat word array; got {shape}."
            )
    elif shape != FIXED_SHAPES[name]:
        raise ValueError(
            f"Component {name} requires {FIXED_SHAPES[name]}; got {shape}."
        )
    nbytes = math.prod(shape) * expected_dtype.itemsize
    if nbytes >= 2**32:
        raise ValueError(
            f"Component {name} exceeds a bounded chunk record; got {nbytes} bytes."
        )
    return {
        "name": name,
        "shape": list(shape),
        "dtype": expected_dtype.str,
        "nbytes": nbytes,
    }


def validate_layout(layout: dict, *, require_complete: bool = False) -> None:
    """Reject incomplete coverage, overlaps and incompatible scalar extents.

    Parameters
    ----------
    layout
        The planned or completed checkpoint manifest's layout section.
    require_complete
        A loading boundary must require published completion and record hashes.
        CPU layout planning permits hashes to remain unset.

    Examples
    --------
    >>> validate_layout(layout, require_complete=False)
    """
    if (
        not _matches_format(layout["format"], LAYOUT_FORMAT)
        or layout["profile"] not in ("query-ready", "source-only")
        or layout["shape"] != [66, 512, 512, 192, 192]
        or layout["source_dtype"] != "<u2"
        or layout["byte_order"] != "little"
        or layout["chunk_scans"] != 16384
        or layout["stream_scans"] != 512
        or layout["component_alignment"] != 64
        or layout["record_alignment"] != 4096
        or layout["shard_period"] not in (2, 3)
        or len(layout["files"]) != 2
        or len(layout["chunks"]) != 1056
    ):
        raise ValueError(
            "Checkpoint layout does not describe the supported complete prepared native format."
        )
    if require_complete and layout["complete"] is not True:
        raise ValueError(
            "Checkpoint export is incomplete; finish and publish it before loading."
        )
    selected = COMPONENTS if layout["profile"] == "query-ready" else SOURCE_COMPONENTS
    ends, totals = [0, 0], {name: 0 for name in selected}
    for chunk, row in enumerate(layout["chunks"]):
        if set(row) != {
            "chunk",
            "acquisition",
            "first_scan",
            "scan_count",
            "shard",
            "file_offset",
            "record_bytes",
            "sha256",
            "components",
        }:
            raise ValueError(
                "Chunk descriptors must contain only the portable relative-offset schema."
            )
        shard = 0 if chunk % layout["shard_period"] == 0 else 1
        if (
            row["chunk"] != chunk
            or row["acquisition"] != chunk // 16
            or row["first_scan"] != (chunk % 16) * 16384
            or row["scan_count"] != 16384
            or row["shard"] != shard
            or row["file_offset"] != ends[shard]
            or row["record_bytes"] <= 0
            or row["record_bytes"] % 4096
            or [entry["name"] for entry in row["components"]] != list(selected)
        ):
            raise ValueError(
                f"Chunk {chunk} has changed coverage, shard order or record extents."
            )
        if require_complete:
            digest = row["sha256"]
            if (
                not isinstance(digest, str)
                or len(digest) != 64
                or any(c not in "0123456789abcdef" for c in digest)
            ):
                raise ValueError(f"Chunk {chunk} is missing its completed record hash.")
        cursor = 0
        for entry in row["components"]:
            if set(entry) != {"name", "shape", "dtype", "nbytes", "offset"}:
                raise ValueError(
                    "Component descriptors must not contain process pointers or extra fields."
                )
            expected = _spec(entry["name"], entry["shape"], entry["dtype"])
            if entry["nbytes"] != expected["nbytes"] or entry["offset"] != _align(
                cursor, 64
            ):
                raise ValueError(
                    f"Chunk {chunk} contains overlapping or invalid component byte extents."
                )
            cursor = entry["offset"] + entry["nbytes"]
            totals[entry["name"]] += entry["nbytes"]
        if _align(cursor, 4096) != row["record_bytes"]:
            raise ValueError(
                f"Chunk {chunk} record length differs from all component bytes."
            )
        ends[shard] += row["record_bytes"]
    if (
        any(
            entry != {"name": f"data-{index}.bin", "nbytes": ends[index]}
            for index, entry in enumerate(layout["files"])
        )
        or layout["logical_component_bytes"] != totals
        or layout["logical_payload_bytes"] != sum(totals.values())
        or layout["padded_pack_bytes"] != sum(ends)
        or layout["maximum_record_bytes"]
        != max(row["record_bytes"] for row in layout["chunks"])
    ):
        raise ValueError(
            "Checkpoint aggregate bytes disagree with the exact per-chunk layout."
        )


def record_views(buffer: np.ndarray, row: dict) -> dict[str, np.ndarray]:
    """View canonical chunk components inside one bounded contiguous host record.

    The manifest must first pass ``validate_layout``. Returned views borrow
    the host slot and stay valid only until its next read or export fill. The
    corresponding device views use the same shapes, types and relative offsets.

    Examples
    --------
    >>> products = record_views(host_slot, layout["chunks"][0])
    >>> products["field_descriptors"].shape
    (16, 1032)
    """
    if (
        buffer.dtype != np.uint8
        or buffer.ndim != 1
        or not buffer.flags.c_contiguous
        or buffer.nbytes < row["record_bytes"]
    ):
        raise ValueError(
            "Supply one contiguous byte slot covering the complete checkpoint record."
        )
    result = {}
    for entry in row["components"]:
        if (
            not 0
            <= entry["offset"]
            <= entry["offset"] + entry["nbytes"]
            <= row["record_bytes"]
        ):
            raise ValueError(
                "A component extends beyond its bounded checkpoint record."
            )
        result[entry["name"]] = np.ndarray(
            tuple(entry["shape"]),
            np.dtype(entry["dtype"]),
            buffer=buffer,
            offset=entry["offset"],
        )
    return result


def read_record(descriptor: int, row: dict, buffer: np.ndarray) -> None:
    """Read one entire record directly into a caller-owned reusable host slot.

    The caller opens the file in buffered or ``O_DIRECT`` mode. Direct mode
    additionally requires the supplied host pointer to meet device alignment.
    Short reads raise before the slot can be uploaded. The caller must retain
    this slot until its upload-completion fence.

    Examples
    --------
    >>> read_record(descriptor, layout["chunks"][0], host_slot)
    """
    record_views(buffer, row)
    view = memoryview(buffer)[: row["record_bytes"]]
    done = 0
    try:
        while done < len(view):
            count = os.preadv(descriptor, [view[done:]], row["file_offset"] + done)
            if count <= 0:
                raise OSError(
                    "Checkpoint record ended early; retain resident owners and restore the complete file."
                )
            done += count
    finally:
        view.release()
