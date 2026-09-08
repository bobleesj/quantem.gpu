"""Stream prepared records directly into final device owners."""

from __future__ import annotations

import os
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cupy as cp
import numpy as np

from .groups import Groups, PaletteGroups
from .layout import read_record
from .metadata import read_metadata
from .planner import NativePlanner
from .source import CompactSeries, build_kernels

OWNER_KEYS = ("dense", "dense_offsets", "sparse", "sparse_offsets", "fields", "total")


def extents(layout):
    """Word starts derived only from the checkpoint layout.

    Examples
    --------
    >>> starts = extents(layout)
    >>> starts["fields"].shape
    (1057,)
    """
    starts = {}
    for name in (*OWNER_KEYS, "coarse"):
        words = np.fromiter(
            (
                next(
                    entry["nbytes"]
                    for entry in row["components"]
                    if entry["name"] == name
                )
                // 4
                for row in layout["chunks"]
            ),
            np.int64,
            len(layout["chunks"]),
        )
        starts[name] = np.concatenate(([0], np.cumsum(words)))
    return starts


def load(
    cache: Path,
    *,
    device: int,
    slots: int = 4,
    workers: int = 4,
    direct: bool = True,
    progress=None,
) -> CompactSeries:
    """Stream a completed prepared series onto the explicitly selected device."""
    with cp.cuda.Device(device):
        return _load(Path(cache), device, slots, workers, direct, progress)


def _load(cache, device, slots, workers, direct, progress):
    began = time.perf_counter()
    manifest, metadata = read_metadata(cache)
    layout = manifest["layout"]
    chunks = layout["chunks"]
    for entry in layout["files"]:
        path = cache / entry["name"]
        if path.stat().st_size != entry["nbytes"]:
            raise ValueError(f"Prepared pack length differs from manifest: {path}.")
    starts = extents(layout)
    d = CompactSeries(device)
    d.owners = {name: cp.empty(int(starts[name][-1]), cp.uint32) for name in OWNER_KEYS}
    d.coarse_index = cp.empty(int(starts["coarse"][-1]), cp.uint32)
    field_descriptors = np.empty((1056, 16, 1032), np.uint32)
    field_words = np.empty((1056, 1033), np.uint32)
    coarse_widths = np.empty((1056, 36), np.uint8)
    coarse_offsets = np.empty((1056, 37), np.uint32)
    capacity = ((layout["maximum_record_bytes"] + (4 << 20) - 1) // (4 << 20)) * (
        4 << 20
    )
    pinned = [cp.cuda.alloc_pinned_memory(capacity) for _ in range(slots)]
    slot_arrays = [np.frombuffer(block, np.uint8, count=capacity) for block in pinned]
    if any(array.ctypes.data % 4096 for array in slot_arrays):
        raise ValueError("Direct reads need page-aligned pinned slots.")
    paths = [cache / entry["name"] for entry in layout["files"]]
    descriptors = []
    stream = cp.cuda.Stream(non_blocking=True)
    events = [cp.cuda.Event(disable_timing=True) for _ in range(slots)]
    read_seconds = 0.0
    allocated = time.perf_counter()
    try:
        for path, entry in zip(paths, layout["files"]):
            descriptor = os.open(path, os.O_RDONLY | (os.O_DIRECT if direct else 0))
            descriptors.append(descriptor)
            if os.fstat(descriptor).st_size != entry["nbytes"]:
                raise ValueError(f"Prepared pack length differs from manifest: {path}.")
        with ThreadPoolExecutor(
            max_workers=workers, thread_name_prefix="cold-read"
        ) as pool:
            futures = [None] * slots

            def submit(chunk):
                row = chunks[chunk]
                index = chunk % slots
                futures[index] = pool.submit(
                    _read_timed, descriptors[row["shard"]], row, slot_arrays[index]
                )

            for chunk in range(min(slots, len(chunks))):
                submit(chunk)
            for chunk, row in enumerate(chunks):
                index = chunk % slots
                read_seconds += futures[index].result()
                slot = slot_arrays[index]
                for entry in row["components"]:
                    host = (
                        slot[entry["offset"] : entry["offset"] + entry["nbytes"]]
                        .view(np.dtype(entry["dtype"]))
                        .reshape(entry["shape"])
                    )
                    name = entry["name"]
                    if name in OWNER_KEYS:
                        d.owners[name][
                            starts[name][chunk] : starts[name][chunk + 1]
                        ].set(host.reshape(-1), stream=stream)
                    elif name == "coarse":
                        d.coarse_index[
                            starts["coarse"][chunk] : starts["coarse"][chunk + 1]
                        ].set(host.reshape(-1), stream=stream)
                    elif name == "field_descriptors":
                        field_descriptors[chunk] = host
                    elif name == "field_word_starts":
                        field_words[chunk] = host
                    elif name == "coarse_widths":
                        coarse_widths[chunk] = host
                    elif name == "coarse_offsets":
                        coarse_offsets[chunk] = host
                    else:
                        raise ValueError(f"Unknown checkpoint component {name}.")
                events[index].record(stream)
                following = chunk + slots
                if following < len(chunks):
                    # The slot is reusable once its uploads have been consumed.
                    events[index].synchronize()
                    submit(following)
                if progress is not None and chunk % 64 == 63:
                    progress(chunk + 1, len(chunks), time.perf_counter() - began)
        stream.synchronize()
    finally:
        # Drain copies before releasing a pinned slot, including failed reads.
        stream.synchronize()
        for descriptor in descriptors:
            os.close(descriptor)
    streamed = time.perf_counter()
    slot_arrays.clear()
    pinned.clear()
    _assemble(
        d,
        metadata,
        starts,
        field_descriptors,
        field_words,
        coarse_widths,
        coarse_offsets,
    )
    # Consumers may query from another stream after load returns.
    cp.cuda.get_current_stream().synchronize()
    d.load_seconds = time.perf_counter() - began
    d.load_timing = {
        "allocate_seconds": allocated - began,
        "stream_seconds": streamed - allocated,
        "assemble_seconds": time.perf_counter() - streamed,
        "read_worker_seconds_sum": read_seconds,
        "pack_paths": [str(path) for path in paths],
        "slots": slots,
        "workers": workers,
        "direct": direct,
        "pack_bytes": layout["padded_pack_bytes"],
    }
    return d


def _read_timed(descriptor, row, slot):
    before = time.perf_counter()
    read_record(descriptor, row, slot)
    return time.perf_counter() - before


def _assemble(
    d, metadata, starts, field_descriptors, field_words, coarse_widths, coarse_offsets
):
    """Rebuild process-local addresses and bounded query workspace."""
    d.blocks = [
        (
            d.owners["dense"][starts["dense"][i] : starts["dense"][i + 1]],
            d.owners["dense_offsets"][
                starts["dense_offsets"][i] : starts["dense_offsets"][i + 1]
            ],
        )
        for i in range(1056)
    ]
    d.sparse = [
        (
            d.owners["sparse"][starts["sparse"][i] : starts["sparse"][i + 1]],
            d.owners["sparse_offsets"][
                starts["sparse_offsets"][i] : starts["sparse_offsets"][i + 1]
            ],
        )
        for i in range(1056)
    ]
    for name, key in (
        ("payload_addresses", "dense"),
        ("offset_addresses", "dense_offsets"),
        ("event_addresses", "sparse"),
        ("sparse_addresses", "sparse_offsets"),
    ):
        setattr(
            d,
            name,
            cp.asarray(
                np.uint64(d.owners[key].data.ptr)
                + starts[key][:-1].astype(np.uint64) * 4
            ),
        )
    for name, array in metadata.items():
        if name.startswith("planner__"):
            setattr(
                d,
                name.removeprefix("planner__"),
                array.item() if array.ndim == 0 else array,
            )
    d.decoding = cp.asarray(metadata["codec__decoding"])
    d.cached = metadata["cached"]
    ids = metadata["model_ids"]
    d.field_widths = cp.asarray(field_descriptors)
    d.field_count = field_descriptors.shape[-1]

    d.field_addresses = cp.asarray(
        np.uint64(d.owners["fields"].data.ptr)
        + 4
        * (
            starts["fields"][:-1, None].astype(np.uint64)
            + field_words[:, :1032].astype(np.uint64)
        )
    )
    d.tile_indices = cp.empty(d.field_count, cp.int32)
    d.tile_signs = cp.empty(d.field_count, cp.int8)
    d.sparse_indices = cp.empty(36864, cp.int32)
    d.sparse_coefficients = cp.empty(36864, cp.int8)
    d.total = d.owners["total"].reshape(66, 512, 512)
    d.output = cp.empty_like(d.total)
    d.groups = PaletteGroups(Groups(ids, len(metadata["codec__decoding"])))
    build_kernels(d)
    d.coarse_addresses = cp.asarray(
        np.uint64(d.coarse_index.data.ptr)
        + 4
        * (
            starts["coarse"][:-1, None].astype(np.uint64)
            + coarse_offsets[:, :36].astype(np.uint64)
        )
    )
    d.coarse_widths = cp.asarray(coarse_widths)
    d.coarse_selected = cp.empty(36, cp.int32)
    d.coarse_signs = cp.empty(36, cp.int8)
    d.native_planner = NativePlanner.from_metadata(
        **{
            name: metadata[f"planner__{name}"].item()
            if metadata[f"planner__{name}"].ndim == 0
            else metadata[f"planner__{name}"]
            for name in (
                "valid",
                "leaf_of",
                "column_cost",
                "tile_cost",
                "parents",
                "omitted",
                "stored_positions",
            )
        }
    )
    d.previous = None
    d.start, d.group_ready, d.seed_ready, d.dense_ready, d.ready = [
        cp.cuda.Event() for _ in range(5)
    ]
    d.patterns = cp.empty((66, 192, 192), cp.uint16)
    d.pattern_ids = cp.asarray(ids)
    d.pattern_cache_map = cp.asarray(d.cache_map)
    d.resident_bytes = (
        sum(value.nbytes for value in d.owners.values()) + d.coarse_index.nbytes
    )
