"""Read selected compressed detector frames of one HDF5 master into page-locked memory.

Bounded encoded and precision loads decode one block of scan positions at a
time. Reading only those frames' HDF5 chunks, straight into a page-locked
buffer, keeps host memory and transfer time proportional to the block rather
than the acquisition. The GPU decoders (``io.hdf5.cuda.decode``,
``io.hdf5.mps.decode``) consume the prepared plan returned here.
"""

import bisect
import ctypes
import os
import time
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import Self

import numpy as np

from quantem.gpu.device.cuda_runtime import _alloc_pinned_fast, _get_libc
from quantem.gpu.formats.hdf5.frames import (
    BLOCK_SIZE,
    master_frame_sources,
    parse_headers,
)

# One read task per 8 MiB keeps a large shard from serializing on one worker.
_HDF5_READ_TASK_BYTES = 8 * 1024**2
_POSIX_FADV_WILLNEED = 3


class FrameReader:
    """Own reusable host resources for selected reads from one HDF5 master.

    Bounded encoded and precision loads read one block of frames at a time;
    the reader keeps the immutable source shards open and one worker pool
    alive so each block does not reopen files or recreate threads.
    """

    def __init__(self, filepath: str) -> None:
        self.closed = False
        started = time.perf_counter()
        source_started = time.perf_counter()
        self.source_infos = master_frame_sources(filepath)
        source_seconds = time.perf_counter() - source_started
        if not self.source_infos:
            raise ValueError(f"No detector data chunks found in {filepath}")
        chunk_index_started = time.perf_counter()
        self.chunk_index_arrays = tuple(
            np.asarray(info["chunk_infos"], dtype=np.uint64)
            for info in self.source_infos
        )
        chunk_index_seconds = time.perf_counter() - chunk_index_started
        self.source_starts = np.cumsum(
            [0] + [info["n_frames"] for info in self.source_infos],
            dtype=np.int64,
        )

        file_open_started = time.perf_counter()
        self.fds: dict[int, int] = {}
        try:
            for source_index, info in enumerate(self.source_infos):
                self.fds[source_index] = os.open(info["path"], os.O_RDONLY)
        except OSError:
            for descriptor in self.fds.values():
                os.close(descriptor)
            raise
        file_open_seconds = time.perf_counter() - file_open_started

        pool_started = time.perf_counter()
        # Workers start lazily. One large shard can also supply independent
        # required-byte reads, so concurrency must not depend on file count.
        self.thread_pool = ThreadPoolExecutor(max_workers=12)
        pool_seconds = time.perf_counter() - pool_started
        self._initial_timing = {
            "source_index": source_seconds,
            "chunk_index_array_build": chunk_index_seconds,
            "persistent_file_open": file_open_seconds,
            "persistent_thread_pool_create": pool_seconds,
            "persistent_session_total": time.perf_counter() - started,
        }

    def prepare(self, frame_indices: np.ndarray) -> dict:
        """Read selected compressed detector frames and index them for GPU decode.

        Only the HDF5 chunks of the requested flattened scan-frame indices are
        read, through the open shards and worker pool. The returned plan is the
        input of ``io.hdf5.cuda.decode.decompress_prepared`` and the Metal
        ``load_prepared_frames``.
        """
        total_started = time.perf_counter()
        selected = np.asarray(frame_indices, dtype=np.int64).reshape(-1)
        if selected.size == 0:
            raise ValueError("frame_indices must contain at least one frame")
        if np.any(selected < 0):
            raise ValueError("frame_indices must be non-negative")

        if self.closed:
            raise RuntimeError("Sparse frame read session is already closed")
        source_infos = self.source_infos
        # One-time initialization costs are reported with the first prepared batch.
        session_timing = self._initial_timing
        self._initial_timing = {name: 0.0 for name in session_timing}
        source_seconds = session_timing.pop("source_index")
        frame_shape = source_infos[0]["frame_shape"]
        dtype = source_infos[0]["dtype"]
        for info in source_infos[1:]:
            if info["frame_shape"] != frame_shape or np.dtype(info["dtype"]) != np.dtype(dtype):
                raise ValueError("Detector chunk files have inconsistent shape or dtype")

        source_starts = self.source_starts
        total_available = int(source_starts[-1])
        if int(selected.max()) >= total_available:
            raise ValueError(
                f"Requested frame {int(selected.max())}, but only {total_available} frames are available"
            )

        plan_started = time.perf_counter()
        # Reading a gap this small is cheaper than issuing another read.
        max_gap_bytes = 4096
        fast_plan = _contiguous_frame_read_plan(
            source_infos,
            source_starts,
            selected,
            max_gap_bytes=max_gap_bytes,
            chunk_index_arrays=self.chunk_index_arrays,
        )
        if fast_plan is not None:
            (
                chunk_offsets_arr,
                chunk_sizes_arr,
                read_plan_by_source,
                cursor,
            ) = fast_plan
        else:
            chunk_offsets_arr = np.empty(selected.size, dtype=np.uint64)
            chunk_sizes_arr = np.empty(selected.size, dtype=np.uint32)
            entries_by_source: dict[int, list[tuple[int, int, int]]] = {}
            # Python ints throughout: NumPy promotes uint64 offsets mixed with int64 to float64.
            for order_position, global_index in enumerate(selected):
                source_index = bisect.bisect_right(source_starts, int(global_index)) - 1
                local_index = int(global_index) - int(source_starts[source_index])
                chunk_infos = source_infos[source_index]["chunk_infos"]
                if local_index >= len(chunk_infos):
                    raise ValueError(
                        f"Requested local frame {local_index}, but only "
                        f"{len(chunk_infos)} HDF5 chunks were indexed"
                    )
                byte_offset, chunk_size = chunk_infos[local_index]
                entries_by_source.setdefault(source_index, []).append(
                    (order_position, int(byte_offset), int(chunk_size))
                )

            read_plan_by_source = {}
            cursor = 0

            def append_span(
                source_index: int,
                start: int,
                stop: int,
                span: list[tuple[int, int, int]],
                destination: int,
            ) -> int:
                span_nbytes = int(stop - start)
                read_plan_by_source.setdefault(source_index, []).append(
                    (int(start), int(destination), span_nbytes)
                )
                for order_position, byte_offset, chunk_size in span:
                    chunk_offsets_arr[order_position] = destination + int(byte_offset - start)
                    chunk_sizes_arr[order_position] = int(chunk_size)
                return destination + span_nbytes

            for source_index, unsorted_entries in entries_by_source.items():
                entries = sorted(unsorted_entries, key=lambda item: item[1])
                span_start = entries[0][1]
                span_end = entries[0][1] + entries[0][2]
                span_entries = [entries[0]]

                for entry in entries[1:]:
                    _, byte_offset, chunk_size = entry
                    next_end = int(byte_offset + chunk_size)
                    if int(byte_offset) <= span_end + max_gap_bytes:
                        span_end = max(span_end, next_end)
                        span_entries.append(entry)
                    else:
                        cursor = append_span(
                            source_index, span_start, span_end, span_entries, cursor
                        )
                        span_start = int(byte_offset)
                        span_end = next_end
                        span_entries = [entry]
                cursor = append_span(source_index, span_start, span_end, span_entries, cursor)
        plan_seconds = time.perf_counter() - plan_started

        total_compressed = int(cursor)
        alloc_started = time.perf_counter()
        read_buffer = _alloc_pinned_fast(total_compressed)
        alloc_seconds = time.perf_counter() - alloc_started
        libc = _get_libc()

        def read_exact_at(fd: int, destination_offset: int, nbytes: int, file_offset: int) -> None:
            # preadv writes straight into the page-locked buffer (Linux and macOS).
            destination_view = memoryview(read_buffer)[destination_offset:destination_offset + nbytes]
            remaining = int(nbytes)
            view_offset = 0
            while remaining > 0:
                bytes_read = os.preadv(
                    fd,
                    [destination_view[view_offset:view_offset + remaining]],
                    int(file_offset + view_offset),
                )
                if bytes_read == 0:
                    raise OSError("short read while loading selected HDF5 chunks")
                remaining -= bytes_read
                view_offset += bytes_read

        def read_source(
            item: tuple[int, list[tuple[int, int, int]]],
        ) -> tuple[float, float]:
            source_index, reads = item
            fd = self.fds[source_index]
            advice_seconds = 0.0
            if libc is not None:
                advice_started = time.perf_counter()
                for file_offset, _, nbytes in reads:
                    libc.posix_fadvise(
                        fd,
                        ctypes.c_long(file_offset),
                        ctypes.c_long(nbytes),
                        _POSIX_FADV_WILLNEED,
                    )
                advice_seconds = time.perf_counter() - advice_started
            pread_started = time.perf_counter()
            for file_offset, destination_offset, nbytes in reads:
                read_exact_at(fd, destination_offset, nbytes, file_offset)
            return advice_seconds, time.perf_counter() - pread_started

        # Split only the already-selected spans. Independent preadv calls write
        # disjoint slices of the same registered buffer without a second host copy.
        # A large HDF5 shard should not force the whole batch through one worker.
        read_jobs = []
        for source_index, reads in read_plan_by_source.items():
            for file_offset, destination, count in reads:
                for offset in range(0, count, _HDF5_READ_TASK_BYTES):
                    length = min(_HDF5_READ_TASK_BYTES, count - offset)
                    read_jobs.append(
                        (source_index, [(file_offset + offset, destination + offset, length)])
                    )
        read_started = time.perf_counter()
        if len(read_jobs) > 1:
            read_metrics = list(self.thread_pool.map(read_source, read_jobs))
        else:
            read_metrics = [read_source(item) for item in read_jobs]
        read_seconds = time.perf_counter() - read_started
        fadvise_seconds = sum(metric[0] for metric in read_metrics)
        pread_seconds = sum(metric[1] for metric in read_metrics)

        frame_bytes = int(np.prod(frame_shape) * np.dtype(dtype).itemsize)
        n_blocks_per_frame = (frame_bytes + BLOCK_SIZE - 1) // BLOCK_SIZE
        block_starts_flat = np.zeros(selected.size * n_blocks_per_frame, dtype=np.uint32)
        block_counts = np.zeros(selected.size, dtype=np.uint32)
        block_offsets_arr = np.zeros(selected.size + 1, dtype=np.uint32)
        headers_started = time.perf_counter()
        parse_headers(
            read_buffer,
            chunk_sizes_arr,
            chunk_offsets_arr,
            block_starts_flat,
            block_counts,
            int(selected.size),
            n_blocks_per_frame,
            thread_pool=self.thread_pool,
        )
        block_offsets_arr[1:selected.size + 1] = np.cumsum(block_counts[:selected.size])
        total_blocks = int(block_offsets_arr[selected.size])
        header_seconds = time.perf_counter() - headers_started
        total_seconds = time.perf_counter() - total_started

        return {
            "read_buffer": read_buffer[:total_compressed],
            "chunk_offsets": chunk_offsets_arr,
            "block_starts": block_starts_flat[:total_blocks],
            "block_counts": block_counts,
            "block_offsets": block_offsets_arr,
            "total_frames": int(selected.size),
            "frame_shape": frame_shape,
            "frame_bytes": frame_bytes,
            "dtype": dtype,
            "prepare_timing_s": {
                "source_index": source_seconds,
                "read_plan": plan_seconds,
                "pinned_alloc": alloc_seconds,
                "compressed_read": read_seconds,
                # float: an empty job list sums to the integer 0.
                "compressed_pread_cpu": float(pread_seconds),
                "posix_fadvise_cpu": float(fadvise_seconds),
                "header_parse": header_seconds,
                "total": total_seconds,
                **session_timing,
            },
        }

    def close(self) -> None:
        """Close the persistent descriptors and preparation worker pool."""
        if self.closed:
            return
        self.closed = True
        self.thread_pool.shutdown(wait=True)
        for descriptor in self.fds.values():
            os.close(descriptor)
        self.fds.clear()

    def __enter__(self) -> Self:
        return self

    def __exit__(self, _exc_type, _exc_value, _traceback) -> None:
        self.close()


def _contiguous_frame_read_plan(
    source_infos: list[dict],
    source_starts,
    selected,
    *,
    max_gap_bytes: int,
    chunk_index_arrays: Sequence[np.ndarray] | None = None,
) -> tuple[
    np.ndarray,
    np.ndarray,
    dict[int, list[tuple[int, int, int]]],
    int,
] | None:
    """Plan monotonic contiguous frame reads without per-frame Python work.

    Encoded loads request consecutive scan positions block by block.  A
    vectorized planner is substantially cheaper for that common case, while
    arbitrary order, duplicates, and non-monotonic physical chunk layouts
    deliberately fall back to the general selector-preserving planner.
    """
    selected = np.asarray(selected, dtype=np.int64).reshape(-1)
    if selected.size == 0:
        return None
    first = int(selected[0])
    if int(selected[-1]) - first + 1 != int(selected.size):
        return None
    if selected.size > 1 and not np.all(selected[1:] == selected[:-1] + 1):
        return None

    starts = np.asarray(source_starts, dtype=np.int64).reshape(-1)
    if starts.size != len(source_infos) + 1:
        return None
    if chunk_index_arrays is not None and len(chunk_index_arrays) != len(source_infos):
        return None

    chunk_offsets = np.empty(selected.size, dtype=np.uint64)
    chunk_sizes = np.empty(selected.size, dtype=np.uint32)
    read_plan: dict[int, list[tuple[int, int, int]]] = {}
    cursor = 0
    stop = first + int(selected.size)

    for source_index, info in enumerate(source_infos):
        source_start = int(starts[source_index])
        source_stop = int(starts[source_index + 1])
        selected_start = max(first, source_start)
        selected_stop = min(stop, source_stop)
        if selected_start >= selected_stop:
            continue

        local_start = selected_start - source_start
        local_stop = selected_stop - source_start
        all_chunks = (
            np.asarray(chunk_index_arrays[source_index], dtype=np.uint64)
            if chunk_index_arrays is not None
            else np.asarray(info["chunk_infos"], dtype=np.uint64)
        )
        if (
            all_chunks.ndim != 2
            or all_chunks.shape[1:] != (2,)
            or local_stop > int(all_chunks.shape[0])
        ):
            return None
        chunks = all_chunks[local_start:local_stop]
        offsets = chunks[:, 0]
        sizes_u64 = chunks[:, 1]
        if np.any(sizes_u64 > np.iinfo(np.uint32).max):
            raise ValueError("Compressed HDF5 chunk exceeds uint32 size range")
        if offsets.size > 1 and np.any(offsets[1:] < offsets[:-1]):
            return None

        ends = offsets + sizes_u64
        if np.any(ends < offsets):
            raise ValueError("Compressed HDF5 chunk byte range overflowed uint64")
        running_ends = np.maximum.accumulate(ends)
        split_points = np.flatnonzero(
            offsets[1:] > running_ends[:-1] + np.uint64(max_gap_bytes)
        ) + 1
        group_starts = np.concatenate((np.array([0]), split_points))
        group_stops = np.concatenate((split_points, np.array([offsets.size])))
        group_bounds = list(zip(group_starts.tolist(), group_stops.tolist(), strict=True))

        destination_start = cursor
        source_plan: list[tuple[int, int, int]] = []
        for group_start, group_stop in group_bounds:
            span_start = int(offsets[group_start])
            span_stop = int(running_ends[group_stop - 1])
            span_nbytes = span_stop - span_start
            source_plan.append((span_start, cursor, span_nbytes))
            cursor += span_nbytes
        read_plan[source_index] = source_plan

        order_start = selected_start - first
        order_stop = selected_stop - first
        group_destinations = np.empty(offsets.size, dtype=np.uint64)
        group_cursors = np.cumsum(
            np.asarray([0] + [item[2] for item in source_plan[:-1]], dtype=np.uint64)
        ) + np.uint64(destination_start)
        for group_number, (group_start, group_stop) in enumerate(group_bounds):
            group_destinations[group_start:group_stop] = (
                group_cursors[group_number]
                + offsets[group_start:group_stop]
                - offsets[group_start]
            )
        chunk_offsets[order_start:order_stop] = group_destinations
        # The header parser and decoders read uint32 sizes; the range was checked above.
        chunk_sizes[order_start:order_stop] = sizes_u64.astype(np.uint32, copy=False)

    if sum(len(plan) for plan in read_plan.values()) == 0:
        return None
    return chunk_offsets, chunk_sizes, read_plan, int(cursor)
