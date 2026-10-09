"""Joint translated detector queries over runtime-shaped encoded acquisitions."""

import math
import threading
import time

import numpy as np

from quantem.gpu.detector.cuda.mask_plan import CudaMaskPlanner
from quantem.gpu.detector.cuda.series import CudaSeriesCompute
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.resident.cuda.counts import StreamedCounts, field_count, kernels
from quantem.gpu.resident.queries import weight_digits


def plan(mask: np.ndarray):
    """Express an exact signed mask as 32/8-pixel tiles plus pixel residuals."""
    rows, cols = mask.shape
    leaf_rows, leaf_cols = math.ceil(rows / 8), math.ceil(cols / 8)
    padded = np.zeros((leaf_rows * 8, leaf_cols * 8), np.int32)
    padded[:rows, :cols] = mask
    tiles = padded.reshape(leaf_rows, 8, leaf_cols, 8).transpose(0, 2, 1, 3)
    counts = np.stack([(tiles == value).sum(axis=(2, 3)) for value in (0, 1, -1)])
    leaves = np.array([0, 1, -1], np.int32)[counts.argmax(axis=0)]
    residual = mask - np.repeat(np.repeat(leaves, 8, axis=0), 8, axis=1)[:rows, :cols]
    root_rows, root_cols = math.ceil(leaf_rows / 4), math.ceil(leaf_cols / 4)
    coarse = np.zeros((root_rows * 4, root_cols * 4), np.int32)
    coarse[:leaf_rows, :leaf_cols] = leaves
    tiles = coarse.reshape(root_rows, 4, root_cols, 4).transpose(0, 2, 1, 3)
    counts = np.stack([(tiles == value).sum(axis=(2, 3)) for value in (0, 1, -1)])
    roots = np.array([0, 1, -1], np.int32)[counts.argmax(axis=0)]
    leaves = leaves - np.repeat(np.repeat(roots, 4, axis=0), 4, axis=1)[:leaf_rows, :leaf_cols]
    fields = np.concatenate((leaves.ravel(), roots.ravel()))
    selected_fields = np.flatnonzero(fields).astype(np.uint32)
    selected_pixels = np.flatnonzero(residual).astype(np.uint32)
    return (
        selected_fields,
        fields[selected_fields],
        selected_pixels,
        residual.ravel()[selected_pixels],
    )


class StreamedSeriesCompute(CudaSeriesCompute):
    """Borrow encoded chunks and launch each query jointly across acquisitions.

    Spatial indexes and signed pixel corrections are integer exact. A private
    previous sum can seed small mask changes; caller-owned output buffers never
    become future baselines. Request and presentation ownership remains in the
    detector session and native client.
    """

    def __init__(self, acquisitions):
        self.owners = tuple(acquisitions)
        sources = [
            item.data if isinstance(item, Dataset4dstemGPU) else item
            for item in acquisitions
        ]
        self.index_owners = tuple(sources)
        if not sources or not all(isinstance(source, StreamedCounts) for source in sources):
            raise TypeError(
                "Streamed joint queries require complete streamed count sources."
            )
        first = sources[0]
        self.device = first.device
        self.scan_shape, self.det_shape = first.shape[:2], first.shape[2:]
        self.series_shape = (len(sources),)
        self.n_frames, self.pixels = (
            math.prod(self.scan_shape),
            math.prod(self.det_shape),
        )
        self.interval = min(source.interval for source in sources)
        self.fields = field_count(self.det_shape)
        self.frame_dtype = np.result_type(*[source.dtype for source in sources])
        maximum = self.pixels * int(np.iinfo(self.frame_dtype).max)
        if maximum * len(sources) > np.iinfo(np.uint64).max:
            raise OverflowError("The complete series exceeds exact uint64 sum bounds.")
        self.sum_dtype = np.dtype(
            "uint32" if maximum <= np.iinfo(np.uint32).max else "uint64"
        )
        self.keepalive, rows = [], []
        self.max_scans = 0
        for acquisition, source in enumerate(sources):
            if source.device != self.device or source.shape != first.shape:
                raise ValueError(
                    "Linked acquisitions must have equal native shapes and share one CUDA device."
                )
            if source.is_released or source.ready_scans != self.n_frames:
                raise ValueError(
                    "Finish loading every native scan before preparing detector queries."
                )
            self.keepalive.extend((source.valid, source.decoding))
            if source._detector_total is not None:
                self.keepalive.append(source._detector_total)
            for chunk in source.chunks:
                self.keepalive.extend(chunk.arrays)
                payload, offsets, models, words, starts, widths = chunk.arrays
                row = [
                    payload.data.ptr,
                    offsets.data.ptr,
                    models.data.ptr,
                    source.decoding.data.ptr,
                    words.data.ptr,
                    starts.data.ptr,
                    widths.data.ptr,
                    chunk.first,
                    chunk.scans,
                    acquisition,
                    source.valid.data.ptr,
                    source.interval,
                ]
                rows.append(row)
                self.max_scans = max(self.max_scans, chunk.scans)
        if len(rows) > 65535:
            raise ValueError(
                "This chunk count exceeds the joint CUDA launch limit; use larger loading chunks."
            )
        self.residual_warps = min(4, math.ceil(self.max_scans / self.interval))
        self.valid_pixels = np.stack([source.valid_pixels for source in sources])
        with cp.cuda.Device(self.device):
            self.descriptors = cp.asarray(rows, dtype=cp.uint64)
            self.chunk_count = len(rows)
            self.error_slots = cp.zeros(8, cp.uint32)   # one counter per query in flight
            self.errors = self.error_slots[:1]
            self.previous = cp.empty(
                (*self.series_shape, *self.scan_shape), self.sum_dtype
            )
            self.selected_fields = cp.empty(self.fields, cp.uint32)
            self.field_coefficients = cp.empty(self.fields, cp.int32)
            self.selected_pixels = cp.empty(self.pixels, cp.uint32)
            self.pixel_coefficients = cp.empty(self.pixels, cp.int32)
            self.begin, self.end = cp.cuda.Event(), cp.cuda.Event()
            self.kernels = kernels(self.device)
            cp.cuda.get_current_stream().synchronize()
            # Pinned staging so selection uploads and error readbacks are asynchronous: a
            # synchronous memcpy waits for every kernel queued ahead of it, which would
            # serialise queries meant to overlap the host's planning of the next one.
            slots = len(self.error_slots)
            sizes = (self.fields, self.fields, self.pixels, self.pixels)
            self._staging = [cp.cuda.alloc_pinned_memory(slots * size * 4) for size in sizes]
            self._staging_views = [
                np.frombuffer(memory, dtype, slots * size).reshape(slots, size)
                for memory, dtype, size in zip(
                    self._staging, (np.uint32, np.int32, np.uint32, np.int32), sizes
                )
            ]
            self._error_pinned = cp.cuda.alloc_pinned_memory(slots * 4)
            self._error_host = np.frombuffer(self._error_pinned, np.uint32, slots)
        self.previous_mask = None
        self.block_stride = 1
        self.previous_stride = 1
        self._mask_planner = None
        self._inflight = []   # (begin, end, done, errors_slot, started, info) for queries launched with wait=False
        self._launches = 0
        self._tainted_from = None   # launch index of a failed query; later incremental queries built on it
        # Every query's error slot, not only the first: an ``out`` overlapping any of them is refused.
        self.keepalive.extend(
            (
                self.descriptors,
                self.error_slots,
                self.previous,
                self.selected_fields,
                self.field_coefficients,
                self.selected_pixels,
                self.pixel_coefficients,
            )
        )
        regions = {(array.data.ptr, array.data.ptr + array.nbytes) for array in self.keepalive}
        self.regions = np.asarray(sorted(regions), np.uint64)
        self.backend_metadata = {
            "backend": "cuda",
            "device": f"cuda:{self.device}",
            "query_abi": "streamed-spatial-counts-v1",
            "frame_dtype": self.frame_dtype.name,
            "sum_dtype": self.sum_dtype.name,
            "series_shape": self.series_shape,
            "resident_bytes": sum(end - start for start, end in regions),
            "index_bytes": sum(source.index_nbytes for source in sources),
        }
        self.lock, self.last = threading.Lock(), {}

    def _plan(self, values):
        """Decompose a signed mask into index fields plus residual pixels."""
        if self.pixels >= 512 * 512:
            if self._mask_planner is None:
                self._mask_planner = CudaMaskPlanner(self.det_shape)
                self.backend_metadata["mask_plan_backend"] = "cuda"
            return self._mask_planner(values)
        return plan(values)

    def _cost(self, selection):
        """Tile reads are random access; pixel residuals decode whole streams."""
        return len(selection[0]) + len(selection[2]) * 4

    def _launch(self):
        """Fresh timing events and an error counter for one query; the counter is zeroed on the stream."""
        slot = self._launches % len(self.error_slots)
        if any(item[3] == slot for item in self._inflight):
            raise ValueError("Finish an outstanding detector query before queuing more than eight updates.")
        self._launches += 1
        self._current_launch = self._launches
        self._current_slot = slot
        errors = self.error_slots[slot : slot + 1]
        errors.fill(0)
        begin = cp.cuda.Event()
        begin.record()
        return begin, errors

    def _finish(self, begin, errors, started, info, wait):
        """Record the end of a query; block for it and publish timings unless the caller finishes later."""
        end = cp.cuda.Event()
        end.record()
        slot = self._current_slot
        stream = cp.cuda.get_current_stream()
        errors.data.copy_to_host_async(self._error_pinned.ptr + slot * 4, 4, stream)   # ordered after the kernels
        done = cp.cuda.Event(disable_timing=True)
        done.record()
        if not wait:
            self._inflight.append((begin, end, done, slot, started, info))
            return
        done.synchronize()
        self._publish(begin, end, slot, started, info)

    def _publish(self, begin, end, slot, started, info):
        """Publish timings, or raise when this query or the delta it was planned against failed."""
        launch = info.get("launch", 0)
        built_on_failure = (info.get("incremental") and self._tainted_from is not None and launch > self._tainted_from)
        if int(self._error_host[slot]) or built_on_failure:
            # A failed decode poisons the incremental state: plan the next query in full,
            # and refuse queued queries that were planned as deltas against this result.
            self.previous_mask = None
            if self._tainted_from is None:
                self._tainted_from = launch
            raise ValueError(
                "An encoded stream failed exact decoding; this result is not ready."
            )
        if not info.get("incremental"):
            self._tainted_from = None
        self.last = {
            "gpu_ms": float(cp.cuda.get_elapsed_time(begin, end)),
            "wall_ms": (time.perf_counter() - started) * 1000,
            "acquisitions": len(self.owners),
            **info,
        }

    def finish(self) -> dict:
        """Wait for the oldest query launched with ``wait=False``; raise if a stream failed exact decoding.

        Returns that query's timings (also in ``last``). Results and the incremental
        planning state stay ordered on the stream, so several queries may be in
        flight while the host plans the next one.
        """
        if not self._inflight:
            return dict(self.last)
        begin, end, done, slot, started, info = self._inflight.pop(0)
        done.synchronize()   # this query and its error readback only, not anything queued after it
        self._publish(begin, end, slot, started, info)
        return dict(self.last)

    def _upload(self, selection):
        """Copy a plan into the launch's pinned slot, then onto the device on the query stream."""
        stream = cp.cuda.get_current_stream()
        for target, staging, values in zip(
            (
                self.selected_fields,
                self.field_coefficients,
                self.selected_pixels,
                self.pixel_coefficients,
            ),
            (views[self._current_slot] for views in self._staging_views),
            selection,
        ):
            if len(values):
                staging[: len(values)] = values
                target[: len(values)].set(staging[: len(values)], stream=stream)

    def _add_residuals(self, bits, count, result, errors):
        """Decode only the residual pixels' streams and add their signed counts to ``result``."""
        groups = math.ceil(count / 32) * math.ceil(
            self.max_scans / self.interval / self.residual_warps
        )
        self.kernels[f"residual_u{bits}"](
            (groups, self.chunk_count),
            (32 * self.residual_warps,),
            (
                self.descriptors,
                self.selected_pixels,
                self.pixel_coefficients,
                np.uint32(count),
                result,
                errors,
                np.uint64(self.n_frames),
                np.uint32(self.pixels),
                np.uint32(self.interval),
            ),
        )

    def detector_total(self):
        """Return one exact uint64 valid-pixel sum of all patterns per acquisition."""
        with self.lock, cp.cuda.Device(self.device):
            result = cp.stack([source.detector_total_device() for source in self.index_owners])
            return result[0] if not self.series_shape else result

    def mean_dp(self):
        """Return one valid-pixel mean diffraction pattern per acquisition."""
        with cp.cuda.Device(self.device):
            return (self.detector_total() / self.n_frames).astype(cp.float32)

    def reduce_frames(self, indices, reduce="mean"):
        """Reduce a scan selection without expanding the full acquisition."""
        indices = np.sort(np.asarray(indices, dtype=np.int64).reshape(-1))
        if not len(indices) or indices[0] < 0 or indices[-1] >= self.n_frames:
            raise ValueError(f"Select scan indices from 0 through {self.n_frames - 1}.")
        if reduce not in {"sum", "mean", "max"}:
            raise ValueError(f"Unknown frame reduction {reduce!r}; use mean, sum, or max.")
        with self.lock, cp.cuda.Device(self.device):
            results = []
            for source in self.index_owners:
                total = cp.zeros(self.det_shape, cp.uint64)
                # Group by codec block so sparse selections decode each block once.
                boundaries = np.flatnonzero(np.diff(indices // 512)) + 1
                for selected in np.split(indices, boundaries):
                    first = int(selected[0] // 512 * 512)
                    stop = min(first + 512, self.n_frames)
                    raw = source.decode_scan_range_device(first, stop)
                    values = raw[cp.asarray(selected - first)]
                    if reduce == "max":
                        cp.maximum(total, values.max(axis=0), out=total)
                    else:
                        total += values.sum(axis=0, dtype=cp.uint64)
                total *= source.valid.reshape(self.det_shape)
                # The mean divides the exact total in float64 and rounds once, as MPS does;
                # rounding the total to float32 first loses counts above 2^24.
                results.append((total / len(indices)).astype(cp.float32) if reduce == "mean" else total)
            result = cp.stack(results).get()
            return result[0] if not self.series_shape else result

    def reduce_frames_exact(self, indices):
        """Return the integer scan-ROI sum used by the public detector API."""
        return self.reduce_frames(indices, reduce="sum")

    def reduce_frames_max(self, indices):
        """Return the integer scan-ROI maximum used by the public detector API."""
        return self.reduce_frames(indices, reduce="max")

    def weighted_sum_exact(self, weights):
        """Return exact uint64 per-scan sums for nonnegative integer pixel weights.

        The residual decoders add 32 pixels' ``weight * count`` in int32, so the
        weights go through in binary digits small enough to keep that exact
        (``weight_digits``, as on MPS): one query per digit, shifted back and
        added. Row and column weights below 1024 on uint16 counts are one query.
        """
        values = np.asarray(weights)
        if values.shape != self.det_shape or values.dtype.kind not in "uib":
            raise ValueError(
                f"Detector weights must be nonnegative integers with shape {self.det_shape}."
            )
        if np.any(values < 0) or np.any(values > np.iinfo(np.int32).max):
            raise ValueError("Detector weights must fit nonnegative int32 values.")
        shape = (*self.series_shape, *self.scan_shape)
        with cp.cuda.Device(self.device):
            result = cp.zeros(shape, cp.uint64)
            for shift, digit in weight_digits(values, np.iinfo(self.frame_dtype).max):
                part = self._weighted_query(digit.astype(np.int32), shape)
                result += part << np.uint64(shift)
            return result

    def _weighted_query(self, values, shape):
        """Exact uint64 sums of one digit of the weights: index fields plus residual pixels."""
        selection = self._plan(values)
        fields, _, pixels, _ = selection
        with self.lock, cp.cuda.Device(self.device):
            started = time.perf_counter()
            result = self._output(None, shape, np.dtype(np.uint64))
            begin, errors = self._launch()
            self._upload(selection)
            self.kernels["index_u64"](
                ((self.max_scans + 127) // 128, self.chunk_count),
                (128,),
                (
                    self.descriptors,
                    self.selected_fields,
                    self.field_coefficients,
                    np.uint32(len(fields)),
                    result,
                    result,
                    np.uint64(self.n_frames),
                    np.uint32(self.fields),
                    np.uint32(self.interval),
                    np.int32(0),
                ),
            )
            if len(pixels):
                self._add_residuals(64, len(pixels), result, errors)
            info = {
                "query_launches": 1 + bool(len(pixels)),
                "residual_pixels": len(pixels),
                "spatial_fields": len(fields),
                "incremental": False,
                "launch": self._current_launch,
                "weighted": True,
            }
            self._finish(begin, errors, started, info, True)
            return result

    def center_of_mass(self, mask=None):
        """Return exact count-weighted detector centers without dense expansion."""
        valid = np.ones(self.det_shape, dtype=np.uint32)
        if mask is not None:
            selected = np.asarray(mask, dtype=bool)
            if selected.shape != self.det_shape:
                raise ValueError(
                    f"Detector mask shape {selected.shape} does not match {self.det_shape}."
                )
            valid *= selected
        rows, cols = np.indices(self.det_shape, dtype=np.uint32)
        total = self.weighted_sum_exact(valid)
        row_moment = self.weighted_sum_exact(valid * rows)
        col_moment = self.weighted_sum_exact(valid * cols)
        denominator = cp.maximum(total.astype(cp.float64), 1.0)
        com_row = (row_moment.astype(cp.float64) / denominator).astype(cp.float32)
        com_col = (col_moment.astype(cp.float64) / denominator).astype(cp.float32)
        return com_col, com_row

    def masked_sum_native(self, mask, *, out=None, wait=True, block_stride=1):
        """Return exact counts for a binary mask, planned as a delta of the previous mask when cheaper."""
        values = np.asarray(mask)
        if values.shape != self.det_shape or not np.all((values == 0) | (values == 1)):
            raise ValueError(
                f"Provide a binary detector mask with shape {self.det_shape}."
            )
        values = values.astype(np.int32)
        with self.lock, cp.cuda.Device(self.device):
            started = time.perf_counter()
            selection, delta = None, False
            stride = int(block_stride)
            if stride < 1:
                raise ValueError("block_stride counts 512-scan blocks; use 1 or more.")
            self.block_stride = stride   # the paired launch wrappers read it
            if stride != self.previous_stride:
                # The baseline holds sums on the previous stride's rows only; a query on
                # other rows must start from a full plan.
                self.previous_mask = None
            if self.previous_mask is not None:
                change = self._plan(values - self.previous_mask)
                # A small change (a nudged detector) is always cheaper than re-planning the
                # whole mask, so the full plan is not even computed for it.
                if len(change[2]) <= 1024 and len(change[0]) <= 256:
                    selection, delta = change, True
                else:
                    selection = self._plan(values)
                    if self._cost(change) < self._cost(selection):
                        selection, delta = change, True
            if selection is None:
                selection = self._plan(values)
            fields, _, pixels, _ = selection
            result = self._output(
                out, (*self.series_shape, *self.scan_shape), self.sum_dtype
            )
            begin, errors = self._launch()
            self._upload(selection)
            bits = self.sum_dtype.itemsize * 8
            if delta and not len(fields):
                # A thin change ring touches no whole index leaf: the index pass would only
                # copy the previous sums, which a device memcpy does in a tenth of the time.
                cp.copyto(result, self.previous)
            else:
                self.kernels[f"index_u{bits}"](
                    ((self.max_scans + 127) // 128, self.chunk_count),
                    (128,),
                    (
                        self.descriptors,
                        self.selected_fields,
                        self.field_coefficients,
                        np.uint32(len(fields)),
                        self.previous,
                        result,
                        np.uint64(self.n_frames),
                        np.uint32(self.fields),
                        np.uint32(self.interval),
                        np.int32(delta),
                    ),
                )
            if len(pixels):
                self._add_residuals(bits, len(pixels), result, errors)
            cp.copyto(self.previous, result)   # ordered on the stream after the sums; the next delta plan reads it
            self.previous_mask = values.copy()
            self.previous_stride = stride
            info = {
                "query_launches": 1 + bool(len(pixels)),
                "residual_pixels": len(pixels),
                "spatial_fields": len(fields),
                "incremental": delta,
                "launch": self._current_launch,
                "block_stride": stride,
            }
            self._finish(begin, errors, started, info, wait)
            return result

    def frame_native(self, index, *, out=None, wait=True):
        """Return the scan position's native counts for every acquisition in one launch."""
        index = int(index)
        if not 0 <= index < self.n_frames:
            raise IndexError(f"Choose a scan index inside {self.scan_shape}.")
        with self.lock, cp.cuda.Device(self.device):
            started = time.perf_counter()
            result = self._output(
                out, (*self.series_shape, *self.det_shape), self.frame_dtype
            )
            begin, errors = self._launch()
            self.kernels[f"frame_u{self.frame_dtype.itemsize * 8}"](
                ((self.pixels + 127) // 128, self.chunk_count),
                (128,),
                (
                    self.descriptors,
                    result,
                    errors,
                    np.uint32(index),
                    np.uint32(self.pixels),
                    np.uint32(self.interval),
                ),
            )
            info = {"query_launches": 1, "launch": self._current_launch}
            self._finish(begin, errors, started, info, wait)
            return result
