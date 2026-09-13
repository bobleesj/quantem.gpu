"""Joint translated detector queries over runtime-shaped encoded acquisitions."""

import math
import threading
import time

import numpy as np

from quantem.gpu.detector.backends.cuda.series import CudaSeriesCompute

from .streamed import StreamedCounts, field_count, kernels


def plan(mask: np.ndarray):
    """Express an exact signed mask as 32/8-pixel tiles plus pixel residuals."""
    rows, cols = mask.shape
    nr, nc = math.ceil(rows / 8), math.ceil(cols / 8)
    padded = np.zeros((nr * 8, nc * 8), np.int32)
    padded[:rows, :cols] = mask
    tiles = padded.reshape(nr, 8, nc, 8).transpose(0, 2, 1, 3)
    counts = np.stack([(tiles == value).sum(axis=(2, 3)) for value in (0, 1, -1)])
    leaves = np.array([0, 1, -1], np.int32)[counts.argmax(axis=0)]
    residual = mask - np.repeat(np.repeat(leaves, 8, axis=0), 8, axis=1)[:rows, :cols]
    cr, cc = math.ceil(nr / 4), math.ceil(nc / 4)
    coarse = np.zeros((cr * 4, cc * 4), np.int32)
    coarse[:nr, :nc] = leaves
    tiles = coarse.reshape(cr, 4, cc, 4).transpose(0, 2, 1, 3)
    counts = np.stack([(tiles == value).sum(axis=(2, 3)) for value in (0, 1, -1)])
    roots = np.array([0, 1, -1], np.int32)[counts.argmax(axis=0)]
    leaves = leaves - np.repeat(np.repeat(roots, 4, axis=0), 4, axis=1)[:nr, :nc]
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
        import cupy as cp

        from quantem.gpu.io.backends.cuda._ans import CudaANSResidentCounts

        self.owners = tuple(acquisitions)
        sources = []
        for item in acquisitions:
            source = (
                item.data
                if hasattr(item, "_fields") and "data" in item._fields
                else item
            )
            if not isinstance(source, StreamedCounts):
                metadata = getattr(item, "metadata", {})
                valid = np.ones(source.shape[2:], bool)
                from quantem.gpu.io._hot_pixels import correction_is_applied

                corrected = correction_is_applied(metadata)
                if metadata.get("pixel_mask") is not None and not corrected:
                    valid &= np.asarray(metadata["pixel_mask"]) == 0
                excluded = metadata.get("excluded_detector_pixels", ())
                if len(excluded):
                    valid.ravel()[np.asarray(excluded, np.intp)] = False
                source = StreamedCounts.index_encoded(source, valid)
            sources.append(source)
        self.index_owners = tuple(sources)
        if not sources or not all(isinstance(s, StreamedCounts) for s in sources):
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
        self.interval = min(s.interval for s in sources)
        self.fields = field_count(self.det_shape)
        self.frame_dtype = np.result_type(*[s.dtype for s in sources])
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
                    0,
                    source.interval,
                    0,
                    *([0] * 9),
                ]
                if source.native_source is not None:
                    native = source.native_source
                    row[11] = 1 if isinstance(native, CudaANSResidentCounts) else 2
                    row[13] = getattr(native, "scale", 0)
                    row[14 : 14 + len(native._arrays)] = [
                        a.data.ptr for a in native._arrays
                    ]
                    self.keepalive.extend(native._arrays)
                rows.append(row)
                self.max_scans = max(self.max_scans, chunk.scans)
        if len(rows) > 65535:
            raise ValueError(
                "This chunk count exceeds the joint CUDA launch limit; use larger loading chunks."
            )
        self.residual_warps = min(4, math.ceil(self.max_scans / self.interval))
        self.valid_pixels = np.stack([s.valid_pixels for s in sources])
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
        self.previous_mask = None
        self.block_stride = 1
        self.previous_stride = 1
        self._inflight = []   # (begin, end, done, errors_slot, started, info) for queries launched with wait=False
        self._launches = 0
        self._tainted_from = None   # launch index of a failed query; later incremental queries built on it
        import cupy as cp

        with cp.cuda.Device(self.device):
            # Pinned staging so selection uploads and error readbacks are asynchronous: a
            # synchronous memcpy waits for every kernel queued ahead of it, which would
            # serialise queries meant to overlap the host's planning of the next one.
            self._staging = [cp.cuda.alloc_pinned_memory(n * 4) for n in (self.fields, self.fields, self.pixels, self.pixels)]
            self._staging_views = [np.frombuffer(m, dtype, n) for m, dtype, n in zip(self._staging, (np.uint32, np.int32, np.uint32, np.int32), (self.fields, self.fields, self.pixels, self.pixels))]
            self._error_pinned = cp.cuda.alloc_pinned_memory(len(self.error_slots) * 4)
            self._error_host = np.frombuffer(self._error_pinned, np.uint32, len(self.error_slots))
        self.keepalive.extend(
            (
                self.descriptors,
                self.errors,
                self.previous,
                self.selected_fields,
                self.field_coefficients,
                self.selected_pixels,
                self.pixel_coefficients,
            )
        )
        regions = {(a.data.ptr, a.data.ptr + a.nbytes) for a in self.keepalive}
        self.regions = np.asarray(sorted(regions), np.uint64)
        self.backend_metadata = {
            "backend": "cuda",
            "device": f"cuda:{self.device}",
            "query_abi": "streamed-spatial-counts-v1",
            "frame_dtype": self.frame_dtype.name,
            "sum_dtype": self.sum_dtype.name,
            "series_shape": self.series_shape,
            "resident_bytes": sum(b - a for a, b in regions),
            "index_bytes": sum(s.index_nbytes for s in sources),
        }
        self.lock, self.last = threading.Lock(), {}

    def _plan(self, values):
        """Decompose a signed mask into index fields plus residual pixels."""
        return plan(values)

    def _cost(self, selection):
        """Tile reads are random access; pixel residuals decode whole streams."""
        return len(selection[0]) + len(selection[2]) * 4

    def _launch(self):
        """Fresh timing events and an error counter for one query; the counter is zeroed on the stream."""
        import cupy as cp

        slot = self._launches % len(self.error_slots)
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
        import cupy as cp

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
        import cupy as cp

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

    def mean_dp(self):
        """Return one valid-pixel mean diffraction pattern per acquisition."""
        import cupy as cp

        with self.lock, cp.cuda.Device(self.device):
            result = cp.empty(
                (len(self.index_owners), *self.det_shape), cp.float32
            )
            for index, source in enumerate(self.index_owners):
                total = source.detector_total_device()
                result[index] = (total / self.n_frames).astype(cp.float32)
            return result[0] if not self.series_shape else result

    def _weighted_sum_native(self, weights):
        """Return exact per-scan sums for nonnegative integer pixel weights."""
        import cupy as cp

        values = np.asarray(weights)
        if values.shape != self.det_shape or values.dtype.kind not in "uib":
            raise ValueError(
                f"Detector weights must be nonnegative integers with shape {self.det_shape}."
            )
        if np.any(values < 0) or np.any(values > np.iinfo(np.int32).max):
            raise ValueError("Detector weights must fit nonnegative int32 values.")
        values = values.astype(np.int32, copy=False)
        selection = self._plan(values)
        fi, _fc, pi, _pc = selection
        with self.lock, cp.cuda.Device(self.device):
            started = time.perf_counter()
            result = self._output(
                None, (*self.series_shape, *self.scan_shape), np.dtype(np.uint64)
            )
            begin, errors = self._launch()
            stream = cp.cuda.get_current_stream()
            for target, staging, source in zip(
                (
                    self.selected_fields,
                    self.field_coefficients,
                    self.selected_pixels,
                    self.pixel_coefficients,
                ),
                self._staging_views,
                selection,
            ):
                if len(source):
                    staging[: len(source)] = source
                    target[: len(source)].set(staging[: len(source)], stream=stream)
            self.kernels["index_u64"](
                ((self.max_scans + 127) // 128, self.chunk_count),
                (128,),
                (
                    self.descriptors,
                    self.selected_fields,
                    self.field_coefficients,
                    np.uint32(len(fi)),
                    result,
                    result,
                    np.uint64(self.n_frames),
                    np.uint32(self.fields),
                    np.uint32(self.interval),
                    np.int32(0),
                ),
            )
            if len(pi):
                groups = math.ceil(len(pi) / 32) * math.ceil(
                    self.max_scans / self.interval / self.residual_warps
                )
                self.kernels["residual_u64"](
                    (groups, self.chunk_count),
                    (32 * self.residual_warps,),
                    (
                        self.descriptors,
                        self.selected_pixels,
                        self.pixel_coefficients,
                        np.uint32(len(pi)),
                        result,
                        errors,
                        np.uint64(self.n_frames),
                        np.uint32(self.pixels),
                        np.uint32(self.interval),
                    ),
                )
            self._finish(
                begin,
                errors,
                started,
                dict(
                    query_launches=1 + bool(len(pi)),
                    residual_pixels=len(pi),
                    spatial_fields=len(fi),
                    incremental=False,
                    launch=self._current_launch,
                    weighted=True,
                ),
                True,
            )
            return result

    def center_of_mass(self, mask=None):
        """Return exact count-weighted detector centers without dense expansion."""
        import cupy as cp

        valid = np.ones(self.det_shape, dtype=np.uint32)
        if mask is not None:
            selected = np.asarray(mask, dtype=bool)
            if selected.shape != self.det_shape:
                raise ValueError(
                    f"Detector mask shape {selected.shape} does not match {self.det_shape}."
                )
            valid *= selected
        rows, cols = np.indices(self.det_shape, dtype=np.uint32)
        total = self._weighted_sum_native(valid)
        row_moment = self._weighted_sum_native(valid * rows)
        col_moment = self._weighted_sum_native(valid * cols)
        denominator = cp.maximum(total.astype(cp.float64), 1.0)
        com_row = (row_moment.astype(cp.float64) / denominator).astype(cp.float32)
        com_col = (col_moment.astype(cp.float64) / denominator).astype(cp.float32)
        return com_col, com_row

    def masked_sum_native(self, mask, *, out=None, wait=True, block_stride=1):
        import cupy as cp

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
            fi, _fc, pi, _pc = selection
            result = self._output(
                out, (*self.series_shape, *self.scan_shape), self.sum_dtype
            )
            begin, errors = self._launch()
            stream = cp.cuda.get_current_stream()
            for target, staging, source in zip(
                (
                    self.selected_fields,
                    self.field_coefficients,
                    self.selected_pixels,
                    self.pixel_coefficients,
                ),
                self._staging_views,
                selection,
            ):
                if len(source):
                    staging[: len(source)] = source
                    target[: len(source)].set(staging[: len(source)], stream=stream)
            bits = self.sum_dtype.itemsize * 8
            u32 = np.uint32
            if delta and not len(fi):
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
                        u32(len(fi)),
                        self.previous,
                        result,
                        np.uint64(self.n_frames),
                        u32(self.fields),
                        u32(self.interval),
                        np.int32(delta),
                    ),
                )
            if len(pi):
                groups = math.ceil(len(pi) / 32) * math.ceil(
                    self.max_scans / self.interval / self.residual_warps
                )
                self.kernels[f"residual_u{bits}"](
                    (groups, self.chunk_count),
                    (32 * self.residual_warps,),
                    (
                        self.descriptors,
                        self.selected_pixels,
                        self.pixel_coefficients,
                        u32(len(pi)),
                        result,
                        errors,
                        np.uint64(self.n_frames),
                        u32(self.pixels),
                        u32(self.interval),
                    ),
                )
            cp.copyto(self.previous, result)   # ordered on the stream after the sums; the next delta plan reads it
            self.previous_mask = values.copy()
            self.previous_stride = stride
            self._finish(begin, errors, started, dict(query_launches=1 + bool(len(pi)), residual_pixels=len(pi),
                                                      spatial_fields=len(fi), incremental=delta, launch=self._current_launch,
                                                      block_stride=stride), wait)
            return result

    def frame_native(self, index, *, out=None, wait=True):
        import cupy as cp

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
            self._finish(begin, errors, started, dict(query_launches=1, launch=self._current_launch), wait)
            return result
