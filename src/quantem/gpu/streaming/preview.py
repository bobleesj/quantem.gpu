"""Bounded detector queries on the received part of a live ANS acquisition."""

import copy
import threading

import numpy as np

from quantem.gpu.detector import prepare
from quantem.gpu.detector.cuda.dense import CudaKernelCompute
from quantem.gpu.device.cuda_runtime import cp


class LiveDetectorSession:
    """Borrow a live scan without representing unacquired positions as counts.

    Completed compression blocks stay ANS encoded. Queries borrow the received
    prefix of the current block while acquisition appends beyond that prefix.
    Scientific operations use CUDA; only the resulting 2D arrays reach the host.
    """

    def __init__(self, acquisition):
        self.acquisition = acquisition
        self.scan_shape = acquisition.scan_shape
        self.detector_shape = acquisition.detector_shape
        self.shape = acquisition.shape
        self.device = f"cuda:{acquisition.device}"
        self.metadata = acquisition.metadata
        self.lock = threading.RLock()
        self.frame_lock = threading.RLock()
        self.received = -1
        self.encoded = 0
        self.session = self.source = None
        self.tail = None
        self.last_mask = self.last_image = None
        self.last_received = -1
        with cp.cuda.Device(acquisition.device):
            self.stream = cp.cuda.Stream(non_blocking=True)
            self.frame_stream = cp.cuda.Stream(non_blocking=True)
        with acquisition.lock:
            if acquisition._retired or acquisition.source.is_released:
                raise ValueError("This live acquisition was retired or released; select another acquisition.")
            acquisition._query_sessions += 1
        self._closed = False

    def _snapshot(self):
        if self._closed:
            raise ValueError("This live detector session is closed; open another session.")
        live = self.acquisition
        with live.lock:
            if live.source.is_released:
                raise ValueError("This live acquisition was released; start another acquisition.")
            if self.received == live.received:
                return
            if live.updated is not None:
                self.stream.wait_event(live.updated)
            ready = live.source.ready_scans
            if ready != self.encoded:
                if self.session is not None:
                    self.session.close()
                    self.session = None
                source = copy.copy(live.source)
                source.shape = (1, ready, *self.detector_shape)
                source.chunks = list(live.source.chunks)
                source._detector_total = live.source._detector_total.copy()
                self.source = source
                self.encoded = ready
            # Received patterns are immutable. The receiver retires the staging
            # allocation when it encodes a block; this view keeps it alive.
            self.tail = None if not live.staged else live.staging[:live.staged]
            self.received = live.received

    def _encoded_session(self):
        if self.session is None:
            self.session = prepare(self.source)
        return self.session

    def masked_sum(self, mask):
        """Exact integer detector sums, rounded to float32 for display."""
        with self.lock, cp.cuda.Device(self.acquisition.device), self.stream:
            self._snapshot()
            unchanged = self.last_mask is not None and np.array_equal(mask, self.last_mask)
            if self.last_received == self.received and unchanged:
                return self.last_image
            result = self.last_image.copy() if unchanged else np.full(self.scan_shape, np.nan, np.float32)
            flat = result.reshape(-1)
            first = self.last_received if unchanged else 0
            # An unchanged aperture only needs newly received counts. Decode a
            # bounded interval if those counts already crossed an ANS boundary.
            # Keep the full encoded reduction for a changed aperture.
            if self.encoded and not unchanged:
                flat[:self.encoded] = self._encoded_session().masked_sum(mask).reshape(-1)
                first = self.encoded
            valid_mask = np.asarray(mask, dtype=bool) & self.acquisition.source.valid_pixels
            step = max(1, (32 << 20) // (np.prod(self.detector_shape) * self.acquisition.dtype.itemsize))
            for start in range(first, self.encoded, step):
                stop = min(start + step, self.encoded)
                counts = self.source.decode_scan_range_device(start, stop)
                flat[start:stop] = CudaKernelCompute(counts).masked_sum(valid_mask).reshape(-1)
            if self.tail is not None:
                start = max(first, self.encoded)
                if start < self.received:
                    counts = self.tail[start - self.encoded:]
                    flat[start:self.received] = CudaKernelCompute(counts).masked_sum(valid_mask).reshape(-1)
            # Reuse only an unchanged acquisition and detector. Never mutate a
            # result already handed to the widget's asynchronous transport.
            self.last_mask = np.array(mask, copy=True)
            self.last_image, self.last_received = result, self.received
            return result

    def frame(self, index: int):
        """Return a measured DP, or NaN where that position has not arrived."""
        if not 0 <= index < self.acquisition.total:
            raise IndexError(f"Scan index {index} is outside {self.scan_shape}.")
        live = self.acquisition
        # Point queries must not wait behind a virtual-image reduction or
        # rebuild its scan-sized query buffers when a compression block arrives.
        with self.frame_lock, cp.cuda.Device(live.device), self.frame_stream:
            if self._closed:
                raise ValueError("This live detector session is closed; open another session.")
            with live.lock:
                if live.source.is_released:
                    raise ValueError("This live acquisition was released; start another acquisition.")
                if index >= live.received:
                    return np.full(self.detector_shape, np.nan, np.float32)
                if live.updated is not None:
                    self.frame_stream.wait_event(live.updated)
                ready = live.source.ready_scans
                if index < ready:
                    source = copy.copy(live.source)
                    source.chunks = list(live.source.chunks)
                else:
                    frame = live.staging[index - ready]
                    valid = live.source.valid
            if index < ready:
                return source.decode_scan_range_device(index, index + 1)[0].get()
            return (frame * valid).get()

    def mean_dp(self):
        """Mean of arrived positions only, with exact integer accumulation."""
        return self.reduce_frames(range(self.acquisition.received))

    def reduce_frames(self, indices, mode: str = "mean"):
        """Reduce only acquired positions in the selection; omit missing ones."""
        if mode not in {"mean", "sum", "max"}:
            raise ValueError("Use mode='mean', 'sum' or 'max'.")
        with self.lock, cp.cuda.Device(self.acquisition.device), self.stream:
            self._snapshot()
            selected = np.asarray(list(indices), np.int64)
            if selected.size and (selected.min() < 0 or selected.max() >= self.acquisition.total):
                raise IndexError("A selected position is outside the scan.")
            selected = selected[selected < self.received]
            if not selected.size:
                return np.full(self.detector_shape, np.nan, np.float32)
            encoded = selected[selected < self.encoded]
            tail = selected[selected >= self.encoded] - self.encoded
            values = cp.zeros(self.detector_shape, cp.uint64)
            if encoded.size:
                values = cp.asarray(self._encoded_session().reduce_frames(encoded, mode="max" if mode == "max" else "sum"), dtype=cp.uint64)
            if tail.size:
                pending = self.tail[cp.asarray(tail)]
                reduced = pending.max(axis=0) if mode == "max" else pending.sum(axis=0, dtype=cp.uint64)
                values = cp.maximum(values, reduced) if mode == "max" else values + reduced
            values *= self.acquisition.source.valid
            if mode == "mean":
                return (values.astype(cp.float64) / selected.size).astype(cp.float32).get()
            return values.get()

    def close(self):
        """Release query buffers and, for the final viewer, any discarded counts."""
        with self.lock, self.frame_lock, cp.cuda.Device(self.acquisition.device), self.stream:
            if self._closed:
                return
            self.stream.synchronize()
            self.frame_stream.synchronize()
            if self.session is not None:
                self.session.close()
            self.session = self.source = self.tail = None
            self.last_mask = self.last_image = None
            self.last_received = -1
            self.received, self.encoded = -1, 0
            self._closed = True
            with self.acquisition.lock:
                self.acquisition._query_sessions -= 1
                if self.acquisition._retired and not self.acquisition._query_sessions:
                    self.acquisition.release()
