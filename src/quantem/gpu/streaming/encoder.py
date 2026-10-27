"""Encode one scan from detector batches as they arrive: the GPU backend of live acquisition.

A receiver (the Holoscan RDMA pipeline, outside this package) hands over consecutive batches of
uncompressed diffraction patterns that are already on the GPU, each labelled by the image id of its
first pattern. ``LiveEncoder`` treats each batch exactly as ``io.load`` treats a block read from an HDF5 file:
the detector's flagged pixels are corrected by the same CUDA corrector and the
counts are appended to the same lossless resident store. ``finish`` returns the
``Dataset4dstemGPU`` that ``io.load`` would have returned for the same counts, so
every downstream algorithm (DirectPtycho, detector products) sees no difference
between a live acquisition and a recorded one.

No streaming is handled here: receiving, queues, threads and the acquisition lifecycle belong to the
caller; this module owns scan placement, ANS encoding and residency only.
"""

import math
import threading
import time
import uuid
import weakref

import numpy as np

from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.formats.qem.snapshot import INTEGER_CODEC
from quantem.gpu.io import arrays
from quantem.gpu.io.dataset import Dataset4dstemGPU, resident_metadata
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.resident.cuda.hot_pixels import CUDAHotPixelCorrector
from quantem.gpu.resident.hot_pixels import normalize_hot_pixel_correction


class IncompleteAcquisition(RuntimeError):
    """The series ended before every scan position arrived."""


class LiveEncoder:
    """Assemble one scan from consecutive decoded batches into an ``io.load``-equivalent dataset.

    ``scan_shape`` and ``detector_shape`` come from the acquisition (the detector's START
    message gives the detector shape; the scan shape is configured, because the
    detector announces only an image count). ``pixel_mask`` is the detector's START
    pixel mask (nonzero = flagged), handled by ``hot_pixel_correction`` exactly as
    ``io.load`` handles the mask stored in a master file. Batches must arrive in scan
    order: image id ``k`` is scan position ``(k // cols, k % cols)``.
    """

    CHUNK_PATTERNS = 4096  # patterns encoded together; see __init__

    def __init__(self, scan_shape, detector_shape, *, pixel_mask=None, device="cuda:0",
                 hot_pixel_correction: str = "median", dtype="uint16", metadata=None):
        self.scan_shape = tuple(int(side) for side in scan_shape)
        self.detector_shape = tuple(int(side) for side in detector_shape)
        self.shape = (*self.scan_shape, *self.detector_shape)
        self.dtype = np.dtype(dtype)
        self.device = int(str(device).split(":")[-1])
        self.pixel_mask = None if pixel_mask is None else np.asarray(pixel_mask)
        self.metadata = dict(metadata or {})
        self.identifier = uuid.uuid4().hex
        self.lock = threading.RLock()
        self._query_sessions = 0
        self._retired = False
        self.updated = None
        self.received = 0
        # Small batches wait here and are encoded together: each encode call costs ~15 ms whatever its size, and
        # ANS codes each pixel over 512-scan runs, so chunks of 4096 keep both the rate and the compression of
        # encoding the whole frame (64-pattern batches alone: 4,500 patterns/s and 58% larger).
        self.staging, self.staged = None, 0
        self.started = time.perf_counter()
        method = normalize_hot_pixel_correction(hot_pixel_correction)
        with cp.cuda.Device(self.device):
            self.corrector = CUDAHotPixelCorrector(self.pixel_mask, method)
            valid = np.ones(self.detector_shape, bool)
            # As in io.load: pixels the corrector does not repair are excluded from scientific sums.
            if self.pixel_mask is not None and not self.corrector.record["applied"]:
                valid &= self.pixel_mask == 0
            self.source = StreamedCounts(self.shape, self.dtype, valid)

    @property
    def total(self) -> int:
        return math.prod(self.scan_shape)

    @property
    def complete(self) -> bool:
        return self.received == self.total

    def memory_snapshot(self) -> dict:
        """Describe published encoding progress and owned buffers without device work.

        Progress reflects completed batches. The staging tail has arrived but
        has not yet been ANS encoded. A source filename is supplied only when
        recorded in the acquisition metadata.

        Returns
        -------
        dict
            Identity, source provenance, encoding, completed DP counts and
            owned buffer sizes in bytes. Index bytes are part of resident bytes.

        Examples
        --------
        >>> acquisition.memory_snapshot()["encoded"]
        """
        with self.lock:
            return {
                "id": self.identifier,
                "name": str(self.metadata.get("name") or self.metadata.get("series") or "Acquisition"),
                "source_path": str(self.metadata.get("source_path") or ""),
                "device": f"cuda:{self.device}",
                "shape": list(self.shape),
                "dtype": self.dtype.name,
                "encoding": "ANS",
                "received": self.received,
                "encoded": 0 if self.source.is_released else self.source.ready_scans,
                "total": self.total,
                "resident_bytes": int(self.source.nbytes),
                "index_bytes": int(self.source.index_nbytes),
                "staging_bytes": 0 if self.staging is None else int(self.staging.nbytes),
            }

    def add(self, first_image_id: int, patterns) -> None:
        """Correct and retain one batch of decoded patterns ``(n, det_row, det_col)``.

        Flagged pixels are corrected in place in ``patterns`` before encoding, as ``io.load``
        does with its decode buffer; pass a batch this encoder may modify (copy a borrowed receiver
        buffer first). ``first_image_id`` must be the next expected image: the resident store encodes
        consecutive scan chunks, and a gap or repeat is a transport fault the receiver
        must resolve (reorder or reject) before handing the batch over.
        """
        with self.lock, cp.cuda.Device(self.device):
            self._add(first_image_id, patterns)
            self.updated = cp.cuda.Event()
            self.updated.record()

    def _add(self, first_image_id: int, patterns) -> None:
        if self.source.is_released:
            raise ValueError("This live acquisition was released; start another acquisition.")
        if first_image_id != self.received:
            raise ValueError(
                f"Expected image {self.received}, got a batch starting at {first_image_id}: "
                "batches must arrive in scan order without gaps or repeats."
            )
        raw = cp.asarray(patterns)
        if raw.dtype != self.dtype or raw.ndim != 3 or raw.shape[1:] != self.detector_shape:
            raise ValueError(
                f"Batch must be {self.dtype} (n, {self.detector_shape[0]}, {self.detector_shape[1]}); "
                f"got {raw.dtype} {raw.shape}."
            )
        if raw.device.id != self.device:
            raise ValueError(f"Batch is on cuda:{raw.device.id}; this acquisition lives on cuda:{self.device}.")
        count = raw.shape[0]
        if self.received + count > self.total:
            raise ValueError(f"Batch overruns the {self.total}-pattern scan.")
        with cp.cuda.Device(self.device):
            raw = cp.ascontiguousarray(raw)
            self.corrector.apply(raw)
            offset = 0
            while offset < count:
                if self.staged == 0 and count - offset >= self.CHUNK_PATTERNS:
                    # whole chunks of a large batch are encoded where they are, without a copy
                    whole = (count - offset) // self.CHUNK_PATTERNS * self.CHUNK_PATTERNS
                    self.source.append(raw[offset:offset + whole])
                    offset += whole
                    continue
                if self.staging is None:
                    self.staging = cp.empty((self.CHUNK_PATTERNS, *self.detector_shape), self.dtype)
                take = min(count - offset, self.CHUNK_PATTERNS - self.staged)
                self.staging[self.staged:self.staged + take] = raw[offset:offset + take]
                self.staged += take
                offset += take
                if self.staged == self.CHUNK_PATTERNS:
                    self._encode_staged()
            if self.received + count == self.total and self.staged:
                self._encode_staged()
            # The batch may belong to another thread's stream (a receiver's buffer, freed when the caller drops
            # it): the queued correction and staging copy must have read it before add returns, or the memory
            # pool can hand the block back to that stream and the next batch overwrites it mid-copy.
            cp.cuda.get_current_stream().synchronize()
        self.received += count

    def _encode_staged(self) -> None:
        """Encode the staged patterns as one chunk; the next chunk stages into a new buffer."""
        self.source.append(self.staging[:self.staged])
        # append only queues work on this stream; a new buffer keeps the queued encode's input intact
        self.staging, self.staged = None, 0

    def finish(self) -> Dataset4dstemGPU:
        """Return the completed acquisition as the dataset ``io.load`` returns for the same counts."""
        if not self.complete:
            raise IncompleteAcquisition(
                f"{self.received} of {self.total} patterns arrived; an incomplete scan is not returned as data."
            )
        metadata = dict(self.metadata)
        metadata.update(
            resident_metadata(self.shape, self.dtype, self.source.nbytes, backend="cuda"),
            source_dtype=self.dtype.name,
            acquisition_id=self.identifier,
            source_read_passes=1,
            source_logical_tensor_bytes=math.prod(self.shape) * self.dtype.itemsize,
            index_bytes=self.source.index_nbytes,
            pixel_mask=self.pixel_mask,
            resident_profile=INTEGER_CODEC,
            **arrays.correction_metadata(self.corrector.record),
            load_timings={"resident_ready_seconds": time.perf_counter() - self.started},
        )
        return Dataset4dstemGPU(self.source, metadata)

    def detector_session(self):
        """Query arrived positions while the acquisition is still receiving.

        Unacquired positions return NaN, distinct from a measured zero. Queries
        borrow immutable ANS chunks and the bounded, unencoded tail. Close the
        session when its viewer retires so discarded counts can be released.
        """
        from quantem.gpu.streaming.preview import LiveDetectorSession

        return LiveDetectorSession(self)

    def release(self) -> None:
        """Free the resident counts of an acquisition that will not be finished."""
        with self.lock:
            self.staging, self.staged = None, 0
            self.source.release()

    def _retire(self) -> None:
        """Release a discarded acquisition after its existing viewers close."""
        with self.lock:
            self._retired = True
            if not self._query_sessions:
                self.release()
