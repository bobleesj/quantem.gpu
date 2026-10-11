"""Raw-Metal detector backend over integer frames held in Metal chunks.

``MetalRawBackend`` answers detector queries for ``ChunkedFrames`` (an integer
Torch MPS tensor viewed as Metal buffers) through ``MetalVirtualImage``. A
no-bin 512x512x192x192 uint16 stack is 2.5e9 elements, more than a Torch MPS
tensor can address, so these sums run in raw Metal on the resident buffers, at
the full detector resolution. Nothing derived from the counts is cached, so a
query always reads the frames as they are.
"""

import numpy as np

from quantem.gpu.detector.tensors import scan_shape_of
from quantem.gpu.resident.queries import DetectorQueries


class MetalRawBackend(DetectorQueries):
    """Detector queries on ``ChunkedFrames``.

    ``frames`` is a ``ChunkedFrames`` or MPS SSB's exact BF-column frames, which
    provide the detector total for the mean pattern and nothing else.
    """

    def __init__(self, frames):
        self.frames = frames
        self.det_shape = tuple(int(size) for size in frames.shape[1:])
        self.n_frames = int(frames.shape[0])
        self.scan_shape = scan_shape_of(self.n_frames)
        self.device = "mps"

    def frame(self, index: int) -> np.ndarray:
        return self.frames.frame(int(index))

    def masked_sum(self, det_mask: np.ndarray) -> np.ndarray:
        return self.masked_sum_exact(det_mask).astype(np.float32)

    def masked_sum_exact(self, det_mask: np.ndarray) -> np.ndarray:
        """Exact uint64 per-frame sums of the selected detector pixels."""
        mask = np.ascontiguousarray(det_mask, dtype=bool)
        if not mask.any():
            return np.zeros(self.scan_shape, dtype=np.uint64)
        image = np.asarray(self.frames.vi.masked_sum(mask))
        return image.reshape(self.scan_shape).astype(np.uint64, copy=False)

    def mean_dp(self) -> np.ndarray:
        # The exact total divided in float64 and rounded once, like every mean pattern.
        return (np.asarray(self.frames.detector_sum) / self.n_frames).astype(np.float32)

    def reduce_frames(self, scan_indices: np.ndarray, reduce: str = "mean") -> np.ndarray:
        """Mean (float32) or exact uint64 sum of the selected patterns; the frames are integer counts."""
        indices = np.asarray(scan_indices, dtype=np.uint32)
        if reduce == "mean":
            # An empty selection keeps its all-zero pattern.
            return (self.frames.vi.sum_frames(indices) / max(indices.size, 1)).astype(np.float32)
        if reduce == "sum":
            return self.frames.vi.sum_frames(indices)
        raise NotImplementedError(
            "MPS reduce_frames(reduce='max') has no Metal kernel. CPU fallback "
            "is disabled; use reduce='mean' or reduce='sum', or implement the "
            "native Metal max reducer."
        )

    def reduce_frames_exact(self, scan_indices: np.ndarray) -> np.ndarray:
        """Exact uint64 sum of the selected patterns."""
        return self.frames.vi.sum_frames(np.asarray(scan_indices, dtype=np.uint32))

    def center_of_mass(self, det_mask: np.ndarray | None = None):
        """Per-scan-position centre of mass (the DPC vector field) on the raw Metal kernel.

        Returns ``(com_col, com_row)``, flat ``(N,)`` float32, the same contract
        as the CUDA and Torch backends.
        """
        mask = None if det_mask is None else np.ascontiguousarray(det_mask)
        return self.frames.vi.center_of_mass(mask)
