"""Integer 4D-STEM frames held in Metal unified-memory chunks.

MPS SSB borrows an integer Torch MPS tensor as ``ChunkedFrames`` so bright-field
column gathers and virtual-detector sums run in raw Metal on the resident
buffers, without a copy or a dtype cast.
"""

import bisect

import numpy as np

from quantem.gpu.resident.mps.virtual_image import MetalVirtualImage


class ChunkedFrames:
    """A 3D ``(N, det_row, det_col)`` view over Metal-buffer chunks.

    ``vi`` runs the reductions; single-frame reads come from the NumPy views
    of the chunks.
    """

    # Chunks hold native detector pixels; only BF-column frames are binned.
    det_bin = 1

    def __init__(self, chunks: list):
        if not chunks:
            raise ValueError("ChunkedFrames requires at least one chunk")
        self.chunks = chunks
        self.metadata = {}
        self.dtype = np.dtype(chunks[0].dtype)
        self.offsets = [0]
        for chunk in chunks:
            self.offsets.append(self.offsets[-1] + int(chunk.shape[0]))
        self.shape = (self.offsets[-1], *(int(size) for size in chunks[0].shape[1:]))
        self.ndim = 3
        self.device = "mps"
        self.vi = MetalVirtualImage(chunks)

    @property
    def detector_sum(self) -> np.ndarray:
        """Exact uint64 per-pixel sum of every frame, reduced in Metal."""
        return self.vi.detector_sum_exact()

    def frame(self, index: int) -> np.ndarray:
        """One diffraction pattern ``(det_row, det_col)`` as a NumPy view."""
        chunk = bisect.bisect_right(self.offsets, index) - 1
        return np.asarray(self.chunks[chunk][index - self.offsets[chunk]])

    def columns_float32_into(self, rows, cols, out: np.ndarray) -> np.ndarray:
        """Gather detector pixels directly into caller-owned unified GPU memory."""
        return self.vi.gather_columns_float32(rows, cols, out=out)
