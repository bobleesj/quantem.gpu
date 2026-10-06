"""Integer 4D-STEM frames held in Metal unified-memory chunks.

MPS SSB borrows an integer Torch MPS tensor as ``ChunkedFrames`` so bright-field
column gathers and virtual-detector sums run in raw Metal on the resident
buffers, without a copy or a dtype cast. On a no-bin detector a detector-binned
copy (the sidecar) makes interactive virtual images real-time; ``bin_mask`` maps
a full-resolution detector mask onto that binned grid.
"""

import bisect
import gc
import subprocess

import numpy as np

from quantem.gpu.resident.mps.virtual_image import MetalVirtualImage


class ChunkedFrames:
    """A 3D ``(N, det_row, det_col)`` view over Metal-buffer chunks.

    ``vi`` runs the full-resolution reductions; ``fast_vi`` runs them on the
    detector-binned sidecar once :meth:`ensure_fast_interaction` builds it.
    Single-frame reads come from the NumPy views of the chunks.
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
        self.fast_vi = None
        # bin4 on a 24 GB Mac (fits, and 4x fewer detector pixels per virtual-image sum), bin2 on bigger boxes.
        self.fast_bin = default_fast_bin()

    @property
    def detector_sum(self) -> np.ndarray:
        """Exact uint64 per-pixel sum of every frame, reduced in Metal."""
        return self.vi.detector_sum_exact()

    def frame(self, index: int) -> np.ndarray:
        """One diffraction pattern ``(det_row, det_col)`` as a NumPy view."""
        chunk = bisect.bisect_right(self.offsets, index) - 1
        return np.asarray(self.chunks[chunk][index - self.offsets[chunk]])

    def columns_float32(self, rows, cols, *, out: np.ndarray | None = None) -> np.ndarray:
        """Gather detector pixels ``(rows, cols)`` over all scan positions as ``(pixel, frame)`` float32."""
        return self.vi.gather_columns_float32(rows, cols, out=out)

    def columns_float32_into(self, rows, cols, out: np.ndarray) -> np.ndarray:
        """Gather detector pixels directly into caller-owned unified GPU memory."""
        return self.vi.gather_columns_float32(rows, cols, out=out)

    def ensure_fast_interaction(self, *, verbose: bool = True) -> MetalVirtualImage:
        """Prepare the detector-bin ``fast_bin`` sidecar for fast virtual images.

        Built on the GPU from the resident no-bin chunks: no disk re-decode, no
        decompress scratch. The only new memory is the sidecar itself (1.2 GB at
        bin4), which lets the no-bin viewer open and scrub on a 24 GB Mac
        without a second 19 GB decode spike.
        """
        if self.fast_vi is None:
            # Collect released Metal views first: the sidecar needs the unified memory now.
            gc.collect()
            self.fast_vi = self.vi.binned(self.fast_bin, verbose=verbose)
        return self.fast_vi


def default_fast_bin() -> int:
    """Sidecar bin factor that fits the host's unified memory.

    The bin2 sidecar of a no-bin 512x512x192x192 stack is 4.8 GB on top of the
    19.3 GB data = 24.1 GB, which does not fit a 24 GB Mac (it swaps and the
    machine freezes). bin4 is 1.2 GB -> ~20.5 GB total, fits. So: bin4 on Macs
    with <= ~32 GB unified memory, bin2 on larger boxes where the sharper
    detector grid is free. Falls back to bin4 (the safe choice) if the memory
    size can't be read.
    """
    try:
        total = int(
            subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True, timeout=3, check=True
            ).stdout.strip()
        )
    except (OSError, ValueError, subprocess.SubprocessError):
        return 4
    return 2 if total > 32 * 1024**3 else 4


def bin_mask(mask: np.ndarray, factor: int) -> np.ndarray:
    """Downsample a full-resolution detector mask to the ``factor`` sidecar grid.

    A binned pixel is in the mask if ANY of its ``factor * factor`` raw pixels
    are, so a virtual-detector edge never disappears at coarser bin factors.
    """
    mask = np.asarray(mask, dtype=bool)
    rows = mask.shape[-2] // factor
    cols = mask.shape[-1] // factor
    return mask[: rows * factor, : cols * factor].reshape(rows, factor, cols, factor).any(axis=(1, 3))
