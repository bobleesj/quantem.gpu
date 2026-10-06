"""Raw-Metal detector backend over integer frames held in Metal chunks.

``MetalRawBackend`` answers detector queries for ``ChunkedFrames`` (an integer
Torch MPS tensor viewed as Metal buffers) through ``MetalVirtualImage``. A
no-bin 512x512x192x192 uint16 stack is 2.5e9 elements, more than a Torch MPS
tensor can address, so these sums run in raw Metal on the resident buffers.

Virtual-image binning contract on MPS:

- No-bin data (``det_bin == 1``) keeps the full detector (e.g. 192x192), so a
  single diffraction pattern keeps its angular resolution. A virtual image sums
  over all frames; at full resolution that is bandwidth-bound (~40 GB/s of
  scattered uint16, ~8-10 fps). So a detector-binned copy of the frames, the
  sidecar ``fast_vi``, is built in the background and virtual images are summed
  on it: 4x fewer pixels to read gives real-time interaction. Binning only the
  mask would still read every full-resolution pixel; the speedup comes from
  reading the smaller copy. Single patterns still read the full resolution.
- Data binned at load (``det_bin >= 2``) is already small, so no sidecar is
  built.
"""

import threading

import numpy as np

from quantem.gpu.detector.tensors import scan_shape_of
from quantem.gpu.resident.mps.frames import ChunkedFrames, bin_mask
from quantem.gpu.resident.queries import DetectorQueries


class MetalRawBackend(DetectorQueries):
    """Detector queries on ``ChunkedFrames``, with the background interaction sidecar.

    ``frames`` is a ``ChunkedFrames`` or MPS SSB's exact BF-column frames, which
    provide the detector total for the mean pattern and nothing else; only
    ``ChunkedFrames`` builds a sidecar.
    """

    def __init__(self, frames):
        self.frames = frames
        self.det_shape = tuple(int(size) for size in frames.shape[1:])
        self.n_frames = int(frames.shape[0])
        self.scan_shape = scan_shape_of(self.n_frames)
        self.device = "mps"
        # Full-detector centre of mass (com_col, com_row), built with the sidecar.
        self._center_of_mass = None
        # Per-frame totals of the full and the sidecar detector: dense masks subtract their complement.
        self._totals = {}
        # On a big no-bin detector full-resolution sums run at ~8-10 fps; the sidecar makes
        # them real-time once ready, and full resolution serves until then. A binned
        # uint32 pixel could overflow, so uint32 frames never get a sidecar.
        self._auto_fast = (
            isinstance(frames, ChunkedFrames)
            and frames.det_bin == 1
            and self.det_shape[0] >= 96
            and np.dtype(frames.dtype) != np.dtype(np.uint32)
        )
        if self._auto_fast and frames.fast_vi is None:
            threading.Thread(target=self._build_fast, daemon=True).start()

    @property
    def has_fast(self) -> bool:
        return isinstance(self.frames, ChunkedFrames) and self.frames.fast_vi is not None

    @property
    def fast_bin(self) -> int:
        return self.frames.fast_bin if isinstance(self.frames, ChunkedFrames) else 2

    def frame(self, index: int) -> np.ndarray:
        return self.frames.frame(int(index))

    def masked_sum(self, det_mask: np.ndarray) -> np.ndarray:
        if self._auto_fast and self.frames.fast_vi is not None:
            # Reduce the smaller resident sidecar with the mask binned to its factor.
            mask = bin_mask(np.ascontiguousarray(det_mask), self.frames.fast_bin)
            image = self._masked_counts(self.frames.fast_vi, mask, "sidecar")
        else:
            image = self._masked_counts(self.frames.vi, np.ascontiguousarray(det_mask), "full")
        return image.reshape(self.scan_shape).astype(np.float32, copy=False)

    def masked_sum_exact(self, det_mask: np.ndarray) -> np.ndarray:
        """Exact full-resolution integer reduction, bypassing display sidecars."""
        image = self._masked_counts(self.frames.vi, np.ascontiguousarray(det_mask), "full")
        return np.asarray(image).reshape(self.scan_shape).astype(np.uint64, copy=False)

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

        Uses the sidecar when ready, so no-bin DPC is real-time instead of an
        ~8 s full-resolution 192^2 pass; the full-detector field is cached when
        the sidecar is built, so the first DPC click is instant. Sidecar pixels
        are scaled by the bin factor back to full-resolution detector pixels;
        the constant half-pixel bin offset cancels under the DPC mean
        subtraction. Returns ``(com_col, com_row)``, flat ``(N,)`` float32, the
        same contract as the CUDA and Torch backends.
        """
        if det_mask is None and self._center_of_mass is not None:
            return self._center_of_mass
        if self._auto_fast and self.frames.fast_vi is not None:
            factor = self.fast_bin
            mask = None if det_mask is None else bin_mask(np.ascontiguousarray(det_mask), factor)
            com_col, com_row = self.frames.fast_vi.center_of_mass(mask)
            return com_col * factor, com_row * factor
        mask = None if det_mask is None else np.ascontiguousarray(det_mask)
        return self.frames.vi.center_of_mass(mask)

    def ensure_fast_sidecar(self, verbose: bool = False) -> bool:
        """Block until the sidecar is ready; True when ready or when data binned at load needs none."""
        if self.frames.det_bin > 1:
            return True
        if not isinstance(self.frames, ChunkedFrames):
            return False
        self.frames.ensure_fast_interaction(verbose=verbose)
        return self.frames.fast_vi is not None

    def cache_fast_presets(self, masks: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
        """Virtual images on the sidecar for named detector masks (``{"bf": mask, ...}``).

        Returns ``(scan_row, scan_col)`` float32 images; the caller stores the bytes.
        """
        if not self.has_fast:
            return {}
        return {
            name: np.asarray(self.frames.fast_vi.masked_sum(bin_mask(np.ascontiguousarray(mask), self.fast_bin)))
            .reshape(self.scan_shape)
            .astype(np.float32, copy=False)
            for name, mask in masks.items()
        }

    # ---

    def _build_fast(self):
        """Background thread body: build the sidecar and the full-detector centre of mass.

        A detector that the bin factor does not divide, or too little unified
        memory for the copy, leaves interaction at full resolution.
        """
        try:
            self.frames.ensure_fast_interaction(verbose=False)
            self._center_of_mass = self.center_of_mass()
        except (MemoryError, ValueError):
            return

    def _masked_counts(self, vi, mask: np.ndarray, totals: str) -> np.ndarray:
        """Per-frame masked counts from one Metal image, dense masks as total minus complement.

        The per-frame total of each detector (full or sidecar) is computed once;
        a mask selecting more than half the detector then reads only its smaller
        complement, which keeps dark-field dragging close to bright-field speed.
        """
        mask = np.ascontiguousarray(mask, dtype=bool)
        selected = int(mask.sum())
        if selected == 0:
            return np.zeros(self.n_frames, dtype=np.uint64)
        dense = selected > mask.size // 2 and mask.size - selected < selected
        if selected == mask.size or dense:
            if totals not in self._totals:
                self._totals[totals] = np.asarray(vi.masked_sum(np.ones(mask.shape, dtype=bool))).copy()
            if selected == mask.size:
                # The cache stays private: the caller gets its own copy.
                return self._totals[totals].copy()
            return self._totals[totals] - np.asarray(vi.masked_sum(~mask))
        return np.asarray(vi.masked_sum(mask))
