"""Detector reductions for Torch tensors on an accelerator and for host arrays.

``TorchBackend`` serves Torch tensors on CUDA or MPS; integer counts stay
integer until the small reduced output, and float patterns sum to float64
accuracy (in float64 on CUDA, in compensated float32 on MPS, which has none).

``ArrayBackend`` serves NumPy arrays, array-likes and CPU Torch tensors (viewed
without a copy). It is the host reference: integer counts sum in uint64, float
intensities in float64, and the centre of mass is computed in float64, each
result cast to float32 once at the end, so host results never lose precision to
a float32 accumulator.

Memory discipline: the chunk-size trap (read before touching chunk math)

Every chunked reduction picks a chunk size from a byte budget divided by the
bytes of one chunk unit. When a sum uses a WIDER accumulator dtype than the
input dtype, the chunk MUST budget for the accumulator's bytes, not the
input's; else the internal cast during ``.sum(dtype=T)`` materializes a
chunk-sized transient in dtype T that oversubscribes device memory.

Concrete regression the rule prevents (fixed 2026-07-02): a uint16 mean
diffraction pattern budgeted 1 GB of uint16 per chunk, about 29K frames at a
192x192 detector. Each 2.15 GB chunk was cast to int64 inside ``sum`` (an
8.6 GB transient), which the CUDA pool then cached: no-bin 512x512x192x192
Show4DSTEM peaked at 29 GB instead of ~21 GB, invisible on a 96 GB card but
out of memory on a 24 GB one. Outputs are bit-identical for any chunk size, so
value-parity tests cannot catch this; reproduce it by capping device memory.

If you add a reduction or change ``dtype=`` on an existing ``.sum()``, budget
for whichever dtype is wider, input or accumulator: float32 sums of uint16
budget 4 bytes per pixel, int64 sums 8 bytes.
"""

import math

import numpy as np
import torch

from quantem.gpu.resident.mps.float_ans import pattern_total
from quantem.gpu.resident.queries import DetectorQueries

# Cap transient float32 memory per reduction chunk (matches the widget budget).
_CHUNK_BYTE_BUDGET = 600 * 1024 * 1024
# Host chunks hold float64 copies of whole frames for the centre of mass.
_HOST_CHUNK_BYTES = 512 * 1024 * 1024
_SPARSE_MASK_CHUNK_BYTE_BUDGET = 64 * 1024 * 1024
# 128 MB of int64 transient per exact chunk: a uint16 chunk expands 4x when cast for the sum.
_INT64_CHUNK_BYTES = 1 << 27


class TorchBackend(DetectorQueries):
    """One chunked Torch path for detector products on CUDA and MPS tensors.

    Masked sums are chunked tensordots (or selected-pixel sums for sparse
    masks), the mean pattern accumulates integers in int64, and every chunk
    stays inside a byte budget.
    """

    def __init__(self, tensor):
        if tensor.ndim == 4:
            self.scan_shape = (int(tensor.shape[0]), int(tensor.shape[1]))
        elif tensor.ndim == 3:
            self.scan_shape = scan_shape_of(int(tensor.shape[0]))
        else:
            raise ValueError(f"expected 3D/4D tensor, got {tuple(tensor.shape)}")
        self.detector_shape = (int(tensor.shape[-2]), int(tensor.shape[-1]))
        self.n_frames = math.prod(self.scan_shape)
        self.device = tensor.device
        self._flat = tensor.reshape(-1, *self.detector_shape)

    def frame(self, index: int) -> np.ndarray:
        return self._flat[int(index)].cpu().numpy()

    def masked_sum(self, det_mask: np.ndarray) -> np.ndarray:
        """Virtual image: sum masked detector pixels per scan position (chunked)."""
        return self._masked_sum_tensor(det_mask).cpu().numpy()

    def masked_sum_native(
        self,
        det_mask: np.ndarray | torch.Tensor,
        *,
        out: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Return the same float32 detector image on the source device.

        Parameters
        ----------
        det_mask
            Detector mask in (row, col) order, as a NumPy array or Torch tensor.
        out
            Optional contiguous float32 tensor with the scan shape on the
            source device. The returned tensor is complete and independently
            owned when this buffer is omitted.

        Returns
        -------
        torch.Tensor
            Float32 virtual image with the scan shape on the source device.
        """
        if out is not None and (
            not isinstance(out, torch.Tensor)
            or tuple(out.shape) != self.scan_shape
            or out.dtype != torch.float32
            or out.device != self.device
            or not out.is_contiguous()
        ):
            raise ValueError(
                "out must be a contiguous float32 Torch tensor with scan shape "
                f"{self.scan_shape} on {self.device}."
            )
        result = self._masked_sum_tensor(det_mask)
        if out is not None:
            out.copy_(result)
            result = out
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)
        elif self.device.type == "mps":
            torch.mps.synchronize()
        return result

    def _masked_sum_tensor(self, det_mask: np.ndarray | torch.Tensor) -> torch.Tensor:
        """Keep the existing bounded reduction and its float32 result on device."""
        values = det_mask if isinstance(det_mask, torch.Tensor) else np.ascontiguousarray(det_mask)
        mask = torch.as_tensor(values, device=self.device, dtype=torch.bool).reshape(-1)
        if not torch.is_floating_point(self._flat):
            selected = torch.nonzero(mask, as_tuple=False).reshape(-1)
            if selected.numel() == 0:
                return torch.zeros(self.scan_shape, dtype=torch.float32, device=self.device)
            # Counts stay integers through the reduction; only the scan image becomes
            # float32, which matches the CUDA paths.
            flat = self._flat.reshape(self.n_frames, -1)
            out = torch.empty(self.n_frames, dtype=torch.float32, device=self.device)
            step = max(1, _SPARSE_MASK_CHUNK_BYTE_BUDGET // max(1, selected.numel() * 8))
            for start in range(0, self.n_frames, step):
                stop = min(self.n_frames, start + step)
                out[start:stop] = flat[start:stop].index_select(1, selected).to(torch.int64).sum(dim=1).to(torch.float32)
            return out.reshape(self.scan_shape)
        # Float intensities: sum the selected pixels of sparse masks directly.
        det_pixels = self.detector_shape[0] * self.detector_shape[1]
        selected = int(mask.sum().item())
        if selected <= 0:
            return torch.zeros(self.scan_shape, dtype=torch.float32, device=self.device)
        # Dense tensordot is better once the ROI covers a large detector fraction.
        if selected <= det_pixels // 4:
            pixel_indices = torch.nonzero(mask, as_tuple=False).reshape(-1)
            flat = self._flat.reshape(self.n_frames, det_pixels)
            out = torch.empty(self.n_frames, dtype=torch.float32, device=self.device)
            step = max(1, _SPARSE_MASK_CHUNK_BYTE_BUDGET // max(1, selected * 4))
            for start in range(0, self.n_frames, step):
                stop = min(self.n_frames, start + step)
                out[start:stop] = flat[start:stop].index_select(1, pixel_indices).sum(dim=1)
            return out.reshape(self.scan_shape)
        weights = torch.as_tensor(values, device=self.device).float()
        out = torch.zeros(self.n_frames, dtype=torch.float32, device=self.device)
        step = self._frames_per_chunk()
        for start in range(0, self.n_frames, step):
            # Never compute in 64 bits: the weights are float32.
            chunk = self._flat[start : start + step].float()
            out[start : start + step] = torch.tensordot(chunk, weights, dims=([1, 2], [0, 1]))
        return out.reshape(self.scan_shape)

    def masked_sum_exact(self, det_mask: np.ndarray) -> np.ndarray:
        """Virtual image with integer counts preserved through host transfer."""
        if torch.is_floating_point(self._flat):
            raise TypeError("Exact detector sums require integer detector data.")
        mask = torch.as_tensor(np.ascontiguousarray(det_mask), device=self.device, dtype=torch.bool).reshape(-1)
        selected = torch.nonzero(mask, as_tuple=False).reshape(-1)
        if selected.numel() == 0:
            return np.zeros(self.scan_shape, dtype=np.uint64)
        flat = self._flat.reshape(self.n_frames, -1)
        out = torch.empty(self.n_frames, dtype=torch.int64, device=self.device)
        step = max(1, _SPARSE_MASK_CHUNK_BYTE_BUDGET // max(1, selected.numel() * 8))
        for start in range(0, self.n_frames, step):
            stop = min(self.n_frames, start + step)
            out[start:stop] = flat[start:stop].index_select(1, selected).to(torch.int64).sum(dim=1)
        return out.reshape(self.scan_shape).cpu().numpy().astype(np.uint64, copy=False)

    def mean_dp(self) -> np.ndarray:
        """Mean pattern with exact integer sums or preserved fractional intensities.

        Integers accumulate in int64 and floats to float64 accuracy; the chunk
        size budgets the 8-byte cast before the sum. The total is divided in
        float64 on the host (MPS has no float64) and rounded once to float32,
        the rule of every mean pattern.
        """
        if torch.is_floating_point(self._flat):
            return (self._float_total() / self.n_frames).astype(np.float32)
        step = max(1, _INT64_CHUNK_BYTES // (self.detector_shape[0] * self.detector_shape[1] * 8))
        total = torch.zeros(self.detector_shape, dtype=torch.int64, device=self.device)
        for start in range(0, self.n_frames, step):
            # Cast first: some devices do not implement mixed uint16/int64 sums.
            total += self._flat[start : start + step].to(torch.int64).sum(dim=0)
        return (total.cpu().numpy() / self.n_frames).astype(np.float32)

    def reduce_frames(self, scan_indices: np.ndarray, reduce: str = "mean") -> np.ndarray:
        """Summed / mean / max DP over a set of scan positions (flat indices).

        Integer counts give exact uint64 ``sum`` and ``max`` and a ``mean`` of the
        exact total divided in float64 and rounded once to float32; float data
        gives float32, its ``sum`` and ``mean`` the float64 total rounded once.
        Selected patterns are gathered in chunks, so a large scan ROI never
        copies all of its patterns at once.
        """
        if reduce not in ("mean", "sum", "max"):
            raise ValueError(f"Unknown frame reduction {reduce!r}; use mean, sum, or max.")
        if not torch.is_floating_point(self._flat):
            # Signed data (dark-subtracted int16) keeps its sign: an unsigned total of negative values wraps.
            exact = np.int64 if self._flat.dtype.is_signed else np.uint64
            if reduce == "max":
                return self._integer_reduce(scan_indices, "max").astype(exact)
            total = self._integer_reduce(scan_indices, "sum")
            return total.astype(exact) if reduce == "sum" else (total / len(scan_indices)).astype(np.float32)
        indices = np.asarray(scan_indices, dtype=np.int64).reshape(-1)
        if reduce == "max":
            step = max(1, _CHUNK_BYTE_BUDGET // (self.detector_shape[0] * self.detector_shape[1] * 4))
            maximum = None
            for start in range(0, indices.size, step):
                frames = self._flat.index_select(0, torch.as_tensor(indices[start : start + step], device=self.device))
                maximum = frames.amax(dim=0) if maximum is None else torch.maximum(maximum, frames.amax(dim=0))
            return maximum.float().cpu().numpy()
        total = self._float_total(indices)
        return (total / indices.size if reduce == "mean" else total).astype(np.float32)

    def _float_total(self, indices=None) -> np.ndarray:
        """Sum every float pattern, or those at ``indices``, per detector pixel; float64 on the host.

        CUDA accumulates in float64, in chunks that budget the 8-byte cast.
        MPS has no float64: its Metal kernel keeps compensated float32 sums
        that add up to the same total and reads float32 where it lies, so the
        whole stack is one window and only a gather is chunked. A float32
        running total is several units in the last place off after thousands
        of patterns.
        """
        pixels = self.detector_shape[0] * self.detector_shape[1]
        count = self.n_frames if indices is None else indices.size
        in_place = self.device.type == "mps" and indices is None and self._flat.dtype == torch.float32
        step = count if in_place else max(1, (_CHUNK_BYTE_BUDGET // 4 if self.device.type == "mps" else _INT64_CHUNK_BYTES // 8) // pixels)
        windows = (
            self._flat[start : start + step] if indices is None
            else self._flat.index_select(0, torch.as_tensor(indices[start : start + step], device=self.device))
            for start in range(0, count, step)
        )
        if self.device.type == "mps":
            # The kernel reads float32; half floats widen first.
            return pattern_total((values.float() for values in windows), pixels).reshape(self.detector_shape)
        total = torch.zeros(self.detector_shape, dtype=torch.float64, device=self.device)
        for values in windows:
            total += values.to(torch.float64).sum(dim=0)
        return total.cpu().numpy()

    def reduce_frames_exact(self, scan_indices: np.ndarray) -> np.ndarray:
        """Return an exact uint64 scan-frame sum with bounded int64 scratch."""
        if torch.is_floating_point(self._flat):
            raise TypeError("Exact scan ROI sums require integer detector data.")
        return self._integer_reduce(scan_indices, "sum").astype(np.uint64, copy=False)

    def reduce_frames_max(self, scan_indices: np.ndarray) -> np.ndarray:
        """Return an exact scan-frame maximum with bounded scratch."""
        if torch.is_floating_point(self._flat):
            raise TypeError("Exact scan ROI maxima require integer detector data.")
        return self._integer_reduce(scan_indices, "max").astype(np.uint32, copy=False)

    def _integer_reduce(self, scan_indices, reduce: str) -> np.ndarray:
        """int64 ``sum`` or ``max`` of the selected integer frames; zeros for an empty selection.

        Frames are gathered in chunks that budget the 8-byte cast, and the
        maximum starts from the first chunk, so negative values survive.
        """
        indices = np.asarray(scan_indices, dtype=np.int64).reshape(-1)
        step = max(1, _INT64_CHUNK_BYTES // (self.detector_shape[0] * self.detector_shape[1] * 8))
        result = None
        for start in range(0, indices.size, step):
            selected = torch.as_tensor(indices[start : start + step], dtype=torch.int64, device=self.device)
            # Cast first: MPS has no mixed uint16/int64 sum ("sum_reduction_outer_ushort_long").
            chunk = self._flat.index_select(0, selected).to(torch.int64)
            if reduce == "max":
                reduced = chunk.max(dim=0).values
                result = reduced if result is None else torch.maximum(result, reduced)
            else:
                result = chunk.sum(dim=0) if result is None else result + chunk.sum(dim=0)
        return np.zeros(self.detector_shape, np.int64) if result is None else result.cpu().numpy()

    def center_of_mass(self, det_mask: np.ndarray | None = None):
        """Per-scan-position centre of mass over the (masked) detector: the DPC vector field.

        Returns ``(com_col, com_row)``, each ``(N,)`` float32 in absolute
        detector pixels (col = sum col*I / sum I, row = sum row*I / sum I), the
        same contract as the Metal and CUDA backends so DPC is single-source.
        ``det_mask`` None means the full detector. Chunked by the same byte
        budget as ``masked_sum``; an empty pattern gives 0.
        """
        mask = None if det_mask is None else torch.as_tensor(np.ascontiguousarray(det_mask), device=self.device).float()
        rows = torch.arange(self.detector_shape[0], device=self.device, dtype=torch.float32)[:, None]
        cols = torch.arange(self.detector_shape[1], device=self.device, dtype=torch.float32)[None, :]
        com_col = torch.zeros(self.n_frames, dtype=torch.float32, device=self.device)
        com_row = torch.zeros(self.n_frames, dtype=torch.float32, device=self.device)
        step = self._frames_per_chunk()
        for start in range(0, self.n_frames, step):
            # Never compute in 64 bits: the mask and coordinates are float32.
            chunk = self._flat[start : start + step].float()
            if mask is not None:
                chunk = chunk * mask
            total = chunk.sum(dim=(1, 2)).clamp(min=1e-12)
            com_row[start : start + step] = (chunk * rows).sum(dim=(1, 2)) / total
            com_col[start : start + step] = (chunk * cols).sum(dim=(1, 2)) / total
        return com_col.cpu().numpy(), com_row.cpu().numpy()

    def _frames_per_chunk(self) -> int:
        """Whole scan rows per chunk, so each float32 chunk stays inside ``_CHUNK_BYTE_BUDGET``."""
        row_length = self.scan_shape[-1]
        bytes_per_row = row_length * self.detector_shape[0] * self.detector_shape[1] * 4
        return max(1, _CHUNK_BYTE_BUDGET // max(1, bytes_per_row)) * row_length


class ArrayBackend(DetectorQueries):
    """Host reference products in NumPy with wide accumulators.

    Integer counts sum in uint64 and float intensities in float64; the centre
    of mass is computed in float64. Per-frame products are chunked over frames,
    which leaves every frame's arithmetic unchanged; the mean pattern sums all
    frames in one NumPy reduction so float sums keep one summation order.
    """

    def __init__(self, data):
        values = np.asarray(data)
        if values.ndim == 4:
            self.scan_shape = (int(values.shape[0]), int(values.shape[1]))
        elif values.ndim == 3:
            self.scan_shape = scan_shape_of(int(values.shape[0]))
        else:
            raise ValueError(f"Expected 3D or 4D 4D-STEM data, got {values.ndim}D with shape {values.shape}.")
        self.detector_shape = (int(values.shape[-2]), int(values.shape[-1]))
        self.n_frames = math.prod(self.scan_shape)
        self.device = "cpu"
        self._flat = values.reshape(-1, *self.detector_shape)

    def frame(self, index: int) -> np.ndarray:
        return np.asarray(self._flat[int(index)])

    def mean_dp(self) -> np.ndarray:
        total = self._flat.sum(axis=0, dtype=_host_accumulator(self._flat.dtype))
        return (total / self.n_frames).astype(np.float32)

    def masked_sum(self, det_mask: np.ndarray) -> np.ndarray:
        mask = self._mask(det_mask).reshape(-1)
        pixels = self._flat.reshape(self.n_frames, -1)
        accumulator = _host_accumulator(pixels.dtype)
        image = np.empty(self.n_frames, dtype=accumulator)
        step = self._frames_per_chunk()
        for start in range(0, self.n_frames, step):
            stop = start + step
            image[start:stop] = pixels[start:stop][:, mask].sum(axis=1, dtype=accumulator)
        return image.astype(np.float32).reshape(self.scan_shape)

    def masked_sum_exact(self, det_mask: np.ndarray) -> np.ndarray:
        mask = self._mask(det_mask).reshape(-1)
        pixels = self._flat.reshape(self.n_frames, -1)
        if not np.issubdtype(pixels.dtype, np.integer):
            raise TypeError("Exact detector sums require integer detector data.")
        image = np.empty(self.n_frames, dtype=np.uint64)
        step = self._frames_per_chunk()
        for start in range(0, self.n_frames, step):
            stop = start + step
            image[start:stop] = pixels[start:stop][:, mask].sum(axis=1, dtype=np.uint64)
        return image.reshape(self.scan_shape)

    def reduce_frames(self, scan_indices, reduce: str = "mean") -> np.ndarray:
        """Exact ``sum`` and ``max`` of integer data (uint64, int64 when signed), float32 for float data; float32 ``mean``.

        The total accumulates in uint64, int64 or float64 and the mean divides
        it once. Frames are gathered in chunks, so a scan region of a
        memory-mapped file is never loaded at once.
        """
        if reduce not in ("mean", "sum", "max"):
            raise ValueError(f"Unknown frame reduction {reduce!r}; use mean, sum, or max.")
        indices = np.asarray(scan_indices, dtype=np.intp).reshape(-1)
        accumulator = _host_accumulator(self._flat.dtype)
        exact = accumulator if accumulator.kind in "ui" else np.dtype(np.float32)
        step = self._frames_per_chunk()
        if reduce == "max":
            maximum = self._flat[indices[:step]].max(axis=0)
            for start in range(step, indices.size, step):
                maximum = np.maximum(maximum, self._flat[indices[start : start + step]].max(axis=0))
            return maximum.astype(exact)
        total = np.zeros(self.detector_shape, accumulator)
        for start in range(0, indices.size, step):
            total += self._flat[indices[start : start + step]].sum(axis=0, dtype=accumulator)
        return total.astype(exact) if reduce == "sum" else (total / indices.size).astype(np.float32)

    def reduce_frames_exact(self, scan_indices) -> np.ndarray:
        selected = self._flat[np.asarray(scan_indices, dtype=np.intp).reshape(-1)]
        if not np.issubdtype(selected.dtype, np.integer):
            raise TypeError("Exact scan ROI sums require integer detector data.")
        return selected.sum(axis=0, dtype=np.uint64)

    def reduce_frames_max(self, scan_indices) -> np.ndarray:
        selected = self._flat[np.asarray(scan_indices, dtype=np.intp).reshape(-1)]
        if not np.issubdtype(selected.dtype, np.integer):
            raise TypeError("Exact scan ROI maxima require integer detector data.")
        return selected.max(axis=0).astype(np.uint32, copy=False)

    def center_of_mass(self, det_mask: np.ndarray | None = None):
        """Per-scan-position centre of mass ``(com_col, com_row)``, flat float32, computed in float64.

        An empty pattern gives 0: its denominator is clamped to 1e-10.
        """
        mask = np.ones(self.detector_shape, dtype=bool) if det_mask is None else self._mask(det_mask)
        rows = np.arange(self.detector_shape[0], dtype=np.float64)[None, :, None]
        cols = np.arange(self.detector_shape[1], dtype=np.float64)[None, None, :]
        com_row = np.empty(self.n_frames, dtype=np.float64)
        com_col = np.empty(self.n_frames, dtype=np.float64)
        step = self._frames_per_chunk()
        for start in range(0, self.n_frames, step):
            stop = start + step
            weighted = np.asarray(self._flat[start:stop], dtype=np.float64) * mask
            total = np.maximum(weighted.sum(axis=(1, 2)), 1e-10)
            com_row[start:stop] = (weighted * rows).sum(axis=(1, 2)) / total
            com_col[start:stop] = (weighted * cols).sum(axis=(1, 2)) / total
        return com_col.astype(np.float32), com_row.astype(np.float32)

    def _mask(self, det_mask) -> np.ndarray:
        mask = np.asarray(det_mask, dtype=bool)
        if mask.shape != self.detector_shape:
            raise ValueError(f"det_mask shape {mask.shape} does not match detector shape {self.detector_shape}.")
        return mask

    def _frames_per_chunk(self) -> int:
        """Frames whose float64 copy fits ``_HOST_CHUNK_BYTES``."""
        return max(1, _HOST_CHUNK_BYTES // (self.detector_shape[0] * self.detector_shape[1] * 8))


def _host_accumulator(dtype) -> np.dtype:
    """uint64 for counts and int64 for signed integers (both exact); float64 for float data.

    An integer sum truncates sub-unity float intensities to 0, and a uint64 sum
    of negative values (dark-subtracted int16) wraps.
    """
    return np.dtype({"u": np.uint64, "i": np.int64}.get(np.dtype(dtype).kind, np.float64))


def scan_shape_of(frames: int) -> tuple[int, ...]:
    """Scan shape of a flat frame stack: the near-square raster that holds every frame, else one row.

    ``rows = round(sqrt(frames))``; when ``rows * (frames // rows)`` covers every
    frame the stack reads as that raster, so a guess never drops frames.
    """
    rows = round(frames**0.5)
    return (rows, frames // rows) if rows * (frames // rows) == frames else (frames,)
