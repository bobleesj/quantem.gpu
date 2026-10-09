"""CUDA detector reductions over dense CuPy 4D-STEM arrays already on the GPU.

Dragging a virtual detector over a resident uint8/uint16/uint32 array must not
allocate CuPy gather temporaries the size of the data. The warp-shuffle kernels
in ``kernels/virtual_image.cu`` read the resident counts in place: selected
detector pixels per frame, per-frame totals, centre-of-mass moments, and sums or
maxima over selected frames. A dense mask (dark field) is computed as the cached
per-frame total minus its smaller complement. Other dtypes and non-contiguous
arrays reduce through ``TorchBackend`` on the same memory.
"""

import math
from collections import OrderedDict
from functools import cache
from pathlib import Path

import numpy as np

from quantem.gpu.detector.cuda.probe import mean_dp
from quantem.gpu.detector.tensors import TorchBackend, scan_shape_of
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.resident.queries import DetectorQueries

_KERNEL_SUFFIX = {np.dtype(np.uint8): "u8", np.dtype(np.uint16): "u16", np.dtype(np.uint32): "u32"}
# Dragging repeats the same few masks; their selected-pixel indices upload once.
_MASK_INDEX_CACHE_SIZE = 32


@cache
def kernels(device: int):
    """Compile the virtual-image kernels once per device."""
    with cp.cuda.Device(device):
        return cp.RawModule(
            code=Path(__file__).with_name("kernels").joinpath("virtual_image.cu").read_text(),
            options=("--std=c++11",),
        )


class CudaKernelCompute(DetectorQueries):
    """Virtual-detector products from a resident CuPy array, reduced by CUDA kernels.

    ``data`` is ``(scan_row, scan_col, det_row, det_col)`` or a flat frame stack
    ``(frames, det_row, det_col)``. Products are copied to the host; the full
    detector total and centre of mass are cached because dense masks and DPC
    reuse them on every interaction.
    """

    def __init__(self, data):
        if data.ndim == 4:
            self.scan_shape = (int(data.shape[0]), int(data.shape[1]))
        elif data.ndim == 3:
            self.scan_shape = scan_shape_of(int(data.shape[0]))
        else:
            raise ValueError(f"expected 3D/4D cupy array, got {tuple(data.shape)}")
        self.det_shape = (int(data.shape[-2]), int(data.shape[-1]))
        self.n_frames = math.prod(self.scan_shape)
        self.device = "cuda"
        self._data = data
        self._frames = data.reshape(-1, *self.det_shape)
        # The kernels read raw C-ordered memory of the native count dtypes only.
        self._suffix = _KERNEL_SUFFIX.get(np.dtype(data.dtype)) if data.flags.c_contiguous else None
        self._totals = None
        self._full_center_of_mass = None
        self._mask_indices = OrderedDict()
        self._torch = None

    def frame(self, index: int) -> np.ndarray:
        return self._frames[int(index)].get()

    def masked_sum(self, det_mask: np.ndarray) -> np.ndarray:
        """Float32 virtual image: selected pixels, or total minus complement for dense masks."""
        mask = self._mask(det_mask)
        selected = int(mask.sum())
        if selected == 0:
            return np.zeros(self.scan_shape, dtype=np.float32)
        if self._suffix is None:
            return self._torch_backend().masked_sum(mask.reshape(self.det_shape))
        if selected == mask.size:
            image = self._total_counts().astype(cp.float32)
        elif selected > int(mask.size * 0.5):
            image = cp.empty(self.n_frames, dtype=cp.float32)
            self._sum_selected("selected_sum_from_total_f32", self._indices(~mask), image, self._total_counts())
        else:
            image = cp.empty(self.n_frames, dtype=cp.float32)
            self._sum_selected("selected_sum_f32", self._indices(mask), image)
        return image.reshape(self.scan_shape).get()

    def masked_sum_exact(self, det_mask: np.ndarray) -> np.ndarray:
        """Exact uint64 virtual image: selected pixels, or total minus complement for dense masks."""
        _require_counts(self._frames, "Exact detector sums")
        mask = self._mask(det_mask)
        selected = int(mask.sum())
        if selected == 0:
            return np.zeros(self.scan_shape, dtype=np.uint64)
        if self._suffix is None:
            image = self._frames.reshape(self.n_frames, -1)[:, self._indices(mask)].sum(axis=1, dtype=cp.uint64)
        elif selected == mask.size:
            image = self._total_counts()
        elif selected > int(mask.size * 0.5):
            image = self._total_counts() - self._selected_sum_uint64(self._indices(~mask))
        else:
            image = self._selected_sum_uint64(self._indices(mask))
        return image.reshape(self.scan_shape).get()

    def mean_dp(self) -> np.ndarray:
        return mean_dp(self._data).get()

    def reduce_frames(self, scan_indices: np.ndarray, reduce: str = "mean") -> np.ndarray:
        """Exact uint64 ``sum`` and ``max`` of integer counts (float32 for float data), float32 ``mean``."""
        indices = cp.asarray(np.asarray(scan_indices, dtype=np.int64))
        frames = self._frames.reshape(self.n_frames, -1).take(indices, axis=0)
        # uint64 keeps counts exact; float data needs a float accumulator or every
        # sub-unity intensity truncates to 0.
        counts = frames.dtype.kind in "ui"
        accumulator = cp.uint64 if counts else cp.float64
        if reduce == "sum":
            pattern = frames.sum(axis=0, dtype=accumulator)
            pattern = pattern if counts else pattern.astype(cp.float32)
        elif reduce == "max":
            pattern = frames.max(axis=0).astype(cp.uint64 if counts else cp.float32)
        else:
            # Divide the exact total in float64 and round once, as MPS does; rounding
            # the total to float32 first loses counts above 2^24.
            pattern = (frames.sum(axis=0, dtype=accumulator) / int(indices.size)).astype(cp.float32)
        return pattern.reshape(self.det_shape).get()

    def reduce_frames_exact(self, scan_indices: np.ndarray) -> np.ndarray:
        """Exact uint64 sum of selected frames without a gathered frame tensor."""
        _require_counts(self._frames, "Exact scan ROI sums")
        indices = np.asarray(scan_indices, dtype=np.int32).reshape(-1)
        if self._suffix is None:
            pattern = self._frames.reshape(self.n_frames, -1).take(cp.asarray(indices), axis=0).sum(axis=0, dtype=cp.uint64)
        else:
            pattern = self._reduce_selected_frames("selected_frame_sum_u64", indices, cp.uint64)
        return pattern.reshape(self.det_shape).get()

    def reduce_frames_max(self, scan_indices: np.ndarray) -> np.ndarray:
        """Exact maximum of selected frames without a gathered frame tensor."""
        _require_counts(self._frames, "Exact scan ROI maxima")
        indices = np.asarray(scan_indices, dtype=np.int32).reshape(-1)
        if self._suffix is None:
            pattern = self._frames.reshape(self.n_frames, -1).take(cp.asarray(indices), axis=0).max(axis=0)
        else:
            pattern = self._reduce_selected_frames("selected_frame_max_u32", indices, cp.uint32)
        return pattern.reshape(self.det_shape).get().astype(np.uint32, copy=False)

    def center_of_mass(self, det_mask: np.ndarray | None = None):
        """Absolute detector ``(com_col, com_row)``, flat float32, from exact integer moments.

        Each pattern is read once; intensity, row and column moments accumulate
        in 64-bit integers and divide in float64. An empty pattern gives 0.
        """
        if det_mask is None and self._full_center_of_mass is not None:
            return self._full_center_of_mass
        if self._suffix is None:
            return self._torch_backend().center_of_mass(det_mask)
        com_row = cp.empty(self.n_frames, dtype=cp.float32)
        com_col = cp.empty(self.n_frames, dtype=cp.float32)
        mask = None if det_mask is None else self._mask(det_mask)
        selected = self.det_shape[0] * self.det_shape[1] if mask is None else int(mask.sum())
        if selected == 0:
            com_row.fill(0)
            com_col.fill(0)
        elif selected == self.det_shape[0] * self.det_shape[1]:
            self._launch(
                f"center_of_mass_full_{self._suffix}_4f",
                (128, 4, 1),
                (self._data, com_row, com_col, np.int32(selected), np.int32(self.det_shape[1]), np.int32(self.n_frames)),
            )
        else:
            self._launch(
                f"center_of_mass_selected_{self._suffix}_4f",
                (128, 4, 1),
                (
                    self._data,
                    cp.asarray(np.flatnonzero(mask).astype(np.int32, copy=False), dtype=cp.int32),
                    com_row,
                    com_col,
                    np.int32(selected),
                    np.int32(self.det_shape[0] * self.det_shape[1]),
                    np.int32(self.det_shape[1]),
                    np.int32(self.n_frames),
                ),
            )
        result = com_col.get(), com_row.get()
        if det_mask is None:
            self._full_center_of_mass = result
        return result

    # ---

    def _mask(self, det_mask) -> np.ndarray:
        """Flat host boolean mask; selected-pixel indices are planned on the host."""
        mask = np.asarray(det_mask, dtype=bool)
        if mask.shape != self.det_shape:
            raise ValueError(f"det_mask shape {mask.shape} does not match detector shape {self.det_shape}.")
        return np.ascontiguousarray(mask.reshape(-1))

    def _indices(self, mask: np.ndarray):
        """Device indices of a flat mask's selected pixels, uploaded once per distinct mask."""
        key = mask.tobytes()
        indices = self._mask_indices.get(key)
        if indices is not None:
            self._mask_indices.move_to_end(key)
            return indices
        indices = cp.asarray(np.flatnonzero(mask).astype(np.int32, copy=False))
        self._mask_indices[key] = indices
        if len(self._mask_indices) > _MASK_INDEX_CACHE_SIZE:
            self._mask_indices.popitem(last=False)
        return indices

    def _total_counts(self):
        """Per-frame total counts in uint64, cached: every dense mask subtracts its complement from it."""
        if self._totals is None:
            self._totals = cp.empty(self.n_frames, dtype=cp.uint64)
            self._launch(
                f"total_sum_{self._suffix}_4f",
                (128, 4, 1),
                (self._data, self._totals, np.int32(self.det_shape[0] * self.det_shape[1]), np.int32(self.n_frames)),
            )
        return self._totals

    def _selected_sum_uint64(self, indices):
        image = cp.empty(self.n_frames, dtype=cp.uint64)
        self._sum_selected("selected_sum_u64", indices, image)
        return image

    def _sum_selected(self, kernel: str, indices, image, total=None):
        """Sum the selected pixels of every frame, one 32-lane warp per frame, into ``image``.

        With ``total`` the kernel writes ``total - selected`` instead.
        """
        counts = (np.int32(indices.size), np.int32(self.det_shape[0] * self.det_shape[1]), np.int32(self.n_frames))
        inputs = (self._data, indices) if total is None else (self._data, indices, total)
        self._launch(f"{kernel}_{self._suffix}_16f", (32, 16, 1), (*inputs, image, *counts))

    def _reduce_selected_frames(self, kernel: str, indices: np.ndarray, dtype):
        """Reduce selected frames into one pattern, one thread per detector pixel.

        Adjacent threads read adjacent detector values, so the reduction stays
        coalesced without a gathered ``selected_frames x pixels`` tensor.
        """
        pixels = self.det_shape[0] * self.det_shape[1]
        if indices.size == 0:
            return cp.zeros(pixels, dtype=dtype)
        pattern = cp.empty(pixels, dtype=dtype)
        kernels(cp.cuda.Device().id).get_function(f"{kernel}_{self._suffix}")(
            ((pixels + 255) // 256, 1, 1),
            (256, 1, 1),
            (self._data, cp.asarray(indices, dtype=cp.int32), pattern, np.int32(indices.size), np.int32(pixels)),
        )
        return pattern

    def _launch(self, name: str, block: tuple[int, int, int], arguments: tuple):
        """Launch a per-frame kernel: ``block[1]`` frames per block, ``block[0]`` threads per frame."""
        grid = ((self.n_frames + block[1] - 1) // block[1], 1, 1)
        kernels(cp.cuda.Device().id).get_function(name)(grid, block, arguments)

    def _torch_backend(self) -> TorchBackend:
        """Torch view of the same memory for arrays the kernels do not read."""
        if self._torch is None:
            import torch

            self._torch = TorchBackend(torch.from_dlpack(self._data))
        return self._torch


def _require_counts(frames, product: str) -> None:
    """Refuse float data for an exact integer product, as the Torch and NumPy backends do.

    A uint64 sum or uint32 maximum of float intensities truncates every
    sub-unity value to 0, so only integer counts have an exact product; float
    data reduces through ``masked_sum`` and ``reduce_frames`` instead.
    """
    if frames.dtype.kind not in "ui":
        raise TypeError(f"{product} require integer detector data.")
