"""CUDA decomposition of full-resolution camera masks into exact spatial sums."""

from functools import cache
from pathlib import Path

import numpy as np


@cache
def _kernels(device):
    import cupy as cp

    with cp.cuda.Device(device):
        module = cp.RawModule(
            code=Path(__file__).with_name("kernels").joinpath("mask_plan.cu").read_text(),
            options=("--std=c++17",),
        )
        return module.get_function("mask_leaves"), module.get_function("mask_roots")


class CudaMaskPlanner:
    """Reuse bounded device scratch for exact 8/32-pixel mask decomposition."""

    def __init__(self, shape):
        import cupy as cp

        self.shape = shape
        self.device = cp.cuda.Device().id
        self.tile_rows, self.tile_cols = [(size + 7) // 8 for size in shape]
        self.tiles = self.tile_rows * self.tile_cols
        self.roots = ((self.tile_rows + 3) // 4) * ((self.tile_cols + 3) // 4)
        self.mask = cp.empty(shape, cp.int32)
        self.leaves = cp.empty(self.tiles, cp.int32)
        self.fields = cp.empty(self.tiles + self.roots, cp.uint32)
        self.field_weights = cp.empty(self.tiles + self.roots, cp.int32)
        self.pixels = cp.empty(np.prod(shape), cp.uint32)
        self.pixel_weights = cp.empty(np.prod(shape), cp.int32)
        self.counts = cp.empty(2, cp.uint32)
        self._host_owners = [cp.cuda.alloc_pinned_memory(array.nbytes) for array in (
            self.fields, self.field_weights, self.pixels, self.pixel_weights,
        )]
        self._host_arrays = [np.frombuffer(owner, array.dtype, count=array.size)
                             for owner, array in zip(self._host_owners, (
                                 self.fields, self.field_weights, self.pixels, self.pixel_weights,
                             ))]
        self.leaf_kernel, self.root_kernel = _kernels(self.device)

    def __call__(self, values):
        import cupy as cp

        with cp.cuda.Device(self.device):
            self.mask.set(values)
            self.counts.fill(0)
            self.leaf_kernel((self.tiles,), (32,), (
                self.mask, self.leaves, self.pixels, self.pixel_weights, self.counts,
                np.uint32(self.shape[0]), np.uint32(self.shape[1]),
            ))
            self.root_kernel((self.roots,), (32,), (
                self.leaves, self.fields, self.field_weights, self.counts,
                np.uint32(self.tile_rows), np.uint32(self.tile_cols),
            ))
            fields, pixels = map(int, self.counts.get())
            # Only the compact mask description crosses back to the existing query
            # scheduler. Raw diffraction counts and spatial indexes remain resident.
            stream = cp.cuda.get_current_stream()
            results = []
            for source, host, count in zip(
                (self.fields, self.field_weights, self.pixels, self.pixel_weights),
                self._host_arrays, (fields, fields, pixels, pixels),
            ):
                if count:
                    source.data.copy_to_host_async(host.ctypes.data, count * 4, stream)
                results.append(host[:count])
            stream.synchronize()
            # The scheduler may retain a plan while another mask is decomposed.
            return tuple(array.copy() for array in results)
