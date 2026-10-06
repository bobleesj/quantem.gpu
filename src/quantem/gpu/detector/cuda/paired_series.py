"""Joint detector queries over paired-count acquisitions with the polar planner.

The paired resident layout (``resident.cuda.paired``) stores its spatial index as
radial-angular pixel groups. Planning a detector mask against that index and
launching the paired query kernels is detector work, so it lives here, above the
resident layer that only stores and decodes the counts.
"""

import math

import numpy as np

from quantem.gpu.detector.cuda.streamed_series import StreamedSeriesCompute
from quantem.gpu.device.cuda_runtime import cp
from quantem.gpu.io.dataset import Dataset4dstemGPU
from quantem.gpu.resident.cuda.paired import (
    QUERY_ABI,
    PairedCounts,
    kernels,
    polar_layout,
)

# Residual pixels per paired launch: the work lists index the selection in uint16.
_RESIDUAL_BATCH = 65535


class PairedSeriesCompute(StreamedSeriesCompute):
    """Joint detector queries over paired-layout sources with the polar planner.

    Selected by ``detector.prepare`` when every acquisition is a
    :class:`PairedCounts`. The descriptor rows, output ownership and the private
    baseline for incremental masks are inherited; the plan, the index summation and
    the residual decoder are the paired kernels.
    """

    def __init__(self, acquisitions):
        sources = [item.data if isinstance(item, Dataset4dstemGPU) else item for item in acquisitions]
        if not sources or not all(isinstance(source, PairedCounts) for source in sources):
            raise TypeError("Paired queries need paired-layout sources only; mix nothing else into the series.")
        super().__init__(acquisitions)
        if any(chunk.scans % source.interval for source in sources for chunk in source.chunks):
            raise ValueError("Every paired chunk must hold complete 512-scan blocks.")
        self.backend_metadata["query_abi"] = QUERY_ABI
        paired = kernels(self.device)
        blocks = math.ceil(self.max_scans / self.interval)
        total = self.chunk_count * blocks
        with cp.cuda.Device(self.device):
            self.work_counts = cp.empty(total, cp.uint32)
            self.work = {"capacity": 0, "array": None}
            weights = cp.zeros(self.pixels, cp.float64)
            paired["weights"](((self.pixels + 255) // 256, self.chunk_count), (256,), (self.descriptors, weights, np.uint32(self.pixels), np.uint32(self.chunk_count)))
            blocks_total = sum(chunk.scans / self.interval for source in self.index_owners for chunk in source.chunks)
            self.pixel_weights = weights.get() / blocks_total
        # Leaves and roots of the polar index weigh each pixel by its expected decode cost.
        self.permutation, self.leaves, self.roots = polar_layout(self.det_shape)
        self.indexed = self.permutation >= 0
        tile_weight = np.zeros(self.permutation.size, np.float64)
        tile_weight[self.indexed] = self.pixel_weights.ravel()[self.permutation[self.indexed]]
        self.tile_weight = tile_weight.reshape(self.leaves, 64)
        shared = 16 * self.fields
        # The inherited query code launches through these names; adapt each launch to the paired kernels.
        self.kernels = dict(self.kernels)
        for bits in (32, 64):
            def index(grid, block, args, bits=bits):
                stride = self.block_stride
                per_block = self.interval // block[0]
                launched = (blocks + stride - 1) // stride * per_block   # thread blocks per chunk at this stride
                paired[f"index_u{bits}"]((launched, grid[1]), block, (*args[:8], args[9], np.uint32(stride)), shared_mem=shared)

            def residual(grid, block, args, bits=bits):
                # The work lists hold uint16 positions into the selected pixels, so a larger
                # selection (moments or an intricate mask on a detector above 256 x 256)
                # runs in batches of 65,535 pixels, each adding its exact counts to the result.
                count = int(args[3])
                capacity = min(count, _RESIDUAL_BATCH)
                if capacity > self.work["capacity"]:
                    self.work["array"] = cp.empty((total, capacity), cp.uint16)
                    self.work["capacity"] = capacity
                stride = self.block_stride
                launched = (blocks + stride - 1) // stride   # 512-scan blocks per chunk at this stride
                items = self.chunk_count * launched
                extra = (self.work["array"], self.work_counts, np.uint32(launched), np.uint32(stride))
                for start in range(0, count, _RESIDUAL_BATCH):
                    stop = min(start + _RESIDUAL_BATCH, count)
                    batch = (args[0], args[1][start:stop], args[2][start:stop], np.uint32(stop - start), *args[4:8])
                    self.work_counts.fill(0)
                    paired[f"plan_u{bits}"]((items,), (256,), (*batch, *extra))
                    paired[f"residual_u{bits}"]((items, (stop - start + 255) // 256), (256,), (*batch, *extra))

            self.kernels[f"index_u{bits}"] = index
            self.kernels[f"residual_u{bits}"] = residual
        for bits in (8, 16):
            def frame(grid, block, args, bits=bits):
                paired[f"frame_u{bits}"](grid, block, args[:5])

            self.kernels[f"frame_u{bits}"] = frame

    def _plan(self, values):
        """Decompose a signed mask into polar index fields plus signed residual pixels.

        Each 64-pixel leaf takes the value (0, 1 or -1) that most of its pixels
        hold, weighted by each pixel's expected decode cost; each 16-leaf root does
        the same over its leaves. Pixels that disagree with their leaf become
        residual corrections, so the plan sums to the mask exactly.
        """
        options = np.array([0, 1, -1], np.int32)
        pixels = self.permutation[self.indexed]
        ordered = np.zeros(self.permutation.size, np.int32)
        ordered[self.indexed] = values.ravel()[pixels]
        tiles = ordered.reshape(self.leaves, 64)
        counts = np.stack([((tiles == value) * self.tile_weight).sum(axis=1) for value in options])
        leaf = options[counts.argmax(axis=0)]
        residual = np.zeros(values.size, np.int32)
        residual[pixels] = (ordered - np.repeat(leaf, 64))[self.indexed]
        padded = np.zeros(self.roots * 16, np.int32)
        padded[: self.leaves] = leaf
        counts = np.stack([(padded.reshape(self.roots, 16) == value).sum(axis=1) for value in options])
        root = options[counts.argmax(axis=0)]
        leaf = leaf - np.repeat(root, 16)[: self.leaves]
        fields = np.concatenate((leaf, root))
        selected_fields = np.flatnonzero(fields).astype(np.uint32)
        selected_pixels = np.flatnonzero(residual).astype(np.uint32)
        return selected_fields, fields[selected_fields], selected_pixels, residual[selected_pixels]

    def _cost(self, selection):
        """Expected decode work: one unit per index field plus each residual pixel's stream cost."""
        return len(selection[0]) + float(self.pixel_weights.ravel()[selection[2]].sum())
