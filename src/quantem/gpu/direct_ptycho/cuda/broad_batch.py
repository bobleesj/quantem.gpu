"""Batch broad-search candidates without changing their reductions."""

import cupy as cp
import numpy as np

from ._broad_shared import kernel as saturated_kernel
from .broad_objective import Objective as Baseline
from .broad_objective import code as baseline_source
from .broad_objective import kernel as baseline_kernel
from .thick import SlabSums

source = baseline_source.replace(
    "    // per parameter set:",
    """    int start_trial = (int)blockIdx.z * 4;
    int tile_count = min(4, B - start_trial);
    int slices = (num_bf + k_chunk - 1) / k_chunk;
    trial += (size_t)start_trial * 7;
    coefficients += (size_t)start_trial * 28;
    partial_numer += (size_t)blockIdx.z * slices * 4 * n_band;
    partial_denom += (size_t)blockIdx.z * slices * 4 * n_band;
    numer += (size_t)start_trial * n_band;
    denom += (size_t)start_trial * n_band;
    arrivals += (size_t)blockIdx.z * gridDim.x;
    B = tile_count;
    // per parameter set:""",
)
assert source != baseline_source
kernel = cp.RawKernel(source, "thick_fit_batch")


class Objective(Baseline):
    def __init__(self, engine, partial_bytes=64 << 20):
        super().__init__(engine)
        self.partial_bytes = partial_bytes
        self.multiprocessors = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)[
            "multiProcessorCount"
        ]

    def score(self, primary, packed, band=(0.2, 0.9)):
        rows = np.atleast_2d(primary)
        packed = np.asarray(packed, dtype=np.float32).reshape(len(rows), 28)
        _, _, flat, qx, qy, weight = self.e.thick._prepare_fit_band(band)
        trial = cp.asarray(
            np.stack(
                [
                    rows[:, 0],
                    rows[:, 1],
                    np.cos(2 * rows[:, 2]),
                    np.sin(2 * rows[:, 2]),
                    rows[:, 5],
                    rows[:, 3] * 1e-3,
                    rows[:, 4] * 1e-3,
                ],
                axis=1,
            ).astype(np.float32)
        )
        coefficients = cp.asarray(packed)
        values = []
        for start, batch, num, den in self._batches(rows, trial, coefficients, flat, qx, qy):
            # Same reduction within each candidate; no mixing across independent trials.
            scores = cp.sum(
                weight[None, :]
                * cp.where(
                    den > 0,
                    cp.abs(num) ** 2 / cp.where(den > 0, den, 1),
                    0,
                ),
                axis=1,
            )
            values.append(scores)
            self.calls += batch
        return cp.asnumpy(cp.concatenate(values)).astype(float)

    def _batches(self, rows, trial, coefficients, flat, qx, qy, stride=1):
        """Per execution batch of candidates: (start, size, numerators, denominators) over the band frequencies.

        The candidates' numerators N_q and denominators D_q from the fused kernels, summed over the slabs of
        bright-field pixels the engine gives (``DirectEngine.band_slabs``; one slab when G is whole), on every
        ``stride``-th pixel (``_pixels``).
        """
        e = self.e
        cache = e._cache
        nb = int(flat.size)
        blocks = (nb + 255) // 256
        num_bf, _, _ = e._bf_grid()
        # the launch layout is the whole G's, so a G held whole launches exactly as before slabs
        whole_chunks = (len(range(0, num_bf, stride)) + 255) // 256
        execution = max(4, (self.partial_bytes // (whole_chunks * nb * 12) // 4) * 4)
        saturated = blocks * whole_chunks >= 2 * self.multiprocessors
        if saturated:
            execution = 8
        kx_all, ky_all = self._pixel_arrays(stride)
        for start in range(0, len(rows), execution):
            batch = min(execution, len(rows) - start)
            sums = SlabSums()
            for first, stop, slab, step in e.band_slabs(stride):
                nk = len(range(first, stop, stride))
                if nk == 0:
                    continue
                chunks = (nk + 255) // 256
                tiles = 1 if saturated else (batch + 3) // 4
                capacity = max(4, batch) if saturated else 4
                # from the memory pool at each launch: buffers kept per launch layout (batch size, stride, resident
                # or streamed slab) added up over a fit to about 300 MiB on a 79,441-frequency band, past the working
                # memory a small device keeps beside its band spectrum (2026-11-13)
                pn = cp.empty((tiles, chunks, capacity, nb), cp.complex64)
                pd = cp.empty((tiles, chunks, capacity, nb), cp.float32)
                numer = cp.empty((tiles * capacity, nb), cp.complex64)
                denom = cp.empty((tiles * capacity, nb), cp.float32)
                counter = cp.zeros((tiles, blocks), cp.uint32)
                launch_kernel = kernel if tiles > 1 else saturated_kernel if saturated else baseline_kernel
                launch_grid = (blocks, chunks, tiles) if tiles > 1 else (blocks, chunks)
                pixels = slice(first // stride, first // stride + nk)
                launch_kernel(
                    launch_grid,
                    (256,),
                    (
                        slab, flat, qx.ravel(), qy.ravel(), kx_all[pixels], ky_all[pixels],
                        trial[start : start + batch].ravel(), pn, pd, counter, numer, denom,
                        np.int32(nk), np.int64(slab.shape[1] * slab.shape[2] * step), np.int32(nb), np.int32(batch),
                        np.float32(e.wavelength), np.float32(cache["semiangle_rad"]), np.float32(cache["ang_y_rad"]),
                        np.float32(cache["ang_x_rad"]), np.float32(e._factor), np.int32(256),
                        coefficients[start : start + batch].ravel(),
                    ),
                )
                sums.add(numer[:batch], denom[:batch])
            numerators, denominators = sums.result()
            yield start, batch, numerators, denominators

    def _pixel_arrays(self, stride):
        """(k row, k column) of every ``stride``-th bright-field pixel, contiguous."""
        cache = self.e._cache
        if stride == 1:
            return cache["kx_bf"], cache["ky_bf"]
        return cp.ascontiguousarray(cache["kx_bf"][::stride]), cp.ascontiguousarray(cache["ky_bf"][::stride])
