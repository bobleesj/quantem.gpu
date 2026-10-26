"""Batch broad-search candidates without changing their reductions."""

import cupy as cp
import numpy as np

from ._broad_shared import kernel as saturated_kernel
from .broad_objective import Objective as Baseline
from .broad_objective import code as baseline_source
from .broad_objective import kernel as baseline_kernel

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
        self.scratch = {}
        self.multiprocessors = cp.cuda.runtime.getDeviceProperties(cp.cuda.Device().id)[
            "multiProcessorCount"
        ]

    def score(self, primary, packed, band=(0.2, 0.9)):
        e = self.e
        cache = e._cache
        rows = np.atleast_2d(primary)
        packed = np.asarray(packed, dtype=np.float32).reshape(len(rows), 28)
        _, _, flat, qx, qy, weight = e.thick._prepare_fit_band(band)
        nk, _, _ = e._bf_grid()
        nb = int(flat.size)
        chunks, blocks = (nk + 255) // 256, (nb + 255) // 256
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
        execution = max(4, (self.partial_bytes // (chunks * nb * 12) // 4) * 4)
        if blocks * chunks >= 2 * self.multiprocessors:
            execution = 8
        values = []
        for start in range(0, len(rows), execution):
            batch = min(execution, len(rows) - start)
            saturated = blocks * chunks >= 2 * self.multiprocessors
            tiles = 1 if saturated else (batch + 3) // 4
            capacity = max(4, batch) if saturated else 4
            key = (chunks, tiles, nb, blocks, capacity)
            if key not in self.scratch:
                self.scratch[key] = (
                    cp.empty((tiles, chunks, capacity, nb), cp.complex64),
                    cp.empty((tiles, chunks, capacity, nb), cp.float32),
                    cp.empty((tiles * capacity, nb), cp.complex64),
                    cp.empty((tiles * capacity, nb), cp.float32),
                    cp.empty((tiles, blocks), cp.uint32),
                )
            pn, pd, numer, denom, counter = self.scratch[key]
            counter.fill(0)
            launch_kernel = (
                kernel
                if tiles > 1
                else saturated_kernel
                if blocks * chunks >= 2 * self.multiprocessors
                else baseline_kernel
            )
            launch_grid = (blocks, chunks, tiles) if tiles > 1 else (blocks, chunks)
            launch_kernel(
                launch_grid,
                (256,),
                (
                    e.G_qk,
                    flat,
                    qx.ravel(),
                    qy.ravel(),
                    cache["kx_bf"],
                    cache["ky_bf"],
                    trial[start : start + batch].ravel(),
                    pn,
                    pd,
                    counter,
                    numer,
                    denom,
                    np.int32(nk),
                    np.int64(e.G_qk.shape[1] * e.G_qk.shape[2]),
                    np.int32(nb),
                    np.int32(batch),
                    np.float32(e.wavelength),
                    np.float32(cache["semiangle_rad"]),
                    np.float32(cache["ang_y_rad"]),
                    np.float32(cache["ang_x_rad"]),
                    np.float32(e._factor),
                    np.int32(256),
                    coefficients[start : start + batch].ravel(),
                ),
            )
            # Same reduction within each candidate; no mixing across independent trials.
            num, den = numer[:batch], denom[:batch]
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
