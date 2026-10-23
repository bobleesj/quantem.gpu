"""CUDA broad-search objective and dimensionless aberration coordinates."""

from types import SimpleNamespace

import cupy as cp
import numpy as np
from quantem.gpu.ssb.cuda.kernels.engine import thick_fit_batch_kernel
from quantem.gpu.ssb.higher_order import CHI_SOURCE
from quantem.gpu.ssb.fifth_order_parameters import Coordinates, public_params, vector

code = thick_fit_batch_kernel.code
code = code.replace(
    "int k_chunk)", "int k_chunk, const float* __restrict__ coefficients)"
)
old1 = "fmaf(m2, c12s[b], fmaf(m1, c12c[b], m0 * c10[b]))"
old2 = "fmaf(p2, c12s[b], fmaf(p1, c12c[b], p0 * c10[b]))"
assert code.count(old1) == 1 and code.count(old2) == 1
code = code.replace(
    old1,
    "(ssb_chi_full(kxv-qx, kyv-qy, wl, coefficients+b*28)-ssb_chi_full(kxv,kyv,wl,coefficients+b*28))",
)
code = code.replace(
    old2,
    "(ssb_chi_full(kxv+qx, kyv+qy, wl, coefficients+b*28)-ssb_chi_full(kxv,kyv,wl,coefficients+b*28))",
)
code = (
    "#define SSB_DEVICE __device__ __forceinline__\n#define SSB_COEFFICIENTS const float*\n"
    + CHI_SOURCE
    + "\n"
    + code
)
kernel = cp.RawKernel(code, "thick_fit_batch")


class Objective(Coordinates):
    def __init__(self, engine):
        self.e = engine
        self.p = SimpleNamespace(
            wavelength=engine.wavelength, semiangle_rad=engine._cache["semiangle_rad"]
        )
        self.calls = 0

    def score(self, primary, packed, band=(0.2, 0.9)):
        e = self.e
        cache = e._cache
        rows = np.atleast_2d(primary)
        packed = np.asarray(packed, dtype=np.float32).reshape(len(rows), 28)
        _, _, flat, qx, qy, weight = e.thick._prepare_fit_band(band)
        nk, _ny, _nx = e._bf_grid()
        nb = int(flat.size)
        chunks = (nk + 255) // 256
        blocks = (nb + 255) // 256
        values = []
        for start in range(0, len(rows), 4):
            r = rows[start : start + 4]
            b = len(r)
            trial = cp.asarray(
                np.stack(
                    [
                        r[:, 0],
                        r[:, 1],
                        np.cos(2 * r[:, 2]),
                        np.sin(2 * r[:, 2]),
                        r[:, 5],
                        r[:, 3] * 1e-3,
                        r[:, 4] * 1e-3,
                    ],
                    axis=1,
                ).astype(np.float32)
            )
            pn = cp.empty((chunks, b, nb), cp.complex64)
            pd = cp.empty((chunks, b, nb), cp.float32)
            numer = cp.empty((b, nb), cp.complex64)
            denom = cp.empty((b, nb), cp.float32)
            kernel(
                (blocks, chunks),
                (256,),
                (
                    e.G_qk,
                    flat,
                    qx.ravel(),
                    qy.ravel(),
                    cache["kx_bf"],
                    cache["ky_bf"],
                    trial.ravel(),
                    pn,
                    pd,
                    cp.zeros(blocks, cp.uint32),
                    numer,
                    denom,
                    np.int32(nk),
                    np.int64(e.G_qk.shape[1] * e.G_qk.shape[2]),
                    np.int32(nb),
                    np.int32(b),
                    np.float32(e.wavelength),
                    np.float32(cache["semiangle_rad"]),
                    np.float32(cache["ang_y_rad"]),
                    np.float32(cache["ang_x_rad"]),
                    np.float32(e._factor),
                    np.int32(256),
                    cp.asarray(packed[start : start + b].ravel()),
                ),
            )
            scores = cp.sum(
                weight[None, :]
                * cp.where(
                    denom > 0, cp.abs(numer) ** 2 / cp.where(denom > 0, denom, 1), 0
                ),
                axis=1,
            )
            values.extend(cp.asnumpy(scores).astype(float))
            self.calls += b
        return np.array(values)

    def evaluate(self, z, order, band=(0.2, 0.9)):
        return self.score(*self.encode(np.atleast_2d(z), order), band=band)
