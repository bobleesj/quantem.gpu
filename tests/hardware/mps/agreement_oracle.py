"""Test oracle: the per-term model agreement of the thick-sample direct ptychography model with higher-order aberrations, on MPS.

The fifth-order objective (``mps.fifth_order``) compiles the same agreement kernel; this module runs it on trial rows
so the tests can check that kernel against independent float64 models and the objective against the kernel.

Same objective as ``thick_sample.thick_fit_batch`` (the least-squares agreement of the direct ptychography model with G(q, k) over a
band of q, larger is better), with the probe phase chi evaluated from the 14-term Krivanek polynomial of
``higher_order`` instead of C10 and C12 only. The fit's angle-sum kernel (``resident_fit``) depends on chi being a
quadratic form, which coma, trefoil and spherical aberration are not, so this is the per-term kernel: chi at k - q
and k + q per (k, q, trial), chi at k once per (k, trial).

Trials are rows of ``(thickness_A, tilt_row_mrad, tilt_col_mrad, z_1 ... z_{2 N})`` where the z are the first ``N``
Krivanek terms (``higher_order.ORDERS`` order) as Cartesian vectors, Angstrom: ``(C cos m phi, C sin m phi)`` for
each term, ``(C, 0)`` for a round term (C10, C30). The vector form is smooth through C = 0, where the azimuth is
undefined, so it is also the coordinate the Newton refinement steps in.
"""

from functools import lru_cache

import numpy as np

from quantem.gpu.direct_ptycho._execution import checkpoint
from quantem.gpu.direct_ptycho.higher_order import ORDERS
from quantem.gpu.direct_ptycho.mps.fifth_order import _AGREEMENT_SOURCE
from quantem.gpu.direct_ptycho.mps.hardware import require_mlx
from quantem.gpu.direct_ptycho.mps.thick_sample import (
    _FIT_DISPATCH_BATCH,
    _K_CHUNK,
    _THICK_FIT_BATCH_HEADER,
    _band,
)

# Execution tile: trials per thread. Larger tiles reuse the (k, q) geometry more but raise register pressure.
_TILE = 4
# Queue of partial sums before a readback, as in ``thick_sample``.
_PENDING_BYTES = 64 << 20

_CHI_HEADER = """
// chi(row, col) / (2 pi / wl) of the first NT Krivanek terms; c holds 2 NT Cartesian coefficients already divided by
// (n + 1), as higher_order.pack_coefficients packs them.
template <int NT>
inline float ho_chi(float row, float col, float wl, const device float* c) {
    float x = row * wl, y = col * wl;
    float r2 = x * x + y * y;
    float radial[4] = {1.0f, r2, r2 * r2, r2 * r2 * r2};
    float re[7], im[7];
    re[0] = 1.0f; im[0] = 0.0f;
    for (int m = 1; m <= 6; ++m) {
        re[m] = re[m - 1] * x - im[m - 1] * y;
        im[m] = im[m - 1] * x + re[m - 1] * y;
    }
    const int azimuth[14] = {0, 2, 1, 3, 0, 2, 4, 1, 3, 5, 0, 2, 4, 6};
    const int power[14] = {1, 0, 1, 0, 2, 1, 0, 2, 1, 0, 3, 2, 1, 0};
    float chi = 0.0f;
    for (int i = 0; i < NT; ++i) {
        chi += radial[power[i]] * (re[azimuth[i]] * c[2 * i] + im[azimuth[i]] * c[2 * i + 1]);
    }
    return (6.283185307179586f / wl) * chi;
}
"""

_PROBE_SOURCE = """
    uint k = thread_position_in_grid.x;
    if (k >= NUM_BF) return;
    float4 gk = tf_geometry(kx[k], ky[k], scalars[0], scalars[1], scalars[2], scalars[3]);
    for (uint i = 0; i < 4; ++i) probe_geometry[k * 4 + i] = gk[i];
    for (int b = 0; b < TOTAL; ++b) {
        probe_chi[k * TOTAL + b] = ho_chi<NT>(kx[k], ky[k], scalars[0], trial + b * STRIDE + 3);
    }
"""






def num_terms(rows: np.ndarray) -> int:
    """Krivanek terms an agreement row carries: (width - 3) / 2."""
    width = np.shape(rows)[-1]
    if width < 5 or (width - 3) % 2 or (width - 3) // 2 > len(ORDERS):
        raise ValueError(f"A trial row is (thickness, tilt_row, tilt_col) and 2 N Cartesian coefficients; got {width}.")
    return (width - 3) // 2


def packed(rows: np.ndarray) -> np.ndarray:
    """Agreement rows -> the kernels' float32 rows: tilt in rad and each Cartesian coefficient divided by (n + 1)."""
    rows = np.atleast_2d(np.asarray(rows, dtype=np.float64))
    terms = num_terms(rows)
    order = np.repeat(np.asarray([n for n, _ in ORDERS[:terms]], dtype=np.float64), 2)
    out = np.empty_like(rows)
    out[:, 0] = rows[:, 0]
    out[:, 1:3] = rows[:, 1:3] * 1e-3
    out[:, 3:] = rows[:, 3:] / (order + 1.0)
    return out.astype(np.float32)


@lru_cache(maxsize=1)
def _probe_kernel():
    return require_mlx().fast.metal_kernel(
        name="direct_higher_order_probe",
        input_names=["kx", "ky", "scalars", "trial"],
        output_names=["probe_geometry", "probe_chi"],
        header=_THICK_FIT_BATCH_HEADER + _CHI_HEADER,
        source=_PROBE_SOURCE,
    )


@lru_cache(maxsize=1)
def _agreement_kernel():
    return require_mlx().fast.metal_kernel(
        name="direct_higher_order_agreement",
        input_names=["G", "band_flat", "band_qx", "band_qy", "kx", "ky", "trial", "scalars", "probe_geometry",
                     "probe_chi"],
        output_names=["numer_real", "numer_imag", "denom"],
        header=_THICK_FIT_BATCH_HEADER + _CHI_HEADER,
        source=_AGREEMENT_SOURCE,
    )


def agreement_batch(prepared, rows, band_inv_A: tuple[float, float] = (0.2, 0.9)) -> np.ndarray:
    """The thick-sample model agreement for each trial row (see the module docstring); larger is better.

    Equals ``thick_sample.thick_fit_batch`` at the same C10, C12, phi12, tilt and depth spread when the higher-order
    coefficients are 0 (the same model, chi evaluated by the polynomial instead of alpha^2 (C10 + C12 cos 2 phi)).
    """
    mx = prepared.mx
    trial = packed(rows)
    terms = num_terms(trial)
    stride = trial.shape[1]
    flat, band_qx, band_qy, weight = _band(prepared, band_inv_A)
    n_band = int(flat.size)
    g = prepared.g_qk
    storage_bf = int(g.shape[0])
    plane = int(g.shape[1]) * int(g.shape[2])
    k_chunks = (storage_bf + _K_CHUNK - 1) // _K_CHUNK
    scalars = mx.array([prepared.wavelength, prepared.semiangle_rad, prepared.ang_y_rad, prepared.ang_x_rad,
                        prepared.factor], dtype=mx.float32)
    per_candidate = (k_chunks + 1) * n_band * 3 * 4
    execution = max(_FIT_DISPATCH_BATCH, (_PENDING_BYTES // per_candidate // _TILE) * _TILE)
    out = np.empty(len(trial))
    for start in range(0, len(trial), execution):
        checkpoint()
        rows_np = trial[start:start + execution]
        batch = len(rows_np)
        tile = min(_TILE, batch)
        rows_mx = mx.array(rows_np.ravel())
        probe_geometry, probe_chi = _probe_kernel()(
            inputs=[prepared.kx, prepared.ky, scalars, rows_mx],
            template=[("NUM_BF", storage_bf), ("TOTAL", batch), ("STRIDE", stride), ("NT", terms)],
            grid=(storage_bf, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(storage_bf, 4), (storage_bf, batch)],
            output_dtypes=[mx.float32, mx.float32],
        )
        numer_real, numer_imag, denom = _agreement_kernel()(
            inputs=[g, flat, band_qx, band_qy, prepared.kx, prepared.ky, rows_mx, scalars, probe_geometry, probe_chi],
            template=[("B", tile), ("TOTAL", batch), ("N_BAND", n_band), ("NUM_BF", storage_bf), ("PLANE", plane),
                      ("K_CHUNK", _K_CHUNK), ("STRIDE", stride), ("NT", terms)],
            grid=(n_band, k_chunks, (batch + tile - 1) // tile),
            threadgroup=(64, 1, 1),
            output_shapes=[(k_chunks, batch, n_band)] * 3,
            output_dtypes=[mx.float32] * 3,
        )
        sums = mx.stack([mx.sum(numer_real, axis=0), mx.sum(numer_imag, axis=0), mx.sum(denom, axis=0)])
        values = np.asarray(sums).astype(np.float64)
        den = values[2]
        ok = den > 0
        power = np.where(ok, (values[0] ** 2 + values[1] ** 2) / np.where(ok, den, 1.0), 0.0)
        out[start:start + batch] = (weight[None] * power).sum(axis=1)
    return out

