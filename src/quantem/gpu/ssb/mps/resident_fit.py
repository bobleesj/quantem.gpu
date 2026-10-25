"""Resident joint-model scores with extended-precision band reduction."""

from functools import lru_cache

import numpy as np

from quantem.gpu.ssb._execution import checkpoint
from quantem.gpu.ssb._resident_search import _Device
from quantem.gpu.ssb.mps.thick_sample import (
    _THICK_FIT_BATCH_HEADER,
    _FIT_DISPATCH_BATCH,
    _FIT_PENDING_BYTES,
    _K_CHUNK,
    _band,
    _thick_fit_batch_kernel,
    _trial_probe_kernel,
)

REDUCE = r"""
if(index>=BATCH*256)return;
int b=int(index)/256,lane=int(index)%256;Real total(0);
for(int q=lane;q<BAND;q+=256){
    Real den(sums[2*BATCH*BAND+b*BAND+q]);
    if(den>Real(0)){
        Real re(sums[b*BAND+q]),im(sums[BATCH*BAND+b*BAND+q]);
        total=total+Real(weight[q])*(re*re+im*im)/den;
    }
}
store_real(out0,int(index),total);
"""


# The joint objective's model terms with the phases split by angle sums (``_ANGLE_SUM_SOURCE``); False keeps the
# per-term kernel of ``thick_sample.thick_fit_batch``.
ANGLE_SUMS = True

# Same model as ``thick_sample._THICK_FIT_BATCH_SOURCE``. The two-term aberration function is a quadratic form, so
#     chi(q - k) - chi(k) = Psi - u.k,   chi(q + k) - chi(k) = Psi + u.k,   Psi = q^T M q,   u = 2 M q,
# and the depth-phase rates are x1 = y - P and x2 = y + P with y = C + B, C = -pi t q.theta, B = t f q.k and
# P = t f |q|^2 / 2 (f = factor wl^2). One precise sincos of u.k and one of y per (q, k, trial) give both sidebands'
# phases and both sines (sin(y -/+ P) = sin y cos P -/+ cos y sin P), with Psi and P once per (q, trial):
#     gamma = exp(i u.k) (a exp(-i Psi) - b exp(i Psi)),   a = w1 A(q - k) A(k),   b = w2 A(q + k) A(k),
#     G conj(gamma) = H ((a - b) cos Psi + i (a + b) sin Psi),   |gamma|^2 = (a - b)^2 cos^2 Psi + (a + b)^2 sin^2 Psi,
# H = G exp(-i u.k), accumulated over k and combined with Psi at the end: two sums of squares, no cancellation in
# the denominator. sinc(x) = sin x / x above |x| = 1 and its Taylor series to x^10 below (truncation 2e-10).
# Stored BF pixels are in detector raster order: along a detector row k advances by a constant step dk, so
# exp(i u.k) and exp(i y) are multiplied by exp(i u.dk) and exp(i t f q.dk) from one pixel to the next and evaluated
# exactly at each chunk's first pixel, after a row break and every ANCHOR pixels. Each phase then carries the float32
# rounding of u.k plus at most ANCHOR - 1 step rotations, below the rounding of the per-term chi difference it
# replaces; the objective matches the per-term kernel and the torch reference to ~1e-7 relative on gold.
_ANGLE_SUM_SOURCE = """
    uint q = thread_position_in_grid.x;
    uint chunk = thread_position_in_grid.y;
    uint base = thread_position_in_grid.z * B;
    if (q >= N_BAND) return;
    float wl = scalars[0], semi = scalars[1], ang_y = scalars[2], ang_x = scalars[3], factor = scalars[4];
    float f = factor * wl * wl;
    float qx = band_qx[q], qy = band_qy[q];
    uint off = band_flat[q];
    float ux[B], uy[B], cos_psi[B], sin_psi[B], c_q[B], cos_p[B], sin_p[B], p_q[B], bx[B], by[B];
    float2 n_minus[B], n_plus[B];
    float s_minus[B], s_plus[B];
    for (int b = 0; b < B; ++b) {
        uint i = metal::min(base + b, uint(TOTAL) - 1u);
        float c10 = trial[i * 7], c12 = trial[i * 7 + 1], c2 = trial[i * 7 + 2], s2 = trial[i * 7 + 3];
        float t = trial[i * 7 + 4], thr = trial[i * 7 + 5], thc = trial[i * 7 + 6];
        float m00 = f * (c10 + c12 * c2), m01 = f * (c12 * s2), m11 = f * (c10 - c12 * c2);
        ux[b] = 2.0f * (m00 * qx + m01 * qy);
        uy[b] = 2.0f * (m01 * qx + m11 * qy);
        float psi = qx * (m00 * qx + m01 * qy) + qy * (m01 * qx + m11 * qy);
        sin_psi[b] = metal::precise::sincos(psi, cos_psi[b]);
        c_q[b] = -3.141592653589793f * t * (qx * thr + qy * thc);
        p_q[b] = 0.5f * t * f * (qx * qx + qy * qy);
        sin_p[b] = metal::precise::sincos(p_q[b], cos_p[b]);
        bx[b] = t * f * qx;
        by[b] = t * f * qy;
        n_minus[b] = n_plus[b] = float2(0.0f);
        s_minus[b] = s_plus[b] = 0.0f;
    }
    uint k0 = chunk * K_CHUNK;
    uint k1 = metal::min(uint(NUM_BF), k0 + uint(K_CHUNK));
    // exp(i u.k) and exp(i y) advance by one detector-row step per BF pixel and are evaluated exactly at the
    // chunk's first pixel, after a row break and every ANCHOR pixels (stored BF pixels are in raster order)
    float2 eu[B], ey[B], su_step[B], sy_step[B];
    for (int b = 0; b < B; ++b) {
        float c, sn = metal::precise::sincos(ux[b] * step_k[0] + uy[b] * step_k[1], c);
        su_step[b] = float2(c, sn);
        sn = metal::precise::sincos(bx[b] * step_k[0] + by[b] * step_k[1], c);
        sy_step[b] = float2(c, sn);
        eu[b] = ey[b] = float2(1.0f, 0.0f);
    }
    for (uint k = k0; k < k1; ++k) {
        float kxv = kx[k], kyv = ky[k];
        bool anchor = k == k0 || neighbour[k] == 0.0f || (k - k0) % ANCHOR == 0;
        for (int b = 0; b < B; ++b) {
            if (anchor) {
                float c, sn = metal::precise::sincos(ux[b] * kxv + uy[b] * kyv, c);
                eu[b] = float2(c, sn);
                sn = metal::precise::sincos(c_q[b] + (bx[b] * kxv + by[b] * kyv), c);
                ey[b] = float2(c, sn);
            } else {
                eu[b] = float2(eu[b].x * su_step[b].x - eu[b].y * su_step[b].y,
                               eu[b].x * su_step[b].y + eu[b].y * su_step[b].x);
                ey[b] = float2(ey[b].x * sy_step[b].x - ey[b].y * sy_step[b].y,
                               ey[b].x * sy_step[b].y + ey[b].y * sy_step[b].x);
            }
        }
        float4 gk = float4(probe_geometry[k * 4], probe_geometry[k * 4 + 1],
                           probe_geometry[k * 4 + 2], probe_geometry[k * 4 + 3]);
        float4 gm = tf_geometry(qx - kxv, qy - kyv, wl, semi, ang_y, ang_x);
        float4 gp = tf_geometry(qx + kxv, qy + kyv, wl, semi, ang_y, ang_x);
        float a1 = gm.w * gk.w, a2 = gp.w * gk.w;
        if (a1 == 0.0f && a2 == 0.0f) continue;      // no double overlap: gamma = 0
        auto g = G[(size_t)k * (size_t)PLANE + (size_t)off];
        float gre = g.real, gim = g.imag;
        for (int b = 0; b < B; ++b) {
            float cu = eu[b].x, su = eu[b].y;
            float hr = gre * cu + gim * su, hi = gim * cu - gre * su;      // H = G exp(-i u.k)
            float wa = a1, wb = a2;
            if (p_q[b] != 0.0f || c_q[b] != 0.0f || bx[b] != 0.0f || by[b] != 0.0f) {
                float y = c_q[b] + (bx[b] * kxv + by[b] * kyv);
                float cy = ey[b].x, sy = ey[b].y;
                if (a1 != 0.0f) wa = depth_sinc(y - p_q[b], sy * cos_p[b] - cy * sin_p[b]) * a1;
                if (a2 != 0.0f) wb = depth_sinc(y + p_q[b], sy * cos_p[b] + cy * sin_p[b]) * a2;
            }
            float d = wa - wb, s = wa + wb;
            n_minus[b] += d * float2(hr, hi);
            n_plus[b] += s * float2(hr, hi);
            s_minus[b] += d * d;
            s_plus[b] += s * s;
        }
    }
    for (int b = 0; b < B; ++b) {
        if (base + b >= TOTAL) break;
        size_t o = ((size_t)chunk * TOTAL + base + b) * N_BAND + q;
        // G conj(gamma) = H ((a - b) cos Psi + i (a + b) sin Psi)
        numer_real[o] = cos_psi[b] * n_minus[b].x - sin_psi[b] * n_plus[b].y;
        numer_imag[o] = cos_psi[b] * n_minus[b].y + sin_psi[b] * n_plus[b].x;
        denom[o] = cos_psi[b] * cos_psi[b] * s_minus[b] + sin_psi[b] * sin_psi[b] * s_plus[b];
    }
"""


_SINC_HEADER = """
inline float depth_sinc(float x, float s) {
    float x2 = x * x;
    float series = 1.0f - x2 * (1.0f / 6.0f) * (1.0f - x2 * (1.0f / 20.0f) * (1.0f - x2 * (1.0f / 42.0f)
                 * (1.0f - x2 * (1.0f / 72.0f) * (1.0f - x2 * (1.0f / 110.0f)))));
    return metal::abs(x) < 1.0f ? series : s / x;
}
"""


# BF pixels between exact evaluations of exp(i u.k) and exp(i y); in between they advance by one detector-row step.
ANCHOR = 8


@lru_cache(maxsize=1)
def _angle_sum_kernel():
    """The joint objective's partial sums per band q, BF chunk and tile of trials, with angle-sum phases."""
    from quantem.gpu.ssb.mps.hardware import require_mlx

    return require_mlx().fast.metal_kernel(
        name="ssb_thick_fit_angle_sums",
        input_names=["G", "band_flat", "band_qx", "band_qy", "kx", "ky", "trial", "scalars", "probe_geometry",
                     "neighbour", "step_k", "batch_count"],
        output_names=["numer_real", "numer_imag", "denom"],
        # Newton changes the number of active candidates at every step. This
        # is dispatch metadata, not a shader specialization: retain the fixed
        # four-candidate tile without compiling a shader for every batch size.
        source="const uint TOTAL = batch_count[0];\n" + _ANGLE_SUM_SOURCE,
        header=_THICK_FIT_BATCH_HEADER + _SINC_HEADER,
    )


def _row_steps(prepared):
    """1 for a stored BF pixel whose predecessor is its left neighbour in the detector row, and the row step dk."""
    mx = prepared.mx
    cached = prepared.row_steps
    if cached is not None and cached[0] is prepared.kx:
        return cached[1], cached[2]

    kx = np.asarray(prepared.kx_np, dtype=np.float64)
    ky = np.asarray(prepared.ky_np, dtype=np.float64)
    neighbour = np.zeros(kx.size, dtype=np.float32)
    step = np.zeros(2)
    if kx.size > 1:
        difference = np.stack((np.diff(kx), np.diff(ky)), axis=1)
        length = np.hypot(difference[:, 0], difference[:, 1])
        guess = np.median(difference[length <= 1.5 * length.min()], axis=0)
        close = np.hypot(*(difference - guess).T) <= 1e-3 * np.hypot(*guess)
        if close.any():
            step = difference[close].mean(axis=0)
            neighbour[1:] = close
    values = (mx.array(neighbour), mx.array(step.astype(np.float32)))
    prepared.row_steps = (prepared.kx, *values)
    return values


@lru_cache(maxsize=1)
def _probe_geometry_kernel():
    """The same probe geometry as the trial kernel, without unused trial phases."""
    from quantem.gpu.ssb.mps.hardware import require_mlx

    return require_mlx().fast.metal_kernel(
        name="ssb_joint_probe_geometry",
        input_names=["kx", "ky", "scalars"],
        output_names=["geometry"],
        header=_THICK_FIT_BATCH_HEADER,
        source="""
        uint k = thread_position_in_grid.x;
        if (k >= NUM_BF) return;
        float4 gk = tf_geometry(kx[k], ky[k], scalars[0], scalars[1], scalars[2], scalars[3]);
        for (uint i = 0; i < 4; ++i) geometry[k * 4 + i] = gk[i];
        """,
    )


def _probe_geometry(prepared, scalars):
    """Reuse detector geometry across trials and repeats at the same rotation."""
    if prepared.fit_probe_geometry is None:
        count = int(prepared.kx.size)
        prepared.fit_probe_geometry = _probe_geometry_kernel()(
            inputs=[prepared.kx, prepared.ky, scalars],
            template=[("NUM_BF", count)],
            grid=(count, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(count, 4)], output_dtypes=[prepared.mx.float32],
        )[0]
    return prepared.fit_probe_geometry


def band_sums(prepared, rows, flat, band_qx, band_qy, scalars):
    """Per band frequency N_q (real, imaginary) and D_q of the encoded trial ``rows``, summed over the BF chunks.

    Shape (3, rows, band frequencies), float32: the joint objective's kernel before its band reduction.
    """
    mx = prepared.mx
    g = prepared.g_qk
    n_band = int(flat.size)
    storage_bf = int(g.shape[0])
    plane = int(g.shape[1]) * int(g.shape[2])
    chunks = (storage_bf + _K_CHUNK - 1) // _K_CHUNK
    batch = len(rows)
    tile = min(_FIT_DISPATCH_BATCH, batch)
    if ANGLE_SUMS:
        probe_geometry = _probe_geometry(prepared, scalars)
        kernel = _angle_sum_kernel()
        extra = [*_row_steps(prepared), mx.array([batch], dtype=mx.uint32)]
    else:
        probe_geometry, probe_chi = _trial_probe_kernel()(
            inputs=[prepared.kx, prepared.ky, scalars, rows.reshape(-1)],
            template=[("NUM_BF", storage_bf), ("B", batch)],
            grid=(storage_bf, 1, 1), threadgroup=(256, 1, 1),
            output_shapes=[(storage_bf, 4), (storage_bf, batch)],
            output_dtypes=[mx.float32] * 2,
        )
        kernel, extra = _thick_fit_batch_kernel(), [probe_chi]
    parts = kernel(
        inputs=[
            g,
            flat,
            band_qx,
            band_qy,
            prepared.kx,
            prepared.ky,
            rows.reshape(-1),
            scalars,
            probe_geometry,
            *extra,
        ],
        template=[
            *([("ANCHOR", ANCHOR)] if ANGLE_SUMS else []),
            ("B", tile),
            *([] if ANGLE_SUMS else [("TOTAL", batch)]),
            ("N_BAND", n_band),
            ("NUM_BF", storage_bf),
            ("PLANE", plane),
            ("K_CHUNK", _K_CHUNK),
        ],
        grid=(n_band, chunks, (batch + tile - 1) // tile),
        threadgroup=(64, 1, 1),
        output_shapes=[(chunks, batch, n_band)] * 3,
        output_dtypes=[mx.float32] * 3,
    )
    sums = mx.stack([mx.sum(part, axis=0) for part in parts])
    return sums


def evaluate(prepared, trial, band_inv_A=(0.2, 0.9)):
    """Evaluate resident encoded trials without a score or parameter readback."""
    device = _Device("mps")
    mx = device.xp
    flat, band_qx, band_qy, weight = _band(prepared, band_inv_A)
    weight = mx.array(weight, dtype=mx.float32)
    n_band = int(flat.size)
    g = prepared.g_qk
    storage_bf = int(g.shape[0])
    chunks = (storage_bf + _K_CHUNK - 1) // _K_CHUNK
    scalars = mx.array(
        [
            prepared.wavelength,
            prepared.semiangle_rad,
            prepared.ang_y_rad,
            prepared.ang_x_rad,
            prepared.factor,
        ],
        dtype=mx.float32,
    )
    # Schedule independent tiles together, including the eight search trials.
    # Keep each thread's four-candidate arithmetic and BF accumulation order;
    # only the grid's independent tile dimension grows. Target the partial-sum
    # budget using the actual band size, with at least one four-candidate tile.
    per_candidate = (chunks + 1) * n_band * 3 * 4
    execution = max(
        _FIT_DISPATCH_BATCH,
        (_FIT_PENDING_BYTES // per_candidate // _FIT_DISPATCH_BATCH)
        * _FIT_DISPATCH_BATCH,
    )
    outputs = []
    for start in range(0, len(trial), execution):
        checkpoint()
        rows = trial[start : start + execution]
        batch = len(rows)
        sums = band_sums(prepared, rows, flat, band_qx, band_qy, scalars)
        constants = f"\n#define BATCH {batch}\n#define BAND {n_band}\n"
        partial = device.run(
            "ssb_joint_reduce_band",
            {"sums": sums, "weight": weight},
            [(batch, 256, 3)],
            REDUCE,
            count=batch * 256,
            constants=constants,
        )[0]
        score = device.run(
            "ssb_joint_reduce_score",
            {"partial": partial},
            [(batch, 3)],
            "if(index>=BATCH)return;Real total(0);for(int i=0;i<256;i++)total=total+load_real(partial,int(index)*256+i);store_real(out0,int(index),total);",
            count=batch,
            constants=constants,
        )[0]
        outputs.append(score)
    return mx.concatenate(outputs, axis=0)
