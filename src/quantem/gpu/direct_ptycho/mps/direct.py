"""Prepared weak-object amplitude/phase systems on Apple GPUs, as ``direct_ptycho.cuda.direct`` on CUDA.

For each measured scan frequency, all supported aliases are fitted together, with the tilted, finite-depth probe
model used by SSB fitting. The layout (dose, detector weights, damping scale, supported aliases of every non-DC
frequency of the half plane, prior) depends on the data and sampling only and is kept on the prepared MPS state. The
normal equations depend on the optics too: Metal evaluates the transfer and accumulates the augmented Gram
V^H W V, V = (T_1 .. T_dim, G), from the retained G(q, k), in float32 per BF pixel and in float-float across BF
batches. A strength change only solves again: float64 Cholesky per frequency on the CPU, then one complex64 inverse
FFT of the packed eta + i phi spectrum. Regularization acts on the solution, never on a rendered image.
"""

import math
import os
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from numba import njit

from quantem.gpu.direct_ptycho import chebyshev
from quantem.gpu.direct_ptycho.higher_order import CHI_SOURCE, pack_coefficients
from quantem.gpu.direct_ptycho.mps.hardware import require_mlx
from quantem.gpu.direct_ptycho.units import higher_order_arrays

# BF pixels per float32 running sum: the CUDA lanes sum about 300 pixels before they meet in float64
_BATCH = 512
# augmented entries a thread of the per-frequency kernel holds; larger systems split into parts
_MAX_SHARE = 96
# frequencies per CPU solve task
_SOLVE_CHUNK = 2048

_HEADER = (
    "#define SSB_DEVICE inline\n#define SSB_COEFFICIENTS threadgroup const float*\n"
    + CHI_SOURCE
    + r"""
// CUDA's thick_probe_geometry: (alpha^2, -, -, soft aperture); on square detector pixels the edge width is the
// constant angular sampling.
inline float4 direct_geometry(float dr, float dc, float wl, float semiangle, float ang_r, float ang_c) {
    const float r2 = dr * dr + dc * dc;
    const float alpha2 = r2 * wl * wl;
    float edge;
    if (ang_r != ang_c) {
        const float r = metal::precise::sqrt(r2);
        const float alpha = r * wl;
        const float inv_r = r > 1e-15f ? 1.0f / r : 0.0f;
        const float denom = metal::precise::sqrt(dr * ang_r * dr * ang_r + dc * ang_c * dc * ang_c) * inv_r;
        edge = denom > 1e-15f ? (semiangle - alpha) / denom + 0.5f : 1.0f;
    } else {
        edge = r2 > 1e-30f ? metal::fma(semiangle - metal::precise::sqrt(r2) * wl, 1.0f / ang_c, 0.5f) : 1.0f;
    }
    return float4(alpha2, 0.0f, 0.0f, metal::clamp(edge, 0.0f, 1.0f));
}

// x reduced to [-pi, pi] (2 pi split in two floats, fused), so the sine and cosine of chi differences of tens of
// radians stay accurate.
inline float direct_reduce(float x) {
    const float turns = metal::rint(x * 0.159154943f);
    return metal::fma(-turns, -1.74845553e-7f, metal::fma(-turns, 6.28318548f, x));
}

inline float direct_sinc(float x) {
    if (metal::abs(x) < 1e-6f) return 1.0f;
    return metal::precise::sin(direct_reduce(x)) / x;
}

// The optics of one preparation, shared by a threadgroup: wavelength, semiangle, angular sampling (row, column),
// pi / wavelength, depth, tilt (row, column), the squared radius beyond which the soft aperture is exactly zero, and
// the 14 packed chi coefficients (37 floats, no padding).
struct DirectOptics {
    float wl, semiangle, ang_r, ang_c, factor, depth, tilt_r, tilt_c, outer2;
    float coefficients[28];
};

// Threads 0 .. 36 of the threadgroup each load one value; every thread must call this (it ends on a barrier), and
// every threadgroup must be full (grids are whole threadgroups).
inline void direct_load(threadgroup DirectOptics& o, const device float* params, const device float* coefficients,
                        uint tid) {
    threadgroup float* values = (threadgroup float*)&o;
    if (tid < 9u) values[tid] = params[tid];
    else if (tid < 37u) values[tid] = coefficients[tid - 9u];
    threadgroup_barrier(mem_flags::mem_threadgroup);
}

inline float2 direct_side(float dr, float dc, float sign, float shift, float4 gk, float ck, threadgroup const DirectOptics& o) {
    const float r2 = dr * dr + dc * dc;
    if (r2 >= o.outer2) return float2(0.0f);
    const float4 g = direct_geometry(dr, dc, o.wl, o.semiangle, o.ang_r, o.ang_c);
    if (g.w * gk.y == 0.0f) return float2(0.0f);
    const float chi = ssb_chi_full(dr, dc, o.wl, o.coefficients);
    const float w = o.depth != 0.0f ? direct_sinc(0.5f * o.depth * (sign * o.factor * (g.x - gk.x) - shift)) : 1.0f;
    const float amplitude = g.w * gk.y * w;
    float c;
    const float s = metal::precise::sincos(direct_reduce(chi - ck), c);
    return float2(amplitude * c, sign * (amplitude * s));
}

// w conj(va) vb onto one running sum, in CUDA's order of operations.
inline float2 direct_product(float2 sum, float2 va, float2 vb, float w) {
    const float ur = w * va.x, ui = -w * va.y;
    return float2(metal::fma(ur, vb.x, metal::fma(-ui, vb.y, sum.x)), metal::fma(ur, vb.y, metal::fma(ui, vb.x, sum.y)));
}

// Sum e of a tiled-kernel thread on alias a (complex systems, C aliases): its entry of the row-major augmented upper
// triangle, and whether the thread owns it. The order is that of the thread's sums: (a, a), (a, P + a), (P + a, P + a),
// then for d = 1 .. C/2 with b = (a + d) mod C the pair entries (eta eta, eta_a phi_b, eta_b phi_a, phi phi), then the
// right sides (a, D), (P + a, D). A pair at d = C/2 of an even C is reached from both aliases; the lower one owns it.
template <int C>
inline uint direct_tiled_entry(uint a, int e, thread bool& mine) {
    constexpr int P = C, D = 2 * C, WIDTH = D + 1, LAST = 3 + 4 * (C / 2);
    #define DIRECT_INDEX(r, c) ((r) * WIDTH - (r) * ((r) - 1) / 2 + ((c) - (r)))
    mine = true;
    if (e == 0) return DIRECT_INDEX(a, a);
    if (e == 1) return DIRECT_INDEX(a, P + a);
    if (e == 2) return DIRECT_INDEX(P + a, P + a);
    if (e == LAST) return DIRECT_INDEX(a, D);
    if (e > LAST) return DIRECT_INDEX(P + a, D);
    const int d = 1 + (e - 3) / 4, kind = (e - 3) % 4;
    const uint b = (a + d) % C, lo = metal::min(a, b), hi = metal::max(a, b);
    mine = 2 * d < C || a < uint(C / 2);
    if (kind == 0) return DIRECT_INDEX(lo, hi);
    if (kind == 1) return DIRECT_INDEX(a, P + b);
    if (kind == 2) return DIRECT_INDEX(b, P + a);
    return DIRECT_INDEX(P + lo, P + hi);
    #undef DIRECT_INDEX
}

// a float32 batch sum onto a float-float (hi re, hi im, lo re, lo im) accumulator (two-sum per component)
inline float4 direct_add(float4 acc, float2 value) {
    const float re = acc.x + value.x, rv = re - acc.x;
    acc.z += (acc.x - (re - rv)) + (value.x - rv);
    const float im = acc.y + value.y, iv = im - acc.y;
    acc.w += (acc.y - (im - iv)) + (value.y - iv);
    acc.x = re;
    acc.y = im;
    return acc;
}
"""
)


@lru_cache(maxsize=1)
def _pixel_kernel():
    """The k-only terms of every stored BF pixel: (kr, kc, chi(k), weight) and (alpha^2(k), aperture(k), 0, 0)."""
    return require_mlx().fast.metal_kernel(
        name="direct_pixel_terms",
        input_names=["k", "weights", "coefficients", "params"],
        output_names=["pixel"],
        header=_HEADER,
        compile_options={"math_mode": "safe"},
        source="""
        threadgroup DirectOptics o;
        direct_load(o, params, coefficients, thread_index_in_threadgroup);
        const uint j = thread_position_in_grid.x;
        if (j >= NBF) return;
        const float kr = k[2 * j], kc = k[2 * j + 1];
        const float4 g = direct_geometry(kr, kc, o.wl, o.semiangle, o.ang_r, o.ang_c);
        device float4* out = (device float4*)pixel;
        out[2 * j] = float4(kr, kc, ssb_chi_full(kr, kc, o.wl, o.coefficients), weights[j]);
        out[2 * j + 1] = float4(g.x, g.w, 0.0f, 0.0f);
        """,
    )


@lru_cache(maxsize=1)
def _normal_kernel():
    """Entries [FIRST, LAST) of the augmented upper triangle (row-major, a <= b) of every frequency, one per thread.

    The thread walks BF pixels ``dims[3] .. dims[4]`` of the stored G(q, k), evaluates the transfer of each of its C
    aliases (log-amplitude columns first for the complex model, then phase, then G) and adds w conj(v_a) v_b to
    float32 running sums, which meet the float-float accumulator after every ``BATCH`` pixels. An alias whose two
    sidebands are closed has zero columns: its products are exact zeros and are skipped.
    """
    return require_mlx().fast.metal_kernel(
        name="direct_normal_equations",
        input_names=["g", "frequency", "alias", "pixel", "coefficients", "params", "dims"],
        output_names=["sums"],
        header=_HEADER,
        compile_options={"math_mode": "safe"},
        source="""
        constexpr int DIM = COMPLEX ? 2 * C : C;
        constexpr int WIDTH = DIM + 1;
        constexpr int SHARE = LAST - FIRST;
        constexpr int CA = C > 0 ? C : 1;
        constexpr int P = DIM - C;  // first phase column
        threadgroup DirectOptics o;
        direct_load(o, params, coefficients, thread_index_in_threadgroup);
        const uint f = thread_position_in_grid.x;
        if (f >= dims[0]) return;
        const uint plane = dims[1], gcols = dims[2], first_pixel = dims[3], last_pixel = dims[4];
        const device float2* gv = (const device float2*)g;
        const device float4* pix = (const device float4*)pixel;
        const device float2* al = (const device float2*)alias;
        device float4* out = (device float4*)sums + (size_t)f * SHARE;
        const uint column = frequency[2 * f] * gcols + frequency[2 * f + 1];
        float qr[CA], qc[CA], shift[CA];
        for (int a = 0; a < C; ++a) {
            const float2 q = al[f * C + a];
            qr[a] = q.x;
            qc[a] = q.y;
            shift[a] = 6.283185307f * (q.x * o.tilt_r + q.y * o.tilt_c);
        }
        for (int i = 0; i < SHARE; ++i) out[i] = float4(0.0f);
        // entry (r, c), r <= c, of the row-major upper triangle; products outside [FIRST, LAST) belong to another part
        #define DIRECT_ENTRY(r, c) ((r) * WIDTH - (r) * ((r) - 1) / 2 + ((c) - (r)))
        #define DIRECT_ADD(i, va, vb) { const int e = (i); if (e >= FIRST && e < LAST) sum[e - FIRST] = direct_product(sum[e - FIRST], va, vb, w); }
        for (uint base = first_pixel; base < last_pixel; base += BATCH) {
            const uint end = metal::min(base + uint(BATCH), last_pixel);
            float2 sum[SHARE];
            for (int i = 0; i < SHARE; ++i) sum[i] = float2(0.0f);
            for (uint j = base; j < end; ++j) {
                const float4 pk = pix[2 * j], gk = pix[2 * j + 1];
                const float w = pk.w;
                float2 v[WIDTH];
                v[DIM] = gv[(size_t)j * plane + column];
                bool open[CA];
                for (int a = 0; a < C; ++a) {
                    const float2 t1 = direct_side(pk.x - qr[a], pk.y - qc[a], -1.0f, shift[a], gk, pk.z, o);
                    const float2 t2 = direct_side(pk.x + qr[a], pk.y + qc[a], 1.0f, shift[a], gk, pk.z, o);
                    // i (t1 - t2): the phase column; t1 + t2: the log-amplitude column
                    v[P + a] = float2(-(t1.y - t2.y), t1.x - t2.x);
                    if (COMPLEX) v[a] = t1 + t2;
                    open[a] = t1.x != 0.0f || t1.y != 0.0f || t2.x != 0.0f || t2.y != 0.0f;
                }
                // an alias with both sidebands closed has zero columns: its products are exact zeros, skipped
                for (int a = 0; a < C; ++a) {
                    if (!open[a]) continue;
                    for (int b = a; b < C; ++b) {
                        if (!open[b]) continue;
                        DIRECT_ADD(DIRECT_ENTRY(P + a, P + b), v[P + a], v[P + b]);
                        if (COMPLEX) {
                            DIRECT_ADD(DIRECT_ENTRY(a, b), v[a], v[b]);
                            DIRECT_ADD(DIRECT_ENTRY(a, P + b), v[a], v[P + b]);
                            if (b != a) DIRECT_ADD(DIRECT_ENTRY(b, P + a), v[b], v[P + a]);
                        }
                    }
                    DIRECT_ADD(DIRECT_ENTRY(P + a, DIM), v[P + a], v[DIM]);
                    if (COMPLEX) DIRECT_ADD(DIRECT_ENTRY(a, DIM), v[a], v[DIM]);
                }
                DIRECT_ADD(DIRECT_ENTRY(DIM, DIM), v[DIM], v[DIM]);
            }
            for (int i = 0; i < SHARE; ++i) out[i] = direct_add(out[i], sum[i]);
        }
        #undef DIRECT_ADD
        #undef DIRECT_ENTRY
        """,
    )


@lru_cache(maxsize=1)
def _tiled_kernel():
    """The complex normal equations with one thread per (frequency, alias), for two or more aliases.

    A threadgroup holds BLOCKS x 32 frequencies of one alias group, SIMD group a evaluating alias a of each. For a
    tile of BF pixels every thread writes its alias's two columns to threadgroup memory; after a barrier it adds the
    products it owns: its alias against itself and against aliases a + 1 .. a + C/2 (cyclically, each pair owned
    once), its right sides and, on alias 0, the signal power. Each product has the per-frequency kernel's arithmetic
    and BF pixel order, so both give the same sums; a thread holds about 2 C sums instead of the whole triangle.
    """
    return require_mlx().fast.metal_kernel(
        name="direct_normal_tiled",
        input_names=["g", "frequency", "alias", "pixel", "coefficients", "params", "dims"],
        output_names=["sums"],
        header=_HEADER,
        compile_options={"math_mode": "safe"},
        source="""
        constexpr int P = C, D = 2 * C, WIDTH = D + 1;
        constexpr int OUTPUTS = WIDTH * (WIDTH + 1) / 2;
        constexpr int PARTNERS = C / 2;
        constexpr int OWNED = 3 + 4 * PARTNERS + 2;
        threadgroup float2 columns[BLOCKS * TILE * C * 2 * 32];
        threadgroup float2 spectra[BLOCKS * TILE * 32];
        const uint tid = thread_index_in_threadgroup;
        const uint group = threadgroup_position_in_grid.x;
        const uint simd = tid / 32u, lane = tid % 32u;
        const uint a = simd % C, block = simd / C;
        const uint nf = dims[0], plane = dims[1], gcols = dims[2], first_pixel = dims[3], last_pixel = dims[4];
        const uint f = (group * BLOCKS + block) * 32u + lane;
        threadgroup float2* cols = columns + block * TILE * C * 2 * 32;
        threadgroup float2* spec = spectra + block * TILE * 32;
        const bool valid = f < nf;
        const uint fc = valid ? f : nf - 1u;
        const uint column = frequency[2 * fc] * gcols + frequency[2 * fc + 1];
        threadgroup DirectOptics o;
        direct_load(o, params, coefficients, tid);
        const float2 q = ((const device float2*)alias)[fc * C + a];
        const float shift = 6.283185307f * (q.x * o.tilt_r + q.y * o.tilt_c);
        const device float2* gv = (const device float2*)g;
        const device float4* pix = (const device float4*)pixel;
        device float4* out = (device float4*)sums + (size_t)fc * OUTPUTS;
        const uint power_at = OUTPUTS - 1;  // (D, D)
        // clamped threads past the last frequency read but never write (their frequency is another thread's)
        if (valid) {
            for (int e = 0; e < OWNED; ++e) {
                bool mine;
                const uint at = direct_tiled_entry<C>(a, e, mine);
                if (mine) out[at] = float4(0.0f);
            }
            if (a == 0u) out[power_at] = float4(0.0f);
        }
        for (uint batch = first_pixel; batch < last_pixel; batch += BATCH) {
            const uint end = metal::min(batch + uint(BATCH), last_pixel);
            float2 sum[OWNED];
            for (int e = 0; e < OWNED; ++e) sum[e] = float2(0.0f);
            float2 power = float2(0.0f);
            for (uint base = batch; base < end; base += TILE) {
                const uint tile = metal::min(uint(TILE), end - base);
                for (uint t = 0; t < tile; ++t) {
                    const uint j = base + t;
                    const float4 pk = pix[2 * j], gk = pix[2 * j + 1];
                    const float2 t1 = direct_side(pk.x - q.x, pk.y - q.y, -1.0f, shift, gk, pk.z, o);
                    const float2 t2 = direct_side(pk.x + q.x, pk.y + q.y, 1.0f, shift, gk, pk.z, o);
                    const uint own = (t * C + a) * 2 * 32 + lane;
                    cols[own + 32] = float2(-(t1.y - t2.y), t1.x - t2.x);
                    cols[own] = t1 + t2;
                }
                // the tile's spectra values, loaded once per threadgroup (SIMD group a takes pixels a, a + C, ...)
                for (uint t = a; t < tile; t += C) spec[t * 32 + lane] = gv[(size_t)(base + t) * plane + column];
                threadgroup_barrier(mem_flags::mem_threadgroup);
                for (uint t = 0; t < tile; ++t) {
                    const float w = pix[2 * (base + t)].w;
                    const uint own = (t * C + a) * 2 * 32 + lane;
                    const float2 phi = cols[own + 32], eta = cols[own];
                    const bool closed = phi.x == 0.0f && phi.y == 0.0f && eta.x == 0.0f && eta.y == 0.0f;
                    if (closed && a != 0u) continue;
                    const float2 gval = spec[t * 32 + lane];
                    if (a == 0u) power = direct_product(power, gval, gval, w);
                    if (closed) continue;
                    int i = 0;
                    sum[i] = direct_product(sum[i], eta, eta, w); ++i;
                    sum[i] = direct_product(sum[i], eta, phi, w); ++i;
                    sum[i] = direct_product(sum[i], phi, phi, w); ++i;
                    for (int d = 1; d <= PARTNERS; ++d) {
                        const uint b = (a + d) % C;
                        const uint other = (t * C + b) * 2 * 32 + lane;
                        const float2 phib = cols[other + 32], etab = cols[other];
                        const float use = 2 * d < C || a < uint(C / 2) ? w : 0.0f;
                        // products in the upper triangle's orientation: the lower column index is conjugated
                        sum[i] = a < b ? direct_product(sum[i], eta, etab, use) : direct_product(sum[i], etab, eta, use); ++i;
                        sum[i] = direct_product(sum[i], eta, phib, use); ++i;
                        sum[i] = direct_product(sum[i], etab, phi, use); ++i;
                        sum[i] = a < b ? direct_product(sum[i], phi, phib, use) : direct_product(sum[i], phib, phi, use); ++i;
                    }
                    sum[i] = direct_product(sum[i], eta, gval, w); ++i;
                    sum[i] = direct_product(sum[i], phi, gval, w); ++i;
                }
                threadgroup_barrier(mem_flags::mem_threadgroup);
            }
            if (valid) {
                for (int e = 0; e < OWNED; ++e) {
                    bool mine;
                    const uint at = direct_tiled_entry<C>(a, e, mine);
                    if (mine) out[at] = direct_add(out[at], sum[e]);
                }
                if (a == 0u) out[power_at] = direct_add(out[power_at], power);
            }
        }
        """,
    )


def _tiled(count: int, complex_object: bool) -> bool:
    """Whether the per-(frequency, alias) kernel serves this group: complex systems of two or more aliases."""
    return complex_object and count >= 2


def _blocks(count: int) -> int:
    """Blocks of 32 frequencies per threadgroup of the tiled kernel: at least 128 threads."""
    return max(1, 4 // count)


def _tile(count: int) -> int:
    """BF pixels per threadgroup tile of the tiled kernel: the tile's columns take about 16 KB."""
    return max(1, min(32, 16384 // (256 * count * 2 * _blocks(count))))


def _parts(count: int, complex_object: bool) -> tuple[int, list[tuple[int, int]]]:
    """Augmented entries of a system with ``count`` aliases, and their split into per-frequency-kernel parts."""
    width = (2 * count if complex_object else count) + 1
    outputs = width * (width + 1) // 2
    if _tiled(count, complex_object):
        return outputs, [(0, outputs)]
    parts = -(-outputs // _MAX_SHARE)
    share = -(-outputs // parts)
    return outputs, [(first, min(first + share, outputs)) for first in range(0, outputs, share)]


@dataclass
class _AliasGroup:
    """The frequencies with one count of supported aliases, and their optics-free terms."""

    count: int
    # (source row, source column) per frequency, uint32
    frequency: np.ndarray
    # (q row, q column) per frequency and alias, 1/A, float32
    alias: np.ndarray
    # output index row * columns + column per frequency and alias
    positions: np.ndarray
    # (1 + |q|^2 / q0^2)^2 per frequency and alias (float32-rounded, as CUDA), float64
    prior: np.ndarray
    # frequency and alias on the device (``_layout``)
    device: tuple | None = None


@dataclass
class _Layout:
    """Everything of a preparation that the data, the sampling and the rotation fix, not the optics."""

    shape: tuple[int, int]
    dose: float
    # detector weight of every stored BF pixel
    weights: np.ndarray
    scale: float
    groups: list[_AliasGroup]


def _alias_groups(scan_shape: tuple[int, int], scan_sampling: tuple[float, float], factor: int, cutoff: float,
                  smoothness_inv_A: float) -> list[_AliasGroup]:
    """The solved frequencies of the source half plane, grouped by their count of supported output aliases.

    Kept: every column v > 0 and rows 1 .. (N - 1) // 2 of column 0, without the DC and, on even grids, the Nyquist
    row and column. The aliases of (u, v) are output indices (u + a N_r, v + b N_c), a, b < factor, in order
    a * factor + b, with frequencies float32(signed index) * float32(step); one is supported when
    |q|^2 < cutoff^2 in float32, as CUDA decides. Frequencies with the same supported aliases sit together, so a
    SIMD group's neighbouring threads take the same aperture branches.
    """
    nrow, ncol = scan_shape
    out_row, out_col = factor * nrow, factor * ncol
    rows, cols = np.indices((nrow, ncol // 2 + 1))
    keep = (cols > 0) | ((rows > 0) & (rows <= (nrow - 1) // 2))
    if nrow % 2 == 0:
        keep &= rows != nrow // 2
    if ncol % 2 == 0:
        keep &= cols != ncol // 2
    rows, cols = rows[keep], cols[keep]
    alias_row, alias_col = (index.ravel() for index in np.indices((factor, factor)))
    output_rows = rows[:, None] + alias_row[None, :] * nrow
    output_cols = cols[:, None] + alias_col[None, :] * ncol
    step_row = np.fft.fftfreq(nrow, scan_sampling[0]).astype(np.float32)[1]
    step_col = np.fft.fftfreq(ncol, scan_sampling[1]).astype(np.float32)[1]
    qr = np.where(output_rows <= (out_row - 1) // 2, output_rows, output_rows - out_row).astype(np.float32) * step_row
    qc = np.where(output_cols <= (out_col - 1) // 2, output_cols, output_cols - out_col).astype(np.float32) * step_col
    active = qr * qr + qc * qc < np.float32(cutoff**2)
    counts = active.sum(axis=1)
    pattern = (active.astype(np.uint64) << np.arange(factor * factor, dtype=np.uint64)[None, :]).sum(axis=1)
    groups = []
    for count in np.unique(counts).tolist():
        selected = np.flatnonzero(counts == count)
        selected = selected[np.argsort(pattern[selected], kind="stable")]
        frequency = np.ascontiguousarray(np.stack((rows[selected], cols[selected]), axis=1).astype(np.uint32))
        if count == 0:
            # frequencies without a supported alias add only to the signal power
            groups.append(_AliasGroup(0, frequency, np.zeros((selected.size, 0, 2), np.float32),
                                      np.zeros((selected.size, 0), np.int64), np.zeros((selected.size, 0))))
            continue
        group_index, alias_index = np.nonzero(active[selected])
        group_index = selected[group_index]
        q_row = qr[group_index, alias_index].reshape(-1, count)
        q_col = qc[group_index, alias_index].reshape(-1, count)
        if smoothness_inv_A:
            frequency2 = q_row**2 + q_col**2
            prior = ((np.float32(1) + frequency2 / np.float32(smoothness_inv_A**2)) ** 2).astype(np.float64)
        else:
            prior = np.ones((selected.size, count))
        positions = (output_rows[group_index, alias_index].astype(np.int64) * out_col
                     + output_cols[group_index, alias_index]).reshape(-1, count)
        groups.append(_AliasGroup(count, frequency, np.ascontiguousarray(np.stack((q_row, q_col), axis=-1)),
                                  positions, prior))
    return groups


def _detector_weights(scan_sums: np.ndarray, positions: int, detector_weighting: str) -> np.ndarray:
    """float32 weight of every selected BF pixel: 1, or 1 / max(mean count, 5 % of the median mean count) / mean."""
    if detector_weighting == "uniform":
        return np.ones(scan_sums.size, np.float32)
    mean = np.maximum(scan_sums.astype(np.float32) / np.float32(positions), np.float32(0))
    floor = np.float32(np.median(mean)) * np.float32(0.05)
    if float(floor) <= 0:
        raise ValueError("Variance weighting needs positive bright-field counts; use uniform weighting.")
    weights = np.float32(1) / np.maximum(mean, floor)
    return (weights / np.float32(weights.mean(dtype=np.float64))).astype(np.float32)


def _scan_sums(backend, indices: np.ndarray) -> np.ndarray:
    """Scan sums (float64) of selected BF pixels that G(q, k) does not store (closed aperture), from the source."""
    selection = backend._selection
    rows, cols = selection.rows[indices], selection.cols[indices]
    positions = int(np.prod(backend._scan_shape))
    values = np.empty((int(indices.size), positions), np.float32)
    backend._frames.columns_float32_into(rows, cols, out=values)
    return values.sum(axis=1, dtype=np.float64)


def _layout(backend, factor: int, smoothness_inv_A: float, detector_weighting: str) -> _Layout:
    """The optics-free part of a preparation, kept on the prepared MPS state per factor, prior, weighting and rotation.

    The dose and the inverse-mean weights use every selected BF pixel, as CUDA does, including pixels G(q, k) does not
    store because the aperture is closed there (their transfer is zero).
    """
    prepared = backend._prepared
    key = (factor, smoothness_inv_A, detector_weighting, backend._scan_detector_rotation_deg)
    cached = prepared.direct_layout
    if cached is not None and cached[0] == key:
        return cached[1]
    mx = prepared.mx
    nrow, ncol = prepared.scan_shape
    if min(nrow, ncol) < 3:
        raise ValueError("Direct inversion needs at least 3 scan positions per side; select a larger region.")
    out_row, out_col = factor * nrow, factor * ncol
    total = int(backend._selection.size)
    stored = prepared.bf_storage_indices_np
    stored = np.arange(total) if stored is None else np.asarray(stored, np.int64)
    # the DC of G(q, k) is each BF pixel's scan sum, float32 as CUDA's
    dc = np.zeros(total, np.float64)
    dc[stored] = np.asarray(prepared.g_qk[:, 0, 0]).real.astype(np.float32)
    missing = np.setdiff1d(np.arange(total), stored)
    if missing.size:
        dc[missing] = _scan_sums(backend, missing)
    aperture2 = np.asarray(prepared.aperture_k_1d, np.float32) ** 2
    dose = float(dc.sum() / (nrow * ncol * aperture2.sum(dtype=np.float64)))
    if not (math.isfinite(dose) and dose > 0):
        raise ValueError("No positive incident intensity estimate; check counts and bright-field calibration.")
    weights = np.ascontiguousarray(_detector_weights(dc, nrow * ncol, detector_weighting)[stored])
    scale = float(np.sum(weights * aperture2 * aperture2, dtype=np.float64))
    cutoff = (2 * prepared.semiangle_rad + max(prepared.ang_y_rad, prepared.ang_x_rad)) / prepared.wavelength
    groups = _alias_groups((nrow, ncol), backend._scan_sampling, factor, cutoff, smoothness_inv_A)
    for group in groups:
        alias = group.alias.reshape(-1, 2) if group.count else np.zeros((1, 2), np.float32)
        group.device = mx.array(group.frequency), mx.array(np.ascontiguousarray(alias))
    layout = _Layout((out_row, out_col), dose, weights, scale, groups)
    prepared.direct_layout = key, layout
    return layout


def _bright_field_frequencies(backend) -> np.ndarray:
    """(k row, k column) in 1/A of every stored BF pixel after the scan rotation: the prepared SSB state's k, formed
    in float64 and rounded once (``prepared.bright_field_frequencies``)."""
    prepared = backend._prepared
    return np.ascontiguousarray(np.stack((prepared.kx_np, prepared.ky_np), axis=1), np.float32)


def _coefficients(aberrations: dict[str, float]) -> np.ndarray:
    """The 14 packed float32 chi coefficients in Angstrom, as the CUDA preparation packs them."""
    higher = higher_order_arrays(aberrations)
    if higher is None:
        magnitudes = np.zeros(14, np.float32)
        angles = np.zeros(14, np.float32)
        magnitudes[:2] = [aberrations.get("C10_nm", 0), aberrations.get("C12_nm", 0)]
        angles[1] = np.deg2rad(aberrations.get("phi12_deg", 0))
    else:
        magnitudes, angles = higher
    return pack_coefficients(magnitudes * 10, angles).ravel()


@dataclass
class _Group:
    count: int
    # float-float augmented upper triangle per frequency: (frequencies, entries, 4) float32
    sums: np.ndarray
    positions: np.ndarray
    prior: np.ndarray


@dataclass
class PreparedDirect:
    """Sufficient statistics for one fixed data, optics and model selection."""

    shape: tuple[int, int]
    factor: int
    groups: list[_Group]
    complex_object: bool
    scale: float
    signal_power: float
    incident_scale: float

    @classmethod
    def prepare(cls, backend, optics, factor: int, *, complex_object: bool, smoothness_inv_A: float,
                detector_weighting: str):
        """Build the normal equations once on the GPU; strengths are applied at evaluation.

        Each scan frequency Q with its supported aliases needs the Gram A = T^H W T, the right side
        b = T^H W g and the signal power sum_k w_k |g_k|^2, where T (BF pixel, alias) is the transfer, W the detector
        weights and g = G / dose the measured spectrum. Everything but T depends on the data alone (`_layout`), so a
        change of optics costs one GPU pass over the BF pixels per alias count.
        """
        mx = require_mlx()
        prepared = backend._prepared
        layout = _layout(backend, factor, smoothness_inv_A, detector_weighting)
        tilt_row, tilt_col = optics["sample_tilt_mrad"]
        wavelength = np.float32(prepared.wavelength)
        radius = (prepared.semiangle_rad + 0.5 * max(prepared.ang_y_rad, prepared.ang_x_rad)) / prepared.wavelength
        params = mx.array(np.array([
            wavelength, prepared.semiangle_rad, prepared.ang_y_rad, prepared.ang_x_rad,
            np.float32(3.141592654) / wavelength, optics["depth_spread_nm"] * 10, tilt_row * 1e-3, tilt_col * 1e-3,
            radius * radius * (1 + 1e-4),
        ], np.float32))
        coefficients = mx.array(_coefficients(optics["aberrations"]))
        nbf = int(layout.weights.size)
        k = mx.array(_bright_field_frequencies(backend))
        pixel = _pixel_kernel()(
            inputs=[k, mx.array(layout.weights), coefficients, params], template=[("NBF", nbf)],
            grid=(-(-nbf // 256) * 256, 1, 1), threadgroup=(256, 1, 1), output_shapes=[(nbf, 8)],
            output_dtypes=[mx.float32],
        )[0]
        g = prepared.g_qk
        plane = int(g.shape[1]) * int(g.shape[2])
        # every group's dispatches in one evaluation; each walks all BF pixels of its frequencies
        pending = [_normal_equations(mx, g, group, pixel, coefficients, params, plane, int(g.shape[2]), nbf,
                                     complex_object) for group in layout.groups]
        mx.eval(*[part for parts in pending for part in parts])
        groups, power = [], 0.0
        for group, parts in zip(layout.groups, pending):
            sums = np.array(parts[0]) if len(parts) == 1 else np.concatenate([np.array(part) for part in parts], axis=1)
            # the last augmented entry is the signal power sum_k w_k |G(Q, k)|^2
            last = sums[:, -1]
            power += float(np.sum(last[:, 0].astype(np.float64) + last[:, 2])) / layout.dose**2
            if group.count:
                groups.append(_Group(group.count, sums, group.positions, group.prior))
        return cls(layout.shape, factor, groups, complex_object, layout.scale, power, layout.dose)

    def solve(self, phase_damping: float, amplitude_damping: float | None, *, compute_loss: bool):
        """Evaluate arbitrary strengths, returning a fresh wave and diagnostics."""
        rows, cols = self.shape
        spectrum = np.zeros((rows * cols, 2), np.float32)
        residual = 0.0
        invalid = False
        complex_object = self.complex_object
        for group in self.groups:
            count = group.count
            dim = 2 * count if complex_object else count
            width = dim + 1
            losses = np.zeros(group.sums.shape[0] if compute_loss else 0)
            failed = _solve_group(
                group.sums, width, count, dim, complex_object, group.prior,
                float(amplitude_damping or 0.0) * self.scale, float(phase_damping) * self.scale,
                self.incident_scale, group.positions, cols, rows, np.float32(self.factor**2), spectrum,
                compute_loss, losses,
            )
            invalid |= bool(failed)
            if compute_loss:
                residual += float(losses.sum())
        if invalid:
            raise ValueError('Direct inversion diverged; increase phase_damping or amplitude_damping, or use method="phase".')
        wave, diagnostics = _wave(spectrum.view(np.complex64).reshape(rows, cols), complex_object,
                                  self.incident_scale)
        if compute_loss:
            if self.signal_power > 0:
                diagnostics["training_relative_sse"] = max(self.signal_power + residual, 0.0) / self.signal_power
            else:
                diagnostics["training_relative_sse"] = None
                diagnostics["prediction_error_status"] = "undefined: no non-DC signal"
        return wave, diagnostics


# ---


class DragWindow:
    """Prepared statistics along one aberration control, for redrawing while it is dragged, as ``direct_ptycho.cuda.direct``.

    Serves: ShowPtycho drags of C10, C12 or phi12 on Apple GPUs at 3x on the full 512 x 512 scan, where an exact frame
    prepares the normal equations again (about 0.3 s). Along one control with everything else fixed, every Gram entry
    and right side is an entire function of the control (`direct_ptycho.chebyshev`), so ``step`` adds one exact preparation at
    a Chebyshev node (one `PreparedDirect.prepare`, 29 of them for a magnitude) and ``solve`` evaluates any value in
    the window from the series in float64 on the CPU, then solves as `PreparedDirect.solve` does. A complex window
    also holds the phase statistics as its lower-right block, so one build serves both regularized methods.
    """

    def __init__(self, backend, optics, control: str, factor: int, *, complex_object: bool, smoothness_inv_A: float,
                 detector_weighting: str):
        prepared = backend._prepared
        self.backend, self.optics, self.control, self.factor = backend, optics, control, factor
        self.complex_object = complex_object
        self.options = dict(complex_object=complex_object, smoothness_inv_A=smoothness_inv_A,
                            detector_weighting=detector_weighting)
        aberrations = optics["aberrations"]
        outer = prepared.semiangle_rad + 0.5 * max(prepared.ang_y_rad, prepared.ang_x_rad)
        self.series = _Series(float(aberrations.get(control, 0.0)),
                              chebyshev.half_width(control, aberrations, prepared.wavelength, outer),
                              chebyshev.order(control))
        # the first node's layout (shape, groups, scale, dose) without its statistics, which the series holds
        self.template = None

    def step(self) -> bool:
        """Add the next node's exact statistics; True once the window is complete."""
        if self.series.ready:
            return True
        aberrations = {**self.optics["aberrations"], self.control: self.series.nodes[self.series.added]}
        prepared = PreparedDirect.prepare(self.backend, {**self.optics, "aberrations": aberrations}, self.factor,
                                          **self.options)
        # float-float (hi re, hi im, lo re, lo im) to float64 (re, im), as `_entry` reads it
        self.series.add(np.concatenate([(group.sums[..., :2].astype(np.float64) + group.sums[..., 2:]).ravel()
                                        for group in prepared.groups]))
        if self.template is None:
            groups = [_Group(group.count, None, group.positions, group.prior) for group in prepared.groups]
            self.template = PreparedDirect(prepared.shape, prepared.factor, groups, prepared.complex_object,
                                           prepared.scale, prepared.signal_power, prepared.incident_scale)
        return self.series.ready

    def solve(self, value: float, phase_damping: float, amplitude_damping: float | None,
              *, complex_object: bool):
        """The wave at control ``value`` (inside the window), as `PreparedDirect.solve` without the loss."""
        if complex_object and not self.complex_object:
            raise ValueError("A phase window cannot serve the complex method; build a complex window.")
        values = self.series.evaluate(value)
        template = self.template
        rows, cols = template.shape
        spectrum = np.zeros((rows * cols, 2), np.float32)
        at, failed = 0, 0
        for group in template.groups:
            count = group.count
            width = (2 * count if self.complex_object else count) + 1
            size = group.positions.shape[0]
            sums = values[at:at + size * width * (width + 1)].reshape(size, width * (width + 1) // 2, 2)
            at += sums.size
            failed += _solve_group(
                sums, width, count, 2 * count if complex_object else count, complex_object, group.prior,
                float(amplitude_damping or 0.0) * template.scale, float(phase_damping) * template.scale,
                template.incident_scale, group.positions, cols, rows, np.float32(self.factor**2), spectrum, False,
                np.zeros(0), count if self.complex_object and not complex_object else 0,
            )
        if failed:
            raise ValueError('Direct inversion diverged; increase regularization or use method="phase".')
        return _wave(spectrum.view(np.complex64).reshape(rows, cols), complex_object, template.incident_scale)


class _Series:
    """`direct_ptycho.chebyshev` coefficients of a float64 vector F(c) on the CPU, as ``direct_ptycho.cuda.chebyshev.Series`` on the GPU.

    The coefficients are (values, N + 1) float64, each value's N + 1 coefficients adjacent, so folding a node in and
    evaluating each stream them once, in ranges of values on the solver threads.
    """

    def __init__(self, center: float, half_width: float, order: int):
        self.center, self.half_width, self.order = center, half_width, order
        self.nodes, weights = chebyshev.nodes_and_weights(center, half_width, order)
        self.weights = np.ascontiguousarray(weights.T)
        self.coefficients = None
        self.added = 0

    @property
    def ready(self) -> bool:
        return self.added == self.order + 1

    def covers(self, value: float) -> bool:
        return abs(value - self.center) <= self.half_width

    def add(self, values: np.ndarray):
        """Fold F at the next node (a flat float64 array) into the coefficients."""
        if self.coefficients is None:
            self.coefficients = np.zeros((values.size, self.order + 1))
        _in_ranges(_fold_node, values.size, self.coefficients, self.weights[self.added], values)
        self.added += 1

    def evaluate(self, value: float) -> np.ndarray:
        """F(value), flat float64."""
        basis = chebyshev.basis(self.center, self.half_width, self.order, value)
        out = np.empty(self.coefficients.shape[0])
        _in_ranges(_evaluate_series, out.size, self.coefficients, basis, out)
        return out


def _in_ranges(function, size: int, *arrays):
    """``function(first, last, *arrays)`` over ranges of ``size`` values on the solver threads."""
    step = max(1 << 16, -(-size // (4 * _threads())))
    tasks = [_solver_pool().submit(function, first, min(first + step, size), *arrays) for first in range(0, size, step)]
    for task in tasks:
        task.result()


@njit(cache=True, nogil=True)
def _fold_node(first, last, coefficients, weight, values):
    """coefficients[m, n] += weight[n] values[m]: one node's values into every coefficient."""
    for m in range(first, last):
        value = values[m]
        for n in range(weight.size):
            coefficients[m, n] += weight[n] * value


@njit(cache=True, nogil=True)
def _evaluate_series(first, last, coefficients, basis, out):
    """out[m] = sum_n basis[n] coefficients[m, n]."""
    for m in range(first, last):
        total = 0.0
        for n in range(basis.size):
            total += basis[n] * coefficients[m, n]
        out[m] = total


def _normal_equations(mx, g, group, pixel, coefficients, params, plane, gcols, nbf, complex_object):
    """One alias group's float-float augmented upper triangles, (frequencies, entries, 4) float32, unevaluated parts."""
    count = group.count
    size = int(group.frequency.shape[0])
    frequency, alias = group.device
    dims = mx.array([size, plane, gcols, 0, nbf], mx.uint32)
    inputs = [g, frequency, alias, pixel, coefficients, params, dims]
    _, parts = _parts(count, complex_object)
    if _tiled(count, complex_object):
        blocks, tile = _blocks(count), _tile(count)
        threads = 32 * count * blocks
        return [_tiled_kernel()(
            inputs=inputs, template=[("C", count), ("BLOCKS", blocks), ("TILE", tile), ("BATCH", _BATCH)],
            grid=(-(-size // (32 * blocks)) * threads, 1, 1), threadgroup=(threads, 1, 1),
            output_shapes=[(size, parts[0][1], 4)], output_dtypes=[mx.float32],
        )[0]]
    return [_normal_kernel()(
        inputs=inputs,
        template=[("C", count), ("COMPLEX", bool(complex_object)), ("FIRST", first), ("LAST", last),
                  ("BATCH", _BATCH)],
        grid=(-(-size // 128) * 128, 1, 1), threadgroup=(128, 1, 1),
        output_shapes=[(size, last - first, 4)], output_dtypes=[mx.float32],
    )[0] for first, last in parts]


@njit(cache=True, inline="always")
def _entry(sums, f, width, a, b):
    """Row a, column b (a <= b) of a stored augmented upper triangle as float64 (re, im).

    ``sums[f, entry]`` is float-float (hi re, hi im, lo re, lo im) as the GPU accumulates it, or (re, im) as a
    `DragWindow` evaluates it.
    """
    index = a * width - a * (a - 1) // 2 + (b - a)
    if sums.shape[2] == 2:
        return np.float64(sums[f, index, 0]), np.float64(sums[f, index, 1])
    return (np.float64(sums[f, index, 0]) + np.float64(sums[f, index, 2]),
            np.float64(sums[f, index, 1]) + np.float64(sums[f, index, 3]))


@lru_cache(maxsize=1)
def _threads() -> int:
    """CPU threads for the solves, the inverse FFT and the wave: ``OMP_NUM_THREADS`` when set, else up to 8."""
    return max(1, int(os.environ.get("OMP_NUM_THREADS") or 0) or min(8, os.cpu_count() or 1))


@lru_cache(maxsize=1)
def _solver_pool() -> ThreadPoolExecutor:
    return ThreadPoolExecutor(_threads(), thread_name_prefix="direct-solve")


def _solve_group(sums, width, count, dim, complex_object, prior, amplitude_strength, phase_strength, dose,
                 positions, columns, rows, gain, spectrum, compute_loss, losses, offset=0):
    """Solve (Gram + diag(strength * prior)) x = rhs / dose per frequency in float64 by Cholesky, and place x.

    Log-amplitude coefficients come first for the complex model. Each coefficient is rounded to complex64, times
    ``gain`` (factor^2), and written with its Hermitian mirror into the packed spectrum eta + i phi (float32 pairs).
    ``losses`` receives x^H Gram x - 2 Re(x^H rhs) per frequency when ``compute_loss``. Returns the number of
    frequencies whose system was not positive definite or whose solution was not finite. Ranges of frequencies run
    on CPU threads; each writes only its own frequencies' spectrum entries, which never coincide. The system is rows
    and columns ``offset .. offset + dim - 1`` of the stored triangle (of size ``width``, right side last): ``offset``
    is ``count`` to solve the phase block of complex statistics.
    """
    size = sums.shape[0]
    arguments = (sums, width, count, dim, complex_object, prior, amplitude_strength, phase_strength, dose, positions,
                 columns, rows, gain, spectrum, compute_loss, losses, offset)
    starts = range(0, size, _SOLVE_CHUNK)
    if len(starts) <= 1:
        return int(_solve_range(0, size, *arguments))
    tasks = [_solver_pool().submit(_solve_range, start, min(start + _SOLVE_CHUNK, size), *arguments)
             for start in starts]
    return sum(int(task.result()) for task in tasks)


@njit(cache=True, nogil=True)
def _solve_range(first, last, sums, width, count, dim, complex_object, prior, amplitude_strength, phase_strength,
                 dose, positions, columns, rows, gain, spectrum, compute_loss, losses, offset):
    """`_solve_group` for frequencies ``first .. last - 1``; returns their failures."""
    failures = 0
    lower = np.empty((dim, dim), np.complex128)
    y = np.empty(dim, np.complex128)
    rhs = np.empty(dim, np.complex128)
    x = np.empty(dim, np.complex128)
    for f in range(first, last):
        # lower triangle of A = Gram + diag(penalty): A[i][j] = conj(M[j][i]) for i > j
        for j in range(dim):
            for i in range(j, dim):
                re, im = _entry(sums, f, width, offset + j, offset + i)
                lower[i, j] = complex(re, 0.0 if i == j else -im)
            strength = amplitude_strength if complex_object and j < count else phase_strength
            lower[j, j] += strength * prior[f, j % count]
            re, im = _entry(sums, f, width, offset + j, width - 1)
            rhs[j] = complex(re / dose, im / dose)
        ok = True
        for j in range(dim):
            d = lower[j, j].real
            for k in range(j):
                d -= lower[j, k].real * lower[j, k].real + lower[j, k].imag * lower[j, k].imag
            if not (d > 0.0 and d < np.inf):
                ok = False
                break
            d = math.sqrt(d)
            lower[j, j] = d
            for i in range(j + 1, dim):
                s = lower[i, j]
                for k in range(j):
                    s -= lower[i, k] * lower[j, k].conjugate()
                lower[i, j] = s / d
        if not ok:
            failures += 1
            continue
        for i in range(dim):
            s = rhs[i]
            for k in range(i):
                s -= lower[i, k] * y[k]
            y[i] = s / lower[i, i].real
        for i in range(dim - 1, -1, -1):
            s = y[i]
            for k in range(i + 1, dim):
                s -= lower[k, i].conjugate() * y[k]
            y[i] = s / lower[i, i].real
        # coefficients rounded to complex64, as CUDA returns them
        for i in range(dim):
            x[i] = complex(np.float64(np.float32(y[i].real)), np.float64(np.float32(y[i].imag)))
        if compute_loss:
            total = 0.0
            for a in range(dim):
                re, im = _entry(sums, f, width, offset + a, offset + a)
                total += re * (x[a].real * x[a].real + x[a].imag * x[a].imag)
                for b in range(a + 1, dim):
                    re, im = _entry(sums, f, width, offset + a, offset + b)
                    total += 2.0 * (x[a].conjugate() * complex(re, im) * x[b]).real
                total -= 2.0 * (x[a].conjugate() * rhs[a]).real
            losses[f] = total
        for a in range(count):
            phi_re = np.float32(x[dim - count + a].real) * gain
            phi_im = np.float32(x[dim - count + a].imag) * gain
            eta_re = np.float32(0.0)
            eta_im = np.float32(0.0)
            if complex_object:
                eta_re = np.float32(x[a].real) * gain
                eta_im = np.float32(x[a].imag) * gain
            if not (np.isfinite(phi_re) and np.isfinite(phi_im) and np.isfinite(eta_re) and np.isfinite(eta_im)):
                failures += 1
                continue
            p = positions[f, a]
            row, column = p // columns, p % columns
            mirror = ((rows - row) % rows) * columns + (columns - column) % columns
            # eta + i phi at q, conj(eta) + i conj(phi) at -q
            spectrum[p, 0] = eta_re - phi_im
            spectrum[p, 1] = eta_im + phi_re
            spectrum[mirror, 0] = eta_re + phi_im
            spectrum[mirror, 1] = phi_re - eta_im
    return failures


def wave_phase(wave: np.ndarray) -> np.ndarray:
    """``np.angle(wave)`` (float32 for a complex64 wave, bit for bit) in row blocks on the solver threads.

    Serves every MPS drag and preview frame: one thread takes about 11 ms for a 1536 x 1536 wave.
    """
    out = np.empty(wave.shape, wave.real.dtype)
    step = -(-wave.shape[0] // 8)

    def fill(rows):
        np.arctan2(wave.imag[rows], wave.real[rows], out=out[rows])

    list(_solver_pool().map(fill, [slice(start, start + step) for start in range(0, wave.shape[0], step)]))
    return out


def _inverse_fft(spectrum: np.ndarray) -> np.ndarray:
    """The complex64 inverse 2-D FFT (1 / size normalization) of a complex64 spectrum, on the CPU."""
    try:
        import scipy.fft
    except ModuleNotFoundError:
        return np.fft.ifft2(spectrum).astype(np.complex64)
    return scipy.fft.ifft2(spectrum, workers=_threads())


def _wave(spectrum: np.ndarray, complex_object: bool, incident_scale: float):
    """exp(eta + i phi) from the packed spectrum eta + i phi (both Hermitian), and the model's description.

    One complex64 inverse FFT returns eta as its real part and phi as its imaginary part; the phase method has no
    log-amplitude (eta = 0 exactly). The transform runs on the CPU (SciPy's pocketfft on the solver threads, else
    NumPy's), which rounds about a third as much as MLX's float32 FFT (1.7e-7 against 4.7e-7 of the image rms at
    1536^2); the spectrum is on the host already. The complex64 exponential is NumPy's (libm exp, cos, sin), the
    arithmetic of CUDA's, in row blocks on the solver threads.
    """
    field = _inverse_fft(spectrum)
    wave = np.empty_like(field)

    def fill(rows):
        part = field[rows]
        np.exp(part if complex_object else 1j * part.imag, out=wave[rows])

    step = -(-field.shape[0] // 8)
    list(_solver_pool().map(fill, [slice(start, start + step) for start in range(0, field.shape[0], step)]))
    if not np.isfinite(wave).all():
        raise ValueError('Direct inversion diverged; increase phase_damping or amplitude_damping, or use method="phase".')
    diagnostics = {
        "model": "T = exp(eta + i phi), linearized about unity",
        "normalization": "zero mean phase and log-amplitude; relative amplitude only",
        "frequency_policy": "supported aliases fitted jointly; DC family and even-grid Nyquist axes excluded",
        "incident_scale": incident_scale,
    }
    return wave, diagnostics
