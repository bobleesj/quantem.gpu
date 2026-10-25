"""Prepared weak-object amplitude/phase systems on the acquisition's GPU.

For each measured scan frequency, all supported aliases are fitted together.
The transfer is the same tilted, finite-depth probe model used by SSB fitting.
Regularization acts on the solution, never on a rendered image.
"""

import weakref
from dataclasses import dataclass

import cupy as cp
import numpy as np

from quantem.gpu.ssb.cuda import chebyshev
from quantem.gpu.ssb.cuda.kernels.engine import _THICK_GEOMETRY_SOURCE
from quantem.gpu.ssb.higher_order import CHI_SOURCE, pack_coefficients
from quantem.gpu.ssb.units import higher_order_arrays

_evaluate = cp.ElementwiseKernel(
    "raw complex128 basis, raw float64 eigen, float64 strength, int32 dim",
    "complex64 coefficient",
    """complex<double> value(0.0, 0.0);
       for (int j = 0; j < dim; ++j) {
           value += basis[i * dim + j] / (eigen[(i / dim) * dim + j] + strength);
       }
       coefficient = complex<float>(value.real(), value.imag());""",
    "direct_ptycho_regularization",
)


_WARPS = 4
_ENTRIES_PER_WARP = 128

# Normal equations of every scan frequency from the augmented vector
# V = (T_1 .. T_dim, G) at each BF pixel: its weighted Gram V^H W V holds the
# normal matrix T^H W T, the right side T^H W G and the signal power G^H W G.
#
# One warp takes one frequency and walks the BF pixels 32 at a time, lane by
# pixel. It evaluates the transfer (the expressions of weak_object._transfer,
# k-only terms precomputed once) and adds every entry a <= b of the augmented
# Gram to running sums held in registers; the system size is a template
# constant so V and the sums never leave registers, and no shared memory or
# barrier is needed until the lanes meet. A transfer term is exactly zero
# outside the lens where the two apertures overlap, so its chi and phase factor
# are skipped there.
#
# The products and the per-lane sums (about 300 terms each) are float32; the 32
# lane sums then meet in float64. Float64 runs at 1/64 of the float32 rate on
# this hardware, and the transfer itself is a float32 model, so the Gram agrees
# with float64 sums to a few float32 ulps of its largest entry (3e-7 on the
# test field), the same as the difference between two float32 evaluations of
# the transfer.
_SOURCE = (
    r"""
#include <cupy/complex.cuh>
#define SSB_DEVICE __device__
#define SSB_COEFFICIENTS const float*
"""
    + CHI_SOURCE
    + _THICK_GEOMETRY_SOURCE
    + r"""
// k-only terms of the transfer, shared by every frequency, as two float4 per pixel:
// (kr, kc, chi(k), w) and (alpha(k)^2, aperture(k), detector row of G_qk, 0)
extern "C" __global__ void direct_pixel_terms(
    const int* order, const float* kr_bf, const float* kc_bf, const float* weights, int nbf,
    float wl, float alpha, float dr, float dc, const float* coefficients, float4* pixel) {
    const int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= nbf) return;
    const int k = order[j];
    const float4 g = thick_probe_geometry(kr_bf[k], kc_bf[k], wl, alpha, dr, dc);
    pixel[2 * j] = make_float4(kr_bf[k], kc_bf[k], ssb_chi_full(kr_bf[k], kc_bf[k], wl, coefficients), weights[k]);
    pixel[2 * j + 1] = make_float4(g.x, g.w, __int_as_float(k), 0.0f);
}

// Block x holds _WARPS frequencies, one per warp; block y sums entries
// [first, last) of each, so very large systems split instead of spilling
// registers (unused sums are dead code and take none).
// Register budget per thread for blocks of 32 _WARPS = 128 threads, measured fastest on
// the test field: 5 resident blocks (96 registers) for phase, 3 (168) for complex, whose
// running sums are 2.5x larger.
template <int COUNT, int COMPLEX, int PARTS>
__global__ void __launch_bounds__(128, COMPLEX ? 3 : 5) direct_normal_equations(
    const complex<float>* spectrum, int plane, const int* frequency, int nfrequency,
    const float* alias_qr, const float* alias_qc,
    const float4* pixel, int nbf,
    float wl, float alpha, float dr, float dc, float depth, float tr, float tc,
    const float* coefficients, double* augmented) {
    constexpr int DIM = COMPLEX ? 2 * COUNT : COUNT;
    constexpr int WIDTH = DIM + 1;
    constexpr int OUTPUTS = WIDTH * (WIDTH + 1) / 2;
    constexpr int SHARE = (OUTPUTS + PARTS - 1) / PARTS;
    const int lane = threadIdx.x % 32;
    const int f = blockIdx.x * (blockDim.x / 32) + threadIdx.x / 32;
    if (f >= nfrequency) return;
    const int first = blockIdx.y * SHARE, last = min(first + SHARE, OUTPUTS);
    const int column = frequency[f];
    float qr[COUNT > 0 ? COUNT : 1], qc[COUNT > 0 ? COUNT : 1];
#pragma unroll
    for (int alias = 0; alias < COUNT; ++alias) {
        qr[alias] = alias_qr[f * COUNT + alias];
        qc[alias] = alias_qc[f * COUNT + alias];
    }
    float sum_re[OUTPUTS], sum_im[OUTPUTS];
#pragma unroll
    for (int i = 0; i < OUTPUTS; ++i) { sum_re[i] = 0.0f; sum_im[i] = 0.0f; }
    for (int k = lane; k < nbf; k += 32) {
        const float4 pk = pixel[2 * k], gk = pixel[2 * k + 1];
        const float kr = pk.x, kc = pk.y, ck = pk.z, w = pk.w;
        complex<float> v[WIDTH];
        v[DIM] = spectrum[(long long)__float_as_int(gk.z) * plane + column];
#pragma unroll
        for (int alias = 0; alias < COUNT; ++alias) {
            float4 m = thick_probe_geometry(kr-qr[alias], kc-qc[alias], wl, alpha, dr, dc);
            float4 pp = thick_probe_geometry(kr+qr[alias], kc+qc[alias], wl, alpha, dr, dc);
            float shift = 6.283185307f * (qr[alias]*tr + qc[alias]*tc);
            complex<float> t1(0.0f, 0.0f), t2(0.0f, 0.0f);
            if (m.w * gk.y != 0.0f) {
                float cm = ssb_chi_full(kr-qr[alias], kc-qc[alias], wl, coefficients);
                float w1 = thick_sinc(0.5f * depth * (-3.141592654f/wl*(m.x-gk.x)-shift));
                t1 = m.w*gk.y*w1*complex<float>(cosf(cm-ck), -sinf(cm-ck));
            }
            if (pp.w * gk.y != 0.0f) {
                float cp_ = ssb_chi_full(kr+qr[alias], kc+qc[alias], wl, coefficients);
                float w2 = thick_sinc(0.5f * depth * (3.141592654f/wl*(pp.x-gk.x)-shift));
                t2 = pp.w*gk.y*w2*complex<float>(cosf(cp_-ck), sinf(cp_-ck));
            }
            v[DIM - COUNT + alias] = complex<float>(0, 1) * (t1 - t2);
            if (COMPLEX) v[alias] = t1 + t2;
        }
        int i = 0;
#pragma unroll
        for (int a = 0; a < WIDTH; ++a) {
            const float ur = w * v[a].real(), ui = -w * v[a].imag();
#pragma unroll
            for (int b = a; b < WIDTH; ++b, ++i) {
                if (i >= first && i < last) {
                    sum_re[i] = fmaf(ur, v[b].real(), fmaf(-ui, v[b].imag(), sum_re[i]));
                    sum_im[i] = fmaf(ur, v[b].imag(), fmaf(ui, v[b].real(), sum_im[i]));
                }
            }
        }
    }
    double* out = augmented + 2 * ((long long)f * WIDTH * WIDTH);
    int i = 0;
#pragma unroll
    for (int a = 0; a < WIDTH; ++a) {
#pragma unroll
        for (int b = a; b < WIDTH; ++b, ++i) {
            if (i < first || i >= last) continue;
            double re = sum_re[i], im = sum_im[i];
            for (int offset = 16; offset > 0; offset /= 2) {
                re += __shfl_xor_sync(0xffffffffu, re, offset);
                im += __shfl_xor_sync(0xffffffffu, im, offset);
            }
            // a Hermitian diagonal is real; the float32 products leave a rounding residue
            if (a == b) im = 0.0;
            if (lane == 0) {
                out[2 * (a * WIDTH + b)] = re;
                out[2 * (a * WIDTH + b) + 1] = im;
                out[2 * (b * WIDTH + a)] = re;
                out[2 * (b * WIDTH + a) + 1] = -im;
            }
        }
    }
}

// Eigenbasis of one phase system per thread: M = D G D / scale with D = diag(prior^-1/2),
// M = V diag(lambda) V^H by cyclic complex Jacobi rotations, and
// basis[i][j] = D_i V_ij (V^H D rhs / scale)_j, so any strength s evaluates as
// x_i = sum_j basis[i][j] / (lambda_j + s). Each rotation first turns M_pq real
// with a phase on column q, then zeroes it with a real rotation (c, s).
// Every loop over N is unrolled so each index is a compile-time constant: with
// runtime indices into these local arrays the optimized NVRTC build returned
// wrong eigenvalues for N >= 5 (correct under -G).
template <int N>
__global__ void direct_eigenbasis(const complex<double>* gram, const complex<double>* rhs,
                                  const double* inverse, double scale, int nfrequency,
                                  double* eigen, complex<double>* basis) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nfrequency) return;
    double mr[N][N], mi[N][N], vr[N][N], vi[N][N], d[N];
    #pragma unroll
    for (int i = 0; i < N; ++i) d[i] = inverse[f * N + i];
    #pragma unroll
    for (int i = 0; i < N; ++i)
        #pragma unroll
        for (int j = 0; j < N; ++j) {
            const complex<double> g = gram[((long long)f * N + i) * N + j];
            mr[i][j] = g.real() * d[i] * d[j] / scale;
            mi[i][j] = g.imag() * d[i] * d[j] / scale;
            vr[i][j] = i == j ? 1.0 : 0.0;
            vi[i][j] = 0.0;
        }
    for (int sweep = 0; sweep < 30; ++sweep) {
        double off = 0.0, diagonal = 0.0;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            diagonal += mr[i][i] * mr[i][i];
            #pragma unroll
            for (int j = i + 1; j < N; ++j) off += mr[i][j] * mr[i][j] + mi[i][j] * mi[i][j];
        }
        if (off <= 1e-34 * diagonal) break;
        #pragma unroll
        for (int p = 0; p < N - 1; ++p)
            #pragma unroll
            for (int q = p + 1; q < N; ++q) {
                const double magnitude = hypot(mr[p][q], mi[p][q]);
                if (magnitude == 0.0) continue;
                // e = M_pq / |M_pq|: column q times conj(e) and row q times e make M_pq real
                const double er = mr[p][q] / magnitude, ei = mi[p][q] / magnitude;
                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    double re = mr[k][q], im = mi[k][q];
                    mr[k][q] = re * er + im * ei; mi[k][q] = im * er - re * ei;
                    re = vr[k][q]; im = vi[k][q];
                    vr[k][q] = re * er + im * ei; vi[k][q] = im * er - re * ei;
                }
                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    const double re = mr[q][k], im = mi[q][k];
                    mr[q][k] = re * er - im * ei; mi[q][k] = im * er + re * ei;
                }
                const double theta = (mr[q][q] - mr[p][p]) / (2.0 * magnitude);
                double t = 1.0 / (fabs(theta) + sqrt(theta * theta + 1.0));
                if (theta < 0.0) t = -t;
                const double c = 1.0 / sqrt(t * t + 1.0), s = t * c;
                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    double a = mr[k][p], b = mr[k][q];
                    mr[k][p] = c * a - s * b; mr[k][q] = s * a + c * b;
                    a = mi[k][p]; b = mi[k][q];
                    mi[k][p] = c * a - s * b; mi[k][q] = s * a + c * b;
                    a = vr[k][p]; b = vr[k][q];
                    vr[k][p] = c * a - s * b; vr[k][q] = s * a + c * b;
                    a = vi[k][p]; b = vi[k][q];
                    vi[k][p] = c * a - s * b; vi[k][q] = s * a + c * b;
                }
                #pragma unroll
                for (int k = 0; k < N; ++k) {
                    double a = mr[p][k], b = mr[q][k];
                    mr[p][k] = c * a - s * b; mr[q][k] = s * a + c * b;
                    a = mi[p][k]; b = mi[q][k];
                    mi[p][k] = c * a - s * b; mi[q][k] = s * a + c * b;
                }
                mr[p][q] = mi[p][q] = mr[q][p] = mi[q][p] = 0.0;
            }
    }
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        // projected = (V^H D rhs / scale)_j
        double pr = 0.0, pi = 0.0;
        #pragma unroll
        for (int i = 0; i < N; ++i) {
            const complex<double> b = rhs[f * N + i];
            const double br = b.real() * d[i] / scale, bi = b.imag() * d[i] / scale;
            pr += vr[i][j] * br + vi[i][j] * bi;
            pi += vr[i][j] * bi - vi[i][j] * br;
        }
        eigen[f * N + j] = mr[j][j];
        #pragma unroll
        for (int i = 0; i < N; ++i)
            basis[((long long)f * N + i) * N + j] = complex<double>(d[i] * (vr[i][j] * pr - vi[i][j] * pi),
                                                                    d[i] * (vr[i][j] * pi + vi[i][j] * pr));
    }
}
"""
)
_pixel_terms = cp.RawKernel(_SOURCE, "direct_pixel_terms", options=("-std=c++17",))

# Serves: every DragWindow frame, about 1.3e5 frequencies of 3-6 aliases (4 x 4
# phase, 8 x 8 complex) at 3x on a 512 x 512 scan, 0.7 / 1.5 ms there.
# The regularized normal equations (G + diag(p)) x = b of one frequency per
# thread, Hermitian positive definite for p > 0, so a float64 Cholesky
# factorization solves them stably (condition below about 1e2 on the test field
# at strength 0.03). Loops are unrolled for the same NVRTC reason as
# direct_eigenbasis.
_CHOLESKY_SOURCE = r"""
#include <cupy/complex.cuh>
template <int N>
__global__ void direct_cholesky(const complex<double>* gram, const complex<double>* rhs,
                                const double* penalty, int nfrequency, complex<float>* out) {
    const int f = blockIdx.x * blockDim.x + threadIdx.x;
    if (f >= nfrequency) return;
    complex<double> L[N][N], y[N];
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        #pragma unroll
        for (int j = 0; j < N; ++j) L[i][j] = gram[((long long)f * N + i) * N + j];
        L[i][i] += penalty[f * N + i];
        y[i] = rhs[f * N + i];
    }
    #pragma unroll
    for (int j = 0; j < N; ++j) {
        double d = L[j][j].real();
        #pragma unroll
        for (int k = 0; k < j; ++k) d -= norm(L[j][k]);
        d = sqrt(d);
        L[j][j] = d;
        #pragma unroll
        for (int i = j + 1; i < N; ++i) {
            complex<double> s = L[i][j];
            #pragma unroll
            for (int k = 0; k < j; ++k) s -= L[i][k] * conj(L[j][k]);
            L[i][j] = s / d;
        }
    }
    #pragma unroll
    for (int i = 0; i < N; ++i) {
        complex<double> s = y[i];
        #pragma unroll
        for (int k = 0; k < i; ++k) s -= L[i][k] * y[k];
        y[i] = s / L[i][i].real();
    }
    #pragma unroll
    for (int i = N - 1; i >= 0; --i) {
        complex<double> s = y[i];
        #pragma unroll
        for (int k = i + 1; k < N; ++k) s -= conj(L[k][i]) * y[k];
        y[i] = s / L[i][i].real();
        out[f * N + i] = complex<float>(y[i].real(), y[i].imag());
    }
}
"""
_kernels = {}


def _normal_equations(count: int, complex_object: bool):
    """The normal-equation kernel for one alias count, and the blocks per frequency.

    Systems of more than _ENTRIES_PER_WARP entries a <= b split over several
    blocks per frequency, each evaluating the transfer again, rather than
    spilling their running sums out of registers.
    """
    width = (2 * count if complex_object else count) + 1
    parts = -(-width * (width + 1) // 2 // _ENTRIES_PER_WARP)
    name = f"direct_normal_equations<{count}, {int(complex_object)}, {parts}>"
    if name not in _kernels:
        module = cp.RawModule(code=_SOURCE, options=("-std=c++17",), name_expressions=[name])
        _kernels[name] = module.get_function(name)
    return _kernels[name], parts


def _cholesky(dim: int):
    """The batched Cholesky solve kernel for regularized systems of one size."""
    name = f"direct_cholesky<{dim}>"
    if name not in _kernels:
        module = cp.RawModule(code=_CHOLESKY_SOURCE, options=("-std=c++17",), name_expressions=[name])
        _kernels[name] = module.get_function(name)
    return _kernels[name]


def _eigenbasis(count: int):
    """The batched Jacobi eigenbasis kernel for phase systems of one alias count."""
    name = f"direct_eigenbasis<{count}>"
    if name not in _kernels:
        module = cp.RawModule(code=_SOURCE, options=("-std=c++17",), name_expressions=[name])
        _kernels[name] = module.get_function(name)
    return _kernels[name]


@dataclass
class _AliasGroup:
    """The frequencies with one count of supported aliases, and their optics-free terms."""

    count: int
    columns: object
    qr: object
    qc: object
    rows: object
    cols: object
    prior: object
    inverse: object


@dataclass
class _Layout:
    """Everything of a preparation that the data and the sampling fix, not the optics."""

    shape: tuple[int, int]
    dose: float
    weights: object
    scale: object
    scale_value: float
    order: object
    groups: list[_AliasGroup]


def _layout(engine, factor: int, smoothness_inv_A: float, detector_weighting: str):
    """The optics-free part of a preparation, computed once per data and sampling.

    The dose, detector weights and their scale, the supported aliases of every
    non-DC frequency in the half plane, the prior (1 + q^2/s^2)^2 and the BF
    pixel order take host round trips that a 30 fps optics change cannot
    afford, so the engine keeps the last one (a few MB).
    """
    # a weak reference, so a replaced G_qk neither matches nor stays alive
    key = (factor, smoothness_inv_A, detector_weighting)
    cached = getattr(engine, "_direct_layout", None)
    if cached is not None and cached[0]() is engine.G_qk and cached[1] == key:
        return cached[2]
    nbf, nrow, ncol = engine._bf_grid()
    if min(nrow, ncol) < 3:
        raise ValueError(
            "Direct inversion needs at least 3 scan positions per side; select a larger region."
        )
    out_row, out_col = factor * nrow, factor * ncol
    cache = engine._cache
    aperture2 = cache["aperture_k_1d"] ** 2
    dose = cp.sum(engine.G_qk[:, 0, 0].real, dtype=cp.float64)
    dose /= nrow * ncol * aperture2.sum()
    if not bool(cp.isfinite(dose) & (dose > 0)):
        raise ValueError(
            "No positive incident intensity estimate; check counts and bright-field calibration."
        )
    if detector_weighting == "uniform":
        weights = cp.ones(nbf, cp.float32)
    else:
        mean = cp.maximum(engine.G_qk[:, 0, 0].real / (nrow * ncol), 0)
        floor = cp.median(mean) * 0.05
        if float(floor) <= 0:
            raise ValueError(
                "Variance weighting needs positive bright-field counts; use uniform weighting."
            )
        weights = 1 / cp.maximum(mean, floor)
        weights /= weights.mean()
    scale = cp.sum(weights * aperture2**2, dtype=cp.float64)
    rows, cols = cp.indices((nrow, ncol // 2 + 1), dtype=cp.int32)
    keep = (cols > 0) | ((rows > 0) & (rows <= (nrow - 1) // 2))
    if nrow % 2 == 0:
        keep &= rows != nrow // 2
    if ncol % 2 == 0:
        keep &= cols != ncol // 2
    rows, cols = rows[keep], cols[keep]
    alias_row, alias_col = cp.indices((factor, factor), dtype=cp.int32)
    output_rows = rows[:, None] + alias_row.ravel()[None, :] * nrow
    output_cols = cols[:, None] + alias_col.ravel()[None, :] * ncol
    qr = (
        cp.where(output_rows <= (out_row - 1) // 2, output_rows, output_rows - out_row).astype(cp.float32)
        * cache["qx_1d"][1]
    )
    qc = (
        cp.where(output_cols <= (out_col - 1) // 2, output_cols, output_cols - out_col).astype(cp.float32)
        * cache["qy_1d"][1]
    )
    cutoff = (2 * cache["semiangle_rad"] + max(cache["ang_y_rad"], cache["ang_x_rad"])) / engine.wavelength
    active = qr * qr + qc * qc < cutoff**2
    counts = active.sum(axis=1)
    columns = rows * np.int32(engine.G_qk.shape[2]) + cols
    groups = []
    for count, size in enumerate(cp.bincount(counts).get().tolist()):
        if not size:
            continue
        selected = cp.flatnonzero(counts == count)
        if not count:
            # frequencies without a supported alias add only to the signal power
            empty = cp.zeros(0, cp.float32)
            groups.append(_AliasGroup(0, columns[selected], empty, empty, None, None, None, None))
            continue
        group_index, alias_index = cp.nonzero(active[selected])
        group_index = selected[group_index]
        prior = cp.ones((size, count), cp.float64)
        if smoothness_inv_A:
            frequency2 = (qr[group_index, alias_index] ** 2 + qc[group_index, alias_index] ** 2).reshape(-1, count)
            prior = (1 + frequency2 / smoothness_inv_A**2) ** 2
        groups.append(
            _AliasGroup(
                count,
                columns[selected],
                cp.ascontiguousarray(qr[group_index, alias_index]),
                cp.ascontiguousarray(qc[group_index, alias_index]),
                output_rows[group_index, alias_index].reshape(-1, count),
                output_cols[group_index, alias_index].reshape(-1, count),
                prior,
                1 / cp.sqrt(prior.astype(cp.float64)),
            )
        )
    # BF pixels in 4 x 8 detector tiles, so the 32 pixels of a warp are compact and
    # the aperture-overlap branches agree across its lanes (about 10 % faster)
    detector_row = cp.asarray(engine.bf_inds_row)
    detector_col = cp.asarray(engine.bf_inds_col)
    order = cp.lexsort(cp.stack((detector_col % 8, detector_row % 4, detector_col // 8, detector_row // 4)))
    layout = _Layout(
        (out_row, out_col), float(dose), weights.astype(cp.float32, copy=False), scale, float(scale),
        order.astype(cp.int32), groups,
    )
    engine._direct_layout = weakref.ref(engine.G_qk), key, layout
    return layout


@dataclass
class _Group:
    rows: object
    cols: object
    gram: object
    rhs: object
    prior: object
    basis: object | None
    eigen: object | None


@dataclass
class PreparedDirect:
    """Sufficient statistics for one fixed data, optics and model selection."""

    shape: tuple[int, int]
    factor: int
    groups: list[_Group]
    complex_object: bool
    scale: object
    signal_power: object
    incident_scale: float

    @classmethod
    def prepare(
        cls,
        engine,
        optics,
        factor: int,
        *,
        complex_object: bool,
        smoothness_inv_A: float,
        detector_weighting: str,
        eigenbasis: bool = True,
    ):
        """Build normal equations once; strengths are applied at evaluation.

        Each scan frequency Q with its supported aliases needs the Gram
        A = T^H W T, the right side b = T^H W g and the signal power
        sum_k w_k |g_k|^2, where T (BF pixel, alias) is the transfer, W the
        detector weights and g = G_qk / dose the measured spectrum. Everything
        but T depends on the data alone and is kept on the engine
        (`_layout`), so a change of optics costs one kernel per alias count,
        which evaluates T and reduces all three (`_normal_equations`), and for
        phase one batched eigendecomposition (skipped with ``eigenbasis=False``
        by a `DragWindow` node, which solves by Cholesky instead).
        """
        layout = _layout(engine, factor, smoothness_inv_A, detector_weighting)
        cache = engine._cache
        higher = higher_order_arrays(optics["aberrations"])
        if higher is None:
            magnitudes = np.zeros(14, np.float32)
            angles = np.zeros(14, np.float32)
            magnitudes[:2] = [
                optics["aberrations"].get("C10_nm", 0),
                optics["aberrations"].get("C12_nm", 0),
            ]
            angles[1] = np.deg2rad(optics["aberrations"].get("phi12_deg", 0))
        else:
            magnitudes, angles = higher
        coefficients = cp.asarray(pack_coefficients(magnitudes * 10, angles))
        tilt_row, tilt_col = optics["sample_tilt_mrad"]
        optics_scalars = (
            np.float32(engine.wavelength),
            np.float32(cache["semiangle_rad"]),
            np.float32(cache["ang_y_rad"]),
            np.float32(cache["ang_x_rad"]),
            np.float32(optics["depth_spread_nm"] * 10),
            np.float32(tilt_row * 1e-3),
            np.float32(tilt_col * 1e-3),
        )
        nbf = layout.weights.size
        pixel = cp.empty((nbf, 2, 4), cp.float32)
        _pixel_terms(
            (-(-nbf // 256),),
            (256,),
            (layout.order, cache["kx_bf"], cache["ky_bf"], layout.weights, np.int32(nbf),
             *optics_scalars[:4], coefficients, pixel),
        )
        spectrum = engine.G_qk.reshape(nbf, -1)
        groups = []
        power = cp.zeros((), cp.float64)
        for group in layout.groups:
            count, size = group.count, group.columns.size
            dim = 2 * count if complex_object else count
            augmented = cp.empty((size, dim + 1, dim + 1), cp.complex128)
            normal_equations, parts = _normal_equations(count, complex_object)
            normal_equations(
                (-(-size // _WARPS), parts),
                (32 * _WARPS,),
                (spectrum, np.int32(spectrum.shape[1]), group.columns, np.int32(size), group.qr, group.qc,
                 pixel, np.int32(nbf), *optics_scalars, coefficients, augmented),
            )
            # G enters unscaled so V stays float32; the dose divides afterwards.
            gram = cp.ascontiguousarray(augmented[:, :dim, :dim])
            rhs = augmented[:, :dim, dim] / layout.dose
            power += augmented[:, dim, dim].real.sum() / layout.dose**2
            if not count:
                continue
            if complex_object or not eigenbasis:
                prior = cp.concatenate((group.prior, group.prior), axis=1) if complex_object else group.prior
                basis = eigen = None
            else:
                prior = group.prior
                eigen = cp.empty((size, count), cp.float64)
                basis = cp.empty((size, count, count), cp.complex128)
                _eigenbasis(count)(
                    (-(-size // 128),),
                    (128,),
                    (gram, cp.ascontiguousarray(rhs), group.inverse, np.float64(layout.scale_value),
                     np.int32(size), eigen, basis),
                )
            groups.append(_Group(group.rows, group.cols, gram, rhs, prior, basis, eigen))
        return cls(layout.shape, factor, groups, complex_object, layout.scale, power, layout.dose)

    def solve(
        self,
        phase_regularization: float,
        amplitude_regularization: float | None,
        *,
        compute_loss: bool,
    ):
        """Evaluate arbitrary strengths, returning fresh wave and diagnostics."""
        spectra = [cp.zeros(self.shape, cp.complex64), cp.zeros(self.shape, cp.complex64)]
        residual = self.signal_power.copy()
        for group in self.groups:
            count = group.rows.shape[1]
            if self.complex_object:
                matrix = group.gram.copy()
                penalty = group.prior.astype(cp.float64) * self.scale
                penalty[:, :count] *= amplitude_regularization
                penalty[:, count:] *= phase_regularization
                diagonal = cp.arange(matrix.shape[-1])
                matrix[:, diagonal, diagonal] += penalty
                coefficients = cp.linalg.solve(matrix, group.rhs[..., None])[
                    ..., 0
                ].astype(cp.complex64)
            else:
                nfrequency, dim = group.eigen.shape
                coefficients = _evaluate(
                    group.basis,
                    group.eigen,
                    phase_regularization,
                    dim,
                    size=nfrequency * dim,
                ).reshape(nfrequency, dim)
            if compute_loss:
                precise = coefficients.astype(cp.complex128)
                prediction = (group.gram @ precise[..., None])[..., 0]
                residual += cp.sum(
                    (precise.conj() * prediction).real
                    - 2 * (precise.conj() * group.rhs).real
                )
            _scatter(spectra, group.rows, group.cols, coefficients, self.complex_object, self.factor)
        wave, diagnostics = _wave(spectra, self.incident_scale)
        if compute_loss:
            if float(self.signal_power) > 0:
                diagnostics["training_relative_sse"] = float(
                    cp.maximum(residual, 0) / self.signal_power
                )
            else:
                diagnostics["training_relative_sse"] = None
                diagnostics["prediction_error_status"] = "undefined: no non-DC signal"
        return wave, diagnostics


# ---


class DragWindow:
    """Prepared statistics along one aberration control, for redrawing while it is dragged.

    Serves: ShowPtycho drags of C10, C12 or phi12 (any other term works the
    same) at 3x on the full 512 x 512 scan, 1 ms phase and 3 ms complex per
    frame after a 0.45 / 0.75 s build.

    Along one control with everything else fixed, every Gram entry and right
    side is an entire function of the control (`chebyshev`), so ``step`` adds
    one exact preparation at a Chebyshev node and ``solve`` evaluates any value
    in the window as one matrix-vector product, then a float64 Cholesky solve
    per frequency, instead of evaluating the transfer at every BF pixel. A
    complex window also holds the phase statistics as its lower-right block,
    so one build serves both regularized methods.
    """

    def __init__(self, engine, optics, control: str, factor: int, *, complex_object: bool,
                 smoothness_inv_A: float, detector_weighting: str):
        self.engine, self.optics, self.control, self.factor = engine, optics, control, factor
        self.complex_object = complex_object
        self.options = dict(complex_object=complex_object, smoothness_inv_A=smoothness_inv_A,
                            detector_weighting=detector_weighting, eigenbasis=False)
        self.series = chebyshev.series(engine, optics["aberrations"], control)
        self.template = None

    def step(self) -> bool:
        """Add the next node's exact statistics; True once the window is complete."""
        if self.series.ready:
            return True
        aberrations = {**self.optics["aberrations"], self.control: self.series.nodes[self.series.added]}
        prepared = PreparedDirect.prepare(self.engine, {**self.optics, "aberrations": aberrations},
                                          self.factor, **self.options)
        self.template = self.template or prepared
        self.series.add(cp.concatenate([part.ravel() for group in prepared.groups for part in (group.gram, group.rhs)]))
        return self.series.ready

    def solve(self, value: float, phase_regularization: float, amplitude_regularization: float | None,
              *, complex_object: bool):
        """The wave at control ``value`` (inside the window), as PreparedDirect.solve without the loss."""
        values = self.series.evaluate(value)
        template = self.template
        spectra = [cp.zeros(template.shape, cp.complex64), cp.zeros(template.shape, cp.complex64)]
        at = 0
        for group in template.groups:
            size, dim = group.rhs.shape
            gram = values[at:at + size * dim * dim].reshape(size, dim, dim)
            at += gram.size
            rhs = values[at:at + size * dim].reshape(size, dim)
            at += rhs.size
            count = group.rows.shape[1]
            penalty = group.prior * template.scale
            if complex_object:
                penalty = penalty * cp.concatenate((cp.full(count, amplitude_regularization), cp.full(count, phase_regularization)))
            else:
                if self.complex_object:
                    gram, rhs, penalty = gram[:, count:, count:], rhs[:, count:], penalty[:, count:]
                penalty = penalty * phase_regularization
            width = rhs.shape[1]
            coefficients = cp.empty((size, width), cp.complex64)
            _cholesky(width)(
                (-(-size // 64),), (64,),
                (cp.ascontiguousarray(gram), cp.ascontiguousarray(rhs), cp.ascontiguousarray(penalty),
                 np.int32(size), coefficients),
            )
            _scatter(spectra, group.rows, group.cols, coefficients, complex_object, self.factor)
        return _wave(spectra, template.incident_scale)

def _scatter(spectra, rows, cols, coefficients, complex_object, factor):
    """Place one group's alias coefficients (log-amplitude first, then phase) and their Hermitian mirrors."""
    count = rows.shape[1]
    shape = spectra[0].shape
    channels = [(spectra[1], coefficients[:, -count:])]
    if complex_object:
        channels.append((spectra[0], coefficients[:, :count]))
    for spectrum, values in channels:
        values = values * factor**2
        spectrum[rows, cols] = values
        spectrum[(-rows) % shape[0], (-cols) % shape[1]] = values.conj()


def _wave(spectra, incident_scale):
    """exp(eta + i phi) from the log-amplitude and phase spectra, and the model's description."""
    amplitude, phase = [cp.fft.ifft2(spectrum).real.astype(cp.float32) for spectrum in spectra]
    wave = cp.exp(amplitude + 1j * phase).astype(cp.complex64)
    if not bool(cp.isfinite(wave).all()):
        raise ValueError('Direct inversion diverged; increase regularization or use method="phase".')
    diagnostics = {
        "model": "T = exp(eta + i phi), linearized about unity",
        "normalization": "zero mean phase and log-amplitude; relative amplitude only",
        "frequency_policy": "supported aliases fitted jointly; DC family and even-grid Nyquist axes excluded",
        "incident_scale": incident_scale,
    }
    return wave, diagnostics
