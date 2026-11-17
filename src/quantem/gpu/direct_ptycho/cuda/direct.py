"""Prepared weak-object amplitude/phase systems on the acquisition's GPU.

For each measured scan frequency, all supported aliases are fitted together.
The transfer is the same tilted, finite-depth probe model used by direct ptychography fitting.
Regularization acts on the solution, never on a rendered image.
"""

import weakref
from dataclasses import dataclass, replace

import cupy as cp
import numpy as np

from quantem.gpu.direct_ptycho.cuda import chebyshev
from quantem.gpu.direct_ptycho.cuda.kernels.engine import _THICK_GEOMETRY_SOURCE
from quantem.gpu.direct_ptycho.higher_order import CHI_SOURCE, pack_coefficients
from quantem.gpu.direct_ptycho.units import higher_order_arrays

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
#define DIRECT_DEVICE __device__
#define DIRECT_COEFFICIENTS const float*
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
    pixel[2 * j] = make_float4(kr_bf[k], kc_bf[k], direct_chi_full(kr_bf[k], kc_bf[k], wl, coefficients), weights[k]);
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
                float cm = direct_chi_full(kr-qr[alias], kc-qc[alias], wl, coefficients);
                float w1 = thick_sinc(0.5f * depth * (-3.141592654f/wl*(m.x-gk.x)-shift));
                t1 = m.w*gk.y*w1*complex<float>(cosf(cm-ck), -sinf(cm-ck));
            }
            if (pp.w * gk.y != 0.0f) {
                float cp_ = direct_chi_full(kr+qr[alias], kc+qc[alias], wl, coefficients);
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

"""
)
_pixel_terms = cp.RawKernel(_SOURCE, "direct_pixel_terms", options=("-std=c++17",))

# Serves: every DragWindow frame, about 1.3e5 frequencies of 3-6 aliases (4 x 4
# phase, 8 x 8 complex) at 3x on a 512 x 512 scan, 0.7 / 1.5 ms there.
# The regularized normal equations (G + diag(p)) x = b of one frequency per
# thread, Hermitian positive definite for p > 0, so a float64 Cholesky
# factorization solves them stably (condition below about 1e2 on the test field
# at strength 0.03). Loops are unrolled: with runtime indices into local
# arrays the optimized NVRTC build has returned wrong results. Checked against
# cp.linalg.solve for 1 to 64 aliases (8x). Every phase solve uses it since
# 11-16; the per-frequency Jacobi eigenbasis before it was wrong at 16 aliases
# (4x with a coarse scan step) and 400x slower there.
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


def _layout(engine, factor: int, smoothness_inv_A: float, detector_weighting: str, support_margin: float = 1.0):
    """The optics-free part of a preparation, computed once per data and sampling.

    The dose, detector weights and their scale, the supported aliases of every
    non-DC frequency in the half plane, the prior (1 + q^2/s^2)^2 and the BF
    pixel order take host round trips that a 30 fps optics change cannot
    afford, so the engine keeps the last one (a few MB).
    """
    # a weak reference, so a replaced G_qk neither matches nor stays alive
    key = (factor, smoothness_inv_A, detector_weighting, support_margin)
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
    active = qr * qr + qc * qc < (support_margin * cutoff)**2
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
        detector_mask=None,
        column_spectra=None,
    ):
        """Build normal equations once; strengths are applied at evaluation.

        ``column_spectra`` serves an encoded scan whose G(q, k) the device cannot hold (``engine`` is then a
        ``StreamedGeometry``): a callable returning batches ``(first, stop, spectra)`` of bright-field pixels in
        the engine's order. The equations are sums over bright-field pixels, so each batch's are added in float64
        and the device holds one batch of spectra and the equations, never G: the same system, no data left out.

        Each scan frequency Q with its supported aliases needs the Gram
        A = T^H W T, the right side b = T^H W g and the signal power
        sum_k w_k |g_k|^2, where T (BF pixel, alias) is the transfer, W the
        detector weights and g = G_qk / dose the measured spectrum. Everything
        but T depends on the data alone and is kept on the engine
        (`_layout`), so a change of optics costs one kernel per alias count,
        which evaluates T and reduces all three (`_normal_equations`). Each
        strength is then one float64 Cholesky solve per frequency (`solve`).
        """
        basis = optics.get("scan_basis")
        transform = None if basis is None else np.linalg.inv(np.asarray(basis)).T
        # Keep a fixed union of aliases for small calibrated scan distortions.
        # The bound also covers larger stretches; the transfer remains exact.
        margin = (1.0 if transform is None else
                  max(1.05, 1 / np.linalg.svd(transform, compute_uv=False).min()))
        layout = _layout(engine, factor, smoothness_inv_A, detector_weighting, margin)
        if detector_mask is not None:
            mask = cp.asarray(detector_mask)
            if mask.dtype != cp.bool_ or mask.shape != layout.weights.shape:
                raise ValueError("detector_mask must be boolean with one entry per bright-field pixel.")
            weights = layout.weights * mask
            scale = cp.sum(weights * engine._cache["aperture_k_1d"] ** 4, dtype=cp.float64)
            if float(scale) <= 0:
                raise ValueError("Select at least one illuminated bright-field pixel.")
            layout = replace(layout, weights=weights, scale=scale, scale_value=float(scale))
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
            # batches come in the engine's pixel order; the resident spectrum is walked in compact detector tiles
            (layout.order if column_spectra is None else cp.arange(nbf, dtype=cp.int32), cache["kx_bf"],
             cache["ky_bf"], layout.weights, np.int32(nbf), *optics_scalars[:4], coefficients, pixel),
        )
        aliases = []
        for group in layout.groups:
            qr, qc = group.qr, group.qc
            if transform is not None:
                qr, qc = (
                    cp.ascontiguousarray(np.float32(transform[0, 0]) * qr + np.float32(transform[0, 1]) * qc),
                    cp.ascontiguousarray(np.float32(transform[1, 0]) * qr + np.float32(transform[1, 1]) * qc),
                )
            aliases.append((qr, qc))
        widths = [(2 * group.count if complex_object else group.count) + 1 for group in layout.groups]
        if column_spectra is None:
            batches = [(pixel, engine.G_qk.reshape(nbf, -1))]
        else:
            _check_batched_memory(engine, layout, widths)
            batches = _pixel_batches(pixel, column_spectra())
        sums = [None] * len(layout.groups)
        for batch_pixel, spectrum in batches:
            for index, (group, (qr, qc), width) in enumerate(zip(layout.groups, aliases, widths)):
                size = group.columns.size
                augmented = cp.empty((size, width, width), cp.complex128)
                normal_equations, parts = _normal_equations(group.count, complex_object)
                normal_equations(
                    (-(-size // _WARPS), parts),
                    (32 * _WARPS,),
                    (spectrum, np.int32(spectrum.shape[1]), group.columns, np.int32(size), qr, qc,
                     batch_pixel, np.int32(batch_pixel.shape[0]), *optics_scalars, coefficients, augmented),
                )
                if sums[index] is None:
                    sums[index] = augmented
                else:
                    sums[index] += augmented
        groups = []
        power = cp.zeros((), cp.float64)
        for group, augmented, width in zip(layout.groups, sums, widths):
            count, dim = group.count, width - 1
            # G enters unscaled so V stays float32; the dose divides afterwards.
            gram = cp.ascontiguousarray(augmented[:, :dim, :dim])
            rhs = augmented[:, :dim, dim] / layout.dose
            power += augmented[:, dim, dim].real.sum() / layout.dose**2
            if not count:
                continue
            prior = cp.concatenate((group.prior, group.prior), axis=1) if complex_object else group.prior
            groups.append(_Group(group.rows, group.cols, gram, rhs, prior))
        return cls(layout.shape, factor, groups, complex_object, layout.scale, power, layout.dose)

    def phase_error(self, phase) -> float:
        """Score a phase against these cached measurements, excluding DC.

        This is relative squared error in the linear weak-phase model, not
        a Poisson likelihood or a nonlinear exit-wave diffraction calculation.
        A no-modulation (zero-phase) prediction scores one.
        """
        phase = cp.asarray(phase)
        if self.complex_object or phase.shape != self.shape or not cp.isrealobj(phase):
            raise ValueError("Use a real phase on the prepared phase-only output grid.")
        if not bool(cp.isfinite(phase).all()) or float(self.signal_power) <= 0:
            raise ValueError("Use finite phase and nonzero measured non-DC signal.")
        spectrum = cp.fft.fft2(phase) / self.factor**2
        residual = self.signal_power.copy()
        for group in self.groups:
            coefficients = spectrum[group.rows, group.cols].astype(cp.complex128)
            prediction = (group.gram @ coefficients[..., None])[..., 0]
            residual += cp.sum((coefficients.conj() * prediction).real
                               - 2 * (coefficients.conj() * group.rhs).real)
        return float(cp.maximum(residual, 0) / self.signal_power)

    def solve(
        self,
        phase_damp: float,
        amplitude_damp: float | None,
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
                penalty[:, :count] *= amplitude_damp
                penalty[:, count:] *= phase_damp
                diagonal = cp.arange(matrix.shape[-1])
                matrix[:, diagonal, diagonal] += penalty
                coefficients = cp.linalg.solve(matrix, group.rhs[..., None])[
                    ..., 0
                ].astype(cp.complex64)
            else:
                # (G + phase_damp scale P) x = b, float64 Cholesky per frequency, as DragWindow solves
                nfrequency, dim = group.rhs.shape
                coefficients = cp.empty((nfrequency, dim), cp.complex64)
                _cholesky(dim)(
                    (-(-nfrequency // 64),), (64,),
                    (group.gram, cp.ascontiguousarray(group.rhs),
                     cp.ascontiguousarray(group.prior * self.scale * phase_damp), np.int32(nfrequency), coefficients),
                )
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


def _pixel_batches(pixel, column_spectra):
    """Each batch of an encoded scan as (its pixels' k-only terms, its spectra as rows): the terms' detector row of
    G becomes the row within the batch."""
    for first, stop, spectra in column_spectra:
        batch = cp.ascontiguousarray(pixel[first:stop])
        batch[:, 1, 2] = cp.arange(stop - first, dtype=cp.int32).view(cp.float32)
        yield batch, spectra.reshape(stop - first, -1)


def _check_batched_memory(engine, layout, widths) -> None:
    """Raise ``DeviceMemoryError`` when the device cannot hold the equations of a batched preparation: every
    frequency's system twice (the sum and one batch's) and one batch of decoded columns and their spectra."""
    from quantem.gpu.direct_ptycho.cuda.streamed import COLUMN_BATCH, DeviceMemoryError, free_device_bytes

    _, rows, cols = engine._bf_grid()
    systems = sum(group.columns.size * width * width * 16 for group, width in zip(layout.groups, widths))
    needed = 2 * systems + COLUMN_BATCH * rows * ((cols + 2) * 4 + cols * 2)
    free_bytes = free_device_bytes()
    if needed > free_bytes * .8:
        raise DeviceMemoryError(f"The damped {layout.shape[0]} x {layout.shape[1]} direct ptychography phase",
                                needed / .8, free_bytes)


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
                            detector_weighting=detector_weighting)
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

    def solve(self, value: float, phase_damp: float, amplitude_damp: float | None,
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
                penalty = penalty * cp.concatenate((cp.full(count, amplitude_damp), cp.full(count, phase_damp)))
            else:
                if self.complex_object:
                    gram, rhs, penalty = gram[:, count:, count:], rhs[:, count:], penalty[:, count:]
                penalty = penalty * phase_damp
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
        raise ValueError('Direct inversion diverged; increase phase_damp or amplitude_damp, or use method="phase".')
    diagnostics = {
        "model": "T = exp(eta + i phi), linearized about unity",
        "normalization": "zero mean phase and log-amplitude; relative amplitude only",
        "frequency_policy": "supported aliases fitted jointly; DC family and even-grid Nyquist axes excluded",
        "incident_scale": incident_scale,
    }
    return wave, diagnostics
