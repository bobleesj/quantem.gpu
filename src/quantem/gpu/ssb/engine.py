"""
GPU kernels for fast SSB ptychography using CuPy.

All kernels are optimized for 256x256 scan size with fused operations
for maximum throughput.
"""

import math
from contextlib import contextmanager
import numpy as np
import cupy as cp

# Mean phase kernel: avoids materializing a full phase buffer.
_mean_phase_kernel = cp.RawKernel(r'''
extern "C" __global__
void mean_phase(
    const float2* __restrict__ corrected,
    float* __restrict__ out,
    int num_bf,
    int ny,
    int nx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny * nx;
    if (idx >= total) return;
    int y = idx / nx;
    int x = idx - y * nx;
    size_t base = (size_t)y * (size_t)nx + (size_t)x;
    size_t plane = (size_t)ny * (size_t)nx;
    float sum = 0.0f;
    for (int b = 0; b < num_bf; ++b) {
        size_t off = (size_t)b * plane + base;
        float2 v = corrected[off];
        sum += atan2f(v.y, v.x);
    }
    out[base] = sum / (float)num_bf;
}
''', 'mean_phase')

# Fused kernel: sum + sumsq of per-BF angles in a single pass.
# Used by reconstruct_with_loss to avoid running the correction pipeline twice.
_sum_sumsq_phase_kernel = cp.RawKernel(r'''
extern "C" __global__
void sum_sumsq_phase(
    const float2* __restrict__ corrected,
    float* __restrict__ sum_out,
    float* __restrict__ sumsq_out,
    int num_bf,
    int ny,
    int nx
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = ny * nx;
    if (idx >= total) return;
    size_t plane = (size_t)ny * (size_t)nx;
    float s = 0.0f, sq = 0.0f;
    for (int b = 0; b < num_bf; ++b) {
        float2 v = corrected[(size_t)b * plane + (size_t)idx];
        float a = atan2f(v.y, v.x);
        s += a;
        sq += a * a;
    }
    sum_out[idx] = s;
    sumsq_out[idx] = sq;
}
''', 'sum_sumsq_phase')

_variance_from_sums_batch_kernel = cp.RawKernel(r'''
extern "C" __global__
void variance_from_sums_batch(
    const float* __restrict__ sum,
    const float* __restrict__ sumsq,
    float* __restrict__ out,
    int num_bf,
    int ny,
    int nx,
    int batch
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int plane = ny * nx;
    int total = batch * plane;
    if (idx >= total) return;
    float mean = sum[idx] / (float)num_bf;
    float mean_sq = sumsq[idx] / (float)num_bf;
    out[idx] = mean_sq - mean * mean;
}
''', 'variance_from_sums_batch')

def _choose_reduce_block(groups: int) -> int:
    if groups <= 32:
        return 32
    if groups <= 64:
        return 64
    if groups <= 128:
        return 128
    return 256

_reduce_group_sums_batch_kernel = cp.RawKernel(r'''
extern "C" __global__
void reduce_group_sums_batch(
    const float* __restrict__ sum_partial,
    const float* __restrict__ sumsq_partial,
    float* __restrict__ sum,
    float* __restrict__ sumsq,
    int groups,
    int ny,
    int nx,
    int batch
) {
    int x = blockIdx.x;
    int y = blockIdx.y;
    int b = blockIdx.z;
    if (x >= nx || y >= ny || b >= batch) return;

    int tid = threadIdx.x;
    int plane = ny * nx;
    int base = y * nx + x;
    int group_base = b * groups;
    float local_sum = 0.0f;
    float local_sumsq = 0.0f;

    for (int g = tid; g < groups; g += blockDim.x) {
        int idx = (group_base + g) * plane + base;
        local_sum += sum_partial[idx];
        local_sumsq += sumsq_partial[idx];
    }

    unsigned mask = 0xffffffffu;
    for (int offset = 16; offset > 0; offset >>= 1) {
        local_sum += __shfl_down_sync(mask, local_sum, offset);
        local_sumsq += __shfl_down_sync(mask, local_sumsq, offset);
    }

    int lane = tid & 31;
    int warp = tid >> 5;
    if (blockDim.x <= 32) {
        if (lane == 0) {
            int out_idx = b * plane + base;
            sum[out_idx] = local_sum;
            sumsq[out_idx] = local_sumsq;
        }
        return;
    }
    __shared__ float warp_sum[8];
    __shared__ float warp_sumsq[8];
    if (lane == 0) {
        warp_sum[warp] = local_sum;
        warp_sumsq[warp] = local_sumsq;
    }
    __syncthreads();

    if (warp == 0) {
        int num_warps = (blockDim.x + 31) >> 5;
        float sum_val = (lane < num_warps) ? warp_sum[lane] : 0.0f;
        float sumsq_val = (lane < num_warps) ? warp_sumsq[lane] : 0.0f;
        for (int offset = 16; offset > 0; offset >>= 1) {
            sum_val += __shfl_down_sync(mask, sum_val, offset);
            sumsq_val += __shfl_down_sync(mask, sumsq_val, offset);
        }
        if (lane == 0) {
            int out_idx = b * plane + base;
            sum[out_idx] = sum_val;
            sumsq[out_idx] = sumsq_val;
        }
    }
}
''', 'reduce_group_sums_batch')

# pk kernel for precomputing pk values (used in fused FFT paths)
_pk_kernel = cp.ElementwiseKernel(
    in_params="""
        float32 alpha_k2, float32 cos2phi_k, float32 sin2phi_k, float32 aperture_k,
        float32 C10, float32 C12, float32 cos2phi12, float32 sin2phi12, float32 factor
        """,
    out_params="complex64 pk",
    operation="""
        float cos_term_k = __fmaf_rn(cos2phi_k, cos2phi12, sin2phi_k * sin2phi12);
        float chi_k = factor * alpha_k2 * __fmaf_rn(C12, cos_term_k, C10);
        float sin_k, cos_k;
        __sincosf(chi_k, &sin_k, &cos_k);
        pk = thrust::complex<float>(aperture_k * cos_k, -aperture_k * sin_k);
    """,
    name="pk_kernel",
)

# Full-aberration pk kernel.  Computes pk = aperture(k) * exp(-i·χ(k)) for all
# 14 Krivanek aberrations up to 5th order.  Used by SSBEngine.reconstruct_full
# for explorer manual higher-order slider drag.  The legacy _pk_kernel above is
# NOT modified and continues to serve the optimization hot path.
#
# mags[14] and angles[14] layout matches aberration.ABERRATION_INDICES:
#   0: C10, 1: C12/phi12,  2: C21/phi21, 3: C23/phi23,
#   4: C30, 5: C32/phi32,  6: C34/phi34,
#   7: C41/phi41, 8: C43/phi43, 9: C45/phi45,
#   10: C50, 11: C52/phi52, 12: C54/phi54, 13: C56/phi56
#
# Input kx, ky are per-BF-pixel reciprocal-space coordinates (1/m); aperture_k
# is the precomputed soft-edge mask; wavelength and kfactor=2π/wavelength are
# passed in to avoid recomputation.
# Fast variant using host-precomputed Chebyshev-ready arrays (see
# fft_common.pack_aberration_coefs).  Replaces 14 cosf per BF pixel with
# ~20 FMAs via direct cos(φ)/sin(φ) from (kx, ky) + Chebyshev recurrence.
_pk_kernel_full = cp.ElementwiseKernel(
    in_params="""
        float32 kx, float32 ky, float32 aperture_k,
        float32 wavelength, float32 kfactor,
        raw float32 abr_mag_scaled, raw float32 abr_cm, raw float32 abr_sm
        """,
    out_params="complex64 pk",
    operation="""
        float r2 = kx * kx + ky * ky;
        float inv_r = (r2 > 1e-30f) ? rsqrtf(r2) : 0.0f;
        float r = r2 * inv_r;
        float alpha = wavelength * r;
        float c1 = kx * inv_r;
        float s1 = ky * inv_r;
        float c2 = fmaf(2.0f * c1, c1, -1.0f);
        float s2 = 2.0f * s1 * c1;
        float c3 = fmaf(c1, c2, -s1 * s2);
        float s3 = fmaf(s1, c2,  c1 * s2);
        float c4 = fmaf(c2, c2, -s2 * s2);
        float s4 = 2.0f * s2 * c2;
        float c5 = fmaf(c1, c4, -s1 * s4);
        float s5 = fmaf(s1, c4,  c1 * s4);
        float c6 = fmaf(c2, c4, -s2 * s4);
        float s6 = fmaf(s2, c4,  c2 * s4);

        float a2 = alpha * alpha;
        float a3 = a2 * alpha;
        float a4 = a2 * a2;
        float a5 = a4 * alpha;
        float a6 = a3 * a3;

        float chi = 0.0f;
        chi = fmaf(a2 * abr_mag_scaled[0], 1.0f,
              fmaf(a2 * abr_mag_scaled[1], fmaf(c2, abr_cm[1],  s2 * abr_sm[1]),  chi));
        chi = fmaf(a3 * abr_mag_scaled[2], fmaf(c1, abr_cm[2],  s1 * abr_sm[2]),
              fmaf(a3 * abr_mag_scaled[3], fmaf(c3, abr_cm[3],  s3 * abr_sm[3]),  chi));
        chi = fmaf(a4 * abr_mag_scaled[4], 1.0f,
              fmaf(a4 * abr_mag_scaled[5], fmaf(c2, abr_cm[5],  s2 * abr_sm[5]),
              fmaf(a4 * abr_mag_scaled[6], fmaf(c4, abr_cm[6],  s4 * abr_sm[6]),  chi)));
        chi = fmaf(a5 * abr_mag_scaled[7], fmaf(c1, abr_cm[7],  s1 * abr_sm[7]),
              fmaf(a5 * abr_mag_scaled[8], fmaf(c3, abr_cm[8],  s3 * abr_sm[8]),
              fmaf(a5 * abr_mag_scaled[9], fmaf(c5, abr_cm[9],  s5 * abr_sm[9]),  chi)));
        chi = fmaf(a6 * abr_mag_scaled[10], 1.0f,
              fmaf(a6 * abr_mag_scaled[11], fmaf(c2, abr_cm[11], s2 * abr_sm[11]),
              fmaf(a6 * abr_mag_scaled[12], fmaf(c4, abr_cm[12], s4 * abr_sm[12]),
              fmaf(a6 * abr_mag_scaled[13], fmaf(c6, abr_cm[13], s6 * abr_sm[13]), chi))));
        chi *= kfactor;

        float sin_k, cos_k;
        __sincosf(chi, &sin_k, &cos_k);
        pk = thrust::complex<float>(aperture_k * cos_k, -aperture_k * sin_k);
    """,
    name="pk_kernel_full",
)

class SSBEngine:
    """
    CuPy-accelerated SSB computation with fused CUDA kernels (256x256 only).

    All computation is done on GPU using CuPy arrays.
    Pre-computes rotation-dependent quantities and caches them for
    fast aberration-dependent gamma factor computation.
    """

    def __init__(
        self,
        G_qk: cp.ndarray,
        bf_inds_row: cp.ndarray,
        bf_inds_col: cp.ndarray,
        bf_center: tuple[float, float] | None,
        dc_value: complex,
        gpts: tuple[int, int],
        sampling: tuple[float, float],
        q_row: cp.ndarray,
        q_col: cp.ndarray,
        wavelength: float,
        semiangle_cutoff: float,
        angular_sampling: tuple[float, float],
    ):
        self.G_qk = G_qk
        self.bf_inds_row = bf_inds_row
        self.bf_inds_col = bf_inds_col
        if bf_center is None:
            bf_center = ((gpts[0] - 1) * 0.5, (gpts[1] - 1) * 0.5)
        self.bf_center = bf_center
        self._dc_value_host = complex(dc_value)
        self.gpts = gpts
        self.sampling = sampling
        self.q_row = q_row
        self.q_col = q_col
        self.wavelength = wavelength
        self.semiangle_cutoff = semiangle_cutoff
        self.angular_sampling = angular_sampling
        # Pre-computed cache
        self._cached_rotation_rad = None
        self._cache = {}
        # Pre-compute factor for aberration phase
        self._factor = float(math.pi / wavelength)  # = (2*pi/wl) * 0.5
        # Work buffers (sized in cache_rotation)
        self._result_buffer = None
        self._corrected_buffer = None
        self._mean_phase_buffer = None
        self._variance_buffer = None
        self._pk_buffer = None
        self._sum_buffer = None
        self._sumsq_buffer = None
        self._partial_sum = None
        self._partial_sumsq = None
        # Custom FFT (initialized in cache_rotation)
        self._custom_fft = None
        self._colvar_group = 32
        # Batch caches
        self._batch_cache: dict[int, dict[str, object]] = {}
        self._batch_chunk_cache: dict[tuple[int, int, int], dict[str, object]] = {}
        self._streaming_cache: dict[tuple[int, int, int], dict[str, object]] = {}
        self._preferred_chunk_bf = 0
        # Empirically tuned on Samsung 512x512 (Blackwell, 2026-04-08):
        # 512 is the sweet spot where the staging buffer
        # (4 × 512 × 512 × 512 × 8 = 2 GB) fits in L2 cache. Sweeping
        # 128 → 2048 showed 512 is fastest AND lowest-peak simultaneously:
        #   stream_bf  peak_GB  optimize_s  refine_s
        #         128      2.0        3.64     20.24
        #         256      3.1        3.61     19.99
        #         512      7.4        3.56     19.83   ← fastest both
        #        1024     16.0        3.58     19.98
        #        2048     61.0        3.59     22.69   ← L2 pressure
        # The old "auto from free_bytes * 0.4" default picked ~1560 BF on
        # 96 GB card, which was simultaneously slower AND much larger peak.
        # 512 works for both L40S 48 GB and RTX PRO 6000 96 GB.
        self._stream_bf = 512

    @property
    def num_bf(self) -> int:
        """Number of bright-field pixels."""
        return int(self._cache["num_bf"])

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector grid shape (n_k_row, n_k_col)."""
        return int(self._cache["ny"]), int(self._cache["nx"])

    def clear_batch_caches(self) -> None:
        """Release batch and chunk caches to free GPU VRAM."""
        self._batch_cache.clear()
        self._batch_chunk_cache.clear()
        self._streaming_cache.clear()

    @contextmanager
    def use_bf_subset(self, indices: "cp.ndarray"):
        """Temporarily swap the accelerator to a BF subset for fast optimize.

        The Optuna trial loop doesn't need full-precision variance loss - it
        only needs a landscape that points toward the right answer. Running
        trials on a 1500-2000 BF subset of the ~9000 full set is ~5× faster
        while preserving the global minimum (refine() on the full set then
        closes the precision gap).

        Parameters
        ----------
        indices : cp.ndarray (int)
            BF pixel indices to use during the `with` block. Should be a
            uniform spatial sample of the full BF disk.

        Usage
        -----
        >>> indices = cp.arange(0, accel.num_bf, 5)  # every 5th BF
        >>> with accel.use_bf_subset(indices):
        ...     best = batch_optimize(accel, ...)   # runs on subset
        >>> accel.refine(...)                        # runs on full BF

        State restored on exit even if an exception is raised.
        """
        indices = cp.asarray(indices, dtype=cp.int32)
        # Snapshot everything we're about to overwrite
        saved_G_qk = self.G_qk
        saved_bf_inds_row = self.bf_inds_row
        saved_bf_inds_col = self.bf_inds_col
        saved_cache = dict(self._cache)
        saved_pk_buffer = self._pk_buffer
        # Fancy-index a view of G_qk and bf_inds. G_qk[indices] is a copy,
        # ~2 GB for 2000 BFs on 512x512 float2 - one-time cost.
        self.G_qk = self.G_qk[indices]
        self.bf_inds_row = self.bf_inds_row[indices]
        self.bf_inds_col = self.bf_inds_col[indices]
        subset_num_bf = int(indices.size)
        # Slice the per-BF cached arrays
        sub_cache = dict(saved_cache)
        for k in ("kx_bf", "ky_bf", "alpha_k2_1d", "cos2phi_k_1d",
                  "sin2phi_k_1d", "aperture_k_1d"):
            if k in sub_cache:
                sub_cache[k] = cp.ascontiguousarray(saved_cache[k][indices])
        sub_cache["num_bf"] = subset_num_bf
        self._cache = sub_cache
        # The pk buffer is sized per num_bf; subset needs its own
        self._pk_buffer = cp.empty((subset_num_bf,), dtype=cp.complex64)
        # Streaming buffers keyed by (batch, stream_bf, num_bf) - clear so
        # they're rebuilt for the subset and then for the full set again.
        self.clear_batch_caches()
        try:
            yield
        finally:
            self.G_qk = saved_G_qk
            self.bf_inds_row = saved_bf_inds_row
            self.bf_inds_col = saved_bf_inds_col
            self._cache = saved_cache
            self._pk_buffer = saved_pk_buffer
            self.clear_batch_caches()
            cp.get_default_memory_pool().free_all_blocks()

    def cache_rotation(
        self,
        rotation_angle_rad: float,
        force: bool = False,
    ) -> None:
        """Pre-compute all rotation-dependent quantities."""
        if self._cached_rotation_rad == rotation_angle_rad and not force:
            return
        # Compute detector k-space coordinates centered on the BF disk.
        recip_y = 1.0 / (self.sampling[0] * self.gpts[0])
        recip_x = 1.0 / (self.sampling[1] * self.gpts[1])
        iy = cp.arange(self.gpts[0], dtype=cp.float32) - self.bf_center[0]
        ix = cp.arange(self.gpts[1], dtype=cp.float32) - self.bf_center[1]
        kxa = iy[:, None] * recip_y
        kya = ix[None, :] * recip_x
        # Passive rotation
        if rotation_angle_rad is not None:
            cos_a = math.cos(-rotation_angle_rad)
            sin_a = math.sin(-rotation_angle_rad)
            kxa_rot = kxa * cos_a + kya * sin_a
            kya_rot = -kxa * sin_a + kya * cos_a
            kxa, kya = kxa_rot, kya_rot
        # Extract BF pixel coordinates
        kx_bf = kxa[self.bf_inds_row, self.bf_inds_col]
        ky_bf = kya[self.bf_inds_row, self.bf_inds_col]
        num_bf = int(kx_bf.shape[0])
        ny, nx = self.q_row.shape
        # Soft aperture (constant, doesn't depend on aberrations)
        semiangle_rad = self.semiangle_cutoff * 1e-3
        ang_y, ang_x = self.angular_sampling
        # Extract 1D q-space coordinates (separable from meshgrid)
        qx_1d = cp.ascontiguousarray(self.q_row[:, 0].astype(cp.float32))
        qy_1d = cp.ascontiguousarray(self.q_col[0, :].astype(cp.float32))
        # Probe at k positions
        k_mag = cp.sqrt(kx_bf**2 + ky_bf**2)
        phi_k = cp.arctan2(ky_bf, kx_bf)
        alpha_k = k_mag * self.wavelength
        alpha_k2 = alpha_k**2
        cos_phi_k = cp.cos(phi_k)
        sin_phi_k = cp.sin(phi_k)
        denom_k = cp.sqrt(
            (cos_phi_k * ang_y * 1e-3)**2 +
            (sin_phi_k * ang_x * 1e-3)**2
        )
        aperture_k = cp.clip((semiangle_rad - alpha_k) / denom_k + 0.5, 0, 1).astype(cp.float32)
        cos2phi_k = cos_phi_k * cos_phi_k - sin_phi_k * sin_phi_k
        sin2phi_k = 2.0 * sin_phi_k * cos_phi_k
        del cos_phi_k, sin_phi_k, denom_k, phi_k
        self._cache = {
            "num_bf": num_bf,
            "ny": ny,
            "nx": nx,
            "alpha_k2_1d": cp.ascontiguousarray(alpha_k2.astype(cp.float32)),
            "cos2phi_k_1d": cp.ascontiguousarray(cos2phi_k.astype(cp.float32)),
            "sin2phi_k_1d": cp.ascontiguousarray(sin2phi_k.astype(cp.float32)),
            "aperture_k_1d": cp.ascontiguousarray(aperture_k),
            "kx_bf": cp.ascontiguousarray(kx_bf.astype(cp.float32)),
            "ky_bf": cp.ascontiguousarray(ky_bf.astype(cp.float32)),
            "qx_1d": qx_1d,
            "qy_1d": qy_1d,
            "wavelength": float(self.wavelength),
            "semiangle_rad": float(semiangle_rad),
            "ang_y_rad": float(ang_y * 1e-3),
            "ang_x_rad": float(ang_x * 1e-3),
        }
        self._cached_rotation_rad = rotation_angle_rad
        # Initialize custom FFT based on scan size
        if self._custom_fft is None:
            if ny == 512 and nx == 512:
                from .fft512 import get_custom_fft_512
                self._custom_fft = get_custom_fft_512()
            elif ny == 256 and nx == 256:
                from .fft256 import get_custom_fft_256
                self._custom_fft = get_custom_fft_256()
            else:
                raise ValueError(f"Custom FFT only supports 256x256 and 512x512 scan, got {ny}x{nx}")
            self._colvar_group = int(self._custom_fft._colvar_group)
        # Work buffers. _result_buffer is (num_bf, ny, nx) complex64 and
        # only used by the reconstruct path - optimize/refine use separate
        # `staging` buffers from _get_streaming_buffers. On small scans
        # (e.g. Steph 256x256, ~600 MB) we pre-allocate it at engine init
        # for a faster reconstruct call path. On large scans (e.g. Samsung
        # 512x512, ~19 GB) we defer the allocation - pre-allocating blows
        # the L40S 48 GB budget during optimize for no reason, since the
        # chunked reconstruct path will allocate a small chunk buffer
        # instead.
        shape = (num_bf, ny, nx)
        full_bytes = num_bf * ny * nx * 8
        if full_bytes < 6 * 1024 ** 3:
            self._result_buffer = cp.empty(shape, dtype=cp.complex64)
        else:
            self._result_buffer = None
        self._corrected_buffer = None
        self._mean_phase_buffer = None
        self._variance_buffer = cp.empty((ny, nx), dtype=cp.float32)
        self._pk_buffer = cp.empty((num_bf,), dtype=cp.complex64)
        self._sum_buffer = cp.empty((ny, nx), dtype=cp.float32)
        self._sumsq_buffer = cp.empty((ny, nx), dtype=cp.float32)
        cp.get_default_memory_pool().free_all_blocks()

    # =====================================================================
    #  Reconstruction
    # =====================================================================

    def _run_correction_pipeline(self, C10: float, C12: float, phi12: float) -> None:
        """Run the aberration-correction pipeline, populating _corrected_buffer."""
        c = self._cache
        num_bf = int(c["num_bf"])
        ny = int(c["ny"])
        nx = int(c["nx"])
        shape = (num_bf, ny, nx)
        if self._result_buffer is None or self._result_buffer.shape != shape:
            self._result_buffer = cp.empty(shape, dtype=cp.complex64)
        if self._pk_buffer is None or self._pk_buffer.shape != (num_bf,):
            self._pk_buffer = cp.empty((num_bf,), dtype=cp.complex64)
        cos2phi12 = math.cos(2.0 * phi12)
        sin2phi12 = math.sin(2.0 * phi12)
        _pk_kernel(
            c["alpha_k2_1d"],
            c["cos2phi_k_1d"],
            c["sin2phi_k_1d"],
            c["aperture_k_1d"],
            cp.float32(C10),
            cp.float32(C12),
            cp.float32(cos2phi12),
            cp.float32(sin2phi12),
            cp.float32(self._factor),
            self._pk_buffer,
        )
        self._custom_fft.ifft2_inplace_fused_pk(
            self._result_buffer,
            self.G_qk,
            self._cache,
            self._pk_buffer,
            C10,
            C12,
            cos2phi12,
            sin2phi12,
            self._factor,
            self._dc_value_host,
        )
        self._corrected_buffer = self._result_buffer

    def _run_correction_pipeline_chunked(
        self, C10: float, C12: float, phi12: float, chunk_bf: int,
    ) -> "cp.ndarray":
        """Chunked reconstruct: processes BF pixels in groups of ``chunk_bf``,
        accumulating the running mean. Avoids ever materializing the full
        ``(num_bf, ny, nx)`` result buffer (~19 GB for 9070 BF × 512 × 512).

        Peak transient buffer is ``chunk_bf × ny × nx × 8`` bytes, e.g.
        ``chunk_bf=1024`` at 512×512 = ~2 GB. For Samsung 512 this cuts
        reconstruct peak by 18 GB with negligible speed cost (~1 ms of extra
        kernel-launch overhead across the chunks).

        Returns the mean complex object directly, shape (ny, nx).
        """
        c = self._cache
        num_bf = int(c["num_bf"])
        n_row = int(c["ny"])
        n_col = int(c["nx"])
        if chunk_bf >= num_bf:
            # Not worth chunking - use the full path.
            self._run_correction_pipeline(C10, C12, phi12)
            return self._corrected_buffer.mean(axis=0)

        # Compute the full pk buffer once (small: num_bf × 8 bytes).
        if self._pk_buffer is None or self._pk_buffer.shape != (num_bf,):
            self._pk_buffer = cp.empty((num_bf,), dtype=cp.complex64)
        cos2phi12 = math.cos(2.0 * phi12)
        sin2phi12 = math.sin(2.0 * phi12)
        _pk_kernel(
            c["alpha_k2_1d"],
            c["cos2phi_k_1d"],
            c["sin2phi_k_1d"],
            c["aperture_k_1d"],
            cp.float32(C10),
            cp.float32(C12),
            cp.float32(cos2phi12),
            cp.float32(sin2phi12),
            cp.float32(self._factor),
            self._pk_buffer,
        )

        # Small work buffer reused across chunks. Release the huge one if cached.
        if self._result_buffer is not None and self._result_buffer.shape[0] > chunk_bf:
            self._result_buffer = None
        chunk_shape = (chunk_bf, n_row, n_col)
        if self._result_buffer is None or self._result_buffer.shape != chunk_shape:
            self._result_buffer = cp.empty(chunk_shape, dtype=cp.complex64)

        accumulator = cp.zeros((n_row, n_col), dtype=cp.complex64)
        # Slice geometry arrays per chunk and call the fused kernel on each
        # slice. We pass a sub-cache that shares qx/qy/wavelength/angles with
        # the main cache but slices the BF geometry arrays to this chunk.
        for bf_start in range(0, num_bf, chunk_bf):
            bf_end = min(bf_start + chunk_bf, num_bf)
            chunk = bf_end - bf_start
            sub_cache = dict(c)
            sub_cache["kx_bf"] = c["kx_bf"][bf_start:bf_end]
            sub_cache["ky_bf"] = c["ky_bf"][bf_start:bf_end]
            chunk_buf = self._result_buffer[:chunk]
            self._custom_fft.ifft2_inplace_fused_pk(
                chunk_buf,
                self.G_qk[bf_start:bf_end],
                sub_cache,
                self._pk_buffer[bf_start:bf_end],
                C10,
                C12,
                cos2phi12,
                sin2phi12,
                self._factor,
                self._dc_value_host,
            )
            accumulator += chunk_buf.sum(axis=0)
        return accumulator / num_bf

    def reconstruct_object(self, C10: float, C12: float, phi12: float) -> "cp.ndarray":
        """Reconstruct complex transmission function.

        Returns the mean of the corrected complex BF images. For large scans
        (num_bf × scan_row × scan_col × 8 bytes > 6 GB), uses a chunked path
        that keeps peak transient VRAM at ~4 GB instead of the full ~19 GB
        staging buffer.

        Returns
        -------
        cp.ndarray
            Complex object (scan_row, scan_col), complex64, stays on GPU.
        """
        c = self._cache
        num_bf = int(c["num_bf"])
        n_row = int(c["ny"])
        n_col = int(c["nx"])
        full_bytes = num_bf * n_row * n_col * 8
        if full_bytes > 6 * 1024 ** 3:
            # Target ~2 GB chunk transient. Speed is flat from 64..9070 BF
            # per chunk on Blackwell (kernel-launch overhead negligible) so
            # we pick the smaller chunk for maximum L40S headroom. Parity
            # is at the float32 summation-order floor (~1e-5 max|Δ|).
            chunk_bf = max(1, (2 * 1024 ** 3) // (n_row * n_col * 8))
            return self._run_correction_pipeline_chunked(C10, C12, phi12, chunk_bf)
        self._run_correction_pipeline(C10, C12, phi12)
        return self._corrected_buffer.mean(axis=0)

    def reconstruct(self, C10: float, C12: float, phi12: float) -> "cp.ndarray":
        """Reconstruct mean phase image.

        Computes ``mean_bf(angle(corrected[b]))`` - the average of the
        per-BF-pixel phase images. For large scans (>6 GB full staging
        buffer) we chunk on the BF axis to stay under the L40S budget;
        mathematically equivalent via sum-then-divide.

        Returns
        -------
        cp.ndarray
            Mean phase image (ny, nx), stays on GPU.
        """
        c = self._cache
        num_bf = int(c["num_bf"])
        ny = int(c["ny"])
        nx = int(c["nx"])
        full_bytes = num_bf * ny * nx * 8
        if full_bytes > 6 * 1024 ** 3:
            return self._reconstruct_chunked(C10, C12, phi12)

        if self._mean_phase_buffer is None:
            self._mean_phase_buffer = cp.empty((ny, nx), dtype=cp.float32)
        self._run_correction_pipeline(C10, C12, phi12)
        total = int(ny * nx)
        block = 256
        grid = (total + block - 1) // block
        _mean_phase_kernel(
            (grid,),
            (block,),
            (
                self._corrected_buffer,
                self._mean_phase_buffer,
                np.int32(num_bf),
                np.int32(ny),
                np.int32(nx),
            ),
        )
        return self._mean_phase_buffer

    def _reconstruct_chunked(
        self, C10: float, C12: float, phi12: float,
    ) -> "cp.ndarray":
        """Chunked mean-phase reconstruction for large scans.

        Uses the fused col-FFT + phase accumulate kernel, shared with
        ``_reconstruct_with_loss_chunked`` so both paths produce
        bit-identical phase images.
        """
        return self._fused_chunked_core(C10, C12, phi12, compute_loss=False)

    # =====================================================================
    #  Full-aberration reconstruct (14 Krivanek coefficients)
    # =====================================================================

    def _run_correction_pipeline_full(
        self, mags_m: cp.ndarray, angles_rad: cp.ndarray,
    ) -> None:
        """Full-aberration correction pipeline.  Mirrors
        :meth:`_run_correction_pipeline` but evaluates ``chi_full`` with all
        14 Krivanek coefficients instead of the 2-term C10/C12 formula.
        Result lands in ``self._corrected_buffer``.
        """
        from .fft_common import pack_aberration_coefs
        c = self._cache
        num_bf = int(c["num_bf"])
        ny = int(c["ny"])
        nx = int(c["nx"])
        shape = (num_bf, ny, nx)
        if self._result_buffer is None or self._result_buffer.shape != shape:
            self._result_buffer = cp.empty(shape, dtype=cp.complex64)
        if self._pk_buffer is None or self._pk_buffer.shape != (num_bf,):
            self._pk_buffer = cp.empty((num_bf,), dtype=cp.complex64)
        # Host-precompute the 14-coef Chebyshev arrays once per pipeline call.
        abr_mag_scaled, abr_cm, abr_sm = pack_aberration_coefs(mags_m, angles_rad)
        kfactor = cp.float32(2.0 * math.pi / c["wavelength"])
        _pk_kernel_full(
            c["kx_bf"], c["ky_bf"], c["aperture_k_1d"],
            cp.float32(c["wavelength"]), kfactor,
            abr_mag_scaled, abr_cm, abr_sm, self._pk_buffer,
        )
        self._custom_fft.ifft2_inplace_fused_pk_full(
            self._result_buffer,
            self.G_qk,
            self._cache,
            self._pk_buffer,
            mags_m, angles_rad,
            self._dc_value_host,
        )
        self._corrected_buffer = self._result_buffer

    def reconstruct_full(
        self, mags_m: cp.ndarray, angles_rad: cp.ndarray,
    ) -> "cp.ndarray":
        """Reconstruct mean phase with all 14 Krivanek aberrations.

        Parameters
        ----------
        mags_m : cp.ndarray
            Aberration magnitudes (meters), shape (14,), float32.
            Order: C10, C12, C21, C23, C30, C32, C34, C41, C43, C45,
            C50, C52, C54, C56.
        angles_rad : cp.ndarray
            Orientation angles (radians), shape (14,), float32.

        Returns
        -------
        cp.ndarray
            Mean phase image (ny, nx), float32, stays on GPU.

        Notes
        -----
        For scans with full staging buffer >6 GB (e.g. Samsung 512²), falls
        back to a chunked BF-axis loop.  The col-accumulate optimization used
        in the legacy chunked path is 2-term-only, so this path always
        materializes each chunk before phase reduction.
        """
        c = self._cache
        num_bf = int(c["num_bf"])
        ny = int(c["ny"])
        nx = int(c["nx"])
        full_bytes = num_bf * ny * nx * 8
        if full_bytes > 6 * 1024 ** 3:
            return self._reconstruct_chunked_full(mags_m, angles_rad)

        if self._mean_phase_buffer is None:
            self._mean_phase_buffer = cp.empty((ny, nx), dtype=cp.float32)
        self._run_correction_pipeline_full(mags_m, angles_rad)
        total = int(ny * nx)
        block = 256
        grid = (total + block - 1) // block
        _mean_phase_kernel(
            (grid,), (block,),
            (self._corrected_buffer, self._mean_phase_buffer,
             np.int32(num_bf), np.int32(ny), np.int32(nx)),
        )
        return self._mean_phase_buffer

    def _reconstruct_chunked_full(
        self, mags_m: cp.ndarray, angles_rad: cp.ndarray,
    ) -> "cp.ndarray":
        """Chunked full-aberration reconstruct for large scans.

        Processes BF pixels in groups of ``chunk_bf`` so the staging buffer
        stays under ~2 GB.  Phase reduction is ``mean(angle(corrected))``
        performed via CuPy per chunk (no dedicated col-accumulate kernel for
        the 14-coef path yet - that belongs to a future perf pass).
        """
        from .fft_common import pack_aberration_coefs
        c = self._cache
        num_bf = int(c["num_bf"])
        ny = int(c["ny"])
        nx = int(c["nx"])

        if self._pk_buffer is None or self._pk_buffer.shape != (num_bf,):
            self._pk_buffer = cp.empty((num_bf,), dtype=cp.complex64)
        abr_mag_scaled, abr_cm, abr_sm = pack_aberration_coefs(mags_m, angles_rad)
        kfactor = cp.float32(2.0 * math.pi / c["wavelength"])
        _pk_kernel_full(
            c["kx_bf"], c["ky_bf"], c["aperture_k_1d"],
            cp.float32(c["wavelength"]), kfactor,
            abr_mag_scaled, abr_cm, abr_sm, self._pk_buffer,
        )

        chunk_bf = max(1, (2 * 1024 ** 3) // (ny * nx * 8))
        if self._result_buffer is not None and self._result_buffer.shape[0] > chunk_bf:
            self._result_buffer = None
        chunk_shape = (chunk_bf, ny, nx)
        if self._result_buffer is None or self._result_buffer.shape != chunk_shape:
            self._result_buffer = cp.empty(chunk_shape, dtype=cp.complex64)

        phase_sum = cp.zeros((ny, nx), dtype=cp.float32)
        for bf_start in range(0, num_bf, chunk_bf):
            bf_end = min(bf_start + chunk_bf, num_bf)
            chunk = bf_end - bf_start
            sub_cache = dict(c)
            sub_cache["kx_bf"] = c["kx_bf"][bf_start:bf_end]
            sub_cache["ky_bf"] = c["ky_bf"][bf_start:bf_end]
            chunk_buf = self._result_buffer[:chunk]
            self._custom_fft.ifft2_inplace_fused_pk_full(
                chunk_buf,
                self.G_qk[bf_start:bf_end],
                sub_cache,
                self._pk_buffer[bf_start:bf_end],
                mags_m, angles_rad,
                self._dc_value_host,
            )
            phase_sum += cp.angle(chunk_buf).sum(axis=0)
        return phase_sum / float(num_bf)

    def reconstruct_full_with_loss(
        self, mags_m: cp.ndarray, angles_rad: cp.ndarray,
    ) -> "tuple[cp.ndarray, float]":
        """Full-aberration reconstruct + variance loss in a single pass.

        Mirrors :meth:`reconstruct_with_loss` but on the 14-coef kernel
        path.  The loss metric is the same BF-pixel phase variance the
        3-param optimizer minimizes, so it is directly comparable to
        ``auto_loss`` / ``reconstruct_with_loss``'s return value:

            mean phase = mean_bf(angle(corrected))
            var/pix    = mean_bf(angle²) - mean²
            loss       = mean over pixels of var/pix

        Used by :meth:`variance_loss_full` as the Optuna objective for
        higher-order aberration optimization.  Falls back to a chunked
        path for scans whose full staging buffer > 6 GB.
        """
        c = self._cache
        num_bf = int(c["num_bf"])
        ny = int(c["ny"])
        nx = int(c["nx"])
        full_bytes = num_bf * ny * nx * 8
        if full_bytes > 6 * 1024 ** 3:
            return self._reconstruct_full_with_loss_chunked(mags_m, angles_rad)

        if self._mean_phase_buffer is None:
            self._mean_phase_buffer = cp.empty((ny, nx), dtype=cp.float32)
        if self._sum_buffer is None:
            self._sum_buffer = cp.empty((ny, nx), dtype=cp.float32)
        if self._sumsq_buffer is None:
            self._sumsq_buffer = cp.empty((ny, nx), dtype=cp.float32)
        self._run_correction_pipeline_full(mags_m, angles_rad)
        total = int(ny * nx)
        block = 256
        grid = (total + block - 1) // block
        _sum_sumsq_phase_kernel(
            (grid,), (block,),
            (self._corrected_buffer, self._sum_buffer, self._sumsq_buffer,
             np.int32(num_bf), np.int32(ny), np.int32(nx)),
        )
        cp.divide(self._sum_buffer, float(num_bf), out=self._mean_phase_buffer)
        var_per_pixel = self._sumsq_buffer / float(num_bf) - self._mean_phase_buffer ** 2
        loss = float(cp.mean(var_per_pixel))
        return self._mean_phase_buffer, loss

    def _reconstruct_full_with_loss_chunked(
        self, mags_m: cp.ndarray, angles_rad: cp.ndarray,
    ) -> "tuple[cp.ndarray, float]":
        """Chunked full-aberration reconstruct + variance loss.

        Accumulates phase sum and phase² sum per BF chunk, then derives the
        variance from the pooled statistics.  Bit-identical to the
        non-chunked path (just sum-then-divide instead of divide-then-sum).
        """
        from .fft_common import pack_aberration_coefs
        c = self._cache
        num_bf = int(c["num_bf"])
        ny = int(c["ny"])
        nx = int(c["nx"])

        if self._pk_buffer is None or self._pk_buffer.shape != (num_bf,):
            self._pk_buffer = cp.empty((num_bf,), dtype=cp.complex64)
        abr_mag_scaled, abr_cm, abr_sm = pack_aberration_coefs(mags_m, angles_rad)
        kfactor = cp.float32(2.0 * math.pi / c["wavelength"])
        _pk_kernel_full(
            c["kx_bf"], c["ky_bf"], c["aperture_k_1d"],
            cp.float32(c["wavelength"]), kfactor,
            abr_mag_scaled, abr_cm, abr_sm, self._pk_buffer,
        )

        chunk_bf = max(1, (2 * 1024 ** 3) // (ny * nx * 8))
        if self._result_buffer is not None and self._result_buffer.shape[0] > chunk_bf:
            self._result_buffer = None
        chunk_shape = (chunk_bf, ny, nx)
        if self._result_buffer is None or self._result_buffer.shape != chunk_shape:
            self._result_buffer = cp.empty(chunk_shape, dtype=cp.complex64)

        phase_sum = cp.zeros((ny, nx), dtype=cp.float32)
        phase_sumsq = cp.zeros((ny, nx), dtype=cp.float32)
        for bf_start in range(0, num_bf, chunk_bf):
            bf_end = min(bf_start + chunk_bf, num_bf)
            chunk = bf_end - bf_start
            sub_cache = dict(c)
            sub_cache["kx_bf"] = c["kx_bf"][bf_start:bf_end]
            sub_cache["ky_bf"] = c["ky_bf"][bf_start:bf_end]
            chunk_buf = self._result_buffer[:chunk]
            self._custom_fft.ifft2_inplace_fused_pk_full(
                chunk_buf,
                self.G_qk[bf_start:bf_end],
                sub_cache,
                self._pk_buffer[bf_start:bf_end],
                mags_m, angles_rad,
                self._dc_value_host,
            )
            angles_chunk = cp.angle(chunk_buf)
            phase_sum += angles_chunk.sum(axis=0)
            phase_sumsq += (angles_chunk ** 2).sum(axis=0)

        mean_phase = phase_sum / float(num_bf)
        var_per_pixel = phase_sumsq / float(num_bf) - mean_phase ** 2
        loss = float(cp.mean(var_per_pixel))
        return mean_phase, loss

    def variance_loss_full(
        self, mags_m: cp.ndarray, angles_rad: cp.ndarray,
    ) -> float:
        """Scalar variance loss at the given 14-coef configuration.

        Thin wrapper around ``reconstruct_full_with_loss`` that discards
        the phase image.  This is the objective function used by the
        :meth:`SSB.optimize_full` Optuna driver.
        """
        _, loss = self.reconstruct_full_with_loss(mags_m, angles_rad)
        return loss

    # =====================================================================
    #  Fused reconstruct + variance loss (single pipeline pass)
    # =====================================================================

    def reconstruct_with_loss(
        self, C10: float, C12: float, phi12: float,
    ) -> tuple["cp.ndarray", float]:
        """Reconstruct mean phase AND compute full-IFFT variance in one pass.

        .. note::

           The loss returned here uses the **full 2D IFFT** pipeline and
           differs from :meth:`variance_loss` / :meth:`variance_loss_batch`,
           which use a sparse row-subsampled FFT optimised for Optuna.
           For loss values consistent with the optimiser, call
           :meth:`variance_loss` separately.

        Returns
        -------
        tuple[cp.ndarray, float]
            (mean_phase (ny, nx), variance_loss_scalar)
        """
        c = self._cache
        num_bf = int(c["num_bf"])
        ny = int(c["ny"])
        nx = int(c["nx"])
        full_bytes = num_bf * ny * nx * 8
        if full_bytes > 6 * 1024 ** 3:
            return self._reconstruct_with_loss_chunked(C10, C12, phi12)

        # Non-chunked: single pipeline pass
        if self._mean_phase_buffer is None:
            self._mean_phase_buffer = cp.empty((ny, nx), dtype=cp.float32)
        self._run_correction_pipeline(C10, C12, phi12)
        total = int(ny * nx)
        block = 256
        grid = (total + block - 1) // block

        # Compute sum and sumsq from corrected buffer
        sum_buf = self._sum_buffer
        sumsq_buf = self._sumsq_buffer
        _sum_sumsq_phase_kernel(
            (grid,), (block,),
            (self._corrected_buffer, sum_buf, sumsq_buf,
             np.int32(num_bf), np.int32(ny), np.int32(nx)),
        )

        # Mean phase = sum / num_bf
        cp.divide(sum_buf, float(num_bf), out=self._mean_phase_buffer)

        # Variance per pixel = sumsq/N - (sum/N)²
        var_per_pixel = sumsq_buf / float(num_bf) - self._mean_phase_buffer ** 2
        loss = float(cp.mean(var_per_pixel))

        return self._mean_phase_buffer, loss

    def _reconstruct_with_loss_chunked(
        self, C10: float, C12: float, phi12: float,
    ) -> tuple["cp.ndarray", float]:
        """Chunked version of reconstruct_with_loss for large scans."""
        return self._fused_chunked_core(C10, C12, phi12, compute_loss=True)

    def _fused_chunked_core(
        self,
        C10: float, C12: float, phi12: float,
        *, compute_loss: bool = False,
    ):
        """Shared chunked core using fused col-FFT + phase accumulate.

        When *compute_loss* is False, returns ``cp.ndarray`` (mean phase).
        When True, returns ``(cp.ndarray, float)`` (mean phase, loss).
        Both paths share the same kernel so phase is bit-identical.
        """
        c = self._cache
        num_bf = int(c["num_bf"])
        ny = int(c["ny"])
        nx = int(c["nx"])

        if self._pk_buffer is None or self._pk_buffer.shape != (num_bf,):
            self._pk_buffer = cp.empty((num_bf,), dtype=cp.complex64)
        cos2phi12 = math.cos(2.0 * phi12)
        sin2phi12 = math.sin(2.0 * phi12)
        _pk_kernel(
            c["alpha_k2_1d"], c["cos2phi_k_1d"], c["sin2phi_k_1d"],
            c["aperture_k_1d"],
            cp.float32(C10), cp.float32(C12),
            cp.float32(cos2phi12), cp.float32(sin2phi12),
            cp.float32(self._factor),
            self._pk_buffer,
        )

        chunk_bf = max(1, (2 * 1024 ** 3) // (ny * nx * 8))
        if self._result_buffer is not None and self._result_buffer.shape[0] > chunk_bf:
            self._result_buffer = None
        chunk_shape = (chunk_bf, ny, nx)
        if self._result_buffer is None or self._result_buffer.shape != chunk_shape:
            self._result_buffer = cp.empty(chunk_shape, dtype=cp.complex64)

        k_bf = 32
        max_groups = (chunk_bf + k_bf - 1) // k_bf
        partial_shape = (max_groups, ny, nx)
        if self._partial_sum is None or self._partial_sum.shape != partial_shape:
            self._partial_sum = cp.empty(partial_shape, dtype=cp.float32)
            self._partial_sumsq = cp.empty(partial_shape, dtype=cp.float32)

        phase_sum = cp.zeros((ny, nx), dtype=cp.float32)
        phase_sumsq = cp.zeros((ny, nx), dtype=cp.float32) if compute_loss else None

        for bf_start in range(0, num_bf, chunk_bf):
            bf_end = min(bf_start + chunk_bf, num_bf)
            chunk = bf_end - bf_start
            sub_cache = dict(c)
            sub_cache["kx_bf"] = c["kx_bf"][bf_start:bf_end]
            sub_cache["ky_bf"] = c["ky_bf"][bf_start:bf_end]
            chunk_buf = self._result_buffer[:chunk]
            n_groups = (chunk + k_bf - 1) // k_bf
            self._custom_fft.ifft2_fused_pk_col_accumulate(
                chunk_buf,
                self.G_qk[bf_start:bf_end],
                sub_cache,
                self._pk_buffer[bf_start:bf_end],
                C10, C12, cos2phi12, sin2phi12,
                self._factor, self._dc_value_host,
                self._partial_sum[:n_groups],
                self._partial_sumsq[:n_groups],
                k_bf,
            )
            phase_sum += self._partial_sum[:n_groups].sum(axis=0)
            if compute_loss:
                phase_sumsq += self._partial_sumsq[:n_groups].sum(axis=0)

        mean_phase = phase_sum / float(num_bf)

        if compute_loss:
            var_per_pixel = phase_sumsq / float(num_bf) - mean_phase ** 2
            loss = float(cp.mean(var_per_pixel))
            return mean_phase, loss
        return mean_phase

    # =====================================================================
    #  Variance loss
    # =====================================================================

    def _release_scalar_buffers(self) -> None:
        if self._result_buffer is None and self._variance_buffer is None:
            return
        self._result_buffer = None
        self._variance_buffer = None
        self._corrected_buffer = None

    def _get_streaming_buffers(self, batch: int, stream_bf: int) -> dict[str, object]:
        """Get small buffers for streaming variance computation.

        The staging buffer is (batch, stream_bf, 256, 256) complex64 - sized to
        fit in L2 cache so the row IFFT → column IFFT data never hits GDDR7.
        """
        c = self._cache
        num_bf = int(c["num_bf"])
        key = (batch, stream_bf, num_bf)
        cached = self._streaming_cache.get(key)
        if cached is not None:
            return cached
        ny, nx = int(c["ny"]), int(c["nx"])
        cached = {
            "staging": cp.empty((batch, stream_bf, ny, nx), dtype=cp.complex64),
            "pk": cp.empty((batch, num_bf), dtype=cp.complex64),
            "sum": cp.empty((batch, ny, nx), dtype=cp.float32),
            "sumsq": cp.empty((batch, ny, nx), dtype=cp.float32),
            "variance": cp.empty((batch, ny, nx), dtype=cp.float32),
        }
        self._streaming_cache[key] = cached
        return cached

    def _get_batch_buffers(self, batch: int) -> dict[str, object]:
        cached = self._batch_cache.get(batch)
        if cached is not None:
            if "sum_buffer" not in cached:
                c = self._cache
                cached["sum_buffer"] = cp.empty((batch, c["ny"], c["nx"]), dtype=cp.float32)
                cached["sumsq_buffer"] = cp.empty((batch, c["ny"], c["nx"]), dtype=cp.float32)
            return cached
        c = self._cache
        shape = (batch, c["num_bf"], c["ny"], c["nx"])
        cached = {
            "result_buffer": cp.empty(shape, dtype=cp.complex64),
            "variance_buffer": cp.empty((batch, c["ny"], c["nx"]), dtype=cp.float32),
            "sum_buffer": cp.empty((batch, c["ny"], c["nx"]), dtype=cp.float32),
            "sumsq_buffer": cp.empty((batch, c["ny"], c["nx"]), dtype=cp.float32),
            "pk_buffer": cp.empty((batch, c["num_bf"]), dtype=cp.complex64),
            "sum_partial_buffer": None,
            "sumsq_partial_buffer": None,
        }
        self._batch_cache[batch] = cached
        return cached

    def _get_batch_chunk_buffers(self, batch: int, chunk_bf: int, num_bf: int) -> dict[str, object]:
        key = (batch, chunk_bf, num_bf)
        cached = self._batch_chunk_cache.get(key)
        if cached is not None:
            if cached.get("sum_partial_buffer") is None:
                groups = (chunk_bf + self._colvar_group - 1) // self._colvar_group
                try:
                    cached["sum_partial_buffer"] = cp.empty((batch * groups, 256, 256), dtype=cp.float32)
                    cached["sumsq_partial_buffer"] = cp.empty((batch * groups, 256, 256), dtype=cp.float32)
                    cached["sum_chunk_buffer"] = cp.empty((batch, 256, 256), dtype=cp.float32)
                    cached["sumsq_chunk_buffer"] = cp.empty((batch, 256, 256), dtype=cp.float32)
                except cp.cuda.memory.OutOfMemoryError:
                    cached["sum_partial_buffer"] = None
                    cached["sumsq_partial_buffer"] = None
                    cached["sum_chunk_buffer"] = None
                    cached["sumsq_chunk_buffer"] = None
            return cached
        c = self._cache
        ny = int(c["ny"])
        nx = int(c["nx"])
        groups = (chunk_bf + self._colvar_group - 1) // self._colvar_group
        sum_partial_buffer = None
        sumsq_partial_buffer = None
        sum_chunk_buffer = None
        sumsq_chunk_buffer = None
        try:
            sum_partial_buffer = cp.empty((batch * groups, 256, 256), dtype=cp.float32)
            sumsq_partial_buffer = cp.empty((batch * groups, 256, 256), dtype=cp.float32)
            sum_chunk_buffer = cp.empty((batch, 256, 256), dtype=cp.float32)
            sumsq_chunk_buffer = cp.empty((batch, 256, 256), dtype=cp.float32)
        except cp.cuda.memory.OutOfMemoryError:
            pass
        cached = {
            "result_buffer": cp.empty((batch, chunk_bf, ny, nx), dtype=cp.complex64),
            "variance_buffer": cp.empty((batch, ny, nx), dtype=cp.float32),
            "sum_buffer": cp.empty((batch, ny, nx), dtype=cp.float32),
            "sumsq_buffer": cp.empty((batch, ny, nx), dtype=cp.float32),
            "pk_buffer": cp.empty((batch, num_bf), dtype=cp.complex64),
            "pk_chunk": cp.empty((batch, chunk_bf), dtype=cp.complex64),
            "sum_partial_buffer": sum_partial_buffer,
            "sumsq_partial_buffer": sumsq_partial_buffer,
            "sum_chunk_buffer": sum_chunk_buffer,
            "sumsq_chunk_buffer": sumsq_chunk_buffer,
        }
        tail = num_bf % chunk_bf
        if tail:
            cached["result_tail"] = cp.empty((batch, tail, ny, nx), dtype=cp.complex64)
            cached["pk_tail"] = cp.empty((batch, tail), dtype=cp.complex64)
        self._batch_chunk_cache[key] = cached
        return cached

    def _variance_loss_batch_chunked(
        self,
        c: dict,
        batch: int,
        chunk_bf: int,
        c10_gpu: cp.ndarray,
        c12_gpu: cp.ndarray,
        cos2phi12_gpu: cp.ndarray,
        sin2phi12_gpu: cp.ndarray,
        out: "cp.ndarray | None",
    ) -> "cp.ndarray":
        buffers = self._get_batch_chunk_buffers(batch, chunk_bf, int(c["num_bf"]))
        pk_buffer = buffers["pk_buffer"]
        _pk_kernel(
            c["alpha_k2_1d"][None, :],
            c["cos2phi_k_1d"][None, :],
            c["sin2phi_k_1d"][None, :],
            c["aperture_k_1d"][None, :],
            c10_gpu[:, None],
            c12_gpu[:, None],
            cos2phi12_gpu[:, None],
            sin2phi12_gpu[:, None],
            cp.float32(self._factor),
            pk_buffer,
        )
        sum_buffer = buffers["sum_buffer"]
        sumsq_buffer = buffers["sumsq_buffer"]
        sum_partial_buffer = buffers.get("sum_partial_buffer")
        sumsq_partial_buffer = buffers.get("sumsq_partial_buffer")
        sum_chunk_buffer = buffers.get("sum_chunk_buffer")
        sumsq_chunk_buffer = buffers.get("sumsq_chunk_buffer")
        variance_buffer = buffers["variance_buffer"]
        sum_buffer.fill(0)
        sumsq_buffer.fill(0)
        use_rowsvar_partial = (
            sum_partial_buffer is not None
            and sumsq_partial_buffer is not None
            and sum_chunk_buffer is not None
            and sumsq_chunk_buffer is not None
        )
        num_bf = int(c["num_bf"])
        for start in range(0, num_bf, chunk_bf):
            end = min(num_bf, start + chunk_bf)
            chunk = end - start
            if chunk == chunk_bf:
                result_buffer = buffers["result_buffer"]
                pk_chunk = buffers["pk_chunk"]
            else:
                result_buffer = buffers["result_tail"]
                pk_chunk = buffers["pk_tail"]
            pk_chunk[...] = pk_buffer[:, start:end]
            cache_chunk = {
                "kx_bf": c["kx_bf"][start:end],
                "ky_bf": c["ky_bf"][start:end],
                "qx_1d": c["qx_1d"],
                "qy_1d": c["qy_1d"],
                "wavelength": c["wavelength"],
                "semiangle_rad": c["semiangle_rad"],
                "ang_y_rad": c["ang_y_rad"],
                "ang_x_rad": c["ang_x_rad"],
            }
            sum_partial_view = sum_partial_buffer
            sumsq_partial_view = sumsq_partial_buffer
            if use_rowsvar_partial:
                groups = (chunk + self._colvar_group - 1) // self._colvar_group
                limit = batch * groups
                sum_partial_view = sum_partial_buffer[:limit]
                sumsq_partial_view = sumsq_partial_buffer[:limit]
            self._custom_fft.ifft2_inplace_batch_fused_pk_variance(
                result_buffer,
                self.G_qk[start:end],
                cache_chunk,
                pk_chunk,
                c10_gpu,
                c12_gpu,
                cos2phi12_gpu,
                sin2phi12_gpu,
                self._factor,
                self._dc_value_host,
                sum_partial_view if use_rowsvar_partial else sum_buffer,
                sumsq_partial_view if use_rowsvar_partial else sumsq_buffer,
            )
            if use_rowsvar_partial:
                block = _choose_reduce_block(groups)
                grid = (c["nx"], c["ny"], batch)
                _reduce_group_sums_batch_kernel(
                    grid,
                    (block,),
                    (
                        sum_partial_view,
                        sumsq_partial_view,
                        sum_chunk_buffer,
                        sumsq_chunk_buffer,
                        np.int32(groups),
                        np.int32(c["ny"]),
                        np.int32(c["nx"]),
                        np.int32(batch),
                    ),
                )
                sum_buffer += sum_chunk_buffer
                sumsq_buffer += sumsq_chunk_buffer
        total = int(batch * c["ny"] * c["nx"])
        block = 256
        grid = (total + block - 1) // block
        _variance_from_sums_batch_kernel(
            (grid,),
            (block,),
            (
                sum_buffer,
                sumsq_buffer,
                variance_buffer,
                np.int32(num_bf),
                np.int32(c["ny"]),
                np.int32(c["nx"]),
                np.int32(batch),
            ),
        )
        if out is None:
            return cp.mean(variance_buffer, axis=(1, 2))
        cp.mean(variance_buffer, axis=(1, 2), out=out)
        return out

    def _compute_chunk_bf(self, batch: int, vram_fraction: float = 0.4) -> int:
        """Decide how many BF pixels to process per chunk based on free VRAM.

        Returns 0 if no chunking needed (all BF pixels fit in one pass).
        """
        num_bf = int(self._cache["num_bf"])
        chunk_bf = self._preferred_chunk_bf if self._preferred_chunk_bf > 0 else 0
        if batch <= 1:
            return chunk_bf if 0 < chunk_bf < num_bf else 0
        try:
            free_mem = cp.cuda.runtime.memGetInfo()[0]
            ny, nx = int(self._cache["ny"]), int(self._cache["nx"])
            bytes_per_bf = batch * ny * nx * 8
            fixed_bytes = batch * num_bf * 8 + batch * ny * nx * 4 * 3
            max_chunk = int((free_mem * vram_fraction - fixed_bytes) // bytes_per_bf)
            if max_chunk >= 256:
                max_chunk = max(256, (max_chunk // 256) * 256)
                max_chunk = min(max_chunk, num_bf)
                if chunk_bf <= 0:
                    chunk_bf = max_chunk
                else:
                    chunk_bf = min(chunk_bf, max_chunk)
                    chunk_bf = max(256, (chunk_bf // 256) * 256)
                    chunk_bf = min(chunk_bf, num_bf)
        except (RuntimeError, AttributeError):
            # memGetInfo() unavailable (no-GPU environment or driver error);
            # fall back to the user-supplied chunk_bf cap without auto-tuning.
            if chunk_bf > 0:
                chunk_bf = min(chunk_bf, num_bf)
        if chunk_bf > 0 and chunk_bf * 2 > num_bf:
            half = (num_bf // 2) // 256 * 256
            if half >= 256:
                chunk_bf = min(chunk_bf, half)
        if chunk_bf >= num_bf:
            chunk_bf = 0
        return chunk_bf

    def variance_loss(
        self,
        C10: float,
        C12: float,
        phi12: float,
        out: "cp.ndarray | None" = None,
    ) -> "cp.ndarray | float":
        """Compute variance loss for a single set of aberration parameters.

        Delegates to the batch quad kernel (batch=4).
        """
        c10_arr = np.full(4, C10, dtype=np.float32)
        c12_arr = np.full(4, C12, dtype=np.float32)
        phi_arr = np.full(4, phi12, dtype=np.float32)
        result = self.variance_loss_batch(c10_arr, c12_arr, phi_arr)
        if out is None:
            return float(result[0])
        out[()] = result[0]
        return out

    def variance_loss_batch(
        self,
        C10: "np.ndarray | list[float] | float",
        C12: "np.ndarray | list[float] | float",
        phi12: "np.ndarray | list[float] | float",
        out: "cp.ndarray | None" = None,
    ) -> "cp.ndarray":
        """Compute variance loss for a batch of parameters in a single GPU pass."""
        c = self._cache
        c10_vals = np.asarray(C10, dtype=np.float32)
        if c10_vals.ndim == 0:
            c10_vals = np.asarray([float(c10_vals)], dtype=np.float32)
        orig_batch = int(c10_vals.size)
        # The pair kernel (batch<4) produces incorrect variance results.
        # For small batches, pad to batch=4 with identical copies.
        _MIN_BATCH = 4
        if orig_batch < _MIN_BATCH:
            C12_arr = np.asarray(C12, dtype=np.float32)
            if C12_arr.ndim == 0:
                C12_arr = np.full(orig_batch, float(C12_arr), dtype=np.float32)
            phi_arr = np.asarray(phi12, dtype=np.float32)
            if phi_arr.ndim == 0:
                phi_arr = np.full(orig_batch, float(phi_arr), dtype=np.float32)
            results = cp.empty(orig_batch, dtype=cp.float32)
            for i in range(orig_batch):
                results[i] = self.variance_loss(
                    float(c10_vals[i]), float(C12_arr[i]), float(phi_arr[i])
                )
            if out is not None:
                out[:orig_batch] = results
                return out
            return results
        batch = int(c10_vals.size)
        c12_vals = np.asarray(C12, dtype=np.float32)
        if c12_vals.ndim == 0:
            c12_vals = np.full(batch, float(c12_vals), dtype=np.float32)
        phi_vals = np.asarray(phi12, dtype=np.float32)
        if phi_vals.ndim == 0:
            phi_vals = np.full(batch, float(phi_vals), dtype=np.float32)
        if c12_vals.size != batch or phi_vals.size != batch:
            raise ValueError("C10, C12, and phi12 must have matching lengths")
        cos2phi12 = np.cos(2.0 * phi_vals).astype(np.float32)
        sin2phi12 = np.sin(2.0 * phi_vals).astype(np.float32)
        c10_gpu = cp.asarray(c10_vals)
        c12_gpu = cp.asarray(c12_vals)
        cos2phi12_gpu = cp.asarray(cos2phi12)
        sin2phi12_gpu = cp.asarray(sin2phi12)
        num_bf = int(c["num_bf"])
        if batch > 1:
            self._release_scalar_buffers()
        # Auto-select streaming group size based on available VRAM.
        # Large stream_bf → fewer kernel launches → faster, but larger buffer.
        # Small stream_bf → staging buffer fits in L2 cache, lower VRAM.
        # _stream_bf acts as a cap (set to num_bf to disable streaming).
        ny_scan, nx_scan = int(c["ny"]), int(c["nx"])
        staging_bytes_per_bf = batch * ny_scan * nx_scan * 8
        try:
            free_bytes = cp.cuda.runtime.memGetInfo()[0]
            # Budget: staging + tail (worst case tail ≈ stream_bf) + pk + sum/sumsq/var
            overhead_bytes = batch * num_bf * 8 + 3 * batch * ny_scan * nx_scan * 4
            usable = max(0, int(free_bytes * 0.4) - overhead_bytes)
            # Need room for staging + tail buffer (2x staging in worst case)
            max_stream_bf = usable // (2 * staging_bytes_per_bf)
            stream_bf = min(self._stream_bf, max(32, max_stream_bf), num_bf)
        except (RuntimeError, AttributeError):
            # memGetInfo() unavailable; fall back to the conservative stream_bf cap.
            stream_bf = min(self._stream_bf, num_bf)
        bufs = self._get_streaming_buffers(batch, stream_bf)
        pk_buffer = bufs["pk"]
        sum_buffer = bufs["sum"]
        sumsq_buffer = bufs["sumsq"]
        variance_buffer = bufs["variance"]
        staging = bufs["staging"]
        _pk_kernel(
            c["alpha_k2_1d"][None, :],
            c["cos2phi_k_1d"][None, :],
            c["sin2phi_k_1d"][None, :],
            c["aperture_k_1d"][None, :],
            c10_gpu[:, None],
            c12_gpu[:, None],
            cos2phi12_gpu[:, None],
            sin2phi12_gpu[:, None],
            cp.float32(self._factor),
            pk_buffer,
        )
        sum_buffer.fill(0)
        sumsq_buffer.fill(0)
        self._custom_fft.ifft2_inplace_batch_fused_pk_variance(
            staging,
            self.G_qk,
            self._cache,
            pk_buffer,
            c10_gpu,
            c12_gpu,
            cos2phi12_gpu,
            sin2phi12_gpu,
            self._factor,
            self._dc_value_host,
            sum_buffer,
            sumsq_buffer,
            stream_bf=stream_bf,
        )
        total = int(batch * c["ny"] * c["nx"])
        block = 256
        grid = (total + block - 1) // block
        _variance_from_sums_batch_kernel(
            (grid,),
            (block,),
            (
                sum_buffer,
                sumsq_buffer,
                variance_buffer,
                np.int32(num_bf),
                np.int32(c["ny"]),
                np.int32(c["nx"]),
                np.int32(batch),
            ),
        )
        if out is None:
            result = cp.mean(variance_buffer, axis=(1, 2))
            return result[:orig_batch] if orig_batch < batch else result
        cp.mean(variance_buffer, axis=(1, 2), out=out)
        return out
