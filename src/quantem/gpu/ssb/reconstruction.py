"""
GPU-accelerated Single-Sideband (SSB) ptychographic reconstruction.

All operations use CuPy exclusively for GPU compute.
Example:
    from quantem.gpu.ssb import SSB

    semiangle = 21.9  # mrad
    bf_radius = 30    # pixels
    det_sampling = (2 * semiangle) / bf_radius  # mrad/px

    ssb = SSB(data_4d, energy=300e3, semiangle=semiangle,
              scan_sampling=0.5, det_sampling=det_sampling)

    # Optimize aberrations (Optuna + Grid refinement)
    ssb.optimize()

    # Reconstruct complex object, then extract phase/amplitude
    obj = ssb.reconstruct_object()
    phase = cp.angle(obj)
    amplitude = cp.abs(obj)
    print(ssb.aberrations)  # {C10, C12, phi12} in nm/radians
"""

import copy
import gc
import math
import pathlib
import time
import numpy as np
from itertools import product
from typing import Self

import cupy as cp

from quantem.gpu.ssb.engine import SSBEngine
from quantem.gpu.ssb.optics.physics import electron_wavelength_angstrom
from quantem.gpu.ssb.results import SSBResult

# =========================================================================
#  Utility functions
# =========================================================================

def _bf_subset_indices(
    full_num_bf: int,
    ratio: float | None,
) -> tuple["cp.ndarray | None", int]:
    """
    Build a uniform-stride BF pixel subset for a given fraction.

    Returns (indices, sub_num_bf). If ratio is None or >= 1, returns
    (None, full_num_bf) meaning "use the full BF disk".

    The stride is round(1 / ratio), so ratio=0.5 -> every 2nd pixel,
    ratio=0.25 -> every 4th, etc. Uniform-stride keeps the angular
    coverage of the BF disk as balanced as possible.
    """
    if ratio is None or ratio >= 1.0:
        return None, full_num_bf
    if ratio <= 0.0:
        raise ValueError(f"bf_subsample must be in (0, 1], got {ratio}")
    stride = max(1, int(round(1.0 / float(ratio))))
    indices = cp.arange(0, full_num_bf, stride, dtype=cp.int32)
    return indices, int(indices.size)


def spatial_frequencies(
    gpts: tuple[int, int],
    sampling: tuple[float, float],
    rotation_angle_rad: float | None = None,
) -> tuple[cp.ndarray, cp.ndarray]:
    """
    Compute spatial frequency grids in Fourier space.

    This is specialized for SSB ptychography and returns 1D arrays by default
    (for efficiency), or rotated 2D arrays if rotation_angle_rad is specified.

    For general 2D frequency grids without rotation, see
    quantem.gpu.ssb.image.freq_grid_2d which always returns 2D meshgrids.

    Parameters
    ----------
    gpts : tuple[int, int]
        Grid points (ny, nx).
    sampling : tuple[float, float]
        Real-space sampling (drow, dcol) in Angstroms.
    rotation_angle_rad : float, optional
        Rotation angle in radians for passive grid rotation.

    Returns
    -------
    k_row, k_col : tuple[cp.ndarray, cp.ndarray]
        1D frequency arrays in inverse-Angstrom (if rotation_angle_rad is None),
        or 2D rotated arrays (if rotation_angle_rad is specified).

    See Also
    --------
    quantem.gpu.ssb.image.freq_grid_2d : General 2D frequency grid utility.
    """
    k_row = cp.fft.fftfreq(gpts[0], sampling[0]).astype(cp.float32)
    k_col = cp.fft.fftfreq(gpts[1], sampling[1]).astype(cp.float32)
    if rotation_angle_rad is not None and rotation_angle_rad != 0:
        # Passive rotation: rotate coordinate grid
        kr_2d, kc_2d = cp.meshgrid(k_row, k_col, indexing='ij')
        cos_r = math.cos(rotation_angle_rad)
        sin_r = math.sin(rotation_angle_rad)
        kr_rot = kr_2d * cos_r - kc_2d * sin_r
        kc_rot = kr_2d * sin_r + kc_2d * cos_r
        return kr_rot, kc_rot

    return k_row, k_col

# =========================================================================
#  SSB - GPU-accelerated single-sideband reconstruction
# =========================================================================

class SSB:
    """
    GPU-accelerated Single-Sideband (SSB) ptychographic reconstruction.

    SSB reconstructs the complex transmission function of a sample from
    4D-STEM data. Each bright-field (BF) pixel sees the sample through a
    slightly different view angle. By correcting for the probe's aberration
    phase at each BF pixel and averaging, SSB recovers the object's phase
    and amplitude at the scan resolution.

    Typical workflow::

        ssb = SSB(data, voltage_kV=300, semiangle=21.9, scan_sampling=0.5)
        ssb.optimize(n_trials=200)  # find aberrations (global search)
        ssb.refine()                # polish aberrations (local search)
        result = ssb.result()       # reconstruct phase
        result.show()               # display

    Parameters
    ----------
    data : cp.ndarray
        3D ``(N, k_row, k_col)`` or 4D ``(scan_row, scan_col, k_row, k_col)``.
        Any dtype - auto-converted to float32. For 3D data with N = perfect
        square, scan_shape is inferred automatically.
    voltage_kV : float
        Accelerating voltage in kV (e.g., 300).
    semiangle : float
        Probe convergence semiangle in mrad.
    scan_sampling : float
        Real-space scan pixel size in Å.
    det_sampling : float, optional
        Detector angular sampling in mrad/px. If None, auto-detected from
        the BF disk radius in the mean diffraction pattern.
    aberrations : dict, optional
        (nm, polar). If None, starts from zero aberrations.
    rotation_angle_deg : float, optional
        Scan rotation from DPC: ``dpc_result.rotation_angle_deg``.
    bf_radius : int, optional
        Limit BF disk to this radius in pixels. If None, uses the full
        detected BF disk. Smaller radius = faster but lower resolution.
        Leave None for bf_radius_sweep to explore all radii.

    Troubleshooting
    ---------------
    **Out of memory**: Use ``SSB(..., bf_radius=30)`` to reduce BF pixel
    count. Each BF pixel costs ``scan_row × scan_col × 8`` bytes of VRAM
    for G_qk. Or restart the kernel to free stale GPU memory.

    **Loss not improving**: Check that ``semiangle`` and ``scan_sampling``
    match the experimental setup. Wrong values produce a wrong probe model,
    so the aberration correction can't converge.

    **Phase looks wrong after optimize**: Run ``ssb.refine()`` - Optuna
    finds the right region but can be ~5nm off on C10. Nelder-Mead polishes
    to the exact minimum.
    """

    # Default optimization ranges
    _DEFAULT_OPTIMIZE_RANGES = {
        "C10_nm": (-400, 400),
        "C12_nm": (0, 100),
        "phi12_deg": (-90, 90),
    }
    _DEFAULT_GRID_HALF_WIDTHS = {
        "C10_nm": 50,
        "C12_nm": 20,
        "phi12_deg": 30,
    }
    _DEFAULT_GRID_POINTS = {
        "C10_nm": 21,
        "C12_nm": 11,
        "phi12_deg": 13,
    }
    _MAX_GRID_BATCH_SIZE = 16

    def __init__(
        self,
        data: cp.ndarray,
        semiangle: float,
        scan_sampling: float | tuple[float, float],
        det_sampling: float | tuple[float, float] | None = None,
        *,
        voltage_kV: float | None = None,
        energy: float | None = None,
        scan_shape: tuple[int, int] | None = None,
        bf_intensity_threshold: float = 0.5,
        bf_radius: int | None = None,
        aberrations: dict[str, float] | None = None,
        rotation_angle_deg: float = 0.0,
    ):
        # Convert rotation angle from degrees (public API) to radians (internal)
        rotation_angle_rad = math.radians(rotation_angle_deg)

        # Resolve energy from voltage_kV or energy
        if voltage_kV is not None and energy is not None:
            raise ValueError("Specify voltage_kV or energy, not both.")
        if voltage_kV is not None:
            energy = voltage_kV * 1e3
        elif energy is None:
            raise ValueError("Specify voltage_kV (kV) or energy (eV).")

        # quantem.gpu SSB is GPU-only. Ensure the input is a CuPy array but
        # keep the native dtype. Casting the raw 4D block to float32 would
        # double memory (e.g. 19 GB uint16 -> 38 GB float32 copy on a
        # 512x512x192x192 scan). Reductions promote internally via uint64
        # accumulators, so the raw block stays in its source dtype.
        data = cp.asarray(data)

        # Reshape 3D → 4D
        if data.ndim == 3:
            if scan_shape is None:
                n = data.shape[0]
                side = int(n ** 0.5)
                if side * side != n:
                    raise ValueError(
                        f"scan_shape is required: {n} frames is not a perfect square. "
                        f"Pass scan_shape=(rows, cols)."
                    )
                scan_shape = (side, side)
            if scan_shape[0] * scan_shape[1] != data.shape[0]:
                raise ValueError("scan_shape does not match number of frames.")
            data = data.reshape(scan_shape[0], scan_shape[1], data.shape[1], data.shape[2])
        elif data.ndim != 4:
            raise ValueError("data must be 3D or 4D.")

        # SSB supports 128x128, 256x256, and 512x512 scan sizes. Auto-pad with mean DP
        # (or center-crop) to the closest supported shape so callers can pass
        # arbitrary scan dims (e.g. drift-corrected cubes).
        H, W = data.shape[0], data.shape[1]
        if (H, W) not in ((128, 128), (256, 256), (512, 512)):
            longest = max(H, W)
            target = 128 if longest <= 128 else 256 if longest <= 256 else 512
            if H > target or W > target:
                # center crop oversize axes
                r0 = max(0, (H - target) // 2)
                c0 = max(0, (W - target) // 2)
                data = data[r0:r0 + min(target, H), c0:c0 + min(target, W)]
                H, W = data.shape[0], data.shape[1]
            if H < target or W < target:
                # Pad with the mean DP — preserves realistic DP statistics so
                # probe detection + BF/DF integrals stay well-conditioned.
                # Chunked int64 sum avoids the 4× float32 transient that
                # `data.reshape(...).mean()` would allocate (would OOM on
                # 17 GB cube → 68 GB transient).
                pad_top = (target - H) // 2
                pad_left = (target - W) // 2
                det_h, det_w = data.shape[2], data.shape[3]
                flat = data.reshape(-1, det_h * det_w)
                is_integer = np.issubdtype(data.dtype, np.integer)
                sum_dtype = cp.int64 if is_integer else cp.float64
                acc = cp.zeros(det_h * det_w, dtype=sum_dtype)
                for s in range(0, flat.shape[0], 16 * W):
                    acc += flat[s:s + 16 * W].astype(sum_dtype).sum(axis=0)
                mean_dp = (acc.reshape(det_h, det_w).astype(cp.float64)
                            / flat.shape[0]).astype(data.dtype)
                padded = cp.broadcast_to(
                    mean_dp[None, None], (target, target, det_h, det_w),
                ).copy()
                padded[pad_top:pad_top + H, pad_left:pad_left + W] = data
                data = padded
            print(
                f"  SSB auto-shape: input {H}x{W} → padded/cropped to {target}x{target}"
            )

        # Handle scalar sampling values
        if isinstance(scan_sampling, (int, float)):
            scan_sampling = (float(scan_sampling), float(scan_sampling))

        # Auto-detect det_sampling from BF radius
        if det_sampling is None:
            from quantem.gpu.detector import detect_bf_radius
            from quantem.gpu.ssb.preprocess import dp_mean
            mean_dp = dp_mean(data)
            _, detected_bf_radius = detect_bf_radius(mean_dp)
            det_sampling = (2 * semiangle) / detected_bf_radius
            det_sampling = (det_sampling, det_sampling)
            print(f"  Auto-detected BF radius: {detected_bf_radius} px, det sampling: {det_sampling[0]:.4f} mrad/px")
        elif isinstance(det_sampling, (int, float)):
            det_sampling = (float(det_sampling), float(det_sampling))
        if aberrations is None:
            aberrations = {"C10": 0.0, "C12": 0.0, "phi12": 0.0}

        # Store user parameters
        self.energy = energy
        self.voltage_kV = voltage_kV
        self.semiangle_mrad = semiangle
        self.semiangle_cutoff = semiangle
        self.scan_sampling = scan_sampling
        self.angular_sampling = det_sampling
        self.bf_intensity_threshold = float(bf_intensity_threshold)
        self.aberrations = aberrations.copy()
        self._rotation_angle_rad = rotation_angle_rad

        # Compute derived parameters
        scan_gpts = data.shape[:2]
        det_gpts = data.shape[2:]
        wavelength = electron_wavelength_angstrom(energy)

        # Convert detector sampling: mrad -> reciprocal space
        reciprocal_sampling = (
            det_sampling[0] * 1e-3 / wavelength,
            det_sampling[1] * 1e-3 / wavelength,
        )
        sampling = (
            1.0 / (reciprocal_sampling[0] * det_gpts[0]),
            1.0 / (reciprocal_sampling[1] * det_gpts[1]),
        )

        # Store internal parameters
        self.gpts = det_gpts
        self.wavelength = wavelength
        self.sampling = sampling

        # BF mask -> G_qk extraction. Raw data stays in its native dtype
        # (usually uint16). _compute_bf_mask reduces over scan axes via
        # dp_mean (integer accumulator); _extract_gqk fancy-indexes the
        # masked BF pixels before casting to complex64, so the float cast
        # only touches the BF disk instead of the full 4D block.
        self.bf_inds_row, self.bf_inds_col, self.bf_center = self._compute_bf_mask(
            data, bf_intensity_threshold, bf_radius,
        )
        self.G_qk, self.dc_value = self._extract_gqk(
            data, self.bf_inds_row, self.bf_inds_col, scan_gpts, det_gpts,
        )
        del data
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

        self._scan_shape = scan_gpts

        # Coordinate setup
        q_row_1d, q_col_1d = spatial_frequencies(scan_gpts, scan_sampling)
        self.q_row, self.q_col = cp.meshgrid(q_row_1d, q_col_1d, indexing='ij')

        # Optimization state
        self._best_loss: float = float('inf')
        self._accelerator: SSBEngine | None = None
        self._elapsed_optimize: float = 0.0
        self._elapsed_grid: float = 0.0
        self._elapsed_refine: float = 0.0
        self._refine_method: str | None = None
        self._refine_nfev: int | None = None
        self._n_trials: int | None = None

    # =====================================================================
    #  VRAM estimation and management
    # =====================================================================

    def _estimate_optimize_vram_gb(self, batch_size: int = 4) -> float:
        """Estimate VRAM needed for optimize() with given batch size.

        The streaming architecture processes stream_bf BF pixels at a time
        (default 512), not all num_bf at once. The staging buffer is the
        dominant allocation: batch × stream_bf × scan_row × scan_col × complex64.
        """
        stream_bf = 512  # default streaming chunk size
        scan_row, scan_col = self._scan_shape
        num_bf = len(self.bf_inds_row)
        # staging buffer: batch × stream_bf × scan_row × scan_col × complex64
        staging_bytes = batch_size * stream_bf * scan_row * scan_col * 8
        # pk_buffer: batch × stream_bf × complex64
        pk_bytes = batch_size * stream_bf * 8
        # sum/sumsq/variance: batch × scan_row × scan_col × float32
        reduce_bytes = 3 * batch_size * scan_row * scan_col * 4
        # G_qk is already allocated (persistent)
        gqk_bytes = num_bf * scan_row * scan_col * 8
        return (staging_bytes + pk_bytes + reduce_bytes + gqk_bytes) / 1e9

    def _free_buffers(self) -> None:
        """Free optimization buffers, keep G_qk for reconstruction."""
        import gc
        if self._accelerator is not None:
            self._accelerator.clear_batch_caches()
            self._accelerator._release_scalar_buffers()
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    def free(self) -> None:
        """
        Release all GPU VRAM held by this SSB engine.

        Frees:
        - G_qk (the FFT of virtual BF stack - the largest allocation, ~7 GB)
        - Engine buffers (correction pipeline, variance computation)
        - Batch caches (optimizer working memory)

        After this call, ``result()`` and ``optimize()`` will fail - the
        engine is no longer usable. Previously returned ``SSBResult``
        objects remain valid (they hold independent copies of the phase).

        Call this when the SSB pipeline is done and you need VRAM for
        the next stage (e.g., iterative ptychography).
        """
        self._free_buffers()
        del self.G_qk
        self.G_qk = None
        if self._accelerator is not None:
            # Clear the engine's internal cache (geometry arrays etc.)
            self._accelerator._cache.clear()
            self._accelerator = None
        cp.get_default_memory_pool().free_all_blocks()

    @staticmethod
    def _compute_bf_mask(
        data: cp.ndarray,
        threshold: float,
        bf_radius: int | None = None,
    ) -> tuple[cp.ndarray, cp.ndarray, tuple[float, float]]:
        """Compute bright-field mask indices and center from mean diffraction pattern.

        Accepts raw integer or float GPU data. Uses ``dp_mean`` with an
        integer accumulator so no float32 copy of the 4D block is made.
        """
        from quantem.gpu.ssb.preprocess import dp_mean
        mean_dp = dp_mean(data)
        return SSB._compute_bf_mask_from_mean_dp(mean_dp, threshold, bf_radius)

    @staticmethod
    def _compute_bf_mask_from_mean_dp(
        mean_dp: cp.ndarray,
        threshold: float,
        bf_radius: int | None = None,
    ) -> tuple[cp.ndarray, cp.ndarray, tuple[float, float]]:
        """Compute BF mask indices from a precomputed mean diffraction pattern."""
        bf_mask = mean_dp > mean_dp.max() * threshold
        bf_inds = cp.nonzero(bf_mask)
        bf_inds_row = bf_inds[0].astype(cp.int32)
        bf_inds_col = bf_inds[1].astype(cp.int32)
        if len(bf_inds_row) == 0:
            raise ValueError(
                f"No bright-field pixels found with threshold "
                f"{threshold:.2f}. Check that the data "
                f"contains a visible BF disk, or lower the threshold."
            )
        weights = mean_dp[bf_inds_row, bf_inds_col].astype(cp.float32)
        weight_sum = float(weights.sum().get())
        if weight_sum > 0:
            center_row = float((bf_inds_row.astype(cp.float32) * weights).sum().get() / weight_sum)
            center_col = float((bf_inds_col.astype(cp.float32) * weights).sum().get() / weight_sum)
        else:
            center_row = float(bf_inds_row.mean().get())
            center_col = float(bf_inds_col.mean().get())

        if bf_radius is not None:
            dist_sq = (bf_inds_row.astype(cp.float32) - center_row) ** 2 + \
                      (bf_inds_col.astype(cp.float32) - center_col) ** 2
            within = dist_sq <= bf_radius ** 2
            bf_inds_row = bf_inds_row[within]
            bf_inds_col = bf_inds_col[within]
            if len(bf_inds_row) == 0:
                raise ValueError(
                    f"No bright-field pixels within bf_radius={bf_radius}. "
                    f"Increase bf_radius or check detector geometry."
                )

        return bf_inds_row, bf_inds_col, (center_row, center_col)

    @staticmethod
    def _extract_gqk(
        data: cp.ndarray,
        bf_inds_row: cp.ndarray,
        bf_inds_col: cp.ndarray,
        scan_gpts: tuple[int, ...],
        det_gpts: tuple[int, ...],
    ) -> tuple[cp.ndarray, complex]:
        """Extract G_qk via virtual BF stack and FFT, chunked on the BF axis.

        For Samsung 512x512 the unchunked transient peak was ~57 GB:
        raw data (19) + vbf_stack complex64 (19) + G_qk (19). That blew
        L40S 48 GB even before optimize ran.

        By chunking on BF pixels we keep only a small complex64 staging
        chunk live at a time. Pre-allocates the full G_qk output first
        (unavoidable, ~19 GB on 512x512), then fills it in chunks. Peak
        transient over init drops to ~40 GB total (raw 19 + G_qk 19 +
        ~2 GB chunk staging), fitting L40S with ~8 GB of headroom.

        Picks chunk_bf to cap the staging buffer at ~2 GB, same L2-cache
        sweet spot that made the reconstruct chunking fast.

        Returns (G_qk, dc_value).
        """
        num_bf = int(len(bf_inds_row))
        scan_row, scan_col = int(scan_gpts[0]), int(scan_gpts[1])
        det_row, det_col = int(det_gpts[0]), int(det_gpts[1])

        # Flat view: (N_scan, det_row, det_col) - shares storage with raw data.
        flat_data = data.reshape(-1, det_row, det_col)

        # Pre-allocate the full G_qk output. This allocation is unavoidable -
        # optimize/refine need the whole thing resident.
        G_qk = cp.empty((num_bf, scan_row, scan_col), dtype=cp.complex64)

        # Chunk sized so the complex64 staging buffer is ~2 GB (L2-friendly).
        bytes_per_bf = scan_row * scan_col * 8  # complex64
        target_chunk_bytes = 2 * 1024 ** 3
        chunk_bf = max(1, min(num_bf, target_chunk_bytes // bytes_per_bf))

        dc_accum = 0j
        for bf_start in range(0, num_bf, chunk_bf):
            bf_end = min(bf_start + chunk_bf, num_bf)
            row_chunk = bf_inds_row[bf_start:bf_end]
            col_chunk = bf_inds_col[bf_start:bf_end]
            # Fancy-index only this chunk's BF pixels (~few hundred MB uint16).
            vbf_flat = flat_data[:, row_chunk, col_chunk]
            k = bf_end - bf_start
            # Transpose + reshape + contiguous copy stays in native dtype.
            vbf_int = cp.ascontiguousarray(vbf_flat.T.reshape(k, scan_row, scan_col))
            del vbf_flat
            # Cast only this chunk to complex64 (~2 GB max).
            vbf_stack = vbf_int.astype(cp.complex64)
            del vbf_int
            # FFT in place into the pre-allocated G_qk slice.
            G_qk[bf_start:bf_end] = cp.fft.fft2(vbf_stack)
            del vbf_stack

        dc_value = complex(G_qk[:, 0, 0].mean().get())
        cp.get_default_memory_pool().free_all_blocks()
        return G_qk, dc_value

    def _resolve_coefs(
        self,
        C10: float | None = None,
        C12: float | None = None,
        phi12: float | None = None,
    ) -> tuple[float, float, float]:
        """Resolve aberrations, falling back to stored values."""
        if C10 is None:
            C10 = self.aberrations.get("C10", 0.0)
        if C12 is None:
            C12 = self.aberrations.get("C12", 0.0)
        if phi12 is None:
            phi12 = self.aberrations.get("phi12", 0.0)
        return C10, C12, phi12

    def _subset_for_radius(self, radius: int) -> "SSB":
        """Create a lightweight SSB clone using a BF pixel subset.

        Reuses the existing G_qk by selecting only the rows whose BF pixels
        fall within the given radius. No data re-loading or FFT needed.
        """
        dist_sq = (
            (self.bf_inds_row.astype(cp.float32) - self.bf_center[0]) ** 2
            + (self.bf_inds_col.astype(cp.float32) - self.bf_center[1]) ** 2
        )
        within = dist_sq <= radius ** 2
        if int(within.sum()) == 0:
            raise ValueError(
                f"No BF pixels within radius={radius}. "
                f"Available: {self.bf_inds_row.shape[0]} pixels."
            )

        # Shallow copy shares all arrays, then override what differs
        clone = copy.copy(self)
        clone.aberrations = self.aberrations.copy()
        if bool(within.all()):
            # All pixels selected - reuse parent arrays directly (no copy)
            clone.dc_value = self.dc_value
        else:
            clone.bf_inds_row = self.bf_inds_row[within]
            clone.bf_inds_col = self.bf_inds_col[within]
            clone.G_qk = self.G_qk[within]
            clone.dc_value = complex(clone.G_qk[:, 0, 0].mean().get())
        clone._best_loss = float('inf')
        clone._accelerator = None
        clone._elapsed_optimize = 0.0
        clone._elapsed_grid = 0.0
        clone._elapsed_refine = 0.0
        clone._refine_method = None
        clone._refine_nfev = None
        clone._n_trials = None
        return clone

    def _get_accelerator(self) -> SSBEngine:
        """Get or create CuPy accelerator."""
        if self._accelerator is None:
            self._accelerator = SSBEngine(
                G_qk=self.G_qk,
                bf_inds_row=self.bf_inds_row,
                bf_inds_col=self.bf_inds_col,
                bf_center=self.bf_center,
                dc_value=self.dc_value,
                gpts=self.gpts,
                sampling=self.sampling,
                q_row=self.q_row,
                q_col=self.q_col,
                wavelength=self.wavelength,
                semiangle_cutoff=self.semiangle_cutoff,
                angular_sampling=self.angular_sampling,
            )
        return self._accelerator

    # =====================================================================
    #  Core computation
    # =====================================================================

    def _prepare_accel(
        self,
        C10: float | None,
        C12: float | None,
        phi12: float | None,
        rotation_angle_rad: float | None,
    ) -> tuple[SSBEngine, float, float, float]:
        """Resolve coefs, prepare accelerator with rotation. Returns (accel, C10, C12, phi12)."""
        C10, C12, phi12 = self._resolve_coefs(C10, C12, phi12)
        if rotation_angle_rad is None:
            rotation_angle_rad = self._rotation_angle_rad
        accel = self._get_accelerator()
        accel.cache_rotation(rotation_angle_rad)
        return accel, C10, C12, phi12

    def _reconstruct(
        self,
        C10: float | None = None,
        C12: float | None = None,
        phi12: float | None = None,
        rotation_angle_rad: float | None = None,
    ) -> cp.ndarray:
        """Reconstruct mean phase image."""
        accel, C10, C12, phi12 = self._prepare_accel(C10, C12, phi12, rotation_angle_rad)
        return accel.reconstruct(C10, C12, phi12)

    def optimize_full(
        self,
        aberrations: dict[str, float | tuple[float, float]] | None = None,
        n_trials: int = 100,
        seed: int = 42,
        verbose: bool = True,
        rotation_angle_deg: float | None = None,
    ) -> Self:
        """
        Optuna TPE search over any subset of the 14 Krivanek aberrations.

        This is the higher-order analogue of :meth:`optimize`.  Unlike the
        legacy 3-param search, which uses the fast batched variance kernel,
        this driver evaluates one 14-coef configuration per trial using
        ``SSBEngine.variance_loss_full`` (the same scalar variance metric
        the 3-param path minimizes, so losses are directly comparable).

        Convergence usually takes fewer trials than full 3-param TPE because
        you typically lock most higher-order coefs and only search 2-4 of
        them at a time (e.g. ``{"C30": (-2000, 2000)}`` while C10/C12/phi12
        are already refined).  ``n_trials=100`` is a reasonable default.

        Parameters
        ----------
        aberrations : dict, optional
            Mapping from Krivanek name to either:

            - A scalar (float) - lock the coefficient to that value
              (magnitude in nm for engine-convention units, angle in radians).
            - A ``(low, high)`` tuple - search magnitude uniformly in [low, high]
              for m = 0 aberrations (C10, C30, C50), or magnitude in [0, high]
              + angle in [-π/2, π/2] for m ≠ 0 (so we search the 2D space).
            - For explicit angle search, pass ``"<name>_angle"`` with a
              ``(low_deg, high_deg)`` tuple.

            Missing names default to locked-at-zero.  For example::

                ssb.optimize_full({
                    "C10": (-400, 400),              # free: ±400 nm
                    "C12": (0, 100),                 # free: 0-100 nm mag
                    "phi12": (-math.pi/2, math.pi/2),# free: rad
                    "C30": (-2000, 2000),            # free: ±2000 nm
                    "C32_mag": (0, 500),             # free mag
                    "C32_angle": (-90, 90),          # free angle in DEG
                })
        n_trials : int, default 100
            Number of Optuna trials.
        seed : int, default 42
            TPE random seed.
        verbose : bool, default True
            Print a tqdm progress bar.
        rotation_angle_deg : float, optional
            Scan rotation override.  Default uses the engine's current rotation.

        Returns
        -------
        SSB
            Self (for chaining: ``ssb.optimize_full({...}).result()``).
        """
        import optuna
        from quantem.gpu.ssb.optics.aberration import ABERRATION_INDICES, N_ABERRATIONS

        if aberrations is None:
            aberrations = {
                "C10": tuple(self._DEFAULT_OPTIMIZE_RANGES["C10_nm"]),
                "C12": (0.0, self._DEFAULT_OPTIMIZE_RANGES["C12_nm"][1]),
                "phi12": (math.radians(-90.0), math.radians(90.0)),
            }
        # Build layout once: maps canonical name → (index, has_angle_flag).
        layout = {name: (i, has_ang) for i, (_, _, name, has_ang) in enumerate(ABERRATION_INDICES)}

        # Parse the request into three dicts: locked magnitudes, locked angles,
        # and free params (name → (low, high) to sample each trial).
        locked_mag: dict[str, float] = {}
        locked_ang: dict[str, float] = {}
        free_mag: dict[str, tuple[float, float]] = {}
        free_ang_deg: dict[str, tuple[float, float]] = {}

        for key, spec in aberrations.items():
            if key.endswith("_angle"):
                base = key[:-len("_angle")]
                if base not in layout:
                    raise ValueError(f"Unknown aberration angle key '{key}'")
                if isinstance(spec, (tuple, list)):
                    free_ang_deg[base] = (float(spec[0]), float(spec[1]))
                else:
                    locked_ang[base] = math.radians(float(spec))
            elif key.endswith("_mag"):
                base = key[:-len("_mag")]
                if base not in layout:
                    raise ValueError(f"Unknown aberration magnitude key '{key}'")
                if isinstance(spec, (tuple, list)):
                    free_mag[base] = (float(spec[0]), float(spec[1]))
                else:
                    locked_mag[base] = float(spec)
            else:
                # Bare name: refers to the magnitude for m ≠ 0 aberrations and
                # to the coefficient for m = 0 aberrations.
                if key not in layout:
                    if key == "phi12":
                        # Special shorthand: phi12 is angle of C12.
                        if isinstance(spec, (tuple, list)):
                            free_ang_deg["C12"] = (
                                math.degrees(float(spec[0])),
                                math.degrees(float(spec[1])),
                            )
                        else:
                            locked_ang["C12"] = float(spec)
                        continue
                    raise ValueError(f"Unknown aberration key '{key}'")
                if isinstance(spec, (tuple, list)):
                    free_mag[key] = (float(spec[0]), float(spec[1]))
                else:
                    locked_mag[key] = float(spec)

        rotation_angle_rad = (
            math.radians(rotation_angle_deg)
            if rotation_angle_deg is not None
            else self._rotation_angle_rad
        )
        accel = self._get_accelerator()
        accel.cache_rotation(rotation_angle_rad)

        # Warm-up call so the CUDA kernels are JIT-compiled and memory pools
        # are pre-allocated before we start measuring trial times.
        zero_m = cp.zeros(N_ABERRATIONS, dtype=cp.float32)
        zero_a = cp.zeros(N_ABERRATIONS, dtype=cp.float32)
        _ = accel.variance_loss_full(zero_m, zero_a)
        cp.cuda.Device().synchronize()

        # Pre-allocate mags/angs arrays reused across trials (avoid cp alloc per trial).
        mags = cp.zeros(N_ABERRATIONS, dtype=cp.float32)
        angs = cp.zeros(N_ABERRATIONS, dtype=cp.float32)

        def objective(trial: "optuna.Trial") -> float:
            mags[:] = 0.0
            angs[:] = 0.0
            for name, val in locked_mag.items():
                mags[layout[name][0]] = cp.float32(val)
            for name, val in locked_ang.items():
                angs[layout[name][0]] = cp.float32(val)
            for name, (lo, hi) in free_mag.items():
                v = trial.suggest_float(f"{name}_mag", lo, hi)
                mags[layout[name][0]] = cp.float32(v)
            for name, (lo_deg, hi_deg) in free_ang_deg.items():
                v_deg = trial.suggest_float(f"{name}_angle_deg", lo_deg, hi_deg)
                angs[layout[name][0]] = cp.float32(math.radians(v_deg))
            return float(accel.variance_loss_full(mags, angs))

        optuna.logging.set_verbosity(optuna.logging.WARNING)
        sampler = optuna.samplers.TPESampler(seed=seed)
        study = optuna.create_study(direction="minimize", sampler=sampler)
        t0 = time.perf_counter()
        if verbose:
            try:
                from tqdm.auto import tqdm
                pbar = tqdm(total=n_trials, desc="optimize_full", leave=True)
                def _cb(study, trial):
                    pbar.update(1)
                    pbar.set_postfix_str(f"best={study.best_value:.6g}")
                study.optimize(objective, n_trials=n_trials, callbacks=[_cb])
                pbar.close()
            except ImportError:
                study.optimize(objective, n_trials=n_trials)
        else:
            study.optimize(objective, n_trials=n_trials)
        elapsed = time.perf_counter() - t0

        # Write best values back into self.aberrations.  Keep legacy
        # C10/C12/phi12 keys populated so `result()` and downstream code
        # that reads those three still works; add higher-order names as
        # additional keys so ``reconstruct_full(self.aberrations)`` can
        # reproduce the result.
        best = study.best_params
        for name, (_, has_ang) in layout.items():
            if name in free_mag:
                key = f"{name}_mag"
                if has_ang:
                    val = best.get(key, 0.0)
                    self.aberrations[name] = float(val)
                else:
                    self.aberrations[name] = float(best.get(key, 0.0))
            elif name in locked_mag:
                self.aberrations[name] = float(locked_mag[name])
            if has_ang:
                if name in free_ang_deg:
                    ang_key = f"{name}_angle_deg"
                    self.aberrations[f"phi{name[1:]}"] = math.radians(best.get(ang_key, 0.0))
                elif name in locked_ang:
                    self.aberrations[f"phi{name[1:]}"] = float(locked_ang[name])
        # Also sync C10/C12/phi12 (legacy keys) so .result() and .refine() still work
        if "C10" in self.aberrations:
            pass  # already written above
        else:
            self.aberrations["C10"] = 0.0
        if "C12" not in self.aberrations:
            self.aberrations["C12"] = 0.0
        if "phi12" not in self.aberrations:
            self.aberrations["phi12"] = 0.0

        self._best_loss = float(study.best_value)
        self._n_trials = n_trials
        self._elapsed_optimize = elapsed
        # Preserve the compact trial list that explore() reads.  We pack every
        # trial's (loss, free params) so the explorer can show the
        # higher-order search history.
        self._optuna_trials = [
            {"loss": float(t.value) if t.value is not None else float("inf"),
             "params": dict(t.params)}
            for t in study.trials
        ]
        if verbose:
            print(f"  optimize_full: best loss={self._best_loss:.6f}, {elapsed:.1f}s")
        return self

    def reconstruct_full(
        self,
        aberrations: dict[str, float | tuple[float, float]] | None = None,
        rotation_angle_deg: float | None = None,
    ) -> cp.ndarray:
        """
        Reconstruct mean phase with all 14 Krivanek aberrations.

        Unlike :meth:`result` which only supports C10/C12/phi12 (the trio
        Optuna optimizes over), this method accepts the full 14-coefficient
        Krivanek set.  Intended for the explorer UI and for validating
        reconstructions against manually-specified higher-order values.

        The legacy 2-term path (:meth:`result`, :meth:`optimize`) is
        unchanged.  At the default ``aberrations=None`` this produces the
        same phase as ``SSB.result()`` at float32 precision.

        Parameters
        ----------
        aberrations : dict, optional
            Mapping from aberration name to value.  Keys are a subset of
            ``{"C10", "C12", "C21", "C23", "C30", "C32", "C34", "C41",
            "C43", "C45", "C50", "C52", "C54", "C56"}``.  Each value may be:

            - A scalar (magnitude in meters for ``C*0`` / ``Cn0``); orientation
              angle defaults to 0.
            - A ``(magnitude_m, angle_rad)`` tuple for m ≠ 0 aberrations.

            Missing keys default to ``(0.0, 0.0)``.  If ``None``, this
            reconstructs with every coefficient at zero and should match
            ``result()`` with zero aberrations.
        rotation_angle_deg : float, optional
            Scan rotation override.  If None, uses the engine's current
            rotation angle.

        Returns
        -------
        cp.ndarray
            Mean phase image (ny, nx), float32, on GPU.
        """
        from quantem.gpu.ssb.optics.aberration import ABERRATION_INDICES, N_ABERRATIONS

        mags = cp.zeros(N_ABERRATIONS, dtype=cp.float32)
        angs = cp.zeros(N_ABERRATIONS, dtype=cp.float32)
        if aberrations is not None:
            name_to_idx = {name: i for i, (_, _, name, _) in enumerate(ABERRATION_INDICES)}
            for name, val in aberrations.items():
                if name not in name_to_idx:
                    raise ValueError(
                        f"Unknown aberration '{name}'.  Valid names: "
                        f"{sorted(name_to_idx)}"
                    )
                idx = name_to_idx[name]
                has_angle = ABERRATION_INDICES[idx][3]
                if has_angle and isinstance(val, (tuple, list)):
                    mags[idx] = cp.float32(val[0])
                    angs[idx] = cp.float32(val[1])
                elif has_angle and not isinstance(val, (tuple, list)):
                    # Scalar for an m ≠ 0 aberration: treat as magnitude only
                    mags[idx] = cp.float32(val)
                else:
                    # Rotationally symmetric (m = 0) - scalar magnitude
                    if isinstance(val, (tuple, list)):
                        mags[idx] = cp.float32(val[0])
                    else:
                        mags[idx] = cp.float32(val)

        rotation_angle_rad = (
            math.radians(rotation_angle_deg)
            if rotation_angle_deg is not None
            else self._rotation_angle_rad
        )
        accel = self._get_accelerator()
        accel.cache_rotation(rotation_angle_rad)
        return accel.reconstruct_full(mags, angs)

    def _ho_arrays_from_aberrations(self) -> "tuple[cp.ndarray, cp.ndarray, bool]":
        """Pack ``self.aberrations`` into (mags, angles_rad, any_active) arrays
        shaped for ``SSBEngine.reconstruct_full``.

        Reads the widget's save format: ``C10``/``C12``/``phi12`` for the
        legacy trio (phi12 already in radians), and for higher orders:
          - ``Cn0`` stored as a scalar (nm)          → single slot
          - ``Cnm_mag`` + ``Cnm_angle`` (nm + DEG)   → split slots
        Angles get converted to radians.  Returns ``any_active=True`` iff any
        higher-order magnitude is non-zero, which is the signal for
        ``result()`` to route through the 14-coef kernel instead of the fast
        2-term path.
        """
        a = self.aberrations
        mags = cp.zeros(14, dtype=cp.float32)
        angs = cp.zeros(14, dtype=cp.float32)
        mags[0] = cp.float32(a.get("C10", 0.0))
        mags[1] = cp.float32(a.get("C12", 0.0))
        angs[1] = cp.float32(a.get("phi12", 0.0))

        layout = [
            ("C21",  2, True),  ("C23", 3, True),
            ("C30",  4, False), ("C32", 5, True), ("C34", 6, True),
            ("C41",  7, True),  ("C43", 8, True), ("C45", 9, True),
            ("C50", 10, False), ("C52", 11, True),
            ("C54", 12, True),  ("C56", 13, True),
        ]
        any_active = False
        for name, idx, has_angle in layout:
            if has_angle:
                mag = float(a.get(f"{name}_mag", 0.0))
                ang_deg = float(a.get(f"{name}_angle", 0.0))
            else:
                mag = float(a.get(name, 0.0))
                ang_deg = 0.0
            if mag != 0.0:
                any_active = True
            mags[idx] = cp.float32(mag)
            angs[idx] = cp.float32(math.radians(ang_deg))
        return mags, angs, any_active

    def result(self) -> SSBResult:
        """
        Reconstruct the phase image with current aberrations.

        Applies the aberration correction to all BF images and averages
        them to produce the complex object transmission function. The
        returned ``SSBResult`` has ``.phase``, ``.amplitude``, and
        ``.show()`` for immediate visualization.

        Routes through the 14-coefficient Krivanek kernel automatically
        whenever any higher-order magnitude in ``self.aberrations`` is
        non-zero - so calibrations loaded from the explorer with C21...C56
        values apply correctly in the live path.

        Call this after ``optimize()`` + ``refine()`` to get the final
        reconstruction. Each call reconstructs from scratch - results
        are not cached.

        Returns
        -------
        SSBResult
            Contains ``object_wave`` (complex), ``phase``, ``amplitude``,
            ``aberrations``, ``loss``, and ``elapsed`` time.
        """
        mags, angs, ho_active = self._ho_arrays_from_aberrations()
        try:
            if ho_active:
                # 14-coef kernel returns phase directly; synthesize object_wave
                # with unit amplitude so downstream SSBResult consumers that read
                # cp.angle(object_wave) still work.  SSB's amplitude channel is
                # not physically meaningful at this codebase's precision anyway.
                accel = self._get_accelerator()
                accel.cache_rotation(self._rotation_angle_rad)
                phase = accel.reconstruct_full(mags, angs)
                obj = cp.exp(1j * phase.astype(cp.float32)).astype(cp.complex64)
            else:
                obj = self._reconstruct_object()
        except cp.cuda.memory.OutOfMemoryError:
            num_bf = len(self.bf_inds_row)
            free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
            raise MemoryError(
                f"Out of GPU VRAM during SSB reconstruction "
                f"({num_bf} BF pixels, {free_gb:.1f} GB free).\n"
                f"Try: SSB(..., bf_radius=<smaller>) to reduce BF pixel count, "
                f"or restart the kernel to free stale GPU memory."
            ) from None
        try:
            if ho_active:
                # Loss metric is only defined for (C10, C12, phi12); report None
                # when HO is active, matching the widget's "loss manual" convention.
                loss = None
            else:
                loss = self._compute_loss()
        except (ValueError, ZeroDivisionError, FloatingPointError):
            # Loss is a diagnostic metric; a numerical failure here should not
            # abort the reconstruction. GPU errors (MemoryError, RuntimeError)
            # are intentionally not caught so they surface to the caller (#130).
            loss = None
        elapsed = self._elapsed_optimize + self._elapsed_grid + self._elapsed_refine
        scan_sampling_scalar = self.scan_sampling[0] if isinstance(self.scan_sampling, tuple) else self.scan_sampling
        return SSBResult(
            object_wave=obj,
            aberrations=self.aberrations.copy(),
            rotation_angle_deg=math.degrees(self._rotation_angle_rad),
            loss=loss,
            elapsed=elapsed if elapsed > 0 else None,
            n_trials=self._n_trials,
            num_bf=len(self.bf_inds_row),
            refine_method=self._refine_method,
            refine_nfev=self._refine_nfev,
            refine_elapsed=self._elapsed_refine if self._elapsed_refine > 0 else None,
            voltage_kV=self.voltage_kV,
            semiangle_mrad=self.semiangle_mrad,
            scan_sampling_A=scan_sampling_scalar,
            optuna_trials=getattr(self, "_optuna_trials", None),
        )

    def _defocus_sweep(
        self,
        c10_range_nm: tuple[float, float] = (-100, 100),
        n_steps: int = 21,
    ) -> "DefocusSweepResult":
        """Sweep defocus (C10) around current aberrations.

        Reconstructs phase images at evenly-spaced C10 values while keeping
        C12 and phi12 fixed at the current stored values. Returns a result
        object that can be passed directly to ``Show3D``.

        Parameters
        ----------
        c10_range_nm : tuple[float, float]
            Min and max defocus in nm (default ``(-100, 100)``).
        n_steps : int
            Number of defocus values (default 21).

        Returns
        -------
        DefocusSweepResult
        """
        from quantem.gpu.ssb.results import DefocusSweepResult

        t0 = time.perf_counter()
        c12 = self.aberrations["C12"]
        phi12 = self.aberrations["phi12"]
        c10_values = np.linspace(c10_range_nm[0], c10_range_nm[1], n_steps)

        accel = self._get_accelerator()
        accel.cache_rotation(self._rotation_angle_rad)

        # Batch variance losses (4 at a time) - much faster than sequential
        c12_arr = np.full(len(c10_values), c12, dtype=np.float32)
        phi12_arr = np.full(len(c10_values), phi12, dtype=np.float32)
        losses = np.empty(len(c10_values), dtype=np.float32)
        for i in range(0, len(c10_values), 4):
            batch = c10_values[i:i+4]
            batch_losses = accel.variance_loss_batch(
                batch.astype(np.float32),
                c12_arr[i:i+4],
                phi12_arr[i:i+4],
            )
            losses[i:i+len(batch)] = cp.asnumpy(batch_losses[:len(batch)])

        # Reconstruct phase images (sequential - each needs full BF buffer)
        images = []
        for c10 in c10_values:
            images.append(cp.asnumpy(self._reconstruct(C10=c10, C12=c12, phi12=phi12)))
        images = np.stack(images)

        best_idx = int(np.argmin(losses))
        return DefocusSweepResult(
            c10_values_nm=c10_values,
            losses=losses,
            images=images,
            best_c10_nm=float(c10_values[best_idx]),
            best_loss=float(losses[best_idx]),
            elapsed=time.perf_counter() - t0,
            labels=[f"C10={c10:.1f} nm" for c10 in c10_values],
        )

    def bf_radius_sweep(
        self,
        radii: list[int] | None = None,
        *,
        optimize_aberrations: dict[str, tuple[float, float]] | None = None,
        n_trials: int = 200,
        verbose: bool = True,
    ) -> "BFRadiusSweepResult":
        """
        Optimize aberrations independently at multiple BF radii.

        Each BF radius includes a different number of bright-field pixels,
        capturing different spatial frequency bands. Smaller radii are
        robust but low-resolution; larger radii resolve finer features but
        need more accurate aberration correction.

        Processes largest radius first, then progressively subsets the
        BF pixels down - each step frees the previous G_qk so VRAM usage
        *decreases* as radii shrink.

        The returned ``BFRadiusSweepResult`` has ``.show()`` for side-by-side
        comparison and ``.average()`` to combine the best reconstructions.

        Parameters
        ----------
        radii : list[int], optional
            BF radii to sweep in pixels. If None, auto-generates ~5 radii
            from 15 to the full BF disk radius.
        n_trials : int, default 50
            Optuna trials per radius. 50-200 recommended.
        verbose : bool, default True
            Print progress per radius.

        Returns
        -------
        BFRadiusSweepResult
            Contains phase images, losses, and ``SSBResult`` at each radius.
        """
        from quantem.gpu.ssb.results import BFRadiusSweepResult

        t0 = time.perf_counter()

        if optimize_aberrations is None:
            optimize_aberrations = self._DEFAULT_OPTIMIZE_RANGES.copy()

        # Compute max BF radius from current instance
        dist_sq = (
            (self.bf_inds_row.astype(cp.float32) - self.bf_center[0]) ** 2
            + (self.bf_inds_col.astype(cp.float32) - self.bf_center[1]) ** 2
        )
        max_bf_radius = int(cp.sqrt(dist_sq.max()).get())

        if radii is None:
            # Generate radii from 15 up to max, stepping by ~10-15
            radii = list(range(15, max_bf_radius, max(5, max_bf_radius // 5)))
            if radii[-1] != max_bf_radius:
                radii.append(max_bf_radius)
            if verbose:
                print(f"  Auto radii: {radii}")
        else:
            # Warn if any requested radius exceeds the parent's BF disk
            too_large = [r for r in radii if r > max_bf_radius]
            if too_large:
                print(
                    f"  Warning: radii {too_large} exceed max BF radius {max_bf_radius}. "
                    f"Clamping to {max_bf_radius}."
                )
                radii = sorted(set(min(r, max_bf_radius) for r in radii))

        sweep_results: dict[int, SSBResult] = {}
        sweep_phases: list[np.ndarray] = []
        sweep_losses: list[float] = []
        sweep_labels: list[str] = []

        # Free parent engine buffers to maximize VRAM for sweep
        if self._accelerator is not None:
            self._accelerator.clear_batch_caches()
            self._accelerator._release_scalar_buffers()
            cp.get_default_memory_pool().free_all_blocks()

        # Process largest → smallest: each radius subsets from the previous
        # one, so we never hold two large G_qk copies simultaneously.
        radii_descending = sorted(radii, reverse=True)
        prev_aberrations = self.aberrations.copy()
        current_ssb = self  # start from parent (largest BF set)

        from quantem.gpu.ssb.batch_optuna import batch_nelder_mead

        for i, r in enumerate(radii_descending):
            ssb_r = current_ssb._subset_for_radius(r)
            ssb_r.aberrations = prev_aberrations.copy()
            num_bf_r = len(ssb_r.bf_inds_row)
            # Free previous G_qk immediately after subsetting
            if current_ssb is not self:
                current_ssb.free()
                del current_ssb
            else:
                # First iteration: free parent engine buffers only
                self._free_buffers()
            current_ssb = ssb_r
            cp.get_default_memory_pool().free_all_blocks()
            if verbose:
                vram_free, vram_total = cp.cuda.runtime.memGetInfo()
                free_gb, total_gb = vram_free / 1e9, vram_total / 1e9
                print(f"  [{i+1}/{len(radii)}] radius={r} ({num_bf_r} BF pixels)  "
                      f"VRAM: {free_gb:.1f} GB available of {total_gb:.1f} GB")
            ssb_r.optimize(
                aberrations=optimize_aberrations,
                n_trials=n_trials,
                verbose=False,
            )
            # Quick refine with loose tolerance for radius comparison
            accel_r = ssb_r._get_accelerator()
            accel_r.cache_rotation(ssb_r._rotation_angle_rad)
            x0 = np.array([ssb_r.aberrations["C10"], ssb_r.aberrations["C12"], ssb_r.aberrations["phi12"]])
            best_x, best_loss, _ = batch_nelder_mead(accel_r, x0, xatol=1.0, fatol=1e-5)
            ssb_r.aberrations["C10"] = float(best_x[0])
            ssb_r.aberrations["C12"] = float(best_x[1])
            ssb_r.aberrations["phi12"] = float(best_x[2])
            ssb_r._best_loss = float(best_loss)
            ssb_r._free_buffers()
            result_r = ssb_r.result()
            prev_aberrations = result_r.aberrations.copy()
            sweep_results[r] = result_r
            sweep_phases.append(cp.asnumpy(result_r.phase))
            sweep_losses.append(result_r.loss)
            aberr = result_r.aberrations
            sweep_labels.append(
                f'r={r} | C10={aberr["C10"]:.0f}nm '
                f'C12={aberr["C12"]:.0f}nm | loss={result_r.loss:.6f}'
            )
            if verbose:
                print(
                    f'  C10={aberr["C10"]:.1f}nm, '
                    f'C12={aberr["C12"]:.1f}nm, loss={result_r.loss:.6f}'
                )
            # Free engine but keep G_qk for next subset
            ssb_r._accelerator = None
            cp.get_default_memory_pool().free_all_blocks()

        # Free the last intermediate
        if current_ssb is not self:
            current_ssb.free()
            del current_ssb
            cp.get_default_memory_pool().free_all_blocks()

        # Reorder results back to ascending radius order
        sweep_phases_ordered = []
        sweep_losses_ordered = []
        sweep_labels_ordered = []
        for r in radii:
            idx = radii_descending.index(r)
            sweep_phases_ordered.append(sweep_phases[idx])
            sweep_losses_ordered.append(sweep_losses[idx])
            sweep_labels_ordered.append(sweep_labels[idx])

        losses_arr = np.array(sweep_losses_ordered)
        best_idx = int(np.argmin(losses_arr))

        return BFRadiusSweepResult(
            radii=radii,
            results=sweep_results,
            images=np.stack(sweep_phases_ordered),
            losses=losses_arr,
            best_radius=radii[best_idx],
            best_loss=float(losses_arr[best_idx]),
            elapsed=time.perf_counter() - t0,
            labels=sweep_labels_ordered,
        )

    def _reconstruct_object(
        self,
        C10: float | None = None,
        C12: float | None = None,
        phi12: float | None = None,
        rotation_angle_rad: float | None = None,
    ) -> cp.ndarray:
        """Reconstruct complex transmission function."""
        accel, C10, C12, phi12 = self._prepare_accel(C10, C12, phi12, rotation_angle_rad)
        return accel.reconstruct_object(C10, C12, phi12)

    # =====================================================================
    #  Optimization helpers
    # =====================================================================

    def _print_summary(self, stage: str, elapsed: float) -> None:
        """Print one-line optimization summary."""
        a = self.aberrations
        print(
            f"  {stage}: loss={self._best_loss:.6f}  "
            f"C10={a['C10']:.1f} nm  C12={a['C12']:.1f} nm  "
            f"phi12={math.degrees(a['phi12']):.1f}°  "
            f"{elapsed:.1f}s"
        )

    # =====================================================================
    #  Optimization
    # =====================================================================

    def optimize(
        self,
        aberrations: dict[str, tuple[float, float] | float] | None = None,
        rotation_angle_deg: tuple[float, float] | float | None = None,
        n_trials: int = 200,
        seed: int = 42,
        verbose: bool = True,
        bf_subsample: float | None = None,
    ) -> Self:
        """
        Global search for aberration parameters using Optuna TPE.

        Uses Tree-structured Parzen Estimator (TPE) to explore the aberration
        parameter space stochastically. Evaluates ``n_trials`` candidate
        aberration sets in batches of 4 on the GPU, measuring the variance
        of phase estimates across BF pixels (lower = better correction).

        This finds the right region of parameter space quickly (~200 trials
        in 1-2s), but may be ~5 nm imprecise on C10. Follow with ``refine()``
        to find the exact minimum.

        Parameters
        ----------
        aberrations : dict, optional
            Search ranges per parameter. Keys: ``"C10_nm"``, ``"C12_nm"``,
            ``"phi12_deg"``. Values: ``(low, high)`` tuple to search, or a
            fixed ``float`` to lock the parameter. If None, searches
            C10 ±400 nm, C12 0-100 nm, phi12 ±90 deg.
        rotation_angle_deg : float or tuple[float, float], optional
            If a tuple, Optuna searches the rotation angle in that range.
            If a float, forces the rotation angle to that value. If None,
            uses the engine's current rotation angle.
        n_trials : int, default 200
            Number of Optuna trials. 200 is the production default for
            reliable convergence. Below 200 the exploration of the
            (C10, C12, phi12) variance landscape becomes unreliable.
        seed : int, default 42
            Random seed for reproducibility of the TPE sampler.
        verbose : bool, default True
            Print the tqdm progress bar and VRAM status header.
        bf_subsample : float or None, default None
            Fraction of BF pixels to use, in (0, 1]. None uses the full
            BF disk (legacy, bit-identical to prior behavior). A ratio like
            0.25 runs the optimizer on every 4th BF pixel (uniform stride)
            and is ~3-4x faster on large-BF datasets with aberration parity
            within 0.05 nm on C10/C12. Small BF disks (< 2000 pixels) and
            flat-loss samples (lamella) can silently drift the aberrations
            with matching loss, so only set this when you can validate the
            result against a full-BF run. See
            ``docs/bf_subsampling_case_study.md``.

        Returns
        -------
        SSB
            Self (for chaining: ``ssb.optimize().refine()``).
        """
        t0 = time.perf_counter()
        if aberrations is None:
            aberrations = dict(self._DEFAULT_OPTIMIZE_RANGES)
        accel = self._get_accelerator()
        accel.cache_rotation(self._rotation_angle_rad)
        _ = accel.variance_loss(0, 50, 0)
        cp.cuda.Device().synchronize()
        full_num_bf = int(accel.num_bf)
        # Build a uniform stride BF subset if requested. refine() later runs
        # on the full BF disk so any small precision gap closes there.
        sub_indices, sub_num_bf = _bf_subset_indices(full_num_bf, bf_subsample)
        if verbose:
            vram_free, vram_total = cp.cuda.runtime.memGetInfo()
            free_gb, total_gb = vram_free / 1e9, vram_total / 1e9
            opt_gb = self._estimate_optimize_vram_gb(4)
            if sub_indices is not None:
                ratio = sub_num_bf / full_num_bf
                print(
                    f"Optimizing aberrations ({n_trials} trials, "
                    f"{sub_num_bf} / {full_num_bf} BF pixels [{ratio:.2f} ratio])"
                )
            else:
                print(f"Optimizing aberrations ({n_trials} trials, {full_num_bf} BF pixels)")
            print(f"  VRAM: {free_gb:.1f} GB available of {total_gb:.1f} GB, {opt_gb:.1f} GB needed")
        from quantem.gpu.ssb.batch_optuna import batch_optimize
        def _run_optimize():
            return batch_optimize(
                accel,
                aberrations=aberrations,
                rotation_angle_rad=self._rotation_angle_rad,
                rotation_angle_deg_spec=(
                    rotation_angle_deg
                    if isinstance(rotation_angle_deg, tuple)
                    else None
                ),
                aberration_defaults=self.aberrations,
                n_trials=n_trials,
                batch_size=4,
                seed=seed,
                verbose=verbose,
            )
        try:
            if sub_indices is not None:
                with accel.use_bf_subset(sub_indices):
                    best_params, best_value, trial_history = _run_optimize()
            else:
                best_params, best_value, trial_history = _run_optimize()
            # Store full Optuna trial history (#26) on the engine so the
            # caller can persist it to the sidecar. Cheap to keep: ~200
            # dicts x 3 floats each = ~5 KB per file.
            self._optuna_trials = trial_history
        except cp.cuda.memory.OutOfMemoryError:
            num_bf = len(self.bf_inds_row)
            free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
            raise MemoryError(
                f"Out of GPU VRAM during SSB optimization "
                f"({num_bf} BF pixels, {free_gb:.1f} GB free).\n"
                f"Try: SSB(..., bf_radius=<smaller>) to reduce BF pixel count, "
                f"or restart the kernel to free stale GPU memory."
            ) from None
        self._best_loss = best_value
        # Update aberrations: use optimized value if present, else fixed value
        for opt_key, aberr_key, convert in [
            ("C10_nm", "C10", None),
            ("C12_nm", "C12", None),
            ("phi12_deg", "phi12", math.radians),
        ]:
            if opt_key in best_params:
                val = best_params[opt_key]
                self.aberrations[aberr_key] = convert(val) if convert else val
            elif opt_key in aberrations and not isinstance(aberrations[opt_key], tuple):
                val = aberrations[opt_key]
                self.aberrations[aberr_key] = convert(val) if convert else val
        if "rotation_angle_deg" in best_params:
            self._rotation_angle_rad = math.radians(best_params["rotation_angle_deg"])
        elif rotation_angle_deg is not None and not isinstance(rotation_angle_deg, tuple):
            self._rotation_angle_rad = math.radians(rotation_angle_deg)
        # Free batch buffers - not needed after optimization
        accel.clear_batch_caches()
        self._elapsed_optimize = time.perf_counter() - t0
        self._n_trials = n_trials
        if verbose:
            self._print_summary("Optimize", self._elapsed_optimize)
        return self

    def refine(
        self,
        verbose: bool = True,
        xatol: float = 0.1,
        fatol: float = 1e-8,
        lock: list[str] | None = None,
        bf_subsample: float | None = None,
    ) -> Self:
        """
        Local refinement using GPU-batched Nelder-Mead.

        Starts from the current aberrations (typically set by ``optimize()``)
        and walks downhill to the nearest minimum. Nelder-Mead is
        derivative-free and handles the scale mismatch between C10 (nm)
        and phi12 (radians) via simplex geometry. On a 512x512 scan with a
        good Optuna starting point it typically converges in ~30 evals (~3 s
        at the current V13 variance kernel throughput).

        Call after ``optimize()`` which finds the right region. Calling
        ``refine()`` alone (without ``optimize()``) will only find the
        nearest local minimum, which may not be the global one.

        Why Nelder-Mead and not a gradient method: the phi12 direction has
        weak curvature when C12 is small, and finite-difference gradients
        in that direction are numerically noisy. This implementation batches
        loss evaluations through the GPU path used by live screening.

        Parameters
        ----------
        verbose : bool, default True
            Show the tqdm progress bar and print the summary at the end.
        xatol : float, default 0.1
            Stop when the simplex spread is below this (nm for C10/C12,
            radians for phi12). Loosening to 0.5 saves ~37% of evals on
            gold but drifts ~1 nm on Samsung; leave at 0.1 for calibrations.
        fatol : float, default 1e-8
            Stop when the loss spread across simplex vertices is below this.
        lock : list[str], optional
            Aberration keys to hold fixed (e.g. ``["C12", "phi12"]``).
            Locked params keep their current value while others are refined.
            Useful for defocus series where only C10 should change.
        bf_subsample : float or None, default None
            Fraction of BF pixels to use, in (0, 1]. None uses the full
            BF disk. A ratio like 0.25 runs the refiner on every 4th BF
            pixel and is 1.5-3x faster; the final loss is re-evaluated on
            the full BF disk before being stored so the reported number
            stays comparable to a full-BF run. Same stability caveats as
            in ``optimize()``: avoid on small BF disks and flat-loss
            samples. See ``docs/bf_subsampling_case_study.md``.

        Returns
        -------
        SSB
            Self (for chaining).
        """
        t0 = time.perf_counter()
        accel = self._get_accelerator()
        accel.cache_rotation(self._rotation_angle_rad)
        lock = set(lock or [])
        # Build lists of free (optimized) and fixed (locked) params
        all_keys = ["C10", "C12", "phi12"]
        free_keys = [k for k in all_keys if k not in lock]
        x0 = np.array([self.aberrations[k] for k in free_keys])
        full_num_bf = int(accel.num_bf)
        sub_indices, sub_num_bf = _bf_subset_indices(full_num_bf, bf_subsample)
        if verbose and sub_indices is not None:
            ratio = sub_num_bf / full_num_bf
            print(
                f"Refining aberrations "
                f"({sub_num_bf} / {full_num_bf} BF pixels [{ratio:.2f} ratio])"
            )
        if free_keys != all_keys:
            locked = ", ".join(sorted(lock)) or "unknown"
            raise ValueError(
                "GPU-only SSB nmead does not support locked refinement yet "
                f"(locked: {locked}). Use unlocked --ssb-refine nmead, "
                "locked SSB reference mode, or add a GPU-batched locked refiner."
            )

        from quantem.gpu.ssb.batch_optuna import batch_nelder_mead

        def _run_batched():
            return batch_nelder_mead(
                accel,
                x0.astype(np.float64),
                xatol=xatol,
                fatol=fatol,
                max_iter=300,
            )

        if sub_indices is not None:
            with accel.use_bf_subset(sub_indices):
                best_x, best_loss, n_evals = _run_batched()
        else:
            best_x, best_loss, n_evals = _run_batched()
        for i, k in enumerate(free_keys):
            self.aberrations[k] = float(best_x[i])
        if sub_indices is not None:
            # Re-evaluate the final loss on the FULL BF set so the reported
            # number is comparable to the non-subsampled path.
            self._best_loss = float(accel.variance_loss(
                self.aberrations["C10"],
                self.aberrations["C12"],
                self.aberrations["phi12"],
            ))
        else:
            self._best_loss = float(best_loss)
        nfev = int(n_evals)
        method = "gpu-batched-nmead"
        elapsed = time.perf_counter() - t0
        self._elapsed_refine = elapsed
        self._refine_method = method
        self._refine_nfev = nfev
        if verbose:
            self._print_summary(f"Refine ({method}, {nfev} evals)", elapsed)
        return self

    def grid_search(
        self,
        window: dict[str, float] | None = None,
        n_points: dict[str, int] | None = None,
        verbose: bool = True,
    ) -> Self:
        """
        Exhaustive grid search around current aberrations.

        Evaluates every combination of C10, C12, phi12 on a dense grid
        centered on the current values. More thorough than ``optimize()``
        + ``refine()`` but slower (3003 evaluations at default resolution).

        Useful as a second opinion when ``refine()`` might be stuck in a
        local minimum, or to visualize the loss landscape shape.

        Parameters
        ----------
        window : dict, optional
            Half-width per parameter. Default: C10 ±50nm, C12 ±20nm,
            phi12 ±30°.
        n_points : dict, optional
            Grid density per parameter. Default: C10=21, C12=11, phi12=13
            (= 3003 total evaluations).
        verbose : bool, default True
            Print grid dimensions and best parameters.

        Returns
        -------
        SSB
            Self with updated aberrations.
        """
        t0 = time.perf_counter()
        if window is None:
            window = {}
        if n_points is None:
            n_points = {}
        # Build grid centered on current aberrations
        hw = {k: window.get(k, v) for k, v in self._DEFAULT_GRID_HALF_WIDTHS.items()}
        np_ = {k: n_points.get(k, v) for k, v in self._DEFAULT_GRID_POINTS.items()}
        c10_center = self.aberrations["C10"]
        c12_center = self.aberrations["C12"]
        phi12_center_deg = math.degrees(self.aberrations["phi12"])
        c10_vals = np.linspace(c10_center - hw["C10_nm"], c10_center + hw["C10_nm"], np_["C10_nm"])
        c12_vals = np.linspace(max(0, c12_center - hw["C12_nm"]), c12_center + hw["C12_nm"], np_["C12_nm"])
        phi12_deg_vals = np.linspace(phi12_center_deg - hw["phi12_deg"], phi12_center_deg + hw["phi12_deg"], np_["phi12_deg"])
        phi12_vals = [math.radians(p) for p in phi12_deg_vals]
        total = len(c10_vals) * len(c12_vals) * len(phi12_vals)
        if verbose:
            print(f"Grid search: {total} combinations ({len(c10_vals)}x{len(c12_vals)}x{len(phi12_vals)})")
        accel = self._get_accelerator()
        # Warm up
        accel.cache_rotation(self._rotation_angle_rad)
        _ = accel.variance_loss(0, 50, 0)
        cp.cuda.Device().synchronize()
        # VRAM budget report
        if verbose:
            vram_free, vram_total = cp.cuda.runtime.memGetInfo()
            free_gb, total_gb = vram_free / 1e9, vram_total / 1e9
            batch_size_preview = min(self._MAX_GRID_BATCH_SIZE, total)
            grid_gb = self._estimate_optimize_vram_gb(batch_size_preview)
            print(f"  VRAM: {free_gb:.1f} GB available of {total_gb:.1f} GB, {grid_gb:.1f} GB needed")
        aberr_params = list(product(c10_vals, c12_vals, phi12_vals))
        cp.get_default_memory_pool().free_all_blocks()
        batch_size = min(self._MAX_GRID_BATCH_SIZE, total)
        chunk_bf = accel._compute_chunk_bf(batch_size, vram_fraction=0.3)
        accel._preferred_chunk_bf = chunk_bf
        if verbose and batch_size > 1:
            msg = f"  Batch size: {batch_size}"
            if chunk_bf:
                msg += f" (BF chunk: {chunk_bf})"
            print(msg)
        accel.clear_batch_caches()
        cp.get_default_memory_pool().free_all_blocks()
        losses_gpu = cp.empty(total, dtype=cp.float32)
        from tqdm.auto import tqdm
        pbar = tqdm(total=total, desc="SSB grid", disable=not verbose,
                    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}]")
        for start in range(0, total, batch_size):
            end = min(start + batch_size, total)
            chunk = aberr_params[start:end]
            c10_chunk = [float(p[0]) for p in chunk]
            c12_chunk = [float(p[1]) for p in chunk]
            phi_chunk = [float(p[2]) for p in chunk]
            try:
                losses_chunk = accel.variance_loss_batch(c10_chunk, c12_chunk, phi_chunk)
            except cp.cuda.memory.OutOfMemoryError:
                accel.clear_batch_caches()
                cp.get_default_memory_pool().free_all_blocks()
                new_chunk = max(256, chunk_bf // 2) if chunk_bf else 256
                accel._preferred_chunk_bf = new_chunk
                if verbose:
                    print(f"  OOM - retrying with BF chunk: {new_chunk}")
                losses_chunk = accel.variance_loss_batch(c10_chunk, c12_chunk, phi_chunk)
            losses_gpu[start:end] = losses_chunk
            pbar.update(end - start)
        pbar.close()
        losses = cp.asnumpy(losses_gpu)
        # Find best
        best_idx = int(np.argmin(losses))
        best_loss = float(losses[best_idx])
        best_c10, best_c12, best_phi12 = aberr_params[best_idx]
        # Store results
        self._best_loss = best_loss
        self.aberrations["C10"] = best_c10
        self.aberrations["C12"] = best_c12
        self.aberrations["phi12"] = best_phi12
        # Cleanup
        accel.clear_batch_caches()
        cp.get_default_memory_pool().free_all_blocks()
        self._elapsed_grid = time.perf_counter() - t0
        if verbose:
            self._print_summary(f"Grid ({total} evals)", self._elapsed_grid)
        return self

    # =====================================================================
    #  Properties
    # =====================================================================

    def _compute_loss(self) -> float:
        """Compute variance loss for the current aberrations."""
        accel = self._get_accelerator()
        accel.cache_rotation(self._rotation_angle_rad)
        return float(accel.variance_loss(
            self.aberrations["C10"],
            self.aberrations["C12"],
            self.aberrations["phi12"],
        ))

    # =====================================================================
    #  Interactive exploration (Jupyter)
    # =====================================================================

    def explore(
        self,
        c10_range: tuple[float, float] | None = None,
        c12_range: tuple[float, float] | None = None,
        phi12_range: tuple[float, float] | None = None,
        rotation_range: tuple[float, float] | None = None,
        drag_bf: int = 2000,
        save_dir: "str | pathlib.Path | None" = None,
        source_file: "str | None" = None,
        size: int = 800,
        fft_on: bool = False,
        calibration: "str | pathlib.Path | object | None" = None,
    ) -> "SSBExplore":
        """Interactive SSB preview is owned by the UI layer.

        ``quantem.gpu`` owns the compute engine only. Use the widget/live SSB
        preview layer for anywidget interaction.
        """
        raise RuntimeError(
            "SSB.explore() is a UI feature and is not part of quantem.gpu. "
            "Use the quantem.gpu/widget SSB preview layer for interactive exploration."
        )
