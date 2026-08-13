"""Private CUDA compute implementation for the public SSB workflow."""

import gc
import math
import time
import numpy as np
from itertools import product
from typing import Literal, Self

import cupy as cp

from .engine import SSBEngine
from ..protocol import SSBExportState, SSBPrecision
from quantem.gpu.optics.physics import electron_wavelength_angstrom
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

class CudaSSBBackend:
    """
    CUDA compute backend for Single-Sideband ptychographic reconstruction.

    SSB reconstructs the complex transmission function of a sample from
    4D-STEM data. Each bright-field (BF) pixel sees the sample through a
    slightly different view angle. By correcting for the probe's aberration
    phase at each BF pixel and averaging, SSB recovers the object's phase
    and amplitude at the scan resolution.

    This class is private compute infrastructure. Scientist code constructs
    :class:`quantem.gpu.SSB`, which owns public units, validation, fitting, and
    shared result types.

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
    gqk_storage : {"herm"}, default "herm"
        Resident Fourier-stack storage. SSB stores only the Hermitian
        half-plane, roughly halving persistent ``G_qk`` memory without changing
        precision or the scientific objective. The old persistent full-plane
        layout has been removed from the public runtime path.

    Troubleshooting
    ---------------
    **Out of memory**: Use the public workflow's ``bf_radius`` option to reduce BF pixel
    count. Default Hermitian ``G_qk`` storage costs roughly
    ``scan_row × (scan_col/2 + 1) × 8`` bytes per BF pixel; phase/loss
    workflows fetch the missing half-plane on demand inside the CUDA kernels.
    Or restart the kernel to free stale GPU memory.

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
    backend = "cuda"
    precision = SSBPrecision()

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
        bf_intensity_threshold: float = 0.0,
        bf_radius: int | None = None,
        aberrations: dict[str, float] | None = None,
        rotation_angle_deg: float = 0.0,
        detector_gain: cp.ndarray | np.ndarray | None = None,
        gqk_storage: Literal["herm"] = "herm",
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

        gain = None
        if detector_gain is not None:
            gain = cp.asarray(detector_gain, dtype=cp.float32)
            if gain.shape != tuple(data.shape[2:]):
                raise ValueError(
                    "detector_gain shape must match detector shape; "
                    f"got {gain.shape}, expected {tuple(data.shape[2:])}."
                )

        # SSB supports 128x128, 256x256, 512x512, and 1024x1024 scan sizes. Auto-pad with mean DP
        # (or center-crop) to the closest supported shape so callers can pass
        # arbitrary scan dims (e.g. drift-corrected cubes).
        H, W = data.shape[0], data.shape[1]
        supported_scan_shapes = ((128, 128), (256, 256), (512, 512), (1024, 1024))
        if (H, W) not in supported_scan_shapes:
            longest = max(H, W)
            target = (
                128 if longest <= 128 else
                256 if longest <= 256 else
                512 if longest <= 512 else
                1024
            )
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

        # Handle scalar sampling values
        if isinstance(scan_sampling, (int, float)):
            scan_sampling = (float(scan_sampling), float(scan_sampling))

        # Auto-detect det_sampling from BF radius
        if det_sampling is None:
            from quantem.gpu.detector.compute.cuda.probe import detect_bf_radius
            from quantem.gpu.detector.compute.cuda.probe import mean_dp as mean_dp_cuda

            mean_dp = mean_dp_cuda(data)
            if gain is not None:
                mean_dp = mean_dp.astype(cp.float32, copy=False) * gain
            _, detected_bf_radius = detect_bf_radius(mean_dp)
            det_sampling = (2 * semiangle) / detected_bf_radius
            det_sampling = (det_sampling, det_sampling)
        elif isinstance(det_sampling, (int, float)):
            det_sampling = (float(det_sampling), float(det_sampling))
        if aberrations is None:
            aberrations = {"C10": 0.0, "C12": 0.0, "phi12": 0.0}
        if gqk_storage != "herm":
            raise ValueError(
                "gqk_storage='full' was removed from the SSB runtime path. "
                "Use the default exact Hermitian storage; low-level tests may "
                "construct canonical full-plane references directly."
            )

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
        self.gqk_storage = gqk_storage

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
        # integer mean-DP reduction (integer accumulator); _extract_gqk fancy-indexes the
        # masked BF pixels before casting to complex64, so the float cast
        # only touches the BF disk instead of the full 4D block.
        self.bf_inds_row, self.bf_inds_col, self.bf_center = self._compute_bf_mask(
            data, bf_intensity_threshold, bf_radius, detector_gain=gain,
        )
        self.G_qk, self.dc_value = self._extract_gqk(
            data, self.bf_inds_row, self.bf_inds_col, scan_gpts, det_gpts,
            detector_gain=gain,
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
        self._optuna_trials: list[dict] = []
        self._optimizer_objective_mode = "exact"

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
        # staging buffer: batch × stream_bf × scan_row × scan_col × complex64
        staging_bytes = batch_size * stream_bf * scan_row * scan_col * 8
        # pk_buffer: batch × stream_bf × complex64
        pk_bytes = batch_size * stream_bf * 8
        # sum/sumsq/variance: batch × scan_row × scan_col × float32
        reduce_bytes = 3 * batch_size * scan_row * scan_col * 4
        # G_qk is already allocated (persistent)
        gqk_bytes = int(self.G_qk.nbytes)
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
        detector_gain: cp.ndarray | None = None,
    ) -> tuple[cp.ndarray, cp.ndarray, tuple[float, float]]:
        """Compute bright-field mask indices and center from mean diffraction pattern.

        Accepts raw integer or float GPU data. Uses ``integer mean-DP reduction`` with an
        integer accumulator so no float32 copy of the 4D block is made.
        """
        frames = data.reshape(-1, data.shape[-2], data.shape[-1])
        sum_dtype = (
            cp.uint64
            if np.issubdtype(data.dtype, np.integer)
            else cp.float64
        )
        mean_dp = (
            frames.sum(axis=0, dtype=sum_dtype).astype(cp.float32)
            / int(frames.shape[0])
        )
        if detector_gain is not None:
            mean_dp = mean_dp.astype(cp.float32, copy=False) * detector_gain
        return CudaSSBBackend._compute_bf_mask_from_mean_dp(
            mean_dp, threshold, bf_radius
        )

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
        if bf_radius is None:
            probe_mask = mean_dp > mean_dp.mean() + mean_dp.std()
            probe_total = int(probe_mask.sum().get())
            if probe_total > 0:
                probe = probe_mask.astype(cp.float32)
                row_coords = cp.arange(
                    mean_dp.shape[0], dtype=cp.float32
                ).reshape(-1, 1)
                col_coords = cp.arange(
                    mean_dp.shape[1], dtype=cp.float32
                ).reshape(1, -1)
                center_row = float(
                    ((row_coords * probe).sum() / probe_total).get()
                )
                center_col = float(
                    ((col_coords * probe).sum() / probe_total).get()
                )
                selected_radius = math.sqrt(probe_total / math.pi)
            else:
                center_row = mean_dp.shape[0] / 2.0
                center_col = mean_dp.shape[1] / 2.0
                selected_radius = min(mean_dp.shape) * 0.25
        else:
            weights = mean_dp[bf_inds_row, bf_inds_col].astype(cp.float32)
            weight_sum = float(weights.sum().get())
            if weight_sum > 0:
                center_row = float(
                    (
                        bf_inds_row.astype(cp.float32) * weights
                    ).sum().get() / weight_sum
                )
                center_col = float(
                    (
                        bf_inds_col.astype(cp.float32) * weights
                    ).sum().get() / weight_sum
                )
            else:
                center_row = float(bf_inds_row.mean().get())
                center_col = float(bf_inds_col.mean().get())
            selected_radius = float(bf_radius)

        dist_sq = (bf_inds_row.astype(cp.float32) - center_row) ** 2 + (
            bf_inds_col.astype(cp.float32) - center_col
        ) ** 2
        within = dist_sq <= selected_radius ** 2
        bf_inds_row = bf_inds_row[within]
        bf_inds_col = bf_inds_col[within]
        if len(bf_inds_row) == 0:
            raise ValueError(
                f"No bright-field pixels within bf_radius={selected_radius}. "
                "Increase bf_radius or check detector geometry."
            )

        return bf_inds_row, bf_inds_col, (center_row, center_col)

    @staticmethod
    def _extract_gqk(
        data: cp.ndarray,
        bf_inds_row: cp.ndarray,
        bf_inds_col: cp.ndarray,
        scan_gpts: tuple[int, ...],
        det_gpts: tuple[int, ...],
        detector_gain: cp.ndarray | None = None,
    ) -> tuple[cp.ndarray, complex]:
        """Extract G_qk via virtual BF stack and FFT, chunked on the BF axis.

        For held-out dataset 512x512 the unchunked transient peak was ~57 GB:
        raw data (19) + vbf_stack complex64 (19) + G_qk (19). That blew
        L40S 48 GB even before optimize ran.

        By chunking on BF pixels we keep only a small complex64 staging
        chunk live at a time. Hermitian storage pre-allocates the nonredundant
        half-plane output first, then fills it in chunks.

        Picks chunk_bf to cap the staging buffer at ~2 GB, same L2-cache
        sweet spot that made the reconstruct chunking fast.

        Returns (G_qk, dc_value).
        """
        num_bf = int(len(bf_inds_row))
        scan_row, scan_col = int(scan_gpts[0]), int(scan_gpts[1])
        det_row, det_col = int(det_gpts[0]), int(det_gpts[1])
        stored_col = scan_col // 2 + 1

        # Flat view: (N_scan, det_row, det_col) - shares storage with raw data.
        flat_data = data.reshape(-1, det_row, det_col)

        # Pre-allocate resident G_qk. The nonredundant scan-frequency
        # half-plane is exact because virtual-BF images are real-valued.
        G_qk = cp.empty((num_bf, scan_row, stored_col), dtype=cp.complex64)

        # Chunk sized so the complex64 staging buffer is ~2 GB (L2-friendly).
        bytes_per_bf = scan_row * scan_col * 8  # complex64
        target_chunk_bytes = 2 * 1024 ** 3
        chunk_bf = max(1, min(num_bf, target_chunk_bytes // bytes_per_bf))

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
            if detector_gain is not None:
                gain_chunk = detector_gain[row_chunk, col_chunk].astype(cp.float32)
                vbf_stack *= gain_chunk[:, None, None]
            # FFT into the pre-allocated G_qk slice. q_col=0..N/2 is the
            # canonical source of truth; CUDA/MPS kernels mirror-conjugate
            # missing columns when a full Fourier coordinate is requested.
            fft_chunk = cp.fft.fft2(vbf_stack)
            half_chunk = cp.ascontiguousarray(fft_chunk[:, :, :scan_col // 2 + 1])
            G_qk[bf_start:bf_end] = half_chunk
            del half_chunk
            del fft_chunk
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

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Reconstruction grid shape in public ``(row, col)`` order."""

        return tuple(int(value) for value in self._scan_shape)

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector grid shape in public ``(row, col)`` order."""

        return tuple(int(value) for value in self.gpts)

    @property
    def num_bf(self) -> int:
        """Number of pixels in the complete detected bright-field disk."""

        return int(self.bf_inds_row.size)

    def cache_rotation(self, rotation_rad: float) -> None:
        """Prepare CUDA geometry for one scan-to-detector rotation."""

        self._rotation_angle_rad = float(rotation_rad)
        self._get_accelerator().cache_rotation(self._rotation_angle_rad)

    def reconstruct(self, c10: float, c12: float, phi12: float):
        """Return the exact full-BF phase reconstructed on CUDA."""

        return self._get_accelerator().reconstruct(c10, c12, phi12)

    def reconstruct_with_loss(
        self,
        c10: float,
        c12: float,
        phi12: float,
    ):
        """Return the phase and exact full-BF variance loss from CUDA."""

        return self._get_accelerator().reconstruct_with_loss(c10, c12, phi12)

    def reconstruct_full(self, mags_m, angles_rad):
        """Return a CUDA phase for the full aberration vector."""

        return self._get_accelerator().reconstruct_full(mags_m, angles_rad)

    def reconstruct_full_with_loss(self, mags_m, angles_rad):
        """Return a CUDA phase and loss for the full aberration vector."""

        return self._get_accelerator().reconstruct_full_with_loss(
            mags_m,
            angles_rad,
        )

    @staticmethod
    def phase_to_numpy(phase) -> np.ndarray:
        """Copy one reconstructed phase image to host float32."""

        return cp.asnumpy(phase).astype(np.float32, copy=False)

    def preview_context(self, num_bf: int):
        """Prepare a reusable reduced-BF CUDA preview context."""

        return self._get_accelerator().prepare_bf_subset(int(num_bf))

    def browser_state(self) -> SSBExportState:
        """Return compact state for browser WebGPU integration."""

        return self._get_accelerator().export_state()

    def export_brightfield(
        self,
        data,
        path_stem,
    ) -> tuple[object, float]:
        """Write exact detector counts in detector-major BF columns."""

        engine = self._get_accelerator()
        path = engine.write_exact_bf_source(data, path_stem)
        return path, engine.bf_source_write_seconds

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

    def reconstruct_phase(
        self,
        C10: float | None = None,
        C12: float | None = None,
        phi12: float | None = None,
        rotation_angle_rad: float | None = None,
    ) -> cp.ndarray:
        """Reconstruct mean phase image."""
        accel, C10, C12, phi12 = self._prepare_accel(C10, C12, phi12, rotation_angle_rad)
        return accel.reconstruct(C10, C12, phi12)

    def _ho_arrays_from_aberrations(self) -> "tuple[cp.ndarray, cp.ndarray, bool]":
        """Pack ``self.aberrations`` into (mags, angles_rad, any_active) arrays
        shaped for ``SSBEngine.reconstruct_full``.

        Reads the widget's save format: ``C10``/``C12``/``phi12`` for the
        shared trio (phi12 already in radians), and for higher orders:
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

    def result(self, *, compute_loss: bool = True) -> SSBResult:
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
                obj = self.reconstruct_object()
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
            if compute_loss and not ho_active:
                loss = self._compute_loss()
            else:
                # Higher-order reconstruction and explicit no-loss calls do not
                # evaluate the C10/C12/phi12 diagnostic loss.
                loss = None
        except (ValueError, ZeroDivisionError, FloatingPointError):
            # Loss is a diagnostic metric; a numerical failure here should not
            # abort the reconstruction. GPU errors (MemoryError, RuntimeError)
            # are intentionally not caught so they surface to the caller (#130).
            loss = None
        elapsed = self._elapsed_optimize + self._elapsed_grid + self._elapsed_refine
        scan_sampling_scalar = self.scan_sampling[0] if isinstance(self.scan_sampling, tuple) else self.scan_sampling
        brightfield = self.browser_state().brightfield
        return SSBResult(
            object_wave=obj,
            backend="cuda",
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
            bf_center=brightfield.center_row_col,
            bf_radius=brightfield.radius_px,
            detected_bf_radius=brightfield.detected_radius_px,
            optuna_trials=self._optuna_trials,
        )

    def fit(
        self,
        *,
        trials: int,
        refinement: str | None,
        search_ranges: dict[str, tuple[float, float] | float] | None,
        refine_lock: list[str] | None,
        seed: int,
        verbose: bool,
    ) -> SSBResult:
        """Run the shared exact optimization contract on CUDA."""

        if trials:
            self.optimize(
                aberrations=search_ranges,
                n_trials=int(trials),
                seed=int(seed),
                verbose=verbose,
                objective_mode="exact",
            )
        if refinement == "nelder-mead":
            self.refine(
                verbose=verbose,
                lock=refine_lock,
                objective_mode="exact",
            )
        return self.result()

    def reconstruct_result(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool = True,
    ) -> SSBResult:
        """Reconstruct the CUDA complex object at fixed aberrations."""

        self.aberrations.update(aberrations)
        return self.result(compute_loss=compute_loss)

    def preview(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool,
        higher_order_magnitudes: np.ndarray | None,
        higher_order_angles: np.ndarray | None,
    ) -> tuple[np.ndarray, float | None]:
        """Return one transient float32 phase and optional exact loss."""

        if higher_order_magnitudes is not None:
            if compute_loss:
                phase, loss = self.reconstruct_full_with_loss(
                    higher_order_magnitudes, higher_order_angles
                )
            else:
                phase = self.reconstruct_full(
                    higher_order_magnitudes, higher_order_angles
                )
                loss = None
        elif compute_loss:
            phase, loss = self.reconstruct_with_loss(
                aberrations["C10"], aberrations["C12"], aberrations["phi12"]
            )
        else:
            phase = self.reconstruct(
                aberrations["C10"], aberrations["C12"], aberrations["phi12"]
            )
            loss = None
        array = self.phase_to_numpy(phase)
        return array, None if loss is None else float(loss)

    def close(self) -> None:
        """Release CUDA-owned session state."""

        self.free()

    def reconstruct_object(
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
        objective_mode: Literal["native", "exact"] = "exact",
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
            BF disk. A ratio like
            0.25 runs the optimizer on every 4th BF pixel (uniform stride)
            and is ~3-4x faster on large-BF datasets with aberration reference agreement
            within 0.05 nm on C10/C12. Small BF disks (< 2000 pixels) and
            flat-loss samples (lamella) can silently drift the aberrations
            with matching loss, so only set this when you can validate the
            result against a full-BF run. See
            ``docs/bf_subsampling_case_study.md``.
        objective_mode : {"exact", "native"}, default "exact"
            ``"exact"`` evaluates every candidate through the full-BF
            phase-variance reconstruction. ``"native"`` is an explicit
            performance-study mode using the CUDA size-specific candidate
            evaluator; it is never selected by default for calibration.

        Returns
        -------
        SSB
            Self (for chaining: ``ssb.optimize().refine()``).
        """
        t0 = time.perf_counter()
        if objective_mode not in {"native", "exact"}:
            raise ValueError(
                "objective_mode must be 'native' or 'exact'; "
                f"got {objective_mode!r}."
            )
        if aberrations is None:
            aberrations = dict(self._DEFAULT_OPTIMIZE_RANGES)
        accel = self._get_accelerator()
        accel.set_optimizer_objective_mode(objective_mode)
        self._optimizer_objective_mode = objective_mode
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
        from .optimizer import batch_optimize
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
        objective_mode: Literal["native", "exact"] | None = None,
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
            gold but drifts ~1 nm on held-out dataset; leave at 0.1 for calibrations.
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
        objective_mode : {"native", "exact"} or None, optional
            Objective used by Nelder-Mead. ``None`` reuses the mode selected
            by the preceding :meth:`optimize` call.

        Returns
        -------
        SSB
            Self (for chaining).
        """
        t0 = time.perf_counter()
        accel = self._get_accelerator()
        if objective_mode is None:
            objective_mode = self._optimizer_objective_mode
        if objective_mode not in {"native", "exact"}:
            raise ValueError(
                "objective_mode must be 'native' or 'exact'; "
                f"got {objective_mode!r}."
            )
        accel.set_optimizer_objective_mode(objective_mode)
        self._optimizer_objective_mode = objective_mode
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
                "GPU-only SSB Nelder-Mead does not support locked refinement yet "
                f"(locked: {locked}). Use no locked aberrations with "
                "refinement='nelder-mead', "
                "locked SSB reference mode, or add a GPU-batched locked refiner."
            )

        from .optimizer import batch_nelder_mead

        def _run_batched():
            exact_fallback = accel.uses_optimizer_reconstruct_fallback
            sparse_1024 = (
                tuple(int(v) for v in self._scan_shape) == (1024, 1024)
                and not exact_fallback
            )
            effective_xatol = xatol
            effective_fatol = fatol
            effective_max_iter = 80 if sparse_1024 else 300
            if exact_fallback and xatol == 0.1 and fatol == 1e-8:
                # The exact full-IFFT objective is smooth enough on held-out dataset
                # 512 full BF that tighter generic tolerances over-solve the
                # phase by hundreds of evals. These defaults preserve the
                # phase image to <1e-3 rad p99.9 in the real-data signoff while
                # avoiding an invisible 200+ eval tail.
                effective_xatol = 0.25
                effective_fatol = 2e-6
                effective_max_iter = 160
            return batch_nelder_mead(
                accel,
                x0.astype(np.float64),
                xatol=effective_xatol,
                fatol=effective_fatol,
                max_iter=effective_max_iter,
                flat_fatol=1e-6 if sparse_1024 else None,
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
        method = "nelder-mead"
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
