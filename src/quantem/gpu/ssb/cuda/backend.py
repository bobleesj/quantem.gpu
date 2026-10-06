"""Private CUDA compute implementation for the public SSB workflow."""

import math
import time
from typing import ClassVar, Self

import cupy as cp
import numpy as np

from quantem.gpu.detector.cuda.probe import mean_dp
from quantem.gpu.optics.physics import wavelength_A_from_kV
from quantem.gpu.ssb.brightfield import disk_edge_radius
from quantem.gpu.ssb.contract import SSBExportState, SSBPrecision
from quantem.gpu.ssb.cuda.brightfield import bright_field_spectra, select_bright_field
from quantem.gpu.ssb.cuda.engine import SSBEngine
from quantem.gpu.ssb.cuda.optimizer import batch_nelder_mead, batch_optimize, fit_sample
from quantem.gpu.ssb.results import SSBResult


class CudaSSBBackend:
    """The CUDA implementation of the SSB backend protocol behind :class:`quantem.gpu.SSB`.

    Each bright-field pixel sees the sample through a slightly different probe tilt; correcting every pixel's image
    for the probe aberrations and averaging recovers the object phase at the scan resolution. This class prepares the
    bright-field spectra ``G_qk`` on the device and runs fits and reconstructions through ``SSBEngine``. Units are the
    engine's (C10, C12 in Angstrom); ``SSB`` owns the public units, validation and result types.

    Parameters
    ----------
    data : cupy.ndarray
        Counts, 3D ``(scan, det_row, det_col)`` (square scan unless ``scan_shape`` is given) or 4D
        ``(scan_row, scan_col, det_row, det_col)``, in their native dtype: reductions accumulate in uint64, and only the
        selected bright-field pixels are cast to complex64. Scans that are not 128, 256, 512 or 1024 square are
        center-cropped or padded with the mean pattern to the next supported size.
    semiangle : float
        Probe convergence semiangle in mrad.
    scan_sampling : float or (float, float)
        Scan step in Angstrom, (row, col).
    det_sampling : float or (float, float), optional
        Detector sampling in mrad per pixel; ``semiangle / disk_edge_radius(mean pattern)`` when omitted.
    voltage_kV : float
        Accelerating voltage in kV.
    bf_intensity_threshold, bf_radius, bf_center
        Bright-field selection (``cuda.brightfield.select_bright_field``).
    aberrations : dict, optional
        Starting C10, C12 (Angstrom) and phi12 (radians); zero when omitted.
    rotation_angle_deg : float
        Physical scan-detector rotation in degrees.

    Out of memory: the resident Hermitian ``G_qk`` costs about ``scan_row x (scan_col / 2 + 1) x 8`` bytes per
    bright-field pixel; a smaller ``bf_radius`` reduces it.
    """

    # search ranges of optimize() when none are given (engine units despite the names)
    _DEFAULT_OPTIMIZE_RANGES: ClassVar[dict[str, tuple[int, int]]] = {
        "C10_nm": (-400, 400),
        "C12_nm": (0, 100),
        "phi12_deg": (-90, 90),
    }
    backend = "cuda"
    precision = SSBPrecision()

    def __init__(
        self,
        data: object,
        semiangle: float,
        scan_sampling: float | tuple[float, float],
        det_sampling: float | tuple[float, float] | None = None,
        *,
        voltage_kV: float,
        scan_shape: tuple[int, int] | None = None,
        bf_intensity_threshold: float = 0.0,
        bf_radius: int | None = None,
        bf_center: tuple[float, float] | None = None,
        aberrations: dict[str, float] | None = None,
        rotation_angle_deg: float = 0.0,
    ):
        # Convert rotation angle from degrees (public API) to radians (internal)
        rotation_angle_rad = math.radians(rotation_angle_deg)

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
                # Pad with the mean DP: preserves realistic DP statistics so
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
                mean_pattern = (acc.reshape(det_h, det_w).astype(cp.float64)
                                / flat.shape[0]).astype(data.dtype)
                padded = cp.broadcast_to(
                    mean_pattern[None, None], (target, target, det_h, det_w),
                ).copy()
                padded[pad_top:pad_top + H, pad_left:pad_left + W] = data
                data = padded

        # Handle scalar sampling values
        if isinstance(scan_sampling, (int, float)):
            scan_sampling = (float(scan_sampling), float(scan_sampling))

        # The bright-field disk edge sits at the semiangle, which calibrates the detector.
        if det_sampling is None:
            det_sampling = semiangle / disk_edge_radius(cp.asnumpy(mean_dp(data)))
            det_sampling = (det_sampling, det_sampling)
        elif isinstance(det_sampling, (int, float)):
            det_sampling = (float(det_sampling), float(det_sampling))
        if aberrations is None:
            aberrations = {"C10": 0.0, "C12": 0.0, "phi12": 0.0}

        # Store user parameters
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
        wavelength = wavelength_A_from_kV(voltage_kV)

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

        # The raw counts stay in their native dtype (usually uint16): the selection reduces them with an integer
        # accumulator and the spectra cast only the selected pixels to complex64, never the full 4D block.
        self.bf_inds_row, self.bf_inds_col, self.bf_center = select_bright_field(
            data, bf_intensity_threshold, bf_radius, bf_center,
        )
        self.G_qk, self.dc_value = bright_field_spectra(
            data, self.bf_inds_row, self.bf_inds_col, scan_gpts, det_gpts,
        )
        del data
        cp.get_default_memory_pool().free_all_blocks()

        self._scan_shape = scan_gpts

        # scan spatial frequencies (1/A)
        q_row_1d = cp.fft.fftfreq(scan_gpts[0], scan_sampling[0]).astype(cp.float32)
        q_col_1d = cp.fft.fftfreq(scan_gpts[1], scan_sampling[1]).astype(cp.float32)
        self.q_row, self.q_col = cp.meshgrid(q_row_1d, q_col_1d, indexing='ij')

        # Optimization state
        self._best_loss: float = float('inf')
        self._accelerator: SSBEngine | None = None
        self._elapsed_optimize: float = 0.0
        self._elapsed_refine: float = 0.0
        self._refine_method: str | None = None
        self._refine_nfev: int | None = None
        self._n_trials: int | None = None
        self._trial_records: list[dict] = []

    def _free_buffers(self) -> None:
        """Free optimization buffers, keep G_qk for reconstruction."""
        import gc
        if self._accelerator is not None:
            self._accelerator.release_reconstruction_buffers()
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
            # Interactive preview/export may be the first operation, before
            # optimize or result has prepared rotation-dependent geometry.
            self._accelerator.cache_rotation(self._rotation_angle_rad)
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

    def result(self, *, compute_loss: bool = True) -> SSBResult:
        """Reconstruct the complex object at the current aberrations, with the phase-variance loss unless disabled.

        Each call reconstructs from scratch (after ``optimize()`` and ``refine()`` this is the final reconstruction).
        """
        accel = self._get_accelerator()
        accel.cache_rotation(self._rotation_angle_rad)
        try:
            obj = accel.reconstruct_object(
                self.aberrations["C10"], self.aberrations["C12"], self.aberrations["phi12"],
            )
        except cp.cuda.memory.OutOfMemoryError:
            num_bf = len(self.bf_inds_row)
            free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
            raise MemoryError(
                f"Out of GPU VRAM during SSB reconstruction "
                f"({num_bf} BF pixels, {free_gb:.1f} GB free).\n"
                f"Try: SSB(..., bf_radius=<smaller>) to reduce BF pixel count, "
                f"or restart the kernel to free stale GPU memory."
            ) from None
        loss = None
        if compute_loss:
            # The fits' evaluator: a fixed-order reduction, so the loss depends only on the data and the aberrations.
            _, loss = accel.reconstruct_with_loss(
                self.aberrations["C10"], self.aberrations["C12"], self.aberrations["phi12"],
            )
        elapsed = self._elapsed_optimize + self._elapsed_refine
        scan_sampling = self.scan_sampling
        if isinstance(scan_sampling, tuple) and scan_sampling[0] == scan_sampling[1]:
            scan_sampling = scan_sampling[0]
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
            scan_sampling_A=scan_sampling,
            bf_center=brightfield.center_row_col,
            bf_radius=brightfield.radius_px,
            detected_bf_radius=brightfield.detected_radius_px,
            trial_records=self._trial_records,
        )

    def fit(
        self,
        *,
        aberrations: dict[str, float] | None,
        trials: int,
        refinement: str | None,
        search_ranges: dict[str, tuple[float, float] | float] | None,
        refine_lock: list[str] | None,
        seed: int,
        verbose: bool,
    ) -> SSBResult:
        """Run the shared exact optimization contract on CUDA."""

        if aberrations is not None:
            self.aberrations = dict(aberrations)
        # Each call owns a new search record, including a zero-trial refinement.
        # Replacing the list preserves records already returned to the caller.
        self._trial_records = []
        self._n_trials = int(trials)
        self._refine_method = None
        self._refine_nfev = None
        self._elapsed_optimize = 0.0
        self._elapsed_refine = 0.0
        if trials:
            self.optimize(
                aberrations=search_ranges,
                n_trials=int(trials),
                seed=int(seed),
                verbose=verbose,
            )
        if refinement == "nelder-mead":
            self.refine(
                verbose=verbose,
                lock=refine_lock,
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
    ) -> tuple[cp.ndarray, float | None]:
        """Return one device-resident float32 phase and optional exact loss."""

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
        return phase, None if loss is None else float(loss)

    def preview_upsampled(
        self,
        aberrations: dict[str, float],
        *,
        upsampling_factor: int,
        compute_loss: bool,
        tilt_mrad: tuple[float, float] = (0.0, 0.0),
        thickness: float = 0.0,
        phase_estimator: str = "mean_phase",
    ) -> tuple[cp.ndarray, float | None]:
        """Upsampled depth-aware SSB with the diagnostic loss kept on the native grid."""
        accel = self._get_accelerator()
        accel.cache_rotation(self._rotation_angle_rad)
        args = tuple(aberrations[key] for key in ("C10", "C12", "phi12"))
        phase, _ = accel.thick.reconstruct(
            *args, tilt_mrad, thickness, compute_loss=False,
            upsampling_factor=upsampling_factor,
            phase_estimator=phase_estimator,
        )
        loss = None
        if compute_loss:
            if thickness > 0:
                _, loss = accel.thick.reconstruct(
                    *args, tilt_mrad, thickness, compute_loss=True,
                )
            else:
                _, loss = self.reconstruct_with_loss(*args)
        return phase, loss

    def preview_sample(
        self,
        aberrations: dict[str, float],
        sample: dict[str, float],
        *,
        compute_loss: bool,
    ) -> tuple[cp.ndarray, float | None]:
        """Phase (and phase-variance loss) for a thick, tilted sample: ``ThickSample.reconstruct``.

        ``sample`` = {"tilt_row_mrad", "tilt_col_mrad", "thickness"} (thickness in the C10 unit; 0 = standard SSB exactly)."""
        accel = self._get_accelerator(); accel.cache_rotation(self._rotation_angle_rad)
        phase, loss = accel.thick.reconstruct(
            aberrations["C10"], aberrations["C12"], aberrations["phi12"],
            (float(sample.get("tilt_row_mrad", 0.0)), float(sample.get("tilt_col_mrad", 0.0))),
            float(sample.get("thickness", 0.0)), compute_loss=compute_loss,
        )
        return phase, loss

    def fit_sample(self, **options) -> dict[str, object]:
        """Fit aberrations, sample tilt and thickness together (``optimizer.fit_sample``)."""
        accel = self._get_accelerator(); accel.cache_rotation(self._rotation_angle_rad)
        return fit_sample(accel, **options)

    def close(self) -> None:
        """Release CUDA-owned session state."""

        self.free()

    def _print_summary(self, stage: str, elapsed: float) -> None:
        """Print one-line optimization summary."""
        a = self.aberrations   # engine units (Angstrom); the public API and this line report nm
        print(
            f"  {stage}: loss={self._best_loss:.6f}  "
            f"C10={a['C10'] / 10.0:.2f} nm  C12={a['C12'] / 10.0:.2f} nm  "
            f"phi12={math.degrees(a['phi12']):.1f}°  "
            f"{elapsed:.1f}s"
        )

    # =====================================================================
    #  Optimization
    # =====================================================================

    def optimize(
        self,
        aberrations: dict[str, tuple[float, float] | float] | None = None,
        n_trials: int = 200,
        seed: int = 42,
        verbose: bool = True,
    ) -> Self:
        """Global search of C10, C12, phi12 with Optuna TPE on the exact phase-variance objective.

        Evaluates ``n_trials`` candidates in batches of 4 on the GPU (lower loss is a better correction). This finds the
        right region of parameter space but may be ~5 nm off on C10; ``refine()`` then walks to the exact minimum.

        Parameters
        ----------
        aberrations : dict, optional
            Search ranges per parameter, keys ``"C10_nm"``, ``"C12_nm"``, ``"phi12_deg"`` (engine Angstrom despite the
            names): a ``(low, high)`` tuple to search, or a fixed float to lock the parameter. None searches C10
            +-400, C12 0-100 and phi12 +-90 degrees.
        n_trials : int, default 200
            Number of Optuna trials; below 200 the exploration of the (C10, C12, phi12) loss landscape becomes
            unreliable.
        seed : int, default 42
            Seed of the TPE sampler.
        verbose : bool, default True
            Print the progress bar and the memory header.
        """
        t0 = time.perf_counter()
        if aberrations is None:
            aberrations = dict(self._DEFAULT_OPTIMIZE_RANGES)
        accel = self._get_accelerator()
        accel.cache_rotation(self._rotation_angle_rad)
        accel.objective.loss(0, 50, 0)
        cp.cuda.Device().synchronize()
        if verbose:
            vram_free, vram_total = cp.cuda.runtime.memGetInfo()
            free_gb, total_gb = vram_free / 1e9, vram_total / 1e9
            print(f"Optimizing aberrations ({n_trials} trials, {int(accel.num_bf)} BF pixels)")
            print(f"  VRAM: {free_gb:.1f} GB available of {total_gb:.1f} GB")
        try:
            best_params, best_value, trial_history = batch_optimize(
                accel.objective,
                aberrations=aberrations,
                aberration_defaults=self.aberrations,
                n_trials=n_trials,
                batch_size=4,
                seed=seed,
                verbose=verbose,
            )
        except cp.cuda.memory.OutOfMemoryError:
            num_bf = len(self.bf_inds_row)
            free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
            raise MemoryError(
                f"Out of GPU VRAM during SSB optimization "
                f"({num_bf} BF pixels, {free_gb:.1f} GB free).\n"
                f"Try: SSB(..., bf_radius=<smaller>) to reduce BF pixel count, "
                f"or restart the kernel to free stale GPU memory."
            ) from None
        # the full trial history is persisted with the result (about 5 KB for 200 trials)
        self._trial_records = trial_history
        self._best_loss = best_value
        # an optimized parameter takes its best value; a locked one keeps the value it was given
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
    ) -> Self:
        """Local Nelder-Mead refinement from the current aberrations on the exact phase-variance objective.

        Walks downhill to the nearest minimum (``optimize()`` first finds the right region). Nelder-Mead is
        derivative-free and handles the scale mismatch between C10 and phi12 through the simplex geometry; the phi12
        direction has weak curvature when C12 is small, where finite-difference gradients are noisy. On a 512 x 512
        scan from a good Optuna start it converges in about 30 evaluations.

        Parameters
        ----------
        verbose : bool, default True
            Print the summary at the end.
        xatol : float, default 0.1
            Stop when the simplex spread is below this (C10, C12 in the engine unit; phi12 in radians).
        fatol : float, default 1e-8
            Stop when the loss spread across the simplex vertices is below this.
        lock : list[str], optional
            Aberrations to hold fixed; locked refinement is not implemented on the GPU and raises.
        """
        t0 = time.perf_counter()
        accel = self._get_accelerator()
        accel.cache_rotation(self._rotation_angle_rad)
        lock = set(lock or [])
        all_keys = ["C10", "C12", "phi12"]
        free_keys = [k for k in all_keys if k not in lock]
        x0 = np.array([self.aberrations[k] for k in free_keys])
        if free_keys != all_keys:
            locked = ", ".join(sorted(lock)) or "unknown"
            raise ValueError(
                "GPU-only SSB Nelder-Mead does not support locked refinement yet "
                f"(locked: {locked}). Use no locked aberrations with "
                "refinement='nelder-mead', "
                "locked SSB reference mode, or add a GPU-batched locked refiner."
            )
        effective_xatol = xatol
        effective_fatol = fatol
        effective_max_iter = 300
        if xatol == 0.1 and fatol == 1e-8:
            # The exact full-IFFT objective is smooth enough on a 512 full-BF
            # acquisition that tighter generic tolerances over-solve the phase
            # by hundreds of evaluations. These defaults keep the phase image
            # within 1e-3 rad p99.9 in the real-data signoff while avoiding an
            # invisible 200+ evaluation tail.
            effective_xatol = 0.25
            effective_fatol = 2e-6
            effective_max_iter = 160
        best_x, best_loss, n_evals = batch_nelder_mead(
            accel.objective,
            x0.astype(np.float64),
            xatol=effective_xatol,
            fatol=effective_fatol,
            max_iter=effective_max_iter,
        )
        for i, k in enumerate(free_keys):
            self.aberrations[k] = float(best_x[i])
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
