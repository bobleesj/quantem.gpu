"""Private CUDA compute implementation for the public SSB workflow."""

import gc
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
        padded with the mean pattern to the next of those sizes, with a printed notice; scans larger than 1024
        raise ``ValueError`` rather than being cropped.
    semiangle : float
        Probe convergence semiangle in mrad.
    scan_sampling : float or (float, float)
        Scan step in Angstrom, (row, col).
    detector_sampling_mrad : float or (float, float), optional
        Detector sampling in mrad per pixel; ``semiangle / disk_edge_radius(mean pattern)`` when omitted.
    voltage_kV : float
        Accelerating voltage in kV.
    bright_field_intensity_threshold, bright_field_radius_px, bright_field_center_px
        Bright-field selection (``cuda.brightfield.select_bright_field``).
    aberrations : dict, optional
        Starting C10, C12 (Angstrom) and phi12 (radians); zero when omitted.
    scan_detector_rotation_deg : float
        Physical scan-detector rotation in degrees.

    Out of memory: the resident Hermitian ``G_qk`` costs about ``scan_row x (scan_col / 2 + 1) x 8`` bytes per
    bright-field pixel; a smaller ``bright_field_radius_px`` reduces it.
    """

    # search ranges of optimize() when none are given, in engine units
    _DEFAULT_OPTIMIZE_RANGES: ClassVar[dict[str, tuple[int, int]]] = {
        "C10_A": (-400, 400),
        "C12_A": (0, 100),
        "phi12_deg": (-90, 90),
    }
    backend = "cuda"
    precision = SSBPrecision()

    def __init__(
        self,
        data: object,
        semiangle: float,
        scan_sampling: float | tuple[float, float],
        detector_sampling_mrad: float | tuple[float, float] | None = None,
        *,
        voltage_kV: float,
        scan_shape: tuple[int, int] | None = None,
        bright_field_intensity_threshold: float = 0.0,
        bright_field_radius_px: int | None = None,
        bright_field_center_px: tuple[float, float] | None = None,
        aberrations: dict[str, float] | None = None,
        scan_detector_rotation_deg: float = 0.0,
    ):
        scan_detector_rotation_rad = math.radians(scan_detector_rotation_deg)

        # quantem.gpu SSB is GPU-only. Ensure the input is a CuPy array but
        # keep the native dtype. Casting the raw 4D block to float32 would
        # double memory (e.g. 19 GB uint16 -> 38 GB float32 copy on a
        # 512x512x192x192 scan). Reductions promote internally via uint64
        # accumulators, so the raw block stays in its source dtype.
        data = cp.asarray(data)
        if data.ndim == 3:
            if scan_shape is None:
                num_frames = data.shape[0]
                side = int(num_frames ** 0.5)
                if side * side != num_frames:
                    raise ValueError(
                        f"scan_shape is required: {num_frames} frames is not a perfect square. "
                        f"Pass scan_shape=(rows, cols)."
                    )
                scan_shape = (side, side)
            if scan_shape[0] * scan_shape[1] != data.shape[0]:
                raise ValueError("scan_shape does not match number of frames.")
            data = data.reshape(scan_shape[0], scan_shape[1], data.shape[1], data.shape[2])
        elif data.ndim != 4:
            raise ValueError("data must be 3D or 4D.")

        # The CUDA FFT kernels take square scans of 128, 256, 512 or 1024 positions. A smaller or non-square scan is
        # padded with its mean pattern to the next of those sizes so callers can pass arbitrary scans (simulations,
        # drift-corrected cubes); every measured position is kept and the padding is announced. A scan larger than
        # 1024 is refused: fitting it would mean dropping measured positions.
        scan_rows, scan_cols = int(data.shape[0]), int(data.shape[1])
        if max(scan_rows, scan_cols) > 1024:
            raise ValueError(
                f"CUDA SSB reconstructs scans of at most 1024 x 1024 positions; got {scan_rows}x{scan_cols}. Select a "
                "square region of 128, 256, 512 or 1024 positions per side, or use backend='mps', which takes any "
                "square scan. SSB never crops a scan."
            )
        if (scan_rows, scan_cols) not in ((128, 128), (256, 256), (512, 512), (1024, 1024)):
            target = next(side for side in (128, 256, 512, 1024) if max(scan_rows, scan_cols) <= side)
            print(
                f"CUDA SSB: padded the {scan_rows}x{scan_cols} scan to {target}x{target} with its mean pattern, as the "
                "CUDA FFT takes 128, 256, 512 or 1024 square scans; the phase covers the padded grid. Pass a square "
                "scan of one of those sizes to reconstruct without padding."
            )
            # Pad with the mean DP: preserves realistic DP statistics so
            # probe detection + BF/DF integrals stay well-conditioned.
            # Chunked int64 sum avoids the 4× float32 transient that
            # `data.reshape(...).mean()` would allocate (would OOM on
            # 17 GB cube → 68 GB transient).
            pad_top = (target - scan_rows) // 2
            pad_left = (target - scan_cols) // 2
            det_rows, det_cols = data.shape[2], data.shape[3]
            flat = data.reshape(-1, det_rows * det_cols)
            is_integer = np.issubdtype(data.dtype, np.integer)
            sum_dtype = cp.int64 if is_integer else cp.float64
            pattern_sum = cp.zeros(det_rows * det_cols, dtype=sum_dtype)
            for start in range(0, flat.shape[0], 16 * scan_cols):
                pattern_sum += flat[start:start + 16 * scan_cols].astype(sum_dtype).sum(axis=0)
            mean_pattern = (pattern_sum.reshape(det_rows, det_cols).astype(cp.float64)
                            / flat.shape[0]).astype(data.dtype)
            padded = cp.broadcast_to(
                mean_pattern[None, None], (target, target, det_rows, det_cols),
            ).copy()
            padded[pad_top:pad_top + scan_rows, pad_left:pad_left + scan_cols] = data
            data = padded

        if isinstance(scan_sampling, (int, float)):
            scan_sampling = (float(scan_sampling), float(scan_sampling))

        # The bright-field disk edge sits at the semiangle, which calibrates the detector.
        if detector_sampling_mrad is None:
            detector_sampling_mrad = semiangle / disk_edge_radius(cp.asnumpy(mean_dp(data)))
            detector_sampling_mrad = (detector_sampling_mrad, detector_sampling_mrad)
        elif isinstance(detector_sampling_mrad, (int, float)):
            detector_sampling_mrad = (float(detector_sampling_mrad), float(detector_sampling_mrad))
        if aberrations is None:
            aberrations = {"C10": 0.0, "C12": 0.0, "phi12": 0.0}
        self.voltage_kV = voltage_kV
        self.semiangle_mrad = semiangle
        self.scan_sampling = scan_sampling
        self.angular_sampling = detector_sampling_mrad
        self.bright_field_intensity_threshold = float(bright_field_intensity_threshold)
        self.aberrations = aberrations.copy()
        self._scan_detector_rotation_rad = scan_detector_rotation_rad
        scan_gpts = data.shape[:2]
        det_gpts = data.shape[2:]
        wavelength = wavelength_A_from_kV(voltage_kV)

        # Convert detector sampling: mrad -> reciprocal space
        reciprocal_sampling = (
            detector_sampling_mrad[0] * 1e-3 / wavelength,
            detector_sampling_mrad[1] * 1e-3 / wavelength,
        )
        sampling = (
            1.0 / (reciprocal_sampling[0] * det_gpts[0]),
            1.0 / (reciprocal_sampling[1] * det_gpts[1]),
        )
        self.gpts = det_gpts
        self.wavelength = wavelength
        self.sampling = sampling

        # The raw counts stay in their native dtype (usually uint16): the selection reduces them with an integer
        # accumulator and the spectra cast only the selected pixels to complex64, never the full 4D block.
        (self.bf_inds_row, self.bf_inds_col, self.bright_field_center_px,
         self.bright_field_radius_px) = select_bright_field(
            data, bright_field_intensity_threshold, bright_field_radius_px, bright_field_center_px,
        )
        self.G_qk, self.dc_value = bright_field_spectra(
            data, self.bf_inds_row, self.bf_inds_col, scan_gpts, det_gpts,
        )
        del data
        cp.get_default_memory_pool().free_all_blocks()

        self._scan_shape = scan_gpts

        # scan spatial frequencies (1/A), float32 as the kernels read them
        q_row_1d = cp.fft.fftfreq(scan_gpts[0], scan_sampling[0]).astype(cp.float32)
        q_col_1d = cp.fft.fftfreq(scan_gpts[1], scan_sampling[1]).astype(cp.float32)
        self.q_row, self.q_col = cp.meshgrid(q_row_1d, q_col_1d, indexing='ij')
        self._best_loss: float = float('inf')
        self._accelerator: SSBEngine | None = None
        self._elapsed_optimize: float = 0.0
        self._elapsed_refine: float = 0.0
        self._refine_method: str | None = None
        self._refine_nfev: int | None = None
        self._num_trials: int | None = None
        self._trial_records: list[dict] = []

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
        if self._accelerator is not None:
            self._accelerator.release_reconstruction_buffers()
        self.G_qk = None
        if self._accelerator is not None:
            # the geometry arrays too, so the pool flush below returns their memory
            self._accelerator._cache.clear()
            self._accelerator = None
        # ThickSample refers back to its engine. Collect that cycle only after
        # dropping our owner reference, before returning unused blocks to CUDA.
        gc.collect()
        cp.get_default_memory_pool().free_all_blocks()

    def _get_accelerator(self) -> SSBEngine:
        """Get or create CuPy accelerator."""
        if self._accelerator is None:
            self._accelerator = SSBEngine(
                G_qk=self.G_qk,
                bf_inds_row=self.bf_inds_row,
                bf_inds_col=self.bf_inds_col,
                bright_field_center_px=self.bright_field_center_px,
                bright_field_radius_px=self.bright_field_radius_px,
                dc_value=self.dc_value,
                gpts=self.gpts,
                sampling=self.sampling,
                q_row=self.q_row,
                q_col=self.q_col,
                wavelength=self.wavelength,
                semiangle_cutoff=self.semiangle_mrad,
                angular_sampling=self.angular_sampling,
            )
            # Interactive preview/export may be the first operation, before
            # optimize or result has prepared rotation-dependent geometry.
            self._accelerator.cache_rotation(self._scan_detector_rotation_rad)
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
    def num_bright_field_pixels(self) -> int:
        """Number of pixels in the complete detected bright-field disk."""

        return int(self.bf_inds_row.size)

    def cache_rotation(self, scan_detector_rotation_rad: float) -> None:
        """Prepare CUDA geometry for one scan-to-detector rotation."""

        self._scan_detector_rotation_rad = float(scan_detector_rotation_rad)
        self._get_accelerator().cache_rotation(self._scan_detector_rotation_rad)

    def preview_context(self, num_bright_field_pixels: int):
        """Prepare a reusable reduced-BF CUDA preview context."""

        return self._get_accelerator().prepare_bf_subset(int(num_bright_field_pixels))

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
        try:
            object_wave = accel.reconstruct_object(
                self.aberrations["C10"], self.aberrations["C12"], self.aberrations["phi12"],
            )
        except cp.cuda.memory.OutOfMemoryError:
            raise self._out_of_memory("reconstruction") from None
        loss = None
        if compute_loss:
            # The fits' evaluator: a fixed-order reduction, so the loss depends only on the data and the aberrations.
            _, loss = accel.reconstruct_with_loss(
                self.aberrations["C10"], self.aberrations["C12"], self.aberrations["phi12"],
            )
        elapsed_s = self._elapsed_optimize + self._elapsed_refine
        brightfield = self.browser_state().brightfield
        return SSBResult(
            object_wave=object_wave,
            backend="cuda",
            aberrations=self.aberrations.copy(),
            scan_detector_rotation_deg=math.degrees(self._scan_detector_rotation_rad),
            loss=loss,
            elapsed_s=elapsed_s if elapsed_s > 0 else None,
            num_trials=self._num_trials,
            num_bright_field_pixels=len(self.bf_inds_row),
            refine_method=self._refine_method,
            refine_nfev=self._refine_nfev,
            refine_elapsed_s=self._elapsed_refine if self._elapsed_refine > 0 else None,
            voltage_kV=self.voltage_kV,
            semiangle_mrad=self.semiangle_mrad,
            object_sampling_A=tuple(float(value) for value in self.scan_sampling),
            bright_field_center_px=brightfield.bright_field_center_px,
            bright_field_radius_px=brightfield.bright_field_radius_px,
            bright_field_edge_radius_px=brightfield.bright_field_edge_radius_px,
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
        self._num_trials = int(trials)
        self._refine_method = None
        self._refine_nfev = None
        self._elapsed_optimize = 0.0
        self._elapsed_refine = 0.0
        if trials:
            self.optimize(
                search_ranges=search_ranges,
                num_trials=int(trials),
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
        accel = self._get_accelerator()
        coefficients = aberrations["C10"], aberrations["C12"], aberrations["phi12"]
        if higher_order_magnitudes is not None:
            if compute_loss:
                phase, loss = accel.reconstruct_full_with_loss(higher_order_magnitudes, higher_order_angles)
            else:
                phase, loss = accel.reconstruct_full(higher_order_magnitudes, higher_order_angles), None
        elif compute_loss:
            phase, loss = accel.reconstruct_with_loss(*coefficients)
        else:
            phase, loss = accel.reconstruct(*coefficients), None
        return phase, None if loss is None else float(loss)

    def preview_upsampled(
        self,
        aberrations: dict[str, float],
        *,
        upsampling_factor: int,
        compute_loss: bool,
        sample_tilt_mrad: tuple[float, float] = (0.0, 0.0),
        thickness: float = 0.0,
        phase_estimator: str = "mean_phase",
        higher_order_magnitudes: np.ndarray | None = None,
        higher_order_angles: np.ndarray | None = None,
    ) -> tuple[cp.ndarray, float | None]:
        """Upsampled depth-aware SSB with the diagnostic loss kept on the native grid."""
        accel = self._get_accelerator()
        coefficients = tuple(aberrations[key] for key in ("C10", "C12", "phi12"))
        phase, _ = accel.thick.reconstruct(
            *coefficients, sample_tilt_mrad, thickness, compute_loss=False,
            upsampling_factor=upsampling_factor,
            phase_estimator=phase_estimator,
            higher_order_magnitudes=higher_order_magnitudes,
            higher_order_angles=higher_order_angles,
        )
        loss = None
        if compute_loss:
            if thickness > 0 or higher_order_magnitudes is not None:
                _, loss = accel.thick.reconstruct(
                    *coefficients, sample_tilt_mrad, thickness, compute_loss=True,
                    higher_order_magnitudes=higher_order_magnitudes,
                    higher_order_angles=higher_order_angles,
                )
            else:
                _, loss = accel.reconstruct_with_loss(*coefficients)
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
        phase, loss = self._get_accelerator().thick.reconstruct(
            aberrations["C10"], aberrations["C12"], aberrations["phi12"],
            (float(sample.get("tilt_row_mrad", 0.0)), float(sample.get("tilt_col_mrad", 0.0))),
            float(sample.get("thickness", 0.0)), compute_loss=compute_loss,
        )
        return phase, loss

    def fit_sample(self, **options) -> dict[str, object]:
        """Fit aberrations, sample tilt and thickness together (``optimizer.fit_sample``)."""
        started = time.perf_counter()
        engine = self._get_accelerator()
        preparation_s = time.perf_counter() - started
        result = fit_sample(engine, **options)
        result.setdefault("timings", {})["fft_preparation_s"] = preparation_s
        return result

    def sample_agreement(self, parameters: np.ndarray, band_inv_A: tuple[float, float]) -> np.ndarray:
        """Evaluate the joint objective on the session's existing evidence."""
        return self._get_accelerator().thick.fit_batch(parameters, band_inv_A)

    def close(self) -> None:
        """Release CUDA-owned session state."""

        self.free()

    def _print_summary(self, stage: str, elapsed_s: float) -> None:
        """Print one-line optimization summary."""
        aberrations = self.aberrations   # engine units (Angstrom); the public API and this line report nm
        print(
            f"  {stage}: loss={self._best_loss:.6f}  "
            f"C10={aberrations['C10'] / 10.0:.2f} nm  C12={aberrations['C12'] / 10.0:.2f} nm  "
            f"phi12={math.degrees(aberrations['phi12']):.1f}°  "
            f"{elapsed_s:.1f}s"
        )

    def _out_of_memory(self, stage: str) -> MemoryError:
        """The error that replaces a device out-of-memory failure during ``stage``.

        It names the two numbers that decide whether SSB fits on the device, the bright-field pixel count (a smaller
        ``bright_field_radius_px`` reduces it) and the free device memory (stale arrays in the session hold it).
        """
        num_bright_field_pixels = len(self.bf_inds_row)
        free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
        return MemoryError(
            f"Out of GPU VRAM during SSB {stage} "
            f"({num_bright_field_pixels} BF pixels, {free_gb:.1f} GB free).\n"
            f"Try: SSB(..., bright_field_radius_px=<smaller>) to reduce BF pixel count, "
            f"or restart the kernel to free stale GPU memory."
        )

    # =====================================================================
    #  Optimization
    # =====================================================================

    def optimize(
        self,
        search_ranges: dict[str, tuple[float, float] | float] | None = None,
        num_trials: int = 200,
        seed: int = 42,
        verbose: bool = True,
    ) -> Self:
        """Global search of C10, C12, phi12 with Optuna TPE on the exact phase-variance objective.

        Evaluates ``num_trials`` candidates in batches of 4 on the GPU (lower loss is a better correction). This finds the
        right region of parameter space but may be ~5 nm off on C10; ``refine()`` then walks to the exact minimum.

        Parameters
        ----------
        search_ranges : dict, optional
            Search ranges per parameter, keys ``"C10_A"``, ``"C12_A"`` (Angstrom) and ``"phi12_deg"``: a
            ``(low, high)`` tuple to search, or a fixed float to lock the parameter. None searches C10
            +-400, C12 0-100 and phi12 +-90 degrees.
        num_trials : int, default 200
            Number of Optuna trials; below 200 the exploration of the (C10, C12, phi12) loss landscape becomes
            unreliable.
        seed : int, default 42
            Seed of the TPE sampler.
        verbose : bool, default True
            Print the progress bar and the memory header.
        """
        started = time.perf_counter()
        if search_ranges is None:
            search_ranges = dict(self._DEFAULT_OPTIMIZE_RANGES)
        accel = self._get_accelerator()
        accel.reconstruct_with_loss(0, 50, 0)
        cp.cuda.Device().synchronize()
        if verbose:
            vram_free, vram_total = cp.cuda.runtime.memGetInfo()
            free_gb, total_gb = vram_free / 1e9, vram_total / 1e9
            print(f"Optimizing aberrations ({num_trials} trials, {accel.num_bright_field_pixels} BF pixels)")
            print(f"  VRAM: {free_gb:.1f} GB available of {total_gb:.1f} GB")
        try:
            best_params, best_value, trial_history = batch_optimize(
                accel,
                aberrations=search_ranges,
                aberration_defaults=self.aberrations,
                num_trials=num_trials,
                batch_size=4,
                seed=seed,
                verbose=verbose,
            )
        except cp.cuda.memory.OutOfMemoryError:
            raise self._out_of_memory("optimization") from None
        # the full trial history is persisted with the result (about 5 KB for 200 trials)
        self._trial_records = trial_history
        self._best_loss = best_value
        # an optimized parameter takes its best value; a locked one keeps the value it was given
        for search_key, aberration_key, convert in [
            ("C10_A", "C10", None),
            ("C12_A", "C12", None),
            ("phi12_deg", "phi12", math.radians),
        ]:
            if search_key in best_params:
                value = best_params[search_key]
            elif search_key in search_ranges and not isinstance(search_ranges[search_key], tuple):
                value = search_ranges[search_key]
            else:
                continue
            self.aberrations[aberration_key] = convert(value) if convert else value
        self._elapsed_optimize = time.perf_counter() - started
        self._num_trials = num_trials
        if verbose:
            self._print_summary("Optimize", self._elapsed_optimize)
        return self

    def refine(
        self,
        verbose: bool = True,
        xatol: float = 0.25,
        fatol: float = 2e-6,
        max_iter: int = 160,
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
        xatol : float, default 0.25
            Stop when the simplex spread is below this (C10, C12 in the engine unit; phi12 in radians).
        fatol : float, default 2e-6
            Stop when the loss spread across the simplex vertices is below this.
        max_iter : int, default 160
            Most simplex iterations. The exact objective is smooth enough on a 512 full-BF acquisition that tighter
            tolerances over-solve the phase by hundreds of evaluations; these keep the phase within 1e-3 rad p99.9 in
            the real-data signoff.
        lock : list[str], optional
            Aberrations to hold fixed; locked refinement is not implemented on the GPU and raises.
        """
        started = time.perf_counter()
        accel = self._get_accelerator()
        lock = set(lock or [])
        all_keys = ["C10", "C12", "phi12"]
        free_keys = [key for key in all_keys if key not in lock]
        x0 = np.array([self.aberrations[key] for key in free_keys])
        if free_keys != all_keys:
            locked = ", ".join(sorted(lock)) or "unknown"
            raise ValueError(
                "GPU-only SSB Nelder-Mead does not support locked refinement yet "
                f"(locked: {locked}). Use no locked aberrations with "
                "refinement='nelder-mead', "
                "locked SSB reference mode, or add a GPU-batched locked refiner."
            )
        best_x, best_loss, n_evals = batch_nelder_mead(
            accel,
            x0.astype(np.float64),  # the simplex is float64 even for integer starting coefficients
            xatol=xatol,
            fatol=fatol,
            max_iter=max_iter,
        )
        for index, key in enumerate(free_keys):
            self.aberrations[key] = float(best_x[index])
        self._best_loss = float(best_loss)
        nfev = n_evals
        method = "nelder-mead"
        elapsed_s = time.perf_counter() - started
        self._elapsed_refine = elapsed_s
        self._refine_method = method
        self._refine_nfev = nfev
        if verbose:
            self._print_summary(f"Refine ({method}, {nfev} evals)", elapsed_s)
        return self
