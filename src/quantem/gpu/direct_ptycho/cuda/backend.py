"""Private CUDA compute implementation for the public Direct Ptycho workflow."""

import math
import time

import cupy as cp
import numpy as np

from quantem.gpu.detector.cuda.probe import mean_dp
from quantem.gpu.optics.physics import wavelength_A_from_kV
from quantem.gpu.detector import disk_edge_radius
from quantem.gpu.direct_ptycho.contract import SSBExportState, SSBPrecision
from quantem.gpu.direct_ptycho.cuda.brightfield import bright_field_spectra, select_bright_field
from quantem.gpu.direct_ptycho.cuda.engine import SSBEngine
from quantem.gpu.direct_ptycho.cuda.optimizer import fit_sample, fit_standard
from quantem.gpu.direct_ptycho.results import SSBResult


class CudaSSBBackend:
    """The CUDA implementation of the Direct Ptycho backend protocol behind :class:`quantem.gpu.DirectPtycho`.

    Each bright-field pixel sees the sample through a slightly different probe tilt; correcting every pixel's image
    for the probe aberrations and averaging recovers the object phase at the scan resolution. This class prepares the
    bright-field spectra ``G_qk`` on the device and runs fits and reconstructions through ``SSBEngine``. Units are the
    engine's (C10, C12 in Angstrom); ``DirectPtycho`` owns the public units, validation and result types.

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

        # quantem.gpu Direct Ptycho is GPU-only. Ensure the input is a CuPy array but
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
        Release all GPU VRAM held by this Direct Ptycho engine.

        Frees:
        - G_qk (the FFT of virtual BF stack - the largest allocation, ~7 GB)
        - Engine buffers (correction pipeline, variance computation)
        - Batch caches (optimizer working memory)

        After this call, ``result()`` and ``fit()`` will fail - the
        engine is no longer usable. Previously returned ``SSBResult``
        objects remain valid (they hold independent copies of the phase).

        Call this when the Direct Ptycho pipeline is done and you need VRAM for
        the next stage (e.g., iterative ptychography).
        """
        if self._accelerator is not None:
            self._accelerator.release_reconstruction_buffers()
        self.G_qk = None
        if self._accelerator is not None:
            # the geometry arrays too, so the pool flush below returns their memory
            self._accelerator._cache.clear()
            # Break the known ownership cycle directly. A global collection
            # would also scan unrelated notebook objects on every region close.
            self._accelerator.thick.engine = None
            self._accelerator = None
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
        """The SSB phase at the current aberrations (``fit()``'s final reconstruction), with the phase-variance loss.

        The phase is the native-grid phase of the bright-field mean wave, as ``preview_upsampled`` computes it at
        factor 1, held as a unit-amplitude wave; ``compute_loss=False`` skips the loss.
        """
        accel = self._get_accelerator()
        try:
            phase, _ = accel.thick.reconstruct(
                self.aberrations["C10"], self.aberrations["C12"], self.aberrations["phi12"], (0.0, 0.0), 0.0,
                compute_loss=False,
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
            object_wave=cp.exp(1j * phase),
            backend="cuda",
            aberrations=self.aberrations.copy(),
            scan_detector_rotation_deg=math.degrees(self._scan_detector_rotation_rad),
            loss=loss,
            elapsed_s=elapsed_s if elapsed_s > 0 else None,
            num_trials=self._num_trials,
            num_bright_field_pixels=len(self.bf_inds_row),
            refinement=self._refine_method,
            refine_nfev=self._refine_nfev,
            refine_elapsed_s=self._elapsed_refine if self._elapsed_refine > 0 else None,
            voltage_kV=self.voltage_kV,
            semiangle_mrad=self.semiangle_mrad,
            object_sampling_A=tuple(float(value) for value in self.scan_sampling),
            bright_field_center_px=brightfield.bright_field_center_px,
            bright_field_radius_px=brightfield.bright_field_radius_px,
            bright_field_edge_radius_px=brightfield.bright_field_edge_radius_px,
            trial_records=self._trial_records,
            phase_estimator="phase_of_mean",
            amplitude_estimated=False,
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
        compute_loss: bool = True,
    ) -> SSBResult:
        """Standard fit (tilt=False) of C10, C12 and phi12, then the complex object and the phase-variance loss there.

        Maximizes the least-squares agreement of the SSB model with G(q, k) at sample tilt and depth spread 0
        (``optimizer.fit_standard``, the resident search shared with MPS; ``refinement`` "newton", the trust-region
        Newton on the exact derivatives, or "bfgs", the exact-gradient quasi-Newton, for a locked or bounded fit). ``search_ranges`` (engine
        keys) bounds the search, a value holding that coefficient in every trial; ``refine_lock`` holds coefficients
        through the refinement; ``aberrations`` (engine units) is a refinement start.
        """
        if aberrations is not None:
            self.aberrations = dict(aberrations)
        # a failed search leaves no previous fit's records or refinement attached to the new aberrations
        self._trial_records, self._num_trials, self._refine_method, self._refine_nfev = [], int(trials), None, None
        self._elapsed_optimize = self._elapsed_refine = 0.0
        accel = self._get_accelerator()
        started = time.perf_counter()
        try:
            fit = fit_standard(accel, aberrations=aberrations, search_ranges=search_ranges, num_trials=int(trials),
                               refine=refinement, refine_lock=refine_lock, seed=int(seed))
            best = {name: fit[name] for name in ("C10", "C12", "phi12")}
            search_s, refine_s = fit["timings"]["coarse_search_s"], fit["timings"]["refinement_s"]
            records = [{"stage": "search", "number": record["number"], "objective": "negative_thick_agreement",
                        "band_inv_A": fit["band_inv_A"],
                        "params": {"C10_A": record["params"]["C10"], "C12_A": record["params"]["C12"],
                                   "phi12_deg": math.degrees(record["params"]["phi12"])},
                        "loss": record["loss"]} for record in fit["trial_records"]]
            nfev, optimizer = int(fit["nfev"]), dict(fit["optimizer"])
            agreement = float(fit["fit"]) / accel.thick.band_power(tuple(fit["band_inv_A"]))
        except cp.cuda.memory.OutOfMemoryError:
            raise self._out_of_memory("optimization") from None
        self.aberrations.update(best)
        # Each call owns a new search record, including a zero-trial refinement.
        # Replacing the list preserves records already returned to the caller.
        self._trial_records = records
        self._num_trials = int(trials)
        self._refine_method = refinement
        self._refine_nfev = nfev if refinement else None
        self._elapsed_optimize = search_s
        self._elapsed_refine = refine_s
        result = self.result(compute_loss=compute_loss)
        result.sample_agreement = agreement
        result.optimizer = optimizer
        result.timings = {"search_seconds": search_s, "refinement_seconds": refine_s}
        self._best_loss = result.loss
        if verbose:
            self._print_summary(f"Fit (agreement, {refinement or 'no refinement'})",
                                time.perf_counter() - started)
        return result

    def preview_upsampled(
        self,
        aberrations: dict[str, float],
        *,
        upsampling_factor: int,
        compute_loss: bool,
        sample_tilt_mrad: tuple[float, float] = (0.0, 0.0),
        depth_spread_A: float = 0.0,
        higher_order_magnitudes: np.ndarray | None = None,
        higher_order_angles: np.ndarray | None = None,
    ) -> tuple[cp.ndarray, float | None]:
        """SSB phase (of the bright-field mean wave) at the output factor, with the diagnostic loss on the native grid.

        The loss is the phase variance over bright-field pixels: the engine's fused two-term pass for a thin sample,
        else ``ThickSample.phase_variance``.
        """
        accel = self._get_accelerator()
        coefficients = tuple(aberrations[key] for key in ("C10", "C12", "phi12"))
        phase, _ = accel.thick.reconstruct(
            *coefficients, sample_tilt_mrad, depth_spread_A, compute_loss=False,
            upsampling_factor=upsampling_factor,
            higher_order_magnitudes=higher_order_magnitudes,
            higher_order_angles=higher_order_angles,
        )
        loss = None
        if compute_loss:
            if depth_spread_A > 0 or higher_order_magnitudes is not None:
                _, loss = accel.thick.phase_variance(
                    *coefficients, sample_tilt_mrad, depth_spread_A,
                    higher_order_magnitudes=higher_order_magnitudes,
                    higher_order_angles=higher_order_angles,
                )
            else:
                _, loss = accel.reconstruct_with_loss(*coefficients)
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

    def band_power(self, band_inv_A: tuple[float, float]) -> float:
        """G's power over the band, the agreement's upper bound (``ThickSample.band_power``)."""
        return self._get_accelerator().thick.band_power(band_inv_A)

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
            f"Try: DirectPtycho(..., bright_field_radius_px=<smaller>) to reduce BF pixel count, "
            f"or restart the kernel to free stale GPU memory."
        )
