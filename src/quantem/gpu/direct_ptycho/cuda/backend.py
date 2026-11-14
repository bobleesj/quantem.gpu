"""Private CUDA compute implementation for the public direct ptychography workflow."""

import math
import time

import cupy as cp
import numpy as np
import torch

from quantem.gpu.detector.cuda.probe import mean_pattern
from quantem.gpu.optics.physics import wavelength_A_from_kV
from quantem.gpu.detector import disk_edge_radius
from quantem.gpu.direct_ptycho.contract import ExportState, BackendPrecision
from quantem.gpu.direct_ptycho.cuda.brightfield import bright_field_spectra, select_bright_field
from quantem.gpu.direct_ptycho.cuda.engine import DirectEngine
from quantem.gpu.direct_ptycho.cuda.thick import BAND_SLAB_ROWS
from quantem.gpu.direct_ptycho.cuda.optimizer import fit_sample, fit_standard
from quantem.gpu.direct_ptycho.padding import pad_scan, padded_side
from quantem.gpu.direct_ptycho.results import DirectPtychoResult



class CudaDirectBackend:
    """The CUDA implementation of the direct ptychography backend protocol behind :class:`quantem.gpu.direct`.

    Each bright-field pixel sees the sample through a slightly different probe tilt; correcting every pixel's image
    for the probe aberrations and averaging recovers the object phase at the scan resolution. This class prepares the
    bright-field spectra ``G_qk`` on the device and runs fits and reconstructions through ``DirectEngine``. Units are the
    engine's (C10, C12 in Angstrom); ``DirectPtycho`` owns the public units, validation and result types.

    Parameters
    ----------
    data : cupy.ndarray
        Counts, 3D ``(scan, det_row, det_col)`` (square scan unless ``scan_shape`` is given) or 4D
        ``(scan_row, scan_col, det_row, det_col)``, in their native dtype: reductions accumulate in uint64, and only the
        selected bright-field pixels are cast to complex64. Scans that are not 128, 256, 512 or 1024 square are
        padded with the mean pattern to the next of those sizes, with a printed line (``direct_ptycho.padding``);
        scans larger than 1024 raise ``ValueError`` rather than being cropped.
    semiangle : float
        Probe convergence semiangle in mrad.
    scan_sampling : float or (float, float)
        Scan step in Angstrom, (row, col).
    detector_sampling_mrad : float or (float, float), optional
        Detector sampling in mrad per pixel; ``semiangle / disk_edge_radius(mean pattern)`` when omitted. The value
        used, given or estimated, is the ``detector_sampling_mrad`` attribute.
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
    precision = BackendPrecision()
    # A full-field region builds its own G(q, k) from its counts on first use; there is no prepared stack to share.
    shares_full_field_preparation = False
    # the fit band whose frequencies alone G holds (from_band_spectra), else None: G is the whole half plane
    fit_band_inv_A = None
    # host copy of a band-only G, which lets release_prepared take it off the device
    _spectra_host = None
    # pixels of a band-only G on the device when the rest streams from _spectra_host (from_band_spectra), else None
    _resident_rows = None

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

        # quantem.gpu direct ptychography is GPU-only. Ensure the input is a CuPy array but
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

        # The CUDA FFT kernels take square scans of 128, 256, 512 or 1024 positions; any other scan is padded with
        # its mean pattern (``direct_ptycho.padding``). Torch and CuPy share the device buffer through DLPack.
        if padded_side(data.shape[:2], any_square=False) is not None:
            data = cp.from_dlpack(pad_scan(torch.from_dlpack(data), any_square=False))

        # The bright-field disk edge sits at the semiangle, which calibrates the detector.
        if detector_sampling_mrad is None:
            detector_sampling_mrad = semiangle / disk_edge_radius(cp.asnumpy(mean_pattern(data)))
        # The raw counts stay in their native dtype (usually uint16): the selection reduces them with an integer
        # accumulator and the spectra cast only the selected pixels to complex64, never the full 4D block.
        bf_inds_row, bf_inds_col, center, radius = select_bright_field(
            data, bright_field_intensity_threshold, bright_field_radius_px, bright_field_center_px,
        )
        scan_gpts, det_gpts = data.shape[:2], data.shape[2:]
        G_qk, dc_value = bright_field_spectra(data, bf_inds_row, bf_inds_col, scan_gpts, det_gpts)
        del data
        cp.get_default_memory_pool().free_all_blocks()
        self._hold(G_qk, dc_value, bf_inds_row, bf_inds_col, center, radius, scan_gpts, det_gpts,
                   semiangle=semiangle, scan_sampling=scan_sampling, detector_sampling_mrad=detector_sampling_mrad,
                   voltage_kV=voltage_kV, bright_field_intensity_threshold=bright_field_intensity_threshold,
                   aberrations=aberrations, scan_detector_rotation_rad=scan_detector_rotation_rad)

    @classmethod
    def from_band_spectra(cls, spectra, *, fit_band_inv_A, dc_value, bf_inds_row, bf_inds_col,
                          bright_field_center_px, bright_field_radius_px, scan_shape, detector_shape, semiangle,
                          scan_sampling, detector_sampling_mrad, voltage_kV, bright_field_intensity_threshold,
                          aberrations, scan_detector_rotation_deg, device_bytes=None) -> "CudaDirectBackend":
        """A backend for fits alone, whose G(q, k) holds only the fit band's frequencies.

        ``spectra`` is (bright-field pixels, band frequencies) in page-locked host memory, in the order
        ``thick.band_frequencies`` lists them; it stays the backend's host copy (``release_prepared``). The fits read
        G nowhere else; on a 512 x 512 scan at 0.27 A the 0.2 to 0.9 1/A band is 23,448 of the half plane's 131,584
        frequencies. ``device_bytes`` bounds the device memory G may take: all of G when it fits, else the first
        bright-field pixels that fit beside two slab buffers, the rest streamed from ``spectra`` slab by slab
        (``DirectEngine.band_slabs``). Reconstructions need every frequency and refuse (``_whole_spectrum``).
        """
        backend = cls.__new__(cls)
        backend.fit_band_inv_A = tuple(float(value) for value in fit_band_inv_A)
        backend._spectra_host = spectra
        if device_bytes is not None and spectra.nbytes > device_bytes:
            rows = max(0, int(device_bytes // (spectra.shape[1] * 8)) - 2 * BAND_SLAB_ROWS)
            backend._resident_rows = rows - rows % BAND_SLAB_ROWS
        evidence = cp.asarray(spectra[:backend._resident_rows])
        backend._hold(evidence.reshape(evidence.shape[0], 1, spectra.shape[1]), dc_value, bf_inds_row, bf_inds_col,
                      bright_field_center_px, bright_field_radius_px, tuple(scan_shape), tuple(detector_shape),
                      semiangle=semiangle, scan_sampling=scan_sampling, detector_sampling_mrad=detector_sampling_mrad,
                      voltage_kV=voltage_kV, bright_field_intensity_threshold=bright_field_intensity_threshold,
                      aberrations=aberrations, scan_detector_rotation_rad=math.radians(scan_detector_rotation_deg))
        return backend

    def _hold(self, G_qk, dc_value, bf_inds_row, bf_inds_col, center, radius, scan_gpts, det_gpts, *, semiangle,
              scan_sampling, detector_sampling_mrad, voltage_kV, bright_field_intensity_threshold, aberrations,
              scan_detector_rotation_rad) -> None:
        """Keep the bright-field spectra and the calibration and geometry the engine reads with them."""
        if isinstance(scan_sampling, (int, float)):
            scan_sampling = (float(scan_sampling), float(scan_sampling))
        if isinstance(detector_sampling_mrad, (int, float)):
            detector_sampling_mrad = (float(detector_sampling_mrad), float(detector_sampling_mrad))
        if aberrations is None:
            aberrations = {"C10": 0.0, "C12": 0.0, "phi12": 0.0}
        self.voltage_kV = voltage_kV
        self.semiangle_mrad = semiangle
        self.scan_sampling = scan_sampling
        self.detector_sampling_mrad = detector_sampling_mrad
        self.bright_field_intensity_threshold = float(bright_field_intensity_threshold)
        self.aberrations = aberrations.copy()
        self._scan_detector_rotation_rad = scan_detector_rotation_rad
        wavelength = wavelength_A_from_kV(voltage_kV)
        # detector sampling, mrad -> the real-space sampling whose reciprocal grid the detector pixels are
        self.sampling = tuple(1.0 / (step * 1e-3 / wavelength * size)
                              for step, size in zip(detector_sampling_mrad, det_gpts))
        self.gpts = det_gpts
        self.wavelength = wavelength
        self.bf_inds_row, self.bf_inds_col = bf_inds_row, bf_inds_col
        self.bright_field_center_px, self.bright_field_radius_px = center, radius
        self.G_qk, self.dc_value = G_qk, dc_value
        self._scan_shape = scan_gpts

        # scan spatial frequencies (1/A), float32 as the kernels read them
        q_row_1d = cp.fft.fftfreq(scan_gpts[0], scan_sampling[0]).astype(cp.float32)
        q_col_1d = cp.fft.fftfreq(scan_gpts[1], scan_sampling[1]).astype(cp.float32)
        self.q_row, self.q_col = cp.meshgrid(q_row_1d, q_col_1d, indexing='ij')
        self._best_loss: float = float('inf')
        self._accelerator: DirectEngine | None = None
        self._elapsed_optimize: float = 0.0
        self._elapsed_refine: float = 0.0
        self._refine_method: str | None = None
        self._refine_nfev: int | None = None
        self._trial_records: list[dict] = []

    def free(self) -> None:
        """
        Release all GPU VRAM held by this direct ptychography engine.

        Frees:
        - G_qk (the FFT of virtual BF stack - the largest allocation, ~7 GB)
        - Engine buffers (correction pipeline, variance computation)
        - Batch caches (optimizer working memory)

        After this call, ``result()`` and ``fit()`` will fail - the
        engine is no longer usable. Previously returned ``DirectPtychoResult``
        objects remain valid (they hold independent copies of the phase).

        Call this when the direct ptychography pipeline is done and you need VRAM for
        the next stage (e.g., iterative ptychography).
        """
        if self._accelerator is not None:
            self._accelerator.release_reconstruction_buffers()
        self.G_qk = self._spectra_host = None
        if self._accelerator is not None:
            # the engine's own G and host copy too: a traceback that still holds the engine (a fit that raised)
            # would otherwise keep G on the device
            self._accelerator.G_qk = self._accelerator._spectra_host = self._accelerator._slab_buffers = None
            # the geometry arrays too, so the pool flush below returns their memory
            self._accelerator._cache.clear()
            # Break the known ownership cycle directly. A global collection
            # would also scan unrelated notebook objects on every region close.
            self._accelerator.thick.engine = None
            self._accelerator = None
        cp.get_default_memory_pool().free_all_blocks()

    def _get_accelerator(self) -> DirectEngine:
        """Get or create CuPy accelerator."""
        if self._accelerator is None:
            self._accelerator = DirectEngine(
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
                angular_sampling=self.detector_sampling_mrad,
                fit_band_inv_A=self.fit_band_inv_A,
                spectra_host=self._spectra_host,
                resident_rows=self._resident_rows,
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

    def browser_state(self) -> ExportState:
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

    def result(self, *, compute_loss: bool = True) -> DirectPtychoResult:
        """The direct ptychography phase at the current aberrations (``fit()``'s final reconstruction), with the phase-variance loss.

        The phase is the native-grid phase of the bright-field mean wave, as ``preview_upsampled`` computes it at
        factor 1, held as a unit-amplitude wave; ``compute_loss=False`` skips the loss.
        """
        self._whole_spectrum("the reconstruction")
        accel = self._get_accelerator()
        try:
            phase, _ = accel.thick.run(
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
        return DirectPtychoResult(
            object_wave=cp.exp(1j * phase),
            backend="cuda",
            aberrations=self.aberrations.copy(),
            scan_detector_rotation_deg=math.degrees(self._scan_detector_rotation_rad),
            loss=loss,
            elapsed_s=elapsed_s if elapsed_s > 0 else None,
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
    ) -> DirectPtychoResult:
        """Standard fit (tilt=False) of C10, C12 and phi12, then the complex object and the phase-variance loss there.

        Maximizes the least-squares agreement of the direct ptychography model with G(q, k) at sample tilt and depth spread 0
        (``optimizer.fit_standard``, the resident search shared with MPS; ``refinement`` "newton", the trust-region
        Newton on the exact derivatives, or "bfgs", the exact-gradient quasi-Newton, for a locked or bounded fit). ``search_ranges`` (engine
        keys) bounds the search, a value holding that coefficient in every trial; ``refine_lock`` holds coefficients
        through the refinement; ``aberrations`` (engine units) is a refinement start.
        """
        if aberrations is not None:
            self.aberrations = dict(aberrations)
        # a failed search leaves no previous fit's records or refinement attached to the new aberrations
        self._trial_records, self._refine_method, self._refine_nfev = [], None, None
        self._elapsed_optimize = self._elapsed_refine = 0.0
        accel = self._get_accelerator()
        started = time.perf_counter()
        try:
            fit = fit_standard(accel, aberrations=aberrations, search_ranges=search_ranges, trials=int(trials),
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
        upsample: int,
        compute_loss: bool,
        sample_tilt_mrad: tuple[float, float] = (0.0, 0.0),
        depth_spread_A: float = 0.0,
        higher_order_magnitudes: np.ndarray | None = None,
        higher_order_angles: np.ndarray | None = None,
    ) -> tuple[cp.ndarray, float | None]:
        """Direct ptychography phase (of the bright-field mean wave) at the output factor, with the diagnostic loss on the native grid.

        The loss is the phase variance over bright-field pixels: the engine's fused two-term pass for a thin sample,
        else ``ThickSample.phase_variance``.
        """
        self._whole_spectrum("the reconstruction")
        accel = self._get_accelerator()
        coefficients = tuple(aberrations[key] for key in ("C10", "C12", "phi12"))
        phase, _ = accel.thick.run(
            *coefficients, sample_tilt_mrad, depth_spread_A, compute_loss=False,
            upsample=upsample,
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

    def phase_variance(self, aberrations: dict[str, float]) -> tuple[cp.ndarray, float]:
        """The phase-variance objective at C10/C12 optics on the native grid: (mean BF phase, loss).

        The fits' own evaluator (``DirectEngine.reconstruct_with_loss``), so the loss matches the one a fit reports.
        """
        self._whole_spectrum("the phase-variance loss")
        return self._get_accelerator().reconstruct_with_loss(aberrations["C10"], aberrations["C12"], aberrations["phi12"])

    def _whole_spectrum(self, stage: str) -> None:
        """Refuse ``stage`` on a band-only G (``from_band_spectra``): it reads every frequency of the half plane."""
        if self.fit_band_inv_A is not None:
            raise NotImplementedError(
                f"This session holds G(q, k) at the fit band's frequencies only, for fitting the whole scan on a small "
                f"device; {stage} needs every frequency. Reconstruct with run(..., compute_loss=False), which reads "
                "the encoded scan in batches of detector columns.")

    def release_cache(self) -> None:
        """Nothing to release between calls: CuPy's pool keeps its blocks so the next fit or preview reuses them."""

    def release_prepared(self) -> None:
        """Nothing to release: the CUDA session keeps G(q, k) resident until ``close``.

        Except a band-only G (``from_band_spectra``): it leaves the device and returns from its host copy when a fit
        next reads it, so a reconstruction can decode the encoded scan in its place on a small device.
        """
        if self.fit_band_inv_A is None:
            return
        # the engine holds G from here on (it reloads it), so it is built before the backend lets go
        self._get_accelerator().release_spectra()
        self.G_qk = None
        cp.get_default_memory_pool().free_all_blocks()

    def release_for_region(self, scan_shape: tuple[int, int]) -> None:
        """Nothing to release before a region fit: the CUDA session keeps G(q, k) resident until ``close``."""

    def inherit_full_field_preparation(self, source) -> bool:
        """CUDA builds G(q, k) from the region's own counts (``shares_full_field_preparation`` is False)."""
        return False

    def close(self) -> None:
        """Release CUDA-owned session state."""

        self.free()

    def _print_summary(self, stage: str, elapsed_s: float) -> None:
        """Print one-line optimization summary."""
        aberrations = self.aberrations   # engine units (Angstrom); the public API and this line report nm
        loss = "not computed" if self._best_loss is None else f"{self._best_loss:.6f}"
        print(
            f"  {stage}: loss={loss}  "
            f"C10={aberrations['C10'] / 10.0:.2f} nm  C12={aberrations['C12'] / 10.0:.2f} nm  "
            f"phi12={math.degrees(aberrations['phi12']):.1f}°  "
            f"{elapsed_s:.1f}s"
        )

    def _out_of_memory(self, stage: str) -> MemoryError:
        """The error that replaces a device out-of-memory failure during ``stage``.

        It names the two numbers that decide whether direct ptychography fits on the device, the bright-field pixel count (a smaller
        ``bright_field_radius_px`` reduces it) and the free device memory (stale arrays in the session hold it).
        """
        num_bright_field_pixels = len(self.bf_inds_row)
        free_gb = cp.cuda.runtime.memGetInfo()[0] / 1e9
        return MemoryError(
            f"Out of GPU VRAM during direct ptychography {stage} "
            f"({num_bright_field_pixels} BF pixels, {free_gb:.1f} GB free).\n"
            f"Try: DirectPtycho(..., bright_field_radius_px=<smaller>) to reduce BF pixel count, "
            f"or restart the kernel to free stale GPU memory."
        )
