"""The MLX/Metal implementation of the SSB backend protocol behind :class:`quantem.gpu.SSB` on Apple GPUs."""

import dataclasses
import math
import time
from pathlib import Path
from typing import Self

import numpy as np

from quantem.gpu.detector import mean as detector_mean
from quantem.gpu.optics.physics import wavelength_A_from_kV
from quantem.gpu.direct_ptycho.contract import SSBExportState, SSBPrecision
from quantem.gpu.direct_ptycho.mps.frames import (
    MpsBfColumnFrames,
    MpsTensorFrames,
    as_chunked_frames,
    frames_scan_shape,
)
from quantem.gpu.direct_ptycho.mps.hardware import (
    default_object_redraw_chunk_bf,
    default_object_setup_chunk_bf,
    effective_phase_loss_chunk_bf,
    require_mlx,
)
from quantem.gpu.direct_ptycho.mps.optimizer import fit_sample
from quantem.gpu.direct_ptycho.mps.optimizer import optimize as optimize_mps
from quantem.gpu.direct_ptycho.mps.padding import pad_scan
from quantem.gpu.direct_ptycho.mps.prepared import (
    PreparedMpsSSB,
    as_sampling,
    prepare_selection,
    resolve_bf_selection,
    retarget_prepared_rotation,
)
from quantem.gpu.direct_ptycho.mps.reconstruct import reconstruct_prepared
from quantem.gpu.direct_ptycho.mps.thick_sample import reconstruct_thick
from quantem.gpu.direct_ptycho.mps.upsampling import _reconstruct_upsampled, phase_variance_higher_order


class MpsSSBBackend:
    """MLX/Metal implementation of ``quantem.gpu.direct_ptycho.contract.SSBProtocol``.

    The prepared state (``prepared.py``) owns MPS preparation and device state. UI and export consumers
    use only the public reconstruction methods and :meth:`browser_state`.
    Dense rectangles are centered on the next supported square FFT grid using
    their mean detector pattern, matching CUDA's padding convention. Padding
    is announced; reconstructed phases include the padded extent.
    """

    backend = "mps"
    precision = SSBPrecision()

    def __init__(
        self,
        data: object,
        *,
        voltage_kV: float,
        semiangle_mrad: float,
        scan_sampling: float | tuple[float, float],
        detector_sampling_mrad: float | tuple[float, float] | None,
        bright_field_intensity_threshold: float,
        bright_field_center_px: tuple[float, float] | None,
        bright_field_radius_px: int | None,
        scan_detector_rotation_deg: float,
    ) -> None:
        self._frames = as_chunked_frames(pad_scan(data))
        self._scan_shape = frames_scan_shape(self._frames)
        self._detector_shape = tuple(int(size) for size in self._frames.shape[-2:])
        self._voltage_kV = float(voltage_kV)
        self._semiangle_mrad = float(semiangle_mrad)
        self._scan_sampling = as_sampling(scan_sampling)
        self._setup_chunk_bf = default_object_setup_chunk_bf()
        self._chunk_bf = default_object_redraw_chunk_bf()
        # phase/loss chunk: the optimizer's rule (``optimize(chunk_bf=16)`` -> the per-scan default), so previews and
        # ``reconstruct_result`` sum BF pixels in the fit's chunks. One BF pixel per dispatch made a 128x128 x 8889-BF
        # preview take 3.3 s instead of ~9 ms.
        self._phase_chunk_bf = effective_phase_loss_chunk_bf(16, self._scan_shape)
        stored_dc = (
            self._frames.dc_value
            if isinstance(self._frames, MpsBfColumnFrames)
            else None
        )
        mean_diffraction = (
            None if stored_dc is not None else np.asarray(detector_mean(
                self._frames.tensor if isinstance(self._frames, MpsTensorFrames) else self._frames
            ))
        )
        calibration = None if detector_sampling_mrad is None else as_sampling(detector_sampling_mrad)
        if calibration is None and isinstance(self._frames, MpsBfColumnFrames):
            calibration = self._frames.detector_sampling_mrad
        self._selection = resolve_bf_selection(
            self._frames,
            bright_field_intensity_threshold,
            bright_field_radius_px,
            center_override=bright_field_center_px,
            mean_diffraction=mean_diffraction,
            bright_field_edge_radius_px=None if calibration is None else self._semiangle_mrad / calibration[0],
        )
        self._detector_sum = (
            None
            if mean_diffraction is None
            else mean_diffraction * np.float64(np.prod(self._scan_shape))
        )
        self._dc_value_override = stored_dc
        if self._dc_value_override is None and self._detector_sum is not None:
            selected_dc = self._detector_sum[
                self._selection.rows,
                self._selection.cols,
            ]
            self._dc_value_override = complex(
                np.complex64(np.asarray(selected_dc, dtype=np.float64).mean())
            )
        if calibration is None:
            # the bright-field disk edge sits at the semiangle
            detector_pixel_mrad = self._semiangle_mrad / self._selection.bright_field_edge_radius_px
            calibration = (detector_pixel_mrad, detector_pixel_mrad)
        self._det_sampling = calibration
        self._scan_detector_rotation_deg = float(scan_detector_rotation_deg)
        self._prepared = None
        # the rotation G(q, k) was first built at, and whether its geometry was retargeted since: a rebuild after
        # release_prepared() repeats both, so its stored pixels and geometry are the same bits
        self._built_rotation_deg = None
        self._retargeted = False
        self._fit_preview_phase = None
        self._fit_preview_loss = None
        self._fit_preview_aberrations = None
        self._bf_source_path = None
        self._bf_source_dtype = None
        self._bf_source_max_value = None
        if isinstance(self._frames, MpsBfColumnFrames):
            self._bf_source_path = self._frames.source_path
            self._bf_source_dtype = np.dtype(self._frames.dtype)
            self._bf_source_max_value = self._frames.max_value

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
    ):
        """Run the standard fit (tilt=False) on MPS/Metal, on this session's prepared evidence: the resident
        model-agreement search (``optimizer.optimize``)."""

        if self._prepared is None:
            self.cache_rotation(math.radians(self._scan_detector_rotation_deg))
        result, phase = optimize_mps(
            self._prepared,
            self._selection,
            voltage_kV=self._voltage_kV,
            semiangle_mrad=self._semiangle_mrad,
            scan_sampling_A=self._scan_sampling,
            aberrations=aberrations,
            search_ranges=search_ranges,
            num_trials=int(trials),
            refine=refinement,
            refine_lock=refine_lock,
            scan_detector_rotation_deg=self._scan_detector_rotation_deg,
            seed=int(seed),
            verbose=verbose,
            compute_loss=compute_loss,
        )
        if not compute_loss:
            result.loss = None
        # The fit's final phase-variance pass answers the first loss at the fitted aberrations without a second one.
        self._fit_preview_phase = phase
        self._fit_preview_loss = None if result.loss is None else float(result.loss)
        self._fit_preview_aberrations = dict(result.aberrations)
        # Fit leaves large candidate-batch temporaries in MLX's allocator
        # cache. The retained prepared FFT remains active; release only unused
        # cache before subsequent slider reconstructions.
        require_mlx().clear_cache()
        return result

    def phase_variance(self, aberrations: dict[str, float]) -> tuple[np.ndarray, float]:
        """The phase-variance objective at C10/C12 optics on the native grid: (mean BF phase, loss).

        A fit's own final pass answers the fitted aberrations without a second evaluation.
        """
        if self._fit_preview_phase is not None and aberrations == self._fit_preview_aberrations:
            return self._fit_preview_phase.copy(), self._fit_preview_loss
        phase, loss = self.reconstruct_with_loss(aberrations["C10"], aberrations["C12"], aberrations["phi12"])
        return np.asarray(phase, dtype=np.float32), float(loss)

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
    ) -> tuple[np.ndarray, float | None]:
        """SSB phase (of the bright-field mean wave) at the output factor, unfolding measured aliases on Metal.

        The loss is the native-grid phase variance over bright-field pixels (``phase_variance``,
        ``sample_phase_variance`` or ``phase_variance_higher_order``).
        """
        if self._prepared is None:
            self.cache_rotation(math.radians(self._scan_detector_rotation_deg))
        higher = dict(higher_order_magnitudes=higher_order_magnitudes,
                      higher_order_angles=higher_order_angles)
        phase = _reconstruct_upsampled(
            self._prepared, **aberrations, sample_tilt_mrad=sample_tilt_mrad,
            thickness=depth_spread_A, upsampling_factor=upsampling_factor, **higher,
        )
        if not compute_loss:
            return phase, None
        if higher_order_magnitudes is not None:
            _, loss = phase_variance_higher_order(
                self._prepared, **aberrations, sample_tilt_mrad=sample_tilt_mrad, thickness=depth_spread_A, **higher,
            )
        elif depth_spread_A > 0.0:
            _, loss = self.sample_phase_variance(
                aberrations, {"tilt_row_mrad": sample_tilt_mrad[0], "tilt_col_mrad": sample_tilt_mrad[1],
                              "depth_spread_A": depth_spread_A},
            )
        else:
            _, loss = self.phase_variance(aberrations)
        return phase, loss

    def sample_phase_variance(self, aberrations: dict[str, float], sample: dict[str, float]) -> tuple[np.ndarray, float]:
        """The phase-variance objective for a thick, tilted sample: (mean BF phase, loss), ``thick_sample.reconstruct_thick``.

        ``sample`` = {"tilt_row_mrad", "tilt_col_mrad", "depth_spread_A"} (depth spread in the C10 unit, Angstrom; 0 = standard SSB).
        """
        if self._prepared is None:
            self.cache_rotation(math.radians(self._scan_detector_rotation_deg))
        phase, loss = reconstruct_thick(
            self._prepared,
            C10=float(aberrations["C10"]),
            C12=float(aberrations["C12"]),
            phi12=float(aberrations["phi12"]),
            sample_tilt_mrad=(float(sample.get("tilt_row_mrad", 0.0)), float(sample.get("tilt_col_mrad", 0.0))),
            thickness=float(sample.get("depth_spread_A", 0.0)),
            compute_loss=True,
            chunk_bf=self._phase_chunk_bf,
        )
        return phase, loss

    def fit_sample(self, **options) -> dict[str, object]:
        """Fit aberrations, sample tilt and thickness together (``optimizer.fit_sample``)."""
        started = time.perf_counter()
        if self._prepared is None:
            self.cache_rotation(math.radians(self._scan_detector_rotation_deg))
        preparation_s = time.perf_counter() - started
        result = fit_sample(self._prepared, **options)
        result.setdefault("timings", {})["fft_preparation_s"] = preparation_s
        # Reuse allocator buffers for the following phase diagnostic and for
        # repeated fits (both 180-degree branches); SSB.find_aberrations calls
        # release_cache() when the fit is done, and close() clears it too.
        return result

    def band_power(self, band_inv_A: tuple[float, float]) -> float:
        """G's power over the band, the agreement's upper bound (``thick_sample.band_power``)."""
        from quantem.gpu.direct_ptycho.mps.thick_sample import band_power

        if self._prepared is None:
            self.cache_rotation(math.radians(self._scan_detector_rotation_deg))
        return band_power(self._prepared, band_inv_A)

    def sample_agreement(self, parameters: np.ndarray, band_inv_A: tuple[float, float]) -> np.ndarray:
        """Evaluate the joint objective on the session's existing evidence."""
        from quantem.gpu.direct_ptycho.mps.thick_sample import thick_fit_batch

        if self._prepared is None:
            self.cache_rotation(math.radians(self._scan_detector_rotation_deg))
        # Four candidates run the diagnostic batches faster on MPS.
        # Preserve each candidate's arithmetic and the search's eight-trial
        # scheduling; these are independent measurements after the fit.
        parameters = np.atleast_2d(parameters)
        agreement = np.empty(len(parameters), dtype=np.float64)
        for start in range(0, len(parameters), 4):
            agreement[start:start + 4] = thick_fit_batch(
                self._prepared, parameters[start:start + 4], band_inv_A,
            )
        return agreement

    def release_cache(self) -> None:
        """Return MLX's cached, unused Metal buffers to the system; the prepared G(q, k) stays."""
        require_mlx().clear_cache()

    def release_prepared(self) -> None:
        """Free G(q, k) and MLX's cached buffers; the next preview or fit rebuilds G(q, k) from the source.

        The rebuild reads the same columns in the same chunks at the rotation of the first build and retargets the
        geometry as the session had, and the fit's final phase stays, so every later result has the same bits. On
        an encoded MPS crop it decodes the bright-field columns again (about 1 s for 8,939 pixels of a 512 x 512
        scan).
        """
        self._prepared = None
        require_mlx().clear_cache()

    def release_for_region(self, scan_shape: tuple[int, int]) -> None:
        """Keep the overview FFT only when a region fit has ample memory headroom.

        A local correction reconstructs the full field again after fitting.
        Rebuilding its unchanged Fourier data on every drag is unnecessary on
        large-memory devices. Reserve four regional Fourier stacks for the
        region and its fitting temporaries, plus 2 GiB for source/decoder work,
        within half the recommended working set. The other half remains free
        for Torch, the browser and other applications. Unknown limits retain
        the existing release policy.
        """
        mx = require_mlx()
        mx.clear_cache()
        if self._prepared is None:
            return
        recommended = mx.device_info().get("max_recommended_working_set_size", 0)
        # Rectangular regions use the ordinary square mean-pattern padding.
        side = max(scan_shape)
        if scan_shape[0] != scan_shape[1]:
            from quantem.gpu.direct_ptycho.mps.kernels import MPS_FFT_CONFIGS

            side = next((size for size in sorted(MPS_FFT_CONFIGS)
                         if size >= side), side)
        parent_pixels = math.prod(self._prepared.scan_shape)
        region_bytes = self._prepared.g_qk.nbytes * side * side // parent_pixels
        reserve_bytes = 4 * region_bytes + (2 << 30)
        if mx.get_active_memory() + reserve_bytes > recommended // 2:
            self.release_prepared()

    def inherit_full_field_preparation(self, source: "MpsSSBBackend") -> None:
        """Share unchanged full-field evidence with an independent fit session.

        The caller establishes that both sessions cover the same source
        positions. Geometry must match a fresh preparation too: retargeted
        geometry can carry different rounding, so it keeps its ordinary path.
        The prepared record is copied; fitting, rotating or closing this
        session cannot replace the source session's arrays or caches.
        """
        if source._prepared is None or source._retargeted:
            return
        attributes = (
            "_scan_shape", "_voltage_kV", "_semiangle_mrad", "_scan_sampling",
            "_det_sampling", "_scan_detector_rotation_deg",
        )
        if any(getattr(self, name) != getattr(source, name) for name in attributes):
            return
        if not (np.array_equal(self._selection.rows, source._selection.rows)
                and np.array_equal(self._selection.cols, source._selection.cols)):
            return
        self._prepared = dataclasses.replace(source._prepared)
        self._built_rotation_deg = source._built_rotation_deg
        self._retargeted = source._retargeted

    def close(self) -> None:
        """Release MPS preparation and cached Metal buffers; the source data stays its owner's."""

        self._prepared = None
        self._frames = None
        self._fit_preview_phase = None
        self._fit_preview_loss = None
        self._fit_preview_aberrations = None
        require_mlx().clear_cache()

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Reconstruction grid shape in public ``(row, col)`` order."""

        return self._scan_shape

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector grid shape in public ``(row, col)`` order."""

        return self._detector_shape

    @property
    def num_bright_field_pixels(self) -> int:
        """Number of selected bright-field detector pixels."""

        return self._selection.size

    def cache_rotation(self, scan_detector_rotation_rad: float) -> None:
        """Prepare the exact selected evidence for one scan rotation."""

        scan_detector_rotation_deg = math.degrees(float(scan_detector_rotation_rad))
        unchanged = abs(scan_detector_rotation_deg - self._scan_detector_rotation_deg) < 1e-9
        if self._prepared is not None and unchanged:
            return
        self._scan_detector_rotation_deg = scan_detector_rotation_deg
        if not unchanged:
            # the fit's final phase answers the first preview only at the rotation it was fitted at (a rebuild after
            # release_prepared keeps it)
            self._fit_preview_phase = None
            self._fit_preview_loss = None
            self._fit_preview_aberrations = None
        if self._prepared is None:
            if self._built_rotation_deg is None:
                self._built_rotation_deg = self._scan_detector_rotation_deg
            self._prepared = prepare_selection(
                self._frames,
                scan_shape=self._scan_shape,
                selection=self._selection,
                voltage_kV=self._voltage_kV,
                semiangle_mrad=self._semiangle_mrad,
                scan_sampling=self._scan_sampling,
                detector_sampling_mrad=self._det_sampling,
                scan_detector_rotation_deg=self._built_rotation_deg,
                chunk_bf=self._setup_chunk_bf,
                compact_inactive=True,
                dc_value_override=self._dc_value_override,
            )
            if self._retargeted or self._built_rotation_deg != self._scan_detector_rotation_deg:
                self._retargeted = True
                retarget_prepared_rotation(
                    self._prepared,
                    selection=self._selection,
                    scan_detector_rotation_deg=self._scan_detector_rotation_deg,
                )
        else:
            self._retargeted = True
            retarget_prepared_rotation(
                self._prepared,
                selection=self._selection,
                scan_detector_rotation_deg=self._scan_detector_rotation_deg,
            )
        # Preparation leaves its FFT and gather temporaries in MLX's buffer cache; the prepared G_qk stays resident.
        require_mlx().clear_cache()

    def reconstruct_with_loss(
        self,
        c10: float,
        c12: float,
        phi12: float,
    ):
        """Return the phase and exact full-BF variance loss from Metal."""

        if self._prepared is None:
            self.cache_rotation(math.radians(self._scan_detector_rotation_deg))

        _object_wave, loss, phase = reconstruct_prepared(
            self._prepared,
            C10=float(c10),
            C12=float(c12),
            phi12=float(phi12),
            # phase images use the fit's phase chunk (one dispatch pair per ~4096 BF): the 128-BF object-redraw chunk
            # spent ~30 of 39 ms of a 128x128 preview on per-chunk synchronisation, and the fit's final phase uses it too
            chunk_bf=self._phase_chunk_bf,
            compute_loss=True,
            compute_object=False,
        )
        return phase, float(loss)

    def preview_context(self, num_bright_field_pixels: int):
        """Reusable reduced-BF context for drag previews (standard and thick): every ``num_bright_field_pixels / N``-th BF pixel.

        Same deterministic subset rule as the CUDA backend's ``prepare_bf_subset``. Inside ``with context:``
        ``preview_upsampled`` reconstructs from the subset (mean over the subset's BF pixels).
        """

        return _MpsBfSubset(self, int(num_bright_field_pixels))

    def export_brightfield(
        self,
        data,
        path_stem,
    ) -> tuple[Path, float] | None:
        """Write the exact selected integer columns and read from them instead of the raw scan."""

        del data
        if isinstance(self._frames, MpsBfColumnFrames):
            return None
        dtype = np.dtype(self._frames.dtype)
        if dtype == np.dtype(np.uint8):
            suffix = "u8"
        elif dtype == np.dtype(np.uint16):
            suffix = "u16"
        else:
            raise TypeError(
                "MPS SSB BF-column export requires uint8 or uint16 detector "
                f"storage, got {dtype}."
            )
        path = Path(f"{Path(path_stem)}.{suffix}")
        path.parent.mkdir(parents=True, exist_ok=True)
        path.unlink(missing_ok=True)
        started = time.perf_counter()
        columns = np.memmap(
            path,
            mode="w+",
            dtype=dtype,
            shape=(self._selection.size, int(np.prod(self._scan_shape))),
        )
        maximum = 0
        mx = require_mlx()
        # The gather writes into one reused MLX block: a gather without an output allocates a Metal buffer that
        # nothing releases, which leaked num_bright_field_pixels x frames x 4 bytes of driver memory per export.
        block_mx = None
        for start in range(0, self._selection.size, 256):
            stop = min(start + 256, self._selection.size)
            if block_mx is None or block_mx.shape[0] != stop - start:
                block_mx = mx.empty((stop - start, columns.shape[1]), dtype=mx.float32)
                mx.eval(block_mx)
            block = self._frames.columns_float32_into(
                self._selection.rows[start:stop],
                self._selection.cols[start:stop],
                out=np.asarray(block_mx),
            )
            # uint8/uint16 detector values are exactly representable in
            # float32; assigning back to the integer memmap is bit-exact.
            columns[start:stop] = block
            if block.size:
                maximum = max(maximum, int(block.max()))
        columns.flush()
        del columns
        if dtype == np.dtype(np.uint16) and maximum <= 255:
            # Counts fit uint8, so the downcast is bit-exact and halves the
            # bytes every browser open reads (same rule as the ShowPtycho
            # folder exporter).
            u8_path = Path(f"{Path(path_stem)}.u8")
            u8_path.unlink(missing_ok=True)
            wide = np.memmap(
                path,
                mode="r",
                dtype=dtype,
                shape=(self._selection.size, int(np.prod(self._scan_shape))),
            )
            narrow = np.memmap(u8_path, mode="w+", dtype=np.uint8, shape=wide.shape)
            for start in range(0, wide.shape[0], 256):
                stop = min(start + 256, wide.shape[0])
                narrow[start:stop] = wide[start:stop]
            narrow.flush()
            del wide, narrow
            path.unlink(missing_ok=True)
            path = u8_path
            dtype = np.dtype(np.uint8)

        # detector_sum here can be the float64 mean_dp * N reconstruction,
        # which is not exact integer counts; the BF-column container insists
        # on exactness. The already-derived dc_value carries the same
        # information, so only exact integer sums are forwarded.
        exact_detector_sum = self._detector_sum
        if exact_detector_sum is not None and not np.issubdtype(
            np.asarray(exact_detector_sum).dtype, np.integer
        ):
            exact_detector_sum = None
        replacement = MpsBfColumnFrames(
            path,
            selection=self._selection,
            scan_shape=self._scan_shape,
            dtype=dtype,
            max_value=maximum,
            detector_sum=exact_detector_sum,
            dc_value=self._dc_value_override,
            detector_sampling_mrad=self._det_sampling,
            verbose=False,
        )
        self._frames = replacement
        if replacement.dc_value is not None:
            self._dc_value_override = replacement.dc_value
        else:
            self._dc_value_override = complex(
                np.complex64(
                    np.asarray(replacement.detector_sum)[
                        self._selection.rows,
                        self._selection.cols,
                    ].astype(np.float64).mean()
                )
            )
        self._bf_source_path = path.resolve()
        self._bf_source_dtype = dtype
        self._bf_source_max_value = maximum
        self._prepared = None
        self._built_rotation_deg = None
        self._retargeted = False
        self._fit_preview_phase = None
        self._fit_preview_loss = None
        self._fit_preview_aberrations = None
        return path.resolve(), time.perf_counter() - started

    def browser_state(self) -> SSBExportState:
        """Return backend-neutral host metadata for a WebGPU consumer."""

        # Browser WebGPU constructs its own exact Fourier evidence from the
        # HDF5/BF-column source. Exporting calibration must therefore remain a
        # metadata-only operation: building the server MPS FFT stack here
        # duplicates tens of GiB and strands driver allocations as users move
        # between files.
        wavelength = float(wavelength_A_from_kV(self._voltage_kV))
        semiangle_rad = self._semiangle_mrad * 1e-3
        ang_y_rad = self._det_sampling[0] * 1e-3
        ang_x_rad = self._det_sampling[1] * 1e-3
        qx_1d = np.fft.fftfreq(
            self._scan_shape[0], self._scan_sampling[0]
        ).astype(np.float32)
        qy_1d = np.fft.fftfreq(
            self._scan_shape[1], self._scan_sampling[1]
        ).astype(np.float32)
        reciprocal_y = ang_y_rad / wavelength
        reciprocal_x = ang_x_rad / wavelength
        kx_bf = (
            self._selection.rows.astype(np.float32)
            - self._selection.bright_field_center_px[0]
        ) * reciprocal_y
        ky_bf = (
            self._selection.cols.astype(np.float32)
            - self._selection.bright_field_center_px[1]
        ) * reciprocal_x
        if self._scan_detector_rotation_deg:
            angle = math.radians(-self._scan_detector_rotation_deg)
            cos_a = math.cos(angle)
            sin_a = math.sin(angle)
            kx_bf, ky_bf = (
                kx_bf * cos_a + ky_bf * sin_a,
                -kx_bf * sin_a + ky_bf * cos_a,
            )
        kx_bf = np.asarray(kx_bf, dtype=np.float32)
        ky_bf = np.asarray(ky_bf, dtype=np.float32)
        alpha_k2, cos2phi_k, sin2phi_k, aperture_k = _bf_geometry_1d_numpy(
            kx_bf,
            ky_bf,
            wavelength=wavelength,
            semiangle_rad=semiangle_rad,
            ang_y_rad=ang_y_rad,
            ang_x_rad=ang_x_rad,
        )
        sampling_A = (
            1.0 / (reciprocal_y * self._detector_shape[0]),
            1.0 / (reciprocal_x * self._detector_shape[1]),
        )
        dc_value = self._dc_value_override
        if dc_value is None:
            raise RuntimeError(
                "MPS WebGPU export requires detector-sum or BF-column DC "
                "metadata from the source loader."
            )
        return SSBExportState(
            backend="mps",
            scan_shape=self.scan_shape,
            brightfield=self._selection,
            kx_bf=kx_bf,
            ky_bf=ky_bf,
            qx_1d=qx_1d,
            qy_1d=qy_1d,
            aperture_k=aperture_k,
            alpha_k2=alpha_k2,
            cos2phi_k=cos2phi_k,
            sin2phi_k=sin2phi_k,
            wavelength_A=wavelength,
            semiangle_rad=semiangle_rad,
            angular_sampling_rad=(
                ang_y_rad,
                ang_x_rad,
            ),
            sampling_A=sampling_A,
            dc_value=complex(dc_value),
            bf_source_path=self._bf_source_path,
            bf_source_dtype=self._bf_source_dtype,
            bf_source_max_value=self._bf_source_max_value,
        )


class _MpsBfSubset:
    """Context manager that swaps an MPS backend onto a reduced-BF ``PreparedMpsSSB`` for drag previews.

    Why: the preview cost is linear in the BF count, so a 25 % subset is ~4x faster while dragging. The subset is built
    on first use and rebuilt only when the backend's prepared evidence or rotation geometry changes, so entering the
    context per preview call is free. ``backend`` is the :class:`MpsSSBBackend` whose prepared evidence is swapped.
    """

    def __init__(self, backend, num_bright_field_pixels: int) -> None:
        self._backend = backend
        self._num_bf = int(num_bright_field_pixels)
        self._source = None
        self._subset = None
        self._saved = None

    @property
    def num_bright_field_pixels(self) -> int:
        """Number of BF pixels in the subset (logical, including compacted ones)."""
        return int(self._current().num_bright_field_pixels)

    def _current(self) -> PreparedMpsSSB:
        """The subset for the backend's current prepared evidence.

        A new preparation or a rotation retarget replaces the prepared arrays, so
        comparing them by identity tells whether the cached subset still applies.
        """
        backend = self._backend
        if backend._prepared is None:
            backend.cache_rotation(math.radians(backend._scan_detector_rotation_deg))
        prepared = backend._prepared
        source = (prepared, prepared.g_qk, prepared.kx, prepared.bf_storage_indices_np)
        if self._num_bf >= int(prepared.num_bright_field_pixels):
            # all BF pixels: use the session itself instead of a full copy of G
            return prepared
        if self._source is None or any(current is not cached for current, cached in zip(source, self._source)):
            self._subset = _subset_prepared(prepared, self._num_bf)
            self._source = source
        return self._subset

    def __enter__(self) -> Self:
        if self._saved is not None:
            raise RuntimeError("The prepared SSB BF subset is already active.")
        subset = self._current()
        self._saved = self._backend._prepared
        self._backend._prepared = subset
        return self

    def __exit__(self, *exc) -> None:
        self._backend._prepared = self._saved
        self._saved = None

    def close(self) -> None:
        """Restore the full bright-field evidence if entered and drop the subset this context built.

        Viewers release a drag preview with ``close()`` on CUDA and MPS alike.
        """
        if self._saved is not None:
            self.__exit__()
        self._subset = None
        self._source = None


def _subset_prepared(prepared: PreparedMpsSSB, num_bright_field_pixels: int) -> PreparedMpsSSB:
    """``prepared`` restricted to every ``num_bright_field_pixels / count``-th logical BF pixel (the CUDA ``prepare_bf_subset`` rule).

    Logical BF pixels that were compacted away (aperture 0, not stored) stay unstored and still count in ``num_bright_field_pixels``,
    so the subset mean has the same definition as the full one.
    """
    mx = prepared.mx
    full = int(prepared.num_bright_field_pixels)
    count = max(1, min(int(num_bright_field_pixels), full))
    step = max(1, full // count)
    logical = np.arange(0, full, step, dtype=np.int64)[:count]
    stored = prepared.bf_storage_indices_np
    if stored is None:
        slots = logical
        storage_indices = None
    else:
        stored = np.asarray(stored, dtype=np.int64)
        keep = np.isin(stored, logical)
        slots = np.nonzero(keep)[0]
        storage_indices = np.searchsorted(logical, stored[keep]).astype(stored.dtype)
        if storage_indices.size == logical.size:
            storage_indices = None
    slots_mx = mx.array(slots.astype(np.int32))
    fields = {
        name: getattr(prepared, name)[slots_mx]
        for name in _PER_BF_FIELDS
        if getattr(prepared, name) is not None
    }
    subset = dataclasses.replace(
        prepared,
        g_qk=prepared.g_qk[slots_mx],
        kx=prepared.kx[slots_mx],
        ky=prepared.ky[slots_mx],
        kx_np=np.asarray(prepared.kx_np)[slots],
        ky_np=np.asarray(prepared.ky_np)[slots],
        num_bright_field_pixels=int(logical.size),
        bf_storage_indices_np=storage_indices,
        # Per-BF preview caches of the full evidence (probe, group bounds, phasor geometry) do not index the subset.
        upsampled_geometry=None,
        upsampled_probe=None,
        separable_geometry=None,
        row_steps=None,
        fit_probe_geometry=None,
        band_power=None,
        **fields,
    )
    mx.eval(subset.g_qk, subset.kx, subset.ky, *fields.values())
    return subset


_PER_BF_FIELDS = (
    "alpha_k2", "cos2_k", "sin2_k", "aperture_k",
    "alpha_m2", "cos2_m", "sin2_m", "ap_m",
    "alpha_p2", "cos2_p", "sin2_p", "ap_p",
    "alpha_k2_1d", "cos2_k_1d", "sin2_k_1d", "aperture_k_1d",
)


def _bf_geometry_1d_numpy(
    kx: np.ndarray,
    ky: np.ndarray,
    *,
    wavelength: float,
    semiangle_rad: float,
    ang_y_rad: float,
    ang_x_rad: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute BF-pixel geometry needed by backend-neutral exporters."""

    dx = np.asarray(kx, dtype=np.float32)
    dy = np.asarray(ky, dtype=np.float32)
    dx2 = dx * dx
    dy2 = dy * dy
    r2 = dx2 + dy2
    r = np.sqrt(r2).astype(np.float32, copy=False)
    alpha = r * np.float32(wavelength)
    alpha2 = alpha * alpha
    inv_r2 = np.zeros_like(r2, dtype=np.float32)
    np.divide(1.0, r2, out=inv_r2, where=r2 > np.float32(1e-30))
    cos2phi = (dx2 - dy2) * inv_r2
    sin2phi = np.float32(2.0) * dx * dy * inv_r2
    denom_num2 = (dx * np.float32(ang_y_rad)) ** 2 + (
        dy * np.float32(ang_x_rad)
    ) ** 2
    inv_r = np.zeros_like(r, dtype=np.float32)
    np.divide(1.0, r, out=inv_r, where=r > np.float32(1e-15))
    denom = np.sqrt(denom_num2).astype(np.float32, copy=False) * inv_r
    edge = np.ones_like(r, dtype=np.float32)
    valid = denom > np.float32(1e-15)
    edge[valid] = (
        (np.float32(semiangle_rad) - alpha[valid]) / denom[valid]
        + np.float32(0.5)
    )
    aperture = np.clip(edge, 0.0, 1.0).astype(np.float32, copy=False)
    return (
        alpha2.astype(np.float32, copy=False),
        cos2phi.astype(np.float32, copy=False),
        sin2phi.astype(np.float32, copy=False),
        aperture,
    )


__all__ = ["MpsSSBBackend"]
