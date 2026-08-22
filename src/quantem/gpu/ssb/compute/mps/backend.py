"""Backend-neutral MPS implementation of interactive SSB reconstruction."""
from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from quantem.gpu.detector import mean_dp
from quantem.gpu.optics.physics import electron_wavelength_angstrom
from quantem.gpu.ssb.results import SSBResult

from ..protocol import SSBExportState, SSBPrecision
from .engine import (
    MpsBfColumnFrames,
    _as_chunked_frames,
    _as_sampling,
    _default_object_redraw_chunk_bf,
    _default_object_setup_chunk_bf,
    _effective_phase_loss_chunk_bf,
    _object_fourier_sum_dynamic,
    _prepare_selection,
    _PreparedMpsSSB,
    _reconstruct_prepared,
    _require_mlx,
    _resolve_bf_selection,
    _retarget_prepared_rotation,
    _scan_shape,
)
from .optimizer import optimize as optimize_mps


def _clear_mps_io_cache() -> None:
    """Release reusable decoder scratch after an SSB source is closed."""

    from quantem.gpu.io.backends.mps.decoder import clear_mps_cache

    clear_mps_cache()


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


class MpsSSBBackend:
    """MLX/Metal implementation of :class:`~quantem.gpu.SSBProtocol`.

    The engine owns MPS preparation and device state. UI and export consumers
    use only the public reconstruction methods and :meth:`browser_state`.
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
        det_sampling: float | tuple[float, float] | None,
        bf_intensity_threshold: float,
        bf_center: tuple[float, float] | None,
        bf_radius: int | None,
        rotation_angle_deg: float,
        aberrations: dict[str, float] | None = None,
        chunk_bf: int | None = None,
        setup_chunk_bf: int | None = None,
        redraw_chunk_bf: int | None = None,
    ) -> None:
        self._frames = _as_chunked_frames(data)
        self._source_data = data
        self._scan_shape = _scan_shape(self._frames)
        self._detector_shape = tuple(int(x) for x in self._frames.shape[-2:])
        self._voltage_kV = float(voltage_kV)
        self._semiangle_mrad = float(semiangle_mrad)
        self._bf_intensity_threshold = float(bf_intensity_threshold)
        self._bf_center = bf_center
        self._bf_radius = bf_radius
        self._scan_sampling = _as_sampling(scan_sampling)
        requested_chunk_bf = 1 if chunk_bf is None else max(1, int(chunk_bf))
        self._setup_chunk_bf = (
            max(1, int(setup_chunk_bf))
            if setup_chunk_bf is not None
            else max(requested_chunk_bf, int(_default_object_setup_chunk_bf()))
        )
        self._chunk_bf = (
            max(1, int(redraw_chunk_bf))
            if redraw_chunk_bf is not None
            else max(requested_chunk_bf, int(_default_object_redraw_chunk_bf()))
        )
        self._phase_chunk_bf = _effective_phase_loss_chunk_bf(
            requested_chunk_bf, self._scan_shape
        )
        stored_dc = (
            self._frames.dc_value
            if isinstance(self._frames, MpsBfColumnFrames)
            else None
        )
        mean_diffraction = (
            None if stored_dc is not None else np.asarray(mean_dp(self._frames))
        )
        self._selection = _resolve_bf_selection(
            self._frames,
            bf_intensity_threshold,
            bf_radius,
            center_override=bf_center,
            mean_diffraction=mean_diffraction,
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
        if det_sampling is None:
            detector_pixel_mrad = (
                2.0 * self._semiangle_mrad / self._selection.detected_radius_px
            )
            self._det_sampling = (detector_pixel_mrad, detector_pixel_mrad)
        else:
            self._det_sampling = _as_sampling(det_sampling)
        self._rotation_angle_deg = float(rotation_angle_deg)
        self._aberrations = dict(aberrations or {})
        self._prepared = None
        self._mean_phase_buffer = None
        self._sumsq_buffer = None
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
        trials: int,
        refinement: str | None,
        search_ranges: dict[str, tuple[float, float] | float] | None,
        refine_lock: list[str] | None,
        seed: int,
        verbose: bool,
    ):
        """Run the shared exact optimization contract on MPS/Metal."""

        result = optimize_mps(
            self._source_data,
            voltage_kV=self._voltage_kV,
            semiangle_mrad=self._semiangle_mrad,
            scan_sampling_A=self._scan_sampling,
            det_sampling=self._det_sampling,
            aberrations=self._aberrations,
            search_ranges=search_ranges,
            n_trials=int(trials),
            refine=refinement,
            refine_lock=refine_lock,
            rotation_angle_deg=self._rotation_angle_deg,
            bf_intensity_threshold=self._bf_intensity_threshold,
            bf_center=self._bf_center,
            bf_radius=self._bf_radius,
            seed=int(seed),
            verbose=verbose,
            _on_complete=self._retain_fit_state,
        )
        # Fit leaves large candidate-batch temporaries in MLX's allocator
        # cache. The retained prepared FFT remains active; release only unused
        # cache before subsequent slider reconstructions.
        _require_mlx().clear_cache()
        self._aberrations = dict(result.aberrations)
        return result

    def _retain_fit_state(
        self,
        prepared: _PreparedMpsSSB,
        phase: np.ndarray,
        loss: float,
        aberrations: dict[str, float],
    ) -> None:
        """Keep the optimizer's exact final state for immediate interaction."""

        self._prepared = prepared
        self._fit_preview_phase = np.asarray(phase, dtype=np.float32)
        self._fit_preview_loss = float(loss)
        self._fit_preview_aberrations = dict(aberrations)

    def reconstruct_result(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool = True,
    ):
        """Reconstruct from the retained source FFT and geometry."""

        started = time.perf_counter()
        if self._prepared is None:
            self.cache_rotation(math.radians(self._rotation_angle_deg))
        object_wave = _object_fourier_sum_dynamic(
            self._prepared,
            C10=float(aberrations["C10"]),
            C12=float(aberrations["C12"]),
            phi12=float(aberrations["phi12"]),
            chunk_bf=self._chunk_bf,
        )
        loss = None
        if compute_loss:
            _unused_object, loss, _unused_phase = _reconstruct_prepared(
                self._prepared,
                C10=float(aberrations["C10"]),
                C12=float(aberrations["C12"]),
                phi12=float(aberrations["phi12"]),
                chunk_bf=self._phase_chunk_bf,
                compute_loss=True,
                compute_object=False,
            )
        self._aberrations = dict(aberrations)
        result = SSBResult(
            object_wave=np.asarray(object_wave).astype(np.complex64, copy=False),
            backend="mps",
            aberrations=dict(aberrations),
            rotation_angle_deg=self._rotation_angle_deg,
            loss=None if loss is None else float(loss),
            elapsed=time.perf_counter() - started,
            num_bf=self._selection.size,
            voltage_kV=self._voltage_kV,
            semiangle_mrad=self._semiangle_mrad,
            scan_sampling_A=self._scan_sampling,
            bf_center=self._selection.center_row_col,
            bf_radius=self._selection.radius_px,
            detected_bf_radius=self._selection.detected_radius_px,
        )
        _require_mlx().clear_cache()
        return result

    def preview(
        self,
        aberrations: dict[str, float],
        *,
        compute_loss: bool,
        higher_order_magnitudes: np.ndarray | None,
        higher_order_angles: np.ndarray | None,
    ) -> tuple[np.ndarray, float | None]:
        """Return one transient float32 phase and optional exact loss."""

        if (
            higher_order_magnitudes is None
            and self._fit_preview_phase is not None
            and aberrations == self._fit_preview_aberrations
        ):
            phase = self._fit_preview_phase.copy()
            loss = self._fit_preview_loss if compute_loss else None
            return phase, loss

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
        """Release MPS preparation, source data, and cached Metal buffers."""

        source = self._source_data
        self._prepared = None
        self._frames = None
        self._mean_phase_buffer = None
        self._sumsq_buffer = None
        self._fit_preview_phase = None
        self._fit_preview_loss = None
        self._fit_preview_aberrations = None
        self._source_data = None
        release = getattr(source, "free", None)
        if callable(release):
            release()
        _clear_mps_io_cache()
        _require_mlx().clear_cache()

    @property
    def scan_shape(self) -> tuple[int, int]:
        """Reconstruction grid shape in public ``(row, col)`` order."""

        return self._scan_shape

    @property
    def detector_shape(self) -> tuple[int, int]:
        """Detector grid shape in public ``(row, col)`` order."""

        return self._detector_shape

    @property
    def num_bf(self) -> int:
        """Number of selected bright-field detector pixels."""

        return self._selection.size

    def cache_rotation(self, rotation_rad: float) -> None:
        """Prepare the exact selected evidence for one scan rotation."""

        rotation_angle_deg = math.degrees(float(rotation_rad))
        if (
            self._prepared is not None
            and abs(rotation_angle_deg - self._rotation_angle_deg) < 1e-9
        ):
            return
        self._rotation_angle_deg = rotation_angle_deg
        self._fit_preview_phase = None
        self._fit_preview_loss = None
        self._fit_preview_aberrations = None
        if self._prepared is None:
            self._prepared = _prepare_selection(
                self._frames,
                scan_shape=self._scan_shape,
                selection=self._selection,
                voltage_kV=self._voltage_kV,
                semiangle_mrad=self._semiangle_mrad,
                scan_sampling=self._scan_sampling,
                det_sampling=self._det_sampling,
                rotation_angle_deg=self._rotation_angle_deg,
                chunk_bf=self._setup_chunk_bf,
                compact_inactive=True,
                dc_value_override=self._dc_value_override,
            )
        else:
            _retarget_prepared_rotation(
                self._prepared,
                selection=self._selection,
                rotation_angle_deg=self._rotation_angle_deg,
            )
            _require_mlx().clear_cache()

    def reconstruct_with_loss(
        self,
        c10: float,
        c12: float,
        phi12: float,
    ):
        """Return the phase and exact full-BF variance loss from Metal."""

        if self._prepared is None:
            self.cache_rotation(math.radians(self._rotation_angle_deg))

        _object_wave, loss, phase = _reconstruct_prepared(
            self._prepared,
            C10=float(c10),
            C12=float(c12),
            phi12=float(phi12),
            chunk_bf=self._chunk_bf,
            compute_loss=True,
            compute_object=False,
        )
        phase_np = np.asarray(phase, dtype=np.float32)
        self._mean_phase_buffer = phase_np
        self._sumsq_buffer = phase_np * phase_np * float(self.num_bf)
        return phase, float(loss)

    def reconstruct(self, c10: float, c12: float, phi12: float):
        """Return the exact full-BF phase reconstructed on Metal."""

        if self._prepared is None:
            self.cache_rotation(math.radians(self._rotation_angle_deg))

        _object_wave, _loss, phase = _reconstruct_prepared(
            self._prepared,
            C10=float(c10),
            C12=float(c12),
            phi12=float(phi12),
            chunk_bf=self._chunk_bf,
            compute_loss=False,
            compute_object=False,
        )
        return phase

    def reconstruct_object(self, c10: float, c12: float, phi12: float):
        """Return the exact complex BF-averaged object wave from Metal."""

        if self._prepared is None:
            self.cache_rotation(math.radians(self._rotation_angle_deg))

        return _object_fourier_sum_dynamic(
            self._prepared,
            C10=float(c10),
            C12=float(c12),
            phi12=float(phi12),
            chunk_bf=self._chunk_bf,
        )

    @staticmethod
    def phase_to_numpy(phase) -> np.ndarray:
        """Expose one MPS reconstruction as a host float32 image."""

        return np.asarray(phase, dtype=np.float32)

    @staticmethod
    def _three_param_from_full(mags_m, angles_rad) -> tuple[float, float, float]:
        mags = np.asarray(mags_m, dtype=np.float32)
        angles = np.asarray(angles_rad, dtype=np.float32)
        if np.any(mags[2:] != 0):
            raise NotImplementedError(
                "MPS SSB currently supports C10/C12/phi12 only. "
                "Higher-order controls require a backend with the "
                "'higher_order' capability."
            )
        return float(mags[0]), float(mags[1]), float(angles[1])

    def reconstruct_full_with_loss(self, mags_m, angles_rad):
        """Reconstruct with the supported subset of full aberration inputs."""

        c10, c12, phi12 = self._three_param_from_full(mags_m, angles_rad)
        return self.reconstruct_with_loss(c10, c12, phi12)

    def reconstruct_full(self, mags_m, angles_rad):
        """Reconstruct with the supported subset of full aberration inputs."""

        c10, c12, phi12 = self._three_param_from_full(mags_m, angles_rad)
        return self.reconstruct(c10, c12, phi12)

    def preview_context(self, num_bf: int):
        """Return no reduced-evidence preview for the exact MPS path."""

        del num_bf

    def export_brightfield(
        self,
        data,
        path_stem,
    ) -> tuple[Path, float] | None:
        """Write the exact selected integer columns and release the raw scan."""

        del data
        if isinstance(self._frames, MpsBfColumnFrames):
            return None
        dtype = np.dtype(self._frames._np_dtype)
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
        for start in range(0, self._selection.size, 256):
            stop = min(start + 256, self._selection.size)
            block = self._frames.columns_float32(
                self._selection.rows[start:stop],
                self._selection.cols[start:stop],
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
            verbose=False,
        )
        raw_source = self._source_data
        self._frames = replacement
        self._source_data = replacement
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
        self._fit_preview_phase = None
        self._fit_preview_loss = None
        self._fit_preview_aberrations = None
        free = getattr(raw_source, "free", None)
        if callable(free):
            free()
        return path.resolve(), time.perf_counter() - started

    def browser_state(self) -> SSBExportState:
        """Return backend-neutral host metadata for a WebGPU consumer."""

        # Browser WebGPU constructs its own exact Fourier evidence from the
        # HDF5/BF-column source. Exporting calibration must therefore remain a
        # metadata-only operation: building the server MPS FFT stack here
        # duplicates tens of GiB and strands driver allocations as users move
        # between files.
        wavelength = float(electron_wavelength_angstrom(self._voltage_kV * 1e3))
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
            - self._selection.center_row_col[0]
        ) * reciprocal_y
        ky_bf = (
            self._selection.cols.astype(np.float32)
            - self._selection.center_row_col[1]
        ) * reciprocal_x
        if self._rotation_angle_deg:
            angle = math.radians(-self._rotation_angle_deg)
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


__all__ = ["MpsSSBBackend"]
