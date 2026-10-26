"""Full-scan SSB with a bounded batch of exact detector columns.

Only detector columns are batched. Every forward transform spans every scan
position; no scan tiles are reconstructed or stitched. The correction kernel
is the same one used by the resident-spectrum Direct Ptycho implementation.
"""

import math
import time

import cupy as cp
import cupyx.scipy.fft
import numpy as np
from cupy.cuda import cufft

from quantem.gpu import detector
from quantem.gpu.optics.physics import wavelength_A_from_kV
from quantem.gpu.resident.cuda.counts import StreamedCounts
from quantem.gpu.direct_ptycho.cuda.brightfield import select_bright_field
from quantem.gpu.direct_ptycho.cuda.kernels.engine import thick_wave_sum_add_kernel, thick_wave_sum_full_add_kernel


def bright_field_pixels(mean_pattern, valid, center, radius, threshold, flipped):
    """The streamed session's bright-field selection from the full-detector mean pattern alone.

    Returns the valid-masked mean pattern, the disk centre and radius, the selected detector rows and columns in
    the compute frame (transposed when ``flipped``) and their row-major indices on the stored detector. A loader
    that keeps only some pixels calls this before the session exists, so it keeps exactly what the session reads.
    """
    valid = np.asarray(valid, dtype=bool)
    mean_pattern = np.where(valid, np.asarray(mean_pattern, dtype=np.float64), 0)
    if radius is None:
        selected = mean_pattern > mean_pattern.mean() + mean_pattern.std()
        rows, cols = np.nonzero(selected)
        if not rows.size:
            raise ValueError("No bright-field disk found. Supply calibrated detector geometry.")
        if center is None:
            center = (float(rows.mean()), float(cols.mean()))
        radius = math.sqrt(rows.size / math.pi)
    elif center is None:
        rows, cols = np.nonzero(mean_pattern > mean_pattern.max() * threshold)
        weights = mean_pattern[rows, cols]
        center = (float(np.average(rows, weights=weights)), float(np.average(cols, weights=weights)))
    mean = mean_pattern.T if flipped else mean_pattern
    compute_center = tuple(center)[::-1] if flipped else tuple(center)
    rows, cols, _, _ = select_bright_field(cp.asarray(mean, dtype=cp.float32)[None, None], threshold, float(radius),
                                           compute_center)
    selected_valid = cp.asarray(valid.T if flipped else valid)[rows, cols]
    rows, cols = rows[selected_valid], cols[selected_valid]
    if not rows.size:
        raise ValueError("The bright-field disk contains no valid detector pixels.")
    original_rows, original_cols = (cols, rows) if flipped else (rows, cols)
    pixels = (original_rows * mean_pattern.shape[1] + original_cols).astype(cp.uint32)
    return mean_pattern, center, radius, rows, cols, pixels


class StreamedDirectPtycho:
    """Borrow a complete exact-count resident and retain its immutable chunks."""

    def __init__(self, data, *, mean_pattern, center, radius, threshold,
                 semiangle_mrad, detector_sampling_mrad, scan_detector_rotation_flipped):
        source = data.data
        if type(source) is not StreamedCounts:
            raise NotImplementedError(
                "Large CUDA SSB currently needs lossless integer ANS storage. "
                "Load a uint8/uint16 acquisition with io.load before creating SSB."
            )
        self.scan_shape = tuple(data.shape[:2])
        self.detector_shape = tuple(data.shape[2:])
        if max(self.scan_shape) > 4096 or any(side % 2 for side in self.scan_shape):
            raise ValueError(
                f"Streamed SSB supports even scan dimensions up to 4096; got {self.scan_shape}."
            )
        self.chunks = tuple(source.chunks)
        position = 0
        for chunk in self.chunks:
            if chunk.first != position or chunk.scans <= 0:
                raise ValueError("SSB needs complete consecutive scan coverage. Finish loading the acquisition first.")
            position += chunk.scans
        if source.is_released or source.released_scans or position != math.prod(self.scan_shape) or source.ready_scans != position:
            raise ValueError("SSB needs every scan position resident. Finish loading the acquisition first.")
        self.device = source.device
        self.interval = source.interval
        self.decoding = source.decoding
        self.decode = source.kernels["columns_u16"]
        self.to_fft = source.kernels["columns_fft"]
        self.descriptors = cp.asarray([
            [*(array.data.ptr for array in chunk.arrays[:3]), chunk.first, chunk.scans]
            for chunk in self.chunks
        ], dtype=cp.uint64)
        self.max_chunk_scans = max(chunk.scans for chunk in self.chunks)
        self.mean_pattern = np.asarray(
            detector.mean(data) if mean_pattern is None else mean_pattern, dtype=np.float64,
        )
        if self.mean_pattern.shape != self.detector_shape or not np.isfinite(self.mean_pattern).all():
            raise ValueError("Supply the finite full-detector mean pattern for this acquisition.")
        valid = np.asarray(source.valid_pixels, dtype=bool)
        self.mean_pattern, center, radius, self.rows, self.cols, self.pixels = bright_field_pixels(
            self.mean_pattern, valid, center, radius, threshold, scan_detector_rotation_flipped)
        self.center = tuple(float(value) for value in center)
        self.radius = float(radius)
        kept = getattr(source, "kept_pixels", None)
        if kept is not None and not bool(np.asarray(kept).ravel()[cp.asnumpy(self.pixels)].all()):
            raise ValueError(
                "This acquisition was loaded with only some detector pixels, and the bright-field selection needs "
                "others. Load every pixel (DirectPtycho.open(..., bright_field_only=False))."
            )
        if detector_sampling_mrad is not None and np.ndim(detector_sampling_mrad) == 0:
            # one number for both axes, as every DirectPtycho entry point accepts
            detector_sampling_mrad = (float(detector_sampling_mrad),) * 2
        self.detector_sampling_mrad = detector_sampling_mrad
        if self.detector_sampling_mrad is None:
            # the module function: the keyword of the same name shadows the import inside this method
            self.detector_sampling_mrad = detector.detector_sampling_mrad(self.mean_pattern, semiangle_mrad)
        self.transposed = bool(scan_detector_rotation_flipped)
        self.num_bright_field_pixels = int(self.pixels.size)
        self.timings = {}

    def _geometry(self, voltage_kV, rotation_deg):
        """Wavelength, detector angular sampling and the BF pixels' rotated k (1/A), in the compute frame."""
        wavelength = wavelength_A_from_kV(voltage_kV)
        angular = self.detector_sampling_mrad[::-1] if self.transposed else self.detector_sampling_mrad
        center = self.center[::-1] if self.transposed else self.center
        # Match the resident-spectrum backend's detector reciprocal coordinates.
        compute_shape = self.detector_shape[::-1] if self.transposed else self.detector_shape
        sampling = tuple(wavelength / (step * 1e-3 * size) for step, size in zip(angular, compute_shape))
        offsets_row = cp.arange(compute_shape[0], dtype=cp.float32) - center[0]
        offsets_col = cp.arange(compute_shape[1], dtype=cp.float32) - center[1]
        reciprocal_row = offsets_row[:, None] * (1.0 / (sampling[0] * compute_shape[0]))
        reciprocal_col = offsets_col[None, :] * (1.0 / (sampling[1] * compute_shape[1]))
        angle = -math.radians(rotation_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        k_row = (reciprocal_row * cosine + reciprocal_col * sine)[self.rows, self.cols]
        k_col = (-reciprocal_row * sine + reciprocal_col * cosine)[self.rows, self.cols]
        return wavelength, angular, k_row, k_col

    @property
    def column_bytes(self) -> int:
        """GPU bytes one detector column of a batch takes: its uint16 counts and its padded float32 FFT plane."""
        rows, cols = self.scan_shape
        return rows * (cols + 2) * 4 + rows * cols * 2

    def batch_size(self, reserve: int = 0, limit: int = 128) -> int:
        """Detector columns per batch that leave ``reserve`` bytes and most free memory to the caller."""
        free = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        return int(max(1, min(limit, self.num_bright_field_pixels, (free * 0.3 - reserve) // self.column_bytes)))

    def batches(self, order, batch_size):
        """Yield ``(first, stop, spectra)``: the half-plane FFTs of BF pixels ``order[first:stop]``, decoded exactly."""
        rows, cols = self.scan_shape
        buffer = cp.empty((batch_size, rows, cols + 2), cp.float32)
        counts = cp.empty(batch_size * rows * cols, cp.uint16)
        spectra = buffer.view(cp.complex64)
        errors = cp.zeros(1, cp.uint32)
        pixels = self.pixels[cp.asarray(order)]
        plan = previous_count = None
        for first in range(0, pixels.size, batch_size):
            stop = min(first + batch_size, pixels.size)
            count = stop - first
            streams = math.ceil(self.max_chunk_scans / self.interval) * count
            self.decode(
                (math.ceil(streams / 128), len(self.chunks)), (128,),
                (self.descriptors, self.decoding, pixels[first:stop], counts, errors,
                 *(np.uint32(value) for value in (count, math.prod(self.detector_shape), self.interval))),
            )
            self.to_fft(
                (math.ceil(rows * cols / 32),), (32, 8),
                (counts, buffer, *(np.uint32(value) for value in (count, rows, cols))),
            )
            if count != previous_count:
                plan = None
                plan = cufft.PlanNd(
                    (rows, cols), (rows, cols + 2), 1, rows * (cols + 2),
                    (rows, cols // 2 + 1), 1, rows * (cols // 2 + 1),
                    cufft.CUFFT_R2C, count, "C", 2, cols // 2 + 1,
                )
                previous_count = count
            plan.fft(buffer[:count], spectra[:count], cufft.CUFFT_FORWARD)
            yield first, stop, spectra[:count]
        if int(errors.get()[0]):
            raise ValueError("An ANS detector column failed decoding. Reload the intact acquisition before reconstructing.")

    def direct(self, optics, factor, policy, *, voltage_kV, semiangle_mrad, scan_sampling_A, rotation_deg,
               compute_loss):
        """The regularized ``phase`` or ``complex`` wave of the full scan (``direct.solve_streamed``)."""
        from types import SimpleNamespace

        from quantem.gpu.direct_ptycho.cuda.direct import solve_streamed

        started = time.perf_counter()
        wavelength, angular, k_row, k_col = self._geometry(voltage_kV, rotation_deg)
        semiangle_rad = semiangle_mrad * 1e-3
        # the resident engine's soft aperture at each BF pixel (engine.cache_rotation)
        alpha = cp.sqrt(k_row**2 + k_col**2) * wavelength
        azimuth = cp.arctan2(k_col, k_row)
        edge = cp.sqrt((cp.cos(azimuth) * angular[0] * 1e-3) ** 2 + (cp.sin(azimuth) * angular[1] * 1e-3) ** 2)
        aperture = cp.clip((semiangle_rad - alpha) / edge + 0.5, 0, 1).astype(cp.float32)
        rows, cols = self.scan_shape
        # G(0, k), the scan sum of each BF pixel, exactly from the detector total
        dc = cp.asarray(self.mean_pattern.ravel(), cp.float64)[self.pixels] * (rows * cols)
        source = SimpleNamespace(
            dc=dc, scan_shape=self.scan_shape, aperture=aperture, wavelength=wavelength,
            q_step=tuple(cp.fft.fftfreq(side, step).astype(cp.float32)[1] for side, step in zip(self.scan_shape, scan_sampling_A)),
            semiangle_rad=semiangle_rad, angular_rad=(angular[0] * 1e-3, angular[1] * 1e-3),
            detector_rows=self.rows, detector_cols=self.cols, k_row=k_row, k_col=k_col,
            column_bytes=self.column_bytes, batch_size=self.batch_size, batches=self.batches,
        )
        wave, diagnostics = solve_streamed(
            source, optics, factor, complex_object=policy["method"] == "complex",
            smoothness_inv_A=policy["smoothness_inv_A"], detector_weighting=policy["detector_weighting"],
            phase_regularization=policy["phase_regularization"],
            amplitude_regularization=policy.get("amplitude_regularization"), compute_loss=compute_loss,
        )
        cp.cuda.get_current_stream().synchronize()
        self.timings = {"reconstruct_s": time.perf_counter() - started, **diagnostics["streamed"]}
        return wave, diagnostics

    def phase(self, aberrations, *, voltage_kV, semiangle_mrad, scan_sampling_A,
              rotation_deg, upsampling_factor, sample_tilt_mrad, depth_spread_A, higher_order=None):
        """Decode, transform and correct every selected count on the source GPU.

        ``higher_order`` holds all 14 Krivanek terms (C10 and C12 included) packed by
        ``higher_order.pack_coefficients``, or None for C10/C12 alone: the same two kernels, and so the same
        corrections, as the resident-spectrum backend.
        """
        started = time.perf_counter()
        rows, cols = self.scan_shape
        output_shape = (rows * upsampling_factor, cols * upsampling_factor)
        output_pixels = math.prod(output_shape)
        free_bytes = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        # The complex128 spectrum, the in-place inverse FFT's work space and the float32 phase.
        fixed_bytes = output_pixels * 40
        per_column = rows * (cols + 2) * 4
        available = int(free_bytes * .8) - fixed_bytes
        if available < per_column * 4:
            raise MemoryError(
                f"The full {output_shape} SSB output needs more free GPU memory. "
                "Close other GPU allocations or explicitly select a smaller output sampling factor."
            )
        batch_size = max(1, min(32, self.num_bright_field_pixels,
                                min(2 << 30, available // 4) // per_column))
        wavelength = wavelength_A_from_kV(voltage_kV)
        angular = self.detector_sampling_mrad[::-1] if self.transposed else self.detector_sampling_mrad
        center = self.center[::-1] if self.transposed else self.center
        # Match the resident-spectrum backend's detector reciprocal coordinates.
        compute_shape = self.detector_shape[::-1] if self.transposed else self.detector_shape
        sampling = tuple(wavelength / (step * 1e-3 * size) for step, size in zip(angular, compute_shape))
        offsets_row = cp.arange(compute_shape[0], dtype=cp.float32) - center[0]
        offsets_col = cp.arange(compute_shape[1], dtype=cp.float32) - center[1]
        reciprocal_row = offsets_row[:, None] * (1.0 / (sampling[0] * compute_shape[0]))
        reciprocal_col = offsets_col[None, :] * (1.0 / (sampling[1] * compute_shape[1]))
        angle = -math.radians(rotation_deg)
        cosine, sine = math.cos(angle), math.sin(angle)
        k_row = (reciprocal_row * cosine + reciprocal_col * sine)[self.rows, self.cols]
        k_col = (-reciprocal_row * sine + reciprocal_col * cosine)[self.rows, self.cols]
        frequencies = []
        for side, output_side, step in zip(self.scan_shape, output_shape, scan_sampling_A):
            native = cp.fft.fftfreq(side, step).astype(cp.float32)
            frequencies.append(native if upsampling_factor == 1 else
                               (cp.fft.fftfreq(output_side) * output_side * native[1]).astype(cp.float32))
        coefficients = (aberrations["C10"], aberrations["C12"],
                        math.cos(2 * aberrations["phi12"]), math.sin(2 * aberrations["phi12"]))
        parameters = tuple(np.float32(value) for value in (
            wavelength, semiangle_mrad * 1e-3, angular[0] * 1e-3, angular[1] * 1e-3,
            *coefficients, math.pi / wavelength, depth_spread_A,
            sample_tilt_mrad[0] * 1e-3, sample_tilt_mrad[1] * 1e-3,
        ))
        buffer = cp.empty((batch_size, rows, cols + 2), cp.float32)
        counts = cp.empty(batch_size * rows * cols, cp.uint16)
        spectra = buffer.view(cp.complex64)
        spectrum = cp.zeros(output_shape, cp.complex128)
        dc = cp.empty(self.num_bright_field_pixels, cp.complex64)
        errors = cp.zeros(1, cp.uint32)
        kernel = thick_wave_sum_add_kernel
        if higher_order is not None:
            kernel = thick_wave_sum_full_add_kernel
            parameters = (*parameters, cp.asarray(higher_order, dtype=cp.float32))
        plan = None
        previous_count = None
        for first in range(0, self.num_bright_field_pixels, batch_size):
            stop = min(first + batch_size, self.num_bright_field_pixels)
            count = stop - first
            streams = math.ceil(self.max_chunk_scans / self.interval) * count
            self.decode(
                (math.ceil(streams / 128), len(self.chunks)), (128,),
                (self.descriptors, self.decoding, self.pixels[first:stop], counts, errors,
                 *(np.uint32(value) for value in
                   (count, math.prod(self.detector_shape), self.interval))),
            )
            self.to_fft(
                (math.ceil(rows * cols / 32),), (32, 8),
                (counts, buffer, *(np.uint32(value) for value in (count, rows, cols))),
            )
            if count != previous_count:
                plan = None
                plan = cufft.PlanNd(
                    (rows, cols), (rows, cols + 2), 1, rows * (cols + 2),
                    (rows, cols // 2 + 1), 1, rows * (cols // 2 + 1),
                    cufft.CUFFT_R2C, count, "C", 2, cols // 2 + 1,
                )
                previous_count = count
            plan.fft(buffer[:count], spectra[:count], cufft.CUFFT_FORWARD)
            dc[first:stop] = spectra[:count, 0, 0]
            # the kernel's arguments in its order: bounds (read only by the bounded variant, so any float array),
            # then num_bf, native rows, cols and stored cols, output rows, cols and stored cols, first BF, group size
            kernel(
                (math.ceil(output_pixels / 128), 1), (128,),
                (spectra, *frequencies, k_row[first:stop], k_col[first:stop], spectrum, k_row,
                 *(np.int32(value) for value in (count, rows, cols, cols // 2 + 1,
                                               *output_shape, output_shape[1], 0, count)),
                 np.complex64(0), *parameters),
            )
        if int(errors.get()[0]):
            raise ValueError("An ANS detector column failed decoding. Reload the intact acquisition before reconstructing.")
        del buffer, counts, spectra, plan
        spectrum /= self.num_bright_field_pixels
        spectrum[0, 0] = dc.mean()
        wave = cupyx.scipy.fft.ifft2(spectrum, overwrite_x=True)
        phase = cp.empty(output_shape, cp.float32)
        cp.arctan2(wave.imag, wave.real, out=phase)
        cp.cuda.get_current_stream().synchronize()
        self.timings = {"reconstruct_s": time.perf_counter() - started,
                        "detector_batch_size": batch_size}
        return phase, None


# The names before the Direct Ptycho rename: the same objects, so existing imports, isinstance checks and pickles work.
StreamedSSB = StreamedDirectPtycho
