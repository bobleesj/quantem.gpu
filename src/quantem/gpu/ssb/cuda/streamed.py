"""Full-scan SSB with a bounded batch of exact detector columns.

Only detector columns are batched. Every forward transform spans every scan
position; no scan tiles are reconstructed or stitched. The correction kernel
is the same one used by the resident-spectrum SSB implementation.
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
from quantem.gpu.ssb.brightfield import disk_edge_radius
from quantem.gpu.ssb.cuda.brightfield import select_bright_field
from quantem.gpu.ssb.cuda.kernels.engine import thick_wave_sum_kernel


class StreamedSSB:
    """Borrow a complete exact-count resident and retain its immutable chunks."""

    def __init__(self, data, *, mean_pattern, center, radius, threshold,
                 semiangle_mrad, detector_sampling_mrad, detector_transposed):
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
        self.decode = source.kernels["columns_f32"]
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
        self.mean_pattern = np.where(valid, self.mean_pattern, 0)
        if radius is None:
            selected = self.mean_pattern > self.mean_pattern.mean() + self.mean_pattern.std()
            rows, cols = np.nonzero(selected)
            if not rows.size:
                raise ValueError("No bright-field disk found. Supply calibrated detector geometry.")
            if center is None:
                center = (float(rows.mean()), float(cols.mean()))
            radius = math.sqrt(rows.size / math.pi)
        elif center is None:
            rows, cols = np.nonzero(self.mean_pattern > self.mean_pattern.max() * threshold)
            weights = self.mean_pattern[rows, cols]
            center = (float(np.average(rows, weights=weights)), float(np.average(cols, weights=weights)))
        self.center = tuple(float(value) for value in center)
        self.radius = float(radius)
        self.detector_sampling_mrad = detector_sampling_mrad
        if self.detector_sampling_mrad is None:
            self.detector_sampling_mrad = (semiangle_mrad / disk_edge_radius(self.mean_pattern),) * 2
        self.transposed = bool(detector_transposed)
        mean = self.mean_pattern.T if self.transposed else self.mean_pattern
        compute_center = self.center[::-1] if self.transposed else self.center
        self.rows, self.cols, _ = select_bright_field(
            cp.asarray(mean, dtype=cp.float32)[None, None], threshold,
            self.radius, compute_center,
        )
        selected_valid = cp.asarray(valid.T if self.transposed else valid)[self.rows, self.cols]
        self.rows, self.cols = self.rows[selected_valid], self.cols[selected_valid]
        if not self.rows.size:
            raise ValueError("The bright-field disk contains no valid detector pixels.")
        original_rows, original_cols = (self.cols, self.rows) if self.transposed else (self.rows, self.cols)
        self.pixels = (original_rows * self.detector_shape[1] + original_cols).astype(cp.uint32)
        self.num_bright_field_pixels = int(self.pixels.size)
        self.timings = {}

    def phase(self, aberrations, *, voltage_kV, semiangle_mrad, scan_sampling_A,
              rotation_deg, upsampling_factor, sample_tilt_mrad, thickness):
        """Decode, transform and correct every selected count on the source GPU."""
        started = time.perf_counter()
        rows, cols = self.scan_shape
        output_shape = (rows * upsampling_factor, cols * upsampling_factor)
        output_pixels = math.prod(output_shape)
        free_bytes = cp.cuda.runtime.memGetInfo()[0] + cp.get_default_memory_pool().free_bytes()
        # Two complex128 output planes, inverse work space and final phase.
        fixed_bytes = output_pixels * 56
        per_column = rows * (cols + 2) * 4
        available = int(free_bytes * .8) - fixed_bytes
        if available < per_column * 3:
            raise MemoryError(
                f"The full {output_shape} SSB output needs more free GPU memory. "
                "Close other GPU allocations or explicitly select a smaller output sampling factor."
            )
        batch_size = max(1, min(32, self.num_bright_field_pixels,
                                min(2 << 30, available // 3) // per_column))
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
            *coefficients, math.pi / wavelength, thickness,
            sample_tilt_mrad[0] * 1e-3, sample_tilt_mrad[1] * 1e-3,
        ))
        buffer = cp.empty((batch_size, rows, cols + 2), cp.float32)
        spectra = buffer.view(cp.complex64)
        spectrum = cp.zeros(output_shape, cp.complex128)
        partial = cp.empty((1, *output_shape), cp.complex128)
        dc = cp.empty(self.num_bright_field_pixels, cp.complex64)
        errors = cp.zeros(1, cp.uint32)
        plan = None
        previous_count = None
        for first in range(0, self.num_bright_field_pixels, batch_size):
            stop = min(first + batch_size, self.num_bright_field_pixels)
            count = stop - first
            streams = math.ceil(self.max_chunk_scans / self.interval) * count
            self.decode(
                (math.ceil(streams / 128), len(self.chunks)), (128,),
                (self.descriptors, self.decoding, self.pixels[first:stop], buffer, errors,
                 *(np.uint32(value) for value in
                   (count, math.prod(self.detector_shape), self.interval, rows, cols))),
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
            thick_wave_sum_kernel(
                (math.ceil(output_pixels / 128), 1), (128,),
                (spectra, *frequencies, k_row[first:stop], k_col[first:stop], partial,
                 *(np.int32(value) for value in (count, rows, cols, cols // 2 + 1,
                                               *output_shape, 0, count)),
                 np.complex64(0), *parameters),
            )
            spectrum += partial[0]
        if int(errors.get()[0]):
            raise ValueError("An ANS detector column failed decoding. Reload the intact acquisition before reconstructing.")
        del buffer, spectra, plan, partial
        spectrum /= self.num_bright_field_pixels
        spectrum[0, 0] = dc.mean()
        wave = cupyx.scipy.fft.ifft2(spectrum, overwrite_x=True)
        phase = cp.empty(output_shape, cp.float32)
        cp.arctan2(wave.imag, wave.real, out=phase)
        cp.cuda.get_current_stream().synchronize()
        self.timings = {"reconstruct_s": time.perf_counter() - started,
                        "detector_batch_size": batch_size}
        return phase, None
