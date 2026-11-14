"""Full-scan direct ptychography with a bounded batch of exact detector columns.

Only detector columns are batched. Every forward transform spans every scan
position; no scan tiles are reconstructed or stitched. The correction kernel
is the same one used by the resident-spectrum direct ptychography implementation.
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
from quantem.gpu.direct_ptycho.cuda.kernels.engine import thick_wave_sum_full_kernel, thick_wave_sum_kernel
from quantem.gpu.direct_ptycho.higher_order import pack_coefficients


# Detector columns decoded and transformed together. The FFT-input kernel (sc_columns_fft) moves one tile of 32
# columns through shared memory, so columns past the 32nd of a batch are never written: batches of 64 and 128
# returned phases wrong by 0.02 to 0.04 rad, and NaN, on a real 512 x 512 scan (2026-11-12).
COLUMN_BATCH = 32


def free_device_bytes() -> int:
    """Bytes this process can still allocate on the current device: free device memory plus the pool's idle blocks.

    A limit set on the CuPy pool (``set_limit``) caps the answer, so a small card can be rehearsed on a large one.
    """
    pool = cp.get_default_memory_pool()
    free_bytes = cp.cuda.runtime.memGetInfo()[0] + pool.free_bytes()
    limit = pool.get_limit()
    return min(free_bytes, limit - pool.used_bytes()) if limit else free_bytes


class StreamedDirect:
    """Borrow a complete exact-count resident and retain its immutable chunks.

    ``full_scan_only`` marks a session that cannot also hold the dense bright-field spectrum (a scan beyond 1024
    pixels per side, or one routed here because the device is too small): its fit reads a scan region and its
    full-scan reconstruction reports no loss.
    """

    def __init__(self, data, *, mean_pattern, center, radius, threshold,
                 semiangle_mrad, detector_sampling_mrad, scan_detector_rotation_flipped, full_scan_only):
        source = data.data
        if type(source) is not StreamedCounts:
            raise NotImplementedError(
                "Large CUDA direct ptychography currently needs lossless integer ANS storage. "
                "Load a uint8/uint16 acquisition with io.load before creating direct ptychography."
            )
        self.full_scan_only = bool(full_scan_only)
        self.scan_shape = tuple(data.shape[:2])
        self.detector_shape = tuple(data.shape[2:])
        if max(self.scan_shape) > 4096 or any(side % 2 for side in self.scan_shape):
            raise ValueError(
                f"Streamed direct ptychography supports even scan dimensions up to 4096; got {self.scan_shape}."
            )
        self.source = source
        position = 0
        for chunk in source.chunks:
            if chunk.first != position or chunk.scans <= 0:
                raise ValueError("Direct ptychography needs complete consecutive scan coverage. Finish loading the acquisition first.")
            position += chunk.scans
        if source.is_released or source.released_scans or position != math.prod(self.scan_shape) or source.ready_scans != position:
            raise ValueError("Direct ptychography needs every scan position resident. Finish loading the acquisition first.")
        self.device = source.device
        self.interval = source.interval
        self.decoding = source.decoding
        self.decode = source.kernels["columns_u16"]
        self.to_fft = source.kernels["columns_fft"]
        self._borrow_chunks()
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
            # a file without a recorded detector sampling: the disk edge sits at the semiangle
            self.detector_sampling_mrad = detector.detector_sampling_mrad(self.mean_pattern, semiangle_mrad)
        self.transposed = bool(scan_detector_rotation_flipped)
        mean = self.mean_pattern.T if self.transposed else self.mean_pattern
        compute_center = self.center[::-1] if self.transposed else self.center
        self.rows, self.cols, _, _ = select_bright_field(
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

    def _borrow_chunks(self):
        """Keep the source's encoded chunks and the device addresses the column decoder reads them at."""
        self.chunks = tuple(self.source.chunks)
        self.descriptors = cp.asarray([
            [*(array.data.ptr for array in chunk.arrays[:3]), chunk.first, chunk.scans]
            for chunk in self.chunks
        ], dtype=cp.uint64)
        self.max_chunk_scans = max(chunk.scans for chunk in self.chunks)

    def park(self):
        """Move the encoded acquisition to host memory, freeing the device for a fit (see StreamedCounts.park)."""
        self.chunks, self.descriptors = (), None
        self.source.park()

    def restore(self, *, keep_host: bool = False):
        """Return the whole encoded acquisition to the device after ``park``.

        ``keep_host`` keeps the host copy, so the next ``park`` only frees the device (no copy back).
        """
        self.source.restore(keep_host=keep_host)
        self._borrow_chunks()

    def band_spectra(self, flat, scan_region=None):
        """G(q, k) of every bright-field pixel at the stored half-plane frequencies ``flat``, in host memory.

        Over every scan position, or over ``scan_region`` (row_start, row_stop, col_start, col_stop) only, whose
        own spectra ``flat`` then indexes.

        One pass of the reconstruction's column batches (``_column_spectra``): each batch's kept frequencies are
        gathered on the device and copied into one page-locked host array, so the device holds one batch of spectra
        at a time, never the whole result. Returns (spectra (bright-field pixels, len(flat)) complex64, the mean DC
        value of G as the dense spectrum reports it).
        """
        memory = cp.cuda.alloc_pinned_memory(self.num_bright_field_pixels * flat.size * 8)
        spectra = np.frombuffer(memory, np.complex64, self.num_bright_field_pixels * flat.size).reshape(
            self.num_bright_field_pixels, flat.size)
        dc = cp.empty(self.num_bright_field_pixels, cp.complex64)
        for first, stop, batch in self._column_spectra(COLUMN_BATCH, scan_region):
            dc[first:stop] = batch[:, 0, 0]
            batch.reshape(stop - first, -1)[:, flat].get(out=spectra[first:stop])
        return spectra, complex(dc.mean().get())

    def _column_spectra(self, batch_size, scan_region=None):
        """Each batch of ``batch_size`` bright-field pixels as (first, stop, spectra): decoded from the encoded
        acquisition, transformed over every scan position, or over ``scan_region`` (row_start, row_stop, col_start,
        col_stop) only, the Hermitian half plane (count, rows, cols / 2 + 1).

        The spectra share one buffer: a batch is valid until the next is drawn.
        """
        rows, cols = self.scan_shape
        row_start, row_stop, col_start, col_stop = (0, rows, 0, cols) if scan_region is None else scan_region
        out_rows, out_cols = row_stop - row_start, col_stop - col_start
        buffer = cp.empty((batch_size, out_rows, out_cols + 2), cp.float32)
        counts = cp.empty(batch_size * rows * cols, cp.uint16)
        spectra = buffer.view(cp.complex64)
        errors = cp.zeros(1, cp.uint32)
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
            # the decoder writes each batch scan-major (position, pixel), so a region is a block of rows of it
            selected = counts if scan_region is None else cp.ascontiguousarray(
                counts[:rows * cols * count].reshape(rows, cols, count)[row_start:row_stop, col_start:col_stop])
            self.to_fft(
                (math.ceil(out_rows * out_cols / 32),), (32, 8),
                (selected, buffer, *(np.uint32(value) for value in (count, out_rows, out_cols))),
            )
            if count != previous_count:
                plan = None
                plan = cufft.PlanNd(
                    (out_rows, out_cols), (out_rows, out_cols + 2), 1, out_rows * (out_cols + 2),
                    (out_rows, out_cols // 2 + 1), 1, out_rows * (out_cols // 2 + 1),
                    cufft.CUFFT_R2C, count, "C", 2, out_cols // 2 + 1,
                )
                previous_count = count
            plan.fft(buffer[:count], spectra[:count], cufft.CUFFT_FORWARD)
            yield first, stop, spectra[:count]
        if int(errors.get()[0]):
            raise ValueError("An ANS detector column failed decoding. Reload the intact acquisition before reconstructing.")

    def phase(self, aberrations, *, voltage_kV, semiangle_mrad, scan_sampling_A,
              rotation_deg, upsample, sample_tilt_mrad, depth_spread_A,
              higher_order_magnitudes=None, higher_order_angles=None):
        """Decode, transform and correct every selected count on the source GPU.

        ``higher_order_magnitudes`` and ``higher_order_angles`` (14 slots, Angstrom and rad, C10 and C12 first)
        select the kernel that evaluates chi through fifth order; without them only C10 and C12 enter.
        """
        started = time.perf_counter()
        rows, cols = self.scan_shape
        output_shape = (rows * upsample, cols * upsample)
        output_pixels = math.prod(output_shape)
        free_bytes = free_device_bytes()
        # Two complex128 output planes, inverse work space and final phase.
        fixed_bytes = output_pixels * 56
        per_column = rows * (cols + 2) * 4
        available = int(free_bytes * .8) - fixed_bytes
        if available < per_column * 4:
            raise MemoryError(
                f"The full {output_shape} direct ptychography output needs more free GPU memory. "
                "Close other GPU allocations or explicitly select a smaller output sampling factor."
            )
        batch_size = max(1, min(COLUMN_BATCH, self.num_bright_field_pixels,
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
            frequencies.append(native if upsample == 1 else
                               (cp.fft.fftfreq(output_side) * output_side * native[1]).astype(cp.float32))
        coefficients = (aberrations["C10"], aberrations["C12"],
                        math.cos(2 * aberrations["phi12"]), math.sin(2 * aberrations["phi12"]))
        parameters = tuple(np.float32(value) for value in (
            wavelength, semiangle_mrad * 1e-3, angular[0] * 1e-3, angular[1] * 1e-3,
            *coefficients, math.pi / wavelength, depth_spread_A,
            sample_tilt_mrad[0] * 1e-3, sample_tilt_mrad[1] * 1e-3,
        ))
        kernel = thick_wave_sum_kernel
        if higher_order_magnitudes is not None:
            kernel = thick_wave_sum_full_kernel
            parameters = (*parameters, cp.asarray(pack_coefficients(higher_order_magnitudes, higher_order_angles)))
        spectrum = cp.zeros(output_shape, cp.complex128)
        partial = cp.empty((1, *output_shape), cp.complex128)
        dc = cp.empty(self.num_bright_field_pixels, cp.complex64)
        for first, stop, spectra in self._column_spectra(batch_size):
            count = stop - first
            dc[first:stop] = spectra[:, 0, 0]
            # the kernel's arguments in its order: bounds (read only by the bounded variant, so any float array),
            # then num_bf, native rows, cols and stored cols, output rows, cols and stored cols, first BF, group size
            kernel(
                (math.ceil(output_pixels / 128), 1), (128,),
                (spectra, *frequencies, k_row[first:stop], k_col[first:stop], partial, k_row,
                 *(np.int32(value) for value in (count, rows, cols, cols // 2 + 1,
                                               *output_shape, output_shape[1], 0, count)),
                 np.complex64(0), *parameters),
            )
            spectrum += partial[0]
        del spectra, partial
        spectrum /= self.num_bright_field_pixels
        spectrum[0, 0] = dc.mean()
        wave = cupyx.scipy.fft.ifft2(spectrum, overwrite_x=True)
        phase = cp.empty(output_shape, cp.float32)
        cp.arctan2(wave.imag, wave.real, out=phase)
        cp.cuda.get_current_stream().synchronize()
        self.timings = {"reconstruct_s": time.perf_counter() - started,
                        "detector_batch_size": batch_size}
        return phase, None
