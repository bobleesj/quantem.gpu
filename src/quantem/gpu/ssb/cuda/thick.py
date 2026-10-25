"""Thick, tilted-sample SSB on CUDA: depth-averaged correction and the least-squares fit objective.

Standard SSB treats the sample as one plane at the probe defocus. A crystal of thickness t tilted by theta sees a range of
defoci and shifts through its depth; averaging the SSB transfer terms over that depth gives each its own sinc weight (see
``kernels/engine.py``). ``ThickSample`` evaluates that model on one engine's prepared bright-field spectra ``G_qk``:
phase images for previews and reconstruction, and the agreement that ``find_aberrations(tilt=True)`` maximises.
"""

import math

import cupy as cp
import cupyx.scipy.fft
import numpy as np

from quantem.gpu.ssb.cuda.kernels.engine import (
    THICK_FIT_MAX_BATCH,
    thick_correct_kernel,
    thick_correct_full_kernel,
    thick_correct_planes_kernel,
    thick_fit_batch_kernel,
    thick_fit_kernel,
    thick_phase_moments_kernel,
    thick_wave_sum_kernel,
    thick_wave_sum_full_kernel,
    thick_wave_sum_bounded_kernel,
    thick_wave_sum_bounded_full_kernel,
)
from quantem.gpu.ssb.cuda import chebyshev
from quantem.gpu.ssb.higher_order import pack_coefficients
from quantem.gpu.ssb.units import ENGINE_PER_NM, aberrations_to_engine, higher_order_arrays, validate_aberrations


class ThickSample:
    """The thick-sample model on one ``SSBEngine``'s prepared ``G_qk`` and rotation geometry.

    The engine owns the evidence and the current rotation; this object owns what only the thick model needs: the
    spatial-frequency band of the fit objective and the float64 wave-sum scratch buffer, each rebuilt only when the
    band or the grid changes.
    """

    def __init__(self, engine) -> None:
        self.engine = engine
        self._band_key = None
        self._band = None
        self._wave_sum_buffer = None
        self._wave_pairing = None
        self._wave_bounds = None
        self._wave_cells = None
        self._wave_lens_cache = None

    def reconstruct(
        self,
        C10: float,
        C12: float,
        phi12: float,
        sample_tilt_mrad: tuple[float, float],
        thickness: float,
        compute_loss: bool = True,
        chunk_bytes: int = 1 << 30,
        upsampling_factor: int = 1,
        phase_estimator: str = "mean_phase",
        higher_order_magnitudes: np.ndarray | None = None,
        higher_order_angles: np.ndarray | None = None,
    ) -> tuple[cp.ndarray, float | None]:
        """Mean phase and variance loss for a thick, tilted crystal (see ``thick_correct_kernel``).

        Same outputs and definitions as ``SSBEngine.reconstruct_with_loss`` (mean over bright-field pixels of the
        per-pixel phase; loss = mean over the image of the per-pixel phase variance), with each pixel's SSB correction
        averaged over the sample depth. ``sample_tilt_mrad`` = (row, col) in the scan frame, ``thickness`` in the C10 unit; thickness
        0 reproduces the standard reconstruction. Reference path: element-wise correction then one inverse FFT per pixel,
        processed in chunks of bright-field pixels, not the fused FFT kernels, so it is slower than the standard path.

        ``phase_of_mean`` sums corrected spectra in bounded BF groups before a
        single inverse FFT and phase extraction; it reads upsampling aliases
        directly from the native spectra without a per-BF expanded image. Linear averaging commutes with
        the inverse FFT, but not with phase extraction. Its diagnostic loss is
        evaluated separately with the legacy estimator on the native grid.
        """
        engine = self.engine
        if phase_estimator not in ("mean_phase", "phase_of_mean"):
            raise ValueError(
                "phase_estimator must be 'mean_phase' or 'phase_of_mean'; "
                f"got {phase_estimator!r}."
            )
        average_wave = phase_estimator == "phase_of_mean"
        cache = engine._cache
        num_bright_field_pixels, ny, nx = engine._bf_grid()
        native_ny, native_nx = ny, nx
        if type(upsampling_factor) is not int or upsampling_factor not in (1, 2, 3, 4, 8):
            raise ValueError("upsampling_factor must be 1, 2, 3, 4, or 8.")
        qrow, qcol = self._frequencies(upsampling_factor)
        ny, nx = qrow.size, qcol.size
        qx, qy = qrow.reshape(1, ny, 1), qcol.reshape(1, 1, nx)
        if average_wave:
            return self._reconstruct_wave(
                C10, C12, phi12, sample_tilt_mrad, thickness, qx.ravel(), qy.ravel(),
                compute_loss=compute_loss, chunk_bytes=chunk_bytes,
                higher_order_magnitudes=higher_order_magnitudes,
                higher_order_angles=higher_order_angles,
            )
        if upsampling_factor == 1 and higher_order_magnitudes is None:
            return self._reconstruct_native(C10, C12, phi12, sample_tilt_mrad, thickness, compute_loss, chunk_bytes)
        half = engine.gqk_is_half_plane()
        # other half of the plane for a real bright-field image: G(-q) = conj(G(q))
        neg_rows = cp.asarray((-np.arange(native_ny)) % native_ny)
        neg_cols = cp.asarray((native_nx - np.arange(native_nx // 2 + 1, native_nx)) % native_nx)
        chunk = max(1, int(chunk_bytes // (ny * nx * 8 * 3)))
        phase_sum = cp.zeros((ny, nx), dtype=cp.float32)
        phase_sumsq = cp.zeros((ny, nx), dtype=cp.float32)
        params = self._kernel_scalars(C10, C12, phi12, sample_tilt_mrad, thickness)
        kernel = thick_correct_kernel
        if higher_order_magnitudes is not None:
            kernel = thick_correct_full_kernel
            params = (*params, cp.asarray(pack_coefficients(higher_order_magnitudes, higher_order_angles)))
        for start in range(0, num_bright_field_pixels, chunk):
            stop = min(num_bright_field_pixels, start + chunk)
            if half:
                source = engine.G_qk[start:stop]
                full = cp.empty((stop - start, native_ny, native_nx), dtype=cp.complex64)
                full[:, :, : native_nx // 2 + 1] = source
                full[:, :, native_nx // 2 + 1:] = cp.conj(source[:, neg_rows][:, :, neg_cols])
            else:
                full = engine.G_qk[start:stop]
            if upsampling_factor != 1:
                full = cp.tile(full, (1, upsampling_factor, upsampling_factor))
            kx = cache["kx_bf"][start:stop].reshape(-1, 1, 1)
            ky = cache["ky_bf"][start:stop].reshape(-1, 1, 1)
            corrected = kernel(full, qx, qy, kx, ky, *params)
            if upsampling_factor != 1:
                # Exclude scan-mean replicas; retain the true DC below.
                corrected[:, ::native_ny, ::native_nx] = 0
            corrected[:, 0, 0] = engine._dc_value_host
            angles = cp.angle(cp.fft.ifft2(corrected, axes=(1, 2)))
            phase_sum += angles.sum(axis=0)
            if compute_loss:
                phase_sumsq += (angles * angles).sum(axis=0)
            del full, corrected, angles
        mean_phase = phase_sum / float(num_bright_field_pixels)
        if not compute_loss:
            return mean_phase, None
        loss = float(cp.mean(phase_sumsq / float(num_bright_field_pixels) - mean_phase * mean_phase))
        return mean_phase, loss

    def _reconstruct_native(self, C10, C12, phi12, sample_tilt_mrad, thickness, compute_loss, chunk_bytes):
        """``reconstruct`` at the native sampling with C10/C12 optics: the same per-pixel correction, phase and loss.

        The fit's native phase diagnostic and every slider-release loss run this. Each group of BF pixels is corrected
        straight into full-plane spectra (``thick_correct_planes_kernel`` reads the stored half plane's mirror in
        place), inverse transformed without the 1 / (ny nx) scale, which the phase does not see (a power of two, so the
        angles are the same bits), and reduced to per-pixel phase sums by ``thick_phase_moments_kernel`` without
        storing the angle images. The reference loop above materialized the full plane, the angles and their squares.
        """
        engine = self.engine
        cache = engine._cache
        num_bright_field_pixels, ny, nx = engine._bf_grid()
        plane = ny * nx
        chunk = min(num_bright_field_pixels, max(1, int(chunk_bytes // (plane * 8 * 3))))
        planes = cp.empty((chunk, ny, nx), dtype=cp.complex64)
        phase_sum = cp.zeros((ny, nx), dtype=cp.float32)
        phase_sumsq = cp.zeros((ny, nx), dtype=cp.float32) if compute_loss else phase_sum
        params = self._kernel_scalars(C10, C12, phi12, sample_tilt_mrad, thickness)
        qrow = cache["qx_1d"].astype(cp.float32, copy=False)
        qcol = cache["qy_1d"].astype(cp.float32, copy=False)
        blocks = (plane + 255) // 256
        for start in range(0, num_bright_field_pixels, chunk):
            count = min(chunk, num_bright_field_pixels - start)
            group = planes[:count]
            thick_correct_planes_kernel(
                (blocks, count), (256,),
                (engine.G_qk, qrow, qcol, cache["kx_bf"], cache["ky_bf"], group, np.int32(ny), np.int32(nx),
                 np.int32(engine.G_qk.shape[-1]), np.int32(start), np.complex64(engine._dc_value_host), *params),
            )
            images = cupyx.scipy.fft.ifft2(group, axes=(1, 2), norm="forward", overwrite_x=True)
            thick_phase_moments_kernel((blocks,), (256,), (images, phase_sum, phase_sumsq, np.int32(count),
                                                           np.int32(plane), np.int32(compute_loss)))
        mean_phase = phase_sum / float(num_bright_field_pixels)
        if not compute_loss:
            return mean_phase, None
        loss = float(cp.mean(phase_sumsq / float(num_bright_field_pixels) - mean_phase * mean_phase))
        return mean_phase, loss

    def _reconstruct_wave(
        self,
        C10: float,
        C12: float,
        phi12: float,
        sample_tilt_mrad: tuple[float, float],
        thickness: float,
        qrow: cp.ndarray,
        qcol: cp.ndarray,
        *,
        compute_loss: bool,
        chunk_bytes: int,
        higher_order_magnitudes: np.ndarray | None,
        higher_order_angles: np.ndarray | None,
    ) -> tuple[cp.ndarray, float | None]:
        """Accumulate float32 corrections in float64 before a single inverse FFT."""
        spectrum = self._wave_spectrum(
            C10, C12, phi12, sample_tilt_mrad, thickness, qrow, qcol, chunk_bytes=chunk_bytes,
            higher_order_magnitudes=higher_order_magnitudes, higher_order_angles=higher_order_angles,
        )
        phase = self._wave_phase(spectrum, qrow.size, qcol.size)
        loss = None
        if compute_loss:
            _, loss = self.reconstruct(
                C10, C12, phi12, sample_tilt_mrad, thickness,
                compute_loss=True, chunk_bytes=chunk_bytes,
                higher_order_magnitudes=higher_order_magnitudes,
                higher_order_angles=higher_order_angles,
            )
        return phase, loss

    def _wave_spectrum(
        self,
        C10: float,
        C12: float,
        phi12: float,
        sample_tilt_mrad: tuple[float, float],
        thickness: float,
        qrow: cp.ndarray,
        qcol: cp.ndarray,
        *,
        chunk_bytes: int,
        higher_order_magnitudes: np.ndarray | None,
        higher_order_angles: np.ndarray | None,
        pairs: int = 0,
    ) -> cp.ndarray:
        """The BF sum of corrected spectra, (rows, unique columns) complex128, before the mirror and the FFT.

        ``pairs`` 1 or 2 sums only the single-sideband or only the triple-overlap
        terms (fine grids); their sum is the full spectrum (0), which the
        SSB drag window uses to interpolate the first and evaluate the second.
        """
        engine = self.engine
        cache = engine._cache
        num_bright_field_pixels = int(cache["num_bright_field_pixels"])
        rows, cols = qrow.size, qcol.size
        unique_cols = self._wave_columns(rows, cols)
        # Fine grids otherwise write and reduce a large partial plane for
        # every 32 BF pixels. Larger groups retain all float32 corrections
        # and float64 accumulation while reducing that intermediate traffic.
        fine_grid = rows > cache["ny"] and cols > cache["nx"]
        group_size = 128 if fine_grid else 32
        order = frequencies = None
        slots = rows * unique_cols
        if pairs == 2:
            # Triple overlap needs |q| <= R: a quarter of the support, so the sum runs over that list of
            # frequencies only, and over BF pixels in compact cells, whose small boxes cull the lens better.
            group_size, order = 64, self._wave_order()
            frequencies = self._wave_lens(qrow, qcol, unique_cols)
            slots = frequencies.size
        # a short frequency list affords more BF groups per launch within the same partial memory
        groups = max(1, min(16 if frequencies is None else 64, math.ceil(num_bright_field_pixels / group_size),
                            chunk_bytes // ((rows * cols if frequencies is None else slots) * 16)))
        shape = (groups, slots)
        if self._wave_sum_buffer is None or self._wave_sum_buffer.shape != shape:
            self._wave_sum_buffer = cp.empty(shape, dtype=cp.complex128)
        partial = self._wave_sum_buffer
        # Native grids do not benefit from group culling; that kernel does not
        # read the bounds argument and keeps its existing execution path.
        if not fine_grid:
            bounds = cache["kx_bf"]
        else:
            bounds = self._wave_group_bounds(group_size) if order is None else self._wave_group_bounds(group_size, order)
        spectrum = cp.zeros((rows, unique_cols), dtype=cp.complex128)
        params = self._kernel_scalars(C10, C12, phi12, sample_tilt_mrad, thickness)
        kernel = thick_wave_sum_bounded_kernel if fine_grid else thick_wave_sum_kernel
        if higher_order_magnitudes is not None:
            kernel = thick_wave_sum_bounded_full_kernel if fine_grid else thick_wave_sum_full_kernel
        if pairs:
            kernel = _pair_kernel(pairs, higher_order_magnitudes is not None)
        if higher_order_magnitudes is not None:
            params = (*params, cp.asarray(pack_coefficients(higher_order_magnitudes, higher_order_angles)))
        if order is not None:
            params = (*params, order, frequencies, np.int32(slots))
        for start in range(0, num_bright_field_pixels, group_size * groups):
            active = min(groups, math.ceil((num_bright_field_pixels - start) / group_size))
            kernel(
                (math.ceil(slots / 128), active), (128,),
                (engine.G_qk, qrow, qcol, cache["kx_bf"], cache["ky_bf"], partial, bounds,
                 np.int32(num_bright_field_pixels), np.int32(cache["ny"]), np.int32(cache["nx"]),
                 np.int32(engine.G_qk.shape[-1]), np.int32(rows), np.int32(cols), np.int32(unique_cols),
                 np.int32(start), np.int32(group_size), np.complex64(engine._dc_value_host),
                 *params),
            )
            if frequencies is None:
                spectrum += partial[:active].sum(axis=0, dtype=cp.complex128).reshape(rows, unique_cols)
            else:
                spectrum.ravel()[frequencies] += partial[:active].sum(axis=0, dtype=cp.complex128)
        return spectrum

    def _wave_phase(self, spectrum: cp.ndarray, rows: int, cols: int) -> cp.ndarray:
        """float32 phase of the BF-averaged wave from ``_wave_spectrum``'s half or full plane."""
        num_bright_field_pixels = int(self.engine._cache["num_bright_field_pixels"])
        unique_cols = spectrum.shape[1]
        if unique_cols != cols:
            # Every term obeys F(-q) = -conj(F(q)): G(-q, k) = conj G(q, k) for real
            # counts, and t1(-q, k) = conj t2(q, k), t2(-q, k) = conj t1(q, k) with
            # swapped depth weights for any chi, odd orders included, so the
            # correction gamma / |gamma| changes to -conj. The guard excludes Nyquist support.
            negative = -cp.conj(spectrum[(-cp.arange(rows)) % rows, 1:cols // 2][:, ::-1])
            spectrum = cp.concatenate((spectrum, negative), axis=1)
        # the BF sum and the inverse FFT run in complex128; the phase is returned float32 like every other path
        return cp.angle(cp.fft.ifft2(spectrum / num_bright_field_pixels)).astype(cp.float32)

    def _frequencies(self, upsampling_factor: int) -> tuple[cp.ndarray, cp.ndarray]:
        """Row and column frequencies (1/A) of the output grid.

        Same field of view at every factor: Fourier spacing is unchanged; the
        measured scan-frequency aliases are tiled and the kernels evaluated at
        the new q.
        """
        cache = self.engine._cache
        if upsampling_factor == 1:
            return cache["qx_1d"], cache["qy_1d"]
        ny, nx = cache["ny"] * upsampling_factor, cache["nx"] * upsampling_factor
        return ((cp.fft.fftfreq(ny) * ny * cache["qx_1d"][1]).astype(cp.float32),
                (cp.fft.fftfreq(nx) * nx * cache["qy_1d"][1]).astype(cp.float32))

    def _wave_group_bounds(self, group_size: int, order: cp.ndarray | None = None) -> cp.ndarray:
        """Cache exact coordinate bounds for each complete BF group on GPU (of the pixels in ``order``, if given)."""
        cache = self.engine._cache
        krow, kcol = cache["kx_bf"], cache["ky_bf"]
        cached = self._wave_bounds
        if (cached is not None and cached[0] == group_size
                and cached[1] is krow and cached[2] is kcol and cached[4] is order):
            return cached[3]
        if order is not None:
            krow, kcol = krow[order], kcol[order]
        # Repeat the final coordinate only to pad the reduction. The correction
        # loop still stops at the actual BF count, with its original sum order.
        padding = (-krow.size) % group_size
        limits = []
        for coordinates in (krow, kcol):
            groups = cp.pad(coordinates, (0, padding), mode="edge").reshape(-1, group_size)
            limits.extend((groups.min(axis=1), groups.max(axis=1)))
        bounds = cp.stack(limits, axis=1)
        # Array identity also invalidates this cache after rotation or an
        # explicitly requested BF subset, even when the BF count is unchanged.
        self._wave_bounds = (group_size, cache["kx_bf"], cache["ky_bf"], bounds, order)
        return bounds

    def _wave_lens(self, qrow: cp.ndarray, qcol: cp.ndarray, unique_cols: int) -> cp.ndarray:
        """Flat indices, row-major, of the stored frequencies with |q| <= R, where both sideband apertures can be open."""
        cache = self.engine._cache
        cached = self._wave_lens_cache
        if cached is not None and cached[0] is qrow and cached[1] is qcol and cached[2] == unique_cols:
            return cached[3]
        radius = (cache["semiangle_rad"] + 0.5 * max(cache["ang_y_rad"], cache["ang_x_rad"])) / self.engine.wavelength
        radius *= 1 + 8 * np.finfo(np.float32).eps
        inside = qrow[:, None] ** 2 + qcol[None, :unique_cols] ** 2 <= radius**2
        frequencies = cp.flatnonzero(inside).astype(cp.int32)
        self._wave_lens_cache = (qrow, qcol, unique_cols, frequencies)
        return frequencies

    def _wave_order(self) -> cp.ndarray:
        """BF pixels sorted into square cells of a quarter BF radius, so 64 consecutive pixels span a small box."""
        cache = self.engine._cache
        krow, kcol = cache["kx_bf"], cache["ky_bf"]
        cached = self._wave_cells
        if cached is not None and cached[0] is krow and cached[1] is kcol:
            return cached[2]
        radius = (cache["semiangle_rad"] + 0.5 * max(cache["ang_y_rad"], cache["ang_x_rad"])) / self.engine.wavelength
        cell_row = cp.floor(krow / (radius / 4)).astype(cp.int32)
        cell_col = cp.floor(kcol / (radius / 4)).astype(cp.int32)
        order = cp.lexsort(cp.stack((kcol, krow, cell_col, cell_row))).astype(cp.int32)
        self._wave_cells = (krow, kcol, order)
        return order

    def _wave_columns(self, rows: int, cols: int) -> int:
        """Use paired frequencies only beyond the full soft-aperture support.

        All BF contributions and their float64 reduction order are retained.
        Native grids use the complete Fourier plane.
        """
        cache = self.engine._cache
        if rows % 2 or cols % 2 or rows == cache["ny"] or cols == cache["nx"]:
            return cols
        key = (id(cache), rows, cols)
        if self._wave_pairing is None or self._wave_pairing[0] != key:
            radius = 2 * (
                cache["semiangle_rad"]
                + .5 * max(cache["ang_y_rad"], cache["ang_x_rad"])
            ) / self.engine.wavelength
            radius *= 1 + 8 * np.finfo(np.float32).eps
            nyquist = cp.minimum(
                cp.abs(cache["qx_1d"][1]) * (rows // 2),
                cp.abs(cache["qy_1d"][1]) * (cols // 2),
            )
            columns = cols // 2 + 1 if float(nyquist) > radius else cols
            self._wave_pairing = (key, columns)
        return self._wave_pairing[1]

    def fit(
        self,
        C10: float,
        C12: float,
        phi12: float,
        sample_tilt_mrad: tuple[float, float],
        thickness: float,
        band_inv_A: tuple[float, float] = (0.2, 0.9),
        chunk_bytes: int = 1 << 30,
    ) -> float:
        """Least-squares fit of the thick-sample SSB model to G over a spatial-frequency band (larger = better).

        sum over q in the band of |sum_k G conj(gamma)|^2 / sum_k |gamma|^2 (see ``thick_fit_kernel``). This is the objective
        that recovers sample tilt: on simulated BaTiO3 15 nm tilted (3, -4) mrad it finds (3.0, -4.1) and ~0 for the untilted
        control, where the phase-variance loss does not. ``band_inv_A`` excludes the lowest frequencies (dominated by the
        probe-overlap geometry, not the lattice) and frequencies beyond the lattice signal.

        Evaluated on the stored Hermitian half-plane only, and only at the band's q: the probe phase is even (P(-v) = P(v)) and
        the two depth weights swap under q -> -q, so gamma(k, -q) = -conj(gamma(k, q)) and G(k, -q) = conj(G(k, q)); the term at
        -q equals the term at q. Columns with a mirror inside the half-plane count once, all others twice. Same value as the
        full-plane sum (checked to float32 precision) at about a third of the work, with no full-plane copy of G.
        """
        engine = self.engine
        cache = engine._cache
        num_bright_field_pixels, _, _ = engine._bf_grid()
        self._prepare_fit_band(band_inv_A)
        _, _, flat, qx, qy, weight = self._band
        n_band = int(flat.size)
        chunk = max(1, int(chunk_bytes // (n_band * 8 * 3)))
        numerator = cp.zeros((n_band,), dtype=cp.complex64)
        denominator = cp.zeros((n_band,), dtype=cp.float32)
        params = self._kernel_scalars(C10, C12, phi12, sample_tilt_mrad, thickness)
        g_flat = engine.G_qk.reshape(num_bright_field_pixels, -1)
        for start in range(0, num_bright_field_pixels, chunk):
            stop = min(num_bright_field_pixels, start + chunk)
            gathered = g_flat[start:stop][:, flat]                      # (chunk, n_band): only the band's q, half-plane
            kx = cache["kx_bf"][start:stop].reshape(-1, 1)
            ky = cache["ky_bf"][start:stop].reshape(-1, 1)
            projected, weight2 = thick_fit_kernel(gathered, qx, qy, kx, ky, *params)
            numerator += projected.sum(axis=0)
            denominator += weight2.sum(axis=0)
            del gathered, projected, weight2
        keep = denominator > 0
        return float((weight[keep] * cp.abs(numerator[keep]) ** 2 / denominator[keep]).sum())

    def _prepare_fit_band(self, band_inv_A):
        """Cache frequency indices without evaluating or downloading an objective."""
        engine = self.engine
        cache = engine._cache
        num_bright_field_pixels, ny, nx = engine._bf_grid()
        half = engine.gqk_is_half_plane()
        cols = nx // 2 + 1 if half else nx
        key = (ny, nx, cols, float(band_inv_A[0]), float(band_inv_A[1]))
        if self._band_key != key:
            q_magnitude = cp.hypot(cache["qx_1d"].reshape(ny, 1), cache["qy_1d"][:cols].reshape(1, cols))
            inside = (q_magnitude > band_inv_A[0]) & (q_magnitude < band_inv_A[1])
            rows_idx, cols_idx = cp.nonzero(inside)
            if rows_idx.size == 0:
                raise ValueError(
                    f"The model-agreement fit needs spatial frequencies in {band_inv_A[0]:g}–{band_inv_A[1]:g} 1/A, "
                    f"but this scan reaches only {float(q_magnitude.max()):.3g} 1/A. "
                    "Use the phase-variance search (tilt=False, objective='phase_variance'), or acquire a finer scan. "
                    "Reconstruction upsampling does not add measured frequencies."
                )
            if half:
                # columns 0 and nx/2 map onto themselves (their mirror is in the stored half): count once there
                self_mirror = (cols_idx == 0) | ((nx % 2 == 0) & (cols_idx == nx // 2))
                weight = cp.where(self_mirror, 1.0, 2.0).astype(cp.float32)
            else:
                weight = cp.ones(rows_idx.shape, dtype=cp.float32)
            self._band = (rows_idx, cols_idx, (rows_idx * cols + cols_idx).astype(cp.int64),
                          cache["qx_1d"][rows_idx].reshape(1, -1), cache["qy_1d"][cols_idx].reshape(1, -1), weight)
            self._band_key = key
        return self._band

    def fit_batch(self, params, band_inv_A: tuple[float, float] = (0.2, 0.9), *, independent: bool = False) -> np.ndarray:
        """``fit`` for many parameter sets at once, with the fused batch kernel (fast path used by the fit search).

        ``params``: (B, 6) rows of (C10, C12, phi12, tilt_row_mrad, tilt_col_mrad, thickness), engine units. Same value as
        ``fit`` row by row; G is read once per group of ``THICK_FIT_MAX_BATCH`` rows.

        ``independent=True`` preserves the singleton reduction used by each
        independent refinement. Copy each score row into aligned storage before
        summing: a sliced row can otherwise choose a different reduction layout
        even when all its values are identical. Initial batched search trials
        keep their existing summation policy.
        """
        params = np.atleast_2d(np.asarray(params, dtype=np.float64))
        trial = np.stack([params[:, 0], params[:, 1], np.cos(2 * params[:, 2]),
                          np.sin(2 * params[:, 2]), params[:, 5],
                          params[:, 3] * 1e-3, params[:, 4] * 1e-3], axis=1).astype(np.float32)
        return cp.asnumpy(self._fit_encoded(cp.asarray(trial), band_inv_A, independent=independent))

    def fit_resident(self, trial, band_inv_A=(0.2, 0.9)):
        """Return extended scores without copying candidate points or scores to the host."""
        return self._fit_encoded(trial, band_inv_A, extended=True)

    def _fit_encoded(self, trial, band_inv_A, *, independent=False, extended=False):
        """Share the model kernel between diagnostic and resident solver calls."""
        engine = self.engine
        cache = engine._cache
        num_bright_field_pixels, ny, nx = engine._bf_grid()
        self._prepare_fit_band(band_inv_A)
        _, _, flat, qx, qy, weight = self._band
        n_band = int(flat.size)
        output = []
        threads = 256
        blocks = (n_band + threads - 1) // threads
        # enough blocks to fill the GPU: split the pixel loop so blocks x k_blocks is a few thousand
        k_chunk = 256
        k_blocks = (num_bright_field_pixels + k_chunk - 1) // k_chunk
        for start in range(0, len(trial), THICK_FIT_MAX_BATCH):
            rows = trial[start:start + THICK_FIT_MAX_BATCH]
            batch = len(rows)
            # one partial sum per 256-pixel slice, added in slice order by the kernel
            partial_numerator = cp.empty((k_blocks, batch, n_band), dtype=cp.complex64)
            partial_denominator = cp.empty((k_blocks, batch, n_band), dtype=cp.float32)
            numerator = cp.empty((batch, n_band), dtype=cp.complex64)
            denominator = cp.empty((batch, n_band), dtype=cp.float32)
            thick_fit_batch_kernel(
                (blocks, k_blocks), (threads,),
                (engine.G_qk, flat, qx.ravel(), qy.ravel(), cache["kx_bf"], cache["ky_bf"], rows.ravel(),
                 partial_numerator, partial_denominator, cp.zeros(blocks, dtype=cp.uint32), numerator, denominator,
                 np.int32(num_bright_field_pixels), np.int64(engine.G_qk.shape[1] * engine.G_qk.shape[2]), np.int32(n_band), np.int32(batch),
                 np.float32(engine.wavelength), np.float32(cache["semiangle_rad"]), np.float32(cache["ang_y_rad"]),
                 np.float32(cache["ang_x_rad"]), np.float32(engine._factor), np.int32(k_chunk)),
            )
            covered = denominator > 0
            if extended:
                real = numerator.real.astype(cp.float64)
                imag = numerator.imag.astype(cp.float64)
                den = denominator.astype(cp.float64)
                contributions = weight[None] * cp.where(covered, (real * real + imag * imag) / cp.where(covered, den, 1.0), 0.0)
            else:
                contributions = weight[None] * cp.where(covered, cp.abs(numerator) ** 2 / cp.where(covered, denominator, 1.0), 0.0)
            # A refinement previously evaluated each candidate alone. Preserve
            # that reduction layout while sharing the expensive model kernel.
            if independent:
                values = cp.concatenate([
                    contributions[index:index + 1].copy().sum(axis=1)
                    for index in range(batch)
                ])
            else:
                values = contributions.sum(axis=1)
            output.append(values)
        result = cp.concatenate(output)
        if extended:
            high = result.astype(cp.float32)
            middle = (result - high).astype(cp.float32)
            low = (result - high - middle).astype(cp.float32)
            return cp.stack((high, middle, low), axis=-1)
        return result

    def _kernel_scalars(self, C10: float, C12: float, phi12: float, sample_tilt_mrad: tuple[float, float],
                        thickness: float) -> tuple[np.float32, ...]:
        """The float32 scalars every thick-sample kernel takes after its arrays.

        Wavelength, aperture semiangle and detector sampling (rad), the mid-depth C10, C12 and (cos, sin) 2 phi12 terms,
        the chi factor pi / lambda, then the thickness and the sample tilt in rad: one place, so the reconstruction, the
        wave sum and the fit objective always evaluate the same model.
        """
        engine = self.engine
        cache = engine._cache
        return tuple(np.float32(value) for value in (
            engine.wavelength, cache["semiangle_rad"], cache["ang_y_rad"], cache["ang_x_rad"],
            C10, C12, math.cos(2.0 * phi12), math.sin(2.0 * phi12), engine._factor,
            thickness, sample_tilt_mrad[0] * 1e-3, sample_tilt_mrad[1] * 1e-3,
        ))


_pair_kernels = {}


def _pair_kernel(pairs: int, higher_order: bool):
    """The fine-grid wave sum restricted to one class of sideband terms (``_wave_spectrum(pairs=)``)."""
    key = pairs, higher_order
    if key not in _pair_kernels:
        from quantem.gpu.ssb.cuda.kernels import engine as kernels

        correction = kernels._HIGHER_CORRECTION_SOURCE if higher_order else kernels._THICK_CORRECTION_SOURCE
        _pair_kernels[key] = cp.RawKernel(
            f"#define SSB_WAVE\n#define SSB_WAVE_BOUNDS\n#define SSB_WAVE_PAIRS {pairs}\n"
            + ("#define SSB_WAVE_ORDER\n#define SSB_WAVE_QLIST\n" if pairs == 2 else "") + correction
            + kernels._THICK_WAVE_SUM_SOURCE, "thick_wave_sum")
    return _pair_kernels[key]


class WaveWindow:
    """The fine-grid SSB spectrum along one aberration control, for redrawing while it is dragged.

    Serves: ShowPtycho SSB drags of C10, C12 or phi12 with a fitted higher-order
    probe held, at 3x on the full 512 x 512 scan.

    A term with one sideband aperture open has gamma / |gamma| = sign(w)
    exp(-i (chi(k -+ q) - chi(k))), an entire function of the control, so the
    BF sum of those terms is kept as a Chebyshev series (`chebyshev.Series`,
    one exact `_wave_spectrum(pairs=1)` per node). Where both apertures are
    open the normalization |gamma| depends on the optics and has zeros on the
    real axis, so those triple-overlap terms (15 % of the nonzero terms, all at
    |q| <= R, on the test field) are summed exactly every frame
    (``pairs=2``). The two parts add to the full spectrum.
    """

    def __init__(self, thick, aberrations_nm: dict, control: str, tilt, depth_nm: float, factor: int):
        self.thick, self.aberrations, self.control = thick, dict(aberrations_nm), control
        self.tilt = tuple(float(value) for value in (tilt or (0.0, 0.0)))
        self.depth_A = float(depth_nm or 0.0) * ENGINE_PER_NM
        self.qrow, self.qcol = thick._frequencies(factor)
        self.series = chebyshev.series(thick.engine, aberrations_nm, control)
        self.shape = None

    def step(self) -> bool:
        """Add the single-sideband spectrum at the next node; True once the window is complete."""
        if self.series.ready:
            return True
        spectrum = self._spectrum(self.series.nodes[self.series.added], pairs=1)
        self.shape = spectrum.shape
        self.series.add(spectrum)
        return self.series.ready

    def phase(self, value: float) -> cp.ndarray:
        """The float32 phase at control ``value`` (inside the window)."""
        spectrum = self.series.evaluate(value).reshape(self.shape) + self._spectrum(value, pairs=2)
        return self.thick._wave_phase(spectrum, self.qrow.size, self.qcol.size)

    def _spectrum(self, value: float, *, pairs: int) -> cp.ndarray:
        """``_wave_spectrum`` at the control value, with the engine arguments ``SSB.preview`` derives."""
        aberrations = validate_aberrations({**self.aberrations, self.control: value}, higher_order=True)
        coefs = aberrations_to_engine(aberrations)
        magnitudes, angles = higher_order_arrays(aberrations) or (None, None)
        if magnitudes is not None and not np.any(magnitudes[2:]):
            # the two-term kernels then read the same float32 coefficients the 14-term kernel would
            coefs = {"C10": float(magnitudes[0]) * ENGINE_PER_NM, "C12": float(magnitudes[1]) * ENGINE_PER_NM,
                     "phi12": float(angles[1])}
            magnitudes = angles = None
        elif magnitudes is not None:
            magnitudes = magnitudes * np.float32(ENGINE_PER_NM)
        return self.thick._wave_spectrum(
            coefs["C10"], coefs["C12"], coefs["phi12"], self.tilt, self.depth_A, self.qrow, self.qcol,
            chunk_bytes=1 << 30, higher_order_magnitudes=magnitudes, higher_order_angles=angles, pairs=pairs,
        )
