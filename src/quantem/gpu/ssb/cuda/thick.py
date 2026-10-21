"""Thick, tilted-sample SSB on CUDA: depth-averaged correction and the least-squares fit objective.

Standard SSB treats the sample as one plane at the probe defocus. A crystal of thickness t tilted by theta sees a range of
defoci and shifts through its depth; averaging the SSB transfer terms over that depth gives each its own sinc weight (see
``kernels/engine.py``). ``ThickSample`` evaluates that model on one engine's prepared bright-field spectra ``G_qk``:
phase images for previews and reconstruction, and the agreement that ``find_aberrations(tilt=True)`` maximises.
"""

import math

import cupy as cp
import numpy as np

from quantem.gpu.ssb.cuda.kernels.engine import (
    THICK_FIT_MAX_BATCH,
    thick_correct_kernel,
    thick_correct_full_kernel,
    thick_fit_batch_kernel,
    thick_fit_kernel,
    thick_wave_sum_kernel,
    thick_wave_sum_full_kernel,
    thick_wave_sum_bounded_kernel,
    thick_wave_sum_bounded_full_kernel,
)
from quantem.gpu.ssb.higher_order import pack_coefficients


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
        if upsampling_factor != 1:
            ny, nx = ny * upsampling_factor, nx * upsampling_factor
            # Same field of view: Fourier spacing is unchanged. Tile measured
            # scan-frequency aliases, then evaluate the kernel at the new q.
            qx = (cp.fft.fftfreq(ny) * ny * cache["qx_1d"][1]).astype(cp.float32).reshape(1, ny, 1)
            qy = (cp.fft.fftfreq(nx) * nx * cache["qy_1d"][1]).astype(cp.float32).reshape(1, 1, nx)
        else:
            qx = cache["qx_1d"].reshape(1, ny, 1)
            qy = cache["qy_1d"].reshape(1, 1, nx)
        if average_wave:
            return self._reconstruct_wave(
                C10, C12, phi12, sample_tilt_mrad, thickness, qx.ravel(), qy.ravel(),
                compute_loss=compute_loss, chunk_bytes=chunk_bytes,
                higher_order_magnitudes=higher_order_magnitudes,
                higher_order_angles=higher_order_angles,
            )
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
        engine = self.engine
        cache = engine._cache
        num_bright_field_pixels = int(cache["num_bright_field_pixels"])
        rows, cols = qrow.size, qcol.size
        unique_cols = (
            self._wave_columns(rows, cols)
            if higher_order_magnitudes is None else cols
        )
        # Fine grids otherwise write and reduce a large partial plane for
        # every 32 BF pixels. Larger groups retain all float32 corrections
        # and float64 accumulation while reducing that intermediate traffic.
        fine_grid = rows > cache["ny"] and cols > cache["nx"]
        group_size = 128 if fine_grid else 32
        groups = max(1, min(16, math.ceil(num_bright_field_pixels / group_size),
                            chunk_bytes // (rows * cols * 16)))
        shape = (groups, rows, unique_cols)
        if self._wave_sum_buffer is None or self._wave_sum_buffer.shape != shape:
            self._wave_sum_buffer = cp.empty(shape, dtype=cp.complex128)
        partial = self._wave_sum_buffer
        # Native grids do not benefit from group culling; that kernel does not
        # read the bounds argument and keeps its existing execution path.
        bounds = self._wave_group_bounds(group_size) if fine_grid else cache["kx_bf"]
        spectrum = cp.zeros((rows, unique_cols), dtype=cp.complex128)
        params = self._kernel_scalars(C10, C12, phi12, sample_tilt_mrad, thickness)
        kernel = thick_wave_sum_bounded_kernel if fine_grid else thick_wave_sum_kernel
        if higher_order_magnitudes is not None:
            kernel = thick_wave_sum_bounded_full_kernel if fine_grid else thick_wave_sum_full_kernel
            params = (*params, cp.asarray(pack_coefficients(higher_order_magnitudes, higher_order_angles)))
        for start in range(0, num_bright_field_pixels, group_size * groups):
            active = min(groups, math.ceil((num_bright_field_pixels - start) / group_size))
            kernel(
                (math.ceil(rows * unique_cols / 128), active), (128,),
                (engine.G_qk, qrow, qcol, cache["kx_bf"], cache["ky_bf"], partial, bounds,
                 np.int32(num_bright_field_pixels), np.int32(cache["ny"]), np.int32(cache["nx"]),
                 np.int32(engine.G_qk.shape[-1]), np.int32(rows), np.int32(cols), np.int32(unique_cols),
                 np.int32(start), np.int32(group_size), np.complex64(engine._dc_value_host),
                 *params),
            )
            spectrum += partial[:active].sum(axis=0, dtype=cp.complex128)
        if unique_cols != cols:
            # Even C10/C12 optics give F(-q) = -conj(F(q)), including the
            # paired depth/tilt weights. The guard excludes Nyquist support.
            negative = -cp.conj(spectrum[(-cp.arange(rows)) % rows, 1:cols // 2][:, ::-1])
            spectrum = cp.concatenate((spectrum, negative), axis=1)
        # the BF sum and the inverse FFT run in complex128; the phase is returned float32 like every other path
        phase = cp.angle(cp.fft.ifft2(spectrum / num_bright_field_pixels)).astype(cp.float32)
        loss = None
        if compute_loss:
            _, loss = self.reconstruct(
                C10, C12, phi12, sample_tilt_mrad, thickness,
                compute_loss=True, chunk_bytes=chunk_bytes,
                higher_order_magnitudes=higher_order_magnitudes,
                higher_order_angles=higher_order_angles,
            )
        return phase, loss

    def _wave_group_bounds(self, group_size: int) -> cp.ndarray:
        """Cache exact coordinate bounds for each complete BF group on GPU."""
        cache = self.engine._cache
        krow, kcol = cache["kx_bf"], cache["ky_bf"]
        cached = self._wave_bounds
        if (cached is not None and cached[0] == group_size
                and cached[1] is krow and cached[2] is kcol):
            return cached[3]
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
        self._wave_bounds = (group_size, krow, kcol, bounds)
        return bounds

    def _wave_columns(self, rows: int, cols: int) -> int:
        """Use paired frequencies only beyond the full soft-aperture support.

        All BF contributions and their float64 reduction order are retained.
        Native grids and higher-order probes use the complete Fourier plane.
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
