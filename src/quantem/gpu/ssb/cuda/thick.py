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
    thick_fit_batch_kernel,
    thick_fit_kernel,
    thick_wave_sum_kernel,
)


class ThickSample:
    """The thick-sample model on one ``SSBEngine``'s prepared ``G_qk`` and rotation geometry.

    The engine owns the evidence and the current rotation; this object owns what only the thick model needs: the
    spatial-frequency band of the fit objective, its kernel-ready copies, and the float64 wave-sum scratch buffer, each
    rebuilt only when the band, the grid or the rotation cache changes.
    """

    def __init__(self, engine) -> None:
        self.engine = engine
        self._band_key = None
        self._band = None
        self._batch_key = None
        self._batch_arrays = None
        self._wave_sum_buffer = None

    def reconstruct(
        self,
        C10: float,
        C12: float,
        phi12: float,
        tilt_mrad: tuple[float, float],
        thickness: float,
        compute_loss: bool = True,
        chunk_bytes: int = 1 << 30,
        upsampling_factor: int = 1,
        phase_estimator: str = "mean_phase",
    ) -> tuple[cp.ndarray, float | None]:
        """Mean phase and variance loss for a thick, tilted crystal (see ``thick_correct_kernel``).

        Same outputs and definitions as ``SSBEngine.reconstruct_with_loss`` (mean over bright-field pixels of the
        per-pixel phase; loss = mean over the image of the per-pixel phase variance), with each pixel's SSB correction
        averaged over the sample depth. ``tilt_mrad`` = (row, col) in the scan frame, ``thickness`` in the C10 unit; thickness
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
        c = engine._cache
        num_bf, ny, nx = int(c["num_bf"]), int(c["ny"]), int(c["nx"])
        native_ny, native_nx = ny, nx
        if type(upsampling_factor) is not int or upsampling_factor not in (1, 2, 3, 4, 8):
            raise ValueError("upsampling_factor must be 1, 2, 3, 4, or 8.")
        if upsampling_factor != 1:
            ny, nx = ny * upsampling_factor, nx * upsampling_factor
            # Same field of view: Fourier spacing is unchanged. Tile measured
            # scan-frequency aliases, then evaluate the kernel at the new q.
            qx = (cp.fft.fftfreq(ny) * ny * c["qx_1d"][1]).astype(cp.float32).reshape(1, ny, 1)
            qy = (cp.fft.fftfreq(nx) * nx * c["qy_1d"][1]).astype(cp.float32).reshape(1, 1, nx)
        else:
            qx = c["qx_1d"].reshape(1, ny, 1); qy = c["qy_1d"].reshape(1, 1, nx)
        if average_wave:
            return self._reconstruct_wave(
                C10, C12, phi12, tilt_mrad, thickness, qx.ravel(), qy.ravel(),
                compute_loss=compute_loss, chunk_bytes=chunk_bytes,
            )
        half = engine.gqk_is_half_plane()
        # other half of the plane for a real bright-field image: G(-q) = conj(G(q))
        neg_rows = cp.asarray((-np.arange(native_ny)) % native_ny)
        neg_cols = cp.asarray((native_nx - np.arange(native_nx // 2 + 1, native_nx)) % native_nx)
        chunk = max(1, int(chunk_bytes // (ny * nx * 8 * 3)))
        phase_sum = cp.zeros((ny, nx), dtype=cp.float32)
        phase_sumsq = cp.zeros((ny, nx), dtype=cp.float32)
        params = (
            cp.float32(engine.wavelength), cp.float32(c["semiangle_rad"]), cp.float32(c["ang_y_rad"]), cp.float32(c["ang_x_rad"]),
            cp.float32(C10), cp.float32(C12), cp.float32(math.cos(2.0 * phi12)), cp.float32(math.sin(2.0 * phi12)),
            cp.float32(engine._factor), cp.float32(thickness), cp.float32(tilt_mrad[0] * 1e-3), cp.float32(tilt_mrad[1] * 1e-3),
        )
        for start in range(0, num_bf, chunk):
            stop = min(num_bf, start + chunk)
            if half:
                source = engine.G_qk[start:stop]
                full = cp.empty((stop - start, native_ny, native_nx), dtype=cp.complex64)
                full[:, :, : native_nx // 2 + 1] = source
                full[:, :, native_nx // 2 + 1:] = cp.conj(source[:, neg_rows][:, :, neg_cols])
            else:
                full = engine.G_qk[start:stop]
            if upsampling_factor != 1:
                full = cp.tile(full, (1, upsampling_factor, upsampling_factor))
            kx = c["kx_bf"][start:stop].reshape(-1, 1, 1); ky = c["ky_bf"][start:stop].reshape(-1, 1, 1)
            corrected = thick_correct_kernel(full, qx, qy, kx, ky, *params)
            corrected[:, 0, 0] = engine._dc_value_host
            angles = cp.angle(cp.fft.ifft2(corrected, axes=(1, 2)))
            phase_sum += angles.sum(axis=0)
            if compute_loss:
                phase_sumsq += (angles * angles).sum(axis=0)
            del full, corrected, angles
        mean = phase_sum / float(num_bf)
        if not compute_loss:
            return mean, None
        loss = float(cp.mean(phase_sumsq / float(num_bf) - mean * mean))
        return mean, loss

    def _reconstruct_wave(
        self,
        C10: float,
        C12: float,
        phi12: float,
        tilt_mrad: tuple[float, float],
        thickness: float,
        qrow: cp.ndarray,
        qcol: cp.ndarray,
        *,
        compute_loss: bool,
        chunk_bytes: int,
    ) -> tuple[cp.ndarray, float | None]:
        """Accumulate float32 corrections in float64 before a single inverse FFT."""
        engine = self.engine
        c = engine._cache
        num_bf = int(c["num_bf"])
        rows, cols = qrow.size, qcol.size
        group_size = 32
        groups = max(1, min(16, math.ceil(num_bf / group_size),
                            chunk_bytes // (rows * cols * 16)))
        shape = (groups, rows, cols)
        if self._wave_sum_buffer is None or self._wave_sum_buffer.shape != shape:
            self._wave_sum_buffer = cp.empty(shape, dtype=cp.complex128)
        partial = self._wave_sum_buffer
        spectrum = cp.zeros((rows, cols), dtype=cp.complex128)
        params = tuple(np.float32(v) for v in (
            engine.wavelength, c["semiangle_rad"], c["ang_y_rad"], c["ang_x_rad"],
            C10, C12, math.cos(2 * phi12), math.sin(2 * phi12), engine._factor,
            thickness, tilt_mrad[0] * 1e-3, tilt_mrad[1] * 1e-3,
        ))
        for start in range(0, num_bf, group_size * groups):
            active = min(groups, math.ceil((num_bf - start) / group_size))
            thick_wave_sum_kernel(
                (math.ceil(rows * cols / 128), active), (128,),
                (engine.G_qk, qrow, qcol, c["kx_bf"], c["ky_bf"], partial,
                 np.int32(num_bf), np.int32(c["ny"]), np.int32(c["nx"]),
                 np.int32(engine.G_qk.shape[-1]), np.int32(rows), np.int32(cols),
                 np.int32(start), np.int32(group_size), np.complex64(engine._dc_value_host),
                 *params),
            )
            spectrum += partial[:active].sum(axis=0, dtype=cp.complex128)
        phase = cp.angle(cp.fft.ifft2(spectrum / num_bf)).astype(cp.float32)
        loss = None
        if compute_loss:
            _, loss = self.reconstruct(
                C10, C12, phi12, tilt_mrad, thickness,
                compute_loss=True, chunk_bytes=chunk_bytes,
            )
        return phase, loss

    def fit(
        self,
        C10: float,
        C12: float,
        phi12: float,
        tilt_mrad: tuple[float, float],
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
        c = engine._cache
        num_bf, ny, nx = int(c["num_bf"]), int(c["ny"]), int(c["nx"])
        half = engine.gqk_is_half_plane()
        cols = nx // 2 + 1 if half else nx
        key = (ny, nx, cols, float(band_inv_A[0]), float(band_inv_A[1]))
        if self._band_key != key:
            qr = c["qx_1d"].reshape(ny, 1); qc = c["qy_1d"][:cols].reshape(1, cols)
            q = cp.hypot(qr, qc)
            inside = (q > band_inv_A[0]) & (q < band_inv_A[1])
            rows_idx, cols_idx = cp.nonzero(inside)
            if half:
                # columns 0 and nx/2 map onto themselves (their mirror is in the stored half): count once there
                self_mirror = (cols_idx == 0) | ((nx % 2 == 0) & (cols_idx == nx // 2))
                weight = cp.where(self_mirror, 1.0, 2.0).astype(cp.float32)
            else:
                weight = cp.ones(rows_idx.shape, dtype=cp.float32)
            self._band = (rows_idx, cols_idx, (rows_idx * cols + cols_idx).astype(cp.int64),
                                c["qx_1d"][rows_idx].reshape(1, -1), c["qy_1d"][cols_idx].reshape(1, -1), weight)
            self._band_key = key
        _, _, flat, qx, qy, weight = self._band
        n_band = int(flat.size)
        chunk = max(1, int(chunk_bytes // (n_band * 8 * 3)))
        numerator = cp.zeros((n_band,), dtype=cp.complex64); denominator = cp.zeros((n_band,), dtype=cp.float32)
        params = (
            cp.float32(engine.wavelength), cp.float32(c["semiangle_rad"]), cp.float32(c["ang_y_rad"]), cp.float32(c["ang_x_rad"]),
            cp.float32(C10), cp.float32(C12), cp.float32(math.cos(2.0 * phi12)), cp.float32(math.sin(2.0 * phi12)),
            cp.float32(engine._factor), cp.float32(thickness), cp.float32(tilt_mrad[0] * 1e-3), cp.float32(tilt_mrad[1] * 1e-3),
        )
        g_flat = engine.G_qk.reshape(num_bf, -1)
        for start in range(0, num_bf, chunk):
            stop = min(num_bf, start + chunk)
            gathered = g_flat[start:stop][:, flat]                      # (chunk, n_band): only the band's q, half-plane
            kx = c["kx_bf"][start:stop].reshape(-1, 1); ky = c["ky_bf"][start:stop].reshape(-1, 1)
            projected, weight2 = thick_fit_kernel(gathered, qx, qy, kx, ky, *params)
            numerator += projected.sum(axis=0); denominator += weight2.sum(axis=0)
            del gathered, projected, weight2
        keep = denominator > 0
        return float((weight[keep] * cp.abs(numerator[keep]) ** 2 / denominator[keep]).sum())

    def fit_batch(self, params, band_inv_A: tuple[float, float] = (0.2, 0.9)) -> np.ndarray:
        """``fit`` for many parameter sets at once, with the fused batch kernel (fast path used by the fit search).

        ``params``: (B, 6) rows of (C10, C12, phi12, tilt_row_mrad, tilt_col_mrad, thickness), engine units. Same value as
        ``fit`` row by row; G is read once per group of ``THICK_FIT_MAX_BATCH`` rows.
        """
        engine = self.engine
        params = np.atleast_2d(np.asarray(params, dtype=np.float64))
        c = engine._cache
        num_bf, ny, nx = int(c["num_bf"]), int(c["ny"]), int(c["nx"])
        cols = nx // 2 + 1 if engine.gqk_is_half_plane() else nx
        if self._band_key != (ny, nx, cols, float(band_inv_A[0]), float(band_inv_A[1])):
            self.fit(0.0, 0.0, 0.0, (0.0, 0.0), 0.0, band_inv_A)      # builds the band index for this band
        _, _, flat, qx, qy, weight = self._band
        n_band = int(flat.size)
        if engine.G_qk.dtype != cp.complex64 or not engine.G_qk.flags.c_contiguous:
            raise TypeError("ThickSample.fit_batch needs a C-contiguous complex64 G_qk")
        # kernel-ready copies of the band and pixel coordinates, built once per band / rotation (not per call)
        cache_key = (self._band_key, id(c))
        if self._batch_key != cache_key:
            self._batch_arrays = (
                cp.ascontiguousarray(qx.ravel().astype(cp.float32)), cp.ascontiguousarray(qy.ravel().astype(cp.float32)),
                cp.ascontiguousarray(c["kx_bf"].astype(cp.float32)), cp.ascontiguousarray(c["ky_bf"].astype(cp.float32)),
                cp.ascontiguousarray(flat.astype(cp.int64)),
            )
            self._batch_key = cache_key
        qx_b, qy_b, kx, ky, flat64 = self._batch_arrays
        out = np.empty(len(params))
        threads = 256
        blocks = (n_band + threads - 1) // threads
        # enough blocks to fill the GPU: split the pixel loop so blocks x k_blocks is a few thousand
        k_chunk = 256
        k_blocks = (num_bf + k_chunk - 1) // k_chunk
        for start in range(0, len(params), THICK_FIT_MAX_BATCH):
            rows = params[start:start + THICK_FIT_MAX_BATCH]
            b = len(rows)
            trial = np.stack([rows[:, 0], rows[:, 1], np.cos(2 * rows[:, 2]), np.sin(2 * rows[:, 2]), rows[:, 5],
                              rows[:, 3] * 1e-3, rows[:, 4] * 1e-3], axis=1).astype(np.float32)
            numer = cp.zeros((b, n_band), dtype=cp.complex64); denom = cp.zeros((b, n_band), dtype=cp.float32)
            thick_fit_batch_kernel(
                (blocks, k_blocks), (threads,),
                (engine.G_qk, flat64, qx_b, qy_b, kx, ky, cp.asarray(trial.ravel()), numer, denom,
                 np.int32(num_bf), np.int64(engine.G_qk.shape[1] * engine.G_qk.shape[2]), np.int32(n_band), np.int32(b),
                 np.float32(engine.wavelength), np.float32(c["semiangle_rad"]), np.float32(c["ang_y_rad"]), np.float32(c["ang_x_rad"]),
                 np.float32(engine._factor), np.int32(k_chunk)),
            )
            ok = denom > 0
            values = (weight[None] * cp.where(ok, cp.abs(numer) ** 2 / cp.where(ok, denom, 1.0), 0.0)).sum(axis=1)
            out[start:start + b] = cp.asnumpy(values)
        return out
