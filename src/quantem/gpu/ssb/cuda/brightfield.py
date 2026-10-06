"""The bright-field pixels the CUDA SSB engine reads: their selection and their spectra.

SSB uses only detector pixels inside the bright-field disk. ``select_bright_field`` picks them on the device from the
mean diffraction pattern, and ``bright_field_spectra`` turns each selected pixel's scan image into its spectrum G(q, k)
(stored as the Hermitian half-plane).
"""

import math

import cupy as cp
import numpy as np


def select_bright_field(
    data: cp.ndarray,
    threshold: float,
    bf_radius: float | None = None,
    bf_center: tuple[float, float] | None = None,
) -> tuple[cp.ndarray, cp.ndarray, tuple[float, float]]:
    """Return the (row, col) detector indices of the bright-field pixels and the disk centre.

    Candidates are the pixels of the mean diffraction pattern above ``threshold`` x max. Without ``bf_radius`` the
    disk is the equal-area disk of the pixels above mean + std, around their unweighted centroid (the rule of
    ``detector.fit_probe``, which the MPS backend calls; here it runs on the device in float32 so the mean pattern never
    leaves the GPU, and CuPy's reductions can round the threshold differently in the last bit, so the two are kept
    apart). With ``bf_radius`` the centre is the intensity-weighted centroid of the candidates; ``bf_center`` (row,
    col, detector pixels) pins it instead, which ``SSB.open`` uses to reproduce the full-detector disk on a detector
    crop, where the mean + std rule would shift. The mean pattern accumulates the raw counts in uint64 (float64 for
    float data), so no float32 copy of the 4D block is made.
    """
    frames = data.reshape(-1, data.shape[-2], data.shape[-1])
    sum_dtype = (
        cp.uint64
        if np.issubdtype(data.dtype, np.integer)
        else cp.float64
    )
    mean_dp = (
        frames.sum(axis=0, dtype=sum_dtype).astype(cp.float32)
        / int(frames.shape[0])
    )
    bf_mask = mean_dp > mean_dp.max() * threshold
    bf_inds = cp.nonzero(bf_mask)
    bf_inds_row = bf_inds[0].astype(cp.int32)
    bf_inds_col = bf_inds[1].astype(cp.int32)
    if len(bf_inds_row) == 0:
        raise ValueError(
            f"No bright-field pixels found with threshold "
            f"{threshold:.2f}. Check that the data "
            f"contains a visible BF disk, or lower the threshold."
        )
    if bf_center is not None:
        if bf_radius is None:
            raise ValueError("bf_center needs bf_radius.")
        center_row, center_col = float(bf_center[0]), float(bf_center[1])
        selected_radius = float(bf_radius)
    elif bf_radius is None:
        probe_mask = mean_dp > mean_dp.mean() + mean_dp.std()
        probe_total = int(probe_mask.sum().get())
        if probe_total > 0:
            probe = probe_mask.astype(cp.float32)
            row_coords = cp.arange(
                mean_dp.shape[0], dtype=cp.float32
            ).reshape(-1, 1)
            col_coords = cp.arange(
                mean_dp.shape[1], dtype=cp.float32
            ).reshape(1, -1)
            center_row = float(
                ((row_coords * probe).sum() / probe_total).get()
            )
            center_col = float(
                ((col_coords * probe).sum() / probe_total).get()
            )
            selected_radius = math.sqrt(probe_total / math.pi)
        else:
            center_row = mean_dp.shape[0] / 2.0
            center_col = mean_dp.shape[1] / 2.0
            selected_radius = min(mean_dp.shape) * 0.25
    else:
        weights = mean_dp[bf_inds_row, bf_inds_col].astype(cp.float32)
        weight_sum = float(weights.sum().get())
        if weight_sum > 0:
            center_row = float(
                (
                    bf_inds_row.astype(cp.float32) * weights
                ).sum().get() / weight_sum
            )
            center_col = float(
                (
                    bf_inds_col.astype(cp.float32) * weights
                ).sum().get() / weight_sum
            )
        else:
            center_row = float(bf_inds_row.mean().get())
            center_col = float(bf_inds_col.mean().get())
        selected_radius = float(bf_radius)
    dist_sq = (bf_inds_row.astype(cp.float32) - center_row) ** 2 + (
        bf_inds_col.astype(cp.float32) - center_col
    ) ** 2
    within = dist_sq <= selected_radius ** 2
    bf_inds_row = bf_inds_row[within]
    bf_inds_col = bf_inds_col[within]
    if len(bf_inds_row) == 0:
        raise ValueError(
            f"No bright-field pixels within bf_radius={selected_radius}. "
            "Increase bf_radius or check detector geometry."
        )
    return bf_inds_row, bf_inds_col, (center_row, center_col)


def bright_field_spectra(
    data: cp.ndarray,
    bf_inds_row: cp.ndarray,
    bf_inds_col: cp.ndarray,
    scan_gpts: tuple[int, ...],
    det_gpts: tuple[int, ...],
) -> tuple[cp.ndarray, complex]:
    """Return G(q, k), the 2D spectrum of each selected pixel's scan image, and its mean DC value.

    The scan images are real, so only the Hermitian half-plane (scan columns 0..N/2) is stored; the kernels mirror and
    conjugate the missing columns when a full Fourier coordinate is requested. The work runs in chunks of bright-field
    pixels sized so the complex64 staging buffer stays near 2 GB: unchunked, a 512 x 512 acquisition peaked at ~57 GB
    (raw counts, the complex stack and G_qk, 19 GB each). Each chunk is gathered and transposed in its native dtype
    and cast to complex64 only then.
    """
    num_bf = len(bf_inds_row)
    scan_row, scan_col = int(scan_gpts[0]), int(scan_gpts[1])
    det_row, det_col = int(det_gpts[0]), int(det_gpts[1])
    stored_col = scan_col // 2 + 1
    # (N_scan, det_row, det_col) view of the raw counts
    flat_data = data.reshape(-1, det_row, det_col)
    G_qk = cp.empty((num_bf, scan_row, stored_col), dtype=cp.complex64)
    bytes_per_bf = scan_row * scan_col * 8  # complex64
    target_chunk_bytes = 2 * 1024 ** 3
    chunk_bf = max(1, min(num_bf, target_chunk_bytes // bytes_per_bf))
    for bf_start in range(0, num_bf, chunk_bf):
        bf_end = min(bf_start + chunk_bf, num_bf)
        row_chunk = bf_inds_row[bf_start:bf_end]
        col_chunk = bf_inds_col[bf_start:bf_end]
        vbf_flat = flat_data[:, row_chunk, col_chunk]
        k = bf_end - bf_start
        vbf_int = cp.ascontiguousarray(vbf_flat.T.reshape(k, scan_row, scan_col))
        del vbf_flat
        vbf_stack = vbf_int.astype(cp.complex64)
        del vbf_int
        fft_chunk = cp.fft.fft2(vbf_stack)
        half_chunk = cp.ascontiguousarray(fft_chunk[:, :, :scan_col // 2 + 1])
        G_qk[bf_start:bf_end] = half_chunk
        del half_chunk
        del fft_chunk
        del vbf_stack
    dc_value = complex(G_qk[:, 0, 0].mean().get())
    cp.get_default_memory_pool().free_all_blocks()
    return G_qk, dc_value
