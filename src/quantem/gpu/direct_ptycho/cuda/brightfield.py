"""The bright-field pixels the CUDA direct ptychography engine reads: their selection and their spectra.

Direct ptychography uses only detector pixels inside the bright-field disk. ``select_bright_field`` picks them on the device from the
mean diffraction pattern, and ``bright_field_spectra`` turns each selected pixel's scan image into its spectrum G(q, k)
(stored as the Hermitian half-plane).
"""

import math

import cupy as cp
from cupy.cuda import cufft
import cupyx.scipy.fftpack
import numpy as np


def select_bright_field(
    data: cp.ndarray,
    threshold: float,
    bright_field_radius_px: float | None = None,
    bright_field_center_px: tuple[float, float] | None = None,
) -> tuple[cp.ndarray, cp.ndarray, tuple[float, float], float]:
    """Return the detector indices, disk centre and exact selection radius.

    Candidates are the pixels of the mean diffraction pattern above ``threshold`` x max. Without ``bright_field_radius_px`` the
    disk is the equal-area disk of the pixels above mean + std, around their unweighted centroid (the rule of
    ``detector.bright_field_disk``, which the MPS backend evaluates in float32; here it runs on the device in float32 so the mean pattern never
    leaves the GPU, and CuPy's reductions can round the threshold differently in the last bit, so the two are kept
    apart). With ``bright_field_radius_px`` the centre is the intensity-weighted centroid of the candidates; ``bright_field_center_px`` (row,
    col, detector pixels) pins it instead, which ``DirectPtycho.open`` uses to reproduce the full-detector disk on a detector
    crop, where the mean + std rule would shift. The mean pattern accumulates the raw counts in uint64 (float64 for
    float data), so no float32 copy of the 4D block is made.
    """
    frames = data.reshape(-1, data.shape[-2], data.shape[-1])
    sum_dtype = (
        cp.uint64
        if np.issubdtype(data.dtype, np.integer)
        else cp.float64
    )
    mean_pattern = (
        frames.sum(axis=0, dtype=sum_dtype).astype(cp.float32)
        / int(frames.shape[0])
    )
    bf_mask = mean_pattern > mean_pattern.max() * threshold
    bf_inds = cp.nonzero(bf_mask)
    bf_inds_row = bf_inds[0].astype(cp.int32)
    bf_inds_col = bf_inds[1].astype(cp.int32)
    if len(bf_inds_row) == 0:
        raise ValueError(
            f"No bright-field pixels found with threshold "
            f"{threshold:.2f}. Check that the data "
            f"contains a visible BF disk, or lower the threshold."
        )
    if bright_field_center_px is not None:
        if bright_field_radius_px is None:
            raise ValueError("bright_field_center_px needs bright_field_radius_px.")
        center_row, center_col = float(bright_field_center_px[0]), float(bright_field_center_px[1])
        selected_radius = float(bright_field_radius_px)
    elif bright_field_radius_px is None:
        probe_mask = mean_pattern > mean_pattern.mean() + mean_pattern.std()
        probe_total = int(probe_mask.sum().get())
        if probe_total > 0:
            probe = probe_mask.astype(cp.float32)
            row_coords = cp.arange(
                mean_pattern.shape[0], dtype=cp.float32
            ).reshape(-1, 1)
            col_coords = cp.arange(
                mean_pattern.shape[1], dtype=cp.float32
            ).reshape(1, -1)
            center_row = float(
                ((row_coords * probe).sum() / probe_total).get()
            )
            center_col = float(
                ((col_coords * probe).sum() / probe_total).get()
            )
            selected_radius = math.sqrt(probe_total / math.pi)
        else:
            center_row = mean_pattern.shape[0] / 2.0
            center_col = mean_pattern.shape[1] / 2.0
            selected_radius = min(mean_pattern.shape) * 0.25
    else:
        weights = mean_pattern[bf_inds_row, bf_inds_col].astype(cp.float32)
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
        selected_radius = float(bright_field_radius_px)
    dist_sq = (bf_inds_row.astype(cp.float32) - center_row) ** 2 + (
        bf_inds_col.astype(cp.float32) - center_col
    ) ** 2
    within = dist_sq <= selected_radius ** 2
    bf_inds_row = bf_inds_row[within]
    bf_inds_col = bf_inds_col[within]
    if len(bf_inds_row) == 0:
        raise ValueError(
            f"No bright-field pixels within bright_field_radius_px={selected_radius}. "
            "Increase bright_field_radius_px or check detector geometry."
        )
    return bf_inds_row, bf_inds_col, (center_row, center_col), selected_radius


def bright_field_spectra(
    data: cp.ndarray,
    bf_inds_row: cp.ndarray,
    bf_inds_col: cp.ndarray,
    scan_gpts: tuple[int, ...],
    det_gpts: tuple[int, ...],
    out: cp.ndarray | None = None,
) -> tuple[cp.ndarray, complex]:
    """Return G(q, k), the 2D spectrum of each selected pixel's scan image, and its mean DC value.

    The scan images are real, so only the Hermitian half-plane (scan columns 0..N/2) is stored; the kernels mirror and
    conjugate the missing columns when a full Fourier coordinate is requested. The work runs in chunks of bright-field
    pixels sized so the complex64 staging buffer stays at or below 256 MiB: unchunked, a 512 x 512 acquisition peaked at ~57 GB
    (raw counts, the complex stack and G_qk, 19 GB each). Each chunk's scan images are written straight into the
    staging buffer as float32 and transformed real-to-complex into their place in G: the half-plane is all that is
    computed, and nothing is copied afterwards. A complex transform of the same images, cropped, moved twice the
    bytes and agrees to float32 rounding (2 parts in 10 million of a typical value).

    ``out``: a G(q, k) of this geometry to write into, for new counts of the same scan and detector (a live preview
    gives one open solver each look's counts). No result is allocated and the memory pool is left as it is, so the
    next look's buffers come from it; the spectra are the same bits.
    """
    num_bright_field_pixels = len(bf_inds_row)
    scan_row, scan_col = int(scan_gpts[0]), int(scan_gpts[1])
    det_row, det_col = int(det_gpts[0]), int(det_gpts[1])
    stored_col = scan_col // 2 + 1
    scans = scan_row * scan_col
    # (N_scan, det_row, det_col) view of the raw counts
    flat_data = data.reshape(-1, det_row, det_col)
    pixels = bf_inds_row.astype(cp.int32) * det_col + bf_inds_col.astype(cp.int32)
    shape = (num_bright_field_pixels, scan_row, stored_col)
    if out is None:
        G_qk = cp.empty(shape, dtype=cp.complex64)
    elif out.shape != shape or out.dtype != cp.complex64:
        raise ValueError(f"out is {out.dtype} {out.shape}; these counts give complex64 {shape}.")
    else:
        G_qk = out
    bytes_per_bf = scan_row * scan_col * 4  # float32
    target_chunk_bytes = 256 * 1024 ** 2
    chunk_bf = max(1, min(num_bright_field_pixels, target_chunk_bytes // bytes_per_bf))
    staging = cp.empty((chunk_bf, scan_row, scan_col), dtype=cp.float32)
    groups = -(-scans // _POSITIONS_PER_THREAD)
    plans = {}
    for bf_start in range(0, num_bright_field_pixels, chunk_bf):
        bf_end = min(bf_start + chunk_bf, num_bright_field_pixels)
        count = bf_end - bf_start
        scan_images = staging[:count]
        _scan_images(flat_data, pixels[bf_start:bf_end], count, scans, det_row * det_col, scan_images,
                     size=groups * count)
        if count not in plans:
            plans[count] = cupyx.scipy.fftpack.get_fft_plan(scan_images, axes=(1, 2), value_type="R2C")
        # real to complex, written where it is kept: the stored half-plane is all the transform computes
        plans[count].fft(scan_images, G_qk[bf_start:bf_end], cufft.CUFFT_FORWARD)
    dc_value = complex(G_qk[:, 0, 0].mean().get())
    del staging, scan_images, plans
    if out is None:
        cp.get_default_memory_pool().free_all_blocks()
    return G_qk, dc_value


# Counts (scan position, detector pixel) become real scan images (pixel, scan position): a transpose, so either the
# reads or the writes are apart. One thread takes a few neighbouring scan positions of one pixel: the threads beside
# it take the next pixels, so they read the same stretch of each pattern together, and each writes one short run of
# its own image. Measured on one workstation GPU, 9298 pixels of a 128 x 128 scan (0.9 GB moved, 0.64 ms at the
# 1.4 TB/s a plain copy moves): 1.1 ms; one thread a scan position walking every pixel took 2.4 ms, one thread 32
# positions 3.8 ms. The values are the same in every layout.
_POSITIONS_PER_THREAD = 8
_scan_images = cp.ElementwiseKernel(
    "raw T counts, raw int32 pixels, int64 count, int64 scans, int64 detector_pixels",
    "raw float32 images",
    """
    long long group = i / count, k = i - group * count;
    long long pixel = pixels[k];
    for (int j = 0; j < %d; ++j) {
        long long position = group * %d + j;
        if (position < scans) {
            images[k * scans + position] = (float)counts[position * detector_pixels + pixel];
        }
    }
    """ % (_POSITIONS_PER_THREAD, _POSITIONS_PER_THREAD),
    "direct_bright_field_scan_images",
)
