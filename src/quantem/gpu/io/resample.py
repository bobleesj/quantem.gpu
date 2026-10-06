"""Resample a decoded CUDA scan crop onto a shared specimen grid.

Drift-aware series (denova) decode one scan crop per acquisition and need every
crop sampled at the same specimen positions, which fall between the scan
positions of each source. Bilinear interpolation over the scan axes does that
on the GPU, reading the crop in its native dtype so no float copy of the whole
crop is made.
"""

import numpy as np

from quantem.gpu.device.cuda_runtime import cp, cuda_module


def resample_scan_crop(
    data,
    *,
    source_scan_region: tuple[int, int, int, int],
    target_scan_region: tuple[int, int, int, int],
    scan_shift_row_col,
    output_dtype: type | np.dtype = np.float32,
):
    """Resample a decoded CUDA scan crop into specimen coordinates.

    Parameters
    ----------
    data
        Decoded CuPy array with shape
        ``(scan_row, scan_col, detector_row, detector_col)``. This function
        does not read HDF5; use :func:`load` for loading.
    source_scan_region
        Source scan bounds of ``data`` in full-frame scan coordinates, as
        ``(row_start, row_stop, col_start, col_stop)``.
    target_scan_region
        Requested output specimen-coordinate bounds, also in ``(row, col)``
        order.
    scan_shift_row_col
        Per-frame scan shift ``(row_shift, col_shift)`` that maps specimen
        coordinates into the source frame.
    output_dtype
        Output dtype. Only ``np.float32`` is currently supported because
        subpixel alignment creates fractional detector counts.

    Returns
    -------
    cupy.ndarray
        Resampled float32 array with shape
        ``(target_rows, aligned_cols, detector_row, detector_col)``.
    """
    if cp is None:  # pragma: no cover - CUDA-only helper
        raise RuntimeError("scan-crop resampling requires CuPy/CUDA")
    if not isinstance(data, cp.ndarray):
        raise TypeError("scan-crop resampling requires a CuPy input array")
    if data.ndim != 4:
        raise ValueError(
            "scan-crop resampling expects decoded data with shape "
            "(scan_row, scan_col, detector_row, detector_col)"
        )
    if np.dtype(output_dtype) != np.dtype(np.float32):
        raise TypeError(
            "output_dtype currently supports only np.float32 because "
            "subpixel scan resampling creates fractional detector counts."
        )
    row_start, row_stop, col_start, col_stop = (
        int(value) for value in target_scan_region
    )
    out_rows = int(row_stop - row_start)
    out_cols = int(col_stop - col_start)
    src_rows, src_cols, det_rows, det_cols = (int(value) for value in data.shape)
    itemsize = int(data.dtype.itemsize)
    src_stride_row, src_stride_col, src_stride_det_row, src_stride_det_col = (
        int(stride // itemsize) for stride in data.strides
    )
    shifts = np.asarray(scan_shift_row_col, dtype=np.float32)
    if shifts.shape != (2,):
        raise TypeError("one decoded crop expects one (row_shift, col_shift) pair")
    out = cp.empty((out_rows, out_cols, det_rows, det_cols), dtype=cp.float32)
    n = int(out.size)
    if n == 0:
        return out
    kernel = _resample_scan_crop_kernel(np.dtype(data.dtype))
    threads = 256
    blocks = (n + threads - 1) // threads
    source_row_start, _source_row_stop, source_col_start, _source_col_stop = (
        int(value) for value in source_scan_region
    )
    kernel(
        (blocks,),
        (threads,),
        (
            data,
            out,
            np.int64(n),
            np.int32(src_rows),
            np.int32(src_cols),
            np.int32(det_rows),
            np.int32(det_cols),
            np.int32(out_rows),
            np.int32(out_cols),
            np.int64(src_stride_row),
            np.int64(src_stride_col),
            np.int64(src_stride_det_row),
            np.int64(src_stride_det_col),
            np.float32(source_row_start),
            np.float32(source_col_start),
            np.float32(row_start),
            np.float32(col_start),
            np.float32(shifts[0]),
            np.float32(shifts[1]),
        ),
    )
    return out


def _resample_scan_crop_kernel(dtype: np.dtype):
    """Return the bilinear scan resampler for ``dtype``, compiled once per CUDA context.

    Drift-aware series need each source crop sampled at fractional scan
    positions of one shared specimen grid; the kernel reads the decoded crop
    in its native dtype so no float copy of the full crop is made.
    """
    ctype_by_dtype = {
        np.dtype(np.uint8): "unsigned char",
        np.dtype(np.uint16): "unsigned short",
        np.dtype(np.uint32): "unsigned int",
        np.dtype(np.float32): "float",
    }
    ctype = ctype_by_dtype.get(dtype)
    if ctype is None:
        raise TypeError(
            "scan-crop resampling supports uint8, uint16, uint32, "
            f"and float32 decoded crops; got {dtype}."
        )
    name = f"quantem_resample_scan_crop_{dtype.name.replace('float', 'f')}"
    code = f"""
    extern "C" __global__
    void {name}(
        const {ctype}* __restrict__ src,
        float* __restrict__ dst,
        const long long n,
        const int src_rows,
        const int src_cols,
        const int det_rows,
        const int det_cols,
        const int out_rows,
        const int out_cols,
        const long long src_stride_row,
        const long long src_stride_col,
        const long long src_stride_det_row,
        const long long src_stride_det_col,
        const float source_row_start,
        const float source_col_start,
        const float target_row_start,
        const float target_col_start,
        const float shift_row,
        const float shift_col
    ) {{
        long long index = (long long)blockDim.x * blockIdx.x + threadIdx.x;
        if (index >= n) {{
            return;
        }}

        int det_col = (int)(index % det_cols);
        long long tmp = index / det_cols;
        int det_row = (int)(tmp % det_rows);
        tmp /= det_rows;
        int out_col = (int)(tmp % out_cols);
        int out_row = (int)(tmp / out_cols);

        float src_row = target_row_start + (float)out_row + shift_row - source_row_start;
        float src_col = target_col_start + (float)out_col + shift_col - source_col_start;
        float max_row = src_rows > 1 ? (float)src_rows - 1.001f : 0.0f;
        float max_col = src_cols > 1 ? (float)src_cols - 1.001f : 0.0f;
        src_row = fminf(fmaxf(src_row, 0.0f), max_row);
        src_col = fminf(fmaxf(src_col, 0.0f), max_col);

        int r0 = (int)floorf(src_row);
        int c0 = (int)floorf(src_col);
        int r1 = r0 + 1 < src_rows ? r0 + 1 : src_rows - 1;
        int c1 = c0 + 1 < src_cols ? c0 + 1 : src_cols - 1;
        float wr = src_row - (float)r0;
        float wc = src_col - (float)c0;

        long long base00 = (long long)r0 * src_stride_row
            + (long long)c0 * src_stride_col
            + (long long)det_row * src_stride_det_row
            + (long long)det_col * src_stride_det_col;
        long long base01 = (long long)r0 * src_stride_row
            + (long long)c1 * src_stride_col
            + (long long)det_row * src_stride_det_row
            + (long long)det_col * src_stride_det_col;
        long long base10 = (long long)r1 * src_stride_row
            + (long long)c0 * src_stride_col
            + (long long)det_row * src_stride_det_row
            + (long long)det_col * src_stride_det_col;
        long long base11 = (long long)r1 * src_stride_row
            + (long long)c1 * src_stride_col
            + (long long)det_row * src_stride_det_row
            + (long long)det_col * src_stride_det_col;

        float p00 = (float)src[base00];
        float p01 = (float)src[base01];
        float p10 = (float)src[base10];
        float p11 = (float)src[base11];
        dst[index] =
            (1.0f - wr) * ((1.0f - wc) * p00 + wc * p01)
            + wr * ((1.0f - wc) * p10 + wc * p11);
    }}
    """
    return cuda_module(code, (name,))[name]
