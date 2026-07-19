"""CUDA virtual-image reductions for resident CuPy 4D-STEM arrays."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np


_CUDA_VI_CODE = r'''
template <typename T, typename OutT>
__device__ __forceinline__
void selected_sum_warp32_16f_impl(
    const T* __restrict__ data,
    const int* __restrict__ indices,
    OutT* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int frame = blockIdx.x * blockDim.y + ty;
    unsigned int s = 0;
    if (frame < nframes) {
        const T* frame_ptr =
            data + (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < nidx; j += 32) {
            s += (unsigned int)frame_ptr[indices[j]];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, offset);
    }
    if (tx == 0 && frame < nframes) {
        out[frame] = (OutT)s;
    }
}

extern "C" __global__
void selected_sum_u8_16f(
    const unsigned char* __restrict__ data,
    const int* __restrict__ indices,
    unsigned int* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_warp32_16f_impl(data, indices, out, nidx, ndet, nframes);
}

extern "C" __global__
void selected_sum_u16_16f(
    const unsigned short* __restrict__ data,
    const int* __restrict__ indices,
    unsigned int* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_warp32_16f_impl(data, indices, out, nidx, ndet, nframes);
}

extern "C" __global__
void selected_sum_f32_u8_16f(
    const unsigned char* __restrict__ data,
    const int* __restrict__ indices,
    float* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_warp32_16f_impl(data, indices, out, nidx, ndet, nframes);
}

extern "C" __global__
void selected_sum_f32_u16_16f(
    const unsigned short* __restrict__ data,
    const int* __restrict__ indices,
    float* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_warp32_16f_impl(data, indices, out, nidx, ndet, nframes);
}

template <typename T>
__device__ __forceinline__
void selected_sum_from_total_f32_warp32_16f_impl(
    const T* __restrict__ data,
    const int* __restrict__ indices,
    const unsigned long long* __restrict__ total,
    float* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int frame = blockIdx.x * blockDim.y + ty;
    unsigned int s = 0;
    if (frame < nframes) {
        const T* frame_ptr =
            data + (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < nidx; j += 32) {
            s += (unsigned int)frame_ptr[indices[j]];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, offset);
    }
    if (tx == 0 && frame < nframes) {
        unsigned long long value = total[frame] - (unsigned long long)s;
        out[frame] = (float)value;
    }
}

extern "C" __global__
void selected_sum_from_total_f32_u8_16f(
    const unsigned char* __restrict__ data,
    const int* __restrict__ indices,
    const unsigned long long* __restrict__ total,
    float* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_from_total_f32_warp32_16f_impl(
        data, indices, total, out, nidx, ndet, nframes
    );
}

extern "C" __global__
void selected_sum_from_total_f32_u16_16f(
    const unsigned short* __restrict__ data,
    const int* __restrict__ indices,
    const unsigned long long* __restrict__ total,
    float* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_from_total_f32_warp32_16f_impl(
        data, indices, total, out, nidx, ndet, nframes
    );
}

template <typename T>
__device__ __forceinline__
void total_sum_warp128_4f_impl(
    const T* __restrict__ data,
    unsigned long long* __restrict__ out,
    int ndet,
    int nframes
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int frame = blockIdx.x * blockDim.y + ty;
    int lane = tx & 31;
    int warp = tx >> 5;
    __shared__ unsigned long long partial[16];
    unsigned long long s = 0;
    if (frame < nframes) {
        const T* frame_ptr =
            data + (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < ndet; j += 128) {
            s += (unsigned long long)frame_ptr[j];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, offset);
    }
    if (lane == 0) {
        partial[ty * 4 + warp] = s;
    }
    __syncthreads();
    unsigned long long v = (tx < 4) ? partial[ty * 4 + tx] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffff, v, offset);
    }
    if (tx == 0 && frame < nframes) {
        out[frame] = v;
    }
}

extern "C" __global__
void total_sum_u8_4f(
    const unsigned char* __restrict__ data,
    unsigned long long* __restrict__ out,
    int ndet,
    int nframes
) {
    total_sum_warp128_4f_impl(data, out, ndet, nframes);
}

extern "C" __global__
void total_sum_u16_4f(
    const unsigned short* __restrict__ data,
    unsigned long long* __restrict__ out,
    int ndet,
    int nframes
) {
    total_sum_warp128_4f_impl(data, out, ndet, nframes);
}

template <typename T>
__device__ __forceinline__
void center_of_mass_full_warp128_4f_impl(
    const T* __restrict__ data,
    float* __restrict__ out_row,
    float* __restrict__ out_col,
    int ndet,
    int det_cols,
    int nframes
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int frame = blockIdx.x * blockDim.y + ty;
    int lane = tx & 31;
    int warp = tx >> 5;
    __shared__ unsigned long long partial_total[16];
    __shared__ unsigned long long partial_row[16];
    __shared__ unsigned long long partial_col[16];
    unsigned long long total = 0;
    unsigned long long row_sum = 0;
    unsigned long long col_sum = 0;
    if (frame < nframes) {
        const T* frame_ptr =
            data + (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < ndet; j += 128) {
            unsigned long long value = (unsigned long long)frame_ptr[j];
            int row = j / det_cols;
            int col = j - row * det_cols;
            total += value;
            row_sum += value * (unsigned long long)row;
            col_sum += value * (unsigned long long)col;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(0xffffffff, total, offset);
        row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        col_sum += __shfl_down_sync(0xffffffff, col_sum, offset);
    }
    if (lane == 0) {
        int slot = ty * 4 + warp;
        partial_total[slot] = total;
        partial_row[slot] = row_sum;
        partial_col[slot] = col_sum;
    }
    __syncthreads();
    unsigned long long t = (tx < 4) ? partial_total[ty * 4 + tx] : 0;
    unsigned long long r = (tx < 4) ? partial_row[ty * 4 + tx] : 0;
    unsigned long long c = (tx < 4) ? partial_col[ty * 4 + tx] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        t += __shfl_down_sync(0xffffffff, t, offset);
        r += __shfl_down_sync(0xffffffff, r, offset);
        c += __shfl_down_sync(0xffffffff, c, offset);
    }
    if (tx == 0 && frame < nframes) {
        if (t == 0) {
            out_row[frame] = 0.0f;
            out_col[frame] = 0.0f;
        } else {
            out_row[frame] = (float)((double)r / (double)t);
            out_col[frame] = (float)((double)c / (double)t);
        }
    }
}

extern "C" __global__
void center_of_mass_full_u8_4f(
    const unsigned char* __restrict__ data,
    float* __restrict__ out_row,
    float* __restrict__ out_col,
    int ndet,
    int det_cols,
    int nframes
) {
    center_of_mass_full_warp128_4f_impl(
        data, out_row, out_col, ndet, det_cols, nframes
    );
}

extern "C" __global__
void center_of_mass_full_u16_4f(
    const unsigned short* __restrict__ data,
    float* __restrict__ out_row,
    float* __restrict__ out_col,
    int ndet,
    int det_cols,
    int nframes
) {
    center_of_mass_full_warp128_4f_impl(
        data, out_row, out_col, ndet, det_cols, nframes
    );
}

template <typename T>
__device__ __forceinline__
void center_of_mass_selected_warp128_4f_impl(
    const T* __restrict__ data,
    const int* __restrict__ indices,
    float* __restrict__ out_row,
    float* __restrict__ out_col,
    int nidx,
    int ndet,
    int det_cols,
    int nframes
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int frame = blockIdx.x * blockDim.y + ty;
    int lane = tx & 31;
    int warp = tx >> 5;
    __shared__ unsigned long long partial_total[16];
    __shared__ unsigned long long partial_row[16];
    __shared__ unsigned long long partial_col[16];
    unsigned long long total = 0;
    unsigned long long row_sum = 0;
    unsigned long long col_sum = 0;
    if (frame < nframes) {
        const T* frame_ptr =
            data + (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < nidx; j += 128) {
            int pixel = indices[j];
            unsigned long long value = (unsigned long long)frame_ptr[pixel];
            int row = pixel / det_cols;
            int col = pixel - row * det_cols;
            total += value;
            row_sum += value * (unsigned long long)row;
            col_sum += value * (unsigned long long)col;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(0xffffffff, total, offset);
        row_sum += __shfl_down_sync(0xffffffff, row_sum, offset);
        col_sum += __shfl_down_sync(0xffffffff, col_sum, offset);
    }
    if (lane == 0) {
        int slot = ty * 4 + warp;
        partial_total[slot] = total;
        partial_row[slot] = row_sum;
        partial_col[slot] = col_sum;
    }
    __syncthreads();
    unsigned long long t = (tx < 4) ? partial_total[ty * 4 + tx] : 0;
    unsigned long long r = (tx < 4) ? partial_row[ty * 4 + tx] : 0;
    unsigned long long c = (tx < 4) ? partial_col[ty * 4 + tx] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        t += __shfl_down_sync(0xffffffff, t, offset);
        r += __shfl_down_sync(0xffffffff, r, offset);
        c += __shfl_down_sync(0xffffffff, c, offset);
    }
    if (tx == 0 && frame < nframes) {
        if (t == 0) {
            out_row[frame] = 0.0f;
            out_col[frame] = 0.0f;
        } else {
            out_row[frame] = (float)((double)r / (double)t);
            out_col[frame] = (float)((double)c / (double)t);
        }
    }
}

extern "C" __global__
void center_of_mass_selected_u8_4f(
    const unsigned char* __restrict__ data,
    const int* __restrict__ indices,
    float* __restrict__ out_row,
    float* __restrict__ out_col,
    int nidx,
    int ndet,
    int det_cols,
    int nframes
) {
    center_of_mass_selected_warp128_4f_impl(
        data, indices, out_row, out_col, nidx, ndet, det_cols, nframes
    );
}

extern "C" __global__
void center_of_mass_selected_u16_4f(
    const unsigned short* __restrict__ data,
    const int* __restrict__ indices,
    float* __restrict__ out_row,
    float* __restrict__ out_col,
    int nidx,
    int ndet,
    int det_cols,
    int nframes
) {
    center_of_mass_selected_warp128_4f_impl(
        data, indices, out_row, out_col, nidx, ndet, det_cols, nframes
    );
}
'''


@lru_cache(maxsize=1)
def _cuda_vi_module():
    import cupy as cp

    return cp.RawModule(code=_CUDA_VI_CODE, options=("--std=c++11",))


def _scan_shape_from_flat(n_frames: int) -> tuple[int, ...]:
    side = int(math.isqrt(n_frames))
    if side * side == n_frames:
        return side, side
    return (n_frames,)


def _flatten_scan(data: Any) -> tuple[Any, tuple[int, ...], tuple[int, int]]:
    if data.ndim == 4:
        scan_shape = (int(data.shape[0]), int(data.shape[1]))
        det_shape = (int(data.shape[2]), int(data.shape[3]))
        return data.reshape(-1, det_shape[0] * det_shape[1]), scan_shape, det_shape
    if data.ndim == 3:
        det_shape = (int(data.shape[1]), int(data.shape[2]))
        scan_shape = _scan_shape_from_flat(int(data.shape[0]))
        return data.reshape(-1, det_shape[0] * det_shape[1]), scan_shape, det_shape
    raise ValueError(
        f"Expected 3D or 4D 4D-STEM data, got {data.ndim}D with shape {data.shape}."
    )


def _as_mask_np(det_mask: Any, det_shape: tuple[int, int]) -> np.ndarray:
    if type(det_mask).__module__.split(".", 1)[0] == "cupy":
        det_mask = det_mask.get()
    mask_np = np.asarray(det_mask, dtype=bool)
    if mask_np.shape != det_shape:
        raise ValueError(
            f"det_mask shape {mask_np.shape} does not match detector shape {det_shape}."
        )
    return np.ascontiguousarray(mask_np.reshape(-1))


def _supported_raw_dtype(dtype: np.dtype) -> str | None:
    dtype = np.dtype(dtype)
    if dtype == np.dtype(np.uint8):
        return "u8"
    if dtype == np.dtype(np.uint16):
        return "u16"
    return None


def _uint32_accum_safe(n_pixels: int, dtype: np.dtype) -> bool:
    """Return whether a selected-pixel sum fits in the RawKernel accumulator."""
    info = np.iinfo(np.dtype(dtype))
    return int(n_pixels) * int(info.max) <= int(np.iinfo(np.uint32).max)


def cuda_sum_all_uint64(data: Any) -> Any | None:
    """Return per-frame total detector counts as uint64 for supported CuPy data."""
    import cupy as cp

    if type(data).__module__.split(".", 1)[0] != "cupy":
        return None
    if not data.flags.c_contiguous:
        return None
    if _supported_raw_dtype(data.dtype) is None:
        return None
    flat, scan_shape, _det_shape = _flatten_scan(data)
    n_frames = int(flat.shape[0])
    n_det = int(flat.shape[1])
    out = cp.empty(n_frames, dtype=cp.uint64)
    block = (128, 4, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    module = _cuda_vi_module()
    dtype_key = _supported_raw_dtype(data.dtype)
    kernel = module.get_function(f"total_sum_{dtype_key}_4f")
    kernel(
        grid,
        block,
        (
            data,
            out,
            np.int32(n_det),
            np.int32(n_frames),
        ),
    )
    return out.reshape(scan_shape)


def cuda_sum_all(data: Any) -> Any | None:
    """Return per-frame total detector counts as float32 for supported CuPy data."""
    total = cuda_sum_all_uint64(data)
    return None if total is None else total.astype("float32")


def cuda_selected_sum_uint32(data: Any, indices: Any) -> Any | None:
    """Sum selected detector pixels as uint32 with a CUDA RawKernel.

    Returns ``None`` for unsupported inputs so callers can fall back to CuPy,
    Torch, or CPU reference paths.
    """
    import cupy as cp

    if type(data).__module__.split(".", 1)[0] != "cupy":
        return None
    if not data.flags.c_contiguous:
        return None
    dtype_key = _supported_raw_dtype(data.dtype)
    if dtype_key is None:
        return None
    flat, scan_shape, _det_shape = _flatten_scan(data)
    n_frames = int(flat.shape[0])
    n_det = int(flat.shape[1])
    indices = cp.asarray(indices, dtype=cp.int32)
    n_idx = int(indices.size)
    if n_idx == 0:
        return cp.zeros(scan_shape, dtype=cp.uint32)
    if not _uint32_accum_safe(n_idx, data.dtype):
        return None

    out = cp.empty(n_frames, dtype=cp.uint32)
    block = (32, 16, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    module = _cuda_vi_module()
    kernel = module.get_function(f"selected_sum_{dtype_key}_16f")
    kernel(
        grid,
        block,
        (
            data,
            indices,
            out,
            np.int32(n_idx),
            np.int32(n_det),
            np.int32(n_frames),
        ),
    )
    return out.reshape(scan_shape)


def cuda_selected_sum(data: Any, indices: Any) -> Any | None:
    """Sum selected detector pixels as float32 with a CUDA RawKernel."""
    import cupy as cp

    if type(data).__module__.split(".", 1)[0] != "cupy":
        return None
    if not data.flags.c_contiguous:
        return None
    dtype_key = _supported_raw_dtype(data.dtype)
    if dtype_key is None:
        return None
    flat, scan_shape, _det_shape = _flatten_scan(data)
    n_frames = int(flat.shape[0])
    n_det = int(flat.shape[1])
    indices = cp.asarray(indices, dtype=cp.int32)
    n_idx = int(indices.size)
    if n_idx == 0:
        return cp.zeros(scan_shape, dtype=cp.float32)
    if not _uint32_accum_safe(n_idx, data.dtype):
        return None

    out = cp.empty(n_frames, dtype=cp.float32)
    block = (32, 16, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    module = _cuda_vi_module()
    kernel = module.get_function(f"selected_sum_f32_{dtype_key}_16f")
    kernel(
        grid,
        block,
        (
            data,
            indices,
            out,
            np.int32(n_idx),
            np.int32(n_det),
            np.int32(n_frames),
        ),
    )
    return out.reshape(scan_shape)


def cuda_selected_sum_from_total(
    data: Any,
    indices: Any,
    total: Any,
) -> Any | None:
    """Return ``total - selected(indices)`` as float32 with one CUDA kernel."""
    import cupy as cp

    if type(data).__module__.split(".", 1)[0] != "cupy":
        return None
    if not data.flags.c_contiguous:
        return None
    dtype_key = _supported_raw_dtype(data.dtype)
    if dtype_key is None:
        return None
    flat, scan_shape, _det_shape = _flatten_scan(data)
    n_frames = int(flat.shape[0])
    n_det = int(flat.shape[1])
    indices = cp.asarray(indices, dtype=cp.int32)
    total = cp.asarray(total, dtype=cp.uint64)
    if int(total.size) != n_frames:
        return None
    n_idx = int(indices.size)
    if not _uint32_accum_safe(n_idx, data.dtype):
        return None

    out = cp.empty(n_frames, dtype=cp.float32)
    if n_idx == 0:
        out[...] = total.reshape(-1).astype(cp.float32)
        return out.reshape(scan_shape)
    block = (32, 16, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    module = _cuda_vi_module()
    kernel = module.get_function(f"selected_sum_from_total_f32_{dtype_key}_16f")
    kernel(
        grid,
        block,
        (
            data,
            indices,
            total.reshape(-1),
            out,
            np.int32(n_idx),
            np.int32(n_det),
            np.int32(n_frames),
        ),
    )
    return out.reshape(scan_shape)


def cuda_masked_sum(
    data: Any,
    det_mask: Any,
    *,
    total: Any | None = None,
    dense_complement_threshold: float = 0.5,
) -> Any | None:
    """Sum a detector mask for every scan position on resident CUDA data.

    Dense masks are evaluated as ``total - unselected`` when the complement is
    smaller than the selected region. This is exact for raw-count virtual images
    and keeps dark-field dragging close to BF latency after the total-count
    image is cached.
    """
    import cupy as cp

    if type(data).__module__.split(".", 1)[0] != "cupy":
        return None
    if not data.flags.c_contiguous:
        return None
    if _supported_raw_dtype(data.dtype) is None:
        return None
    _flat, scan_shape, det_shape = _flatten_scan(data)
    mask_np = _as_mask_np(det_mask, det_shape)
    selected = int(mask_np.sum())
    n_det = int(mask_np.size)
    if selected == 0:
        return cp.zeros(scan_shape, dtype=cp.float32)
    if selected == n_det:
        total_out = total if total is not None else cuda_sum_all_uint64(data)
        return None if total_out is None else total_out.astype(cp.float32)

    if selected > int(n_det * dense_complement_threshold):
        complement = np.flatnonzero(~mask_np).astype(np.int32, copy=False)
        total_out = total if total is not None else cuda_sum_all_uint64(data)
        if total_out is None:
            return None
        return cuda_selected_sum_from_total(data, complement, total_out)

    indices = np.flatnonzero(mask_np).astype(np.int32, copy=False)
    return cuda_selected_sum(data, indices)


def cuda_center_of_mass(data: Any, det_mask: Any | None = None) -> tuple[Any, Any] | None:
    """Return absolute detector-row/column CoM maps for resident CuPy data.

    The kernel reads each diffraction pattern once and accumulates intensity,
    row moment, and column moment in integer registers. Outputs are float32
    absolute detector coordinates shaped like the scan.
    """
    import cupy as cp

    if type(data).__module__.split(".", 1)[0] != "cupy":
        return None
    if not data.flags.c_contiguous:
        return None
    dtype_key = _supported_raw_dtype(data.dtype)
    if dtype_key is None:
        return None
    flat, scan_shape, det_shape = _flatten_scan(data)
    n_frames = int(flat.shape[0])
    n_det = int(flat.shape[1])
    det_cols = int(det_shape[1])
    out_row = cp.empty(n_frames, dtype=cp.float32)
    out_col = cp.empty(n_frames, dtype=cp.float32)
    block = (128, 4, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    module = _cuda_vi_module()

    if det_mask is None:
        kernel = module.get_function(f"center_of_mass_full_{dtype_key}_4f")
        kernel(
            grid,
            block,
            (
                data,
                out_row,
                out_col,
                np.int32(n_det),
                np.int32(det_cols),
                np.int32(n_frames),
            ),
        )
    else:
        mask_np = _as_mask_np(det_mask, det_shape)
        selected = int(mask_np.sum())
        if selected == 0:
            out_row.fill(0)
            out_col.fill(0)
            return out_row.reshape(scan_shape), out_col.reshape(scan_shape)
        if selected == n_det:
            return cuda_center_of_mass(data, None)
        indices = cp.asarray(
            np.flatnonzero(mask_np).astype(np.int32, copy=False),
            dtype=cp.int32,
        )
        kernel = module.get_function(f"center_of_mass_selected_{dtype_key}_4f")
        kernel(
            grid,
            block,
            (
                data,
                indices,
                out_row,
                out_col,
                np.int32(selected),
                np.int32(n_det),
                np.int32(det_cols),
                np.int32(n_frames),
            ),
        )

    return out_row.reshape(scan_shape), out_col.reshape(scan_shape)
