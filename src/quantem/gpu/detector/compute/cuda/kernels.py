"""CUDA virtual-image reductions for resident CuPy 4D-STEM arrays."""
from __future__ import annotations

import math
from functools import lru_cache
from typing import Any

import numpy as np

_CUDA_VI_CODE = r'''
__device__ __forceinline__
unsigned int uint4_at(const unsigned char* __restrict__ data, unsigned long long logical_idx) {
    unsigned char byte = data[logical_idx >> 1];
    return (logical_idx & 1ULL) ? ((unsigned int)(byte >> 4) & 0x0fU) : ((unsigned int)byte & 0x0fU);
}

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

template <typename T, typename OutT>
__device__ __forceinline__
void selected_sum_warp32_16f_u64_impl(
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
    unsigned long long s = 0;
    if (frame < nframes) {
        const T* frame_ptr =
            data + (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < nidx; j += 32) {
            s += (unsigned long long)frame_ptr[indices[j]];
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
void selected_sum_u32_16f(
    const unsigned int* __restrict__ data,
    const int* __restrict__ indices,
    unsigned int* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_warp32_16f_u64_impl(data, indices, out, nidx, ndet, nframes);
}

extern "C" __global__
void selected_sum_u64_u8_16f(
    const unsigned char* __restrict__ data,
    const int* __restrict__ indices,
    unsigned long long* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_warp32_16f_u64_impl(data, indices, out, nidx, ndet, nframes);
}

extern "C" __global__
void selected_sum_u64_u16_16f(
    const unsigned short* __restrict__ data,
    const int* __restrict__ indices,
    unsigned long long* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_warp32_16f_u64_impl(data, indices, out, nidx, ndet, nframes);
}

extern "C" __global__
void selected_sum_u64_u32_16f(
    const unsigned int* __restrict__ data,
    const int* __restrict__ indices,
    unsigned long long* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_warp32_16f_u64_impl(data, indices, out, nidx, ndet, nframes);
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

extern "C" __global__
void selected_sum_f32_u32_16f(
    const unsigned int* __restrict__ data,
    const int* __restrict__ indices,
    float* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_warp32_16f_u64_impl(data, indices, out, nidx, ndet, nframes);
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
void selected_sum_f32_uint4_16f(
    const unsigned char* __restrict__ data,
    const int* __restrict__ indices,
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
        unsigned long long frame_base =
            (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < nidx; j += 32) {
            s += uint4_at(data, frame_base + (unsigned int)indices[j]);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, offset);
    }
    if (tx == 0 && frame < nframes) {
        out[frame] = (float)s;
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
void selected_sum_from_total_f32_warp32_16f_u64_impl(
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
    unsigned long long s = 0;
    if (frame < nframes) {
        const T* frame_ptr =
            data + (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < nidx; j += 32) {
            s += (unsigned long long)frame_ptr[indices[j]];
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, offset);
    }
    if (tx == 0 && frame < nframes) {
        unsigned long long value = total[frame] - s;
        out[frame] = (float)value;
    }
}

extern "C" __global__
void selected_sum_from_total_f32_u32_16f(
    const unsigned int* __restrict__ data,
    const int* __restrict__ indices,
    const unsigned long long* __restrict__ total,
    float* __restrict__ out,
    int nidx,
    int ndet,
    int nframes
) {
    selected_sum_from_total_f32_warp32_16f_u64_impl(
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
void selected_sum_from_total_f32_uint4_16f(
    const unsigned char* __restrict__ data,
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
        unsigned long long frame_base =
            (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < nidx; j += 32) {
            s += uint4_at(data, frame_base + (unsigned int)indices[j]);
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        s += __shfl_down_sync(0xffffffff, s, offset);
    }
    if (tx == 0 && frame < nframes) {
        out[frame] = (float)(total[frame] - (unsigned long long)s);
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

extern "C" __global__
void total_sum_u32_4f(
    const unsigned int* __restrict__ data,
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

template <typename T>
__device__ __forceinline__
void screening_sums_exact_warp128_4f_impl(
    const T* __restrict__ data,
    const unsigned char* __restrict__ detector_band_bits,
    const int* __restrict__ guard_slots,
    T* __restrict__ guard_out,
    unsigned long long* __restrict__ out,
    int guard_count,
    int ndet,
    int det_cols,
    int nframes
) {
    int tx = threadIdx.x;
    int ty = threadIdx.y;
    int frame = blockIdx.x * blockDim.y + ty;
    int lane = tx & 31;
    int warp = tx >> 5;
    __shared__ unsigned long long partial[7][16];
    unsigned long long total = 0;
    unsigned long long row_moment = 0;
    unsigned long long col_moment = 0;
    unsigned long long bright_field = 0;
    unsigned long long annular_bright_field = 0;
    unsigned long long annular_dark_field = 0;
    unsigned long long dark_field = 0;
    if (frame < nframes) {
        const T* frame_ptr =
            data + (unsigned long long)frame * (unsigned int)ndet;
        for (int detector = tx; detector < ndet; detector += 128) {
            unsigned long long value =
                (unsigned long long)frame_ptr[detector];
            int row = detector / det_cols;
            int col = detector - row * det_cols;
            unsigned char bands = detector_band_bits[detector];
            if (guard_count > 0) {
                int guard_slot = guard_slots[detector];
                if (guard_slot >= 0) {
                    guard_out[(unsigned long long)guard_slot * nframes + frame] =
                        (T)value;
                }
            }
            total += value;
            row_moment += value * (unsigned long long)row;
            col_moment += value * (unsigned long long)col;
            if (bands & 1U) bright_field += value;
            if (bands & 2U) annular_bright_field += value;
            if (bands & 4U) annular_dark_field += value;
            if (bands & 8U) dark_field += value;
        }
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        total += __shfl_down_sync(0xffffffff, total, offset);
        row_moment += __shfl_down_sync(0xffffffff, row_moment, offset);
        col_moment += __shfl_down_sync(0xffffffff, col_moment, offset);
        bright_field += __shfl_down_sync(0xffffffff, bright_field, offset);
        annular_bright_field += __shfl_down_sync(
            0xffffffff, annular_bright_field, offset
        );
        annular_dark_field += __shfl_down_sync(
            0xffffffff, annular_dark_field, offset
        );
        dark_field += __shfl_down_sync(0xffffffff, dark_field, offset);
    }
    if (lane == 0) {
        int slot = ty * 4 + warp;
        partial[0][slot] = total;
        partial[1][slot] = row_moment;
        partial[2][slot] = col_moment;
        partial[3][slot] = bright_field;
        partial[4][slot] = annular_bright_field;
        partial[5][slot] = annular_dark_field;
        partial[6][slot] = dark_field;
    }
    __syncthreads();
    unsigned long long values[7];
    #pragma unroll
    for (int product = 0; product < 7; ++product) {
        values[product] = (tx < 4) ? partial[product][ty * 4 + tx] : 0;
    }
    for (int offset = 16; offset > 0; offset >>= 1) {
        #pragma unroll
        for (int product = 0; product < 7; ++product) {
            values[product] += __shfl_down_sync(
                0xffffffff, values[product], offset
            );
        }
    }
    if (tx == 0 && frame < nframes) {
        #pragma unroll
        for (int product = 0; product < 7; ++product) {
            out[(unsigned long long)product * nframes + frame] = values[product];
        }
    }
}

#define DEFINE_SCREENING_SUMS_EXACT(NAME, TYPE)                                    \
extern "C" __global__                                                             \
void NAME(                                                                         \
    const TYPE* __restrict__ data,                                                  \
    const unsigned char* __restrict__ detector_band_bits,                           \
    const int* __restrict__ guard_slots,                                             \
    TYPE* __restrict__ guard_out,                                                    \
    unsigned long long* __restrict__ out,                                           \
    int guard_count,                                                                 \
    int ndet,                                                                       \
    int det_cols,                                                                   \
    int nframes                                                                     \
) {                                                                                 \
    screening_sums_exact_warp128_4f_impl(                                           \
        data, detector_band_bits, guard_slots, guard_out, out,                       \
        guard_count, ndet, det_cols, nframes                                         \
    );                                                                              \
}

DEFINE_SCREENING_SUMS_EXACT(screening_sums_exact_u16_4f, unsigned short)

extern "C" __global__
void total_sum_uint4_4f(
    const unsigned char* __restrict__ data,
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
        unsigned long long frame_base =
            (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < ndet; j += 128) {
            s += (unsigned long long)uint4_at(data, frame_base + (unsigned int)j);
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

extern "C" __global__
void center_of_mass_full_u32_4f(
    const unsigned int* __restrict__ data,
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
void center_of_mass_full_uint4_4f(
    const unsigned char* __restrict__ data,
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
        unsigned long long frame_base =
            (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < ndet; j += 128) {
            unsigned long long value =
                (unsigned long long)uint4_at(data, frame_base + (unsigned int)j);
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
    unsigned long long total2 = (tx < 4) ? partial_total[ty * 4 + tx] : 0;
    unsigned long long row2 = (tx < 4) ? partial_row[ty * 4 + tx] : 0;
    unsigned long long col2 = (tx < 4) ? partial_col[ty * 4 + tx] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        total2 += __shfl_down_sync(0xffffffff, total2, offset);
        row2 += __shfl_down_sync(0xffffffff, row2, offset);
        col2 += __shfl_down_sync(0xffffffff, col2, offset);
    }
    if (tx == 0 && frame < nframes) {
        if (total2 == 0) {
            out_row[frame] = 0.0f;
            out_col[frame] = 0.0f;
        } else {
            out_row[frame] = (float)row2 / (float)total2;
            out_col[frame] = (float)col2 / (float)total2;
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

extern "C" __global__
void center_of_mass_selected_uint4_4f(
    const unsigned char* __restrict__ data,
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
        unsigned long long frame_base =
            (unsigned long long)frame * (unsigned int)ndet;
        for (int j = tx; j < nidx; j += 128) {
            int idx = indices[j];
            unsigned long long value =
                (unsigned long long)uint4_at(data, frame_base + (unsigned int)idx);
            int row = idx / det_cols;
            int col = idx - row * det_cols;
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
    unsigned long long total2 = (tx < 4) ? partial_total[ty * 4 + tx] : 0;
    unsigned long long row2 = (tx < 4) ? partial_row[ty * 4 + tx] : 0;
    unsigned long long col2 = (tx < 4) ? partial_col[ty * 4 + tx] : 0;
    for (int offset = 16; offset > 0; offset >>= 1) {
        total2 += __shfl_down_sync(0xffffffff, total2, offset);
        row2 += __shfl_down_sync(0xffffffff, row2, offset);
        col2 += __shfl_down_sync(0xffffffff, col2, offset);
    }
    if (tx == 0 && frame < nframes) {
        if (total2 == 0) {
            out_row[frame] = 0.0f;
            out_col[frame] = 0.0f;
        } else {
            out_row[frame] = (float)row2 / (float)total2;
            out_col[frame] = (float)col2 / (float)total2;
        }
    }
}

extern "C" __global__
void frame_uint4_to_u8(
    const unsigned char* __restrict__ data,
    unsigned char* __restrict__ out,
    int frame,
    int ndet
) {
    unsigned int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (j >= (unsigned int)ndet) {
        return;
    }
    unsigned long long base = (unsigned long long)frame * (unsigned int)ndet;
    out[j] = (unsigned char)uint4_at(data, base + j);
}

extern "C" __global__
void mean_dp_uint4(
    const unsigned char* __restrict__ data,
    float* __restrict__ out,
    int ndet,
    int nframes
) {
    int det = blockIdx.x;
    int tx = threadIdx.x;
    __shared__ unsigned long long partial[256];
    unsigned long long s = 0;
    for (int frame = tx; frame < nframes; frame += blockDim.x) {
        unsigned long long idx = (unsigned long long)frame * (unsigned int)ndet
            + (unsigned int)det;
        s += (unsigned long long)uint4_at(data, idx);
    }
    partial[tx] = s;
    __syncthreads();
    for (int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (tx < stride) {
            partial[tx] += partial[tx + stride];
        }
        __syncthreads();
    }
    if (tx == 0) {
        out[det] = (float)partial[0] / (float)nframes;
    }
}

extern "C" __global__
void center_of_mass_selected_u32_4f(
    const unsigned int* __restrict__ data,
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

template <typename T>
__device__ __forceinline__
void selected_frame_sum_u64_impl(
    const T* __restrict__ data,
    const int* __restrict__ indices,
    unsigned long long* __restrict__ out,
    int nidx,
    int ndet
) {
    int detector = blockIdx.x * blockDim.x + threadIdx.x;
    if (detector >= ndet) {
        return;
    }
    unsigned long long sum = 0;
    for (int index = 0; index < nidx; ++index) {
        unsigned long long offset =
            (unsigned long long)indices[index] * (unsigned int)ndet
            + (unsigned int)detector;
        sum += (unsigned long long)data[offset];
    }
    out[detector] = sum;
}

#define DEFINE_SELECTED_FRAME_SUM(NAME, TYPE)                                       \
extern "C" __global__                                                               \
void NAME(                                                                           \
    const TYPE* __restrict__ data,                                                    \
    const int* __restrict__ indices,                                                  \
    unsigned long long* __restrict__ out,                                             \
    int nidx,                                                                         \
    int ndet                                                                          \
) {                                                                                  \
    selected_frame_sum_u64_impl(data, indices, out, nidx, ndet);                     \
}

DEFINE_SELECTED_FRAME_SUM(selected_frame_sum_u64_u8, unsigned char)
DEFINE_SELECTED_FRAME_SUM(selected_frame_sum_u64_u16, unsigned short)
DEFINE_SELECTED_FRAME_SUM(selected_frame_sum_u64_u32, unsigned int)

template <typename T>
__device__ __forceinline__
void selected_frame_max_u32_impl(
    const T* __restrict__ data,
    const int* __restrict__ indices,
    unsigned int* __restrict__ out,
    int nidx,
    int ndet
) {
    int detector = blockIdx.x * blockDim.x + threadIdx.x;
    if (detector >= ndet) {
        return;
    }
    unsigned int maximum = 0;
    for (int index = 0; index < nidx; ++index) {
        unsigned long long offset =
            (unsigned long long)indices[index] * (unsigned int)ndet
            + (unsigned int)detector;
        maximum = max(maximum, (unsigned int)data[offset]);
    }
    out[detector] = maximum;
}

#define DEFINE_SELECTED_FRAME_MAX(NAME, TYPE)                                       \
extern "C" __global__                                                               \
void NAME(                                                                           \
    const TYPE* __restrict__ data,                                                    \
    const int* __restrict__ indices,                                                  \
    unsigned int* __restrict__ out,                                                   \
    int nidx,                                                                         \
    int ndet                                                                          \
) {                                                                                  \
    selected_frame_max_u32_impl(data, indices, out, nidx, ndet);                     \
}

DEFINE_SELECTED_FRAME_MAX(selected_frame_max_u32_u8, unsigned char)
DEFINE_SELECTED_FRAME_MAX(selected_frame_max_u32_u16, unsigned short)
DEFINE_SELECTED_FRAME_MAX(selected_frame_max_u32_u32, unsigned int)

extern "C" __global__
void selected_frame_sum_u64_uint4(
    const unsigned char* __restrict__ data,
    const int* __restrict__ indices,
    unsigned long long* __restrict__ out,
    int nidx,
    int ndet
) {
    int detector = blockIdx.x * blockDim.x + threadIdx.x;
    if (detector >= ndet) {
        return;
    }
    unsigned long long sum = 0;
    for (int index = 0; index < nidx; ++index) {
        unsigned long long logical =
            (unsigned long long)indices[index] * (unsigned int)ndet
            + (unsigned int)detector;
        sum += (unsigned long long)uint4_at(data, logical);
    }
    out[detector] = sum;
}

extern "C" __global__
void selected_frame_max_u32_uint4(
    const unsigned char* __restrict__ data,
    const int* __restrict__ indices,
    unsigned int* __restrict__ out,
    int nidx,
    int ndet
) {
    int detector = blockIdx.x * blockDim.x + threadIdx.x;
    if (detector >= ndet) {
        return;
    }
    unsigned int maximum = 0;
    for (int index = 0; index < nidx; ++index) {
        unsigned long long logical =
            (unsigned long long)indices[index] * (unsigned int)ndet
            + (unsigned int)detector;
        maximum = max(maximum, (unsigned int)uint4_at(data, logical));
    }
    out[detector] = maximum;
}
'''


@lru_cache(maxsize=1)
def _cuda_vi_module():
    import cupy as cp

    return cp.RawModule(code=_CUDA_VI_CODE, options=("--std=c++11",))


def _scan_shape_from_flat(n_frames: int) -> tuple[int, ...]:
    side = math.isqrt(n_frames)
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
    if dtype == np.dtype(np.uint32):
        return "u32"
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


def cuda_selected_sum_uint64(data: Any, indices: Any) -> Any | None:
    """Sum selected detector pixels exactly into uint64 outputs."""
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
        return cp.zeros(scan_shape, dtype=cp.uint64)

    out = cp.empty(n_frames, dtype=cp.uint64)
    block = (32, 16, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    kernel = _cuda_vi_module().get_function(f"selected_sum_u64_{dtype_key}_16f")
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


def cuda_selected_frame_sum_uint64(data: Any, indices: Any) -> Any | None:
    """Sum selected scan frames exactly into one uint64 diffraction pattern.

    The kernel writes one detector pixel per CUDA thread and walks only the
    selected scan indices. Adjacent threads therefore read adjacent detector
    values, keeping the hot reduction coalesced without allocating a gathered
    ``selected_frames × detector_pixels`` tensor.
    """
    import cupy as cp

    if type(data).__module__.split(".", 1)[0] != "cupy":
        return None
    if not data.flags.c_contiguous:
        return None
    dtype_key = _supported_raw_dtype(data.dtype)
    if dtype_key is None:
        return None
    flat, _scan_shape, det_shape = _flatten_scan(data)
    scan_indices = cp.asarray(indices, dtype=cp.int32).reshape(-1)
    n_idx = int(scan_indices.size)
    n_det = int(flat.shape[1])
    if n_idx == 0:
        return cp.zeros(det_shape, dtype=cp.uint64)
    out = cp.empty(n_det, dtype=cp.uint64)
    threads = 256
    kernel = _cuda_vi_module().get_function(
        f"selected_frame_sum_u64_{dtype_key}"
    )
    kernel(
        ((n_det + threads - 1) // threads, 1, 1),
        (threads, 1, 1),
        (
            data,
            scan_indices,
            out,
            np.int32(n_idx),
            np.int32(n_det),
        ),
    )
    return out.reshape(det_shape)


def cuda_selected_frame_sum_uint64_uint4(data: Any, indices: Any) -> Any | None:
    """Sum selected packed-uint4 scan frames exactly without unpacking."""
    import cupy as cp

    flattened = _flatten_uint4(data)
    if flattened is None:
        return None
    buffer, _scan_shape, det_shape, _n_frames, n_det = flattened
    if type(buffer).__module__.split(".", 1)[0] != "cupy":
        return None
    if not buffer.flags.c_contiguous:
        return None
    scan_indices = cp.asarray(indices, dtype=cp.int32).reshape(-1)
    n_idx = int(scan_indices.size)
    if n_idx == 0:
        return cp.zeros(det_shape, dtype=cp.uint64)
    out = cp.empty(n_det, dtype=cp.uint64)
    threads = 256
    kernel = _cuda_vi_module().get_function("selected_frame_sum_u64_uint4")
    kernel(
        ((n_det + threads - 1) // threads, 1, 1),
        (threads, 1, 1),
        (
            buffer,
            scan_indices,
            out,
            np.int32(n_idx),
            np.int32(n_det),
        ),
    )
    return out.reshape(det_shape)


def cuda_selected_frame_max_uint32(data: Any, indices: Any) -> Any | None:
    """Take an exact per-detector maximum without a gathered frame tensor."""
    import cupy as cp

    if type(data).__module__.split(".", 1)[0] != "cupy":
        return None
    if not data.flags.c_contiguous:
        return None
    dtype_key = _supported_raw_dtype(data.dtype)
    if dtype_key is None:
        return None
    flat, _scan_shape, det_shape = _flatten_scan(data)
    scan_indices = cp.asarray(indices, dtype=cp.int32).reshape(-1)
    n_idx = int(scan_indices.size)
    n_det = int(flat.shape[1])
    if n_idx == 0:
        return cp.zeros(det_shape, dtype=cp.uint32)
    out = cp.empty(n_det, dtype=cp.uint32)
    threads = 256
    kernel = _cuda_vi_module().get_function(f"selected_frame_max_u32_{dtype_key}")
    kernel(
        ((n_det + threads - 1) // threads, 1, 1),
        (threads, 1, 1),
        (
            data,
            scan_indices,
            out,
            np.int32(n_idx),
            np.int32(n_det),
        ),
    )
    return out.reshape(det_shape)


def cuda_selected_frame_max_uint32_uint4(data: Any, indices: Any) -> Any | None:
    """Take an exact maximum directly from packed uint4 storage."""
    import cupy as cp

    flattened = _flatten_uint4(data)
    if flattened is None:
        return None
    buffer, _scan_shape, det_shape, _n_frames, n_det = flattened
    if type(buffer).__module__.split(".", 1)[0] != "cupy":
        return None
    if not buffer.flags.c_contiguous:
        return None
    scan_indices = cp.asarray(indices, dtype=cp.int32).reshape(-1)
    n_idx = int(scan_indices.size)
    if n_idx == 0:
        return cp.zeros(det_shape, dtype=cp.uint32)
    out = cp.empty(n_det, dtype=cp.uint32)
    threads = 256
    kernel = _cuda_vi_module().get_function("selected_frame_max_u32_uint4")
    kernel(
        ((n_det + threads - 1) // threads, 1, 1),
        (threads, 1, 1),
        (
            buffer,
            scan_indices,
            out,
            np.int32(n_idx),
            np.int32(n_det),
        ),
    )
    return out.reshape(det_shape)


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


def _cuda_screening_sums_exact(
    data: Any,
    detector_band_bits: Any,
    guard_slots: Any | None = None,
    guard_count: int | None = None,
) -> Any | tuple[Any, Any] | None:
    """Return seven exact per-frame screening sums with one detector traversal.

    ``detector_band_bits`` is a detector-shaped uint8 array. Bits 0 through 3
    select BF, ABF, ADF, and DF respectively. The returned product-major
    uint64 array contains total, row moment, column moment, BF, ABF, ADF, and
    DF. When ``guard_slots`` maps detector pixels to non-negative compact
    slots, the same traversal also returns exact frame-major guard counts.
    This helper is private to the prepared-screening pipeline.
    """

    import cupy as cp

    if type(data).__module__.split(".", 1)[0] != "cupy":
        return None
    if not data.flags.c_contiguous:
        return None
    dtype_key = _supported_raw_dtype(data.dtype)
    if dtype_key != "u16":
        return None
    flat, _scan_shape, detector_shape = _flatten_scan(data)
    if type(detector_band_bits).__module__.split(".", 1)[0] == "cupy":
        if detector_band_bits.dtype != cp.uint8:
            raise ValueError("detector_band_bits must have dtype uint8")
        if tuple(int(value) for value in detector_band_bits.shape) != detector_shape:
            raise ValueError(
                f"detector_band_bits shape {detector_band_bits.shape} does not "
                f"match detector shape {detector_shape}."
            )
        band_bits_gpu = detector_band_bits.reshape(-1)
    else:
        band_bits = np.asarray(detector_band_bits, dtype=np.uint8)
        if band_bits.shape != detector_shape:
            raise ValueError(
                f"detector_band_bits shape {band_bits.shape} does not match "
                f"detector shape {detector_shape}."
            )
        if np.any(band_bits & np.uint8(0xF0)):
            raise ValueError("detector_band_bits may use only bits 0 through 3")
        band_bits_gpu = cp.asarray(np.ascontiguousarray(band_bits).reshape(-1))
    n_frames = int(flat.shape[0])
    n_detector = int(flat.shape[1])
    detector_columns = int(detector_shape[1])
    out = cp.empty((7, n_frames), dtype=cp.uint64)
    guard_out = data
    resolved_guard_count = 0
    guard_slots_gpu = band_bits_gpu
    if guard_slots is not None:
        if type(guard_slots).__module__.split(".", 1)[0] == "cupy":
            if guard_slots.dtype != cp.int32:
                raise ValueError("guard_slots must have dtype int32")
            if tuple(int(value) for value in guard_slots.shape) != detector_shape:
                raise ValueError(
                    f"guard_slots shape {guard_slots.shape} does not match "
                    f"detector shape {detector_shape}."
                )
            guard_slots_gpu = guard_slots.reshape(-1)
            resolved_guard_count = (
                int(guard_slots_gpu.max()) + 1
                if guard_count is None
                else int(guard_count)
            )
            observed_min = int(guard_slots_gpu.min())
            observed_max = int(guard_slots_gpu.max())
        else:
            guard_slots_host = np.asarray(guard_slots, dtype=np.int32)
            if guard_slots_host.shape != detector_shape:
                raise ValueError(
                    f"guard_slots shape {guard_slots_host.shape} does not match "
                    f"detector shape {detector_shape}."
                )
            guard_slots_gpu = cp.asarray(guard_slots_host.reshape(-1))
            resolved_guard_count = (
                int(guard_slots_host.max(initial=-1)) + 1
                if guard_count is None
                else int(guard_count)
            )
            observed_min = int(guard_slots_host.min(initial=-1))
            observed_max = int(guard_slots_host.max(initial=-1))
        if resolved_guard_count < 0:
            raise ValueError("guard_count must be non-negative")
        if observed_min < -1 or observed_max >= resolved_guard_count:
            raise ValueError(
                "guard_slots values must be -1 or fall within guard_count"
            )
        if resolved_guard_count > 0:
            guard_out = cp.empty(
                (resolved_guard_count, n_frames),
                dtype=data.dtype,
            )
    block = (128, 4, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    kernel = _cuda_vi_module().get_function(
        f"screening_sums_exact_{dtype_key}_4f"
    )
    kernel(
        grid,
        block,
        (
            data,
            band_bits_gpu,
            guard_slots_gpu,
            guard_out,
            out,
            np.int32(resolved_guard_count),
            np.int32(n_detector),
            np.int32(detector_columns),
            np.int32(n_frames),
        ),
    )
    return (out, guard_out) if resolved_guard_count > 0 else out


def _flatten_uint4(data: Any) -> tuple[Any, tuple[int, ...], tuple[int, int], int, int] | None:
    from quantem.gpu.io.uint4 import is_packed_uint4

    if not is_packed_uint4(data) or data.backend != "cuda":
        return None
    shape = tuple(int(v) for v in data.shape)
    if len(shape) == 4:
        scan_shape = (shape[0], shape[1])
        det_shape = (shape[2], shape[3])
    elif len(shape) == 3:
        det_shape = (shape[1], shape[2])
        scan_shape = _scan_shape_from_flat(shape[0])
    else:
        raise ValueError(
            f"Expected 3D or 4D packed uint4 4D-STEM data, got "
            f"{len(shape)}D with shape {shape}."
        )
    n_frames = int(math.prod(scan_shape))
    n_det = int(det_shape[0] * det_shape[1])
    return data.buffer, scan_shape, det_shape, n_frames, n_det


def cuda_sum_all_uint64_uint4(data: Any) -> Any | None:
    """Return per-frame total detector counts for packed CUDA uint4 data."""
    import cupy as cp

    flattened = _flatten_uint4(data)
    if flattened is None:
        return None
    buffer, scan_shape, _det_shape, n_frames, n_det = flattened
    if type(buffer).__module__.split(".", 1)[0] != "cupy":
        return None
    if not buffer.flags.c_contiguous:
        return None
    out = cp.empty(n_frames, dtype=cp.uint64)
    block = (128, 4, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    kernel = _cuda_vi_module().get_function("total_sum_uint4_4f")
    kernel(
        grid,
        block,
        (
            buffer,
            out,
            np.int32(n_det),
            np.int32(n_frames),
        ),
    )
    return out.reshape(scan_shape)


def cuda_selected_sum_uint4(data: Any, indices: Any) -> Any | None:
    """Sum selected detector pixels as float32 for packed CUDA uint4 data."""
    import cupy as cp

    flattened = _flatten_uint4(data)
    if flattened is None:
        return None
    buffer, scan_shape, _det_shape, n_frames, n_det = flattened
    if type(buffer).__module__.split(".", 1)[0] != "cupy":
        return None
    if not buffer.flags.c_contiguous:
        return None
    indices = cp.asarray(indices, dtype=cp.int32)
    n_idx = int(indices.size)
    if n_idx == 0:
        return cp.zeros(scan_shape, dtype=cp.float32)
    out = cp.empty(n_frames, dtype=cp.float32)
    block = (32, 16, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    kernel = _cuda_vi_module().get_function("selected_sum_f32_uint4_16f")
    kernel(
        grid,
        block,
        (
            buffer,
            indices,
            out,
            np.int32(n_idx),
            np.int32(n_det),
            np.int32(n_frames),
        ),
    )
    return out.reshape(scan_shape)


def cuda_selected_sum_from_total_uint4(
    data: Any,
    indices: Any,
    total: Any,
) -> Any | None:
    """Return ``total - selected(indices)`` for packed CUDA uint4 data."""
    import cupy as cp

    flattened = _flatten_uint4(data)
    if flattened is None:
        return None
    buffer, scan_shape, _det_shape, n_frames, n_det = flattened
    if type(buffer).__module__.split(".", 1)[0] != "cupy":
        return None
    if not buffer.flags.c_contiguous:
        return None
    indices = cp.asarray(indices, dtype=cp.int32)
    total = cp.asarray(total, dtype=cp.uint64).reshape(-1)
    if int(total.size) != n_frames:
        return None
    n_idx = int(indices.size)
    out = cp.empty(n_frames, dtype=cp.float32)
    if n_idx == 0:
        out[...] = total.astype(cp.float32)
        return out.reshape(scan_shape)
    block = (32, 16, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    kernel = _cuda_vi_module().get_function("selected_sum_from_total_f32_uint4_16f")
    kernel(
        grid,
        block,
        (
            buffer,
            indices,
            total,
            out,
            np.int32(n_idx),
            np.int32(n_det),
            np.int32(n_frames),
        ),
    )
    return out.reshape(scan_shape)


def cuda_masked_sum_uint4(
    data: Any,
    det_mask: Any,
    *,
    total: Any | None = None,
    dense_complement_threshold: float = 0.5,
) -> Any | None:
    """Sum a detector mask for every scan position on packed CUDA uint4 data."""
    import cupy as cp

    flattened = _flatten_uint4(data)
    if flattened is None:
        return None
    _buffer, scan_shape, det_shape, _n_frames, _n_det = flattened
    mask_np = _as_mask_np(det_mask, det_shape)
    selected = int(mask_np.sum())
    n_det = int(mask_np.size)
    if selected == 0:
        return cp.zeros(scan_shape, dtype=cp.float32)
    if selected == n_det:
        total_out = total if total is not None else cuda_sum_all_uint64_uint4(data)
        return None if total_out is None else total_out.astype(cp.float32)
    if selected > int(n_det * dense_complement_threshold):
        complement = np.flatnonzero(~mask_np).astype(np.int32, copy=False)
        total_out = total if total is not None else cuda_sum_all_uint64_uint4(data)
        if total_out is None:
            return None
        return cuda_selected_sum_from_total_uint4(data, complement, total_out)
    indices = np.flatnonzero(mask_np).astype(np.int32, copy=False)
    return cuda_selected_sum_uint4(data, indices)


def cuda_frame_uint4_to_u8(data: Any, idx: int) -> Any | None:
    """Return one packed CUDA uint4 diffraction pattern as a CuPy uint8 array."""
    import cupy as cp

    flattened = _flatten_uint4(data)
    if flattened is None:
        return None
    buffer, _scan_shape, det_shape, n_frames, n_det = flattened
    frame = int(idx)
    if frame < 0 or frame >= n_frames:
        raise IndexError(f"frame index {frame} is outside 0..{n_frames - 1}")
    out = cp.empty(n_det, dtype=cp.uint8)
    threads = 256
    kernel = _cuda_vi_module().get_function("frame_uint4_to_u8")
    kernel(
        ((n_det + threads - 1) // threads, 1, 1),
        (threads, 1, 1),
        (
            buffer,
            out,
            np.int32(frame),
            np.int32(n_det),
        ),
    )
    return out.reshape(det_shape)


def cuda_mean_dp_uint4(data: Any) -> Any | None:
    """Return the mean diffraction pattern for packed CUDA uint4 data."""
    import cupy as cp

    flattened = _flatten_uint4(data)
    if flattened is None:
        return None
    buffer, _scan_shape, det_shape, n_frames, n_det = flattened
    out = cp.empty(n_det, dtype=cp.float32)
    kernel = _cuda_vi_module().get_function("mean_dp_uint4")
    kernel(
        (n_det, 1, 1),
        (256, 1, 1),
        (
            buffer,
            out,
            np.int32(n_det),
            np.int32(n_frames),
        ),
    )
    return out.reshape(det_shape)


def cuda_center_of_mass_uint4(
    data: Any,
    det_mask: Any | None = None,
) -> tuple[Any, Any] | None:
    """Return absolute detector-row/column CoM maps for packed CUDA uint4 data."""
    import cupy as cp

    flattened = _flatten_uint4(data)
    if flattened is None:
        return None
    buffer, scan_shape, det_shape, n_frames, n_det = flattened
    det_cols = int(det_shape[1])
    out_row = cp.empty(n_frames, dtype=cp.float32)
    out_col = cp.empty(n_frames, dtype=cp.float32)
    block = (128, 4, 1)
    grid = ((n_frames + block[1] - 1) // block[1], 1, 1)
    module = _cuda_vi_module()
    if det_mask is None:
        kernel = module.get_function("center_of_mass_full_uint4_4f")
        kernel(
            grid,
            block,
            (
                buffer,
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
            return cuda_center_of_mass_uint4(data, None)
        indices = cp.asarray(
            np.flatnonzero(mask_np).astype(np.int32, copy=False),
            dtype=cp.int32,
        )
        kernel = module.get_function("center_of_mass_selected_uint4_4f")
        kernel(
            grid,
            block,
            (
                buffer,
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
