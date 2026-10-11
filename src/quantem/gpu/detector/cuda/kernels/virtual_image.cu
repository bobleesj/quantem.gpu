
// Selected-pixel sums accumulate in 64 bits for every output type: 65,538
// saturated uint16 pixels already pass 2^32, and a float image must round the
// exact count once rather than a wrapped one.
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

// Every family is exported once per native count type; the templates above hold the code.
#define DEFINE_COUNT_KERNELS(SUFFIX, TYPE)                                                          \
extern "C" __global__ void selected_sum_u64_##SUFFIX##_16f(                                         \
    const TYPE* __restrict__ data, const int* __restrict__ indices,                                 \
    unsigned long long* __restrict__ out, int nidx, int ndet, int nframes) {                        \
    selected_sum_warp32_16f_impl(data, indices, out, nidx, ndet, nframes);                          \
}                                                                                                   \
extern "C" __global__ void selected_sum_f32_##SUFFIX##_16f(                                         \
    const TYPE* __restrict__ data, const int* __restrict__ indices,                                 \
    float* __restrict__ out, int nidx, int ndet, int nframes) {                                     \
    selected_sum_warp32_16f_impl(data, indices, out, nidx, ndet, nframes);                          \
}                                                                                                   \
extern "C" __global__ void center_of_mass_full_##SUFFIX##_4f(                                       \
    const TYPE* __restrict__ data, float* __restrict__ out_row, float* __restrict__ out_col,        \
    int ndet, int det_cols, int nframes) {                                                          \
    center_of_mass_full_warp128_4f_impl(data, out_row, out_col, ndet, det_cols, nframes);           \
}                                                                                                   \
extern "C" __global__ void center_of_mass_selected_##SUFFIX##_4f(                                   \
    const TYPE* __restrict__ data, const int* __restrict__ indices,                                 \
    float* __restrict__ out_row, float* __restrict__ out_col,                                       \
    int nidx, int ndet, int det_cols, int nframes) {                                                \
    center_of_mass_selected_warp128_4f_impl(                                                        \
        data, indices, out_row, out_col, nidx, ndet, det_cols, nframes);                            \
}                                                                                                   \
extern "C" __global__ void selected_frame_sum_u64_##SUFFIX(                                         \
    const TYPE* __restrict__ data, const int* __restrict__ indices,                                 \
    unsigned long long* __restrict__ out, int nidx, int ndet) {                                     \
    selected_frame_sum_u64_impl(data, indices, out, nidx, ndet);                                    \
}                                                                                                   \
extern "C" __global__ void selected_frame_max_u32_##SUFFIX(                                         \
    const TYPE* __restrict__ data, const int* __restrict__ indices,                                 \
    unsigned int* __restrict__ out, int nidx, int ndet) {                                           \
    selected_frame_max_u32_impl(data, indices, out, nidx, ndet);                                    \
}

DEFINE_COUNT_KERNELS(u8, unsigned char)
DEFINE_COUNT_KERNELS(u16, unsigned short)
DEFINE_COUNT_KERNELS(u32, unsigned int)
