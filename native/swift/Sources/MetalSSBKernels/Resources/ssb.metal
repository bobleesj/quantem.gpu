#include <metal_stdlib>
using namespace metal;

kernel void ssb_object_phase(device const float2 *object [[buffer(0)]],
    device float *phase [[buffer(1)]], uint index [[thread_position_in_grid]]) {
    phase[index] = atan2(object[index].y, object[index].x);
}

struct FFTParams {
    uint n;
    uint log2n;
    uint batch;
    uint inverse;
};

struct HalfExtractParams {
    uint source_n;
    uint batch;
};

kernel void counts_to_complex(
    device const uchar *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &byte_width [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    if (index < count) {
        uint value = byte_width == 1u ? uint(input[index])
            : byte_width == 2u ? uint(((device const ushort *)input)[index])
            : ((device const uint *)input)[index];
        output[index] = float2(float(value), 0.0f);
    }
}

// 512 = 8^3. This is the parity-checked radix-8 kernel retained from the
// native-device FFT implementation. One threadgroup transforms one row.
// Combine two radix-2 stages in registers between barriers. Input is already
// bit-reversed; 128 uses an initial radix-2 stage followed by three radix-4
// stages, while 256 uses four radix-4 stages. One group has 64 threads.
inline void fft_small_inplace(threadgroup float2 *work,
    device const float2 *twiddle, uint n, uint log2n, bool inverse, uint tid) {
    if (log2n & 1u) {
        const uint a = 2u * tid;
        const float2 x = work[a];
        const float2 y = work[a + 1u];
        work[a] = x + y;
        work[a + 1u] = x - y;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint width = (log2n & 1u) ? 8u : 4u; width <= n; width *= 4u) {
        if (tid < n / 4u) {
            const uint quarter = width / 4u;
            const uint j = tid % quarter;
            const uint a = (tid / quarter) * width + j;
            float2 w1 = twiddle[j * (2u * n / width)];
            float2 w2 = twiddle[j * (n / width)];
            if (inverse) { w1.y = -w1.y; w2.y = -w2.y; }
            const float2 w3 = inverse ? float2(-w2.y, w2.x) : float2(w2.y, -w2.x);
            const float2 x0 = work[a];
            const float2 x1 = complex_mul(w1, work[a + quarter]);
            const float2 x2 = work[a + 2u * quarter];
            const float2 x3 = complex_mul(w1, work[a + 3u * quarter]);
            const float2 u = x0 + x1;
            const float2 v = x0 - x1;
            const float2 p = complex_mul(w2, x2 + x3);
            const float2 q = complex_mul(w3, x2 - x3);
            work[a] = u + p;
            work[a + quarter] = v + q;
            work[a + 2u * quarter] = u - p;
            work[a + 3u * quarter] = v - q;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

// Native 128/256 row FFT. A group owns one row; butterfly pairs are disjoint
// at each stage. Keep the independently tuned radix-8 512 implementation below.
kernel void fft_rows_small_mixed_radix(
    device const float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    device const float2 *twiddle [[buffer(2)]],
    constant FFTParams &params [[buffer(3)]],
    threadgroup float2 *work [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]],
    uint3 threads [[threads_per_threadgroup]]) {
    const uint n = params.n;
    const size_t base = (size_t)group.x * n;
    for (uint i = tid; i < n; i += threads.x) {
        const uint reversed = reverse_bits(i) >> (32u - params.log2n);
        work[reversed] = input[base + i];
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    fft_small_inplace(work, twiddle, n, params.log2n, params.inverse != 0u, tid);
    const float scale = params.inverse ? 1.0f / float(n) : 1.0f;
    for (uint i = tid; i < n; i += threads.x) output[base + i] = work[i] * scale;
}

kernel void fft_rows_radix8(
    device const float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    device const float2 *twiddle [[buffer(2)]],
    constant FFTParams &params [[buffer(3)]],
    threadgroup float2 *scratch [[threadgroup(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint group [[threadgroup_position_in_grid]]) {

    const uint n = params.n;
    const uint eighth_n = n >> 3u;
    const uint radix8_digits = params.log2n / 3u;
    const uint row = group % n;
    const uint batch_index = group / n;
    if (batch_index >= params.batch || params.log2n % 3u != 0u) return;
    const size_t base = ((size_t)batch_index * n + row) * n;
    const bool inverse = params.inverse != 0u;

    for (uint butterfly = tid; butterfly < eighth_n; butterfly += threads) {
        uint pos[8];
        float2 x[8];
        for (uint lane = 0u; lane < 8u; ++lane) {
            pos[lane] = butterfly + lane * eighth_n;
            x[lane] = input[base + pos[lane]];
        }
        float2 y0, y1, y2, y3, y4, y5, y6, y7;
        fft8_butterfly(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], inverse,
            y0, y1, y2, y3, y4, y5, y6, y7);
        scratch[reverse_base8_digits(pos[0], radix8_digits)] = y0;
        scratch[reverse_base8_digits(pos[1], radix8_digits)] = y1;
        scratch[reverse_base8_digits(pos[2], radix8_digits)] = y2;
        scratch[reverse_base8_digits(pos[3], radix8_digits)] = y3;
        scratch[reverse_base8_digits(pos[4], radix8_digits)] = y4;
        scratch[reverse_base8_digits(pos[5], radix8_digits)] = y5;
        scratch[reverse_base8_digits(pos[6], radix8_digits)] = y6;
        scratch[reverse_base8_digits(pos[7], radix8_digits)] = y7;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);

    for (uint width = 64u; width <= n; width <<= 3u) {
        const uint eighth = width >> 3u;
        for (uint butterfly = tid; butterfly < eighth_n; butterfly += threads) {
            const uint block = butterfly / eighth;
            const uint j = butterfly - block * eighth;
            const uint first = block * width + j;
            const uint twiddle_index = j * (n / width);
            float2 x[8];
            x[0] = scratch[first];
            for (uint lane = 1u; lane < 8u; ++lane) {
                float2 w = twiddle[twiddle_index * lane * (512u / n)];
                if (inverse) w.y = -w.y;
                x[lane] = complex_mul(w, scratch[first + lane * eighth]);
            }
            float2 y0, y1, y2, y3, y4, y5, y6, y7;
            fft8_butterfly(x[0], x[1], x[2], x[3], x[4], x[5], x[6], x[7], inverse,
                y0, y1, y2, y3, y4, y5, y6, y7);
            scratch[first] = y0;
            scratch[first + eighth] = y1;
            scratch[first + 2u * eighth] = y2;
            scratch[first + 3u * eighth] = y3;
            scratch[first + 4u * eighth] = y4;
            scratch[first + 5u * eighth] = y5;
            scratch[first + 6u * eighth] = y6;
            scratch[first + 7u * eighth] = y7;
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }

    const float scale = inverse ? 1.0f / float(n) : 1.0f;
    for (uint index = tid; index < n; index += threads) {
        output[base + index] = scratch[index] * scale;
    }
}

kernel void extract_hermitian_half(
    device const float2 *full_g [[buffer(0)]],
    device float2 *half_g [[buffer(1)]],
    constant HalfExtractParams &params [[buffer(2)]],
    uint index [[thread_position_in_grid]]) {
    const uint half_cols = params.source_n / 2u + 1u;
    const uint half_plane = params.source_n * half_cols;
    const uint total = params.batch * half_plane;
    if (index >= total) return;
    const uint local_bf = index / half_plane;
    const uint pixel = index - local_bf * half_plane;
    const uint row = pixel / half_cols;
    const uint col = pixel - row * half_cols;
    const size_t full_plane = (size_t)params.source_n * params.source_n;
    half_g[index] = full_g[(size_t)local_bf * full_plane +
        (size_t)row * params.source_n + col];
}

kernel void transpose_complex32(
    device const float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    constant FFTParams &params [[buffer(2)]],
    uint3 tid [[thread_position_in_threadgroup]],
    uint3 group [[threadgroup_position_in_grid]]) {
    threadgroup float2 tile[32][33];
    const uint n = params.n;
    const uint batch_index = group.z;
    if (batch_index >= params.batch) return;
    const size_t plane = (size_t)batch_index * n * n;
    const uint input_x = group.x * 32u + tid.x;
    const uint input_y = group.y * 32u + tid.y;
    for (uint offset = 0; offset < 32u; offset += 8u) {
        if (input_x < n && input_y + offset < n) {
            tile[tid.y + offset][tid.x] =
                input[plane + (size_t)(input_y + offset) * n + input_x];
        }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint output_x = group.y * 32u + tid.x;
    const uint output_y = group.x * 32u + tid.y;
    for (uint offset = 0; offset < 32u; offset += 8u) {
        if (output_x < n && output_y + offset < n) {
            output[plane + (size_t)(output_y + offset) * n + output_x] =
                tile[tid.x][tid.y + offset];
        }
    }
}
