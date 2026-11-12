#include <metal_stdlib>
using namespace metal;

struct BluesteinPlan {
    uint rows, cols, n, m, axis, kernelOnly;
};

inline float2 bluestein_complex_mul(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline uint bluestein_reverse_bits(uint value, uint m) {
    uint result = 0;
    for (uint mask = m >> 1; mask; mask >>= 1) {
        result = (result << 1) | (value & 1);
        value >>= 1;
    }
    return result;
}

// One butterfly per thread. Every thread reaches every barrier. The complete
// convolution stays in threadgroup memory, with no padded global scan buffer.
inline void bluestein_fft(
    threadgroup float2 *values, device const float2 *twiddle,
    uint m, uint tid, bool inverse
) {
    for (uint width = 2; width <= m; width <<= 1) {
        const uint halfWidth = width >> 1;
        const uint j = tid % halfWidth;
        const uint start = (tid / halfWidth) * width + j;
        const float2 a = values[start];
        float2 w = twiddle[j * (m / width)];
        if (inverse) w.y = -w.y;
        const float2 b = bluestein_complex_mul(values[start + halfWidth], w);
        values[start] = a + b;
        values[start + halfWidth] = a - b;
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}

kernel void direct_ptycho_bluestein(
    device const float2 *input [[buffer(0)]],
    device float2 *output [[buffer(1)]],
    device const float2 *chirp [[buffer(2)]],
    device const float2 *twiddle [[buffer(3)]],
    device const float2 *kernelFFT [[buffer(4)]],
    constant BluesteinPlan &p [[buffer(5)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]],
    threadgroup float2 *scratch [[threadgroup(0)]]
) {
    const uint line = group.x;
    const uint planeOffset = group.y * p.rows * p.cols;
    // Remove each line's DC before convolution, then restore the exact GPU
    // reduction at frequency zero. This avoids background-dependent roundoff
    // in weak nonzero frequencies; it does not remove any Fourier evidence.
    float2 dc = float2(0);
    float2 mean = float2(0);
    if (!p.kernelOnly) {
        const uint index = p.axis == 1 ? line * p.cols + tid : tid * p.cols + line;
        scratch[tid] = tid < p.n ? input[planeOffset + index] : float2(0);
        threadgroup_barrier(mem_flags::mem_threadgroup);
        for (uint stride = p.m / 4; stride > 0; stride >>= 1) {
            if (tid < stride) scratch[tid] += scratch[tid + stride];
            threadgroup_barrier(mem_flags::mem_threadgroup);
        }
        dc = scratch[0];
        mean = dc / float(p.n);
        threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint j = tid; j < p.m; j += p.m / 2) {
        float2 value = float2(0);
        if (p.kernelOnly) {
            // Symmetric convolution kernel, generated once per axis on GPU.
            if (j < p.n) value = float2(chirp[j].x, -chirp[j].y);
            else if (j > p.m - p.n)
                value = float2(chirp[p.m - j].x, -chirp[p.m - j].y);
        } else if (j < p.n) {
            const uint index = p.axis == 1 ? line * p.cols + j : j * p.cols + line;
            value = bluestein_complex_mul(input[planeOffset + index] - mean, chirp[j]);
        }
        scratch[bluestein_reverse_bits(j, p.m)] = value;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bluestein_fft(scratch, twiddle, p.m, tid, false);
    if (p.kernelOnly) {
        output[tid] = scratch[tid];
        output[tid + p.m / 2] = scratch[tid + p.m / 2];
        return;
    }
    const float2 first = bluestein_complex_mul(scratch[tid], kernelFFT[tid]);
    const float2 second = bluestein_complex_mul(
        scratch[tid + p.m / 2], kernelFFT[tid + p.m / 2]);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    scratch[bluestein_reverse_bits(tid, p.m)] = first;
    scratch[bluestein_reverse_bits(tid + p.m / 2, p.m)] = second;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bluestein_fft(scratch, twiddle, p.m, tid, true);
    for (uint j = tid; j < p.n; j += p.m / 2) {
        const uint index = p.axis == 1 ? line * p.cols + j : j * p.cols + line;
        output[planeOffset + index] = j == 0 ? dc
            : bluestein_complex_mul(scratch[j] / float(p.m), chirp[j]);
    }
}
