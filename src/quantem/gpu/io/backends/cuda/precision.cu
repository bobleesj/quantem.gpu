// CUDA kernels for source-wide scaled-uint16 conversion.
// Compile with --fmad=false so restoration has the same float32 rounding as
// the reference CuPy expression used by the scientific parity tests.

extern "C" __device__ inline void precision_max_double(
    double* address, double value
) {
    auto bits = reinterpret_cast<unsigned long long*>(address);
    unsigned long long old = *bits;
    unsigned long long assumed;
    while (value > __longlong_as_double(old)) {
        assumed = old;
        old = atomicCAS(bits, assumed, __double_as_longlong(value));
        if (assumed == old) {
            break;
        }
    }
}

extern "C" __global__ void precision_encode(
    const float* values,
    unsigned short* codes,
    unsigned long long count,
    float scale,
    float offset
) {
    for (
        unsigned long long index =
            static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < count;
        index += static_cast<unsigned long long>(gridDim.x) * blockDim.x
    ) {
        float code = rintf((values[index] - offset) / scale);
        code = fminf(fmaxf(code, 0.0f), 65535.0f);
        codes[index] = static_cast<unsigned short>(code);
    }
}

extern "C" __global__ void precision_encode_measure(
    const float* values,
    unsigned short* codes,
    unsigned long long count,
    float encode_scale,
    float encode_offset,
    double restore_scale,
    double restore_offset,
    double* squared_error,
    double* maximum_error,
    unsigned long long* positive_to_zero,
    unsigned long long* changed,
    unsigned long long* overflow
) {
    extern __shared__ unsigned char scratch[];
    double* sum = reinterpret_cast<double*>(scratch);
    double* maximum = sum + blockDim.x;
    unsigned long long* positive =
        reinterpret_cast<unsigned long long*>(maximum + blockDim.x);
    unsigned long long* changed_local = positive + blockDim.x;
    unsigned long long* overflow_local = changed_local + blockDim.x;

    double local_sum = 0.0;
    double local_maximum = 0.0;
    unsigned long long local_positive = 0;
    unsigned long long local_changed = 0;
    unsigned long long local_overflow = 0;
    for (
        unsigned long long index =
            static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < count;
        index += static_cast<unsigned long long>(gridDim.x) * blockDim.x
    ) {
        float code = rintf((values[index] - encode_offset) / encode_scale);
        code = fminf(fmaxf(code, 0.0f), 65535.0f);
        unsigned short encoded = static_cast<unsigned short>(code);
        codes[index] = encoded;
        float restored = static_cast<float>(
            static_cast<double>(encoded) * restore_scale + restore_offset
        );
        double delta = static_cast<double>(restored) - static_cast<double>(values[index]);
        local_sum += delta * delta;
        local_maximum = fmax(local_maximum, fabs(delta));
        if (values[index] > 0.0f && encoded == 0) {
            ++local_positive;
        }
        if (delta != 0.0) {
            ++local_changed;
        }
        if (!isfinite(restored)) {
            ++local_overflow;
        }
    }

    sum[threadIdx.x] = local_sum;
    maximum[threadIdx.x] = local_maximum;
    positive[threadIdx.x] = local_positive;
    changed_local[threadIdx.x] = local_changed;
    overflow_local[threadIdx.x] = local_overflow;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
            maximum[threadIdx.x] = fmax(
                maximum[threadIdx.x], maximum[threadIdx.x + stride]
            );
            positive[threadIdx.x] += positive[threadIdx.x + stride];
            changed_local[threadIdx.x] += changed_local[threadIdx.x + stride];
            overflow_local[threadIdx.x] += overflow_local[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(squared_error, sum[0]);
        precision_max_double(maximum_error, maximum[0]);
        atomicAdd(positive_to_zero, positive[0]);
        atomicAdd(changed, changed_local[0]);
        atomicAdd(overflow, overflow_local[0]);
    }
}

extern "C" __global__ void precision_measure(
    const float* values,
    const unsigned short* codes,
    unsigned long long count,
    double scale,
    double offset,
    double* squared_error,
    double* maximum_error,
    unsigned long long* positive_to_zero,
    unsigned long long* changed,
    unsigned long long* overflow
) {
    extern __shared__ unsigned char scratch[];
    double* sum = reinterpret_cast<double*>(scratch);
    double* maximum = sum + blockDim.x;
    unsigned long long* positive =
        reinterpret_cast<unsigned long long*>(maximum + blockDim.x);
    unsigned long long* changed_local = positive + blockDim.x;
    unsigned long long* overflow_local = changed_local + blockDim.x;

    double local_sum = 0.0;
    double local_maximum = 0.0;
    unsigned long long local_positive = 0;
    unsigned long long local_changed = 0;
    unsigned long long local_overflow = 0;
    for (
        unsigned long long index =
            static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x;
        index < count;
        index += static_cast<unsigned long long>(gridDim.x) * blockDim.x
    ) {
        // Restore in float64 and round once to float32, matching
        // ``codes.astype(float64) * scale + offset`` followed by the public
        // float32 working representation.
        float restored = static_cast<float>(
            static_cast<double>(codes[index]) * scale + offset
        );
        double delta = static_cast<double>(restored) - static_cast<double>(values[index]);
        local_sum += delta * delta;
        local_maximum = fmax(local_maximum, fabs(delta));
        if (values[index] > 0.0f && codes[index] == 0) {
            ++local_positive;
        }
        if (delta != 0.0) {
            ++local_changed;
        }
        if (!isfinite(restored)) {
            ++local_overflow;
        }
    }

    sum[threadIdx.x] = local_sum;
    maximum[threadIdx.x] = local_maximum;
    positive[threadIdx.x] = local_positive;
    changed_local[threadIdx.x] = local_changed;
    overflow_local[threadIdx.x] = local_overflow;
    __syncthreads();
    for (unsigned int stride = blockDim.x / 2; stride; stride >>= 1) {
        if (threadIdx.x < stride) {
            sum[threadIdx.x] += sum[threadIdx.x + stride];
            maximum[threadIdx.x] = fmax(
                maximum[threadIdx.x], maximum[threadIdx.x + stride]
            );
            positive[threadIdx.x] += positive[threadIdx.x + stride];
            changed_local[threadIdx.x] += changed_local[threadIdx.x + stride];
            overflow_local[threadIdx.x] += overflow_local[threadIdx.x + stride];
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        atomicAdd(squared_error, sum[0]);
        precision_max_double(maximum_error, maximum[0]);
        atomicAdd(positive_to_zero, positive[0]);
        atomicAdd(changed, changed_local[0]);
        atomicAdd(overflow, overflow_local[0]);
    }
}
