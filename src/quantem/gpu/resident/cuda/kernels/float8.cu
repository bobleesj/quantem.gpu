__device__ unsigned float8_bits(unsigned value, unsigned format) {
    if (format == 4) {
        if (value == 255) return 0x7fc00000u;
        return value ? value << 23 : 0x00400000u;
    }
    const unsigned mantissa_bits = (format == 1 || format == 3) ? 2 : 3;
    const int bias = (mantissa_bits == 2 ? 15 : 7) + (format >= 2);
    const unsigned sign = (value & 128u) << 24;
    const unsigned mantissa = value & ((1u << mantissa_bits) - 1);
    const unsigned exponent = (value & 127u) >> mantissa_bits;
    if ((format >= 2 && value == 128) ||
        (format == 0 && (value & 127u) == 127)) return 0x7fc00000u;
    if (format == 1 && exponent == 31)
        return sign | (mantissa ? 0x7fc00000u : 0x7f800000u);
    if (!exponent) {
        if (!mantissa) return sign;
        const unsigned top = 31 - __clz(mantissa);
        return sign | (unsigned(128 - bias - int(mantissa_bits) + int(top)) << 23)
            | ((mantissa << (23 - top)) & 0x7fffffu);
    }
    return sign | (unsigned(int(exponent) - bias + 127) << 23)
        | (mantissa << (23 - mantissa_bits));
}

// Scale through IEEE bits so subnormal inputs and outputs survive device FTZ.
__device__ float float8_scale(float value, int shift) {
    const unsigned bits = __float_as_uint(value);
    const unsigned sign = bits & 0x80000000u;
    int exponent = int((bits >> 23) & 255u);
    unsigned mantissa = bits & 0x7fffffu;
    if (exponent == 255 || !(bits & 0x7fffffffu)) return value;
    if (exponent) mantissa |= 0x800000u;
    else {
        const unsigned normalize = __clz(mantissa) - 8u;
        mantissa <<= normalize;
        exponent = 1 - int(normalize);
    }
    exponent += shift;
    unsigned result;
    if (exponent >= 255) result = 0x7f800000u;
    else if (exponent > 0) result = (unsigned(exponent) << 23) | (mantissa & 0x7fffffu);
    else {
        const unsigned displacement = unsigned(1 - exponent);
        result = 0;
        if (displacement <= 24) {
            const unsigned remainder = mantissa & ((1u << displacement) - 1u);
            const unsigned midpoint = 1u << (displacement - 1u);
            result = mantissa >> displacement;
            result += remainder > midpoint || (remainder == midpoint && (result & 1u));
        }
    }
    return __uint_as_float(result | sign);
}

extern "C" __global__ void float8_decode(
    const void* input, float* output, unsigned* invalid,
    unsigned long long count, unsigned width, unsigned format
) {
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned value = width == 1 ? ((const unsigned char*)input)[index]
                                     : ((const unsigned short*)input)[index];
    if (value > 255) atomicExch(invalid, 1u);
    output[index] = __uint_as_float(float8_bits(value & 255u, format));
}

extern "C" __global__ void float8_bytes(
    const unsigned short* input, unsigned char* output,
    unsigned* errors, unsigned long long count
) {
    const unsigned long long index = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned value = input[index];
    if (value > 255u) atomicOr(errors, 1u);
    output[index] = value;
}

extern "C" __global__ void float8_pattern_total(
    const unsigned short* input, double* total, const unsigned char* selected,
    const unsigned char* valid, unsigned* errors, unsigned scans,
    unsigned pixels, unsigned first, unsigned format, unsigned maximum
) {
    const unsigned pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    if (!valid[pixel]) { total[pixel]=0; return; }
    double sum=total[pixel];
    for (unsigned scan=0; scan<scans; ++scan) {
        if (!selected[first+scan]) continue;
        const unsigned word=input[(unsigned long long)scan*pixels+pixel];
        if (word>255u) atomicOr(errors,1u);
        const double value=format == 4
            ? __longlong_as_double(word == 255u ? 0x7ff8000000000000ull : (896ull + (word & 255u)) << 52)
            : double(__uint_as_float(float8_bits(word & 255u,format)));
        sum=maximum ? ((isnan(value) || value>sum) ? value : sum) : sum+value;
    }
    total[pixel]=sum;
}

extern "C" __global__ void float8_pattern_finish(
    const double* input, float* output, unsigned count, unsigned divisor
) {
    const unsigned index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= count) return;
    const unsigned long long bits = __double_as_longlong(input[index] / double(divisor));
    const unsigned exponent = (bits >> 52) & 2047u;
    const unsigned sign = (bits >> 32) & 0x80000000u;
    const unsigned long long fraction = bits & 0xfffffffffffffull;
    unsigned result = 0;
    if (exponent == 2047u) result = fraction ? 0x7fc00000u : 0x7f800000u;
    else if (bits & 0x7fffffffffffffffull) {
        const int adjusted = int(exponent) - 1023 + 127;
        if (adjusted >= 255) result = 0x7f800000u;
        else if (adjusted > -24) {
            const unsigned shift = adjusted > 0 ? 29u : 30u - adjusted;
            const unsigned long long significand = fraction | 0x10000000000000ull;
            unsigned long long rounded = significand >> shift;
            const unsigned long long remainder = significand & ((1ull << shift) - 1);
            const unsigned long long half = 1ull << (shift - 1);
            rounded += remainder > half || (remainder == half && (rounded & 1ull));
            result = adjusted > 0 ? (unsigned(adjusted - 1) << 23) + unsigned(rounded)
                                  : unsigned(rounded);
        }
    }
    output[index] = __uint_as_float(result | sign);
}
