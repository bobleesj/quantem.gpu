#include <metal_stdlib>
using namespace metal;

struct MetalFFT2DParameters {
    uint width;
    uint height;
    uint log2Size;
    uint stage;
    float direction;
    uint rowAxis;
};

struct DPCPackParameters {
    uint count;
    uint flags;
    uint _padding0;
    uint _padding1;
    float4 rotation;
};

inline float2 complexMultiply(float2 a, float2 b) {
    return float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

inline float2 twiddle(uint index, uint size, float direction) {
    float angle = direction * 2.0f * M_PI_F * float(index) / float(size);
    return float2(cos(angle), sin(angle));
}

inline uint bitReverse(uint value, uint bitCount) {
    uint result = 0u;
    for (uint bit = 0u; bit < bitCount; ++bit) {
        result = (result << 1u) | (value & 1u);
        value >>= 1u;
    }
    return result;
}

kernel void dpc_pack_complex(
    device const float *rowDPC [[buffer(0)]],
    device const float *columnDPC [[buffer(1)]],
    device float2 *complexField [[buffer(2)]],
    constant DPCPackParameters &parameters [[buffer(3)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= parameters.count) return;
    float row = rowDPC[index];
    float column = columnDPC[index];
    float cosine = parameters.rotation.x;
    float sine = parameters.rotation.y;
    float gradientRow;
    float gradientColumn;
    if ((parameters.flags & 1u) != 0u) {
        gradientRow = sine * column + cosine * row;
        gradientColumn = cosine * column - sine * row;
    } else {
        gradientRow = cosine * row - sine * column;
        gradientColumn = sine * row + cosine * column;
    }
    complexField[index] = float2(gradientRow, gradientColumn);
}

kernel void fft_bit_reverse_rows(
    device float2 *data [[buffer(0)]],
    constant MetalFFT2DParameters &parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint column = position.x;
    uint row = position.y;
    if (row >= parameters.height || column >= parameters.width) return;
    uint reversed = bitReverse(column, parameters.log2Size);
    if (column < reversed) {
        uint first = row * parameters.width + column;
        uint second = row * parameters.width + reversed;
        float2 temporary = data[first];
        data[first] = data[second];
        data[second] = temporary;
    }
}

kernel void fft_bit_reverse_columns(
    device float2 *data [[buffer(0)]],
    constant MetalFFT2DParameters &parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint column = position.x;
    uint row = position.y;
    if (row >= parameters.height || column >= parameters.width) return;
    uint reversed = bitReverse(row, parameters.log2Size);
    if (row < reversed) {
        uint first = row * parameters.width + column;
        uint second = reversed * parameters.width + column;
        float2 temporary = data[first];
        data[first] = data[second];
        data[second] = temporary;
    }
}

kernel void fft_butterfly_rows(
    device float2 *data [[buffer(0)]],
    constant MetalFFT2DParameters &parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint index = position.x;
    uint row = position.y;
    if (row >= parameters.height || index >= parameters.width / 2u) return;
    uint halfSize = 1u << parameters.stage;
    uint fullSize = halfSize << 1u;
    uint group = index / halfSize;
    uint offset = index % halfSize;
    uint firstColumn = group * fullSize + offset;
    uint secondColumn = firstColumn + halfSize;
    if (secondColumn >= parameters.width) return;
    uint first = row * parameters.width + firstColumn;
    uint second = row * parameters.width + secondColumn;
    float2 even = data[first];
    float2 odd = complexMultiply(
        twiddle(offset, fullSize, parameters.direction), data[second]
    );
    data[first] = even + odd;
    data[second] = even - odd;
}

kernel void fft_butterfly_columns(
    device float2 *data [[buffer(0)]],
    constant MetalFFT2DParameters &parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    uint column = position.x;
    uint index = position.y;
    if (column >= parameters.width || index >= parameters.height / 2u) return;
    uint halfSize = 1u << parameters.stage;
    uint fullSize = halfSize << 1u;
    uint group = index / halfSize;
    uint offset = index % halfSize;
    uint firstRow = group * fullSize + offset;
    uint secondRow = firstRow + halfSize;
    if (secondRow >= parameters.height) return;
    uint first = firstRow * parameters.width + column;
    uint second = secondRow * parameters.width + column;
    float2 even = data[first];
    float2 odd = complexMultiply(
        twiddle(offset, fullSize, parameters.direction), data[second]
    );
    data[first] = even + odd;
    data[second] = even - odd;
}

kernel void fft_normalize_2d(
    device float2 *data [[buffer(0)]],
    constant MetalFFT2DParameters &parameters [[buffer(1)]],
    uint2 position [[thread_position_in_grid]]
) {
    if (position.x >= parameters.width || position.y >= parameters.height) return;
    uint index = position.y * parameters.width + position.x;
    data[index] /= float(parameters.width * parameters.height);
}

inline float frequency(uint index, uint size) {
    if (index < (size + 1u) / 2u) return float(index) / float(size);
    return -float(size - index) / float(size);
}

kernel void dpc_poisson_frequency(
    device const float2 *gradientFFT [[buffer(0)]],
    device float2 *phaseFFT [[buffer(1)]],
    constant uint4 &shape [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint width = shape.x;
    uint height = shape.y;
    uint count = shape.z;
    if (index >= count) return;
    if (index == 0u) {
        phaseFFT[0] = float2(0.0f);
        return;
    }
    uint row = index / width;
    uint column = index - row * width;
    uint mirrorRow = (height - row) % height;
    uint mirrorColumn = (width - column) % width;
    uint mirror = mirrorRow * width + mirrorColumn;
    float2 value = gradientFFT[index];
    float2 mirrorConjugate = float2(
        gradientFFT[mirror].x, -gradientFFT[mirror].y
    );
    float2 rowFFT = 0.5f * (value + mirrorConjugate);
    float2 difference = value - mirrorConjugate;
    float2 columnFFT = float2(0.5f * difference.y, -0.5f * difference.x);
    float rowFrequency = frequency(row, height);
    float columnFrequency = frequency(column, width);
    float frequencySquared =
        rowFrequency * rowFrequency + columnFrequency * columnFrequency;
    float2 gradient = rowFFT * rowFrequency + columnFFT * columnFrequency;
    float scale = 0.25f / frequencySquared;
    phaseFFT[index] = float2(gradient.y * scale, -gradient.x * scale);
}

kernel void dpc_extract_phase(
    device const float2 *phaseComplex [[buffer(0)]],
    device float *phase [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    phase[index] = -phaseComplex[index].x;
}
