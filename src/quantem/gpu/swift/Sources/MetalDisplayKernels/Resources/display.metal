#include <metal_stdlib>
using namespace metal;

struct MetalDisplayParameters {
    uint rows;
    uint cols;
    uint low;
    uint high;
    uint scaleMode;
    uint lutCount;
    uint _padding0;
    uint _padding1;
};

struct MetalFloatDisplayParameters {
    uint rows;
    uint cols;
    float low;
    float high;
    uint scaleMode;
    uint lutCount;
    uint _padding0;
    uint _padding1;
};

inline float metal_signed_log1p(float value) {
    return copysign(log(1.0f + abs(value)), value);
}

inline float metal_normalize_u32(
    uint value,
    constant MetalDisplayParameters &parameters
) {
    uint high = max(parameters.low, parameters.high);
    if (high == parameters.low) return 0.5f;
    uint clipped = clamp(value, parameters.low, high);
    float low = float(parameters.low);
    float highValue = float(high);
    float displayValue = float(clipped);
    if (parameters.scaleMode == 1u) {
        low = log(1.0f + low);
        highValue = log(1.0f + highValue);
        displayValue = log(1.0f + displayValue);
    }
    return clamp(
        (displayValue - low) / max(1.0e-30f, highValue - low),
        0.0f,
        1.0f
    );
}

inline float metal_normalize_f32(
    float value,
    constant MetalFloatDisplayParameters &parameters
) {
    if (isnan(value)) return 0.0f;
    if (isinf(value)) return value > 0.0f ? 1.0f : 0.0f;
    float low = parameters.low;
    float high = max(low, parameters.high);
    if (!(high > low)) return 0.5f;
    float displayValue = value;
    if (parameters.scaleMode == 1u) {
        displayValue = metal_signed_log1p(displayValue);
        low = metal_signed_log1p(low);
        high = metal_signed_log1p(high);
    }
    float span = high - low;
    float normalized;
    if (isinf(span)) {
        float negativeMagnitude = -low;
        float center;
        if (negativeMagnitude <= high) {
            float ratio = negativeMagnitude / high;
            center = ratio / (1.0f + ratio);
        } else {
            float ratio = high / negativeMagnitude;
            center = 1.0f / (1.0f + ratio);
        }
        normalized = displayValue >= 0.0f
            ? center + (1.0f - center) * (displayValue / high)
            : center * (1.0f - displayValue / low);
    } else {
        normalized = (displayValue - low) / span;
    }
    return clamp(normalized, 0.0f, 1.0f);
}

struct MetalDisplayVertex {
    float4 position [[position]];
    float2 uv;
};

vertex MetalDisplayVertex metal_display_vertex(uint vertexID [[vertex_id]]) {
    const float2 positions[4] = {
        float2(-1.0, -1.0), float2(1.0, -1.0),
        float2(-1.0, 1.0), float2(1.0, 1.0)
    };
    const float2 coordinates[4] = {
        float2(0.0, 1.0), float2(1.0, 1.0),
        float2(0.0, 0.0), float2(1.0, 0.0)
    };
    MetalDisplayVertex output;
    output.position = float4(positions[vertexID], 0.0, 1.0);
    output.uv = coordinates[vertexID];
    return output;
}

fragment float4 metal_display_fragment(
    MetalDisplayVertex input [[stage_in]],
    device const uint *values [[buffer(0)]],
    constant MetalDisplayParameters &parameters [[buffer(1)]],
    device const float4 *lut [[buffer(2)]]
) {
    if (parameters.rows == 0u || parameters.cols == 0u || parameters.lutCount == 0u) {
        return float4(0.0f);
    }
    uint col = min(parameters.cols - 1u, uint(input.uv.x * float(parameters.cols)));
    uint row = min(parameters.rows - 1u, uint(input.uv.y * float(parameters.rows)));
    float normalized = metal_normalize_u32(values[row * parameters.cols + col], parameters);
    uint lutIndex = min(
        parameters.lutCount - 1u,
        uint(normalized * float(parameters.lutCount - 1u))
    );
    return lut[lutIndex];
}

fragment float4 metal_display_fragment_f32(
    MetalDisplayVertex input [[stage_in]],
    device const float *values [[buffer(0)]],
    constant MetalFloatDisplayParameters &parameters [[buffer(1)]],
    device const float4 *lut [[buffer(2)]]
) {
    if (parameters.rows == 0u || parameters.cols == 0u || parameters.lutCount == 0u) {
        return float4(0.0f);
    }
    uint col = min(parameters.cols - 1u, uint(input.uv.x * float(parameters.cols)));
    uint row = min(parameters.rows - 1u, uint(input.uv.y * float(parameters.rows)));
    float value = values[row * parameters.cols + col];
    if (!isfinite(value)) {
        return float4(143.0f / 255.0f, 63.0f / 255.0f, 143.0f / 255.0f, 1.0f);
    }
    float normalized = metal_normalize_f32(value, parameters);
    uint lutIndex = min(
        parameters.lutCount - 1u,
        uint(normalized * float(parameters.lutCount - 1u))
    );
    return lut[lutIndex];
}

kernel void metal_range_u32(
    device const uint *values [[buffer(0)]],
    device atomic_uint *valueRange [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    uint value = values[index];
    atomic_fetch_min_explicit(&valueRange[0], value, memory_order_relaxed);
    atomic_fetch_max_explicit(&valueRange[1], value, memory_order_relaxed);
}

inline uint metal_ordered_float_bits(float value) {
    uint bits = as_type<uint>(value);
    return (bits & 0x80000000u) != 0u ? ~bits : bits ^ 0x80000000u;
}

kernel void metal_range_f32(
    device const float *values [[buffer(0)]],
    device atomic_uint *orderedRange [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    if (index >= count) return;
    float value = values[index];
    if (!isfinite(value)) return;
    uint ordered = metal_ordered_float_bits(value);
    atomic_fetch_min_explicit(&orderedRange[0], ordered, memory_order_relaxed);
    atomic_fetch_max_explicit(&orderedRange[1], ordered, memory_order_relaxed);
}

kernel void metal_histogram_u32(
    device const uint *values [[buffer(0)]],
    device atomic_uint *bins [[buffer(1)]],
    constant MetalDisplayParameters &parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint count = parameters.rows * parameters.cols;
    if (index >= count) return;
    float normalized = metal_normalize_u32(values[index], parameters);
    uint bin = min(255u, uint(normalized * 256.0f));
    atomic_fetch_add_explicit(&bins[bin], 1u, memory_order_relaxed);
}

kernel void metal_histogram_f32(
    device const float *values [[buffer(0)]],
    device atomic_uint *bins [[buffer(1)]],
    constant MetalFloatDisplayParameters &parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint count = parameters.rows * parameters.cols;
    if (index >= count) return;
    float value = values[index];
    if (!isfinite(value)) return;
    float normalized = metal_normalize_f32(value, parameters);
    uint bin = min(255u, uint(normalized * 256.0f));
    atomic_fetch_add_explicit(&bins[bin], 1u, memory_order_relaxed);
}
