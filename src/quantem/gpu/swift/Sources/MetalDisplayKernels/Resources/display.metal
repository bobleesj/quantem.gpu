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

inline float metal_normalize_u32(
    uint value,
    constant MetalDisplayParameters &parameters
) {
    uint high = max(parameters.low, parameters.high);
    uint clipped = clamp(value, parameters.low, high);
    float span = max(1.0f, float(high - parameters.low));
    float shifted = float(clipped - parameters.low);
    if (parameters.scaleMode == 1u) {
        return log(1.0f + shifted) / log(1.0f + span);
    }
    return shifted / span;
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
        uint(normalized * float(parameters.lutCount - 1u) + 0.5f)
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

kernel void metal_histogram_u32(
    device const uint *values [[buffer(0)]],
    device atomic_uint *bins [[buffer(1)]],
    constant MetalDisplayParameters &parameters [[buffer(2)]],
    uint index [[thread_position_in_grid]]
) {
    uint count = parameters.rows * parameters.cols;
    if (index >= count) return;
    float normalized = metal_normalize_u32(values[index], parameters);
    uint bin = min(255u, uint(normalized * 255.0f + 0.5f));
    atomic_fetch_add_explicit(&bins[bin], 1u, memory_order_relaxed);
}
