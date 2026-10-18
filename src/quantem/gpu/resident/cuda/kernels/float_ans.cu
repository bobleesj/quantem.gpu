// IEEE bit lanes, with literal/constant streams directly addressable by scan.
// Entropy and sparse-event streams retain the established StreamReader recurrence.
extern "C" __global__ void float_ans_direct(
    const u8* payload, const u32* offsets, const u8* models,
    u16* output, u32 first, u32 count) {
    u32 at = blockIdx.x * blockDim.x + threadIdx.x;
    if (at >= count * FLOAT_LANES) return;
    u32 lane = at % FLOAT_LANES, model = models[lane];
    if (model < 253u) return;
    u32 position = offsets[lane] + (model == 254u ? 2 * (first + at / FLOAT_LANES) : 0);
    output[at] = model == 253u ? 0 : payload[position] | (u32(payload[position + 1]) << 8);
}

extern "C" __global__ void float_ans_entropy(
    const u8* payload, const u32* offsets, const u8* models,
    const u32* decoding, u16* output, u32* errors,
    u32 scans, u32 first, u32 count) {
    u32 lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= FLOAT_LANES || models[lane] >= 253u) return;
    StreamReader reader(payload, offsets, models, decoding, lane);
    for (u32 scan = 0; scan < first + count; ++scan) {
        u32 value = reader.next();
        if (scan >= first) output[(scan - first) * FLOAT_LANES + lane] = value;
    }
    if (!reader.valid || (first + count == scans && !reader.finished())) atomicOr(errors, 1u);
}

extern "C" __global__ void float_ans_selected_entropy(
    const u8* payload, const u32* offsets, const u8* models,
    const u32* decoding, u16* output, u32* errors, const u8* mask, u32 scans) {
    u32 lane = blockIdx.x * blockDim.x + threadIdx.x;
    if (lane >= FLOAT_LANES || !mask[lane / FLOAT_WORDS] || models[lane] >= 253u) return;
    StreamReader reader(payload, offsets, models, decoding, lane);
    for (u32 scan = 0; scan < scans; ++scan) output[scan * FLOAT_LANES + lane] = reader.next();
    if (!reader.finished()) atomicOr(errors, 1u);
}

__device__ u32 float_lane(const u8* payload, const u32* offsets, const u8* models,
                         const u16* decoded, u32 lane, u32 scan) {
    u32 model = models[lane];
    if (model < 253u) return decoded[scan * FLOAT_LANES + lane];
    if (model == 253u) return 0;
    u32 at = offsets[lane] + (model == 254u ? scan * 2 : 0);
    return payload[at] | (u32(payload[at + 1]) << 8);
}

__device__ void float_accumulate(float value, float& sum, float& residual) {
    float next = sum + value;
    if (isfinite(value) && isfinite(sum) && isfinite(next))
        residual += fabsf(sum) >= fabsf(value) ? (sum - next) + value : (value - next) + sum;
    else residual = 0;
    sum = next;
}

extern "C" __global__ void float_ans_detector(
    const u8* payload, const u32* offsets, const u8* models, const u16* decoded,
    const u8* mask, const float* dark, float* output, u32 offset, u32 corrected
#if FLOAT8_FORMAT >= 0
    , u32* errors
#endif
    ) {
    __shared__ float partials[8];
    u32 local = threadIdx.x, lane = local % 32, group = local / 32, scan = blockIdx.x;
#if FLOAT8_FORMAT == 4
    // Corrected E8M0 uses range normalization; ordinary sums count code 0 separately.
    __shared__ u32 maxima[4], tiny_partials[4];
    u32 scale = 0;
    u32 tiny_count = 0;
    if (corrected) {
        for (u32 pixel = local; pixel < FLOAT_LANES; pixel += 128) {
            if (!mask[pixel]) continue;
            scale = max(scale, float_lane(payload, offsets, models, decoded, pixel, scan) & 255u);
            scale = max(scale, (__float_as_uint(dark[pixel]) >> 23) & 255u);
        }
        for (u32 distance = 16; distance; distance >>= 1)
            scale = max(scale, __shfl_down_sync(0xffffffffu, scale, distance));
        if (lane == 0) maxima[group] = scale;
        __syncthreads();
        scale = max(max(maxima[0], maxima[1]), max(maxima[2], maxima[3]));
    }
#endif
    float sum = 0, residual = 0;
    for (u32 pixel = local; pixel < FLOAT_LANES / FLOAT_WORDS; pixel += 128) {
        if (mask[pixel]) {
#if FLOAT8_FORMAT >= 0
            u32 word = float_lane(payload,offsets,models,decoded,pixel,scan);
            if (word > 255u) atomicOr(errors, 1u);
#if FLOAT8_FORMAT == 4
            if (!corrected && word == 0u) { ++tiny_count; continue; }
            float value = !corrected ? __uint_as_float(float8_bits(word & 255u, 4))
                : word == 255u ? __uint_as_float(0x7fc00000u)
                : ldexpf(1.0f, int(word & 255u) - int(scale));
#else
            float value = __uint_as_float(float8_bits(word & 255u, FLOAT8_FORMAT));
#endif
#else
            u32 word = float_lane(payload,offsets,models,decoded,pixel*2,scan)
                     | (float_lane(payload,offsets,models,decoded,pixel*2+1,scan) << 16);
            float value = __uint_as_float(word);
#endif
#if FLOAT8_FORMAT == 4
            if (corrected) value -= float8_scale(dark[pixel], 127 - int(scale));
#else
            if (corrected) value -= dark[pixel];
#endif
            float_accumulate(value, sum, residual);
        }
    }
#if FLOAT8_FORMAT == 4
    for (u32 distance = 16; distance; distance >>= 1)
        tiny_count += __shfl_down_sync(0xffffffffu, tiny_count, distance);
    if (lane == 0) tiny_partials[group] = tiny_count;
#endif
    float total = 0, compensation = 0;
    for (u32 i = 0; i < 32; ++i) {
        float high = __shfl_sync(0xffffffffu,sum,i), low = __shfl_sync(0xffffffffu,residual,i);
        if (lane == 0) { float_accumulate(high,total,compensation); float_accumulate(low,total,compensation); }
    }
    if (lane == 0) { partials[group*2]=total; partials[group*2+1]=compensation; }
    __syncthreads();
    if (local == 0) {
        total=0; compensation=0;
        for (u32 i=0;i<8;++i) float_accumulate(partials[i],total,compensation);
#if FLOAT8_FORMAT == 4
        float value = total + compensation;
        const u32 tiny = tiny_partials[0] + tiny_partials[1] + tiny_partials[2] + tiny_partials[3];
        if (corrected) value = float8_scale(value, int(scale) - 127);
        else if (tiny && (__float_as_uint(value) & 0x7fffffffu) < 0x3f800000u)
            value = float8_scale(float8_scale(value, 1) + float(tiny) * 0x1p-126f, -1);
        output[offset+scan] = value;
#else
        output[offset+scan] = total + compensation;
#endif
    }
}
