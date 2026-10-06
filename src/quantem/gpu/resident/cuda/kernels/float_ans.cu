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
    if (lane >= FLOAT_LANES || !mask[lane / 2] || models[lane] >= 253u) return;
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
    const u8* mask, const float* dark, float* output, u32 offset, u32 corrected) {
    __shared__ float partials[8];
    u32 local = threadIdx.x, lane = local % 32, group = local / 32, scan = blockIdx.x;
    float sum = 0, residual = 0;
    for (u32 pixel = local; pixel < FLOAT_LANES / 2; pixel += 128) {
        if (mask[pixel]) {
            u32 word = float_lane(payload,offsets,models,decoded,pixel*2,scan)
                     | (float_lane(payload,offsets,models,decoded,pixel*2+1,scan) << 16);
            float value = __uint_as_float(word);
            if (corrected) value -= dark[pixel];
            float_accumulate(value, sum, residual);
        }
    }
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
        output[offset+scan]=total+compensation;
    }
}
