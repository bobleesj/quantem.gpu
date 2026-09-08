// Includes the package-owned rANS Reader and compact_sample implementations.
// Each y block owns one acquisition; source bytes stay in their original buffers.
static const int DESCRIPTOR_WORDS = 20;

__device__ Reader series_reader(const u64* d, u64 stream) {
    return Reader((const u8*)d[9], (const u64*)d[10], (const u32*)d[11],
                  (const u32*)d[12], (const u16*)d[13], (const u16*)d[14],
                  (const u16*)d[15], (const u8*)d[16], stream, (u32)d[3]);
}

__device__ u32 series_count(const u64* d, u64 scan, u32 pixel, u32 pixels) {
    if (d[0] == 0) {
        u64 index = scan * pixels + pixel;
        return d[1] == 1 ? ((const u8*)d[9])[index]
                         : ((const u16*)d[9])[index];
    }
    if (d[0] == 2) {
        u64 stream = (scan / d[2]) * pixels + pixel;
        return packed_count((const u32*)d[9], (const u64*)d[10],
                            (const u8*)d[11], stream, scan % d[2]);
    }
    u32 raw = ((const u32*)d[19])[pixel];
    if (raw != 0xffffffffu) return raw;
    const u64* shards = (const u64*)d[17];
    u64 shard = scan / d[4];
    return compact_sample((const u32*)shards[shard * 2],
                          (const u32*)shards[shard * 2 + 1],
                          d[6], d[7], d[8], d[5], pixel, scan % d[4]);
}

__device__ const u64* series_chunk(const u64* d, u64 index, u32 key) {
    const u64* chunks=(const u64*)d[9];
    u32 left=0,right=d[11];
    while (left+1<right) {
        u32 middle=left+(right-left)/2;
        if (chunks[u64(middle)*6+key]<=index) left=middle;
        else right=middle;
    }
    return chunks+u64(left)*6;
}

template<typename Output>
__device__ void series_sum(const u64* descriptors, const u8* mask,
                          Output* output, u32* errors, u64 scans, u32 pixels) {
    u32 acquisition = blockIdx.y;
    const u64* d = descriptors + (u64)acquisition * DESCRIPTOR_WORDS;
    const u8* valid = (const u8*)d[18];
    Output* image = output + (u64)acquisition * scans;
    if (d[0] == 4) {
        u64 stream=u64(blockIdx.x)*blockDim.x+threadIdx.x;
        if (stream>=d[12]) return;
        const u64* chunk=series_chunk(d,stream,2);
        u32 local=stream-chunk[2],pixel=local%pixels;
        if (!mask[pixel] || !valid[pixel]) return;
        StreamReader reader((const u8*)chunk[3],(const u32*)chunk[4],(const u8*)chunk[5],(const u32*)d[10],local);
        u32 first=(local/pixels)*d[2],count=min(d[2],chunk[1]-first);
        for (u32 scan=0;scan<count && reader.valid;++scan)
            atomicAdd(image+chunk[0]+first+scan,Output(reader.next()));
        if (!reader.finished()) atomicOr(errors,1u);
        return;
    }
    if (d[0] == 1) {
        u64 stream = (u64)blockIdx.x * blockDim.x + threadIdx.x;
        u64 streams = ((scans + d[2] - 1) / d[2]) * pixels;
        if (stream >= streams) return;
        u32 pixel = stream % pixels;
        if (!mask[pixel] || !valid[pixel]) return;
        Reader reader = series_reader(d, stream);
        u64 first = (stream / pixels) * d[2];
        u32 count = min(d[2], scans - first);
        for (u32 scan = 0; scan < count && reader.valid; ++scan)
            atomicAdd(image + first + scan, (Output)reader.next());
        if (!reader.finished()) atomicOr(errors, 1u);
        return;
    }
    u64 scan = blockIdx.x;
    if (scan >= scans) return;
    u64 value = 0;
    for (u32 pixel = threadIdx.x; pixel < pixels; pixel += blockDim.x)
        if (mask[pixel] && valid[pixel])
            value += series_count(d, scan, pixel, pixels);
    __shared__ u64 sums[256];
    sums[threadIdx.x] = value;
    __syncthreads();
    for (u32 stride = 128; stride; stride >>= 1) {
        if (threadIdx.x < stride) sums[threadIdx.x] += sums[threadIdx.x + stride];
        __syncthreads();
    }
    if (!threadIdx.x) image[scan] = (Output)sums[0];
}

extern "C" __global__ void series_sum_u32(
    const u64* d, const u8* mask, u32* out, u32* errors, u64 scans, u32 pixels
) { series_sum(d, mask, out, errors, scans, pixels); }

extern "C" __global__ void series_sum_u64(
    const u64* d, const u8* mask, u64* out, u32* errors, u64 scans, u32 pixels
) { series_sum(d, mask, out, errors, scans, pixels); }

template<typename Output>
__device__ void series_frame(const u64* descriptors, Output* output,
                            u32* errors, u64 scan, u32 pixels) {
    u32 pixel = blockIdx.x * blockDim.x + threadIdx.x;
    if (pixel >= pixels) return;
    u32 acquisition = blockIdx.y;
    const u64* d = descriptors + (u64)acquisition * DESCRIPTOR_WORDS;
    u32 value;
    if (d[0] == 4) {
        const u64* chunk=series_chunk(d,scan,0);
        u32 local=scan-chunk[0];
        StreamReader reader((const u8*)chunk[3],(const u32*)chunk[4],(const u8*)chunk[5],(const u32*)d[10],(local/d[2])*pixels+pixel);
        value=0;
        for (u32 i=0;i<=local%d[2];++i) value=reader.next();
        if (!reader.valid) atomicOr(errors,1u);
    } else     if (d[0] == 1) {
        Reader reader = series_reader(d, (scan / d[2]) * pixels + pixel);
        u32 within = scan % d[2];
        if (reader.raw) {
            reader.cursor += (u64)within * 2;
            value = reader.next();
        } else {
            value = 0;
            for (u32 step = 0; step <= within && reader.valid; ++step)
                value = reader.next();
        }
        if (!reader.valid) atomicOr(errors, 1u);
    } else {
        value = series_count(d, scan, pixel, pixels);
    }
    output[(u64)acquisition * pixels + pixel] = (Output)value;
}

extern "C" __global__ void series_frame_u8(
    const u64* d, u8* out, u32* errors, u64 scan, u32 pixels
) { series_frame(d, out, errors, scan, pixels); }

extern "C" __global__ void series_frame_u16(
    const u64* d, u16* out, u32* errors, u64 scan, u32 pixels
) { series_frame(d, out, errors, scan, pixels); }
