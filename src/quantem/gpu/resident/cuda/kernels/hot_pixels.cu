
// One thread corrects one flagged pixel of one frame. Its neighbors are the
// planned valid pixels (quantem.gpu.resident.hot_pixels.median_plan): flat
// detector indices neighbors[offsets[slot]:offsets[slot + 1]], at least one.
// The two middle values are selected by counting ranks, so any window size
// needs no local array; their mean rounds half to even, as the CPU reference.
template <typename T>
__device__ void correct_one(
    T* frames, const int* offsets, const int* neighbors, const int* bad,
    int bad_count, int height, int width, unsigned long long item,
    bool use_median
) {
    int slot = item % bad_count;
    unsigned long long base = (item / bad_count) * (unsigned long long)height * width;
    int pixel = bad[slot];
    if (!use_median) {
        frames[base + pixel] = T(0);
        return;
    }
    int start = offsets[slot], stop = offsets[slot + 1];
    int count = stop - start;
    int low_rank = (count - 1) / 2, high_rank = count / 2;
    unsigned int low = 0, high = 0;
    for (int i = start; i < stop; ++i) {
        unsigned int value = frames[base + neighbors[i]];
        int less = 0, equal = 0;
        for (int j = start; j < stop; ++j) {
            unsigned int other = frames[base + neighbors[j]];
            less += other < value;
            equal += other == value;
        }
        if (less <= low_rank && low_rank < less + equal) low = value;
        if (less <= high_rank && high_rank < less + equal) high = value;
    }
    unsigned long long total = (unsigned long long)low + high;
    unsigned long long lower_half = total >> 1;
    frames[base + pixel] = T(lower_half + (total & lower_half & 1ull));
}

#define HOT_PIXEL_KERNEL(name, type, median)                                         \
extern "C" __global__ void name(                                                     \
    type* frames, const int* offsets, const int* neighbors, const int* bad,          \
    int bad_count, int height, int width, unsigned long long total                   \
) {                                                                                  \
    unsigned long long item = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x; \
    if (item < total) correct_one(frames, offsets, neighbors, bad, bad_count, height, width, item, median); \
}

HOT_PIXEL_KERNEL(hot_median_u8, unsigned char, true)
HOT_PIXEL_KERNEL(hot_median_u16, unsigned short, true)
HOT_PIXEL_KERNEL(hot_median_u32, unsigned int, true)
HOT_PIXEL_KERNEL(hot_zero_u8, unsigned char, false)
HOT_PIXEL_KERNEL(hot_zero_u16, unsigned short, false)
HOT_PIXEL_KERNEL(hot_zero_u32, unsigned int, false)
