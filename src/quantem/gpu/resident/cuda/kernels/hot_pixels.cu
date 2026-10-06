
template <typename T>
__device__ void correct_one(
    T* frames, const unsigned char* valid, const int* bad,
    int bad_count, int height, int width, unsigned long long item,
    bool use_median
) {
    int bad_slot = item % bad_count;
    unsigned long long frame = item / bad_count;
    int pixel = bad[bad_slot];
    if (!use_median) {
        frames[frame * height * width + pixel] = T(0);
        return;
    }
    int row = pixel / width, column = pixel % width;
    unsigned int values[8];
    int count = 0;
    for (int dr = -1; dr <= 1; ++dr) {
        for (int dc = -1; dc <= 1; ++dc) {
            int rr = row + dr, cc = column + dc;
            if ((dr == 0 && dc == 0) || rr < 0 || rr >= height ||
                cc < 0 || cc >= width) continue;
            int neighbor = rr * width + cc;
            if (!valid[neighbor]) continue;
            values[count++] = frames[frame * height * width + neighbor];
        }
    }
    for (int i = 1; i < count; ++i) {
        unsigned int value = values[i];
        int j = i - 1;
        while (j >= 0 && values[j] > value) {
            values[j + 1] = values[j];
            --j;
        }
        values[j + 1] = value;
    }
    unsigned int result = 0;
    if (count & 1) result = values[count / 2];
    else if (count) result = (values[count / 2 - 1] + values[count / 2]) / 2;
    frames[frame * height * width + pixel] = T(result);
}

extern "C" __global__ void hot_median_u8(
    unsigned char* frames, const unsigned char* valid, const int* bad,
    int bad_count, int height, int width, unsigned long long total
) {
    unsigned long long item = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < total) correct_one(frames, valid, bad, bad_count, height, width, item, true);
}
extern "C" __global__ void hot_median_u16(
    unsigned short* frames, const unsigned char* valid, const int* bad,
    int bad_count, int height, int width, unsigned long long total
) {
    unsigned long long item = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < total) correct_one(frames, valid, bad, bad_count, height, width, item, true);
}
extern "C" __global__ void hot_zero_u8(
    unsigned char* frames, const unsigned char* valid, const int* bad,
    int bad_count, int height, int width, unsigned long long total
) {
    unsigned long long item = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < total) correct_one(frames, valid, bad, bad_count, height, width, item, false);
}
extern "C" __global__ void hot_zero_u16(
    unsigned short* frames, const unsigned char* valid, const int* bad,
    int bad_count, int height, int width, unsigned long long total
) {
    unsigned long long item = (unsigned long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (item < total) correct_one(frames, valid, bad, bad_count, height, width, item, false);
}
