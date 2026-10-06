// Exact signed-mask decomposition for large diffraction cameras.
extern "C" __global__ void mask_leaves(
    const int* mask, int* leaves, unsigned int* pixels, int* coefficients,
    unsigned int* counts, unsigned int rows, unsigned int cols
) {
    unsigned int tile_cols = (cols + 7) / 8;
    unsigned int tile = blockIdx.x, lane = threadIdx.x;
    unsigned int row = (tile / tile_cols) * 8 + lane / 8;
    unsigned int col = (tile % tile_cols) * 8 + lane % 8;
    bool valid0 = row < rows && col < cols;
    bool valid1 = row + 4 < rows && col < cols;
    int value0 = valid0 ? mask[row * cols + col] : 0;
    int value1 = valid1 ? mask[(row + 4) * cols + col] : 0;
    // One warp votes on both halves of a tile. Padding votes for zero, matching
    // the reference tie order (zero, positive, negative) exactly.
    unsigned int zero = __popc(__ballot_sync(0xffffffff, value0 == 0))
                      + __popc(__ballot_sync(0xffffffff, value1 == 0));
    unsigned int positive = __popc(__ballot_sync(0xffffffff, value0 == 1))
                          + __popc(__ballot_sync(0xffffffff, value1 == 1));
    unsigned int negative = __popc(__ballot_sync(0xffffffff, value0 == -1))
                          + __popc(__ballot_sync(0xffffffff, value1 == -1));
    int base = positive > zero ? 1 : 0;
    if (negative > (base ? positive : zero)) base = -1;
    if (!lane) leaves[tile] = base;
    unsigned int active0 = __ballot_sync(0xffffffff, valid0 && value0 != base);
    unsigned int active1 = __ballot_sync(0xffffffff, valid1 && value1 != base);
    unsigned int count0 = __popc(active0), total = count0 + __popc(active1);
    unsigned int start = 0;
    if (!lane && total) start = atomicAdd(counts + 1, total);
    start = __shfl_sync(0xffffffff, start, 0);
    unsigned int lower = (1u << lane) - 1;
    if (valid0 && value0 != base) {
        unsigned int at = start + __popc(active0 & lower);
        pixels[at] = row * cols + col;
        coefficients[at] = value0 - base;
    }
    if (valid1 && value1 != base) {
        unsigned int at = start + count0 + __popc(active1 & lower);
        pixels[at] = (row + 4) * cols + col;
        coefficients[at] = value1 - base;
    }
}

extern "C" __global__ void mask_roots(
    const int* leaves, unsigned int* fields, int* coefficients,
    unsigned int* counts, unsigned int tile_rows, unsigned int tile_cols
) {
    unsigned int root = blockIdx.x, lane = threadIdx.x;
    unsigned int root_cols = (tile_cols + 3) / 4;
    unsigned int row = (root / root_cols) * 4 + lane / 4;
    unsigned int col = (root % root_cols) * 4 + lane % 4;
    int value = (lane < 16 && row < tile_rows && col < tile_cols)
                    ? leaves[row * tile_cols + col] : 0;
    unsigned int zero = __popc(__ballot_sync(0xffffffff, lane < 16 && value == 0));
    unsigned int positive = __popc(__ballot_sync(0xffffffff, lane < 16 && value == 1));
    unsigned int negative = __popc(__ballot_sync(0xffffffff, lane < 16 && value == -1));
    int base = positive > zero ? 1 : 0;
    if (negative > (base ? positive : zero)) base = -1;
    if (!lane && base) {
        unsigned int at = atomicAdd(counts, 1u);
        fields[at] = tile_rows * tile_cols + root;
        coefficients[at] = base;
    }
    if (lane < 16 && row < tile_rows && col < tile_cols && value != base) {
        unsigned int at = atomicAdd(counts, 1u);
        fields[at] = row * tile_cols + col;
        coefficients[at] = value - base;
    }
}
