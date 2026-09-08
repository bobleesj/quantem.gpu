// Compact changed columns across every model context, without a host acquisition loop.
using i8 = signed char;
using u32 = unsigned int;
extern "C" __global__ void count_groups(
    const int* sorted_detectors, const int* starts, const i8* difference,
    int* counts, int* group_counts) {
    const int segment = blockIdx.x, base = segment/2, sign = segment%2 == 0 ? 1 : -1;
    int count = 0;
    for (int at = starts[base]+threadIdx.x; at < starts[base+1]; at += blockDim.x)
        count += difference[sorted_detectors[at]] == sign;
    count = __reduce_add_sync(0xffffffffu, count);
    __shared__ int warp_counts[8];
    if ((threadIdx.x & 31) == 0) warp_counts[threadIdx.x/32] = count;
    __syncthreads();
    if (threadIdx.x == 0) {
        int total = 0;
        for (int warp = 0; warp < 8; ++warp) total += warp_counts[warp];
        counts[segment] = total; group_counts[segment] = (total+31)/32;
    }
}
extern "C" __global__ void scatter_groups(
    const int* sorted_detectors, const int* starts, const i8* difference,
    const int* counts, const int* group_offsets, int model_count,
    int* selected, i8* coefficients, int* group_context, int* group_model) {
    const int segment = blockIdx.x, base = segment/2, sign = segment%2 == 0 ? 1 : -1;
    const int count = counts[segment];
    if (!count) return;
    const int first_group = group_offsets[segment], groups = (count+31)/32;
    for (int group = threadIdx.x; group < groups; group += blockDim.x) {
        group_context[first_group+group] = base/model_count;
        group_model[first_group+group] = base%model_count;
    }
    for (int at = count+threadIdx.x; at < groups*32; at += blockDim.x) {
        selected[first_group*32+at] = 0; coefficients[first_group*32+at] = 0;
    }
    __shared__ int written;
    if (threadIdx.x == 0) written = 0;
    __syncthreads();
    const int begin = starts[base], end = starts[base+1], lane = threadIdx.x & 31;
    for (int tile = begin; tile < end; tile += blockDim.x) {
        const int at = tile+threadIdx.x;
        const int detector = at < end ? sorted_detectors[at] : 0;
        const bool active = at < end && difference[detector] == sign;
        const u32 ballot = __ballot_sync(0xffffffffu, active);
        int first = 0;
        if (lane == 0 && ballot) first = atomicAdd(&written, __popc(ballot));
        first = __shfl_sync(0xffffffffu, first, 0);
        if (active) {
            const int rank = first+__popc(ballot & ((1u << lane)-1));
            selected[first_group*32+rank] = detector; coefficients[first_group*32+rank] = sign;
        }
    }
}
