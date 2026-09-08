// Appended to the unchanged production float shader for this audit only.
// Prefix offsets stay on the GPU; no CPU descriptor scan or codec fallback.
kernel void audit_local_offsets(device uint4* descriptors [[buffer(0)]],
                                device uint* totals [[buffer(1)]],
                                uint block [[thread_position_in_grid]],
                                uint local [[thread_index_in_threadgroup]],
                                uint group [[threadgroup_position_in_grid]],
                                ushort lane [[thread_index_in_simdgroup]],
                                ushort simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint sums[8];
    uint words = descriptors[block].y * 4;
    uint offset = simd_prefix_exclusive_sum(words);
    uint sum = simd_sum(words);
    if (lane == 0) sums[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint prior = 0; prior < simd; ++prior) offset += sums[prior];
    descriptors[block].w = offset;
    if (local == 255) totals[group] = offset + words;
}

kernel void audit_group_offsets(device const uint* totals [[buffer(0)]],
                                device uint* offsets [[buffer(1)]],
                                constant uint& count [[buffer(2)]],
                                uint local [[thread_index_in_threadgroup]],
                                ushort lane [[thread_index_in_simdgroup]],
                                ushort simd [[simdgroup_index_in_threadgroup]]) {
    threadgroup uint sums[8];
    uint words = local < count ? totals[local] : 0;
    uint offset = simd_prefix_exclusive_sum(words);
    uint sum = simd_sum(words);
    if (lane == 0) sums[simd] = sum;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint prior = 0; prior < simd; ++prior) offset += sums[prior];
    if (local < count) offsets[local] = offset;
    if (local == count - 1) offsets[count] = offset + words;
}

kernel void audit_add_offsets(device uint4* descriptors [[buffer(0)]],
                              device const uint* offsets [[buffer(1)]],
                              uint block [[thread_position_in_grid]]) {
    descriptors[block].w += offsets[block / 256];
}

kernel void audit_verify(device const uint* original [[buffer(0)]],
                         device const uint* payload [[buffer(1)]],
                         device const uint4* descriptors [[buffer(2)]],
                         device atomic_uint* mismatch [[buffer(3)]],
                         uint index [[thread_position_in_grid]]) {
    if (empad_word(payload, descriptors, index) != original[index])
        atomic_fetch_add_explicit(mismatch, 1u, memory_order_relaxed);
}
