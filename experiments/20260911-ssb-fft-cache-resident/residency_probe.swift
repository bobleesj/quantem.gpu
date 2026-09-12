import Metal
// Empirical concurrent-residency test: every threadgroup increments `arrived`
// and then spins until it observes G arrivals or a bounded timeout. All groups
// observing G means G groups were simultaneously resident.
let src = """
#include <metal_stdlib>
using namespace metal;
kernel void k(device atomic_uint *arrived [[buffer(0)]],
              device uint *saw [[buffer(1)]],
              constant uint &target [[buffer(2)]],
              threadgroup float *tg [[threadgroup(0)]],
              uint gid [[threadgroup_position_in_grid]],
              uint lid [[thread_index_in_threadgroup]]) {
    tg[lid] = float(lid);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (lid == 0u) {
        atomic_fetch_add_explicit(arrived, 1u, memory_order_relaxed);
        uint seen = 0u;
        for (uint i = 0u; i < (1u << 20); ++i) {
            seen = atomic_load_explicit(arrived, memory_order_relaxed);
            if (seen >= target) break;
        }
        saw[gid] = seen + (uint)tg[lid];
    }
    // Keep every thread of the group alive until the spin finishes.
    threadgroup_barrier(mem_flags::mem_threadgroup);
    tg[lid] += 1.0f;
}
"""
let device = MTLCreateSystemDefaultDevice()!
let lib = try device.makeLibrary(source: src, options: nil)
let pso = try device.makeComputePipelineState(function: lib.makeFunction(name: "k")!)
let queue = device.makeCommandQueue()!
let counter = device.makeBuffer(length: 4, options: .storageModeShared)!
let saw = device.makeBuffer(length: 4 * 65536, options: .storageModeShared)!
func test(groups: Int, threads: Int, tgBytes: Int) -> Bool {
    counter.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    let cb = queue.makeCommandBuffer()!
    let enc = cb.makeComputeCommandEncoder()!
    enc.setComputePipelineState(pso)
    enc.setBuffer(counter, offset: 0, index: 0)
    enc.setBuffer(saw, offset: 0, index: 1)
    var t = UInt32(groups)
    enc.setBytes(&t, length: 4, index: 2)
    enc.setThreadgroupMemoryLength(max(tgBytes, threads * 4), index: 0)
    enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1))
    enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
    let p = saw.contents().bindMemory(to: UInt32.self, capacity: groups)
    return (0..<groups).allSatisfy { p[$0] >= UInt32(groups) }
}
for (threads, tg) in [(256, 27000), (256, 16384), (256, 1024), (64, 4096)] {
    var lo = 1, hi = 1024
    // Find the largest group count where all groups were simultaneously resident.
    while lo < hi {
        let mid = (lo + hi + 1) / 2
        if test(groups: mid, threads: threads, tgBytes: tg) { lo = mid } else { hi = mid - 1 }
    }
    print("threads/group \(threads)  tg mem \(tg) B  -> max concurrently resident groups: \(lo)  (\(lo * threads) threads, \(lo * tg / 1024) KB tg mem, \(Double(lo) / 10.0) groups/core if 10 cores)")
}
