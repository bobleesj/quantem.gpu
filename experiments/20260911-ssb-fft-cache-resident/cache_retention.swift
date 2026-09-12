// Apple M5 write->read retention microbenchmark for the SSB fftA intermediate.
//
// Emulates one objective's per-batch traffic: pass 1 streams X bytes of "G"
// from a large buffer and writes X bytes to intermediate slot k % 2; pass 2
// reads slot k % 2 back. Reports GPU time per byte of intermediate for
//   split : two serial dispatches per batch (current production topology)
//   fused : one dispatch per batch whose first groups run pass 1 on batch k and
//           whose remaining groups run pass 2 on batch k - 1 (candidate)
// If the intermediate is served from on-chip cache, the fused time approaches
// the stream-read-only cost (X bytes) instead of 3X bytes of device memory traffic.
import Foundation
import Metal

let source = """
#include <metal_stdlib>
using namespace metal;
kernel void pass1(device const float4 *stream [[buffer(0)]],
                  device float4 *slot [[buffer(1)]],
                  constant uint &count [[buffer(2)]],
                  uint i [[thread_position_in_grid]]) {
    if (i >= count) return;
    float4 v = stream[i];
    slot[i] = v * 1.0001f + 1.0f;
}
kernel void pass2(device const float4 *slot [[buffer(0)]],
                  device float *sink [[buffer(1)]],
                  constant uint &count [[buffer(2)]],
                  uint i [[thread_position_in_grid]],
                  uint stride [[threads_per_grid]],
                  uint gid [[threadgroup_position_in_grid]],
                  uint lid [[thread_index_in_threadgroup]]) {
    float acc = 0.0f;
    for (uint j = i; j < count; j += stride) {
        float4 v = slot[j];
        acc += v.x + v.y + v.z + v.w;
    }
    acc = simd_sum(acc);
    if (lid == 0u) sink[gid] = acc;
}
kernel void fused(device const float4 *stream [[buffer(0)]],
                  device float4 *slot_write [[buffer(1)]],
                  device const float4 *slot_read [[buffer(2)]],
                  device float *sink [[buffer(3)]],
                  constant uint &count [[buffer(4)]],
                  constant uint &write_groups [[buffer(5)]],
                  constant uint &read_stride [[buffer(6)]],
                  uint gid [[threadgroup_position_in_grid]],
                  uint lid [[thread_index_in_threadgroup]]) {
    if (gid < write_groups) {
        uint i = gid * 256u + lid;
        if (i < count) slot_write[i] = stream[i] * 1.0001f + 1.0f;
    } else {
        uint i = (gid - write_groups) * 256u + lid;
        float acc = 0.0f;
        for (uint j = i; j < count; j += read_stride) {
            float4 v = slot_read[j];
            acc += v.x + v.y + v.z + v.w;
        }
        acc = simd_sum(acc);
        if (lid == 0u) sink[gid - write_groups] = acc;
    }
}
"""

let device = MTLCreateSystemDefaultDevice()!
let library = try device.makeLibrary(source: source, options: nil)
let p1 = try device.makeComputePipelineState(function: library.makeFunction(name: "pass1")!)
let p2 = try device.makeComputePipelineState(function: library.makeFunction(name: "pass2")!)
let pf = try device.makeComputePipelineState(function: library.makeFunction(name: "fused")!)
let queue = device.makeCommandQueue()!
let streamBytes = 3 << 30  // 3 GiB streamed per timed objective
let stream = device.makeBuffer(length: streamBytes, options: .storageModePrivate)!
let slots = (0..<2).map { _ in device.makeBuffer(length: 64 << 20, options: .storageModePrivate)! }
let sink = device.makeBuffer(length: 1 << 20, options: .storageModePrivate)!
// Commit every page of the private buffers; untouched private pages are not
// backed by device memory and would read far faster than any real stream.
do {
    let cb = queue.makeCommandBuffer()!
    let blit = cb.makeBlitCommandEncoder()!
    blit.fill(buffer: stream, range: 0..<streamBytes, value: 0x3f)
    for slot in slots { blit.fill(buffer: slot, range: 0..<slot.length, value: 0x3f) }
    blit.fill(buffer: sink, range: 0..<sink.length, value: 0)
    blit.endEncoding(); cb.commit(); cb.waitUntilCompleted()
}
let xs = CommandLine.arguments.dropFirst().compactMap { Double($0) }
let sizesMB: [Double] = xs.isEmpty ? [0.5, 1, 2, 4, 6, 8, 12, 16, 24, 32, 48, 64] : Array(xs)

func run(_ label: String, sizeMB: Double, fused: Bool, repeats: Int = 5) {
    let bytes = Int(sizeMB * 1048576)
    var count = UInt32(bytes / 16)
    let batches = streamBytes / bytes
    var times: [Double] = []
    for _ in 0..<repeats {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        if fused {
            enc.setComputePipelineState(pf)
            var writeGroups = UInt32((Int(count) + 255) / 256)
            let readGroups = min(4096, (Int(count) + 255) / 256)
            for k in 0..<(batches + 1) {
                if k < batches { enc.setBuffer(stream, offset: k * bytes, index: 0) }
                enc.setBuffer(slots[k % 2], offset: 0, index: 1)
                enc.setBuffer(slots[(k + 1) % 2], offset: 0, index: 2)
                enc.setBuffer(sink, offset: 0, index: 3)
                enc.setBytes(&count, length: 4, index: 4)
                var wg = (k < batches) ? writeGroups : 0
                enc.setBytes(&wg, length: 4, index: 5)
                let rg = (k > 0) ? readGroups : 0
                var rs = UInt32(max(rg, 1) * 256)
                enc.setBytes(&rs, length: 4, index: 6)
                enc.dispatchThreadgroups(MTLSize(width: Int(wg) + rg, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                _ = writeGroups
            }
        } else {
            for k in 0..<batches {
                enc.setComputePipelineState(p1)
                enc.setBuffer(stream, offset: k * bytes, index: 0)
                enc.setBuffer(slots[k % 2], offset: 0, index: 1)
                enc.setBytes(&count, length: 4, index: 2)
                enc.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
                enc.setComputePipelineState(p2)
                enc.setBuffer(slots[k % 2], offset: 0, index: 0)
                enc.setBuffer(sink, offset: 0, index: 1)
                enc.setBytes(&count, length: 4, index: 2)
                let groups = min(4096, (Int(count) + 255) / 256)
                enc.dispatchThreadgroups(MTLSize(width: groups, height: 1, depth: 1),
                    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
            }
        }
        enc.endEncoding()
        cb.commit(); cb.waitUntilCompleted()
        times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
    }
    times.sort()
    let p50 = times[times.count / 2]
    let gbPerS = Double(streamBytes) / (p50 / 1000) / 1e9
    print(String(format: "%@ X=%6.1f MB batches=%5d dispatches=%6d p50=%8.2f ms min=%8.2f  stream-equiv %6.1f GB/s  (3X device memory-equiv %6.1f GB/s)",
        label, sizeMB, batches, fused ? batches + 1 : 2 * batches, p50, times[0], gbPerS, 3 * gbPerS))
}

print(device.name)
// Pure stream read roofline: pass2 only over the stream buffer.
do {
    var times: [Double] = []
    for _ in 0..<5 {
        let cb = queue.makeCommandBuffer()!
        let enc = cb.makeComputeCommandEncoder()!
        enc.setComputePipelineState(p2)
        var count = UInt32(streamBytes / 16)
        enc.setBuffer(stream, offset: 0, index: 0)
        enc.setBuffer(sink, offset: 0, index: 1)
        enc.setBytes(&count, length: 4, index: 2)
        enc.dispatchThreadgroups(MTLSize(width: 4096, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        enc.endEncoding(); cb.commit(); cb.waitUntilCompleted()
        times.append((cb.gpuEndTime - cb.gpuStartTime) * 1000)
    }
    times.sort()
    print(String(format: "read-only 3 GiB p50=%.2f ms  %.1f GB/s", times[2], Double(streamBytes) / (times[2] / 1000) / 1e9))
}
for s in sizesMB {
    run("split", sizeMB: s, fused: false)
    run("fused", sizeMB: s, fused: true)
}
