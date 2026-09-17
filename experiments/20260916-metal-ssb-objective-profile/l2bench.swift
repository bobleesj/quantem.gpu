import Foundation
import Metal

// Memory-system characterization for the Metal SSB profile. Measures achieved
// read, write and read-modify-write bandwidth as a function of the working-set
// size so the fusion question ("can the 9.41 GB blocked intermediate stay
// on-chip?") can be answered with hardware limits instead of intuition.
let device = MTLCreateSystemDefaultDevice()!
print("device=\(device.name)")
print("maxThreadgroupMemoryLength=\(device.maxThreadgroupMemoryLength)")
print("recommendedMaxWorkingSetSize=\(device.recommendedMaxWorkingSetSize)")
print("hasUnifiedMemory=\(device.hasUnifiedMemory)")
print("maxBufferLength=\(device.maxBufferLength)")

let source = """
#include <metal_stdlib>
using namespace metal;

// Every kernel shares one signature so the host argument table is identical.
kernel void m_read(
    device const float4 *p [[buffer(0)]],
    device float *sink [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &reps [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    float4 acc = float4(0.0f);
    for (uint r = 0u; r < reps; ++r) {
        for (uint i = tid; i < count; i += stride) acc += p[i];
    }
    const float s = acc.x + acc.y + acc.z + acc.w;
    if (s == 1.0e30f) sink[0] = s;
}

kernel void m_write(
    device const float4 *p [[buffer(0)]],
    device float *sink [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &reps [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    device float4 *out = const_cast<device float4 *>(p);
    for (uint r = 0u; r < reps; ++r) {
        for (uint i = tid; i < count; i += stride) {
            out[i] = float4(1.0f, 2.0f, 3.0f, 4.0f);
        }
    }
    const float s = float(tid);
    if (s == 1.0e30f) sink[0] = s;
}

kernel void m_rmw(
    device const float4 *p [[buffer(0)]],
    device float *sink [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &reps [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    device float4 *out = const_cast<device float4 *>(p);
    for (uint r = 0u; r < reps; ++r) {
        for (uint i = tid; i < count; i += stride) {
            const float4 v = p[i];
            out[i] = float4(v.x + 1.0f, v.y, v.z, v.w);
        }
    }
    const float s = float(tid);
    if (s == 1.0e30f) sink[0] = s;
}

// Replicates the blocked intermediate store pattern of
// ssb_correct_half_column_ifft512_hermitian: per (bf, col) one threadgroup of
// 64 threads writes 512 float2 at a 1028-float2 pitch, i.e. 32 contiguous
// bytes per 8224-byte stride.
// Bright-field read pattern of the redraw accumulator in the two cache
// layouts: row-major [row][col] versus column-major [col][row].
kernel void m_redraw_pattern(
    device const float2 *p [[buffer(0)]],
    device float *sink [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &reps [[buffer(3)]],
    constant uint &column_major [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    const uint rows = 512u;
    const uint cols = 257u;
    if (tid >= rows * cols) return;
    const uint row = tid / cols;
    const uint col = tid - row * cols;
    float2 sum = float2(0.0f);
    for (uint batch = 0u; batch < reps; ++batch) {
        for (uint plane = 0u; plane < count; ++plane) {
            const size_t base = ((size_t)batch * count + plane) * rows * cols;
            const size_t offset =
                column_major ? (size_t)col * rows + row : (size_t)row * cols + col;
            sum += p[base + offset];
        }
    }
    if (sum.x == 1.0e30f) sink[0] = sum.y;
}

kernel void m_blocked_write(
    device const float *p [[buffer(0)]],
    device float *sink [[buffer(1)]],
    constant uint &count [[buffer(2)]],
    constant uint &reps [[buffer(3)]],
    constant uint &stride [[buffer(4)]],
    uint tid [[thread_position_in_grid]]) {
    device float *out = const_cast<device float *>(p);
    const uint col = tid / 512u;
    const uint row = tid % 512u;
    const uint base = col * 4u + (row % 4u) + (row / 4u) * 1028u;
    for (uint r = 0u; r < reps; ++r) {
        const uint plane_base = (r % (count / 131584u)) * 131584u;
        out[plane_base * 2u + base * 2u] = 1.0f;
        out[plane_base * 2u + base * 2u + 1u] = 2.0f;
    }
    const float s = float(tid);
    if (s == 1.0e30f) sink[0] = s;
}
"""
let library = try device.makeLibrary(source: source, options: nil)
let queue = device.makeCommandQueue()!
func pipeline(_ name: String) throws -> MTLComputePipelineState {
  try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}
let readPS = try pipeline("m_read")
let redrawPS = try pipeline("m_redraw_pattern")
let blockedWritePS = try pipeline("m_blocked_write")
let writePS = try pipeline("m_write")
let rmwPS = try pipeline("m_rmw")

let sink = device.makeBuffer(length: 64, options: .storageModePrivate)!
let threads = 1 << 18
let threadgroup = 256

func measure(
  _ ps: MTLComputePipelineState, buffer: MTLBuffer, count: Int, reps: Int
) -> Double {
  var mutableCount = UInt32(count)
  var mutableReps = UInt32(reps)
  var mutableStride = UInt32(threads)
  let commands = queue.makeCommandBuffer()!
  let encoder = commands.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(ps)
  encoder.setBuffer(buffer, offset: 0, index: 0)
  encoder.setBuffer(sink, offset: 0, index: 1)
  encoder.setBytes(&mutableCount, length: 4, index: 2)
  encoder.setBytes(&mutableReps, length: 4, index: 3)
  encoder.setBytes(&mutableStride, length: 4, index: 4)
  encoder.dispatchThreads(
    MTLSize(width: threads, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: threadgroup, height: 1, depth: 1))
  encoder.endEncoding()
  commands.commit()
  commands.waitUntilCompleted()
  return commands.gpuStartTime == 0 ? 0 : commands.gpuEndTime - commands.gpuStartTime
}

let totalBytes = 3.2e9
var rows: [[String: Any]] = []
for megabytes in [1, 2, 4, 8, 12, 16, 24, 32, 48, 64, 96, 128, 192, 256, 384] {
  let bytes = megabytes * 1_000_000
  let count = bytes / 16
  let buffer = device.makeBuffer(length: count * 16, options: .storageModePrivate)!
  let reps = max(1, Int(totalBytes) / (count * 16))
  // Warm the allocation once so first-touch faults are outside the timing.
  _ = measure(readPS, buffer: buffer, count: count, reps: 1)
  let readSeconds = measure(readPS, buffer: buffer, count: count, reps: reps)
  let writeSeconds = measure(writePS, buffer: buffer, count: count, reps: reps)
  let rmwSeconds = measure(rmwPS, buffer: buffer, count: count, reps: reps)
  let readGB = Double(count * 16 * reps) / 1e9
  let writeGB = readGB
  let rmwGB = readGB * 2
  rows.append([
    "mb": megabytes, "reps": reps,
    "read_gbs": readGB / readSeconds,
    "write_gbs": writeGB / writeSeconds,
    "rmw_gbs": rmwGB / rmwSeconds,
  ])
  print(String(
    format: "%5d MB  read %7.1f GB/s   write %7.1f GB/s   rmw %7.1f GB/s",
    megabytes,
    readGB / readSeconds, writeGB / writeSeconds, rmwGB / rmwSeconds))
}

// Redraw read pattern in both cache layouts over the full 8937-plane cache.
var redraw: [[String: Any]] = []
for columnMajor in [0, 1] {
  let planes = 8937
  let batches = 35
  let perBatch = 256
  let buffer = device.makeBuffer(
    length: planes * 512 * 257 * 8, options: .storageModePrivate)!
  var mutableCount = UInt32(perBatch)
  var mutableReps = UInt32(batches)
  var mutableColumnMajor = UInt32(columnMajor)
  let commands = queue.makeCommandBuffer()!
  let encoder = commands.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(redrawPS)
  encoder.setBuffer(buffer, offset: 0, index: 0)
  encoder.setBuffer(sink, offset: 0, index: 1)
  encoder.setBytes(&mutableCount, length: 4, index: 2)
  encoder.setBytes(&mutableReps, length: 4, index: 3)
  encoder.setBytes(&mutableColumnMajor, length: 4, index: 4)
  encoder.dispatchThreads(
    MTLSize(width: 512 * 257, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  encoder.endEncoding()
  commands.commit()
  commands.waitUntilCompleted()
  let seconds = commands.gpuEndTime - commands.gpuStartTime
  let bytes = Double(batches * perBatch * 512 * 257 * 8)
  redraw.append([
    "column_major": columnMajor, "bytes": bytes,
    "gbs": bytes / 1e9 / seconds, "seconds": seconds,
  ])
  print(String(
    format: "redraw pattern %@  %6.2f GB in %.3f s = %6.1f GB/s",
    columnMajor == 1 ? "column-major" : "row-major   ", bytes / 1e9, seconds,
    bytes / 1e9 / seconds))
}

// Blocked intermediate store pattern versus a linear store of the same volume.
var blocked: [[String: Any]] = []
for planes in [8, 16, 64, 512] {
  let perPlane = 257 * 512
  let buffer = device.makeBuffer(
    length: planes * perPlane * 8, options: .storageModePrivate)!
  var mutableCount = UInt32(planes * perPlane)
  var mutableReps = UInt32(planes)
  var mutableStride = UInt32(257 * 512)
  let commands = queue.makeCommandBuffer()!
  let encoder = commands.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(blockedWritePS)
  encoder.setBuffer(buffer, offset: 0, index: 0)
  encoder.setBuffer(sink, offset: 0, index: 1)
  encoder.setBytes(&mutableCount, length: 4, index: 2)
  encoder.setBytes(&mutableReps, length: 4, index: 3)
  encoder.setBytes(&mutableStride, length: 4, index: 4)
  encoder.dispatchThreads(
    MTLSize(width: 257 * 512, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
  encoder.endEncoding()
  commands.commit()
  commands.waitUntilCompleted()
  let seconds = commands.gpuEndTime - commands.gpuStartTime
  let bytes = Double(planes * perPlane * 8)
  blocked.append([
    "planes": planes, "mb": bytes / 1e6,
    "blocked_write_gbs": bytes / 1e9 / seconds,
  ])
  print(String(
    format: "%5d planes (%6.1f MB)  blocked write %7.1f GB/s",
    planes, bytes / 1e6, bytes / 1e9 / seconds))
}

let report: [String: Any] = [
  "device": device.name,
  "max_threadgroup_memory": device.maxThreadgroupMemoryLength,
  "recommended_max_working_set": device.recommendedMaxWorkingSetSize,
  "rows": rows,
  "blocked_write": blocked,
  "redraw_pattern": redraw,
]
let output = URL(fileURLWithPath: CommandLine.arguments[1], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent("l2.json"))
print("Completed: \(output.path)")
