import Foundation
import Metal

// Pattern-priced skeleton of the SSB cached-loss data movement.
//
// Replicates the production column-pass read pattern (8 bit-reversed float2
// loads per thread out of a contiguous 4 KB run) and the production
// STORE_BLOCKED pattern (32 B contiguous per 8,224 B stride), plus coalesced
// and streaming controls, at the production dispatch geometry
// (257 x batch threadgroups of 64 threads, 8 BF per dispatch, 1118 dispatches).
//
// No FFT, no chi/cross corrections: this isolates memory-system behaviour.
// Usage: ghost-traffic <out-dir> [reps] [tiles]

let args = CommandLine.arguments
let outDir = URL(fileURLWithPath: args.count >= 2 ? args[1] : ".", isDirectory: true)
let reps = args.count >= 3 ? (Int(args[2]) ?? 3) : 3
let tileCount = args.count >= 4 ? (Int(args[3]) ?? 1118) : 1118
try FileManager.default.createDirectory(at: outDir, withIntermediateDirectories: true)

let source = """
#include <metal_stdlib>
using namespace metal;

inline uint octal_reverse_512(uint value) {
    const uint d0 = value & 7u;
    const uint d1 = (value >> 3u) & 7u;
    const uint d2 = value >> 6u;
    return (d0 << 6u) | (d1 << 3u) | d2;
}

constant constexpr uint n = 512u;
constant constexpr uint half_cols = 257u;
constant constexpr uint block_rows = 4u;
constant constexpr uint plane = n * half_cols;

inline float2 ghost_value(uint tid, uint lane) {
    return float2((float)(tid * 8u + lane), (float)(tid ^ lane));
}

// Column-pass skeleton: bit-reversed loads + production blocked stores.
kernel void ghost_blocked(
    device const float2 *half_g [[buffer(0)]],
    device float2 *intermediate [[buffer(1)]],
    constant uint &tile_base [[buffer(2)]],
    constant uint &tiles [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    const uint col = group.x;
    const uint local_bf = group.y;
    const uint tile = (tile_base + group.y) % tiles;
    const size_t read_base = (size_t)tile * plane;
    const size_t write_base = (size_t)local_bf * plane;
    float2 r[8];
    for (uint lane = 0u; lane < 8u; ++lane) {
        const uint src = octal_reverse_512(tid * 8u + lane);
        r[lane] = half_g[read_base + (size_t)col * n + src];
    }
    const size_t store_base = write_base + (size_t)col * block_rows + (tid % block_rows);
    constexpr size_t block_stride = (size_t)half_cols * block_rows;
    for (uint lane = 0u; lane < 8u; ++lane) {
        intermediate[store_base + (size_t)((tid + 64u * lane) / block_rows) * block_stride] = r[lane];
    }
}

// Same loads, same byte count, coalesced destination.
kernel void ghost_coalesced(
    device const float2 *half_g [[buffer(0)]],
    device float2 *intermediate [[buffer(1)]],
    constant uint &tile_base [[buffer(2)]],
    constant uint &tiles [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    const uint col = group.x;
    const uint local_bf = group.y;
    const uint tile = (tile_base + group.y) % tiles;
    const size_t read_base = (size_t)tile * plane;
    const size_t write_base = (size_t)local_bf * plane;
    float2 r[8];
    for (uint lane = 0u; lane < 8u; ++lane) {
        const uint src = octal_reverse_512(tid * 8u + lane);
        r[lane] = half_g[read_base + (size_t)col * n + src];
    }
    for (uint lane = 0u; lane < 8u; ++lane) {
        const uint linear = (tid / 32u) * 256u + (tid % 32u) + lane * 32u;
        intermediate[write_base + (size_t)col * n + linear] = r[lane];
    }
}

// Same loads and blocked stores as production, but the 4 KB column run is
// staged through threadgroup memory with a coalesced cooperative load first.
kernel void ghost_staged(
    device const float2 *half_g [[buffer(0)]],
    device float2 *intermediate [[buffer(1)]],
    constant uint &tile_base [[buffer(2)]],
    constant uint &tiles [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup float2 staged[512];
    const uint col = group.x;
    const uint local_bf = group.y;
    const uint tile = (tile_base + group.y) % tiles;
    const size_t read_base = (size_t)tile * plane + (size_t)col * n;
    for (uint i = tid; i < 512u; i += 64u) { staged[i] = half_g[read_base + i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 r[8];
    for (uint lane = 0u; lane < 8u; ++lane) {
        r[lane] = staged[octal_reverse_512(tid * 8u + lane)];
    }
    const size_t write_base = (size_t)local_bf * plane;
    const size_t store_base = write_base + (size_t)col * block_rows + (tid % block_rows);
    constexpr size_t block_stride = (size_t)half_cols * block_rows;
    for (uint lane = 0u; lane < 8u; ++lane) {
        intermediate[store_base + (size_t)((tid + 64u * lane) / block_rows) * block_stride] = r[lane];
    }
}

// Staged read plus coalesced store (the fully-fixed column pass skeleton).
kernel void ghost_staged_coalesced(
    device const float2 *half_g [[buffer(0)]],
    device float2 *intermediate [[buffer(1)]],
    constant uint &tile_base [[buffer(2)]],
    constant uint &tiles [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    threadgroup float2 staged[512];
    const uint col = group.x;
    const uint local_bf = group.y;
    const uint tile = (tile_base + group.y) % tiles;
    const size_t read_base = (size_t)tile * plane + (size_t)col * n;
    for (uint i = tid; i < 512u; i += 64u) { staged[i] = half_g[read_base + i]; }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float2 r[8];
    for (uint lane = 0u; lane < 8u; ++lane) {
        r[lane] = staged[octal_reverse_512(tid * 8u + lane)];
    }
    const size_t write_base = (size_t)local_bf * plane;
    for (uint lane = 0u; lane < 8u; ++lane) {
        const uint linear = (tid / 32u) * 256u + (tid % 32u) + lane * 32u;
        intermediate[write_base + (size_t)col * n + linear] = r[lane];
    }
}

// Loads only, DCE-proofed by one guarded store per threadgroup.
kernel void ghost_read(
    device const float2 *half_g [[buffer(0)]],
    device float2 *sink [[buffer(1)]],
    constant uint &tile_base [[buffer(2)]],
    constant uint &tiles [[buffer(3)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    const uint tile = (tile_base + group.y) % tiles;
    const size_t read_base = (size_t)tile * plane;
    float2 value = float2(0.0f);
    for (uint lane = 0u; lane < 8u; ++lane) {
        const uint src = octal_reverse_512(tid * 8u + lane);
        value += half_g[read_base + (size_t)group.x * n + src];
    }
    // The guard is data dependent so the loads can never be sunk into it.
    if (value.x == 3.5e38f && value.y == -3.5e38f) {
        sink[group.y % 4096u] = value;
    }
}

// Production blocked stores, no loads.
kernel void ghost_write_blocked(
    device float2 *intermediate [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    const uint col = group.x;
    const uint local_bf = group.y;
    const size_t write_base = (size_t)local_bf * plane;
    const size_t store_base = write_base + (size_t)col * block_rows + (tid % block_rows);
    constexpr size_t block_stride = (size_t)half_cols * block_rows;
    for (uint lane = 0u; lane < 8u; ++lane) {
        intermediate[store_base + (size_t)((tid + 64u * lane) / block_rows) * block_stride] =
            ghost_value(tid, lane);
    }
}

// Coalesced stores, no loads.
kernel void ghost_write_coalesced(
    device float2 *intermediate [[buffer(0)]],
    uint tid [[thread_index_in_threadgroup]],
    uint2 group [[threadgroup_position_in_grid]])
{
    const uint col = group.x;
    const uint local_bf = group.y;
    const size_t write_base = (size_t)local_bf * plane;
    for (uint lane = 0u; lane < 8u; ++lane) {
        const uint linear = (tid / 32u) * 256u + (tid % 32u) + lane * 32u;
        intermediate[write_base + (size_t)col * n + linear] = ghost_value(tid, lane);
    }
}

// Row-pass skeleton: one 256-thread threadgroup stages each (BF, row block) as
// one contiguous 1028-float2 run, exactly like ssb_fetch_row_block.
kernel void ghost_row_read(
    device const float2 *intermediate [[buffer(0)]],
    device float2 *sink [[buffer(1)]],
    constant uint &batch [[buffer(2)]],
    uint tid [[thread_index_in_threadgroup]],
    uint row_block [[threadgroup_position_in_grid]])
{
    threadgroup float2 staged[5];
    const size_t block_base = (size_t)row_block * (half_cols * block_rows);
    float2 value = float2(0.0f);
    for (uint local_bf = 0u; local_bf < batch; ++local_bf) {
        for (uint j = 0u; j < 5u; ++j) {
            const uint e = tid + j * 256u;
            if (e < half_cols * block_rows) {
                staged[j] = intermediate[block_base + (size_t)local_bf * plane + e];
            }
        }
        for (uint j = 0u; j < 5u; ++j) { value += staged[j]; }
    }
    if (value.x == 3.5e38f && value.y == -3.5e38f) { sink[row_block % 4096u] = value; }
}

// Streaming write control: the same volume, coalesced, no reuse.
kernel void ghost_stream_write(
    device float2 *half_g [[buffer(0)]],
    uint index [[thread_position_in_grid]])
{
    const size_t base = (size_t)index * 8u;
    for (uint i = 0u; i < 8u; ++i) {
        half_g[base + i] = float2((float)(index & 1023u), (float)(index >> 10u));
    }
}

// Pure streaming read control at the same volume.
kernel void ghost_stream_read(
    device const float2 *half_g [[buffer(0)]],
    device float2 *sink [[buffer(1)]],
    uint index [[thread_position_in_grid]])
{
    const size_t base = (size_t)index * 8u;
    float2 value = float2(0.0f);
    for (uint i = 0u; i < 8u; ++i) { value += half_g[base + i]; }
    if (value.x == 3.5e38f && value.y == -3.5e38f) { sink[index % 4096u] = value; }
}
"""

setvbuf(stdout, nil, _IONBF, 0)
print("mark: start")
let device = MTLCreateSystemDefaultDevice()!
print("mark: device \(device.name)")
let queue = device.makeCommandQueue()!
print("mark: compiling library")
let library = try device.makeLibrary(source: source, options: nil)
print("mark: library ok")

let halfPlaneBytes = 512 * 257 * MemoryLayout<SIMD2<Float>>.stride
let batch = 8
let dispatches = 1118
let planes = batch

func pipeline(_ name: String) throws -> MTLComputePipelineState {
  guard let function = library.makeFunction(name: name) else {
    throw NSError(domain: "ghost", code: 1, userInfo: [NSLocalizedDescriptionKey: "missing \(name)"])
  }
  return try device.makeComputePipelineState(function: function)
}

let cacheBytes = tileCount * halfPlaneBytes
print("read window: \(tileCount) tiles = " + String(format: "%.2f", Double(cacheBytes) / 1e9) + " GB; intermediate: \(planes) planes = " + String(format: "%.1f", Double(planes * halfPlaneBytes) / 1e6) + " MB")

guard let cache = device.makeBuffer(length: cacheBytes, options: .storageModePrivate),
  let intermediate = device.makeBuffer(
    length: planes * halfPlaneBytes, options: .storageModePrivate),
  let sink = device.makeBuffer(length: 65536 * 8, options: .storageModePrivate)
else {
  FileHandle.standardError.write(Data("buffer allocation failed at \(cacheBytes) bytes\n".utf8))
  exit(3)
}

print("mark: buffers allocated")
// Page the read window in once so page faults are not counted as traffic.
do {
  guard let commands = queue.makeCommandBuffer(), let blit = commands.makeBlitCommandEncoder()
  else { exit(4) }
  blit.fill(buffer: cache, range: 0..<cache.length, value: 1)
  blit.fill(buffer: intermediate, range: 0..<intermediate.length, value: 2)
  blit.endEncoding()
  commands.commit()
  commands.waitUntilCompleted()
}

print("mark: prefill done")
var tileValue = UInt32(tileCount)
var batchValue = UInt32(batch)

struct Variant {
  let name: String
  let bytesRead: Double
  let bytesWritten: Double
}

let windowBytes = Double(tileCount) * Double(halfPlaneBytes)
let fullBytes = Double(dispatches) * Double(batch) * Double(halfPlaneBytes)
let variants: [Variant] = [
  Variant(name: "ghost_read", bytesRead: fullBytes, bytesWritten: 0),
  Variant(name: "ghost_write_blocked", bytesRead: 0, bytesWritten: fullBytes),
  Variant(name: "ghost_write_coalesced", bytesRead: 0, bytesWritten: fullBytes),
  Variant(name: "ghost_blocked", bytesRead: fullBytes, bytesWritten: fullBytes),
  Variant(name: "ghost_coalesced", bytesRead: fullBytes, bytesWritten: fullBytes),
  Variant(name: "ghost_staged", bytesRead: fullBytes, bytesWritten: fullBytes),
  Variant(name: "ghost_staged_coalesced", bytesRead: fullBytes, bytesWritten: fullBytes),
  Variant(name: "ghost_row_read", bytesRead: fullBytes, bytesWritten: 0),
  Variant(name: "ghost_stream_read", bytesRead: windowBytes, bytesWritten: 0),
  Variant(name: "ghost_stream_write", bytesRead: 0, bytesWritten: windowBytes),
]

func run(_ variant: Variant, dispatches count: Int) throws -> (gpuMs: Double, wallMs: Double) {
  let start = Date()
  guard let commands = queue.makeCommandBuffer() else { exit(5) }
  let state = try pipeline(variant.name)
  if variant.name == "ghost_stream_read" {
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(state)
    encoder.setBuffer(cache, offset: 0, index: 0)
    encoder.setBuffer(sink, offset: 0, index: 1)
    encoder.dispatchThreads(
      MTLSize(width: tileCount * 512 * 257 / 8, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
  } else if variant.name == "ghost_stream_write" {
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(state)
    encoder.setBuffer(cache, offset: 0, index: 0)
    encoder.dispatchThreads(
      MTLSize(width: tileCount * 512 * 257 / 8, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
  } else if variant.name == "ghost_row_read" {
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(state)
    encoder.setBuffer(intermediate, offset: 0, index: 0)
    encoder.setBuffer(sink, offset: 0, index: 1)
    encoder.setBytes(&batchValue, length: 4, index: 2)
    for _ in 0..<count {
      encoder.dispatchThreadgroups(
        MTLSize(width: 128, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    encoder.endEncoding()
  } else {
    // One encoder per dispatch, exactly like the production cache walk: each
    // dispatch binds the next 8 tiles of the read window.
    for dispatch in 0..<count {
      let encoder = commands.makeComputeCommandEncoder()!
      encoder.setComputePipelineState(state)
      if variant.name == "ghost_read" {
        encoder.setBuffer(cache, offset: 0, index: 0)
        encoder.setBuffer(sink, offset: 0, index: 1)
      } else if variant.name.hasPrefix("ghost_write") {
        encoder.setBuffer(intermediate, offset: 0, index: 0)
      } else {
        encoder.setBuffer(cache, offset: 0, index: 0)
        encoder.setBuffer(intermediate, offset: 0, index: 1)
      }
      if !variant.name.hasPrefix("ghost_write") {
        var tileBase = UInt32((dispatch * batch) % tileCount)
        encoder.setBytes(&tileBase, length: 4, index: 2)
        encoder.setBytes(&tileValue, length: 4, index: 3)
      }
      encoder.dispatchThreadgroups(
        MTLSize(width: 257, height: batch, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
      encoder.endEncoding()
    }
  }
  commands.commit()
  commands.waitUntilCompleted()
  if let error = commands.error {
    throw NSError(domain: "ghost", code: 6, userInfo: [NSLocalizedDescriptionKey: error.localizedDescription])
  }
  let wallMs = Date().timeIntervalSince(start) * 1000
  let gpuMs = commands.gpuEndTime > commands.gpuStartTime
    ? (commands.gpuEndTime - commands.gpuStartTime) * 1000 : wallMs
  return (gpuMs, wallMs)
}

// Warm every variant once at short length (pipeline state creation, etc.).
for variant in variants { _ = try run(variant, dispatches: 16) }

var samples: [[String: Any]] = []
for rep in 0..<reps {
  for variant in variants {
    let timing = try run(variant, dispatches: dispatches)
    let bytes = variant.bytesRead + variant.bytesWritten
    var loads = [Double](repeating: 0, count: 3)
    let loadCount = getloadavg(&loads, 3)
    samples.append([
      "variant": variant.name, "rep": rep,
      "gpu_ms": timing.gpuMs, "wall_ms": timing.wallMs,
      "bytes": bytes, "gb_per_s": bytes / timing.gpuMs / 1e6,
      "gb_per_s_wall": bytes / timing.wallMs / 1e6,
      "load1": loadCount >= 1 ? loads[0] : -1,
      "epoch": Date().timeIntervalSince1970,
    ])
    let label = variant.name.padding(toLength: 22, withPad: " ", startingAt: 0)
    print(String(format: "%@ rep %d  gpu %8.2f ms  %7.1f GB/s", label, rep,
      timing.gpuMs, bytes / timing.gpuMs / 1e6))
  }
}

let report: [String: Any] = [
  "device": device.name, "tiles": tileCount, "reps": reps,
  "dispatches": dispatches, "batch": batch,
  "threadgroups_per_dispatch": 257 * batch,
  "samples": samples,
]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
  .write(to: outDir.appendingPathComponent("ghost-report.json"))
print("wrote \(outDir.appendingPathComponent("ghost-report.json").path)")
