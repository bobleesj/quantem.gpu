import Foundation
import Metal

let root = URL(fileURLWithPath: CommandLine.arguments[1])
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
func load(_ name: String) throws -> MTLBuffer {
  let bytes = try Data(contentsOf: root.appendingPathComponent(name))
  return bytes.withUnsafeBytes {
    device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
  }
}
let payload = try load("payload.c64")
let edges = try load("edges.c64")
let rows = try load("rows.u32")
let expected = try Data(contentsOf: root.appendingPathComponent("expected.c64"))
let output = device.makeBuffer(length: expected.count, options: .storageModeShared)!
let code = """
#include <metal_stdlib>
using namespace metal;
kernel void restore(device const uint2 *payload [[buffer(0)]],
                    device const uint2 *edges [[buffer(1)]],
                    device const uint2 *rows [[buffer(2)]],
                    device uint2 *output [[buffer(3)]],
                    constant uint &count [[buffer(4)]],
                    uint index [[thread_position_in_grid]]) {
  if (index >= count) return;
  uint row = index / 257u;
  uint col = index % 257u;
  uint2 result = uint2(0u);
  if (col == 0u || col == 256u) {
    result = edges[row * 2u + (col == 256u ? 1u : 0u)];
  } else {
    uint2 descriptor = rows[row];
    uint start = descriptor.y & 65535u;
    uint length = descriptor.y >> 16u;
    if (col >= start && col - start < length)
      result = payload[descriptor.x + col - start];
  }
  output[index] = result;
}
"""
let library = try device.makeLibrary(source: code, options: nil)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "restore")!)
var count = UInt32(expected.count / 8)
var milliseconds: [Double] = []
for repetition in 0..<21 {
  let commands = queue.makeCommandBuffer()!
  let encoder = commands.makeComputeCommandEncoder()!
  encoder.setComputePipelineState(pipeline)
  for (index, buffer) in [payload, edges, rows, output].enumerated() {
    encoder.setBuffer(buffer, offset: 0, index: index)
  }
  encoder.setBytes(&count, length: 4, index: 4)
  encoder.dispatchThreads(MTLSize(width: Int(count), height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  encoder.endEncoding()
  commands.commit()
  commands.waitUntilCompleted()
  precondition(commands.status == .completed)
  let actual = Data(bytes: output.contents(), count: expected.count)
  precondition(actual == expected, "Compact interval Metal lookup changed bits")
  if repetition > 0 { milliseconds.append((commands.gpuEndTime-commands.gpuStartTime)*1000) }
}
let report: [String: Any] = ["device": device.name, "cases": 21, "bit_exact": true,
  "sample_bf_count": 8, "gpu_ms": milliseconds,
  "compact_sample_bytes": payload.length+edges.length+rows.length,
  "dense_sample_bytes": expected.count,
  "scope": "eight real full-scan BF format lookup only; not full SSB objective timing"]
let json = try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
try json.write(to: root.appendingPathComponent("metal-result.json"))
print(String(data: json, encoding: .utf8)!)
