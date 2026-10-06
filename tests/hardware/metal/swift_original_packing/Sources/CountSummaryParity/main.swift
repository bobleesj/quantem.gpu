import Foundation
import Metal
import Metal4DSTEMKernels

let device = MTLCreateSystemDefaultDevice()!
let library = try Metal4DSTEMKernels.makeCompactH5Library(device: device)
let pipeline = try device.makeComputePipelineState(function: library.makeFunction(name: "compact_h5_total_counts")!)
let queue = device.makeCommandQueue()!
var cases = [[UInt64](repeating: 0, count: 1), [UInt64.max], [UInt64.max - 1, 1]]
for n in [2, 255, 256, 257, 1003, 262144] {
  cases.append((0..<n).map { UInt64($0 % 11) + (UInt64(1) << 33) })
}
for values in cases {
  var words = [UInt32](repeating: .max, count: values.count * 8)
  for (i, value) in values.enumerated() {
    words[i * 8] = UInt32(truncatingIfNeeded: value)
    words[i * 8 + 1] = UInt32(truncatingIfNeeded: value >> 32)
  }
  let input = words.withUnsafeBytes { device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)! }
  let output = device.makeBuffer(length: 8)!
  let command = queue.makeCommandBuffer()!
  let encoder = command.makeComputeCommandEncoder()!
  var count = UInt32(values.count)
  encoder.setComputePipelineState(pipeline)
  encoder.setBuffer(input, offset: 0, index: 0)
  encoder.setBuffer(output, offset: 0, index: 1)
  encoder.setBytes(&count, length: 4, index: 2)
  encoder.dispatchThreadgroups(MTLSize(width: 1, height: 1, depth: 1),
    threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  encoder.endEncoding()
  command.commit()
  command.waitUntilCompleted()
  if let error = command.error { throw error }
  let total = output.contents().load(as: UInt64.self)
  precondition(total == values.reduce(0, +), "Exact UInt64 count mismatch")
  print("PASS scans=\(values.count) total=\(total) GPU_ms=\((command.gpuEndTime - command.gpuStartTime) * 1000)")
}
