import Foundation
import Metal
import MetalDisplayKernels

let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let library = try MetalDisplayKernels.makeLibrary(device: device)
func pipeline(_ name: String) throws -> MTLComputePipelineState {
  try device.makeComputePipelineState(function: library.makeFunction(name: name)!)
}
let rangeKernel = try pipeline(MetalDisplayKernels.simdRangeFunction)
let histogramKernel = try pipeline(MetalDisplayKernels.histogramFromRangeFunction)
let referenceHistogram = try pipeline(MetalDisplayKernels.histogramFunction)
let fixtures: [[UInt32]] = [
  [0], [65535], Array(repeating: 0, count: 33), Array(repeating: 65535, count: 31),
  (0..<257).map(UInt32.init),
  (0..<65537).map { UInt32($0 % 2 == 0 ? $0 : 65537 - $0) },
  [0, 1, 255, 256, 32767, 32768, 65535, 1 << 24, UInt32.max],
]
for values in fixtures {
  for scale in [MetalDisplayScale.linear, .logarithmic] {
    try autoreleasepool {
      let input = values.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count)!
      }
      let range = device.makeBuffer(length: 8)!
      let rangeValues = range.contents().bindMemory(to: UInt32.self, capacity: 2)
      rangeValues[0] = .max
      rangeValues[1] = 0
      let bins = device.makeBuffer(length: 1024)!
      let expectedBins = device.makeBuffer(length: 1024)!
      memset(bins.contents(), 0, 1024)
      memset(expectedBins.contents(), 0, 1024)
      let command = queue.makeCommandBuffer()!
      let rangeEncoder = command.makeComputeCommandEncoder()!
      var count = UInt32(values.count)
      rangeEncoder.setComputePipelineState(rangeKernel)
      rangeEncoder.setBuffer(input, offset: 0, index: 0)
      rangeEncoder.setBuffer(range, offset: 0, index: 1)
      rangeEncoder.setBytes(&count, length: 4, index: 2)
      rangeEncoder.dispatchThreadgroups(
        MTLSize(width: (values.count + 127) / 128, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      rangeEncoder.endEncoding()
      var parameters = MetalDisplayParameters(
        rows: 1, cols: values.count, low: values.min()!, high: values.max()!,
        scale: scale)
      for (kernel, output) in [(histogramKernel, bins), (referenceHistogram, expectedBins)] {
        let encoder = command.makeComputeCommandEncoder()!
        encoder.setComputePipelineState(kernel)
        encoder.setBuffer(input, offset: 0, index: 0)
        encoder.setBuffer(output, offset: 0, index: 1)
        encoder.setBytes(&parameters, length: MemoryLayout.stride(ofValue: parameters), index: 2)
        if kernel === histogramKernel { encoder.setBuffer(range, offset: 0, index: 3) }
        encoder.dispatchThreadgroups(
          MTLSize(width: (values.count + 127) / 128, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        encoder.endEncoding()
      }
      command.commit()
      command.waitUntilCompleted()
      precondition(command.status == .completed)
      precondition(rangeValues[0] == values.min() && rangeValues[1] == values.max())
      precondition(memcmp(bins.contents(), expectedBins.contents(), 1024) == 0)
      let counts = UnsafeBufferPointer(
        start: bins.contents().assumingMemoryBound(to: UInt32.self), count: 256)
      precondition(counts.reduce(0, +) == values.count)
    }
  }
}
print("DISPLAY_RANGE_HISTOGRAM_EXACT_PASS")
