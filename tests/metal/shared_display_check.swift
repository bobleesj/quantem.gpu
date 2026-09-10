import Foundation
import Metal
import MetalDisplayKernels
import MetalImageRuntime

// Standalone second consumer: no Live4DSTEM, AppKit, SwiftUI, or XCTest.
// Run with scripts/check_metal_display.sh on an Apple GPU.
@main enum SharedDisplayCheck {
  static func main() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      fatalError("An Apple GPU is required; this is not a CPU fallback test")
    }
    let statistics = try MetalDisplayStatistics(device: device, commandQueue: queue)
    func buffer<T>(_ samples: [T]) -> MTLBuffer {
      samples.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)!
      }
    }
    func bins(_ buffer: MTLBuffer) -> [UInt32] {
      Array(
        UnsafeBufferPointer(
          start: buffer.contents().assumingMemoryBound(to: UInt32.self), count: 256))
    }
    func submit(_ requests: [MetalStatisticsRequest], updateRange: Bool = true) throws {
      let command = queue.makeCommandBuffer()!
      try statistics.encode(requests, into: command, updateRange: updateRange)
      precondition(command.status == .notEnqueued, "encode must not submit")
      command.commit()
      command.waitUntilCompleted()
      if let error = command.error { throw error }
    }
    let integers: [[UInt32]] = [
      [0, 0, 0, 0], [UInt32.max, UInt32.max, UInt32.max],
      [0, 1, 255, 65535, UInt32.max], (0..<1024).map { UInt32($0 % 253) },
    ]
    let floats: [[Float]] = [
      [-8, -2, 0, 1, 7], [-1, -1, -1], [.nan, .infinity, -.infinity],
      [-5, .nan, 5, .infinity, 0],
      [-Float.greatestFiniteMagnitude, 0, Float.greatestFiniteMagnitude],
    ]
    let sources = integers.map(buffer) + floats.map(buffer)
    let ranges = sources.map { _ in buffer([UInt32.max, UInt32.max]) }
    let histograms = sources.map { _ in buffer(Array(repeating: UInt32.max, count: 256)) }
    var checks = 0
    for scale in [MetalDisplayScale.linear, .logarithmic] {
      let requests = sources.indices.map { index in
        MetalStatisticsRequest(
          values: sources[index], rows: 1,
          columns: index < integers.count
            ? integers[index].count : floats[index - integers.count].count,
          scalarType: index < integers.count ? .uint32 : .float32,
          scale: scale, valueRange: ranges[index], histogram: histograms[index])
      }
      // The same output allocations must survive repeated full/histogram-only updates.
      for updateRange in [true, false, true] {
        try submit(requests, updateRange: updateRange)
        for index in sources.indices {
          let actual = bins(histograms[index])
          if index < integers.count {
            let values = integers[index]
            let range = ranges[index].contents().assumingMemoryBound(to: UInt32.self)
            precondition(range[0] == values.min()! && range[1] == values.max()!)
            let serial = try statistics.analyzeUInt32(
              values: sources[index], rows: 1, columns: values.count, scale: scale)
            precondition(actual == serial.bins)
            precondition(actual.reduce(0, +) == UInt32(values.count))
            if range[0] == range[1] { precondition(actual[128] == UInt32(values.count)) }
          } else {
            let values = floats[index - integers.count].filter(\.isFinite)
            let range = ranges[index].contents().assumingMemoryBound(to: Float.self)
            precondition(range[0] == (values.min() ?? 0) && range[1] == (values.max() ?? 0))
            precondition(actual.reduce(0, +) == UInt32(values.count))
            let serial = try statistics.analyzeFloat32(
              values: sources[index], rows: 1,
              columns: floats[index - integers.count].count, scale: scale)
            precondition(actual == serial.bins)
          }
          checks += 1
        }
      }
    }
    // Independent, exact histogram oracle: endpoints/midpoint avoid floating bin edges.
    let source = buffer([UInt32(0), 0, 128, 256])
    let known = try statistics.analyzeUInt32(values: source, rows: 2, columns: 2, scale: .linear)
    var expected = [UInt32](repeating: 0, count: 256)
    expected[0] = 2
    expected[128] = 1
    expected[255] = 1
    precondition(known.bins == expected)
    let signed = try statistics.analyzeFloat32(
      values: buffer([Float(-8), -8, 0, 8, .nan]), rows: 1, columns: 5, scale: .linear)
    precondition(signed.bins == expected)
    let signedLog = try statistics.analyzeFloat32(
      values: buffer([Float(-8), -8, 0, 8, .nan]), rows: 1, columns: 5, scale: .logarithmic)
    precondition(signedLog.bins == expected)
    // Reject overlapping output buffers before encoding any GPU work.
    let invalidCommand = queue.makeCommandBuffer()!
    do {
      try statistics.encode([
        MetalStatisticsRequest(
          values: source, rows: 2, columns: 2, scalarType: .uint32,
          scale: .linear, valueRange: histograms[0], histogram: histograms[0])
      ], into: invalidCommand)
      preconditionFailure("Aliased statistics outputs must be rejected")
    } catch MetalImageRuntimeError.invalidStatisticsBuffers {
      precondition(invalidCommand.status == .notEnqueued)
    }
    let allMax = try statistics.analyzeUInt32Batch(
      values: [buffer([UInt32.max])], rows: 1, columns: 1)
    precondition(
      allMax[0].allSatisfy { $0.minimum == .max && $0.maximum == .max && $0.bins[128] == 1 })
    let selected = MetalPercentileRange(low: 0.05, high: 0.95)
    precondition(selected.window(bins: [1, 8, 1]) == MetalHistogramContrast(low: 0, high: 0.5))
    precondition(MetalPercentileRange.percentile(at: 0.5, bins: [90, 0, 10]) == 0.9)
    precondition(MetalPercentileRange.percentile(at: 0.5, bins: [10, 0, 90]) == 0.1)
    for logarithmic in [false, true] {
      for value in [-8.0, -2, 0, 1, 8] {
        let bounds = SIMD2<Double>(-8, 8)
        let fraction = MetalFloatDisplayMapping.fraction(
          value, bounds: bounds, logarithmic: logarithmic)
        let restored = MetalFloatDisplayMapping.rawValue(
          fraction, bounds: bounds, logarithmic: logarithmic)
        precondition(abs(value - restored) < 1e-12)
      }
    }
    precondition(
      MetalDisplayLimits.resolvedInteger(SIMD2(100.2, 100.3), bounds: SIMD2(0, 1000))
        == SIMD2<UInt32>(100, 101))
    for map in MetalColormap.allCases {
      let lut = try MetalDisplayKernels.lut(map)
      precondition(
        lut.count == 256
          && lut.allSatisfy { color in
            (0..<4).allSatisfy { color[$0].isFinite && color[$0] >= 0 && color[$0] <= 1 }
          })
    }
    let gray = try MetalDisplayKernels.lut(.gray)
    precondition(gray.first == SIMD4<Float>(0, 0, 0, 1) && gray.last == SIMD4<Float>(1, 1, 1, 1))
    print(
      "PASS device=\(device.name) mixed_statistics_checks=\(checks) exact_oracles=4 percentile_mapping=pass colormaps=\(MetalColormap.allCases.count) buffer_reuse=pass alias_validation=pass"
    )
  }
}
