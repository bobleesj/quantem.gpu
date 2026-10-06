import Metal
import MetalDisplayKernels
import MetalImageRuntime
import XCTest

final class MetalBatchStatisticsTests: XCTestCase {
  func testBatchMatchesEverySerialRangeAndHistogramExactly() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let statistics = try MetalDisplayStatistics(device: device)
    let sources: [[UInt32]] = [
      Array(repeating: 0, count: 36864),
      Array(repeating: 65535, count: 36864),
      (0..<36864).map { UInt32($0 * 973) },
      (0..<36864).map { $0 % 19 == 0 ? UInt32.max : UInt32($0 % 7) },
    ]
    let buffers = try sources.map { source in
      try XCTUnwrap(
        source.withUnsafeBytes {
          device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
    }
    let scales: [MetalDisplayScale] = [.linear, .logarithmic]
    let batch = try statistics.analyzeUInt32Batch(values: buffers, rows: 192, columns: 192)
    for index in buffers.indices {
      for (j, scale) in scales.enumerated() {
        let serial = try statistics.analyzeUInt32(
          values: buffers[index], rows: 192, columns: 192, scale: scale)
        XCTAssertEqual(batch[index][j].minimum, serial.minimum)
        XCTAssertEqual(batch[index][j].maximum, serial.maximum)
        XCTAssertEqual(batch[index][j].bins, serial.bins)
        XCTAssertEqual(batch[index][j].bins.reduce(UInt64(0)) { $0 + UInt64($1) }, 36864)
      }
    }
    XCTAssertTrue(try statistics.analyzeUInt32Batch(values: [], rows: 192, columns: 192).isEmpty)
    XCTAssertThrowsError(
      try statistics.analyzeUInt32Batch(values: buffers, rows: 193, columns: 192))
  }
}
