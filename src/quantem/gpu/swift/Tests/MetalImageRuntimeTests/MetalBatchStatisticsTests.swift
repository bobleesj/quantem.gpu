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

  /// The fused copy-and-analyze batch copies every image exactly and returns
  /// the same ranges and bins as the two-round-trip batch on the copies, also
  /// after its reused statistics buffers wrap around.
  func testCopyAndAnalyzeMatchesTheBatchOnTheCopiesAcrossPoolReuse() throws {
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
    let reference = try statistics.analyzeUInt32Batch(values: buffers, rows: 192, columns: 192)
    for round in 0..<(MetalDisplayStatistics.statisticsPoolDepth + 1) {
      let destinations = try buffers.map { buffer in
        let destination = try XCTUnwrap(
          device.makeBuffer(length: buffer.length, options: .storageModeShared))
        memset(destination.contents(), 0xA5, destination.length)
        return destination
      }
      let fused = try statistics.copyAndAnalyzeUInt32Batch(
        sources: buffers, destinations: destinations, rows: 192, columns: 192)
      for index in buffers.indices {
        XCTAssertEqual(
          memcmp(buffers[index].contents(), destinations[index].contents(), buffers[index].length),
          0, "round \(round) image \(index) copy")
        for scale in 0..<2 {
          XCTAssertEqual(fused[index][scale].minimum, reference[index][scale].minimum)
          XCTAssertEqual(fused[index][scale].maximum, reference[index][scale].maximum)
          XCTAssertEqual(fused[index][scale].bins, reference[index][scale].bins, "round \(round)")
          let range = fused[index][scale].valueRange.contents().assumingMemoryBound(to: UInt32.self)
          XCTAssertEqual([range[0], range[1]], [sources[index].min()!, sources[index].max()!])
        }
      }
    }
    XCTAssertThrowsError(
      try statistics.copyAndAnalyzeUInt32Batch(
        sources: buffers, destinations: buffers, rows: 192, columns: 192))
    XCTAssertThrowsError(
      try statistics.copyAndAnalyzeUInt32Batch(
        sources: buffers, destinations: Array(buffers.prefix(1)), rows: 192, columns: 192))
  }
}
