import Metal
import XCTest

@testable import MetalDirectPtychoKernels

final class MetalSSBBrightfieldEstimateTests: XCTestCase {
  func testMeasuredDiskMatchesIndependentFullDetectorReference() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    // Non-square detector, fractional/off-center disk, soft edge and an invalid
    // hot pixel. Counts exceed uint32; no narrowed copy is permitted.
    let rows = 96
    let columns = 112
    var counts = (0..<(rows * columns)).map { i -> UInt64 in
      let radius = hypot(Double(i / columns) - 43.25, Double(i % columns) - 61.75)
      let intensity = 10 + 1100 / (1 + exp((radius - 23.4) * 2))
      return UInt64(intensity) * (1 << 28)
    }
    counts[0] = UInt64.max
    let measured = try MetalDirectPtychoBrightfieldEstimate.measure(
      device: device, detectorSum: counts,
      rows: rows, columns: columns, excluded: [0])
    counts[0] = 0
    let values = counts.map { Double($0) }
    let mean = values.reduce(0, +) / Double(values.count)
    let variance = values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)
    let selected = counts.indices.filter { values[$0] > mean + sqrt(variance) }
    let plateau = selected.map { values[$0] }.sorted()
    let threshold = (plateau[(plateau.count - 1) / 2] + plateau[plateau.count / 2]) / 4
    let edgeCount = values.filter { $0 > threshold }.count
    XCTAssertEqual(measured.selectionCount, selected.count)
    XCTAssertEqual(measured.edgeCount, edgeCount)
    XCTAssertEqual(
      measured.centerRow, selected.reduce(0) { $0 + Double($1 / columns) } / Double(selected.count),
      accuracy: 1e-12)
    XCTAssertEqual(
      measured.centerColumn,
      selected.reduce(0) { $0 + Double($1 % columns) } / Double(selected.count), accuracy: 1e-12)
    XCTAssertEqual(measured.edgeRadius, sqrt(Double(edgeCount) / .pi), accuracy: 1e-12)
    let angle = try measured.semiangle(rowSampling: 0.2, columnSampling: 0.8)
    XCTAssertEqual(angle, measured.edgeRadius * 0.4, accuracy: 1e-12)
    XCTAssertEqual(try measured.detectorSampling(semiangle: angle), 0.4, accuracy: 1e-12)
    XCTAssertThrowsError(try measured.detectorSampling(semiangle: 0))
    XCTAssertThrowsError(try measured.semiangle(rowSampling: .nan, columnSampling: 0.2))
  }

  func testNoInventedCalibrationForEmptyUniformOrClippedPattern() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let clipped: [UInt64] = (0..<(32 * 32)).map { i in
      let radius = hypot(Double(i / 32) - 1, Double(i % 32) - 16)
      return radius < 7 ? 100 : 0
    }
    for counts in [
      [UInt64](repeating: 0, count: 32 * 32), [UInt64](repeating: 10, count: 32 * 32), clipped,
    ] {
      XCTAssertThrowsError(
        try MetalDirectPtychoBrightfieldEstimate.measure(
          device: device, detectorSum: counts, rows: 32, columns: 32))
    }
    XCTAssertThrowsError(
      try MetalDirectPtychoBrightfieldEstimate.measure(
        device: device, detectorSum: [0], rows: 32, columns: 32))
  }
}
