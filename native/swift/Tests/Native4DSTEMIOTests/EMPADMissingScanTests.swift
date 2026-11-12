import Foundation
import Metal
import Native4DSTEMIO
import XCTest

@testable import Metal4DSTEMStreamingIO

final class EMPADMissingScanTests: XCTestCase {
  func testMissingFrameMaskInterpolationAndQEMRoundTrip() throws {
    guard let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue() else {
      throw XCTSkip("Metal required")
    }
    let dir = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    let raw = dir.appendingPathComponent("scan_x5_y3.raw")
    var original = [Float](repeating: 0, count: 15 * 130 * 128)
    for f in 0..<15 {
      for p in 0..<16384 { original[f * 16640 + p] = Float(f * f) + Float(p % 17) * 0.125 - 4 }
    }
    for p in 0..<16384 { original[7 * 16640 + p] = Float(bitPattern: 0x7fc0_1234) }
    try original.withUnsafeBytes { try Data($0).write(to: raw) }
    let source = try NativeEMPADSource.open(raw)
    let resident = try MetalEMPADResidentSource.load(
      source, device: device, memoryBudgetBytes: 512 << 20)
    XCTAssertEqual(resident.scanValidity.invalidFrames, [7])
    XCTAssertEqual(resident.scanValidity.interpolatedFrames, [7])
    let pixels = [0, 1, 7, 128, 16383]
    for corrected in [false, true] {
      let output = try XCTUnwrap(
        device.makeBuffer(length: pixels.count * 15 * 4, options: .storageModeShared))
      let command = try XCTUnwrap(queue.makeCommandBuffer())
      try resident.encodeDetectorColumns(
        pixels: pixels, into: output, command: command, interpolateMissing: corrected)
      command.commit()
      command.waitUntilCompleted()
      XCTAssertNil(command.error)
      let values = output.contents().assumingMemoryBound(to: Float.self)
      for (slot, p) in pixels.enumerated() {
        for f in 0..<15 {
          if f == 7 && corrected {
            let expected = [2, 12, 6, 8].reduce(Float(0)) { $0 + original[$1 * 16640 + p] * 0.25 }
            XCTAssertEqual(values[slot * 15 + f], expected)
          } else {
            XCTAssertEqual(values[slot * 15 + f].bitPattern, original[f * 16640 + p].bitPattern)
          }
        }
      }
    }
    let mean = try XCTUnwrap(device.makeBuffer(length: 16384 * 4, options: .storageModeShared))
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    try resident.encodeMeanDiffraction(into: mean, command: command)
    command.commit()
    command.waitUntilCompleted()
    XCTAssertNil(command.error)
    for p in pixels {
      let expected =
        (0..<15).filter { $0 != 7 }.reduce(0.0) { $0 + Double(original[$1 * 16640 + p]) } / 14
      XCTAssertEqual(
        Double(mean.contents().assumingMemoryBound(to: Float.self)[p]), expected, accuracy: 2e-5)
    }
    let qem = dir.appendingPathComponent("saved.qem")
    try resident.saveQEM(to: qem)
    let restored = try MetalEMPADResidentSource.load(
      NativeEMPADSource.open(qem), device: device, memoryBudgetBytes: 512 << 20)
    XCTAssertEqual(restored.scanValidity.invalidFrames, [7])
    XCTAssertEqual(restored.sourceIdentitySHA256, resident.sourceIdentitySHA256)
    XCTAssertEqual(
      restored.scanValidity.metadata["scanValidity"], resident.scanValidity.metadata["scanValidity"]
    )
    XCTAssertNotNil(restored.source.microscopeMetadata["scanValidity"])
    let dp = try XCTUnwrap(device.makeBuffer(length: 16384 * 4, options: .storageModeShared))
    let verify = try XCTUnwrap(queue.makeCommandBuffer())
    try restored.encodeDiffraction(scanRow: 1, scanColumn: 2, into: dp, command: verify)
    verify.commit()
    verify.waitUntilCompleted()
    XCTAssertEqual(dp.contents().assumingMemoryBound(to: UInt32.self)[0], 0x7fc0_1234)
    // Raw acquisition remained byte-for-byte unchanged.
    XCTAssertEqual(try Data(contentsOf: raw), original.withUnsafeBytes { Data($0) })
  }

  func testAdjacentAndBoundaryGapsAreNotInterpolated() {
    let report = EMPADScanValidity(scanRows: 5, scanColumns: 6, invalidFrames: [0, 8, 9, 22])
    XCTAssertEqual(report.interpolatedFrames, [22])
    XCTAssertEqual(report.neighbors(of: 22), [16, 28, 21, 23])
  }

  func testDPCMeanIgnoresMissingPositionAndRetainsGap() {
    let row: [Float] = [1, .nan, 3]
    let column: [Float] = [4, .nan, 8]
    var r = [Float](repeating: 0, count: 3)
    var c = r
    let valid = row.withUnsafeBufferPointer { rp in
      column.withUnsafeBufferPointer { cp in
        r.withUnsafeMutableBufferPointer { rr in
          c.withUnsafeMutableBufferPointer { cc in
            ScanVectorStatistics.center(
              row: rp.baseAddress!, column: cp.baseAddress!, count: 3,
              intoRow: rr.baseAddress!, intoColumn: cc.baseAddress!)
          }
        }
      }
    }
    XCTAssertEqual(valid, 2)
    XCTAssertEqual(r[0], -1)
    XCTAssertEqual(r[2], 1)
    XCTAssertEqual(c[0], -2)
    XCTAssertEqual(c[2], 2)
    XCTAssertTrue(r[1].isNaN && c[1].isNaN)
  }
}
