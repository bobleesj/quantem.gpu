import Metal
import Metal4DSTEMStreamingIO
import XCTest

final class MetalEncodedSourceTests: XCTestCase {
  func testFullDetectorReadsPreserveNativeCountsAcrossChunks() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Requires Metal")
    }
    let frames = 530
    let rows = 7
    let columns = 9
    let pixels = rows * columns
    for itemBytes in [1, 2] {
      let source = try MetalEncodedSource(
        shape: [1, frames, rows, columns], itemBytes: itemBytes, device: device)
      defer { source.releaseResidentStorage() }
      let values = (0..<(frames * pixels)).map { index -> UInt16 in
        let frame = index / pixels
        let pixel = index % pixels
        if pixel == 0 { return 0 }
        if pixel == 1 { return itemBytes == 1 ? 255 : 65535 }
        return UInt16((frame * 37 + pixel * 13) % (itemBytes == 1 ? 256 : 65536))
      }
      for chunk in [0..<517, 517..<frames] {
        let raw = try XCTUnwrap(
          device.makeBuffer(length: chunk.count * pixels * itemBytes, options: .storageModeShared))
        for (index, value) in values[(chunk.lowerBound * pixels)..<(chunk.upperBound * pixels)]
          .enumerated()
        {
          if itemBytes == 1 {
            raw.contents().storeBytes(of: UInt8(value), toByteOffset: index, as: UInt8.self)
          } else {
            raw.contents().storeBytes(of: value, toByteOffset: index * 2, as: UInt16.self)
          }
        }
        try source.append(raw, frames: chunk.count, verify: true)
      }
      for range in [0..<frames, 510..<523, 518..<frames] {
        let decoded = try source.read(range)
        let actual: [UInt16]
        if itemBytes == 1 {
          actual = Array(
            UnsafeBufferPointer(
              start: decoded.contents().assumingMemoryBound(to: UInt8.self),
              count: range.count * pixels)
          ).map(UInt16.init)
        } else {
          actual = Array(
            UnsafeBufferPointer(
              start: decoded.contents().assumingMemoryBound(to: UInt16.self),
              count: range.count * pixels))
        }
        XCTAssertEqual(
          actual, Array(values[(range.lowerBound * pixels)..<(range.upperBound * pixels)]))
      }
    }
  }
}
