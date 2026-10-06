import Foundation
import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMKernels
import XCTest

@testable import Metal4DSTEMStreamingIO

final class TANSExactTileIndexTests: XCTestCase {
  func testBlockedWidthsConstantsAndAcquisitionOffsetsAreExact() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    let images = try (0..<2).map { acquisition -> MTLBuffer in
      let values: [UInt32] = (0..<262144).map { i in
        let width = (i / 256 + acquisition) % 33
        let maximum: UInt32 = width == 32 ? .max : (1 << width) - 1
        let base = UInt32.max - maximum
        return base + (UInt32(i) &* 2_654_435_761 & maximum)
      }
      return try values.withUnsafeBytes { bytes in
        try XCTUnwrap(
          device.makeBuffer(
            bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared))
      }
    }
    let index = try TANSExactTileIndex(
      device: device, queue: queue, library: library, blockedPacking: true)
    try index.append(tile: .init(row: 0, col: 0, side: 32), images: images, maximumBytes: 4 << 20)
    XCTAssertLessThan(index.residentBytes, images.reduce(0) { $0 + $1.length })
    let command = try XCTUnwrap(queue.makeCommandBuffer())
    try index.encodeAdd(
      command: command, images: images, acquisitions: [0, 1], selected: [(0, -1)])
    command.commit()
    command.waitUntilCompleted()
    XCTAssertEqual(command.status, .completed)
    for image in images {
      let values = UnsafeBufferPointer(
        start: image.contents().assumingMemoryBound(to: UInt32.self), count: 262144)
      XCTAssertTrue(values.allSatisfy { $0 == 0 })
    }
  }

  func testEveryWidthRoundtripsAndSignedAdditionIsExact() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    for width in 1...32 {
      try autoreleasepool {
        let maximum: UInt32 = width == 32 ? .max : (1 << width) - 1
        let values = (0..<262144).map { UInt32($0) &* 2_654_435_761 & maximum }
        let image = try values.withUnsafeBytes { bytes in
          try XCTUnwrap(
            device.makeBuffer(
              bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared))
        }
        image.contents().storeBytes(of: maximum, as: UInt32.self)
        let index = try TANSExactTileIndex(device: device, queue: queue, library: library)
        try index.append(
          tile: .init(row: 0, col: 0, side: 32), images: [image], maximumBytes: 16 << 20)
        XCTAssertEqual(index.fields[0].width, UInt32(width))
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        try index.encodeAdd(
          command: command, images: [image], acquisitions: [0], selected: [(0, -1)])
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed)
        let result = UnsafeBufferPointer(
          start: image.contents().assumingMemoryBound(to: UInt32.self), count: 262144)
        XCTAssertTrue(result.allSatisfy { $0 == 0 }, "signed exact width \(width)")
      }
    }
  }

  func testBudgetRejectsWithoutPublishingAField() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let index = try TANSExactTileIndex(
      device: device, queue: queue,
      library: Metal4DSTEMKernels.makeTANSLibrary(device: device))
    let image = try XCTUnwrap(device.makeBuffer(length: 262144 * 4, options: .storageModeShared))
    memset(image.contents(), 255, image.length)
    XCTAssertThrowsError(
      try index.append(
        tile: .init(row: 0, col: 0, side: 32),
        images: [image], maximumBytes: 4))
    XCTAssertEqual(index.residentBytes, 0)
  }
}
