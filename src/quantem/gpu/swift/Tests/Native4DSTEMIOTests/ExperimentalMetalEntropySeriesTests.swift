import CryptoKit
import Foundation
import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMStreamingIO
import XCTest

final class ExperimentalMetalEntropySeriesTests: XCTestCase {
  func testRealSPIIndexDeltaSelectionAndReleaseWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_SPI_FIXTURE"] else {
      throw XCTSkip(
        "Requires the complete sealed 66-acquisition archive and 128 GB-class Metal device")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let capacity = min(
      device.recommendedMaxWorkingSetSize,
      ProcessInfo.processInfo.physicalMemory * 4 / 5)
    let allocated = UInt64(device.currentAllocatedSize)
    let source = try ExperimentalMetalEntropySeries(
      directory: URL(fileURLWithPath: path),
      acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: capacity > allocated ? capacity - allocated : 0)
    defer { source.release() }
    XCTAssertEqual(source.shape, [66, 512, 512, 192, 192])
    let valid = source.validDetectorMask
    func mask(_ row: Double, _ column: Double, _ inner: Double, _ outer: Double) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - column
        let radius2 = r * r + c * c
        return valid[q] != 0 && radius2 >= inner * inner && radius2 <= outer * outer ? 1 : 0
      }
    }
    func hashes(_ images: [MTLBuffer]) -> [String] {
      images.map { buffer in
        SHA256.hash(
          data: Data(
            bytesNoCopy: buffer.contents(), count: buffer.length,
            deallocator: .none)
        ).map { String(format: "%02x", $0) }.joined()
      }
    }
    let masks = [
      mask(95.5, 95.5, 0, 28), mask(95.5, 95.5, 14, 28),
      mask(95.5, 95.5, 40, 80), mask(96.5, 96.5, 40, 81),
      mask(0, 0, 0, 50), mask(191, 191, 20, 80),
    ]
    let references = try masks.map { selection in
      try autoreleasepool {
        hashes(
          try source.detectorImages(
            mask: selection,
            maximumAdditionalBytes: 1 << 30, rebase: true))
      }
    }
    let sourceBytes = source.residentBytes
    try source.prepareDetectorIndex(maximumIndexBytes: 2 << 30)
    XCTAssertGreaterThan(source.detectorIndexBytes, 0)
    XCTAssertLessThanOrEqual(source.detectorIndexBytes, 2 << 30)
    XCTAssertGreaterThan(source.residentBytes, sourceBytes)
    for (index, selection) in masks.enumerated() {
      try autoreleasepool {
        let result = try source.detectorImages(
          mask: selection,
          maximumAdditionalBytes: 1 << 30)
        XCTAssertEqual(result.count, 66)
        XCTAssertEqual(hashes(result), references[index])
        let prior = hashes(result)
        XCTAssertThrowsError(
          try source.detectorImages(
            mask: selection,
            maximumAdditionalBytes: 1 << 30, selectedAcquisitions: [0, 0]))
        XCTAssertThrowsError(try source.prepareDetectorIndex(maximumIndexBytes: 1))
        XCTAssertEqual(hashes(result), prior, "Failure must preserve published buffers")
      }
    }
    for selection in [[65, 0, 17], Array((0..<66).reversed()), [0]] {
      let result = try source.detectorImages(
        mask: masks[0],
        maximumAdditionalBytes: 1 << 30, selectedAcquisitions: selection)
      XCTAssertEqual(hashes(result), selection.map { references[0][$0] })
    }
    let restored = try source.detectorImages(mask: masks[0], maximumAdditionalBytes: 1 << 30)
    XCTAssertEqual(
      hashes(restored), references[0], "Subset/full transitions must not reuse stale seeds")
    let dp = try source.diffractionImages(scanRow: 511, scanColumn: 511)
    XCTAssertEqual(dp.count, 66)
    XCTAssertTrue(dp.allSatisfy { $0.length == 192 * 192 * 4 })
    source.release()
    XCTAssertEqual(source.residentBytes, 0)
    XCTAssertThrowsError(try source.diffractionImages(scanRow: 0, scanColumn: 0))
    XCTAssertThrowsError(try source.detectorImages(mask: masks[0], maximumAdditionalBytes: 1 << 30))
  }

  /// The atlas is built into temporary outputs: a buffer returned before the
  /// build is unchanged after it, and seeded and unseeded queries afterwards,
  /// including ones that start from a stored image, equal an unseeded
  /// recompute made before the atlas existed.
  func testRealSPIAtlasKeepsReturnedImagesAndStaysExactWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_SPI_FIXTURE"] else {
      throw XCTSkip(
        "Requires the complete sealed 66-acquisition archive and 128 GB-class Metal device")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let capacity = min(
      device.recommendedMaxWorkingSetSize,
      ProcessInfo.processInfo.physicalMemory * 4 / 5)
    let allocated = UInt64(device.currentAllocatedSize)
    let source = try ExperimentalMetalEntropySeries(
      directory: URL(fileURLWithPath: path),
      acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: capacity > allocated ? capacity - allocated : 0)
    defer { source.release() }
    let valid = source.validDetectorMask
    func mask(_ row: Double, _ column: Double) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - column
        let radius2 = r * r + c * c
        return valid[q] != 0 && radius2 >= 40 * 40 && radius2 <= 80 * 80 ? 1 : 0
      }
    }
    func hashes(_ images: [MTLBuffer]) -> [String] {
      images.map { buffer in
        SHA256.hash(
          data: Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none)
        ).map { String(format: "%02x", $0) }.joined()
      }
    }
    try source.prepareDetectorIndex(maximumIndexBytes: 2 << 30)
    source.configureInteractiveGrouping(mixedModelTails: true, chooseCheaperBase: true)
    // The first target is 2 px from the published geometry and 0.2 px from a
    // stored centre, so the planner starts it from the atlas; the second is a
    // small seeded step after it.
    let targets = [mask(97.3, 97.4), mask(97.1, 97.8)]
    let references = try targets.map { target in
      try autoreleasepool {
        hashes(
          try source.detectorImages(mask: target, maximumAdditionalBytes: 1 << 30, rebase: true))
      }
    }
    let published = try source.detectorImages(
      mask: mask(95.5, 96.5), maximumAdditionalBytes: 1 << 30)
    let publishedHashes = hashes(published)
    XCTAssertThrowsError(try source.appendDetectorAtlasField(mask: mask(95.5, 95.5)))
    try source.beginDetectorAtlas(maximumBytes: 1 << 30)
    for (row, column) in [(95.5, 95.5), (95.5, 97.5), (97.5, 95.5), (97.5, 97.5)] {
      try source.appendDetectorAtlasField(mask: mask(row, column))
    }
    XCTAssertEqual(source.detectorAtlasFieldCount, 4)
    XCTAssertGreaterThan(source.detectorAtlasBytes, 0)
    XCTAssertEqual(
      hashes(published), publishedHashes, "Building the atlas must not write a returned image")
    var atlasFrames = 0
    for (index, target) in targets.enumerated() {
      try autoreleasepool {
        let seeded = try source.detectorImages(mask: target, maximumAdditionalBytes: 1 << 30)
        if source.lastDetectorAtlasField != nil { atlasFrames += 1 }
        XCTAssertEqual(hashes(seeded), references[index], "seeded query \(index)")
      }
    }
    XCTAssertGreaterThan(atlasFrames, 0, "The planner never started from the atlas")
    let unseeded = try source.detectorImages(
      mask: targets[0], maximumAdditionalBytes: 1 << 30, rebase: true)
    XCTAssertEqual(hashes(unseeded), references[0], "unseeded query with the atlas")
    XCTAssertThrowsError(try source.appendDetectorAtlasField(mask: [1, 0]))
  }
}
