import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// A restored exact tile index must answer every query exactly like the index
/// it was exported from, and a corrupt or foreign cache must be refused.
final class TANSTileIndexCacheTests: XCTestCase {
  func testExportedIndexRestoresExactlyAndRejectsCorruptionWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_CACHE_FIXTURE"] else {
      throw XCTSkip("Requires the complete sealed 66-acquisition archive and a 128 GB-class device")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let budget =
      ProcessInfo.processInfo.physicalMemory * 4 / 5 - UInt64(device.currentAllocatedSize)
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: budget)
    defer { source.releaseResidentStorage() }
    let cache = FileManager.default.temporaryDirectory.appendingPathComponent(
      "tans-index-cache-\(UUID().uuidString)")
    defer { try? FileManager.default.removeItem(at: cache) }
    let buildStart = ProcessInfo.processInfo.systemUptime
    try source.prepareExperimentalTileIndex(maximumIndexBytes: 2 << 30)
    let buildSeconds = ProcessInfo.processInfo.systemUptime - buildStart
    source.experimentalUseTileIndex = true
    source.experimentalDetectorStreamsPerLane = 32
    let exportStart = ProcessInfo.processInfo.systemUptime
    try source.exportExperimentalTileIndex(to: cache)
    let exportSeconds = ProcessInfo.processInfo.systemUptime - exportStart
    let valid = source.validDetectorMask
    func mask(_ row: Double, _ col: Double, _ inner: Double, _ outer: Double) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - col
        let radius2 = r * r + c * c
        return valid[q] != 0 && radius2 >= inner * inner && radius2 <= outer * outer ? 1 : 0
      }
    }
    func hashes(_ images: [MTLBuffer]) -> [String] {
      images.map {
        SHA256.hash(data: Data(bytesNoCopy: $0.contents(), count: $0.length, deallocator: .none))
          .map { String(format: "%02x", $0) }.joined()
      }
    }
    let masks = [mask(95.5, 95.5, 0, 28), mask(95.5, 95.5, 40, 80), mask(96.5, 97.5, 40, 80)]
    var built: [[String]] = []
    var builtTiles: [Int] = []
    for (index, selection) in masks.enumerated() {
      built.append(
        hashes(
          try source.detectorImages(
            mask: selection, maximumAdditionalBytes: 1 << 30, rebase: index == 0)))
      builtTiles.append(source.lastDetectorTileFields)
    }
    XCTAssertGreaterThan(builtTiles[0], 0, "a full-mask query must use index fields")
    let builtBytes = source.experimentalTileIndexBytes

    // Restore into the same series after discarding the built index.
    try source.discardExperimentalTileIndex()
    let importStart = ProcessInfo.processInfo.systemUptime
    try source.importExperimentalTileIndex(from: cache, maximumIndexBytes: 2 << 30)
    let importSeconds = ProcessInfo.processInfo.systemUptime - importStart
    source.experimentalUseTileIndex = true
    XCTAssertEqual(source.experimentalTileIndexBytes, builtBytes)
    for (index, selection) in masks.enumerated() {
      let restored = hashes(
        try source.detectorImages(
          mask: selection, maximumAdditionalBytes: 1 << 30, rebase: index == 0))
      XCTAssertEqual(restored, built[index], "restored index must reproduce mask \(index) exactly")
      XCTAssertEqual(
        source.lastDetectorTileFields, builtTiles[index], "same tile plan for mask \(index)")
    }
    print(
      "TILE_INDEX_CACHE build_s=\(buildSeconds) export_s=\(exportSeconds) import_s=\(importSeconds) bytes=\(builtBytes) full_mask_fields=\(builtTiles[0]) exact=true"
    )
    fflush(stdout)

    // A wrong selection, a corrupt payload byte and a foreign manifest are all refused.
    let subset = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: [0, 1], device: device,
      maximumAdditionalBytes: 8 << 30)
    defer { subset.releaseResidentStorage() }
    XCTAssertThrowsError(
      try subset.importExperimentalTileIndex(from: cache, maximumIndexBytes: 2 << 30))
    let payloadURL = cache.appendingPathComponent(MetalTANSResidentSeries.tileIndexCachePayloadName)
    let handle = try FileHandle(forWritingTo: payloadURL)
    try handle.seek(toOffset: 4096)
    var original = [UInt8](repeating: 0, count: 1)
    let reader = try FileHandle(forReadingFrom: payloadURL)
    try reader.seek(toOffset: 4096)
    original = [UInt8](try XCTUnwrap(reader.read(upToCount: 1)))
    try reader.close()
    try handle.write(contentsOf: Data([original[0] ^ 0x01]))
    try handle.close()
    try source.discardExperimentalTileIndex()
    XCTAssertThrowsError(
      try source.importExperimentalTileIndex(from: cache, maximumIndexBytes: 2 << 30))
    XCTAssertEqual(source.experimentalTileIndexBytes, 0, "a refused cache leaves no index behind")
  }
}
