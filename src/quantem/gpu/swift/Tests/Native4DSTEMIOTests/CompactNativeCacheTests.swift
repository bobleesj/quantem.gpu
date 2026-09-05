import Foundation
import XCTest

@testable import Metal4DSTEMStreamingIO

final class CompactNativeCacheTests: XCTestCase {
  func testSignatureBindsIndexBytesEvenWhenSourceSizeAndMtimeMatch() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    var bytes = Data(repeating: 0, count: 128)
    var offset = UInt32(48).littleEndian
    var count = UInt32(76).littleEndian
    withUnsafeBytes(of: &offset) { bytes.replaceSubrange(16..<20, with: $0) }
    withUnsafeBytes(of: &count) { bytes.replaceSubrange(20..<24, with: $0) }
    let date = Date(timeIntervalSince1970: 1)
    try bytes.write(to: url)
    try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: url.path)
    let original = try CompactNativeCache.signature(sourceURL: url)
    XCTAssertEqual(try CompactNativeCache.signature(sourceURL: url), original)
    bytes[80] ^= 1
    try bytes.write(to: url)
    try FileManager.default.setAttributes([.modificationDate: date], ofItemAtPath: url.path)
    XCTAssertNotEqual(try CompactNativeCache.signature(sourceURL: url), original)
  }

  func testCacheHeaderRejectsTruncationChecksumAndRangeMismatch() throws {
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: directory) }
    let url = directory.appendingPathComponent("fixture.qgmc")
    let data = Data(repeating: 0, count: CompactNativeCache.headerBytes + 8)
    try data.write(to: url)
    let file = try FileHandle(forUpdating: url)
    defer { try? file.close() }
    let cache = CompactNativeCache(
      schema: "quantem.gpu.native-compact-cache/v1",
      sourceSignature: String(repeating: "1", count: 64),
      sourceManifestSHA256: String(repeating: "2", count: 64),
      fileBytes: UInt64(data.count),
      shards: [
        CompactNativeCache.Shard(
          payloadOffset: UInt64(CompactNativeCache.headerBytes), payloadBytes: 4,
          payloadSHA256: CompactNativeCache.digest(Data(repeating: 0, count: 4)),
          descriptorsOffset: UInt64(CompactNativeCache.headerBytes + 4), descriptorsBytes: 4,
          descriptorsSHA256: CompactNativeCache.digest(Data(repeating: 0, count: 4))
        )
      ]
    )
    try cache.writeHeader(to: file)
    try file.seek(toOffset: 0)
    XCTAssertEqual(try CompactNativeCache.read(from: file).fileBytes, UInt64(data.count))
    let original = try Data(contentsOf: url)
    var changed = original
    changed[50] ^= 1
    try changed.write(to: url)
    try file.seek(toOffset: 0)
    XCTAssertThrowsError(try CompactNativeCache.read(from: file))
    try original.write(to: url)
    try file.truncate(atOffset: UInt64(data.count - 1))
    try file.seek(toOffset: 0)
    XCTAssertThrowsError(try CompactNativeCache.read(from: file))
    try file.truncate(atOffset: 12)
    try file.seek(toOffset: 0)
    XCTAssertThrowsError(try CompactNativeCache.read(from: file))
  }

  func testCacheHeaderRejectsOverlappingRangesEvenWithValidMetadataHash() throws {
    let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: url) }
    try Data(repeating: 0, count: CompactNativeCache.headerBytes + 8).write(to: url)
    let file = try FileHandle(forUpdating: url)
    defer { try? file.close() }
    let cache = CompactNativeCache(
      schema: "quantem.gpu.native-compact-cache/v1",
      sourceSignature: String(repeating: "1", count: 64),
      sourceManifestSHA256: String(repeating: "2", count: 64),
      fileBytes: UInt64(CompactNativeCache.headerBytes + 8),
      shards: [
        CompactNativeCache.Shard(
          payloadOffset: UInt64(CompactNativeCache.headerBytes), payloadBytes: 4,
          payloadSHA256: String(repeating: "3", count: 64),
          descriptorsOffset: UInt64(CompactNativeCache.headerBytes), descriptorsBytes: 4,
          descriptorsSHA256: String(repeating: "4", count: 64)
        )
      ]
    )
    try cache.writeHeader(to: file)
    try file.seek(toOffset: 0)
    XCTAssertThrowsError(try CompactNativeCache.read(from: file))
  }
}
