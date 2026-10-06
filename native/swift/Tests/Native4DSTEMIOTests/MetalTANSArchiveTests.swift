import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

final class MetalTANSArchiveTests: XCTestCase {
  func testPublishedProfilesAreExplicitAndUnknownFormatsFailClosed() {
    XCTAssertEqual(TANSArchive.profile(format: "metal-entropy-source-v1"), .source)
    XCTAssertEqual(TANSArchive.profile(format: "metal-entropy-prepared-v1"), .prepared)
    XCTAssertNil(TANSArchive.profile(format: "metal-entropy-source-v2"))
    XCTAssertNil(TANSArchive.profile(format: ""))
  }

  func testArchiveShapeByteCountsRejectOverflowAndUnsupportedDtypes() throws {
    XCTAssertEqual(
      try TANSArchive.byteCount(shape: [512, 512, 192, 192], dtype: "<u2"), 19_327_352_832)
    XCTAssertThrowsError(try TANSArchive.byteCount(shape: [Int.max, 2], dtype: "<u2"))
    XCTAssertThrowsError(try TANSArchive.byteCount(shape: [-1], dtype: "<u4"))
    XCTAssertThrowsError(try TANSArchive.byteCount(shape: [0], dtype: "<u4"))
    XCTAssertThrowsError(try TANSArchive.byteCount(shape: [1], dtype: ">u2"))
  }

  func testEntropyOffsetsRejectOverrunAndBrokenTerminal() throws {
    let dense: [UInt32] = [0, 0, 0, 0, 0, 0, 0, 0, 0]
    try dense.withUnsafeBytes {
      try TANSStreamValidation.offsets($0, columns: 1, sparse: false, limit: 32)
    }
    XCTAssertThrowsError(
      try dense.withUnsafeBytes {
        try TANSStreamValidation.offsets($0, columns: 1, sparse: false, limit: 31)
      })
    let sparse: [UInt32] = [0, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    XCTAssertThrowsError(
      try sparse.withUnsafeBytes {
        try TANSStreamValidation.offsets($0, columns: 1, sparse: true, limit: 0)
      })
    XCTAssertThrowsError(
      try dense.withUnsafeBytes {
        try TANSStreamValidation.offsets($0, columns: 2, sparse: false, limit: 32)
      })
  }

  func testMalformedArchiveMetadataFailsBeforeResidentPublicationWhenConfigured() throws {
    guard let fixture = ProcessInfo.processInfo.environment["QUANTEM_TANS_ARCHIVE_FIXTURE"] else {
      throw XCTSkip("Metadata rejection test requires a sealed archive")
    }
    let original = URL(fileURLWithPath: fixture)
    let checkpoint = try Data(contentsOf: original.appendingPathComponent("checkpoint.json"))
    let root = FileManager.default.temporaryDirectory.appendingPathComponent(
      "tans-negative-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: root) }
    try FileManager.default.linkItem(
      at: original.appendingPathComponent("global-state.npz"),
      to: root.appendingPathComponent("global-state.npz"))
    let cases = [
      "incomplete", "wrongEndian", "wrongShape", "overflowShape", "missingChunk",
      "overlappingComponents", "wrongDtype", "escapedGlobal", "wrongGlobalHash", "wrongNPYShape",
      "missingSemanticContract", "sourceMaskApplied", "wrongSourceCodec",
    ]
    for name in cases {
      var json = try XCTUnwrap(JSONSerialization.jsonObject(with: checkpoint) as? [String: Any])
      var layout = try XCTUnwrap(json["layout"] as? [String: Any])
      switch name {
      case "missingSemanticContract":
        json.removeValue(forKey: "semantics")
        json.removeValue(forKey: "index_rebuild")
      case "sourceMaskApplied", "wrongSourceCodec":
        if var semantics = json["semantics"] as? [String: Any] {
          if name == "sourceMaskApplied" {
            semantics["apply_mask_to_source"] = true
          } else {
            semantics["codec"] = "unsupported"
          }
          json["semantics"] = semantics
        } else {
          var contract = try XCTUnwrap(json["index_rebuild"] as? [String: Any])
          if name == "sourceMaskApplied" {
            contract["source_mask_applied"] = true
          } else {
            contract["source_codec"] = "unsupported"
          }
          json["index_rebuild"] = contract
        }
      case "incomplete": json["complete"] = false
      case "wrongEndian": layout["byte_order"] = "big"
      case "wrongShape": layout["shape"] = [66, 511, 512, 192, 192]
      case "overflowShape": layout["shape"] = [Int.max, 512, 512, 192, 192]
      case "missingChunk":
        var chunks = try XCTUnwrap(layout["chunks"] as? [[String: Any]])
        chunks.removeFirst()
        layout["chunks"] = chunks
      case "overlappingComponents", "wrongDtype":
        var chunks = try XCTUnwrap(layout["chunks"] as? [[String: Any]])
        var components = try XCTUnwrap(chunks[0]["components"] as? [[String: Any]])
        if name == "overlappingComponents" {
          components[1]["offset"] = 0
        } else {
          components[0]["dtype"] = "<f8"
        }
        chunks[0]["components"] = components
        layout["chunks"] = chunks
      case "escapedGlobal": json["global_state_file"] = "../global-state.npz"
      case "wrongGlobalHash": json["global_state_file_sha256"] = String(repeating: "0", count: 64)
      case "wrongNPYShape":
        var globals = try XCTUnwrap(json["global_state"] as? [String: [String: Any]])
        globals["planner__valid"]?["shape"] = [192, 192]
        json["global_state"] = globals
      default: XCTFail("Unrecognized rejection case")
      }
      json["layout"] = layout
      try JSONSerialization.data(withJSONObject: json).write(
        to: root.appendingPathComponent("checkpoint.json"))
      XCTAssertThrowsError(try TANSArchive(directory: root, acquisitions: [0]), name)
    }
    try checkpoint.write(to: root.appendingPathComponent("checkpoint.json"))
    XCTAssertThrowsError(try TANSArchive(directory: root, acquisitions: [0, 0]))
    XCTAssertThrowsError(try TANSArchive(directory: root, acquisitions: [66]))
    XCTAssertThrowsError(try TANSArchive(directory: root, acquisitions: []))
    let archive = try TANSArchive(directory: root, acquisitions: [0])
    try Data([1, 2, 3, 4]).write(to: root.appendingPathComponent("data-0.bin"))
    XCTAssertThrowsError(try archive.read(archive.chunks[0]), "Truncated payload must fail closed")
    let outside = root.appendingPathComponent("escape.npz")
    try FileManager.default.createSymbolicLink(
      at: outside, withDestinationURL: original.appendingPathComponent("global-state.npz"))
    XCTAssertThrowsError(try TANSArchive.child("escape.npz", under: root))
  }

  private struct Oracle: Decodable {
    struct Record: Decodable {
      let scanRow: Int
      let scanColumn: Int
      let sha256: String
      let total: UInt64
      let maximum: UInt16
      let minimum: UInt16
    }
    let records: [Record]
  }

  func testConcurrentUploadFailureJoinsAndAllowsExactReuseWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_ARCHIVE_FIXTURE"] else {
      throw XCTSkip("Requires a sealed archive for joined failure and retry")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let queue = try XCTUnwrap(device.makeCommandQueue())
    let archive = try TANSArchive(directory: URL(fileURLWithPath: path), acquisitions: [0])
    let window = try TANSUploadWindow(
      archive: archive, device: device, queue: queue,
      stageBytes: archive.chunks[0].recordBytes, concurrency: 2)
    let good = archive.chunks[0]
    let bad = TANSArchive.Chunk(
      chunk: good.chunk, acquisition: good.acquisition, firstScan: good.firstScan,
      scanCount: good.scanCount, shard: good.shard, fileOffset: good.fileOffset,
      recordBytes: good.recordBytes, sha256: String(repeating: "0", count: 64),
      components: good.components)
    // A successful sibling may already be uploading when the other worker
    // rejects SHA. The window must join and drop both results before throwing.
    let before = device.currentAllocatedSize
    try autoreleasepool {
      let previous = try window.load([good])
      let beforeFailure = device.currentAllocatedSize
      XCTAssertThrowsError(try window.load([good, bad]))
      // Allow only small Metal runtime command/descriptor caches, not an
      // accidentally retained ~80 MB successful sibling from the failed load.
      XCTAssertLessThan(device.currentAllocatedSize, beforeFailure + 8 * 1024 * 1024)
      let loaded = try window.load([good])
      XCTAssertEqual(loaded.count, 1)
      for privateRecord in [
        try XCTUnwrap(previous.first?.resident), try XCTUnwrap(loaded.first?.resident),
      ] {
        let readback = try XCTUnwrap(
          device.makeBuffer(length: privateRecord.length, options: .storageModeShared))
        let command = try XCTUnwrap(queue.makeCommandBuffer())
        let blit = try XCTUnwrap(command.makeBlitCommandEncoder())
        blit.copy(
          from: privateRecord, sourceOffset: 0, to: readback, destinationOffset: 0,
          size: privateRecord.length)
        blit.endEncoding()
        command.commit()
        command.waitUntilCompleted()
        XCTAssertEqual(command.status, .completed)
        let bytes = Data(
          bytesNoCopy: readback.contents(), count: readback.length, deallocator: .none)
        try TANSArchive.verify(bytes, sha256: good.sha256)
      }
    }
    XCTAssertLessThan(device.currentAllocatedSize, before + 8 * 1024 * 1024)
  }

  func testDirectStagingRejectsUndersizedBufferAndOverflowingOffsetWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_ARCHIVE_FIXTURE"] else {
      throw XCTSkip("Requires sealed archive metadata")
    }
    let archive = try TANSArchive(directory: URL(fileURLWithPath: path), acquisitions: [0])
    let good = archive.chunks[0]
    let overflow = TANSArchive.Chunk(
      chunk: good.chunk, acquisition: good.acquisition, firstScan: good.firstScan,
      scanCount: good.scanCount, shard: good.shard, fileOffset: UInt64.max,
      recordBytes: 16, sha256: good.sha256, components: good.components)
    var sentinel = [UInt8](repeating: 0xA5, count: 16)
    var metrics = TANSArchive.ReadMetrics()
    try sentinel.withUnsafeMutableBytes { destination in
      XCTAssertThrowsError(try archive.read(good, into: destination, metrics: &metrics))
      XCTAssertThrowsError(try archive.read(overflow, into: destination, metrics: &metrics))
    }
    XCTAssertEqual(sentinel, [UInt8](repeating: 0xA5, count: 16))
    XCTAssertEqual(metrics.sourceBytesRead, 0)
  }

  func testTransferAdmissionRejectsInvalidWorkerCountsBeforeReading() throws {
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    for workers in [0, -1, 5, Int.max] {
      XCTAssertThrowsError(
        try MetalTANSResidentSeries(
          directory: URL(fileURLWithPath: "/nonexistent-tans-fixture"), acquisitions: [0],
          device: device, maximumAdditionalBytes: 0, transferConcurrency: workers))
    }
  }

  func testRealArchiveDiffractionMatchesIndependentOriginalCountsWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let path = environment["QUANTEM_TANS_ARCHIVE_FIXTURE"],
      let oraclePath = environment["QUANTEM_TANS_ORIGINAL_ORACLE"]
    else {
      throw XCTSkip("Provide a sealed archive and independent original-source DP oracle")
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Physical Metal device required")
    }
    let oracleBytes = try Data(contentsOf: URL(fileURLWithPath: oraclePath))
    try TANSArchive.verify(
      oracleBytes, sha256: "0dbc03e28c3ccc0361f3117668cea83bb836d04a8b4e98c9082a17b25388ea25")
    let oracle = try JSONDecoder().decode(Oracle.self, from: oracleBytes)
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: [0], device: device,
      maximumAdditionalBytes: 3 * 1024 * 1024 * 1024, transferConcurrency: 4)
    defer { source.releaseResidentStorage() }
    XCTAssertEqual(source.shape, [1, 512, 512, 192, 192])
    XCTAssertEqual(source.acquisitionIndices, [0])
    let residentBytes = source.residentBytes
    print(
      "TANS_LOAD load_seconds=\(source.loadSeconds) authenticated_read_seconds=\(source.readAndAuthenticationSeconds) private_upload_seconds=\(source.privateUploadSeconds) resident_bytes=\(residentBytes) acquisition_count=1 source_pages=unspecified prepared_archive=true"
    )
    for record in oracle.records {
      let values = try source.extractDiffraction(
        scanRow: record.scanRow, scanColumn: record.scanColumn)
      let digest = values.withUnsafeBytes {
        SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
      }
      XCTAssertEqual(digest, record.sha256, "Native DP (\(record.scanRow),\(record.scanColumn))")
      XCTAssertEqual(values.reduce(UInt64(0)) { $0 + UInt64($1) }, record.total)
      XCTAssertEqual(values.max(), record.maximum)
      XCTAssertEqual(values.min(), record.minimum)
      let image = try XCTUnwrap(
        source.diffractionImages(
          scanRow: record.scanRow, scanColumn: record.scanColumn
        ).first)
      let displayCounts = Array(
        UnsafeBufferPointer(
          start: image.contents().assumingMemoryBound(to: UInt32.self), count: 36864))
      XCTAssertEqual(
        displayCounts, values.map(UInt32.init), "Renderer must receive exact native counts")
      XCTAssertEqual(source.residentBytes, residentBytes)
      print(
        "TANS_DP row=\(record.scanRow) col=\(record.scanColumn) sha256=\(digest) gpu_seconds=\(source.lastQueryGPUSeconds)"
      )
    }
    source.releaseResidentStorage()
    XCTAssertEqual(source.residentBytes, 0)
    XCTAssertThrowsError(try source.extractDiffraction(scanRow: 0, scanColumn: 0))
  }

  func testCompleteNativeAcquisitionMatchesIndependentOriginalHashWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let path = environment["QUANTEM_TANS_ARCHIVE_FIXTURE"],
      let expected = environment["QUANTEM_TANS_FULL_ORIGINAL_SHA256"]
    else {
      throw XCTSkip("Full-count audit requires an independently frozen original HDF5 hash")
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw XCTSkip("Physical Metal device required")
    }
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: [0], device: device,
      maximumAdditionalBytes: 3 * 1024 * 1024 * 1024)
    defer { source.releaseResidentStorage() }
    let before = source.residentBytes
    let started = CFAbsoluteTimeGetCurrent()
    let actual = try source.auditFullCountSHA256(acquisitionIndex: 0)
    XCTAssertEqual(actual, expected, "Every native count, scan1/detector1/cropnone")
    XCTAssertEqual(source.residentBytes, before)
    print(
      "TANS_FULL_AUDIT sha256=\(actual) logical_bytes=19327352832 scratch_bytes=37748736 resident_bytes=\(before) audit_wall_seconds=\(CFAbsoluteTimeGetCurrent() - started) performance_claim=false"
    )
  }
}
