import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Explicitly opt-in capacity test, not a UI/FPS or original-source cold claim.
final class MetalTANSFullSeriesTests: XCTestCase {
  func testCompleteSealedSeriesLoadsAndQueriesAllAcquisitionsWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let path = environment["QUANTEM_TANS_FULL_SERIES_FIXTURE"] else {
      throw XCTSkip("Requires the complete hash-verified 66-acquisition entropy archive")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let root = URL(fileURLWithPath: path)
    let archive = try TANSArchive(directory: root, acquisitions: Array(0..<66))
    XCTAssertEqual(archive.metadata.layout.shape, [66, 512, 512, 192, 192])
    XCTAssertEqual(archive.chunks.count, 1056)
    for file in archive.metadata.layout.files {
      let attributes = try FileManager.default.attributesOfItem(
        atPath: root.appendingPathComponent(file.name).path)
      XCTAssertEqual((attributes[.size] as? NSNumber)?.uint64Value, file.nbytes)
    }
    let initialAllocated = device.currentAllocatedSize
    let recommended = device.recommendedMaxWorkingSetSize
    let capacity = min(recommended, ProcessInfo.processInfo.physicalMemory * 4 / 5)
    let budget = capacity > UInt64(initialAllocated) ? capacity - UInt64(initialAllocated) : 0
    print(
      "TANS66_PREFLIGHT device=\(device.name) physical_bytes=\(ProcessInfo.processInfo.physicalMemory) recommended_bytes=\(recommended) current_allocated_bytes=\(initialAllocated) budget_bytes=\(budget) max_buffer_bytes=\(device.maxBufferLength)"
    )
    let source: MetalTANSResidentSeries
    if environment["QUANTEM_TANS_TRANSFER_WORKERS"] == nil,
      environment["QUANTEM_TANS_READ_POLICY"] == nil
    {
      // Final acceptance exercises ordinary defaults without tuning arguments.
      source = try MetalTANSResidentSeries(
        directory: root, acquisitions: Array(0..<66), device: device,
        maximumAdditionalBytes: budget)
      XCTAssertEqual(source.transferConcurrency, 4)
      XCTAssertEqual(source.sourceReadPolicy, .avoidCaching)
    } else {
      source = try MetalTANSResidentSeries(
        directory: root, acquisitions: Array(0..<66), device: device,
        maximumAdditionalBytes: budget,
        transferConcurrency: try XCTUnwrap(
          Int(environment["QUANTEM_TANS_TRANSFER_WORKERS"] ?? "4")),
        sourceReadPolicy: try XCTUnwrap(
          TANSArchive.SourceReadPolicy(
            rawValue: environment["QUANTEM_TANS_READ_POLICY"] ?? "avoidCaching")))
    }
    defer { source.releaseResidentStorage() }
    XCTAssertEqual(source.shape, [66, 512, 512, 192, 192])
    XCTAssertEqual(source.acquisitionIndices, Array(0..<66))
    let residentBytes = source.residentBytes
    print(
      "TANS66_RESIDENT acquisition_count=66 record_count=1056 shape=66x512x512x192x192 dtype=uint16 scan_bin=1 detector_bin=1 crop=none raw_resident=false resident_bytes=\(residentBytes) allocated_bytes=\(device.currentAllocatedSize) load_seconds=\(source.loadSeconds) read_auth_seconds=\(source.readAndAuthenticationSeconds) upload_seconds=\(source.privateUploadSeconds) source_pages=unspecified prepared_archive=true"
    )
    print(
      "TANS66_STAGES io_seconds=\(source.readMetrics.ioSeconds) sha256_seconds=\(source.readMetrics.hashSeconds) structure_seconds=\(source.readMetrics.validationSeconds) workers=\(source.transferConcurrency) payload_read_policy=\(source.sourceReadPolicy.rawValue) stage_times_are_worker_sums=true"
    )
    XCTAssertEqual(source.readMetrics.authenticatedRecords, 1056)
    XCTAssertEqual(source.readMetrics.sourceBytesRead, 84_666_114_048)
    print(
      "TANS66_TRANSFER source_bytes_read=\(source.readMetrics.sourceBytesRead) authenticated_records=\(source.readMetrics.authenticatedRecords) encoded_staging_bytes=\(source.stagingBytes) explicit_payload_memcpy_bytes=0"
    )
    for (row, column) in [(0, 0), (256, 256), (511, 511)] {
      let start = CFAbsoluteTimeGetCurrent()
      let values = try source.extractDiffraction(scanRow: row, scanColumn: column)
      let wall = CFAbsoluteTimeGetCurrent() - start
      XCTAssertEqual(values.count, 66 * 36864)
      var hashes = Set<String>()
      for index in 0..<66 {
        let plane = Array(values[(index * 36864)..<((index + 1) * 36864)])
        let sha = plane.withUnsafeBytes {
          SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
        }
        hashes.insert(sha)
        let total = plane.reduce(UInt64(0)) { $0 + UInt64($1) }
        print(
          "TANS66_DP acquisition=\(index) row=\(row) column=\(column) sha256=\(sha) total=\(total) maximum=\(plane.max() ?? 0)"
        )
      }
      XCTAssertGreaterThan(hashes.count, 1, "Different source acquisitions must not reuse one DP")
      XCTAssertEqual(source.residentBytes, residentBytes)
      print(
        "TANS66_QUERY row=\(row) column=\(column) acquisitions=66 distinct_hashes=\(hashes.count) wall_seconds=\(wall) gpu_seconds=\(source.lastQueryGPUSeconds) ui_fps_claim=false"
      )
    }
    // Independent full-source hash was frozen before this source-only adapter.
    // Audit only acquisition index 0 here; no independent all-66 count claim.
    XCTAssertEqual(
      try source.auditFullCountSHA256(acquisitionIndex: 0),
      "7f43a32b35f82205c09c24078ed8a1c43ba72fe5d3de0823dd5f74ab5b85a68e")
    source.releaseResidentStorage()
    XCTAssertEqual(source.residentBytes, 0)
    print("TANS66_RELEASE resident_bytes=0 allocated_bytes=\(device.currentAllocatedSize)")
  }
}
