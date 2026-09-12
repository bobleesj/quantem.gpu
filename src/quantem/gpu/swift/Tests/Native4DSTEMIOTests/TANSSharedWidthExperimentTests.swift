import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Local one-factor experiment, not a default-selection or native FPS test.
final class TANSSharedWidthExperimentTests: XCTestCase {
  func testAll66SharedWidthABBAWhenConfigured() throws {
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_WIDTH_FIXTURE"] else {
      throw XCTSkip("Requires complete all-66 entropy fixture and uncontended Metal device")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.releaseResidentStorage() }
    let valid = source.validDetectorMask
    let sourceBytes = source.residentBytes
    let prepared = ProcessInfo.processInfo.systemUptime
    try source.prepareExperimentalTileIndex(maximumIndexBytes: 2 << 30)
    source.experimentalUseTileIndex = true
    source.experimentalDetectorStreamsPerLane = 32
    print(
      "WIDTH_LOAD acquisitions=66 shape=512x512x192x192 dtype=uint16 scan_bin=1 detector_bin=1 crop=none representation=entropy source_bytes=\(sourceBytes) index_bytes=\(source.experimentalTileIndexBytes) load_s=\(source.loadSeconds) index_s=\(ProcessInfo.processInfo.systemUptime-prepared) source_pages=unspecified route=package_not_headed"
    )
    fflush(stdout)
    func hashes(_ images: [MTLBuffer]) -> [String] {
      images.map { image in
        SHA256.hash(
          data: Data(
            bytesNoCopy: image.contents(), count: image.length,
            deallocator: .none)
        ).map { String(format: "%02x", $0) }.joined()
      }
    }
    func mask(_ row: Double, _ col: Double, _ inner: Double, _ outer: Double) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - col
        let d = r * r + c * c
        return valid[q] != 0 && d >= inner * inner && d <= outer * outer ? 1 : 0
      }
    }
    // Bind the reference path to frozen independent full Linux products first.
    let frozen = try Data(
      contentsOf: URL(fileURLWithPath: path)
        .appendingPathComponent("series-products/shared-resident-images.npy"),
      options: .mappedIfSafe)
    XCTAssertEqual(
      SHA256.hash(data: frozen).map { String(format: "%02x", $0) }.joined(),
      "06b96168b62d89ba5a893416dcf7b0000d34647310da91db243ceab945cf89ac")
    let offset = 10 + Int(frozen[8]) + (Int(frozen[9]) << 8)
    XCTAssertEqual(frozen.count - offset, 66 * 4 * 512 * 512 * 4)
    for (product, inner, outer) in [(0, 0.0, 28.0), (1, 40.0, 80.0), (2, 14.0, 28.0)] {
      try autoreleasepool {
        let images = try source.detectorImages(
          mask: mask(95.5, 95.5, inner, outer),
          maximumAdditionalBytes: 1 << 30, rebase: true)
        for acquisition in 0..<66 {
          let start = offset + (acquisition * 4 + product) * 512 * 512 * 4
          let image = images[acquisition]
          let actual = Data(bytesNoCopy: image.contents(), count: image.length, deallocator: .none)
          XCTAssertEqual(
            actual, frozen.subdata(in: start..<(start + image.length)),
            "Frozen product \(product) acquisition \(acquisition)")
        }
      }
    }
    // Large jumps, edge crossing, resizing and a smaller delta are distinct cases.
    let cases = [
      ("BF_jump", 0.0, 28.0, 119.5, 119.5, 28.0),
      ("ABF_jump", 14.0, 28.0, 119.5, 119.5, 28.0),
      ("ADF_jump", 40.0, 80.0, 119.5, 119.5, 80.0),
      ("ADF_edge", 40.0, 80.0, 159.5, 159.5, 80.0),
      ("BF_resize", 0.0, 28.0, 95.5, 95.5, 40.0),
      ("ADF_resize", 40.0, 80.0, 95.5, 95.5, 96.0),
      ("ADF_small", 40.0, 80.0, 96.5, 96.5, 80.0),
    ]
    let bases = cases.map { mask(95.5, 95.5, $0.1, $0.2) }
    let targets = cases.map { mask($0.3, $0.4, $0.1, $0.5) }
    let references = try targets.map { target in
      try autoreleasepool {
        hashes(
          try source.detectorImages(
            mask: target, maximumAdditionalBytes: 1 << 30,
            rebase: true))
      }
    }
    let indexBytes = source.experimentalTileIndexBytes
    for (arm, width) in [("A1", 128), ("B64", 64), ("B256", 256), ("A2", 128)] {
      source.experimentalSharedModelThreadgroupWidth = width
      for i in cases.indices {
        // One untimed complete warmup per compiled candidate/case.
        try autoreleasepool {
          _ = try source.detectorImages(
            mask: bases[i], maximumAdditionalBytes: 1 << 30,
            rebase: true)
          _ = try source.detectorImages(mask: targets[i], maximumAdditionalBytes: 1 << 30)
        }
        for cycle in 0..<5 {
          try autoreleasepool {
            let prior = try source.detectorImages(
              mask: bases[i], maximumAdditionalBytes: 1 << 30,
              rebase: true)
            let priorHashes = hashes(prior)
            let started = ProcessInfo.processInfo.systemUptime
            let images = try source.detectorImages(
              mask: targets[i], maximumAdditionalBytes: 1 << 30)
            let wall = (ProcessInfo.processInfo.systemUptime - started) * 1000
            let gpu = source.lastDetectorGPUSeconds * 1000
            XCTAssertEqual(images.count, 66)
            XCTAssertEqual(hashes(images), references[i], "\(arm) \(cases[i].0) \(cycle)")
            XCTAssertEqual(hashes(prior), priorHashes, "Prior publication is immutable")
            XCTAssertEqual(source.experimentalTileIndexBytes, indexBytes)
            print(
              "WIDTH_SAMPLE arm=\(arm) threads=\(width) case=\(cases[i].0) cycle=\(cycle) wall_ms=\(wall) gpu_ms=\(gpu) columns=\(source.lastDetectorDecodedColumns) tile_fields=\(source.lastDetectorTileFields) scratch_bytes=\(source.lastDetectorScratchBytes) allocated_bytes=\(device.currentAllocatedSize) images=66 exact=true"
            )
            fflush(stdout)
          }
        }
      }
    }
  }
}
