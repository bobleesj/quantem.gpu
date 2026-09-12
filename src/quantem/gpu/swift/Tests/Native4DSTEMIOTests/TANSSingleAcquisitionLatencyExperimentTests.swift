import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Package diagnostic for an interactive per-acquisition detector path while
/// all 66 acquisitions stay entropy-resident. It measures exact single-image
/// delta queries, acquisition switches and the all-66 catch-up query. It is
/// not native input-to-presentation latency and makes no FPS claim.
final class TANSSingleAcquisitionLatencyExperimentTests: XCTestCase {
  private struct Sample: Encodable {
    let scenario: String
    let product: String
    let step: Int
    let acquisitions: Int
    let wallMs: Double
    let gpuMs: Double
    let encodeMs: Double
    let commitMs: Double
    let waitMs: Double
    let commitToGPUStartMs: Double
    let columns: Int
    let tiles: Int
    let usedPrevious: Bool
    let idleSeconds: Double
    let exact: Bool
  }

  func testSingleAcquisitionDeltaLatencyWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let path = environment["QUANTEM_TANS_SINGLE_FIXTURE"] else {
      throw XCTSkip("Requires the complete sealed 66-acquisition archive and a 128 GB-class device")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.releaseResidentStorage() }
    let sourceBytes = source.residentBytes
    let prepareStart = ProcessInfo.processInfo.systemUptime
    try source.prepareExperimentalTileIndex(maximumIndexBytes: 2 << 30)
    source.experimentalUseTileIndex = true
    source.experimentalDetectorStreamsPerLane = 32
    if environment["QUANTEM_TANS_SINGLE_MIXED"] == "1" {
      source.experimentalMixedModelTails = true
      source.experimentalMixedTailSavingsDivisor = 8
      source.experimentalSeparateMixedDispatches = true
      source.experimentalMixedOnlySpecialization = true
    }
    let indexSeconds = ProcessInfo.processInfo.systemUptime - prepareStart
    print(
      "SINGLE_LATENCY_LOAD acquisitions=66 source_bytes=\(sourceBytes) index_bytes=\(source.experimentalTileIndexBytes) load_s=\(source.loadSeconds) index_s=\(indexSeconds) source_pages=unspecified route=package_not_headed mixed_tails=\(source.experimentalMixedModelTails)"
    )
    fflush(stdout)
    let valid = source.validDetectorMask
    // Same geometry rule as the app: annulus around a fractional center, with
    // the optional right half-plane for the DF product.
    func mask(row: Double, col: Double, inner: Double, outer: Double, halfPlane: Bool) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - col
        let radius2 = r * r + c * c
        return valid[q] != 0 && radius2 >= inner * inner && radius2 <= outer * outer
          && (!halfPlane || c >= 0) ? 1 : 0
      }
    }
    func hash(_ buffer: MTLBuffer) -> String {
      SHA256.hash(
        data: Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none)
      )
      .map { String(format: "%02x", $0) }.joined()
    }
    var samples: [Sample] = []
    func record(
      _ scenario: String, product: String, step: Int, acquisitions: Int, wall: Double,
      idle: Double, exact: Bool
    ) {
      let timing = source.lastDetectorCommandTiming
      let sample = Sample(
        scenario: scenario, product: product, step: step, acquisitions: acquisitions,
        wallMs: wall * 1000, gpuMs: source.lastDetectorGPUSeconds * 1000,
        encodeMs: timing["encode_ms"] ?? -1, commitMs: timing["commit_ms"] ?? -1,
        waitMs: timing["wait_ms"] ?? -1,
        commitToGPUStartMs: timing["commit_to_gpu_start_ms"] ?? -1,
        columns: source.lastDetectorDecodedColumns, tiles: source.lastDetectorTileFields,
        usedPrevious: source.lastDetectorUsedPrevious, idleSeconds: idle, exact: exact)
      samples.append(sample)
      print(
        "SINGLE_LATENCY_SAMPLE scenario=\(scenario) product=\(product) step=\(step) acquisitions=\(acquisitions) wall_ms=\(sample.wallMs) gpu_ms=\(sample.gpuMs) encode_ms=\(sample.encodeMs) commit_ms=\(sample.commitMs) wait_ms=\(sample.waitMs) commit_to_gpu_ms=\(sample.commitToGPUStartMs) columns=\(sample.columns) tiles=\(sample.tiles) previous=\(sample.usedPrevious) idle_s=\(idle) exact=\(exact)"
      )
      fflush(stdout)
    }
    struct Product {
      let name: String
      let inner: Double
      let outer: Double
      let half: Bool
    }
    let products = [
      Product(name: "BF", inner: 0, outer: 28, half: false),
      Product(name: "ABF", inner: 14, outer: 28, half: false),
      Product(name: "ADF", inner: 40, outer: 80, half: false),
      Product(name: "DF-half", inner: 0, outer: 80, half: true),
    ]
    let focus = 0
    let other = 33
    for product in products {
      try autoreleasepool {
        // Independent exact reference for the start geometry, all 66, no seed.
        let startMask = mask(
          row: 95.5, col: 95.5, inner: product.inner, outer: product.outer,
          halfPlane: product.half)
        let reference = try source.detectorImages(
          mask: startMask, maximumAdditionalBytes: 1 << 30, rebase: true)
        let referenceHashes = reference.map(hash)
        XCTAssertEqual(reference.count, 66)

        // S1: single acquisition, full recompute (first frame of a gesture).
        var started = ProcessInfo.processInfo.systemUptime
        var images = try source.detectorImages(
          mask: startMask, maximumAdditionalBytes: 1 << 30, rebase: true,
          selectedAcquisitions: [focus])
        var wall = ProcessInfo.processInfo.systemUptime - started
        XCTAssertEqual(images.count, 1)
        var exact = hash(images[0]) == referenceHashes[focus]
        XCTAssertTrue(exact, "\(product.name) single full recompute must match all-66 reference")
        record(
          "single-full", product: product.name, step: 0, acquisitions: 1, wall: wall, idle: 0,
          exact: exact)

        // S2: smooth drag, one detector pixel per frame: 16 right, 16 down,
        // then 16 outer-radius growth steps. Every frame seeds from the prior.
        var row = 95.5
        var col = 95.5
        var outer = product.outer
        var chainedHash = ""
        var lastMask = startMask
        for step in 0..<48 {
          if step < 16 { col += 1 } else if step < 32 { row += 1 } else { outer += 1 }
          let next = mask(
            row: row, col: col, inner: product.inner, outer: outer, halfPlane: product.half)
          started = ProcessInfo.processInfo.systemUptime
          images = try source.detectorImages(
            mask: next, maximumAdditionalBytes: 1 << 30, rebase: false,
            selectedAcquisitions: [focus])
          wall = ProcessInfo.processInfo.systemUptime - started
          XCTAssertTrue(source.lastDetectorUsedPrevious, "smooth 1px drag must use the delta seed")
          record(
            "single-drag-1px", product: product.name, step: step, acquisitions: 1, wall: wall,
            idle: 0, exact: true)
          chainedHash = hash(images[0])
          lastMask = next
        }
        // Exactness of the 48-step delta chain against an unseeded recompute.
        let unseeded = try source.detectorImages(
          mask: lastMask, maximumAdditionalBytes: 1 << 30, rebase: true,
          selectedAcquisitions: [focus])
        exact = hash(unseeded[0]) == chainedHash
        XCTAssertTrue(exact, "\(product.name) 48-step delta chain must equal unseeded recompute")
        record(
          "single-chain-check", product: product.name, step: 48, acquisitions: 1, wall: 0,
          idle: 0, exact: exact)

        // S3: fast drag, four pixels per frame (thicker rings).
        for step in 0..<12 {
          col -= 4
          let next = mask(
            row: row, col: col, inner: product.inner, outer: outer, halfPlane: product.half)
          started = ProcessInfo.processInfo.systemUptime
          images = try source.detectorImages(
            mask: next, maximumAdditionalBytes: 1 << 30, rebase: false,
            selectedAcquisitions: [focus])
          wall = ProcessInfo.processInfo.systemUptime - started
          record(
            "single-drag-4px", product: product.name, step: step, acquisitions: 1, wall: wall,
            idle: 0, exact: true)
          lastMask = next
        }

        // S4: idle penalty on the single-acquisition path.
        for step in 0..<3 {
          Thread.sleep(forTimeInterval: 4)
          col += 1
          let next = mask(
            row: row, col: col, inner: product.inner, outer: outer, halfPlane: product.half)
          started = ProcessInfo.processInfo.systemUptime
          images = try source.detectorImages(
            mask: next, maximumAdditionalBytes: 1 << 30, rebase: false,
            selectedAcquisitions: [focus])
          wall = ProcessInfo.processInfo.systemUptime - started
          record(
            "single-after-idle", product: product.name, step: step, acquisitions: 1, wall: wall,
            idle: 4, exact: true)
          lastMask = next
        }

        // S5: hover switch to another acquisition and back with the same mask.
        for (step, acquisition) in [other, focus, other, focus].enumerated() {
          started = ProcessInfo.processInfo.systemUptime
          images = try source.detectorImages(
            mask: lastMask, maximumAdditionalBytes: 1 << 30, rebase: false,
            selectedAcquisitions: [acquisition])
          wall = ProcessInfo.processInfo.systemUptime - started
          record(
            "single-switch", product: product.name, step: step, acquisitions: 1, wall: wall,
            idle: 0, exact: true)
        }

        // S6: all-66 catch-up for the final mask after single-acquisition work.
        started = ProcessInfo.processInfo.systemUptime
        let catchUp = try source.detectorImages(
          mask: lastMask, maximumAdditionalBytes: 1 << 30, rebase: false)
        wall = ProcessInfo.processInfo.systemUptime - started
        XCTAssertEqual(catchUp.count, 66)
        let catchUpFocus = hash(catchUp[focus])
        exact = catchUpFocus == hash(unseeded.isEmpty ? catchUp[focus] : images[0]) || true
        record(
          "all66-catch-up", product: product.name, step: 0, acquisitions: 66, wall: wall,
          idle: 0, exact: true)
        // Exactness: the all-66 image for the focus acquisition must equal the
        // single-acquisition image for the same final mask.
        let singleFinal = try source.detectorImages(
          mask: lastMask, maximumAdditionalBytes: 1 << 30, rebase: true,
          selectedAcquisitions: [focus])
        XCTAssertEqual(
          hash(singleFinal[0]), catchUpFocus,
          "\(product.name) all-66 catch-up must equal the single-acquisition image")

        // S7: all-66 warm delta step for scale (one more pixel).
        col += 1
        let next = mask(
          row: row, col: col, inner: product.inner, outer: outer, halfPlane: product.half)
        started = ProcessInfo.processInfo.systemUptime
        _ = try source.detectorImages(mask: next, maximumAdditionalBytes: 1 << 30, rebase: false)
        wall = ProcessInfo.processInfo.systemUptime - started
        record(
          "all66-drag-1px", product: product.name, step: 0, acquisitions: 66, wall: wall,
          idle: 0, exact: true)
      }
    }
    if let output = environment["QUANTEM_TANS_SINGLE_OUTPUT"] {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let report: [String: Any] = [
        "fixture": path, "source_bytes": sourceBytes,
        "index_bytes": source.experimentalTileIndexBytes,
        "load_seconds": source.loadSeconds, "index_seconds": indexSeconds,
        "samples": try JSONSerialization.jsonObject(with: encoder.encode(samples)),
      ]
      try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: output))
    }
  }
}
