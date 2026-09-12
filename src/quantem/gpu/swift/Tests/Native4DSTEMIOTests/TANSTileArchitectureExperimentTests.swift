import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// One representation/index change, complete outputs, never a native FPS claim.
final class TANSTileArchitectureExperimentTests: XCTestCase {
  func testAll66UniformIndexABAWhenConfigured() throws {
    let env = ProcessInfo.processInfo.environment
    guard let path = env["QUANTEM_TANS_ARCH_FIXTURE"] else {
      throw XCTSkip("Requires the complete authenticated all66 entropy fixture")
    }
    let candidate = try XCTUnwrap(
      TANSExactTileIndex.Layout(rawValue: env["QUANTEM_TANS_ARCH_LAYOUT"] ?? "uniform16"))
    guard candidate != .centerFine else { throw TANSArchive.invalid("Select a uniform candidate") }
    let planAfterIndex = env["QUANTEM_TANS_ARCH_PLAN"] == "1"
    let baselineOnly = env["QUANTEM_TANS_ARCH_BASELINE_ONLY"] == "1"
    let pairLookupBits = Int(env["QUANTEM_TANS_ARCH_PAIR_LOOKUP"] ?? "0") ?? -1
    guard [0, 4, 6].contains(pairLookupBits), !(pairLookupBits > 0 && planAfterIndex) else {
      throw TANSArchive.invalid("Select one exact planner or pair-lookup experiment")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.releaseResidentStorage() }
    let sourceBytes = source.residentBytes
    let indexCeiling: UInt64 = 2_108_620_800
    source.experimentalDetectorStreamsPerLane = 32
    source.experimentalDetectorRecordsPerCommand = 64
    source.experimentalMixedModelTails = true
    source.experimentalSeparateMixedDispatches = true
    source.experimentalMixedOnlySpecialization = true
    source.experimentalMixedTailSavingsDivisor = 8
    print(
      "ARCH_LOAD source_bytes=\(sourceBytes) load_s=\(source.loadSeconds) acquisitions=66 source_pages=unspecified"
    )
    fflush(stdout)
    func hash(_ data: Data) -> String {
      SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    func hashes(_ images: [MTLBuffer]) -> [String] {
      images.map { hash(Data(bytesNoCopy: $0.contents(), count: $0.length, deallocator: .none)) }
    }
    func mask(_ row: Double, _ col: Double, _ inner: Double, _ outer: Double) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - col
        let d = r * r + c * c
        return source.validDetectorMask[q] != 0 && d >= inner * inner && d <= outer * outer ? 1 : 0
      }
    }
    // Authenticate independent Linux reference before choosing any architecture.
    let frozen = try Data(
      contentsOf: URL(fileURLWithPath: path).appendingPathComponent(
        "series-products/shared-resident-images.npy"), options: .mappedIfSafe)
    XCTAssertEqual(hash(frozen), "06b96168b62d89ba5a893416dcf7b0000d34647310da91db243ceab945cf89ac")
    let offset = 10 + Int(frozen[8]) + (Int(frozen[9]) << 8)
    XCTAssertEqual(frozen.count - offset, 66 * 4 * 512 * 512 * 4)
    for (product, inner, outer) in [(0, 0.0, 28.0), (1, 40.0, 80.0), (2, 14.0, 28.0)] {
      try autoreleasepool {
        let images = try source.detectorImages(
          mask: mask(95.5, 95.5, inner, outer), maximumAdditionalBytes: 1 << 30, rebase: true)
        for acquisition in 0..<66 {
          let image = images[acquisition]
          let begin = offset + (acquisition * 4 + product) * 512 * 512 * 4
          XCTAssertEqual(
            Data(bytesNoCopy: image.contents(), count: image.length, deallocator: .none),
            frozen.subdata(in: begin..<(begin + image.length)))
        }
        print("ARCH_BASELINE product=\(product) complete_acquisitions=66 exact=true")
        fflush(stdout)
      }
    }
    if baselineOnly {
      source.releaseResidentStorage()
      XCTAssertEqual(source.residentBytes, 0)
      print("ARCH_DIAGNOSTIC_RELEASE source_bytes=0")
      fflush(stdout)
      return
    }
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
    // No index in the reference, so candidate and reference cannot share an index bug.
    let references = try targets.map { target in
      try autoreleasepool {
        hashes(
          try source.detectorImages(mask: target, maximumAdditionalBytes: 1 << 30, rebase: true))
      }
    }
    let arms: [(String, TANSExactTileIndex.Layout, Bool)] =
      planAfterIndex || pairLookupBits > 0
      ? [("A1", candidate, true), ("B_plan", candidate, true), ("A2", candidate, true)]
      : [
        ("A1", TANSExactTileIndex.Layout.centerFine, false),
        ("B_index", candidate, true),
        ("A2", TANSExactTileIndex.Layout.centerFine, false),
      ]
    for (arm, layout, blocked) in arms {
      source.experimentalPairLookupBits = 0
      source.experimentalPlanAfterIndex = false
      // No simultaneous retained indices or concealed increase of source residency.
      try source.discardExperimentalTileIndex()
      XCTAssertEqual(source.experimentalTileIndexBytes, 0)
      let indexStarted = ProcessInfo.processInfo.systemUptime
      do {
        try source.prepareExperimentalTileIndex(
          maximumIndexBytes: indexCeiling, blockedPacking: blocked, layout: layout)
      } catch {
        print(
          "ARCH_ADMISSION arm=\(arm) layout=\(layout.rawValue) rejected=true ceiling=\(indexCeiling) allocated_bytes=\(device.currentAllocatedSize) error=\(error)"
        )
        fflush(stdout)
        throw error
      }
      source.experimentalUseTileIndex = true
      source.experimentalPlanAfterIndex = planAfterIndex && arm == "B_plan"
      if pairLookupBits > 0 && source.experimentalPairLookupBytes == 0 {
        let started = ProcessInfo.processInfo.systemUptime
        try source.prepareExperimentalPairLookup(
          bits: pairLookupBits, maximumMetadataBytes: indexCeiling)
        print(
          "ARCH_LOOKUP bits=\(pairLookupBits) bytes=\(source.experimentalPairLookupBytes) prepare_s=\(ProcessInfo.processInfo.systemUptime-started)"
        )
        fflush(stdout)
      }
      source.experimentalPairLookupBits = arm == "B_plan" ? pairLookupBits : 0
      XCTAssertLessThanOrEqual(
        UInt64(source.experimentalTileIndexBytes + source.experimentalPairLookupBytes), indexCeiling
      )
      XCTAssertLessThanOrEqual(UInt64(source.experimentalTileIndexBytes), indexCeiling)
      print(
        "ARCH_INDEX arm=\(arm) layout=\(layout.rawValue) blocked=\(blocked) bytes=\(source.experimentalTileIndexBytes) prepare_s=\(ProcessInfo.processInfo.systemUptime-indexStarted) allocated_bytes=\(device.currentAllocatedSize)"
      )
      fflush(stdout)
      for i in cases.indices {
        try autoreleasepool {
          _ = try source.detectorImages(
            mask: bases[i], maximumAdditionalBytes: 1 << 30, rebase: true)
          _ = try source.detectorImages(mask: targets[i], maximumAdditionalBytes: 1 << 30)
        }
        for cycle in 0..<5 {
          try autoreleasepool {
            let prior = try source.detectorImages(
              mask: bases[i], maximumAdditionalBytes: 1 << 30, rebase: true)
            let priorHashes = hashes(prior)
            let start = ProcessInfo.processInfo.systemUptime
            let images = try source.detectorImages(
              mask: targets[i], maximumAdditionalBytes: 1 << 30)
            let wall = (ProcessInfo.processInfo.systemUptime - start) * 1000
            XCTAssertEqual(images.count, 66)
            XCTAssertEqual(hashes(images), references[i])
            XCTAssertEqual(hashes(prior), priorHashes)
            print(
              "ARCH_SAMPLE arm=\(arm) case=\(cases[i].0) cycle=\(cycle) wall_ms=\(wall) gpu_ms=\(source.lastDetectorGPUSeconds*1000) columns=\(source.lastDetectorDecodedColumns) tile_fields=\(source.lastDetectorTileFields) index_bytes=\(source.experimentalTileIndexBytes) allocated_bytes=\(device.currentAllocatedSize) complete66_sha256=\(hash(Data(hashes(images).joined(separator: ";").utf8))) exact=true"
            )
            let timing = source.lastDetectorCommandTiming.sorted { $0.key < $1.key }
              .map { "\($0.key)=\($0.value)" }.joined(separator: " ")
            print("ARCH_COMMAND arm=\(arm) case=\(cases[i].0) cycle=\(cycle) \(timing)")
            fflush(stdout)
          }
        }
      }
    }
    source.releaseResidentStorage()
    XCTAssertEqual(source.residentBytes, 0)
    print("ARCH_RELEASE source_bytes=0 allocated_bytes=\(device.currentAllocatedSize)")
    fflush(stdout)
  }
}
