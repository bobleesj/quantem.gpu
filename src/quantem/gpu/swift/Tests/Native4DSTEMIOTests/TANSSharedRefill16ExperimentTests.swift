import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Local one-factor experiment, not a default-selection or native FPS test.
final class TANSSharedRefill16ExperimentTests: XCTestCase {
  func testAll66SharedRefill16ABAWhenConfigured() throws {
    let packetMajor = ProcessInfo.processInfo.environment["QUANTEM_TANS_PACKET_MAJOR"] == "1"
    let bitExtract = ProcessInfo.processInfo.environment["QUANTEM_TANS_BIT_EXTRACT"] == "1"
    let prefetch = ProcessInfo.processInfo.environment["QUANTEM_TANS_PREFETCH_CODE"] == "1"
    let deferred =
      Int(ProcessInfo.processInfo.environment["QUANTEM_TANS_DEFERRED_PAIRS"] ?? "1") ?? -1
    guard [1, 2, 4, 8].contains(deferred) else {
      throw TANSArchive.invalid("Select deferred exact pair block1,2,4or8")
    }
    let concurrent = ProcessInfo.processInfo.environment["QUANTEM_TANS_CONCURRENT_DISPATCH"] == "1"
    let staged = ProcessInfo.processInfo.environment["QUANTEM_TANS_STAGED_REDUCTION"] == "1"
    let directTable = ProcessInfo.processInfo.environment["QUANTEM_TANS_DIRECT_TABLE"] == "1"
    let zeroRuns = ProcessInfo.processInfo.environment["QUANTEM_TANS_ZERO_RUNS"] == "1"
    let zeroArithmetic = ProcessInfo.processInfo.environment["QUANTEM_TANS_ZERO_ARITHMETIC"] == "1"
    let mixedTails = ProcessInfo.processInfo.environment["QUANTEM_TANS_MIXED_TAILS"] == "1"
    let separateMixed = ProcessInfo.processInfo.environment["QUANTEM_TANS_SEPARATE_MIXED"] == "1"
    let mixedOnly = ProcessInfo.processInfo.environment["QUANTEM_TANS_MIXED_ONLY"] == "1"
    let signedPair = ProcessInfo.processInfo.environment["QUANTEM_TANS_SIGNED_PAIR"] == "1"
    let preparedZero = ProcessInfo.processInfo.environment["QUANTEM_TANS_PREPARED_ZERO"] == "1"
    let unroll = Int(ProcessInfo.processInfo.environment["QUANTEM_TANS_PAIR_UNROLL"] ?? "1") ?? -1
    let compilerLimit =
      Int(ProcessInfo.processInfo.environment["QUANTEM_TANS_COMPILER_LIMIT"] ?? "0") ?? -1
    guard [0, 128, 256, 512].contains(compilerLimit),
      compilerLimit == 0
        || (mixedTails && separateMixed && mixedOnly && unroll == 1
          && !preparedZero && !signedPair && !zeroRuns && !directTable && !zeroArithmetic)
    else {
      throw TANSArchive.invalid(
        "Compiler limit comparison requires only frozen mixed-tail controls")
    }
    guard [1, 2, 4, 8].contains(unroll) else {
      throw TANSArchive.invalid("Select pair unroll factor1,2,4or8")
    }
    let mixedSavings =
      Int(ProcessInfo.processInfo.environment["QUANTEM_TANS_MIXED_SAVINGS_DIVISOR"] ?? "0") ?? -1
    guard
      !(concurrent || staged || prefetch || deferred > 1 || bitExtract || packetMajor)
        || (mixedTails && separateMixed && mixedOnly && mixedSavings == 8
          && unroll == 1 && compilerLimit == 0 && !signedPair && !preparedZero
          && !directTable && !zeroRuns && !zeroArithmetic)
    else {
      throw TANSArchive.invalid("Concurrent dispatch requires only the frozen mixed-tail baseline")
    }
    guard
      [concurrent, staged, prefetch, deferred > 1, bitExtract, packetMajor].filter({ $0 }).count
        <= 1
    else {
      throw TANSArchive.invalid("Select one experiment")
    }
    guard [0, 8].contains(mixedSavings) else {
      throw TANSArchive.invalid("Use mixed savings divisor zero or eight")
    }
    guard [directTable, zeroRuns, zeroArithmetic, mixedTails].filter({ $0 }).count <= 1 else {
      throw TANSArchive.invalid("Select only one exact decoder experiment per run")
    }
    guard !signedPair || (mixedTails && separateMixed && mixedOnly && mixedSavings == 8) else {
      throw TANSArchive.invalid(
        "Signed-pair control requires the frozen adaptive separated mixed baseline")
    }
    guard
      !preparedZero
        || (mixedTails && separateMixed && mixedOnly && mixedSavings == 8 && !signedPair
          && !zeroRuns)
    else {
      throw TANSArchive.invalid(
        "Prepared zero comparison requires the frozen separated mixed baseline")
    }
    guard let path = ProcessInfo.processInfo.environment["QUANTEM_TANS_REFILL16_FIXTURE"] else {
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
      "REFILL16_LOAD acquisitions=66 shape=512x512x192x192 dtype=uint16 scan_bin=1 detector_bin=1 crop=none representation=entropy source_bytes=\(sourceBytes) index_bytes=\(source.experimentalTileIndexBytes) load_s=\(source.loadSeconds) index_s=\(ProcessInfo.processInfo.systemUptime-prepared) source_pages=unspecified route=package_not_headed"
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
    for (arm, refill16) in [("A1", false), ("B16", true), ("A2", false)] {
      source.experimentalConcurrentDetectorDispatches = concurrent && refill16
      source.experimentalStagedDetectorReduction = staged && refill16
      source.experimentalPrefetchDecoderEntry = prefetch && refill16
      source.experimentalDeferredReductionPairs = refill16 ? deferred : 1
      source.experimentalDecoderBitExtract = bitExtract && refill16
      source.experimentalPacketMajorGrid = packetMajor && refill16
      source.experimentalDetectorRecordsPerCommand =
        concurrent || staged || prefetch || deferred > 1 || bitExtract || packetMajor ? 64 : 0
      guard
        unroll == 1
          || (mixedTails && separateMixed && mixedOnly && mixedSavings == 8
            && !preparedZero && !signedPair && !zeroRuns && !directTable && !zeroArithmetic)
      else {
        throw TANSArchive.invalid("Unroll comparison requires only frozen mixed-tail controls")
      }
      source.experimentalPairLoopUnroll = refill16 ? unroll : 1
      source.experimentalCompilerThreadgroupLimit = refill16 ? compilerLimit : 0
      source.experimentalSharedRefill16 =
        !directTable && !zeroRuns && !zeroArithmetic && !mixedTails && refill16
      source.experimentalDirectDecodingTable = directTable && refill16
      source.experimentalZeroRunDecoding = (zeroRuns || preparedZero) && refill16
      source.experimentalPreparedZeroRuns = preparedZero && refill16
      source.experimentalZeroBitArithmetic = zeroArithmetic && refill16
      source.experimentalMixedModelTails =
        mixedTails
        && (refill16 || signedPair || preparedZero || unroll > 1 || compilerLimit > 0 || concurrent
          || staged || prefetch || deferred > 1 || bitExtract || packetMajor)
      source.experimentalSeparateMixedDispatches =
        mixedTails
        && (refill16 || signedPair || preparedZero || unroll > 1 || compilerLimit > 0 || concurrent
          || staged || prefetch || deferred > 1 || bitExtract || packetMajor)
        && separateMixed
      source.experimentalMixedOnlySpecialization =
        mixedTails
        && (refill16 || signedPair || preparedZero || unroll > 1 || compilerLimit > 0 || concurrent
          || staged || prefetch || deferred > 1 || bitExtract || packetMajor)
        && mixedOnly
      source.experimentalSignedPairReduction = signedPair && refill16
      source.experimentalMixedTailSavingsDivisor = mixedSavings
      print(
        "MIXED_DISPATCH arm=\(arm) separate=\(source.experimentalSeparateMixedDispatches) savings_divisor=\(mixedSavings) signed_pair=\(source.experimentalSignedPairReduction)"
      )
      print("PREPARED_ZERO arm=\(arm) enabled=\(source.experimentalPreparedZeroRuns)")
      print("PAIR_UNROLL arm=\(arm) factor=\(source.experimentalPairLoopUnroll)")
      print("COMPILER_LIMIT arm=\(arm) requested=\(source.experimentalCompilerThreadgroupLimit)")
      fflush(stdout)
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
              "REFILL16_SAMPLE arm=\(arm) refill16=\(source.experimentalSharedRefill16) direct_table=\(source.experimentalDirectDecodingTable) zero_runs=\(source.experimentalZeroRunDecoding) zero_arithmetic=\(source.experimentalZeroBitArithmetic) mixed_tails=\(source.experimentalMixedModelTails) model_groups=\(source.lastDetectorModelGroups) mixed_groups=\(source.lastDetectorMixedModelGroups) padded_lanes=\(source.lastDetectorPaddedModelLanes) case=\(cases[i].0) cycle=\(cycle) wall_ms=\(wall) gpu_ms=\(gpu) columns=\(source.lastDetectorDecodedColumns) tile_fields=\(source.lastDetectorTileFields) scratch_bytes=\(source.lastDetectorScratchBytes) allocated_bytes=\(device.currentAllocatedSize) images=66 exact=true"
            )
            fflush(stdout)
            if concurrent || staged || prefetch || deferred > 1 || bitExtract || packetMajor {
              let timing = source.lastDetectorCommandTiming.sorted { $0.key < $1.key }
                .map { "\($0.key)=\($0.value)" }.joined(separator: " ")
              print("CONCURRENT_COMMAND arm=\(arm) case=\(cases[i].0) cycle=\(cycle) \(timing)")
              fflush(stdout)
            }
          }
        }
      }
    }
  }
}
