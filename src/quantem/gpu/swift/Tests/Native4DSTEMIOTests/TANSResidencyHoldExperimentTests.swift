import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Same masks as retained native gestures; package diagnostics, not native FPS.
final class TANSResidencyHoldExperimentTests: XCTestCase {
  private struct Request: Decodable {
    let geometry: [String: Double]
    let columns: Int
    let previous: Bool
    let probes: String
    let maskSha256: String
  }
  private struct Replay: Decodable { let rows: [Request] }

  func testAll66ResidencyHoldIdleABAWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    let batch256 = environment["QUANTEM_TANS_BATCH256_ABA"] == "1"
    let narrowOutputs = environment["QUANTEM_TANS_NARROW_OUTPUTS_ABA"] == "1"
    let prefetch = environment["QUANTEM_TANS_PREFETCH_REPLAY"] == "1"
    let indexReplay = environment["QUANTEM_TANS_INDEX_REPLAY"] == "1"
    let invocationOwned = environment["QUANTEM_TANS_INVOCATION_OWNED"] == "1"
    let sourcePolicyRun = environment["QUANTEM_TANS_SOURCE_POLICY_RUN"]
    let sparsePrefix = environment["QUANTEM_TANS_SPARSE_PREFIX"] == "1"
    let singleCommand = environment["QUANTEM_TANS_SINGLE_COMMAND_ENCODERS"] == "1"
    let metal4Submission = environment["QUANTEM_TANS_METAL4_SUBMISSION"] == "1"
    let singlePass = environment["QUANTEM_TANS_SINGLE_COMPUTE_PASS"] == "1"
    let submissionFloor = environment["QUANTEM_TANS_SUBMISSION_FLOOR"] == "1"
    guard [sparsePrefix, singleCommand, metal4Submission, singlePass].filter({ $0 }).count <= 1,
      !(sparsePrefix || singleCommand || metal4Submission || singlePass) || sourcePolicyRun != nil
    else { throw TANSArchive.invalid("Select one isolated audit-derived source-policy experiment") }
    let untrackedSource = environment["QUANTEM_TANS_UNTRACKED_SOURCE"] == "1"
    let coalescedSource = environment["QUANTEM_TANS_COALESCED_SOURCE"] == "1"
    let coalescedHold = environment["QUANTEM_TANS_COALESCED_HOLD"] == "1"
    let requestOnly = environment["QUANTEM_TANS_HOLD_REQUEST_ONLY"] == "1"
    guard !requestOnly || coalescedHold else {
      throw TANSArchive.invalid("Request-only hold requires the isolated coalesced hold")
    }
    let sharedSource = environment["QUANTEM_TANS_SHARED_SOURCE"] == "1"
    let multiQueueRun = environment["QUANTEM_TANS_MULTI_QUEUE_RUN"] == "1"
    let queryQueues = Int(environment["QUANTEM_TANS_QUERY_QUEUES"] ?? "1") ?? -1
    guard [1, 4].contains(queryQueues), multiQueueRun || queryQueues == 1,
      !multiQueueRun
        || (coalescedSource && sourcePolicyRun != nil && !coalescedHold && !sharedSource)
    else {
      throw TANSArchive.invalid(
        "Queue comparison requires an isolated private-coalesced source run")
    }
    guard !sharedSource || (coalescedSource && sourcePolicyRun != nil && !coalescedHold) else {
      throw TANSArchive.invalid("Shared source requires an isolated coalesced source-policy run")
    }
    guard !coalescedHold || (coalescedSource && sourcePolicyRun != nil) else {
      throw TANSArchive.invalid("Coalesced hold requires a coalesced single-arm source-policy run")
    }
    guard sourcePolicyRun != nil || !(untrackedSource || coalescedSource),
      !(untrackedSource && coalescedSource)
    else {
      throw TANSArchive.invalid(
        "Untracked source requires an explicit single-arm source-policy run")
    }
    let sourceSetOnly = environment["QUANTEM_TANS_HOLD_SOURCE_SET"] == "1"
    let heapReplay = environment["QUANTEM_TANS_HEAP_REPLAY"] == "1"
    let heapDeclaration = environment["QUANTEM_TANS_HEAP_DECLARATION_ABA"] == "1"
    let recordsPerHeap = Int(environment["QUANTEM_TANS_RECORDS_PER_HEAP"] ?? "0") ?? -1
    guard [0, 16, 64].contains(recordsPerHeap), heapReplay || recordsPerHeap == 0 else {
      throw TANSArchive.invalid("Heap backing requires the isolated heap replay")
    }
    let mixedTails = environment["QUANTEM_TANS_REPLAY_MIXED_TAILS"] == "1"
    let separateMixed = environment["QUANTEM_TANS_SEPARATE_MIXED"] == "1"
    let mixedOnly = environment["QUANTEM_TANS_MIXED_ONLY"] == "1"
    let mixedSavings = Int(environment["QUANTEM_TANS_MIXED_SAVINGS_DIVISOR"] ?? "0") ?? -1
    let mixedBaseSplit = Int(environment["QUANTEM_TANS_REPLAY_BASE_SPLIT"] ?? "0") ?? -1
    guard
      !(batch256 || narrowOutputs || prefetch || indexReplay || invocationOwned
        || sourcePolicyRun != nil)
        || (mixedTails && separateMixed && mixedOnly && mixedSavings == 8
          && mixedBaseSplit == 64 && !sourceSetOnly && !heapReplay && !heapDeclaration)
    else {
      throw TANSArchive.invalid("Batch256 requires unchanged mixed-tail submission64 controls")
    }
    guard
      [batch256, narrowOutputs, prefetch, indexReplay, invocationOwned, sourcePolicyRun != nil]
        .filter({ $0 }).count <= 1
    else {
      throw TANSArchive.invalid("Select one submission experiment")
    }
    guard
      !heapDeclaration
        || (heapReplay && recordsPerHeap == 16 && mixedTails && separateMixed && mixedOnly
          && mixedSavings == 8 && mixedBaseSplit == 64)
    else {
      throw TANSArchive.invalid(
        "Heap declaration comparison requires the unchanged heap16 mixed-tail submission64 baseline"
      )
    }
    let splitCommands = environment["QUANTEM_TANS_SPLIT_COMMANDS"] == "1"
    guard [0, 8].contains(mixedSavings), [0, 64].contains(mixedBaseSplit),
      !mixedTails || (!splitCommands && !sourceSetOnly)
    else {
      throw TANSArchive.invalid(
        "Mixed-tail replay must isolate grouping from submission and residency changes")
    }
    let recordsPerCommand = Int(environment["QUANTEM_TANS_RECORDS_PER_COMMAND"] ?? "64") ?? -1
    let idleSeconds = Double(environment["QUANTEM_TANS_HOLD_IDLE_SECONDS"] ?? "4") ?? -1
    let cycles = Int(environment["QUANTEM_TANS_HOLD_CYCLES"] ?? "2") ?? 0
    guard
      !submissionFloor
        || (sourcePolicyRun != nil && coalescedSource
          && !coalescedHold && !sharedSource && !multiQueueRun && !sparsePrefix
          && !singleCommand && !metal4Submission && !singlePass && idleSeconds == 4 && cycles == 2)
    else {
      throw TANSArchive.invalid("Submission floor requires the frozen isolated paused baseline")
    }
    guard [0.0, 4.0].contains(idleSeconds), (1...5).contains(cycles),
      [16, 64].contains(recordsPerCommand)
    else {
      throw TANSArchive.invalid(
        "Replay requires zero/four second idle, one through five cycles and 16/64 records per command"
      )
    }
    guard let path = environment["QUANTEM_TANS_HOLD_FIXTURE"],
      let replayPath = environment["QUANTEM_TANS_REPLAY_MASKS"]
    else { throw XCTSkip("Requires complete all66 fixture and recorded native mask metadata") }
    let decoder = JSONDecoder()
    decoder.keyDecodingStrategy = .convertFromSnakeCase
    let replay = try decoder.decode(
      Replay.self, from: Data(contentsOf: URL(fileURLWithPath: replayPath)))
    XCTAssertEqual(replay.rows.count, 9)
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize), experimentalRecordsPerHeap: recordsPerHeap,
      experimentalUntrackedEncodedSource: untrackedSource,
      experimentalCoalescedEncodedSource: coalescedSource,
      experimentalSharedEncodedSource: sharedSource)
    defer {
      source.releaseResidentStorage()
      print(
        "SOURCE_FINAL source_bytes=\(source.residentBytes) allocated_bytes=\(device.currentAllocatedSize)"
      )
      fflush(stdout)
    }
    let sourceBytes = source.residentBytes
    XCTAssertEqual(source.experimentalUntrackedRecordCount, untrackedSource ? 1056 : 0)
    XCTAssertEqual(source.experimentalEncodedAllocationCount, coalescedSource ? 66 : 1056)
    XCTAssertEqual(sourceBytes, 84_681_338_880)
    print(
      "SOURCE_POLICY untracked=\(untrackedSource) records=\(source.experimentalUntrackedRecordCount) explicit_upload_completion=true"
    )
    fflush(stdout)
    print(
      "COALESCED_SOURCE enabled=\(coalescedSource) allocations=\(source.experimentalEncodedAllocationCount)"
    )
    print("ENCODED_STORAGE shared=\(sharedSource) unified_memory=\(device.hasUnifiedMemory)")
    if let sourcePolicyRun {
      // Full selected-DP fingerprints across chunk boundaries, outside timing.
      for (row, col) in [(0, 0), (31, 511), (32, 0), (255, 255), (256, 0), (511, 511)] {
        try autoreleasepool {
          let counts = try source.extractDiffraction(scanRow: row, scanColumn: col)
          XCTAssertEqual(counts.count, 66 * 36864)
          let fingerprint = counts.withUnsafeBytes {
            SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
          }
          print(
            "SOURCE_DP arm=\(sourcePolicyRun) row=\(row) col=\(col) values=\(counts.count) sha256=\(fingerprint)"
          )
        }
      }
      fflush(stdout)
    }
    let prepareStart = ProcessInfo.processInfo.systemUptime
    try source.prepareExperimentalTileIndex(maximumIndexBytes: 2 << 30)
    source.experimentalUseTileIndex = true
    source.experimentalDetectorStreamsPerLane = 32
    print(
      "RESIDENCY_HOLD_LOAD acquisitions=66 source_bytes=\(sourceBytes) index_bytes=\(source.experimentalTileIndexBytes) load_s=\(source.loadSeconds) index_s=\(ProcessInfo.processInfo.systemUptime-prepareStart) source_pages=unspecified route=package_not_headed"
    )
    fflush(stdout)
    print(
      "HEAP_REPLAY_LOAD records_per_heap=\(recordsPerHeap) heap_count=\(source.experimentalRecordHeapCount) heap_bytes=\(source.experimentalRecordHeapBytes) staging_bytes=\(source.stagingBytes) source_bytes_read=\(source.readMetrics.sourceBytesRead) authenticated_records=\(source.readMetrics.authenticatedRecords) allocated_bytes=\(device.currentAllocatedSize)"
    )
    fflush(stdout)
    func hash(_ data: Data) -> String {
      SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    func hashes(_ images: [MTLBuffer]) -> [String] {
      images.map { hash(Data(bytesNoCopy: $0.contents(), count: $0.length, deallocator: .none)) }
    }
    func mask(row: Double, col: Double, inner: Double, outer: Double) -> [UInt8] {
      (0..<36864).map { q in
        let radius2 = pow(Double(q / 192) - row, 2) + pow(Double(q % 192) - col, 2)
        return source.validDetectorMask[q] != 0 && radius2 >= inner * inner
          && radius2 <= outer * outer ? 1 : 0
      }
    }
    let masks = try replay.rows.map { request in
      let geometry = request.geometry
      let values = try mask(
        row: XCTUnwrap(geometry["center-row"]), col: XCTUnwrap(geometry["center-column"]),
        inner: XCTUnwrap(geometry["inner"]), outer: XCTUnwrap(geometry["outer"]))
      XCTAssertEqual(hash(Data(values)), request.maskSha256)
      return values
    }
    let frozen = try Data(
      contentsOf: URL(fileURLWithPath: path).appendingPathComponent(
        "series-products/shared-resident-images.npy"), options: .mappedIfSafe)
    XCTAssertEqual(hash(frozen), "06b96168b62d89ba5a893416dcf7b0000d34647310da91db243ceab945cf89ac")
    let offset = 10 + Int(frozen[8]) + (Int(frozen[9]) << 8)
    XCTAssertEqual(frozen.count - offset, 66 * 4 * 512 * 512 * 4)
    for (product, inner, outer) in [(0, 0.0, 28.0), (1, 40.0, 80.0), (2, 14.0, 28.0)] {
      try autoreleasepool {
        let images = try source.detectorImages(
          mask: mask(row: 95.5, col: 95.5, inner: inner, outer: outer),
          maximumAdditionalBytes: 1 << 30, rebase: true)
        for acquisition in 0..<66 {
          let start = offset + (acquisition * 4 + product) * 512 * 512 * 4
          let image = images[acquisition]
          XCTAssertEqual(
            Data(bytesNoCopy: image.contents(), count: image.length, deallocator: .none),
            frozen.subdata(in: start..<(start + image.length)))
        }
      }
    }
    let referenceHashes = try masks.enumerated().map { index, mask in
      try autoreleasepool {
        let images = try source.detectorImages(
          mask: mask, maximumAdditionalBytes: 1 << 30, rebase: true)
        let nativeProbes = replay.rows[index].probes.split(separator: ";")
        XCTAssertEqual(nativeProbes.count, 66)
        for acquisition in 0..<66 {
          let entry = nativeProbes[acquisition].split(separator: ":")
          XCTAssertEqual(Int(entry[0]), acquisition)
          let expected = entry[1].split(separator: ",").map { UInt32($0)! }
          for (point, scan) in [0, 131328, 262143].enumerated() {
            XCTAssertEqual(
              images[acquisition].contents().load(fromByteOffset: scan * 4, as: UInt32.self),
              expected[point])
          }
        }
        let completeHashes = hashes(images)
        if heapReplay {
          print(
            "HEAP_REPLAY_REFERENCE case=\(index) complete66_sha256=\(hash(Data(completeHashes.joined(separator: ";").utf8)))"
          )
          fflush(stdout)
        }
        return completeHashes
      }
    }
    if environment["QUANTEM_TANS_SUBMISSION_FAILURES"] == "1" {
      source.experimentalDetectorQueueCount = queryQueues
      source.experimentalNarrowOutputDeclarations = multiQueueRun
      source.experimentalInvocationOwnedSubmissions = invocationOwned
      source.experimentalDetectorRecordsPerCommand = 64
      source.experimentalMixedModelTails = mixedTails
      source.experimentalSeparateMixedDispatches = mixedTails && separateMixed
      source.experimentalMixedOnlySpecialization = mixedTails && mixedOnly
      source.experimentalMixedTailSavingsDivisor = mixedSavings
      for failureAfter in [0, 1, 2] {
        try autoreleasepool {
          let prior = try source.detectorImages(
            mask: masks[0], maximumAdditionalBytes: 1 << 30, rebase: true)
          XCTAssertEqual(hashes(prior), referenceHashes[0])
          let allocationBefore = device.currentAllocatedSize
          source.experimentalFailAfterSubmittedCommands = failureAfter
          XCTAssertThrowsError(
            try autoreleasepool {
              _ = try source.detectorImages(mask: masks[1], maximumAdditionalBytes: 1 << 30)
            }
          ) { error in
            XCTAssertTrue(
              String(describing: error).contains("Injected detector submission failure"))
          }
          source.experimentalFailAfterSubmittedCommands = nil
          XCTAssertEqual(hashes(prior), referenceHashes[0])
          let allocationAfter = device.currentAllocatedSize
          XCTAssertEqual(allocationAfter, allocationBefore)
          let recovered = try source.detectorImages(mask: masks[1], maximumAdditionalBytes: 1 << 30)
          XCTAssertEqual(hashes(recovered), referenceHashes[1])
          XCTAssertEqual(hashes(prior), referenceHashes[0])
          print(
            "SUBMISSION_FAILURE after=\(failureAfter) images=66 prior_exact=true recovered_exact=true allocated_before=\(allocationBefore) allocated_after=\(allocationAfter)"
          )
          fflush(stdout)
        }
      }
      return
    }
    let arms =
      sourcePolicyRun != nil
      ? [(sourcePolicyRun!, true)]
      : heapDeclaration
        ? [("A1", false), ("B_heap", true), ("A2", false)]
        : heapReplay
          ? [("HEAP_\(recordsPerHeap)", true)]
          : [
            ("A1", false), (mixedTails ? "B_mixed" : (splitCommands ? "B_split" : "B_hold"), true),
            ("A2", false),
          ]
    for (arm, candidate) in arms {
      source.endExperimentalResidencyHold()
      let setupStart = ProcessInfo.processInfo.systemUptime
      if indexReplay && arm != "A1" {
        try source.discardExperimentalTileIndex()
        try source.prepareExperimentalTileIndex(
          maximumIndexBytes: 2_108_620_800, blockedPacking: candidate,
          layout: candidate ? .uniform16 : .centerFine)
        source.experimentalUseTileIndex = true
      }
      if indexReplay {
        print(
          "ARCH_INDEX arm=\(arm) layout=\(candidate ? "uniform16" : "centerFine") bytes=\(source.experimentalTileIndexBytes) prepare_s=\(ProcessInfo.processInfo.systemUptime-setupStart)"
        )
        fflush(stdout)
      }
      let holding = coalescedHold || (candidate && !splitCommands && !mixedTails)
      if holding { try source.beginExperimentalResidencyHold(attachToQueue: !requestOnly) }
      source.experimentalDetectorRecordsPerCommand =
        batch256
        ? (candidate ? 256 : 64)
        : mixedTails ? mixedBaseSplit : (candidate && splitCommands ? recordsPerCommand : 0)
      source.experimentalNarrowOutputDeclarations = multiQueueRun || (narrowOutputs && candidate)
      source.experimentalDetectorQueueCount = queryQueues
      source.experimentalPrefetchDecoderEntry = prefetch && candidate
      source.experimentalSparsePrefixCarry = sparsePrefix
      source.experimentalSingleCommandEncoders = singleCommand
      source.experimentalSingleComputePass = singlePass
      source.experimentalMetal4Submission = metal4Submission
      if metal4Submission { try source.prepareExperimentalMetal4Submission() }
      source.experimentalInvocationOwnedSubmissions = invocationOwned && candidate
      source.experimentalMixedModelTails =
        mixedTails
        && (candidate || heapDeclaration || batch256 || narrowOutputs || prefetch || indexReplay
          || invocationOwned || sourcePolicyRun != nil)
      source.experimentalSeparateMixedDispatches =
        mixedTails
        && (candidate || heapDeclaration || batch256 || narrowOutputs || prefetch || indexReplay
          || invocationOwned || sourcePolicyRun != nil)
        && separateMixed
      source.experimentalMixedOnlySpecialization =
        mixedTails
        && (candidate || heapDeclaration || batch256 || narrowOutputs || prefetch || indexReplay
          || invocationOwned || sourcePolicyRun != nil)
        && mixedOnly
      source.experimentalDeclareEncodedHeaps = heapDeclaration && candidate
      source.experimentalMixedTailSavingsDivisor = mixedSavings
      source.experimentalUseResidentSetForSourceReads = holding && sourceSetOnly
      print(
        "RESIDENCY_HOLD_SETUP arm=\(arm) records_per_command=\(source.experimentalDetectorRecordsPerCommand) source_set_only=\(source.experimentalUseResidentSetForSourceReads) setup_s=\(ProcessInfo.processInfo.systemUptime-setupStart) set_bytes=\(source.experimentalResidencySetBytes) allocations=\(source.experimentalResidencyAllocationCount) allocated_bytes=\(device.currentAllocatedSize)"
      )
      fflush(stdout)
      print(
        "MIXED_REPLAY arm=\(arm) enabled=\(source.experimentalMixedModelTails) minimum_savings_divisor=\(mixedSavings) records_per_command=\(source.experimentalDetectorRecordsPerCommand)"
      )
      print(
        "RESIDENCY_ATTACHMENT arm=\(arm) attached=\(source.experimentalResidencyAttachedToQueue) request_only=\(requestOnly)"
      )
      print("HEAP_DECLARATION arm=\(arm) enabled=\(source.experimentalDeclareEncodedHeaps)")
      fflush(stdout)
      if submissionFloor {
        var diagnosticHashes: [String: String] = [:]
        for cycle in 0..<cycles {
          for index in masks.indices {
            // Rotate and reverse order; every mode gets the same untimed
            // scientific prior, followed by the declared idle interval.
            let modes = ["bindings", "denseWords", "full"]
            let rotated = (0..<3).map { modes[($0 + index) % 3] }
            let order = cycle == 0 ? rotated : Array(rotated.reversed())
            for mode in order {
              try autoreleasepool {
                let priorIndex = index == 0 ? masks.count - 1 : index - 1
                let prior = try source.detectorImages(
                  mask: masks[priorIndex], maximumAdditionalBytes: 1 << 30, rebase: true)
                XCTAssertEqual(hashes(prior), referenceHashes[priorIndex])
                let priorResidentBytes = source.residentBytes
                let priorSourceReads = source.readMetrics.sourceBytesRead
                Thread.sleep(forTimeInterval: idleSeconds)
                let started = ProcessInfo.processInfo.systemUptime
                let timing: [String: Double]
                var words: [UInt32] = []
                var images: [MTLBuffer] = []
                if mode == "full" {
                  images = try source.detectorImages(
                    mask: masks[index], maximumAdditionalBytes: 1 << 30, rebase: index == 0)
                  timing = source.lastDetectorCommandTiming
                } else {
                  let result = try source.measureSubmissionProbe(
                    mask: masks[index], mode: mode == "bindings" ? .bindings : .denseWords,
                    maximumAdditionalBytes: 1 << 30, rebase: index == 0)
                  timing = result.timing
                  words = result.records
                }
                let wall = (ProcessInfo.processInfo.systemUptime - started) * 1000
                XCTAssertEqual(hashes(prior), referenceHashes[priorIndex])
                XCTAssertEqual(source.lastDetectorDecodedColumns, replay.rows[index].columns)
                XCTAssertEqual(source.lastDetectorUsedPrevious, replay.rows[index].previous)
                XCTAssertEqual(source.residentBytes, priorResidentBytes)
                XCTAssertEqual(source.readMetrics.sourceBytesRead, priorSourceReads)
                XCTAssertEqual(source.experimentalEncodedAllocationCount, 66)
                XCTAssertEqual(source.experimentalTileIndexBytes, 2_108_620_800)
                var denseWords: UInt64 = 0
                var streams: UInt64 = 0
                var fingerprint = ""
                if mode == "full" {
                  XCTAssertEqual(images.count, 66)
                  XCTAssertEqual(hashes(images), referenceHashes[index])
                  fingerprint = hash(Data(hashes(images).joined(separator: ";").utf8))
                } else {
                  guard words.count == 1056 * 4 else {
                    throw TANSArchive.invalid("Incomplete all66 diagnostic record output")
                  }
                  for record in 0..<1056 {
                    let base = record * 4
                    if mode == "bindings" {
                      XCTAssertEqual(Array(words[base..<(base + 3)]), [0, 0, 0])
                      XCTAssertEqual(words[base + 3], 0x51A7_0000 + UInt32(record))
                    } else {
                      XCTAssertEqual(words[base + 3], 0)
                      denseWords += UInt64(words[base + 1])
                      streams += UInt64(words[base + 2])
                    }
                  }
                  if mode == "denseWords" {
                    XCTAssertGreaterThan(denseWords, 0)
                    XCTAssertEqual(Double(streams), timing["dense_stream_count"])
                  }
                  fingerprint = words.withUnsafeBytes { hash(Data($0)) }
                  let key = "\(mode)-\(index)"
                  if let previous = diagnosticHashes[key] { XCTAssertEqual(fingerprint, previous) }
                  diagnosticHashes[key] = fingerprint
                }
                let stages = timing.sorted { $0.key < $1.key }
                  .map { "\($0.key)=\($0.value)" }.joined(separator: " ")
                print(
                  "SUBMISSION_FLOOR_SAMPLE mode=\(mode) cycle=\(cycle) case=\(index) wall_ms=\(wall) idle_s=\(idleSeconds) dense_words=\(denseWords) streams=\(streams) fingerprint=\(fingerprint) prior_exact=true scientific=\(mode == "full") images=\(images.count) source_bytes=\(sourceBytes) index_bytes=\(source.experimentalTileIndexBytes) allocated_bytes=\(device.currentAllocatedSize)"
                )
                print(
                  "SUBMISSION_FLOOR_COMMAND mode=\(mode) cycle=\(cycle) case=\(index) \(stages)")
                fflush(stdout)
                if mode != "full" {
                  // Untimed recovery also verifies that diagnostic completion
                  // did not replace the internal prior mask or output.
                  let recovered = try source.detectorImages(
                    mask: masks[index], maximumAdditionalBytes: 1 << 30, rebase: index == 0)
                  XCTAssertEqual(hashes(recovered), referenceHashes[index])
                }
              }
            }
          }
        }
        return
      }
      for cycle in 0..<cycles {
        for index in masks.indices {
          try autoreleasepool {
            if idleSeconds > 0 { Thread.sleep(forTimeInterval: idleSeconds) }
            if let auditRoot = environment["QUANTEM_TANS_MEMORY_AUDIT"] {
              source.experimentalMemoryAuditURL = URL(fileURLWithPath: auditRoot)
                .appendingPathComponent("case-\(index)-cycle-\(cycle).json")
            }
            let started = ProcessInfo.processInfo.systemUptime
            let images = try source.detectorImages(
              mask: masks[index], maximumAdditionalBytes: 1 << 30, rebase: index == 0)
            let wall = (ProcessInfo.processInfo.systemUptime - started) * 1000
            let gpu = source.lastDetectorGPUSeconds * 1000
            XCTAssertEqual(images.count, 66)
            XCTAssertEqual(hashes(images), referenceHashes[index])
            if !indexReplay {
              XCTAssertEqual(source.lastDetectorDecodedColumns, replay.rows[index].columns)
              XCTAssertEqual(source.lastDetectorUsedPrevious, replay.rows[index].previous)
            }
            // An index may change decomposition, never the full output above.
            XCTAssertLessThanOrEqual(source.experimentalTileIndexBytes, 2_108_620_800)
            print(
              "RESIDENCY_HOLD_SAMPLE arm=\(arm) cycle=\(cycle) case=\(index) idle_s=\(idleSeconds) wall_ms=\(wall) gpu_ms=\(gpu) columns=\(source.lastDetectorDecodedColumns) previous=\(source.lastDetectorUsedPrevious) allocated_bytes=\(device.currentAllocatedSize) source_bytes=\(sourceBytes) index_bytes=\(source.experimentalTileIndexBytes) images=66 exact=true"
            )
            fflush(stdout)
            let stages = source.lastDetectorCommandTiming.sorted { $0.key < $1.key }
              .map { "\($0.key)=\($0.value)" }.joined(separator: " ")
            print("RESIDENCY_HOLD_COMMAND arm=\(arm) cycle=\(cycle) case=\(index) \(stages)")
            fflush(stdout)
          }
        }
      }
    }
    if heapReplay {
      source.releaseResidentStorage()
      XCTAssertEqual(source.residentBytes, 0)
      XCTAssertEqual(source.experimentalRecordHeapCount, 0)
      print(
        "HEAP_REPLAY_RELEASE source_bytes=\(source.residentBytes) heap_bytes=\(source.experimentalRecordHeapBytes) allocated_bytes=\(device.currentAllocatedSize)"
      )
      fflush(stdout)
    }
  }
}
