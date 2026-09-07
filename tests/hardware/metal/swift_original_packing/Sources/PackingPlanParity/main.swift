import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

struct Oracle: Decodable {
  let countsSHA256: String
  let detectorSHA256: [String]
  let dpcSHA256: String
  let sumSHA256: String
  let maximum: Int
}

func sha(_ words: [UInt32]) -> String {
  words.withUnsafeBytes { SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined() }
}

do {
  let master = URL(fileURLWithPath: CommandLine.arguments[1])
  let plan = URL(fileURLWithPath: CommandLine.arguments[2])
  let oracle = try JSONDecoder().decode(
    Oracle.self,
    from: Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[3])))
  let mode = CommandLine.arguments[4]
  let device = MTLCreateSystemDefaultDevice()!
  let native = try Native4DSTEMCatalogBuilder(
    cacheDirectory:
      plan.deletingLastPathComponent().appendingPathComponent("Indexes")
  )
  .prepare(input: master).datasets[0]
  let source = try Native4DSTEMIndexedSource.open(dataset: native)
  let scans = native.scanRows * native.scanCols
  let pixels = native.detectorRows * native.detectorCols
  let receipt = plan.deletingLastPathComponent().appendingPathComponent(
    "baseline-resident-bytes.json")

  func check(_ resident: MetalCompactH5ResidentSource) throws {
    defer { resident.releaseResidentStorage() }
    precondition(resident.metadata.workingDtype == (oracle.maximum <= 255 ? "uint8" : "uint16"))
    var digest = SHA256()
    for frame in 0..<scans {
      let values = try resident.extractDiffraction(
        scanRow: frame / native.scanCols, scanColumn: frame % native.scanCols)
      precondition(values.count == pixels)
      values.withUnsafeBytes { digest.update(bufferPointer: $0) }
    }
    precondition(
      digest.finalize().map { String(format: "%02x", $0) }.joined() == oracle.countsSHA256,
      "Every-DP original-count SHA differs")
    for offset in 0..<3 {
      let mask = (0..<pixels).map { UInt8(($0 + offset) % 3 == 0 ? 1 : 0) }
      _ = try resident.updateVirtualDetector(mask: mask)
      let detectorValues = try resident.virtualDetectorValues()
      precondition(sha(detectorValues) == oracle.detectorSHA256[offset])
    }
    let moments = try resident.preparedDPCMomentValues()!
    var dpc = SHA256()
    for frame in 0..<scans {
      let values = [
        moments.total[frame].littleEndian, moments.detectorRowMoment[frame].littleEndian,
        moments.detectorColumnMoment[frame].littleEndian, UInt64(0),
      ]
      values.withUnsafeBytes { dpc.update(bufferPointer: $0) }
    }
    precondition(dpc.finalize().map { String(format: "%02x", $0) }.joined() == oracle.dpcSHA256)
    let sums = try resident.meanDiffractionPattern().detectorSum
    let actualSum = sums.withUnsafeBytes {
      SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
    }
    precondition(actualSum == oracle.sumSHA256, "Fresh detector sums differ")
    if FileManager.default.fileExists(atPath: receipt.path) {
      let bytes = try JSONDecoder().decode(UInt64.self, from: Data(contentsOf: receipt))
      precondition(
        resident.loadMetrics.residentBytes == bytes, "Plan changed retained original-count memory")
    } else {
      // Optional detector-region products may fit on a cheaper reopen even
      // when they did not fit initial decode staging. Count storage is stable.
      try JSONEncoder().encode(resident.loadMetrics.residentBytes).write(to: receipt)
    }
    print("PACKING_PLAN_EXACT_ALL_DP_DPC_SUMS_PASS \(scans * pixels)")
  }

  if mode.hasPrefix("bitshuffle-") {
    precondition(native.sourceDtype == "uint16")
    let windows = try source.windows(
      maximumDecodedBytes: UInt64(4096 * pixels * 2), alignToScanRows: false)
    precondition(
      windows.count == 2 && windows[0].slices.map { $0.globalFrameRange.count } == [204, 3892],
      "Fixture must exercise short and long slices in the same packed window")
    let seed = try MetalCompactH5Loader.load(
      source: source, device: device,
      packingPlanURL: ["bitshuffle-audit", "bitshuffle-overlap-fault"].contains(mode) ? nil : plan)
    let savedDPC = try seed.preparedDPCMomentValues()!
    try check(seed)
    func candidate(shouldCancel: () -> Bool = { false }, progress: (Int, Int) -> Void = { _, _ in })
      throws -> MetalCompactH5ResidentSource
    {
      try MetalCompactH5Loader.load(
        source: source, device: device,
        maximumAdditionalBytes: UInt64(2) << 30, preparedDPC: savedDPC,
        packingPlanURL: plan, shouldCancel: shouldCancel, progress: progress)
    }
    if mode == "bitshuffle-overlap-cancel" {
      precondition(ProcessInfo.processInfo.environment["QGPU_ORIGINAL_PLAN_OVERLAP"] == "1")
      precondition(ProcessInfo.processInfo.environment["QGPU_ORIGINAL_READ_AHEAD"] == "0")
      var firstProgress = false
      var firstWindowPolls = 0
      let control = try candidate(
        shouldCancel: {
          if !firstProgress { firstWindowPolls += 1 }
          return false
        }, progress: { _, _ in firstProgress = true })
      precondition((control.loadMetrics.gpuDecodeAndPackingMilliseconds ?? 0) > 0)
      try check(control)
      precondition(firstWindowPolls >= 3)
      // Direct overlap polls immediately before/after next-plan preparation,
      // then waits and performs its final poll before progress. With count
      // read-ahead disabled, these positions are repeatable without test hooks.
      for target in [firstWindowPolls - 2, firstWindowPolls - 1] {
        var polls = 0
        var progressed = false
        var cancelled = false
        do {
          let unexpected = try candidate(
            shouldCancel: {
              polls += 1
              return polls == target
            }, progress: { _, _ in progressed = true })
          unexpected.releaseResidentStorage()
        } catch Metal4DSTEMStreamingIOError.cancelled { cancelled = true }
        precondition(
          cancelled && !progressed && polls == target,
          "Prefetch cancellation must drain the pending command before returning")
        try check(candidate())
      }
      print("PACKING_PLAN_OVERLAP_PREFETCH_CANCEL_DRAIN_RECOVERY_PASS polls=\(firstWindowPolls)")
    } else if mode == "bitshuffle-recover" {
      var rejectedBudget = false
      do {
        let unexpected = try MetalCompactH5Loader.load(
          source: source, device: device,
          maximumAdditionalBytes: 1, preparedDPC: savedDPC, packingPlanURL: plan)
        unexpected.releaseResidentStorage()
      } catch Metal4DSTEMStreamingIOError.invalidRequest { rejectedBudget = true }
      precondition(rejectedBudget, "Optional direct path must respect the load budget")
      var progressed = false
      var cancelled = false
      do {
        let unexpected = try candidate(
          shouldCancel: { progressed }, progress: { _, _ in progressed = true })
        unexpected.releaseResidentStorage()
      } catch Metal4DSTEMStreamingIOError.cancelled { cancelled = true }
      precondition(progressed && cancelled, "Do not publish a partially decoded direct resident")
      try check(candidate())
      let changed = URL(fileURLWithPath: native.dataFiles[0])
      var attributes = stat()
      precondition(lstat(changed.path, &attributes) == 0)
      var originalTimes = [attributes.st_atimespec, attributes.st_mtimespec]
      var changedOnce = false
      var rejected = false
      do {
        defer { precondition(utimensat(AT_FDCWD, changed.path, &originalTimes, 0) == 0) }
        do {
          let unexpected = try candidate(progress: { _, _ in
            if !changedOnce {
              var advanced = originalTimes
              advanced[1].tv_sec += 10
              precondition(utimensat(AT_FDCWD, changed.path, &advanced, 0) == 0)
              precondition(utimensat(AT_FDCWD, changed.path, &originalTimes, 0) == 0)
              changedOnce = true
            }
          })
          unexpected.releaseResidentStorage()
        } catch Metal4DSTEMStreamingIOError.invalidRequest {
          rejected = true
        } catch Native4DSTEMIOError.invalidData { rejected = true }
      }
      precondition(changedOnce && rejected, "Restored mtime must not hide source change")
      try check(candidate())
      print("DIRECT_BITSHUFFLE_CANCEL_SOURCE_FRESHNESS_RECOVERY_PASS")
    } else if mode == "bitshuffle-malformed" {
      // Record a complete eligible direct reopen before introducing a fault.
      // This is source-mutation recovery coverage; the separate decoder suite
      // proves malformed-LZ4 handling without source freshness interference.
      try check(candidate())
      // Mutate only this synthetic fixture after the first window completes.
      // Token 0 followed by match distance 0 must fail the checked LZ4 decoder.
      precondition(ProcessInfo.processInfo.environment["QGPU_ORIGINAL_READ_AHEAD"] == "0")
      let slice = windows[1].slices[0]
      let shard = source.shards[windows[1].slices[0].shardIndex]
      let offset =
        slice.chunkCompressedByteRange.lowerBound
        + UInt64(shard.index.metadataWords[slice.metadataWordRange.lowerBound])
      let descriptor = open(shard.sourceURL.path, O_RDWR)
      precondition(descriptor >= 0)
      defer { close(descriptor) }
      var original = [UInt8](repeating: 0, count: 3)
      var attributes = stat()
      precondition(pread(descriptor, &original, original.count, off_t(offset)) == original.count)
      precondition(fstat(descriptor, &attributes) == 0)
      var originalTimes = [attributes.st_atimespec, attributes.st_mtimespec]
      var injected = false
      var rejected = false
      do {
        defer {
          precondition(
            pwrite(descriptor, &original, original.count, off_t(offset)) == original.count)
          precondition(futimens(descriptor, &originalTimes) == 0)
        }
        do {
          let unexpected = try candidate(progress: { done, _ in
            if !injected {
              precondition(done == 4096)
              var invalid: [UInt8] = [0, 0, 0]
              precondition(
                pwrite(descriptor, &invalid, invalid.count, off_t(offset)) == invalid.count)
              precondition(futimens(descriptor, &originalTimes) == 0)
              injected = true
            }
          })
          unexpected.releaseResidentStorage()
        } catch Metal4DSTEMStreamingIOError.invalidRequest {
          rejected = true
        } catch Native4DSTEMIOError.invalidData { rejected = true }
      }
      precondition(injected && rejected, "Malformed original counts must never publish")
      try check(candidate())
      print("DIRECT_BITSHUFFLE_MIDSTREAM_SOURCE_MUTATION_RECOVERY_PASS")
    } else {
      precondition(
        ["bitshuffle-cycle", "bitshuffle-audit", "bitshuffle-overlap-fault"].contains(mode))
      let loaded = try candidate()
      if mode == "bitshuffle-overlap-fault" {
        precondition(
          (loaded.loadMetrics.gpuDecodeAndPackingMilliseconds ?? 0) > 0,
          "Fallback must retain GPU timing of the already submitted direct command")
        print("PACKING_PLAN_OVERLAP_FALLBACK_COMBINED_INTERVAL_PASS")
      }
      try check(loaded)
      print("DIRECT_BITSHUFFLE_EXACT_REOPEN_PASS")
    }
  } else if mode == "cycle" {
    precondition(!FileManager.default.fileExists(atPath: plan.path))
    try check(MetalCompactH5Loader.load(source: source, device: device, packingPlanURL: plan))
    precondition(FileManager.default.fileExists(atPath: plan.path), "Cache miss did not store plan")
    try check(MetalCompactH5Loader.load(source: source, device: device, packingPlanURL: plan))
  } else if mode == "recover" {
    var rejectedBudget = false
    do {
      _ = try MetalCompactH5Loader.load(
        source: source, device: device,
        maximumAdditionalBytes: 1, packingPlanURL: plan)
    } catch Metal4DSTEMStreamingIOError.invalidRequest { rejectedBudget = true }
    precondition(rejectedBudget)
    var complete = false
    var rejectedCancel = false
    do {
      _ = try MetalCompactH5Loader.load(
        source: source, device: device, packingPlanURL: plan,
        shouldCancel: { complete }, progress: { _, _ in complete = true })
    } catch Metal4DSTEMStreamingIOError.cancelled { rejectedCancel = true }
    precondition(rejectedCancel)
    try check(MetalCompactH5Loader.load(source: source, device: device, packingPlanURL: plan))
    print("PACKING_PLAN_BUDGET_CANCEL_RECOVERY_PASS")
  } else if mode == "payload-budget" {
    precondition(native.sourceDtype == "uint16")
    let frames = 4096
    let base = UInt64(frames * pixels * 4 + frames * (pixels / 4096) * 32) + (768 << 20)
    let windows = try source.windows(
      maximumDecodedBytes: UInt64(frames * pixels * 2), alignToScanRows: false)
    var readReserve: UInt64 = 0
    for slice in windows.flatMap(\.slices) {
      let words = Array(
        source.shards[slice.shardIndex].index.metadataWords[slice.metadataWordRange])
      var first = UInt64.max
      var last: UInt64 = 0
      for offset in stride(from: 0, to: words.count, by: 2) {
        first = min(first, UInt64(words[offset]))
        last = max(last, UInt64(words[offset]) + UInt64(words[offset + 1]))
      }
      readReserve = max(readReserve, last - first + UInt64(words.count * 8))
    }
    let normalBytes = try JSONDecoder().decode(UInt64.self, from: Data(contentsOf: receipt))
    let budget = base + (64 << 20) + readReserve + normalBytes
    let repaired = try MetalCompactH5Loader.load(
      source: source, device: device,
      maximumAdditionalBytes: budget, packingPlanURL: plan)
    precondition(
      repaired.loadMetrics.plannedAdditionalBytes <= budget,
      "Repaired layout exceeded the unchanged loading budget")
    try check(repaired)
    print("PACKING_PLAN_OVERSIZED_PAYLOAD_BUDGET_RECOVERY_PASS")
  } else if mode == "reserve" {
    // This exact fixture admits normal two-window low/high-u16 residency but
    // cannot afford the optional plan's additional 64 MiB staging reservation.
    precondition(native.sourceDtype == "uint16")
    // Scalar decoding and fused DPC are production defaults, not prerequisites
    // that callers must enable through experimental environment variables.
    let frames = 4096
    let base = UInt64(frames * pixels * 4 + frames * (pixels / 4096) * 32) + (768 << 20)
    let budget = base + (63 << 20)
    try check(
      MetalCompactH5Loader.load(source: source, device: device, maximumAdditionalBytes: budget))
    try check(
      MetalCompactH5Loader.load(
        source: source, device: device,
        maximumAdditionalBytes: budget, packingPlanURL: plan))
    print("PACKING_PLAN_OPTIONAL_RESERVE_FALLBACK_PASS")
  } else if mode == "source-change" {
    let changed = URL(fileURLWithPath: native.dataFiles[0])
    var original = stat()
    precondition(changed.path.withCString { lstat($0, &original) } == 0)
    var times = [original.st_atimespec, original.st_mtimespec]
    var changedOnce = false
    var rejected = false
    do {
      defer { precondition(utimensat(AT_FDCWD, changed.path, &times, 0) == 0) }
      do {
        let unexpected = try MetalCompactH5Loader.load(
          source: source, device: device, packingPlanURL: plan,
          progress: { _, _ in
            if !changedOnce {
              var advanced = times
              advanced[1].tv_sec += 10
              precondition(utimensat(AT_FDCWD, changed.path, &advanced, 0) == 0)
              // Hide the mtime change again before returning to the loader.
              // The changed ctime must still prevent publication.
              precondition(utimensat(AT_FDCWD, changed.path, &times, 0) == 0)
              changedOnce = true
            }
          })
        unexpected.releaseResidentStorage()
      } catch Metal4DSTEMStreamingIOError.invalidRequest {
        rejected = true
      } catch Native4DSTEMIOError.invalidData { rejected = true }
    }
    precondition(changedOnce && rejected)
    let fresh = try Native4DSTEMIndexedSource.open(dataset: native)
    try check(MetalCompactH5Loader.load(source: fresh, device: device, packingPlanURL: plan))
    print("PACKING_PLAN_SOURCE_CHANGE_RECOVERY_PASS")
  } else {
    precondition(mode == "audit")
    try check(MetalCompactH5Loader.load(source: source, device: device, packingPlanURL: plan))
  }
} catch {
  fputs("ERROR: \(error)\n", stderr)
  exit(1)
}
