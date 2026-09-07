import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

do {
  let input = URL(fileURLWithPath: CommandLine.arguments[1])
  let destination = URL(fileURLWithPath: CommandLine.arguments[2])
  let cache = destination.deletingLastPathComponent().appendingPathComponent("Indexes")
  let native = try Native4DSTEMCatalogBuilder(cacheDirectory: cache).prepare(input: input).datasets[
    0]
  print("CATALOG \(native.label)")
  fflush(stdout)
  let device = MTLCreateSystemDefaultDevice()!
  let indexed = try Native4DSTEMIndexedSource.open(dataset: native)
  print("INDEXED")
  fflush(stdout)
  if CommandLine.arguments.last == "direct-regression" {
    precondition(
      native.scanRows * native.scanCols == 12288
        && native.detectorRows * native.detectorCols == 4096)
    precondition(ProcessInfo.processInfo.environment["QGPU_ORIGINAL_SCALAR_DECODE"] != "1")
    let frames = 4096
    let pixels = 4096
    let windows = try indexed.windows(
      maximumDecodedBytes: UInt64(frames * pixels * 2), alignToScanRows: false)
    precondition(windows.count == 3)
    var readReserve: UInt64 = 0
    for slice in windows.flatMap(\.slices) {
      let words = Array(
        indexed.shards[slice.shardIndex].index.metadataWords[slice.metadataWordRange])
      var first = UInt64.max
      var last: UInt64 = 0
      for i in stride(from: 0, to: words.count, by: 2) {
        first = min(first, UInt64(words[i]))
        last = max(last, UInt64(words[i]) + UInt64(words[i + 1]))
      }
      readReserve = max(readReserve, last - first + UInt64(words.count * 8))
    }
    let shardBytes = UInt64(frames * pixels * 2 + pixels * 20 * 4)
    let budget =
      UInt64(frames * pixels * 2) + (768 << 20) + readReserve + shardBytes + shardBytes / 2
    var progressCount = 0
    var budgetRejected = false
    let readAhead = ProcessInfo.processInfo.environment["QGPU_ORIGINAL_READ_AHEAD"] != "0"
    do {
      let admitted = try MetalCompactH5Loader.load(
        source: indexed, device: device, maximumAdditionalBytes: budget,
        progress: { _, _ in progressCount += 1 })
      defer { admitted.releaseResidentStorage() }
      precondition(readAhead, "Sequential admission must retain its original reserve")
      precondition(admitted.loadMetrics.plannedAdditionalBytes <= budget)
      for frame in [0, 4095, 4096, 8192, 12287] {
        let dp = try admitted.extractDiffraction(
          scanRow: frame / native.scanCols, scanColumn: frame % native.scanCols)
        precondition(dp.count == pixels && dp.allSatisfy { $0 == 65535 })
      }
    } catch Metal4DSTEMStreamingIOError.invalidRequest(let message) {
      precondition(message.contains("Exact packed counts exceed"), message)
      budgetRejected = true
    }
    if readAhead {
      precondition(!budgetRejected && progressCount == 3)
      print("DIRECT_READ_AHEAD_OLD_RESERVE_REGRESSION_PASS")
    } else {
      precondition(
        budgetRejected && progressCount == 1,
        "Sequential budget must admit first shard and reject second before payload allocation")
    }
    func recover() throws {
      let fresh = try Native4DSTEMIndexedSource.open(dataset: native)
      let restored = try MetalCompactH5Loader.load(source: fresh, device: device)
      defer { restored.releaseResidentStorage() }
      for frame in [0, 4095, 4096, 8192, 12287] {
        let dp = try restored.extractDiffraction(
          scanRow: frame / native.scanCols, scanColumn: frame % native.scanCols)
        precondition(dp.count == pixels && dp.allSatisfy { $0 == 65535 })
      }
      let basis = try restored.preparedDPCMomentValues()!
      let total = UInt64(pixels) * 65535
      let weighted = UInt64(64 * 64 * 63 / 2) * 65535
      precondition(basis.total.allSatisfy { $0 == total })
      precondition(
        basis.detectorRowMoment.allSatisfy { $0 == weighted }
          && basis.detectorColumnMoment.allSatisfy { $0 == weighted })
    }
    try recover()
    for changedURL in [URL(fileURLWithPath: native.dataFiles[0]), input] {
      var originalStat = stat()
      precondition(changedURL.path.withCString { lstat($0, &originalStat) } == 0)
      var restoreTimes = [originalStat.st_atimespec, originalStat.st_mtimespec]
      var changed = false
      var mutationRejected = false
      var mutationFailed = false
      do {
        defer { precondition(utimensat(AT_FDCWD, changedURL.path, &restoreTimes, 0) == 0) }
        do {
          _ = try MetalCompactH5Loader.load(
            source: indexed, device: device,
            progress: { _, _ in
              if !changed {
                var times = restoreTimes
                times[1].tv_sec += 10
                mutationFailed = utimensat(AT_FDCWD, changedURL.path, &times, 0) != 0
                changed = true
              }
            })
        } catch Native4DSTEMIOError.invalidData {
          mutationRejected = true
        } catch Metal4DSTEMStreamingIOError.invalidRequest { mutationRejected = true }
      }
      precondition(
        changed && !mutationFailed && mutationRejected,
        "Changed raw shard must not publish residency")
      try recover()
    }
    var didProgress = false
    var didCancel = false
    do {
      _ = try MetalCompactH5Loader.load(
        source: indexed, device: device,
        shouldCancel: { didProgress }, progress: { _, _ in didProgress = true })
    } catch Metal4DSTEMStreamingIOError.cancelled { didCancel = true }
    precondition(didCancel)
    try recover()
    // Count only polls after the last source progress event. Read-ahead's timed
    // polling before that event cannot influence this deterministic late gate.
    // With no auxiliary budget, four polls cover pack completion, pre-library,
    // resident completion, and the final source-bound public publication.
    var completedSource = false
    var completionPolls = 0
    let lateBaseline = try MetalCompactH5Loader.load(
      source: indexed, device: device,
      shouldCancel: {
        if completedSource { completionPolls += 1 }
        return false
      }, progress: { done, total in completedSource = done == total })
    precondition(
      completedSource && completionPolls == 4,
      "Update the phase proof if completion cancellation polls change")
    lateBaseline.releaseResidentStorage()
    completedSource = false
    completionPolls = 0
    var lateCancelled = false
    var latePublished = false
    do {
      let unexpected = try MetalCompactH5Loader.load(
        source: indexed, device: device,
        shouldCancel: {
          guard completedSource else { return false }
          completionPolls += 1
          return completionPolls == 4
        }, progress: { done, total in completedSource = done == total })
      latePublished = true
      unexpected.releaseResidentStorage()
    } catch Metal4DSTEMStreamingIOError.cancelled { lateCancelled = true }
    precondition(lateCancelled && !latePublished && completedSource && completionPolls == 4)
    try recover()
    print("FINAL_PUBLICATION_CANCELLATION_AND_EXACT_RECOVERY_PASS")
    precondition(!FileManager.default.fileExists(atPath: destination.path))
    print("DIRECT_MIDSTREAM_BUDGET_SOURCE_MUTATION_AND_RECOVERY_PASS budget=\(budget)")
    exit(0)
  }
  if CommandLine.arguments.last == "direct-budget" {
    let frames = 2048
    let pixels = 192 * 192
    let scans = frames * 6
    precondition(native.scanRows * native.scanCols == scans)
    precondition(native.detectorRows == 192 && native.detectorCols == 192)
    precondition(ProcessInfo.processInfo.environment["QGPU_ORIGINAL_READ_AHEAD"] == "1")
    precondition(ProcessInfo.processInfo.environment["QGPU_ORIGINAL_SCALAR_DECODE"] == "1")
    let windowPayload = UInt64(frames * pixels * 2)
    let windowHeaders = UInt64(pixels * 10 * 4)
    let shardBytes = windowPayload + windowHeaders
    func verify(_ resident: MetalCompactH5ResidentSource) throws {
      precondition(resident.metadata.workingDtype == "uint16")
      precondition(resident.metadata.shardCount == 6)
      precondition(resident.metadata.residentBytes == shardBytes * 6 + UInt64(scans * 32))
      for frame in [0, 4095, 4096, 8191, 8192, scans - 1] {
        let dp = try resident.extractDiffraction(
          scanRow: frame / native.scanCols, scanColumn: frame % native.scanCols)
        precondition(dp.count == pixels && dp.allSatisfy { $0 == 65535 })
      }
      let total = UInt64(pixels) * 65535
      let weighted = UInt64(192 * 192 * 191 / 2) * 65535
      let dpc = try resident.preparedDPCMomentValues()!
      precondition(dpc.total.count == scans && dpc.total.allSatisfy { $0 == total })
      precondition(dpc.detectorRowMoment.allSatisfy { $0 == weighted })
      precondition(dpc.detectorColumnMoment.allSatisfy { $0 == weighted })
      _ = try resident.updateVirtualDetector(mask: [UInt8](repeating: 1, count: pixels))
      let virtual = try resident.virtualDetectorValues()
      precondition(virtual.count == scans && virtual.allSatisfy { $0 == UInt32(total) })
    }
    let seed = try MetalCompactH5Loader.load(source: indexed, device: device)
    try verify(seed)
    let planned = seed.loadMetrics.plannedAdditionalBytes
    let staging = seed.loadMetrics.maximumTransientBytes
    seed.releaseResidentStorage()
    let budget = planned + (8 << 20)
    let partialDPC = UInt64(frames * (pixels / 4096) * 32)
    let oldStaging = windowPayload * 2 + partialDPC + (768 << 20)
    // This is below the old window-6 admission even without its input reserve.
    // It is above the unchanged initial guard and the reported conservative
    // publication plan, so an old-code failure is specifically over-reservation.
    precondition(budget > oldStaging && budget < oldStaging + shardBytes * 6)
    var visited = [Int]()
    let admitted = try MetalCompactH5Loader.load(
      source: indexed, device: device, maximumAdditionalBytes: budget,
      progress: { done, _ in visited.append(done) })
    try verify(admitted)
    precondition(visited == [2048, 4096, 6144, 8192, 10240, 12288])
    precondition(admitted.loadMetrics.plannedAdditionalBytes <= budget)
    admitted.releaseResidentStorage()
    let insufficient = budget - shardBytes / 2
    precondition(insufficient > oldStaging)
    visited.removeAll()
    var rejected = false
    do {
      let unexpected = try MetalCompactH5Loader.load(
        source: indexed, device: device, maximumAdditionalBytes: insufficient,
        progress: { done, _ in visited.append(done) })
      unexpected.releaseResidentStorage()
    } catch Metal4DSTEMStreamingIOError.invalidRequest(let message) {
      precondition(message.contains("Exact packed counts exceed"), message)
      rejected = true
    }
    precondition(
      rejected && visited == [2048, 4096, 6144, 8192, 10240], "Reject window 6 before allocation")
    let restored = try MetalCompactH5Loader.load(
      source: indexed, device: device, maximumAdditionalBytes: budget)
    try verify(restored)
    restored.releaseResidentStorage()
    precondition(!FileManager.default.fileExists(atPath: destination.path))
    print(
      "DIRECT_TIGHT_BUDGET_EXACT_RECOVERY_PASS planned=\(planned) staging=\(staging) budget=\(budget) insufficient=\(insufficient)"
    )
    exit(0)
  }
  let started = CFAbsoluteTimeGetCurrent()
  let cached = CommandLine.arguments.last == "direct-cache"
  let direct = cached || CommandLine.arguments.last == "direct"
  var preparedDPC: MetalCompactH5ExactDPCMoments?
  if cached {
    let seed = try MetalCompactH5Loader.load(source: indexed, device: device)
    preparedDPC = try seed.preparedDPCMomentValues()
    precondition(preparedDPC?.sourceIdentitySHA256 == native.sourceIdentitySHA256)
    seed.releaseResidentStorage()
  }
  let resident: MetalCompactH5ResidentSource
  if direct {
    resident = try MetalCompactH5Loader.load(
      source: indexed, device: device,
      maximumAdditionalBytes: UInt64(4) << 30,
      preparedDPC: preparedDPC,
      progress: { done, total in
        print("PACK \(done)/\(total)")
        fflush(stdout)
      })
    precondition(resident.metadata.sourceRawLogicalSHA256 == nil)
    precondition(resident.metadata.workingLogicalSHA256 == nil)
    precondition(resident.loadMetrics.nativeCacheStatus == "originalDirect")
    precondition(resident.loadMetrics.reusedPreparedDPC == cached)
    precondition(!FileManager.default.fileExists(atPath: destination.path))
  } else {
    try MetalCompactH5Loader.prepare(
      source: indexed, destinationURL: destination, device: device,
      progress: { done, total in
        print("PACK \(done)/\(total)")
        fflush(stdout)
      })
    resident = try MetalCompactH5Loader.load(sourceURL: destination, device: device)
  }
  let packed = CFAbsoluteTimeGetCurrent() - started
  let publicMetrics = try JSONSerialization.data(
    withJSONObject: [
      "gpu_decode_and_header_ms": resident.loadMetrics.gpuDecodeAndHeaderMilliseconds
    ], options: [.sortedKeys])
  print("PUBLIC_LOAD_METRICS \(String(data: publicMetrics, encoding: .utf8)!)")
  _ = try Metal4DSTEMResidentCapabilities.compact(resident)
  print(
    "READY \(native.label) \(resident.metadata.workingDtype) \(resident.loadMetrics.totalResidentBytes) bytes pack=\(packed)s"
  )
  if CommandLine.arguments.count > 3 {
    let oracle = try Data(contentsOf: URL(fileURLWithPath: CommandLine.arguments[3]))
    let expected = oracle.withUnsafeBytes { Array($0.bindMemory(to: UInt16.self)) }
    precondition(resident.metadata.workingDtype == (expected.max()! <= 255 ? "uint8" : "uint16"))
    let scans = native.scanRows * native.scanCols
    let pixels = native.detectorRows * native.detectorCols
    precondition(expected.count == scans * pixels)
    for frame in 0..<scans {
      let actual = try resident.extractDiffraction(
        scanRow: frame / native.scanCols, scanColumn: frame % native.scanCols)
      precondition(
        actual == expected[frame * pixels..<(frame + 1) * pixels].map(UInt32.init),
        "DP mismatch \(frame)")
    }
    for offset in [0, 1, 2] {
      let mask = (0..<pixels).map { UInt8(($0 + offset) % 3 == 0 ? 1 : 0) }
      _ = try resident.updateVirtualDetector(mask: mask)
      let actual = try resident.virtualDetectorValues()
      for frame in 0..<scans {
        let sum = (0..<pixels).reduce(UInt32(0)) {
          $0 + (mask[$1] != 0 ? UInt32(expected[frame * pixels + $1]) : 0)
        }
        precondition(actual[frame] == sum, "detector mismatch")
      }
    }
    let moments = try resident.preparedDPCMomentValues()!
    for frame in 0..<scans {
      var total: UInt64 = 0
      var row: UInt64 = 0
      var column: UInt64 = 0
      for pixel in 0..<pixels {
        let value = UInt64(expected[frame * pixels + pixel])
        total += value
        row += value * UInt64(pixel / native.detectorCols)
        column += value * UInt64(pixel % native.detectorCols)
      }
      precondition(
        moments.total[frame] == total && moments.detectorRowMoment[frame] == row
          && moments.detectorColumnMoment[frame] == column, "DPC mismatch")
    }
    print(
      "EXACT_PARITY_PASS \(expected.count) original counts, 3 arbitrary detectors, all DPC sums")
  }
  resident.releaseResidentStorage()
  if let preparedDPC {
    let invalid = [
      MetalCompactH5ExactDPCMoments(
        total: preparedDPC.total,
        detectorRowMoment: preparedDPC.detectorRowMoment,
        detectorColumnMoment: preparedDPC.detectorColumnMoment,
        sourceIdentitySHA256: String(repeating: "0", count: 64),
        detectorMaskSHA256: preparedDPC.detectorMaskSHA256),
      MetalCompactH5ExactDPCMoments(
        total: preparedDPC.total,
        detectorRowMoment: preparedDPC.detectorRowMoment,
        detectorColumnMoment: preparedDPC.detectorColumnMoment,
        sourceIdentitySHA256: preparedDPC.sourceIdentitySHA256,
        detectorMaskSHA256: String(repeating: "0", count: 64)),
      MetalCompactH5ExactDPCMoments(
        total: Array(preparedDPC.total.dropLast()),
        detectorRowMoment: preparedDPC.detectorRowMoment,
        detectorColumnMoment: preparedDPC.detectorColumnMoment,
        sourceIdentitySHA256: preparedDPC.sourceIdentitySHA256,
        detectorMaskSHA256: preparedDPC.detectorMaskSHA256),
      MetalCompactH5ExactDPCMoments(
        total: [UInt64.max] + preparedDPC.total.dropFirst(),
        detectorRowMoment: preparedDPC.detectorRowMoment,
        detectorColumnMoment: preparedDPC.detectorColumnMoment,
        sourceIdentitySHA256: preparedDPC.sourceIdentitySHA256,
        detectorMaskSHA256: preparedDPC.detectorMaskSHA256),
    ]
    for hint in invalid {
      let checked = try MetalCompactH5Loader.load(
        source: indexed, device: device, preparedDPC: hint)
      precondition(!checked.loadMetrics.reusedPreparedDPC)
      let checkedDPC = try checked.preparedDPCMomentValues()
      precondition(checkedDPC == preparedDPC)
      checked.releaseResidentStorage()
    }
    print("CACHED_DPC_REUSE_AND_REJECTION_PASS")
  }
  if direct {
    for cancelAfterProgress in [false, true] {
      var progressed = false
      var rejected = false
      do {
        _ = try MetalCompactH5Loader.load(
          source: indexed, device: device,
          shouldCancel: { !cancelAfterProgress || progressed },
          progress: { _, _ in progressed = true })
      } catch Metal4DSTEMStreamingIOError.cancelled { rejected = true }
      precondition(rejected)
    }
    var rejectedBudget = false
    do {
      _ = try MetalCompactH5Loader.load(source: indexed, device: device, maximumAdditionalBytes: 1)
    } catch { rejectedBudget = true }
    precondition(rejectedBudget && !FileManager.default.fileExists(atPath: destination.path))
    print("DIRECT_CANCELLATION_BUDGET_AND_NO_CACHE_PASS")
    exit(0)
  }
  let before = try Data(contentsOf: destination)
  var rejectedOverwrite = false
  do {
    try MetalCompactH5Loader.prepare(source: indexed, destinationURL: destination, device: device)
  } catch { rejectedOverwrite = true }
  let after = try Data(contentsOf: destination)
  precondition(
    rejectedOverwrite && after == before,
    "Existing packed output must not be overwritten")
  for cancelAfterProgress in [false, true] {
    let cancelledURL = destination.appendingPathExtension(
      cancelAfterProgress ? "cancel-late" : "cancel-early")
    var progressed = false
    var rejected = false
    do {
      try MetalCompactH5Loader.prepare(
        source: indexed, destinationURL: cancelledURL, device: device,
        shouldCancel: { !cancelAfterProgress || progressed },
        progress: { _, _ in progressed = true })
    } catch Metal4DSTEMStreamingIOError.cancelled { rejected = true }
    precondition(rejected && !FileManager.default.fileExists(atPath: cancelledURL.path))
  }
  let names = try FileManager.default.contentsOfDirectory(
    atPath: destination.deletingLastPathComponent().path)
  precondition(!names.contains { $0.hasPrefix(".packing-") && $0.hasSuffix(".partial") })
  print("CANCELLATION_AND_IMMUTABILITY_PASS")
} catch {
  fputs("ERROR: \(error.localizedDescription)\n", stderr)
  exit(1)
}
