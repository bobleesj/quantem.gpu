import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

struct Oracle: Decodable {
  let countsSHA256: String
  let detectorSHA256: [String]
  let dpcSHA256: String
}

/// Test-owned, source-bound reduction metadata; never a cached count volume.
struct SavedDPC: Codable {
  let total: [UInt64]
  let row: [UInt64]
  let column: [UInt64]
  let sourceIdentity: String
  let maskIdentity: String

  init(_ values: MetalCompactH5ExactDPCMoments) {
    total = values.total
    row = values.detectorRowMoment
    column = values.detectorColumnMoment
    sourceIdentity = values.sourceIdentitySHA256!
    maskIdentity = values.detectorMaskSHA256!
  }

  var values: MetalCompactH5ExactDPCMoments {
    MetalCompactH5ExactDPCMoments(
      total: total, detectorRowMoment: row,
      detectorColumnMoment: column, sourceIdentitySHA256: sourceIdentity,
      detectorMaskSHA256: maskIdentity)
  }
}

func digest(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

do {
  let folder = URL(fileURLWithPath: CommandLine.arguments[1])
  let mode = CommandLine.arguments[2]
  let sourceCount = Int(CommandLine.arguments[3])!
  let device = MTLCreateSystemDefaultDevice()!
  let catalog = Native4DSTEMCatalogBuilder(cacheDirectory: folder.appendingPathComponent("Indexes"))
  let oracle = try (0..<sourceCount).map {
    try JSONDecoder().decode(
      Oracle.self, from: Data(contentsOf: folder.appendingPathComponent("oracle-\($0).json")))
  }
  let sources = try (0..<sourceCount).map { index -> Native4DSTEMIndexedSource in
    let dataset = try catalog.prepare(
      input: folder.appendingPathComponent("sample-\(index)_master.h5")
    ).datasets[0]
    return try Native4DSTEMIndexedSource.open(dataset: dataset)
  }
  let receipt = folder.appendingPathComponent("baseline-budget.json")
  let dpcReceipt = folder.appendingPathComponent("budget-source-dpc.json")
  let plan = folder.appendingPathComponent("budget-source.qgplan")
  let directBudgetModes = ["plan-baseline", "tight-baseline", "budget"]
  let defaultBudget = UInt64(2) << 30
  let budget: UInt64
  if ["tight-baseline", "budget"].contains(mode) {
    budget = try JSONDecoder().decode(UInt64.self, from: Data(contentsOf: receipt))
  } else {
    budget = defaultBudget
  }
  var savedDPC: MetalCompactH5ExactDPCMoments?
  if mode == "plan-baseline" {
    precondition(sourceCount == 1 && sources[0].dataset.sourceDtype == "uint16")
    let seed = try MetalCompactH5Loader.load(
      source: sources[0], device: device,
      maximumAdditionalBytes: defaultBudget, packingPlanURL: plan)
    let saved = try SavedDPC(seed.preparedDPCMomentValues()!)
    seed.releaseResidentStorage()
    try JSONEncoder().encode(saved).write(to: dpcReceipt)
    savedDPC = saved.values
    precondition(
      FileManager.default.fileExists(atPath: plan.path), "Budget test requires a real saved layout")
  } else if directBudgetModes.contains(mode) {
    savedDPC = try JSONDecoder().decode(SavedDPC.self, from: Data(contentsOf: dpcReceipt)).values
  }
  if mode == "cancel" {
    var successfulPolls = 0
    let control = try MetalCompactH5Loader.load(
      source: sources[0], device: device,
      maximumAdditionalBytes: budget,
      shouldCancel: {
        successfulPolls += 1
        return false
      })
    control.releaseResidentStorage()
    precondition(successfulPolls > 4)
    // The final three polls are builder completion, resident construction,
    // and public publication. The preceding poll follows the last GPU wait.
    let finalCommandPoll = successfulPolls - 3
    var polls = 0
    var cancelled = false
    do {
      let unexpected = try MetalCompactH5Loader.load(
        source: sources[0], device: device,
        maximumAdditionalBytes: budget,
        shouldCancel: {
          polls += 1
          return polls == finalCommandPoll
        })
      unexpected.releaseResidentStorage()
    } catch Metal4DSTEMStreamingIOError.cancelled { cancelled = true }
    precondition(
      cancelled && polls == finalCommandPoll,
      "Cancellation after the last aggregate command must drain without publishing")
    print("DETECTOR_REGIONS_CANCEL_DRAIN_RECOVERY polls=\(polls)")
    let changed = sources[0].shards[0].sourceURL
    var stamp = stat()
    precondition(lstat(changed.path, &stamp) == 0)
    var timestamps = [stamp.st_atimespec, stamp.st_mtimespec]
    var mutated = false
    var rejected = false
    polls = 0
    do {
      defer { if mutated { precondition(utimensat(AT_FDCWD, changed.path, &timestamps, 0) == 0) } }
      do {
        let unexpected = try MetalCompactH5Loader.load(
          source: sources[0], device: device,
          maximumAdditionalBytes: budget,
          shouldCancel: {
            polls += 1
            if polls == finalCommandPoll {
              var advanced = timestamps
              advanced[1].tv_sec += 10
              precondition(utimensat(AT_FDCWD, changed.path, &advanced, 0) == 0)
              mutated = true
            }
            return false
          })
        unexpected.releaseResidentStorage()
      } catch Metal4DSTEMStreamingIOError.invalidRequest(let message) {
        rejected = message.contains("Original data changed during resident preparation")
      }
    }
    precondition(
      mutated && rejected, "Source mutation during auxiliary readiness must reject publication")
    print("DETECTOR_REGIONS_LATE_SOURCE_MUTATION_REJECTED")
  }
  var residents: [MetalCompactH5ResidentSource] = []
  defer { for resident in residents { resident.releaseResidentStorage() } }
  for source in sources {
    let resident = try MetalCompactH5Loader.load(
      source: source, device: device,
      maximumAdditionalBytes: budget, preparedDPC: savedDPC,
      packingPlanURL: directBudgetModes.contains(mode) ? plan : nil)
    precondition(resident.loadMetrics.plannedAdditionalBytes <= budget)
    print(
      "DETECTOR_REGIONS_LOAD retained=\(resident.loadMetrics.totalResidentBytes) planned=\(resident.loadMetrics.plannedAdditionalBytes) budget=\(budget)"
    )
    residents.append(resident)
  }
  if mode == "plan-baseline" {
    try JSONEncoder().encode(residents[0].loadMetrics.plannedAdditionalBytes).write(to: receipt)
  }
  let first = residents[0].metadata
  let masks = try Data(contentsOf: folder.appendingPathComponent("masks.bin"))
  precondition(masks.count == oracle[0].detectorSHA256.count * first.detectorPixelCount)
  var snapshotBuffers: [MTLBuffer] = []
  for maskIndex in oracle[0].detectorSHA256.indices {
    let start = maskIndex * first.detectorPixelCount
    let mask = Array(masks[start..<(start + first.detectorPixelCount)])
    let metrics = try MetalCompactH5ResidentSource.updateVirtualDetectors(
      residents, mask: mask, snapshots: &snapshotBuffers)
    precondition(metrics.submissionCount == 1 && metrics.fftDispatchCount == 0)
    for (index, resident) in residents.enumerated() {
      let values = try resident.virtualDetectorValues()
      let actual = values.withUnsafeBytes { digest(Data($0)) }
      precondition(
        actual == oracle[index].detectorSHA256[maskIndex],
        "Exact detector counts differ from independent NumPy")
      let snapshot = Data(
        bytes: snapshotBuffers[index].contents(), count: snapshotBuffers[index].length)
      precondition(digest(snapshot) == actual, "Published snapshot differs from completed counts")
    }
    // Full rebase deliberately bypasses auxiliary/prepared evidence, after the
    // candidate image has already been checked against an independent oracle.
    _ = try MetalCompactH5ResidentSource.updateVirtualDetectors(
      residents, mask: mask, forceRebase: true)
    for (index, resident) in residents.enumerated() {
      let raw = try resident.virtualDetectorValues()
      precondition(
        raw.withUnsafeBytes { digest(Data($0)) } == oracle[index].detectorSHA256[maskIndex])
    }
  }
  for (index, resident) in residents.enumerated() {
    var counts = SHA256()
    for scan in 0..<resident.metadata.scanCount {
      let values = try resident.extractDiffraction(
        scanRow: scan / resident.metadata.scanColumns,
        scanColumn: scan % resident.metadata.scanColumns)
      values.withUnsafeBytes { counts.update(bufferPointer: $0) }
    }
    precondition(
      counts.finalize().map { String(format: "%02x", $0) }.joined() == oracle[index].countsSHA256,
      "Auxiliary preparation changed original uint16 or DP counts")
    let moments = try resident.preparedDPCMomentValues()!
    var dpc = SHA256()
    for scan in 0..<resident.metadata.scanCount {
      let row = [
        moments.total[scan], moments.detectorRowMoment[scan], moments.detectorColumnMoment[scan],
        UInt64(0),
      ]
      row.withUnsafeBytes { dpc.update(bufferPointer: $0) }
    }
    precondition(
      dpc.finalize().map { String(format: "%02x", $0) }.joined() == oracle[index].dpcSHA256)
  }
  // Snapshot ownership survives releasing all source evidence.
  let snapshotsBefore = snapshotBuffers.map { digest(Data(bytes: $0.contents(), count: $0.length)) }
  for resident in residents { resident.releaseResidentStorage() }
  precondition(
    snapshotBuffers.map { digest(Data(bytes: $0.contents(), count: $0.length)) } == snapshotsBefore)
  print("DETECTOR_REGIONS_EXACT_COUNTS_MASKS_DPC_SNAPSHOTS_PASS sources=\(sourceCount)")
} catch {
  fputs("ERROR: \(error)\n", stderr)
  exit(1)
}
