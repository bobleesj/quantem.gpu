import CryptoKit
import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMStreamingIO
import Native4DSTEMIO

@main
@available(macOS 15.0, *)
enum MetalPairedRuntimeTANSBenchmark {
  static func main() throws {
    let arguments = Array(CommandLine.arguments.dropFirst())
    guard arguments.count == 2 else {
      throw failure("Usage: metal-paired-runtime-tans-benchmark INPUT INDEX_DIRECTORY")
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw failure("A physical Metal device is required")
    }
    let catalog = try Native4DSTEMCatalogBuilder(
      cacheDirectory: URL(fileURLWithPath: arguments[1])
    ).prepare(input: URL(fileURLWithPath: arguments[0]))
    guard let dataset = catalog.datasets.first else {
      throw failure("No indexed HDF5 acquisition was found")
    }
    let indexed = try Native4DSTEMIndexedSource.open(dataset: dataset)
    let resident = try MetalPairedRuntimeTANSResidentSource.load(
      source: indexed, device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory)

    let reference = try MetalRuntimeANSResidentSource.load(
      source: indexed, device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory)
    defer { reference.releaseResidentStorage() }
    let scanCount = dataset.scanRows * dataset.scanCols
    let sampleFrames = [0, 1, dataset.scanCols - 1, scanCount / 2, scanCount - 2, scanCount - 1]
    var sampleParity = true
    var dpcMomentSampleParity = true
    var sampleHashes: [String] = []
    for frame in sampleFrames {
      let row = frame / dataset.scanCols
      let column = frame % dataset.scanCols
      let actual = try resident.extractRawDiffraction(scanRow: row, scanColumn: column)
      let expected = try reference.extractRawDiffraction(scanRow: row, scanColumn: column)
      sampleParity = sampleParity && actual == expected
      var total: UInt64 = 0
      var rowMoment: UInt64 = 0
      var columnMoment: UInt64 = 0
      for (pixel, value) in actual.enumerated() {
        let count = UInt64(value)
        total += count
        rowMoment += count * UInt64(pixel / dataset.detectorCols)
        columnMoment += count * UInt64(pixel % dataset.detectorCols)
      }
      dpcMomentSampleParity =
        dpcMomentSampleParity
        && resident.dpcMoments.total[frame] == total
        && resident.dpcMoments.detectorRowMoment[frame] == rowMoment
        && resident.dpcMoments.detectorColumnMoment[frame] == columnMoment
      sampleHashes.append(hash(actual))
    }

    var dpMilliseconds: [Double] = []
    for trial in 0..<60 {
      let row = (trial * 37) % dataset.scanRows
      let column = (trial * 61) % dataset.scanCols
      let started = CFAbsoluteTimeGetCurrent()
      _ = try resident.extractRawDiffraction(scanRow: row, scanColumn: column)
      dpMilliseconds.append((CFAbsoluteTimeGetCurrent() - started) * 1_000)
    }

    let smallMasks = [
      circularMask(dataset: dataset, innerRadius: 0, outerRadius: 5),
      circularMask(dataset: dataset, centerColumnOffset: 1, innerRadius: 0, outerRadius: 5),
    ]
    let wideMasks = [
      circularMask(dataset: dataset, innerRadius: 45, outerRadius: 70),
      circularMask(dataset: dataset, innerRadius: 45, outerRadius: 71),
    ]
    let small = try detectorTrials(resident: resident, masks: smallMasks)
    let wide = try detectorTrials(resident: resident, masks: wideMasks)

    // The polar-index phase is opt-in so the shipped schema and its runtime stay
    // unchanged, and so the two-arm lock experiment can be reproduced alone.
    if ProcessInfo.processInfo.environment["QG_BENCH_POLAR_INDEX"] == "1" {
      try polarIndexPhase(indexed: indexed, dataset: dataset, device: device)
    }

    let values = try resident.updateVirtualDetector(mask: wideMasks[0]).values
    var detectorSampleParity = true
    for frame in sampleFrames {
      let dp = try resident.extractRawDiffraction(
        scanRow: frame / dataset.scanCols, scanColumn: frame % dataset.scanCols)
      let expected = dp.indices.reduce(UInt64(0)) {
        $0 + (wideMasks[0][$1] == 1 ? UInt64(dp[$1]) : 0)
      }
      detectorSampleParity = detectorSampleParity && UInt64(values[frame]) == expected
    }

    let output: [String: Any] = [
      "schema": "quantem-gpu-paired-runtime-tans-benchmark/v1",
      "shape": resident.shape,
      "logical_dtype": resident.logicalDtype.rawValue,
      "source_identity": resident.sourceIdentitySHA256,
      "load_seconds": resident.loadMetrics.totalSeconds,
      "fused_decode_encode_seconds": resident.loadMetrics.fusedDecodeAndSizeSeconds,
      "provisional_cpu_prefix_seconds": resident.loadMetrics.provisionalCPUPrefixSeconds,
      "compact_seconds": resident.loadMetrics.compactSeconds,
      "resident_bytes": resident.residentBytes,
      "sample_frames": sampleFrames,
      "sample_hashes_u32_le": sampleHashes,
      "sample_dp_parity": sampleParity,
      "dpc_moment_sample_parity": dpcMomentSampleParity,
      "detector_sample_parity": detectorSampleParity,
      "selected_dp_p50_milliseconds": percentile(dpMilliseconds, 0.50),
      "selected_dp_p95_milliseconds": percentile(dpMilliseconds, 0.95),
      "small_bf_wall_p95_milliseconds": percentile(small.wall, 0.95),
      "small_bf_gpu_p95_milliseconds": percentile(small.gpu, 0.95),
      "wide_adf_wall_p95_milliseconds": percentile(wide.wall, 0.95),
      "wide_adf_gpu_p95_milliseconds": percentile(wide.gpu, 0.95),
      "crop": NSNull(),
      "scan_bin": 1,
      "detector_bin": 1,
      "saved_ans_file": false,
    ]
    print(
      String(
        data: try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]),
        encoding: .utf8)!)
  }


  /// Thread-safe slot for the probe's background error.
  final class ProbeBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: Error?
    func store(_ error: Error?) {
      lock.lock()
      stored = error
      lock.unlock()
    }
    var error: Error? {
      lock.lock()
      defer { lock.unlock() }
      return stored
    }
  }

  /// Thread-safe slot for the scheduled build's outcome.
  final class OutcomeBox: @unchecked Sendable {
    private let lock = NSLock()
    private var stored: ResidentDetectorIndexPreparationOutcome?
    func store(_ outcome: ResidentDetectorIndexPreparationOutcome) {
      lock.lock()
      stored = outcome
      lock.unlock()
    }
    var outcome: ResidentDetectorIndexPreparationOutcome? {
      lock.lock()
      defer { lock.unlock() }
      return stored
    }
  }

  struct ProbeOutcome {
    let samples: [Double]
    let seconds: Double
    let backgroundError: Error?
    var p50: Double { percentile(samples, 0.50) }
    var p95: Double { percentile(samples, 0.95) }
    var max: Double { percentile(samples, 1.0) }
  }

  /// Run `body` on another thread and sample the cheapest `stateLock` query from
  /// this thread. `extractRawDiffraction` takes exactly the lock the interaction
  /// paths take, so a build that holds that lock for its whole duration shows up
  /// here as one long sample. The 2 ms gap keeps the probe from starving the
  /// background body, which would corrupt the very timing being measured.
  static func interactionProbe(
    resident: MetalPairedRuntimeTANSResidentSource,
    scanRows: Int,
    scanColumns: Int,
    body: @escaping @Sendable () throws -> Void
  ) -> ProbeOutcome {
    let box = ProbeBox()
    let finished = DispatchSemaphore(value: 0)
    DispatchQueue.global(qos: .userInitiated).async {
      do { try body() } catch { box.store(error) }
      finished.signal()
    }
    var samples: [Double] = []
    let started = CFAbsoluteTimeGetCurrent()
    var step = 0
    while finished.wait(timeout: .now()) == .timedOut {
      let row = (step * 37) % scanRows
      let column = (step * 61) % scanColumns
      step += 1
      let began = CFAbsoluteTimeGetCurrent()
      guard (try? resident.extractRawDiffraction(scanRow: row, scanColumn: column)) != nil else {
        break
      }
      samples.append((CFAbsoluteTimeGetCurrent() - began) * 1_000)
      usleep(2_000)
    }
    return ProbeOutcome(
      samples: samples, seconds: CFAbsoluteTimeGetCurrent() - started,
      backgroundError: box.error)
  }

  /// Large detector deltas modelled on the reference trajectories: a moving
  /// centre, then a resizing annulus. These are the moves the region index is
  /// meant to accelerate.
  static func largeMasks(dataset: Native4DSTEMDataset) -> [[UInt8]] {
    let centerRow = dataset.detectorRows / 2
    let centerColumn = dataset.detectorCols / 2
    var masks: [[UInt8]] = []
    for step in 0..<16 {
      let phase = Double(step) / 15 * 2 * Double.pi
      let offset = Int((12 * sin(phase)).rounded())
      let growth = Int((10 * cos(phase)).rounded())
      masks.append(
        annulus(
          dataset: dataset, centerRow: centerRow + offset, centerColumn: centerColumn,
          innerRadius: 40, outerRadius: 84))
      masks.append(
        annulus(
          dataset: dataset, centerRow: centerRow, centerColumn: centerColumn,
          innerRadius: 40 + growth, outerRadius: 84 - growth))
    }
    return masks.map { maskExcludingMarkedPixels($0, markedPixels: dataset.badPixelIndices) }
  }

  /// The indexed query is only exact while both masks keep invalid detector
  /// pixels at zero, the same policy the aperture formula has to follow.
  static func maskExcludingMarkedPixels(_ raw: [UInt8], markedPixels: [Int]) -> [UInt8] {
    guard !markedPixels.isEmpty else { return raw }
    var result = raw
    for pixel in markedPixels where result.indices.contains(pixel) { result[pixel] = 0 }
    return result
  }

  static func annulus(
    dataset: Native4DSTEMDataset, centerRow: Int, centerColumn: Int,
    innerRadius: Int, outerRadius: Int
  ) -> [UInt8] {
    (0..<(dataset.detectorRows * dataset.detectorCols)).map { pixel in
      let row = pixel / dataset.detectorCols - centerRow
      let column = pixel % dataset.detectorCols - centerColumn
      let squared = row * row + column * column
      return squared >= innerRadius * innerRadius && squared <= outerRadius * outerRadius ? 1 : 0
    }
  }

  /// Lock-scope and interaction experiment for the polar region index, on one
  /// Normal resident.
  ///
  /// The control arm runs `makePackedResident`, which takes `stateLock` and holds
  /// it across its whole conversion. It exists to prove the probe really does
  /// detect a held state lock, so the treatment arm's "no stall" is a
  /// measurement rather than an absence of one.
  static func polarIndexPhase(
    indexed: Native4DSTEMIndexedSource,
    dataset: Native4DSTEMDataset,
    device: MTLDevice
  ) throws {
    let budget = UInt64(6) << 30
    let masks = largeMasks(dataset: dataset)
    let zero = [UInt8](repeating: 0, count: dataset.detectorRows * dataset.detectorCols)

    func sweep(_ source: MetalPairedRuntimeTANSResidentSource) throws
      -> (wall: [Double], values: [[UInt32]])
    {
      _ = try source.updateVirtualDetector(mask: zero)
      var wall: [Double] = []
      var values: [[UInt32]] = []
      for mask in masks {
        let result = try source.updateVirtualDetector(mask: mask)
        wall.append(result.wallMilliseconds)
        values.append(result.values)
      }
      return (wall, values)
    }

    let resident = try MetalPairedRuntimeTANSResidentSource.load(
      source: indexed, device: device, maximumAdditionalBytes: budget,
      interaction: .normal)
    defer { resident.releaseResidentStorage() }
    let identity = resident.sourceIdentitySHA256

    let unindexed = try sweep(resident)

    // Control arm: an operation that genuinely holds `stateLock` throughout.
    let controlProbe = interactionProbe(
      resident: resident, scanRows: dataset.scanRows, scanColumns: dataset.scanCols
    ) {
      _ = try resident.makePackedResident(maximumAdditionalBytes: budget)
    }

    // Treatment arm: the deferred index build, which must hold nothing.
    let outcomeBox = OutcomeBox()
    let completed = DispatchSemaphore(value: 0)
    let treatmentProbe = interactionProbe(
      resident: resident, scanRows: dataset.scanRows, scanColumns: dataset.scanCols
    ) {
      resident.scheduleResidentDetectorIndexPreparation(
        maximumAdditionalBytes: budget,
        completion: { outcome in
          outcomeBox.store(outcome)
          completed.signal()
        })
      completed.wait()
    }
    let installed = resident.residentDetectorIndexPrepared
    let indexedValues = try sweep(resident)
    let parity = indexedValues.values == unindexed.values
    let polarFields = resident.lastPolarFieldCount
    let polarResiduals = resident.lastPolarResidualCount
    let scheduled: (addedBytes: UInt64, seconds: Double)
    switch outcomeBox.outcome {
    case .installed(let bytes, let seconds): scheduled = (bytes, seconds)
    default: scheduled = (0, 0)
    }

    let output: [String: Any] = [
      "schema": "quantem-gpu-paired-runtime-polar-index-benchmark/v1",
      "source_identity": identity,
      "trajectory_samples": masks.count,
      "unindexed_wall_p50_milliseconds": percentile(unindexed.wall, 0.50),
      "unindexed_wall_p95_milliseconds": percentile(unindexed.wall, 0.95),
      "unindexed_wall_max_milliseconds": percentile(unindexed.wall, 1.0),
      "indexed_wall_p50_milliseconds": percentile(indexedValues.wall, 0.50),
      "indexed_wall_p95_milliseconds": percentile(indexedValues.wall, 0.95),
      "indexed_wall_max_milliseconds": percentile(indexedValues.wall, 1.0),
      "index_installed": installed,
      "index_parity": parity,
      "last_polar_field_count": polarFields,
      "last_polar_residual_count": polarResiduals,
      "masked_detector_pixels": dataset.badPixelIndices.count,
      "scheduled_outcome": String(describing: outcomeBox.outcome),
      "scheduled_added_bytes": scheduled.addedBytes,
      "scheduled_build_seconds": scheduled.seconds,
      "control_probe_samples": controlProbe.samples.count,
      "control_probe_seconds": controlProbe.seconds,
      "control_probe_p50_milliseconds": controlProbe.p50,
      "control_probe_p95_milliseconds": controlProbe.p95,
      "control_probe_max_milliseconds": controlProbe.max,
      "treatment_probe_samples": treatmentProbe.samples.count,
      "treatment_probe_seconds": treatmentProbe.seconds,
      "treatment_probe_p50_milliseconds": treatmentProbe.p50,
      "treatment_probe_p95_milliseconds": treatmentProbe.p95,
      "treatment_probe_max_milliseconds": treatmentProbe.max,
      "control_probe_background_error": String(describing: controlProbe.backgroundError),
      "treatment_probe_background_error": String(describing: treatmentProbe.backgroundError),
    ]
    print(
      String(
        data: try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]),
        encoding: .utf8)!)
  }

  static func detectorTrials(
    resident: MetalPairedRuntimeTANSResidentSource, masks: [[UInt8]]
  ) throws -> (wall: [Double], gpu: [Double]) {
    _ = try resident.updateVirtualDetector(mask: masks[0])
    var wall: [Double] = []
    var gpu: [Double] = []
    for trial in 0..<12 {
      let result = try resident.updateVirtualDetector(mask: masks[(trial + 1) % masks.count])
      wall.append(result.wallMilliseconds)
      gpu.append(result.gpuMilliseconds)
    }
    return (wall, gpu)
  }

  static func circularMask(
    dataset: Native4DSTEMDataset, centerColumnOffset: Int = 0,
    innerRadius: Int, outerRadius: Int
  ) -> [UInt8] {
    let centerRow = dataset.detectorRows / 2
    let centerColumn = dataset.detectorCols / 2 + centerColumnOffset
    return (0..<(dataset.detectorRows * dataset.detectorCols)).map { pixel in
      let row = pixel / dataset.detectorCols - centerRow
      let column = pixel % dataset.detectorCols - centerColumn
      let squared = row * row + column * column
      return squared >= innerRadius * innerRadius && squared <= outerRadius * outerRadius ? 1 : 0
    }
  }

  static func hash(_ values: [UInt32]) -> String {
    values.withUnsafeBytes {
      SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
    }
  }

  static func percentile(_ values: [Double], _ fraction: Double) -> Double {
    guard !values.isEmpty else { return .nan }
    let sorted = values.sorted()
    return sorted[min(sorted.count - 1, Int(ceil(Double(sorted.count) * fraction)) - 1)]
  }

  static func failure(_ message: String) -> NSError {
    NSError(domain: message, code: 1)
  }
}
