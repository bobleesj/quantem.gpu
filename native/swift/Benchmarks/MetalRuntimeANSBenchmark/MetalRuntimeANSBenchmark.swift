import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

/// Measure native original-HDF5 to in-memory ANS without a frontend cache.
@main
@available(macOS 15.0, *)
enum MetalRuntimeANSBenchmark {
  static func main() throws {
    let arguments = Array(CommandLine.arguments.dropFirst())
    if arguments.count == 2 && arguments[1] == "--qem-load-only" {
      guard let device = MTLCreateSystemDefaultDevice() else {
        throw failure("A physical Metal device is required")
      }
      let ioBefore = try processUsage()
      let started = CFAbsoluteTimeGetCurrent()
      let source = try MetalRuntimeANSResidentSource.load(
        snapshot: NativeANSSnapshot(url: URL(fileURLWithPath: arguments[0])), device: device)
      let elapsed = CFAbsoluteTimeGetCurrent() - started
      let ioAfter = try processUsage()
      defer { source.releaseResidentStorage() }
      let checkStarted = CFAbsoluteTimeGetCurrent()
      let scans = source.shape[0] * source.shape[1]
      let frames = [0, scans / 2, scans - 1]
      let hashes = try frames.map { frame in
        let values = try source.extractRawDiffraction(
          scanRow: frame / source.shape[1], scanColumn: frame % source.shape[1])
        return values.withUnsafeBytes {
          SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
        }
      }
      let output: [String: Any] = [
        "schema": "quantem-gpu-native-qem-load-benchmark/v1",
        "device": device.name, "shape": source.shape,
        "load_seconds": elapsed, "resident_bytes": source.residentBytes,
        "physical_read_bytes": ioAfter.ri_diskio_bytesread - ioBefore.ri_diskio_bytesread,
        "physical_footprint_bytes": ioAfter.ri_phys_footprint,
        "peak_physical_footprint_bytes": ioAfter.ri_lifetime_max_phys_footprint,
        "metal_allocated_bytes": device.currentAllocatedSize,
        "source_dtype": source.dataset.sourceDtype,
        "sample_frames": frames, "sample_hashes_u32_le": hashes,
        "verification_seconds_excluded": CFAbsoluteTimeGetCurrent() - checkStarted,
        "verification_scope": "three raw diffraction frame hashes; not full-volume parity",
      ]
      print(
        String(
          data: try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]),
          encoding: .utf8)!)
      return
    }
    if arguments.count == 2 && arguments[0] == "--camera" {
      try CameraBenchmark.run(URL(fileURLWithPath: arguments[1]))
      return
    }
    let loadOnly = arguments.count == 3 && arguments[2] == "--load-only"
    guard arguments.count == 2 || loadOnly else {
      throw failure("Usage: metal-runtime-ans-benchmark INPUT INDEX_DIRECTORY [--load-only]")
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw failure("A physical Metal device is required")
    }
    let input = URL(fileURLWithPath: arguments[0])
    let indexes = URL(fileURLWithPath: arguments[1])
    let ioBeforePreparation = try processUsage()
    let preparationStarted = CFAbsoluteTimeGetCurrent()
    let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: indexes).prepare(input: input)
    let preparationSeconds = CFAbsoluteTimeGetCurrent() - preparationStarted
    guard !catalog.datasets.isEmpty else { throw failure("No original indexed HDF5 acquisition") }
    var residents: [MetalRuntimeANSResidentSource] = []
    defer {
      for source in residents { source.releaseResidentStorage() }
    }
    var records: [[String: Any]] = []
    let seriesStarted = CFAbsoluteTimeGetCurrent()
    for dataset in catalog.datasets {
      let indexed = try Native4DSTEMIndexedSource.open(dataset: dataset)
      let ioBefore = try processUsage()
      let resident = try MetalRuntimeANSResidentSource.load(
        source: indexed, device: device, includeSpatialIndex: loadOnly,
        maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory)
      let ioAfter = try processUsage()
      residents.append(resident)
      let readySeconds = CFAbsoluteTimeGetCurrent() - seriesStarted
      let checkStarted = CFAbsoluteTimeGetCurrent()
      let scans = indexed.logicalFrameCount
      let frames = [0, scans / 2, scans - 1]
      let hashes = try frames.map { frame in
        let values = try resident.extractRawDiffraction(
          scanRow: frame / dataset.scanCols, scanColumn: frame % dataset.scanCols)
        return values.withUnsafeBytes {
          SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
        }
      }
      records.append([
        "shape": resident.shape,
        "ready_seconds_including_preparation": preparationSeconds + readySeconds,
        "source_dtype": dataset.sourceDtype,
        "source_identity": resident.sourceIdentitySHA256,
        "load_seconds": resident.loadMetrics.totalSeconds,
        "physical_read_bytes": ioAfter.ri_diskio_bytesread - ioBefore.ri_diskio_bytesread,
        "physical_read_bytes_including_preparation": ioAfter.ri_diskio_bytesread
          - ioBeforePreparation.ri_diskio_bytesread,
        "physical_footprint_bytes": ioAfter.ri_phys_footprint,
        "peak_physical_footprint_bytes": ioAfter.ri_lifetime_max_phys_footprint,
        "metal_allocated_bytes": device.currentAllocatedSize,
        "fused_decode_encode_seconds": resident.loadMetrics.fusedDecodeAndEncodeSeconds,
        "prefix_seconds": resident.loadMetrics.prefixSeconds,
        "compact_seconds": resident.loadMetrics.compactSeconds,
        "logical_bytes": resident.loadMetrics.logicalBytes,
        "resident_bytes": resident.loadMetrics.residentBytes,
        "sample_frames": frames,
        "sample_hashes_u32_le": hashes,
        "verification_seconds_excluded": CFAbsoluteTimeGetCurrent() - checkStarted,
      ])
    }
    let seriesLoadSeconds = CFAbsoluteTimeGetCurrent() - seriesStarted
    if loadOnly {
      let output: [String: Any] = [
        "schema": "quantem-gpu-native-load-benchmark/v1",
        "device": device.name,
        "catalog_preparation_seconds": preparationSeconds,
        "acquisitions": records,
        "include_spatial_index": true,
        "verification_scope": "three raw diffraction frame hashes; not full-volume parity",
      ]
      print(
        String(
          data: try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]),
          encoding: .utf8)!)
      return
    }
    let series = try MetalRuntimeANSSeries(sources: residents)
    defer { series.release() }
    var selectedDPMilliseconds: [Double] = []
    var allDPMilliseconds: [Double] = []
    for trial in 0..<60 {
      let row = (trial * 37) % catalog.datasets[0].scanRows
      let column = (trial * 61) % catalog.datasets[0].scanCols
      var started = CFAbsoluteTimeGetCurrent()
      _ = try series.updatePriorityDiffractionBuffer(
        scanRow: row, scanColumn: column, priorityIndex: trial % residents.count)
      selectedDPMilliseconds.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
      started = CFAbsoluteTimeGetCurrent()
      _ = try series.updateDiffractionBuffers(scanRow: row, scanColumn: column)
      allDPMilliseconds.append((CFAbsoluteTimeGetCurrent() - started) * 1000)
    }
    let detectorRows = catalog.datasets[0].detectorRows
    let detectorColumns = catalog.datasets[0].detectorCols
    let smallMasks = [
      circularMask(
        rows: detectorRows, columns: detectorColumns,
        centerRow: detectorRows / 2, centerColumn: detectorColumns / 2, outerRadius: 5),
      circularMask(
        rows: detectorRows, columns: detectorColumns,
        centerRow: detectorRows / 2, centerColumn: detectorColumns / 2 + 1, outerRadius: 5),
    ]
    let wideMasks = [
      circularMask(
        rows: detectorRows, columns: detectorColumns,
        centerRow: detectorRows / 2, centerColumn: detectorColumns / 2,
        innerRadius: 45, outerRadius: 70),
      circularMask(
        rows: detectorRows, columns: detectorColumns,
        centerRow: detectorRows / 2, centerColumn: detectorColumns / 2,
        innerRadius: 45, outerRadius: 71),
    ]
    let jumpMasks = [
      circularMask(
        rows: detectorRows, columns: detectorColumns,
        centerRow: detectorRows / 2, centerColumn: detectorColumns / 2 - 16,
        innerRadius: 45, outerRadius: 70),
      circularMask(
        rows: detectorRows, columns: detectorColumns,
        centerRow: detectorRows / 2, centerColumn: detectorColumns / 2 + 16,
        innerRadius: 45, outerRadius: 70),
    ]
    let small = try detectorTrials(series: series, masks: smallMasks)
    let wide = try detectorTrials(series: series, masks: wideMasks)
    let priorityWide = try priorityDetectorTrials(
      series: series, masks: wideMasks, priorityIndex: residents.count / 2)
    let jumps = try detectorTrials(series: series, masks: jumpMasks)
    let parityBuffers = try series.updateVirtualDetectorBuffers(
      mask: wideMasks[0], forceRebase: true
    ).buffers
    var detectorParity = true
    var detectorParityMismatches: [[String: Any]] = []
    let parityFrames = [
      0, catalog.datasets[0].scanCols * catalog.datasets[0].scanRows / 2,
      catalog.datasets[0].scanCols * catalog.datasets[0].scanRows - 1,
    ]
    for sourceIndex in residents.indices {
      let values = parityBuffers[sourceIndex].contents().bindMemory(
        to: UInt32.self, capacity: catalog.datasets[0].scanRows * catalog.datasets[0].scanCols)
      for frame in parityFrames {
        let dp = try residents[sourceIndex].extractRawDiffraction(
          scanRow: frame / catalog.datasets[0].scanCols,
          scanColumn: frame % catalog.datasets[0].scanCols)
        let valid = residents[sourceIndex].detectorValidityMask
        let expected = dp.indices.reduce(UInt64(0)) { sum, pixel in
          sum + (wideMasks[0][pixel] == 1 && valid[pixel] == 1 ? UInt64(dp[pixel]) : 0)
        }
        let actual = UInt64(values[frame])
        detectorParity = detectorParity && expected == actual
        if expected != actual {
          detectorParityMismatches.append([
            "acquisition": sourceIndex, "frame": frame,
            "expected": expected, "actual": actual,
          ])
        }
      }
    }
    let output: [String: Any] = [
      "schema": "quantem-gpu-runtime-ans-benchmark/v1",
      "acquisitions": records,
      "acquisition_count": records.count,
      "series_load_seconds": seriesLoadSeconds,
      "series_resident_bytes": residents.reduce(0) { $0 + $1.loadMetrics.residentBytes },
      "selected_dp_p95_milliseconds": percentile(selectedDPMilliseconds, 0.95),
      "all_dp_p95_milliseconds": percentile(allDPMilliseconds, 0.95),
      "small_bf_wall_p95_milliseconds": percentile(small.wall, 0.95),
      "small_bf_gpu_p95_milliseconds": percentile(small.gpu, 0.95),
      "small_bf_changed_pixels": small.changed,
      "wide_adf_wall_p95_milliseconds": percentile(wide.wall, 0.95),
      "wide_adf_gpu_p95_milliseconds": percentile(wide.gpu, 0.95),
      "wide_adf_changed_pixels": wide.changed,
      "priority_wide_adf_wall_p95_milliseconds": percentile(priorityWide.wall, 0.95),
      "priority_wide_adf_gpu_p95_milliseconds": percentile(priorityWide.gpu, 0.95),
      "large_jump_wall_p95_milliseconds": percentile(jumps.wall, 0.95),
      "large_jump_gpu_p95_milliseconds": percentile(jumps.gpu, 0.95),
      "large_jump_changed_pixels": jumps.changed,
      "detector_sample_parity": detectorParity,
      "detector_sample_parity_mismatches": detectorParityMismatches,
      "series_product_bytes": series.productBytes,
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

  static func processUsage() throws -> rusage_info_v4 {
    var usage = rusage_info_v4()
    let result = withUnsafeMutablePointer(to: &usage) { pointer in
      pointer.withMemoryRebound(to: rusage_info_t?.self, capacity: 1) {
        proc_pid_rusage(getpid(), RUSAGE_INFO_V4, $0)
      }
    }
    guard result == 0 else { throw failure("Cannot read process IO counters") }
    return usage
  }

  static func failure(_ message: String) -> NSError {
    NSError(domain: message, code: 1)
  }

  static func percentile(_ values: [Double], _ fraction: Double) -> Double {
    let sorted = values.sorted()
    return sorted[min(sorted.count - 1, Int(ceil(Double(sorted.count) * fraction)) - 1)]
  }

  static func circularMask(
    rows: Int, columns: Int, centerRow: Int, centerColumn: Int,
    innerRadius: Int = 0, outerRadius: Int
  ) -> [UInt8] {
    let innerSquared = innerRadius * innerRadius
    let outerSquared = outerRadius * outerRadius
    return (0..<(rows * columns)).map { pixel in
      let row = pixel / columns - centerRow
      let column = pixel % columns - centerColumn
      let squared = row * row + column * column
      return squared >= innerSquared && squared <= outerSquared ? 1 : 0
    }
  }

  static func detectorTrials(
    series: MetalRuntimeANSSeries, masks: [[UInt8]]
  ) throws -> (wall: [Double], gpu: [Double], changed: Int) {
    _ = try series.updateVirtualDetectorBuffers(mask: masks[0], forceRebase: true)
    var wall: [Double] = []
    var gpu: [Double] = []
    var changed = 0
    let trials = Int(ProcessInfo.processInfo.environment["QGPU_RUNTIME_ANS_TRIALS"] ?? "60") ?? 60
    for trial in 0..<trials {
      let result = try series.updateVirtualDetectorBuffers(mask: masks[(trial + 1) % masks.count])
      wall.append(result.metrics.wallMilliseconds)
      gpu.append(result.metrics.gpuMilliseconds)
      changed = max(changed, result.metrics.changedDetectorPixels)
    }
    return (wall, gpu, changed)
  }

  static func priorityDetectorTrials(
    series: MetalRuntimeANSSeries, masks: [[UInt8]], priorityIndex: Int
  ) throws -> (wall: [Double], gpu: [Double]) {
    _ = try series.updatePriorityVirtualDetectorBuffer(
      mask: masks[0], priorityIndex: priorityIndex, forceRebase: true)
    var wall: [Double] = []
    var gpu: [Double] = []
    let trials = Int(ProcessInfo.processInfo.environment["QGPU_RUNTIME_ANS_TRIALS"] ?? "60") ?? 60
    for trial in 0..<trials {
      let result = try series.updatePriorityVirtualDetectorBuffer(
        mask: masks[(trial + 1) % masks.count], priorityIndex: priorityIndex)
      wall.append(result.metrics.wallMilliseconds)
      gpu.append(result.metrics.gpuMilliseconds)
    }
    return (wall, gpu)
  }
}
