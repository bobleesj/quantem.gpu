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
      source: indexed, device: device, maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory)

    let reference = try MetalRuntimeANSResidentSource.load(
      source: indexed, device: device, maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory)
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
      dpcMomentSampleParity = dpcMomentSampleParity
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
    let sorted = values.sorted()
    return sorted[min(sorted.count - 1, Int(ceil(Double(sorted.count) * fraction)) - 1)]
  }

  static func failure(_ message: String) -> NSError {
    NSError(domain: message, code: 1)
  }
}
