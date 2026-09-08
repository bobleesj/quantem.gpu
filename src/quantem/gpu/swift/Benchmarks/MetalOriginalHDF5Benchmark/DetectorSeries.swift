import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

extension MetalOriginalHDF5Benchmark {
  /// Simultaneously resident exact sources; every full map is externally hashed.
  /// Independent updates deliberately desynchronize histories before a batch.
  static func benchmarkSeries(
    _ datasets: [Native4DSTEMDataset], options: Options, device: MTLDevice
  ) throws {
    var sources: [MetalCompactH5ResidentSource] = []
    defer { for source in sources { source.releaseResidentStorage() } }
    var residentBytes: UInt64 = 0
    for dataset in datasets {
      let identity = dataset.sourceIdentitySHA256!
      let start = CFAbsoluteTimeGetCurrent()
      let source = try autoreleasepool {
        try MetalCompactH5Loader.load(
          source: Native4DSTEMIndexedSource.open(dataset: dataset), device: device,
          maximumAdditionalBytes: options.budget - residentBytes,
          packingPlanURL: options.planDirectory?.appendingPathComponent(identity + ".qgplan"))
      }
      sources.append(source)
      let receipt = try Metal4DSTEMResidentCapabilities.compact(source).residentReceipt
      guard receipt.losslessExact, receipt.scanBin == 1, receipt.detectorBin == 1,
        receipt.crop == nil, receipt.sourceShape == receipt.workingShape,
        receipt.detectorMaskCount == 0
      else { throw failure("Series lost complete source counts") }
      residentBytes += source.loadMetrics.totalResidentBytes
      guard residentBytes < options.budget else { throw failure("Series exceeded its resident budget") }
      try emit(["phase": "series_load", "source_identity": identity,
        "seconds": CFAbsoluteTimeGetCurrent() - start, "resident_count": sources.count,
        "resident_bytes": residentBytes, "device_allocated_bytes": device.currentAllocatedSize])
    }
    guard let first = sources.first else { throw failure("No sources for series test") }
    let metadata = first.metadata
    let shapes: [(String, Double, Double)] = [("BF", 0, 13.5), ("ABF", 16, 64), ("ADF", 32, 90)]
    let offsets: [(Double, Double)] = [(0, 0), (2, 3), (18, -14), (-15, 18), (0, 0)]
    var hashes: [String: String] = [:]
    for trial in 0..<max(2, options.detectorTrials) {
      for (name, inner, outer) in shapes {
        for (step, offset) in offsets.enumerated() {
          try autoreleasepool {
            let mask: [UInt8] = (0..<metadata.detectorPixelCount).map { pixel in
              let row = Double(pixel / metadata.detectorColumns) - Double(metadata.detectorRows - 1) / 2 - offset.0
              let column = Double(pixel % metadata.detectorColumns) - Double(metadata.detectorColumns - 1) / 2 - offset.1
              let r2 = row * row + column * column
              return (inner == 0 || r2 > inner * inner) && r2 <= outer * outer ? 1 : 0
            }
            // The last trial must take the safe heterogeneous-history path.
            // Earlier trials are identical histories and test shared preparation.
            if trial == max(2, options.detectorTrials) - 1, sources.count > 1 {
              var alternate = mask
              alternate[metadata.detectorPixelCount / 2] ^= 1
              _ = try sources[0].updateVirtualDetector(mask: alternate)
            }
            let started = CFAbsoluteTimeGetCurrent()
            var snapshots: [MTLBuffer] = []
            let metrics = try MetalCompactH5ResidentSource.updateVirtualDetectors(
              sources, mask: mask, snapshots: &snapshots)
            let call = (CFAbsoluteTimeGetCurrent() - started) * 1000
            guard snapshots.count == sources.count else { throw failure("Missing series snapshot") }
            for (source, snapshot) in zip(sources, snapshots) {
              let values = Array(UnsafeBufferPointer(
                start: snapshot.contents().assumingMemoryBound(to: UInt32.self), count: source.metadata.scanCount))
              let hash = digest(values)
              let key = "\(source.metadata.sourceIdentitySHA256):\(name):\(step)"
              guard hashes[key] == nil || hashes[key] == hash else { throw failure("Series repeat or heterogeneous fallback differs") }
              hashes[key] = hash
              try emit(["phase": "series_detector", "source_identity": source.metadata.sourceIdentitySHA256,
                "trial": trial, "case": name, "step": step, "sha256_u32_le": hash,
                "gpu_ms": metrics.gpuMilliseconds, "call_ms": call,
                "wall_after_prepare_ms": metrics.wallMilliseconds,
                "source_count": sources.count, "submission_count": metrics.submissionCount,
                "heterogeneous_history": trial == max(2, options.detectorTrials) - 1,
                "ui_present_measured": false])
            }
          }
        }
      }
    }
    for fill: UInt8 in [0, 1, 1, 0] {
      try autoreleasepool {
      let mask = [UInt8](repeating: fill, count: metadata.detectorPixelCount)
      var snapshots: [MTLBuffer] = []
      _ = try MetalCompactH5ResidentSource.updateVirtualDetectors(sources, mask: mask, snapshots: &snapshots)
      for (source, snapshot) in zip(sources, snapshots) {
        guard let dpc = try source.preparedDPCMomentValues() else { throw failure("Missing exact totals") }
        let values = UnsafeBufferPointer(start: snapshot.contents().assumingMemoryBound(to: UInt32.self), count: source.metadata.scanCount)
        guard values.indices.allSatisfy({ UInt64(values[$0]) == (fill == 0 ? 0 : dpc.total[$0]) })
        else { throw failure("Empty/full/unchanged mask differs from exact totals") }
      }
      }
    }
    try emit(["phase": "series_boundary_parity", "empty_full_unchanged_masks": true,
      "sources": sources.count, "full_scan_counts_preserved": true])
    for source in sources { source.releaseResidentStorage() }
    try emit(["phase": "complete", "series": true, "resident_bytes": residentBytes,
      "released_device_allocated_bytes": device.currentAllocatedSize])
  }
}
