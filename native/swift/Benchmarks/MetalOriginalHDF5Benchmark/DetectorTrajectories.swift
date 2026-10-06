import Foundation
import Metal4DSTEMStreamingIO

extension MetalOriginalHDF5Benchmark {
  /// Measure full-scan mask sums after loading, separately from presentation.
  /// Each repeat follows the same masks; exact output hashes must remain stable.
  static func benchmarkDetectors(_ source: MetalCompactH5ResidentSource, trials: Int, cycle: Int)
    throws
  {
    let metadata = source.metadata
    let centerRow = Double(metadata.detectorRows - 1) / 2
    let centerColumn = Double(metadata.detectorColumns - 1) / 2
    let shapes: [(String, Double, Double)] = [
      ("BF", 0, 13.5), ("ABF", 16, 64), ("ADF", 32, 90),
    ]
    let offsets: [(Double, Double)] = [(0, 0), (2, 3), (18, -14), (-15, 18), (0, 0)]
    var hashes: [String: String] = [:]
    // Sample raw diffraction independently of the detector reduction. This is
    // a bounded cross-operation check, not an independent HDF5 decoder oracle.
    let scans = [0, metadata.scanCount / 2, metadata.scanCount - 1]
    let diffraction = try scans.map {
      try source.extractDiffraction(
        scanRow: $0 / metadata.scanColumns, scanColumn: $0 % metadata.scanColumns)
    }
    for trial in 0..<trials {
      for (name, inner, outer) in shapes {
        for (step, offset) in offsets.enumerated() {
          let row = centerRow + offset.0
          let column = centerColumn + offset.1
          let mask: [UInt8] = (0..<metadata.detectorPixelCount).map { pixel in
            let dr = Double(pixel / metadata.detectorColumns) - row
            let dc = Double(pixel % metadata.detectorColumns) - column
            let radiusSquared = dr * dr + dc * dc
            return (inner == 0 || radiusSquared > inner * inner)
              && radiusSquared <= outer * outer ? 1 : 0
          }
          let started = CFAbsoluteTimeGetCurrent()
          let metrics = try source.updateVirtualDetector(mask: mask)
          let callMilliseconds = (CFAbsoluteTimeGetCurrent() - started) * 1000
          let actual = try source.virtualDetectorValues()
          let hash = digest(actual)
          let key = "\(name)-\(step)"
          if let previous = hashes[key], previous != hash {
            throw failure("Detector return visit changed exact values for \(key)")
          }
          hashes[key] = hash
          for (index, scan) in scans.enumerated() {
            let expected = mask.indices.reduce(UInt64(0)) {
              $0 + (mask[$1] == 0 ? 0 : UInt64(diffraction[index][$1]))
            }
            guard UInt64(actual[scan]) == expected else {
              throw failure("Detector sum differs from selected diffraction for \(key)")
            }
          }
          try emit([
            "phase": "detector", "source_identity": metadata.sourceIdentitySHA256,
            "cycle": cycle, "trial": trial, "case": name, "step": step,
            "center_row": row, "center_column": column,
            "inner_radius": inner, "outer_radius": outer,
            "mode": metrics.mode, "changed_pixels": metrics.changedDetectorPixels,
            "gpu_ms": metrics.gpuMilliseconds, "wall_ms": metrics.wallMilliseconds,
            "call_ms_including_mask_preparation": callMilliseconds,
            "sha256_u32_le": hash, "selected_dp_sum_parity": true,
            "repeat_parity": true, "independent_hdf5_parity": false,
            "ui_present_measured": false,
          ])
        }
      }
    }
  }
}
