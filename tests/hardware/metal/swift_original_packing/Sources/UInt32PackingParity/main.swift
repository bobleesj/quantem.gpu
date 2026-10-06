import Foundation
import Metal
import Native4DSTEMIO
import Metal4DSTEMStreamingIO

do {
  let input = URL(fileURLWithPath: CommandLine.arguments[1])
  let oracleURL = URL(fileURLWithPath: CommandLine.arguments[2])
  let oracle = try Data(contentsOf: oracleURL).withUnsafeBytes { Array($0.bindMemory(to: UInt32.self)) }
  let native = try Native4DSTEMCatalogBuilder(cacheDirectory: oracleURL.deletingLastPathComponent().appendingPathComponent("indexes"))
    .prepare(input: input).datasets[0]
  precondition(native.sourceDtype == "uint32")
  let indexed = try Native4DSTEMIndexedSource.open(dataset: native)
  let device = MTLCreateSystemDefaultDevice()!
  let scans = native.scanRows * native.scanCols, pixels = native.detectorRows * native.detectorCols
  precondition(oracle.count == scans * pixels)
  for cycle in 0..<2 {
    let resident = try MetalCompactH5Loader.load(source: indexed, device: device,
      packingPlanURL: oracleURL.appendingPathExtension("plan"))
    defer { resident.releaseResidentStorage() }
    for scan in 0..<scans {
      let actual = try resident.extractDiffraction(scanRow: scan / native.scanCols, scanColumn: scan % native.scanCols)
      precondition(actual == Array(oracle[(scan * pixels)..<((scan + 1) * pixels)]), "exact diffraction mismatch at \(scan)")
    }
    let average = try resident.meanDiffractionPattern()
    for pixel in 0..<pixels {
      let expected = (0..<scans).reduce(UInt64(0)) { $0 + UInt64(oracle[$1 * pixels + pixel]) }
      precondition(average.detectorSum[pixel] == expected, "exact average sum mismatch")
      precondition(average.mean[pixel] == Float(expected) / Float(scans), "average display mismatch")
    }
    let dpc = try resident.preparedDPCMomentValues()!
    var totalCount: UInt64 = 0
    for scan in 0..<scans {
      var total: UInt64 = 0, row: UInt64 = 0, column: UInt64 = 0
      for pixel in 0..<pixels {
        let value = UInt64(oracle[scan * pixels + pixel])
        total += value; row += value * UInt64(pixel / native.detectorCols)
        column += value * UInt64(pixel % native.detectorCols)
      }
      precondition(dpc.total[scan] == total && dpc.detectorRowMoment[scan] == row
        && dpc.detectorColumnMoment[scan] == column, "exact DPC mismatch")
      totalCount += total
    }
    let summary = try resident.countSummary()
    precondition(summary.totalCounts == totalCount)
    for selection in [0, 1, 2, 3, 0] {
      let mask: [UInt8] = (0..<pixels).map { pixel in
        selection == 0 ? 1 : (selection == 3 ? 0 : (pixel % 3 == selection ? 1 : 0))
      }
      var snapshots: [MTLBuffer] = []
      _ = try MetalCompactH5ResidentSource.updateVirtualDetectors([resident], mask: mask, snapshots: &snapshots)
      let actual = try resident.virtualDetectorValues64()
      let display = snapshots[0].contents().assumingMemoryBound(to: Float.self)
      for scan in 0..<scans {
        var expected: UInt64 = 0
        for pixel in 0..<pixels where mask[pixel] != 0 { expected += UInt64(oracle[scan * pixels + pixel]) }
        precondition(actual[scan] == expected, "exact detector mismatch")
        precondition(display[scan] == Float(expected), "display conversion mismatch")
      }
      _ = try resident.updateVirtualDetector(mask: mask, forceRebase: true)
      let rebased = try resident.virtualDetectorValues64()
      precondition(rebased == actual, "delta/rebase mismatch")
    }
    print("UINT32_EXACT_PASS cycle=\(cycle) counts=\(oracle.count) bytes=\(resident.loadMetrics.totalResidentBytes)")
  }
} catch {
  fputs("UINT32_PARITY_FAILED: \(error)\n", stderr)
  exit(1)
}
