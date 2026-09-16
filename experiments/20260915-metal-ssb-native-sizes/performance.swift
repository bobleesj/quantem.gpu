import Foundation
import Metal
import MetalSSBKernels

// Native-resolution deterministic counts and a full 8,937-term disk. This
// measures kernel scaling, not acquisition I/O or an end-to-end app workflow.
@main enum NativeSizePerformance {
  static func main() throws {
    let args = CommandLine.arguments
    guard args.count == 2, let device = MTLCreateSystemDefaultDevice() else {
      fatalError("Usage: ssb-native-size-performance <report.json>")
    }
    var reports: [[String: Any]] = []
    for n in [128, 256, 512] {
      let record = try autoreleasepool { () throws -> [String: Any] in
        var calibration = MetalSSBCalibration(beamEnergyKeV: 300, semiangleMrad: 30,
          scanStepRowAngstroms: 0.264, scanStepColumnAngstroms: 0.264,
          detectorStepRowMrad: 1, detectorStepColumnMrad: 1,
          centerRow: 94.88451385498047, centerColumn: 96.35952758789062,
          brightfieldRadiusPixels: 53.35992814757164, excludedDetectorPixels: [78 * 192 + 74])
        try calibration.matchApertureToBrightfieldDisk()
        let setup = try calibration.geometry(detectorRows: 192, detectorColumns: 192,
          detectorSum: Array(repeating: UInt64(100 * n * n), count: 192 * 192),
          scanRows: n, scanColumns: n)
        precondition(setup.pixels.count == 8937)
        let engine = try MetalSSBEngine(device: device, geometry: setup.geometry)
        let source = device.makeBuffer(length: 8937 * n * n, options: .storageModeShared)!
        let bytes = source.contents().assumingMemoryBound(to: UInt8.self)
        for index in 0..<source.length {
          bytes[index] = UInt8((index * 17 + index / 113 + 3) % 251)
        }
        let start = Date()
        try engine.prepare(brightfield: source)
        let prepare = -start.timeIntervalSinceNow
        var redraw: [Double] = [], objective: [Double] = []
        var redrawGPU: [Double] = [], objectiveGPU: [Double] = []
        var cacheBytes = 0
        for iteration in 0..<23 {
          let result = try engine.reconstruct(aberrations: .init(
            c10Nanometers: 55 + Float(iteration % 5), c12Nanometers: 13, phi12Radians: 0.23))
          cacheBytes = result.provenance.cacheBytes
          if iteration >= 3 { redraw.append(result.wallSeconds); redrawGPU.append(result.gpuSeconds) }
        }
        for iteration in 0..<23 {
          let result = try engine.phaseVariance(aberrations: .init(
            c10Nanometers: 55 + Float(iteration % 5), c12Nanometers: 13, phi12Radians: 0.23))
          if iteration >= 3 { objective.append(result.wallSeconds); objectiveGPU.append(result.gpuSeconds) }
        }
        func metrics(_ values: [Double]) -> [String: Double] {
          let ordered = values.sorted().map { $0 * 1000 }
          return ["mean_ms": ordered.reduce(0, +) / Double(ordered.count),
            "p50_ms": ordered[ordered.count / 2], "p95_ms": ordered[ordered.count - 1]]
        }
        let record: [String: Any] = ["scan_size": n, "selected_bf": 8937,
          "active_bf": engine.executedBrightfieldCount, "prepare_seconds": prepare,
          "object_wall": metrics(redraw), "objective_wall": metrics(objective),
          "object_gpu": metrics(redrawGPU), "objective_gpu": metrics(objectiveGPU),
          "cache_bytes": cacheBytes, "sampled_metal_allocated_bytes": device.currentAllocatedSize]
        print(record)
        return record
      }
      reports.append(record)
    }
    let report: [String: Any] = ["device": device.name,
      "data": "synthetic uint8 native scans; real full-aperture calibration, not real acquisition counts",
      "warmups": 3, "repeats": 20, "bin": 1, "crop": "none", "results": reports]
    try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
      .write(to: URL(fileURLWithPath: args[1]), options: .atomic)
  }
}
