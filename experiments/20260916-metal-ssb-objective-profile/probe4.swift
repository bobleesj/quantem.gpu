import Foundation
import Metal
import MetalSSBKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// RECORD OF A REJECTED ARM. This probe needs `MetalSSBEngine.phaseVarianceBatch`
// and `maximumObjectiveBatchSize`, which were written for the multi-candidate
// experiment and reverted after the arm measured slower than the single-candidate
// path (0.90x at the default per-candidate batch, 0.78x at 4 planes, 0.55x at 2).
// Keep the source as the arm's record; re-apply the reverted API to rebuild it.

// Batched-objective probe: identical input, BF selection and controls to
// probe.swift. Groups of candidate aberrations are evaluated once through the
// single-candidate path and once through the batched path, then compared bit
// for bit and timed.
let args = CommandLine.arguments
let output = URL(fileURLWithPath: args[2], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let device = MTLCreateSystemDefaultDevice()!
let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: output.appendingPathComponent("index"))
  .prepare(input: URL(fileURLWithPath: args[1]))
let indexed = try Native4DSTEMIndexedSource.open(dataset: catalog.datasets[0])
let source = try MetalCompactH5Loader.load(source: indexed, device: device)
let mean = try source.meanDiffractionPattern()
var calibration = MetalSSBCalibration(beamEnergyKeV: 300, semiangleMrad: 30,
  scanStepRowAngstroms: 0.264, scanStepColumnAngstroms: 0.264,
  detectorStepRowMrad: 1, detectorStepColumnMrad: 1,
  centerRow: 94.88451385498047, centerColumn: 96.35952758789062,
  brightfieldRadiusPixels: 53.35992814757164, excludedDetectorPixels: [78 * 192 + 74])
try calibration.matchApertureToBrightfieldDisk()
let setup = try calibration.geometry(detectorRows: 192, detectorColumns: 192, detectorSum: mean.detectorSum)
precondition(setup.pixels.count == 8937)
let engine = try MetalSSBEngine(device: device, geometry: setup.geometry, cacheBudgetBytes: nil)
try engine.prepare(countType: .uint32) { indices, destination, commands in
  try source.encodeDetectorColumns(pixels: indices.map { setup.pixels[$0] }, into: destination, commands: commands)
}
let rotation: Float = 158.88268568029937
func point(_ c10: Float, _ c12: Float, _ phi: Float) -> MetalSSBAberrations {
  MetalSSBAberrations(c10Nanometers: c10, c12Nanometers: c12, phi12Radians: phi)
}
// The control triple used by the objective probe plus nearby aberrations, so
// every group spans the range the optimizer actually visits.
let base: [(Float, Float, Float)] = [
  (0, 42.96961, 0.293384),
  (55, 42.96961, 0.293384),
  (155.96977, 42.96961, 0.293384),
  (12.5, 50, 0.293384),
  (0, 42.96961, 0.0),
  (55, 35, 0.5),
  (155.96977, 50, 0.1),
  (27.5, 46.5, 0.6),
]
var groups: [[(Float, Float, Float)]] = []
for start in stride(from: 0, to: base.count, by: 4) {
  groups.append(Array(base[start..<min(start + 4, base.count)]))
}
print("batch capacity \(engine.maximumObjectiveBatchSize)")
var records: [[String: Any]] = []
var maxAbs = 0.0
var maxRel = 0.0
var bitMismatch = 0
var peakAllocated = device.currentAllocatedSize
func noteAllocation() {
  peakAllocated = Swift.max(peakAllocated, device.currentAllocatedSize)
}
let rounds = Int(ProcessInfo.processInfo.environment["SSB_PROBE4_ROUNDS"] ?? "3") ?? 3
for round in 0..<rounds {
  for (index, group) in groups.enumerated() {
    // Balance the arm order across (round, group) parity so neither arm owns a
    // fixed position in every comparison.
    let batchedFirst = (round + index) % 2 == 1
    noteAllocation()
    var sequential: [Float] = []
    var batched: [MetalSSBPhaseVarianceResult] = []
    var sequentialMs = 0.0
    var batchedMs = 0.0
    if batchedFirst {
      let batchedStart = Date()
      batched = try engine.phaseVarianceBatch(
        aberrations: group.map { point($0.0, $0.1, $0.2) },
        rotationDegrees: rotation
      )
      batchedMs = Date().timeIntervalSince(batchedStart) * 1000
    }
    noteAllocation()
    let sequentialStart = Date()
    for candidate in group {
      sequential.append(
        try engine.phaseVariance(
          aberrations: point(candidate.0, candidate.1, candidate.2),
          rotationDegrees: rotation
        ).loss)
    }
    noteAllocation()
    sequentialMs = Date().timeIntervalSince(sequentialStart) * 1000
    if !batchedFirst {
      let batchedStart = Date()
      batched = try engine.phaseVarianceBatch(
        aberrations: group.map { point($0.0, $0.1, $0.2) },
        rotationDegrees: rotation
      )
      batchedMs = Date().timeIntervalSince(batchedStart) * 1000
    }
    noteAllocation()
    var deviations: [Double] = []
    for (single, many) in zip(sequential, batched.map(\.loss)) {
      let delta = Double(many) - Double(single)
      let rel = delta / max(Swift.abs(Double(single)), 1e-30)
      maxAbs = Swift.max(maxAbs, Swift.abs(delta))
      maxRel = Swift.max(maxRel, Swift.abs(rel))
      if single.bitPattern != many.bitPattern { bitMismatch += 1 }
      deviations.append(delta)
    }
    records.append([
      "round": round, "group": index, "count": group.count,
      "batched_first": batchedFirst,
      "sequential_ms": sequentialMs, "batched_ms": batchedMs,
      "per_candidate_sequential_ms": sequentialMs / Double(group.count),
      "deviations": deviations,
      "batched_gpu_ms": batched.map(\.gpuSeconds).max()! * 1000,
    ])
    print(String(
      format: "round %d group %d  n=%d  sequential %7.1f ms  batched %7.1f ms  speedup %.3fx  maxabs %.3e",
      round, index, group.count, sequentialMs, batchedMs,
      sequentialMs / batchedMs, deviations.map { $0.magnitude }.max() ?? 0))
  }
}
let report: [String: Any] = [
  "device": device.name, "bf": 8937, "rounds": rounds,
  "multi_batch_planes": ProcessInfo.processInfo.environment["SSB_MULTI_BATCH_PLANES"] ?? "default",
  "peak_allocated_bytes": peakAllocated,
  "load_average": (try? String(contentsOfFile: "/proc/loadavg", encoding: .utf8)) ?? "n/a",
  "maximum_objective_batch_size": engine.maximumObjectiveBatchSize,
  "max_abs_deviation": maxAbs, "max_rel_deviation": maxRel,
  "bit_mismatches": bitMismatch, "records": records,
  "allocated_bytes": device.currentAllocatedSize,
]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent("report.json"))
print("Completed: \(output.path)")
