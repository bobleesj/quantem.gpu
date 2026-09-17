import Foundation
import Metal
import MetalSSBKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Fit A/B: identical input, controls, seed and start point to probe.swift.
// Runs the 200-trial TPE search plus Nelder-Mead refinement, records a
// timestamped stage split, and (optionally) re-evaluates every trial point
// through the single-candidate path to bound the batched objective's drift.
let args = CommandLine.arguments
let output = URL(fileURLWithPath: args[2], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let environment = ProcessInfo.processInfo.environment
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
precondition(engine.executedBrightfieldCount == 8937)
let rotation: Float = 158.88268568029937

func point(_ p: SSBOptimizationPoint) -> MetalSSBAberrations {
  MetalSSBAberrations(
    c10Nanometers: Float(p.c10Nanometers),
    c12Nanometers: Float(p.c12Nanometers),
    phi12Radians: Float(p.phi12Radians))
}

// Stage split: the TPE stage ends when the last "tpe" progress event fires.
var firstStamp: Date?
var trialStageEnd: Date?
var lastStage = ""
let progress: (SSBOptimizationProgress) -> Void = { update in
  if firstStamp == nil { firstStamp = Date() }
  if update.stage != lastStage {
    lastStage = update.stage
  }
  if update.stage == "tpe" && update.completed == update.total {
    trialStageEnd = Date()
  }
}

let fitStart = Date()
let fit = try engine.optimize(
  start: MetalSSBAberrations(c10Nanometers: 0, c12Nanometers: 50, phi12Radians: 0),
  rotationDegrees: rotation, globalTrials: 200, progress: progress)
let fitEnd = Date()

let trialStageSeconds = (trialStageEnd ?? fitEnd).timeIntervalSince(firstStamp ?? fitStart)
let refinementSeconds = fitEnd.timeIntervalSince(trialStageEnd ?? fitEnd)
let tpeTrials = fit.trials.filter { $0.stage == "tpe" }

var report: [String: Any] = [
  "device": device.name,
  "bf": 8937,
  "optimize_batch": environment["SSB_OPTIMIZE_BATCH"] ?? "off",
  "batch_trials": environment["SSB_BATCH_TRIALS"] ?? "2",
  "multi_batch_planes": environment["SSB_MULTI_BATCH_PLANES"] ?? "default",
  "load_average": (try? String(contentsOfFile: "/proc/loadavg", encoding: .utf8)) ?? "n/a",
  "fit_seconds": fit.elapsedSeconds,
  "trial_stage_seconds": trialStageSeconds,
  "refinement_stage_seconds": refinementSeconds,
  "refinement_evaluations": fit.refinementEvaluations,
  "fit_loss": fit.loss,
  "fit_best": [fit.best.c10Nanometers, fit.best.c12Nanometers, fit.best.phi12Radians],
  "trial_count": tpeTrials.count,
  "peak_allocated_bytes": device.currentAllocatedSize,
  "trials": tpeTrials.map {
    ["point": [$0.point.c10Nanometers, $0.point.c12Nanometers, $0.point.phi12Radians],
     "loss": $0.loss]
  },
]

if environment["SSB_PROBE5_REEVAL"] == "1" {
  var maxAbs = 0.0
  var maxRel = 0.0
  var mismatches = 0
  var rows: [[String: Any]] = []
  for (index, trial) in tpeTrials.enumerated() {
    let single = try engine.phaseVariance(
      aberrations: point(trial.point), rotationDegrees: rotation).loss
    let delta = Double(single) - trial.loss
    let rel = delta / max(Swift.abs(trial.loss), 1e-30)
    maxAbs = max(maxAbs, Swift.abs(delta))
    maxRel = max(maxRel, Swift.abs(rel))
    if single.bitPattern != Float(trial.loss).bitPattern { mismatches += 1 }
    rows.append([
      "index": index, "loss": trial.loss, "single": Double(single),
      "abs": Swift.abs(delta), "rel": Swift.abs(rel),
    ])
  }
  report["reeval"] = [
    "max_abs": maxAbs, "max_rel": maxRel, "bit_mismatches": mismatches, "rows": rows,
  ]
}

try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent("report.json"))
print(String(format: "fit %.2f s (trials %.2f s, refine %.2f s, %d evaluations) loss %.17g",
  fit.elapsedSeconds, trialStageSeconds, refinementSeconds,
  fit.refinementEvaluations, fit.loss))
print("Completed: \(output.path)")
