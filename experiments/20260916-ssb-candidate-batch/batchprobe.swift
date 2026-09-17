import Foundation
import Metal
import MetalSSBKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Paired A/B harness for the shared-sweep candidate-batch objective.
//
//   batchprobe points <input.h5> <output-dir>   evaluate fixed candidates sequentially and in batches
//   batchprobe fit    <input.h5> <output-dir>   full TPE + Nelder-Mead fit (sequential when k <= 1)
//
// Every arm uses the production 8937-executed-term ARINA objective, the same
// calibration and controls as probe5.swift, and the same Float32 conversion the
// optimizer applies. Env knobs: SSB_CANDBATCH_K, SSB_CANDBATCH_ORDER,
// SSB_CANDBATCH_REPS, SSB_CANDBATCH_POINTS, SSB_CANDBATCH_TRIALS,
// SSB_CANDBATCH_VERIFY, SSB_CANDBATCH_OUT.
let args = CommandLine.arguments
guard args.count >= 4 else {
  FileHandle.standardError.write(
    "usage: batchprobe <points|fit> <input.h5> <output-dir>\n".data(using: .utf8)!)
  exit(2)
}
let mode = args[1]
let output = URL(fileURLWithPath: args[3], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let environment = ProcessInfo.processInfo.environment
let batchSize = Int(environment["SSB_CANDBATCH_K"] ?? "8") ?? 8
let reps = Int(environment["SSB_CANDBATCH_REPS"] ?? "3") ?? 3
let pointCount = Int(environment["SSB_CANDBATCH_POINTS"] ?? "16") ?? 16
let globalTrials = Int(environment["SSB_CANDBATCH_TRIALS"] ?? "200") ?? 200
let verifyCount = Int(environment["SSB_CANDBATCH_VERIFY"] ?? "0") ?? 0
let order =
  SSBCandidateBatchOrder(rawValue: environment["SSB_CANDBATCH_ORDER"] ?? "groupedPasses")
  ?? .groupedPasses
let planesPerRange = Int(environment["SSB_CANDBATCH_PLANES"] ?? "0") ?? 0

func loadAverage() -> [Double] {
  var loads = [Double](repeating: 0, count: 3)
  guard getloadavg(&loads, 3) == 3 else { return [] }
  return loads.map { (($0 * 100).rounded() / 100) }
}

let device = MTLCreateSystemDefaultDevice()!
let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: output.appendingPathComponent("index"))
  .prepare(input: URL(fileURLWithPath: args[2]))
let indexed = try Native4DSTEMIndexedSource.open(dataset: catalog.datasets[0])
let source = try MetalCompactH5Loader.load(source: indexed, device: device)
let mean = try source.meanDiffractionPattern()
var calibration = MetalSSBCalibration(
  beamEnergyKeV: 300, semiangleMrad: 30,
  scanStepRowAngstroms: 0.264, scanStepColumnAngstroms: 0.264,
  detectorStepRowMrad: 1, detectorStepColumnMrad: 1,
  centerRow: 94.88451385498047, centerColumn: 96.35952758789062,
  brightfieldRadiusPixels: 53.35992814757164, excludedDetectorPixels: [78 * 192 + 74])
try calibration.matchApertureToBrightfieldDisk()
let setup = try calibration.geometry(
  detectorRows: 192, detectorColumns: 192, detectorSum: mean.detectorSum)
precondition(setup.pixels.count == 8937)
let engine = try MetalSSBEngine(device: device, geometry: setup.geometry, cacheBudgetBytes: nil)
try engine.prepare(countType: .uint32) { indices, destination, commands in
  try source.encodeDetectorColumns(
    pixels: indices.map { setup.pixels[$0] }, into: destination, commands: commands)
}
precondition(engine.executedBrightfieldCount == 8937)
let rotation: Float = 158.88268568029937

func point(_ c10: Double, _ c12: Double, _ phi12: Double) -> SSBOptimizationPoint {
  SSBOptimizationPoint(c10Nanometers: c10, c12Nanometers: c12, phi12Radians: phi12)
}

func aberrations(_ p: SSBOptimizationPoint) -> MetalSSBAberrations {
  MetalSSBAberrations(
    c10Nanometers: Float(p.c10Nanometers),
    c12Nanometers: Float(p.c12Nanometers),
    phi12Radians: Float(p.phi12Radians))
}

// Deterministic fixtures: production start point, the pinned optimum, probe
// controls and a spread of the uniform draw bounds.
let fixtures: [SSBOptimizationPoint] = [
  point(0, 50, 0),
  point(6.603671591975075, 0.09848762314231685, 1.1341200767509283),
  point(55, 42.96961, 0.293384),
  point(155.96977, 42.96961, 0.293384),
  point(0, 42.96961, 0.293384),
  point(400, 100, 1.5707963267948966),
  point(-400, 100, -1.5707963267948966),
  point(200, 25, 0.5),
  point(-200, 75, 2.0),
  point(100, 0, 1.0),
  point(-100, 50, -1.0),
  point(6.7, 5.0, 1.1),
  point(0, 0.1, 0.0),
  point(-50, 90, -1.4),
  point(300, 10, 1.2),
  point(-300, 30, -0.2),
  point(12, 60, 0.9),
  point(-12, 3, -0.9),
  point(250, 80, -1.1),
  point(-250, 20, 1.45),
].prefix(max(2, pointCount)).map { $0 }

struct ArmSample: Encodable {
  let rep: Int
  let arm: String
  let chunkSizes: [Int]
  let batchWallSeconds: [Double]
  let perEvalSeconds: [Double]
  let gpuSeconds: [Double]
  let losses: [Double]
  let loadAverage: [Double]
}

var samples: [ArmSample] = []
var batchStats: [String: Any] = [:]

if mode == "points" {
  var sequentialReference: [UInt32] = []
  let started = Date()
  for rep in 0..<reps {
    // Rotate arm order so no arm owns a fixed position.
    let sequentialFirst = rep % 2 == 0
    func runSequential() throws {
      let loads = loadAverage()
      var perEval = [Double]()
      var gpu = [Double]()
      var losses = [Double]()
      var bits = [UInt32]()
      for fixture in fixtures {
        let t0 = Date()
        let result = try engine.phaseVariance(
          aberrations: aberrations(fixture), rotationDegrees: rotation)
        let elapsed = Date().timeIntervalSince(t0)
        perEval.append(elapsed)
        gpu.append(result.gpuSeconds)
        losses.append(Double(result.loss))
        bits.append(result.loss.bitPattern)
      }
      if sequentialReference.isEmpty {
        sequentialReference = bits
      }
      samples.append(
        ArmSample(
          rep: rep, arm: "sequential", chunkSizes: fixtures.map { _ in 1 },
          batchWallSeconds: perEval, perEvalSeconds: perEval, gpuSeconds: gpu,
          losses: losses, loadAverage: loads))
    }
    func runBatched() throws {
      let loads = loadAverage()
      var chunkSizes = [Int]()
      var walls = [Double]()
      var perEval = [Double]()
      var gpu = [Double]()
      var losses = [Double]()
      var cursor = 0
      while cursor < fixtures.count {
        let chunk = Array(fixtures[cursor..<min(cursor + max(1, batchSize), fixtures.count)])
        let t0 = Date()
        let result = try engine.phaseVarianceBatch(
          aberrations: chunk.map(aberrations), rotationDegrees: rotation, order: order,
          planesPerRange: planesPerRange)
        let elapsed = Date().timeIntervalSince(t0)
        chunkSizes.append(chunk.count)
        walls.append(elapsed)
        perEval.append(elapsed / Double(chunk.count))
        gpu.append(result.gpuSeconds)
        losses.append(contentsOf: result.losses.map(Double.init))
        if batchStats.isEmpty {
          batchStats = [
            "candidate_count": result.stats.candidateCount,
            "ranged_pass_count": result.stats.rangedPassCount,
            "encoder_count": result.stats.encoderCount,
            "cached_brightfield_count": result.stats.cachedBrightfieldCount,
            "streamed_brightfield_count": result.stats.streamedBrightfieldCount,
            "used_sequential_fallback": result.stats.usedSequentialFallback,
          ]
        }
        cursor += chunk.count
      }
      samples.append(
        ArmSample(
          rep: rep, arm: "batched", chunkSizes: chunkSizes, batchWallSeconds: walls,
          perEvalSeconds: perEval, gpuSeconds: gpu, losses: losses, loadAverage: loads))
    }
    if sequentialFirst {
      try runSequential()
      try runBatched()
    } else {
      try runBatched()
      try runSequential()
    }
  }
  // Bit-parity gate: every batched loss must equal the sequential loss exactly.
  var mismatches: [[String: Any]] = []
  for sample in samples where sample.arm == "batched" {
    for (index, loss) in sample.losses.enumerated() {
      if Float(loss).bitPattern != sequentialReference[index] {
        mismatches.append([
          "rep": sample.rep, "index": index,
          "sequential": Float(bitPattern: sequentialReference[index]),
          "batched": Float(loss),
        ])
      }
    }
  }
  var report: [String: Any] = [
    "mode": mode,
    "k": batchSize,
    "order": order.rawValue,
    "planes_per_range": planesPerRange,
    "reps": reps,
    "fixture_count": fixtures.count,
    "device": device.name,
    "bf": 8937,
    "scan": 512,
    "wall_seconds": Date().timeIntervalSince(started),
    "bit_identical": mismatches.isEmpty,
    "mismatches": mismatches,
    "batch_stats": batchStats,
    "samples": try samples.map { sample -> [String: Any] in
      let data = try JSONEncoder().encode(sample)
      return try JSONSerialization.jsonObject(with: data) as! [String: Any]
    },
  ]
  report["load_average_end"] = loadAverage()
  let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
  let out = environment["SSB_CANDBATCH_OUT"].map { output.appendingPathComponent($0) }
    ?? output.appendingPathComponent("batch.json")
  try data.write(to: out)
  print("wrote \(out.path)")
  print("bit_identical=\(mismatches.isEmpty) k=\(batchSize) order=\(order.rawValue)")
  for sample in samples where sample.arm == "batched" {
    let sorted = sample.perEvalSeconds.sorted()
    let p50 = sorted[sorted.count / 2]
    print(
      "batched rep \(sample.rep): per-eval mean "
        + String(format: "%.1f", 1000 * sample.perEvalSeconds.reduce(0, +) / Double(sample.perEvalSeconds.count))
        + " ms p50 " + String(format: "%.1f", 1000 * p50) + " ms")
  }
  for sample in samples where sample.arm == "sequential" {
    let sorted = sample.perEvalSeconds.sorted()
    let p50 = sorted[sorted.count / 2]
    print(
      "sequential rep \(sample.rep): per-eval mean "
        + String(format: "%.1f", 1000 * sample.perEvalSeconds.reduce(0, +) / Double(sample.perEvalSeconds.count))
        + " ms p50 " + String(format: "%.1f", 1000 * p50) + " ms")
  }
} else {
  var stageStamps: [[String: Double]] = []
  var firstStamp: Date?
  var trialStageEnd: Date?
  var lastStage = ""
  let progress: (SSBOptimizationProgress) -> Void = { update in
    if firstStamp == nil { firstStamp = Date() }
    if update.stage != lastStage { lastStage = update.stage }
    if update.stage == "tpe" && update.completed == update.total { trialStageEnd = Date() }
  }
  let started = Date()
  let fit = try engine.optimize(
    start: MetalSSBAberrations(c10Nanometers: 0, c12Nanometers: 50, phi12Radians: 0),
    rotationDegrees: rotation,
    globalTrials: globalTrials,
    candidateBatchSize: batchSize,
    candidateBatchOrder: order,
    candidateBatchPlanes: planesPerRange,
    progress: progress)
  let finished = Date()
  stageStamps.append([
    "trial_stage_seconds": (trialStageEnd ?? finished).timeIntervalSince(firstStamp ?? started),
    "refinement_stage_seconds": finished.timeIntervalSince(trialStageEnd ?? finished),
    "write_seconds": 0,
  ])
  var trials: [[String: Any]] = []
  for trial in fit.trials {
    trials.append([
      "c10": trial.point.c10Nanometers,
      "c12": trial.point.c12Nanometers,
      "phi12": trial.point.phi12Radians,
      "loss": Double(trial.loss),
      "stage": trial.stage,
    ])
  }
  var verification: [String: Any] = ["verified": 0, "mismatches": []]
  if verifyCount > 0 {
    var mismatches: [[String: Any]] = []
    let tpeTrials = fit.trials.filter { $0.stage == "tpe" }
    var probes = tpeTrials.prefix(verifyCount).map {
      (point: $0.point, loss: Double($0.loss), stage: $0.stage)
    }
    probes.append(
      (
        point: SSBOptimizationPoint(
          c10Nanometers: fit.best.c10Nanometers,
          c12Nanometers: fit.best.c12Nanometers,
          phi12Radians: fit.best.phi12Radians),
        loss: Double(fit.loss), stage: "best"
      ))
    for trial in probes {
      let sequential = try engine.phaseVariance(
        aberrations: aberrations(trial.point), rotationDegrees: rotation)
      if sequential.loss.bitPattern != Float(trial.loss).bitPattern {
        mismatches.append([
          "stage": trial.stage,
          "recorded": trial.loss,
          "sequential": Double(sequential.loss),
        ])
      }
    }
    verification = ["verified": probes.count, "mismatches": mismatches]
  }
  let report: [String: Any] = [
    "mode": mode,
    "k": batchSize,
    "order": order.rawValue,
    "planes_per_range": planesPerRange,
    "global_trials": globalTrials,
    "device": device.name,
    "bf": 8937,
    "scan": 512,
    "fit_seconds": fit.elapsedSeconds,
    "wall_seconds": Date().timeIntervalSince(started),
    "refinement_evaluations": fit.refinementEvaluations,
    "fit_loss": Double(fit.loss),
    "fit_best": [
      fit.best.c10Nanometers, fit.best.c12Nanometers, fit.best.phi12Radians,
    ],
    "stages": stageStamps,
    "trials": trials,
    "sequential_verification": verification,
    "load_average_start": loadAverage(),
    "load_average_end": loadAverage(),
    "peak_allocated_bytes": device.currentAllocatedSize,
  ]
  let data = try JSONSerialization.data(withJSONObject: report, options: [.sortedKeys])
  let out = environment["SSB_CANDBATCH_OUT"].map { output.appendingPathComponent($0) }
    ?? output.appendingPathComponent("fit.json")
  try data.write(to: out)
  print("wrote \(out.path)")
  print(
    "k=\(batchSize) fit_seconds=\(String(format: "%.2f", fit.elapsedSeconds)) "
      + "loss=\(fit.loss) best=\(fit.best.c10Nanometers),\(fit.best.c12Nanometers),\(fit.best.phi12Radians)")
}
