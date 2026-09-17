import Foundation
import Metal
import MetalSSBKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Traffic-attribution probe.
//
// One process, one engine prepare, then an interleaved schedule of fractional
// pass ablations. Every arm runs the production kernels; the arms differ only in
// how many cached batch dispatches encode a given pass. Because all arms are
// cycled round-robin inside one thermal/memory/load state, load drift is
// common-mode rather than arm-specific.
//
// Usage: probe_traffic <master.h5> <output-dir> [reps]

let args = CommandLine.arguments
guard args.count >= 3 else {
  FileHandle.standardError.write(Data("usage: probe_traffic <master.h5> <out-dir> [reps]\n".utf8))
  exit(2)
}
let output = URL(fileURLWithPath: args[2], isDirectory: true)
setvbuf(stdout, nil, _IONBF, 0)
let reps = args.count >= 4 ? (Int(args[3]) ?? 4) : 4
let label = args.count >= 5 ? args[4] : "session"
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)

func loadAverages() -> [Double] {
  var loads = [Double](repeating: 0, count: 3)
  let count = getloadavg(&loads, 3)
  return count >= 1 ? loads : [-1, -1, -1]
}

let device = MTLCreateSystemDefaultDevice()!
let catalog = try Native4DSTEMCatalogBuilder(
  cacheDirectory: output.appendingPathComponent("index")
).prepare(input: URL(fileURLWithPath: args[1]))
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
precondition(setup.pixels.count == 8937, "expected the production 8937-term BF disk")

let engine = try MetalSSBEngine(device: device, geometry: setup.geometry, cacheBudgetBytes: nil)
let prepareStart = Date()
try engine.prepare(countType: .uint32) { indices, destination, commands in
  try source.encodeDetectorColumns(
    pixels: indices.map { setup.pixels[$0] }, into: destination, commands: commands)
}
let prepareSeconds = Date().timeIntervalSince(prepareStart)

let rotation: Float = 158.88268568029937
func params(_ c10: Float) -> MetalSSBAberrations {
  MetalSSBAberrations(c10Nanometers: c10, c12Nanometers: 42.96961, phi12Radians: 0.293384)
}

let gateNames = [
  "SSB_PROFILE_COLUMN_EVERY", "SSB_PROFILE_NYQUIST_EVERY",
  "SSB_PROFILE_ROWS_EVERY", "SSB_PROFILE_TRIG_EVERY", "SSB_PHASE_BATCH",
]
func applyEnvironment(_ settings: [String: String]) {
  for name in gateNames { unsetenv(name) }
  for (name, value) in settings { setenv(name, value, 1) }
}

struct Sample {
  let arm: String
  let phase: String
  let rep: Int
  let c10: Float
  let gpuMs: Double
  let wallMs: Double
  let loss: Float
  let load1: Double
  let load5: Double
  let epoch: Double
  let env: [String: String]
}

func measure(arm: String, phase: String, rep: Int, c10: Float,
             env: [String: String]) throws -> Sample {
  applyEnvironment(env)
  let loads = loadAverages()
  let c10Value: Float = c10
  let start = Date()
  let result = try engine.phaseVariance(
    aberrations: params(c10Value), rotationDegrees: rotation)
  let wallMs = Date().timeIntervalSince(start) * 1000
  return Sample(
    arm: arm, phase: phase, rep: rep, c10: c10Value,
    gpuMs: result.gpuSeconds * 1000, wallMs: wallMs,
    loss: result.loss, load1: loads[0], load5: loads[1],
    epoch: Date().timeIntervalSince1970, env: env)
}

// Pinned losses from the frozen objective-profile probe at these controls.
// probe3 alternated c10 = 155.96977 on odd indices and 55 on even ones.
let pinned: [(Float, Float)] = [
  (0, 0.14511984586715698),
  (55, 0.13808111846446991),
  (155.96977, 0.13864889740943909),
]

var samples: [Sample] = []

// Gate: reproduce the frozen full-path losses before any ablation.
var gateRows: [[String: Any]] = []
var gateFailures: [String] = []
for (c10, expected) in pinned {
  let sample = try measure(arm: "gate_full", phase: "gate", rep: 0, c10: c10, env: [:])
  samples.append(sample)
  let match = sample.loss == expected
  if !match {
    gateFailures.append("c10=\(c10): \(sample.loss) != \(expected)")
  }
  gateRows.append([
    "c10": Double(c10), "expected_loss": Double(expected), "loss": Double(sample.loss),
    "match": match, "gpu_ms": sample.gpuMs, "wall_ms": sample.wallMs, "load1": sample.load1,
  ])
}

// Warm the pipeline with the production path so no arm pays first-call cost.
_ = try measure(arm: "warmup", phase: "warmup", rep: 0, c10: 155.96977, env: [:])
_ = try measure(arm: "warmup", phase: "warmup", rep: 1, c10: 155.96977, env: [:])

let arms: [(String, [String: String])] = [
  ("full", [:]),
  ("col_half", ["SSB_PROFILE_COLUMN_EVERY": "2"]),
  ("col_none", ["SSB_PROFILE_COLUMN_EVERY": "0"]),
  ("nyq_half", ["SSB_PROFILE_NYQUIST_EVERY": "2"]),
  ("nyq_none", ["SSB_PROFILE_NYQUIST_EVERY": "0"]),
  ("row_half", ["SSB_PROFILE_ROWS_EVERY": "2"]),
  ("row_none", ["SSB_PROFILE_ROWS_EVERY": "0"]),
  ("trig_none", ["SSB_PROFILE_TRIG_EVERY": "0"]),
  ("floor", [
    "SSB_PROFILE_COLUMN_EVERY": "0", "SSB_PROFILE_ROWS_EVERY": "0",
    "SSB_PROFILE_TRIG_EVERY": "0",
  ]),
  ("pb32", ["SSB_PHASE_BATCH": "32"]),
  ("pb16", ["SSB_PHASE_BATCH": "16"]),
]

let ablationC10: Float = 155.96977
for rep in 0..<reps {
  let offset = rep % arms.count
  for step in 0..<arms.count {
    let arm = arms[(offset + step) % arms.count]
    samples.append(try measure(
      arm: arm.0, phase: "ablation", rep: rep, c10: ablationC10, env: arm.1))
  }
}

applyEnvironment([:])

func summarize(_ rows: [Sample]) -> [String: Any] {
  let times = rows.map { $0.gpuMs }.sorted()
  guard !times.isEmpty else { return [:] }
  func percentile(_ q: Double) -> Double {
    let position = q * Double(times.count - 1)
    let lower = Int(position.rounded(.down))
    let upper = Int(position.rounded(.up))
    if lower == upper { return times[lower] }
    let fraction = position - Double(lower)
    return times[lower] * (1 - fraction) + times[upper] * fraction
  }
  return [
    "n": times.count,
    "min": times.first!, "p50": percentile(0.5), "p95": percentile(0.95),
    "max": times.last!, "mean": times.reduce(0, +) / Double(times.count),
  ]
}

var perArm: [String: Any] = [:]
let byArm = Dictionary(grouping: samples.filter { $0.phase == "ablation" }, by: { $0.arm })
for (arm, rows) in byArm {
  var summary = summarize(rows)
  let loads = rows.map { $0.load1 }
  summary["load1_min"] = loads.min()
  summary["load1_max"] = loads.max()
  summary["wall_p50"] = summarize(rows.map { Sample(
    arm: $0.arm, phase: $0.phase, rep: $0.rep, c10: $0.c10, gpuMs: $0.wallMs,
    wallMs: $0.wallMs, loss: $0.loss, load1: $0.load1, load5: $0.load5,
    epoch: $0.epoch, env: $0.env) })["p50"] ?? 0
  summary["loss_first"] = Double(rows.first!.loss)
  perArm[arm] = summary
}

let report: [String: Any] = [
  "device": device.name,
  "bf": setup.pixels.count,
  "reps": reps,
  "prepare_seconds": prepareSeconds,
  "allocated_bytes": device.currentAllocatedSize,
  "gate": ["rows": gateRows, "failures": gateFailures],
  "summary": perArm,
  "samples": samples.map { [
    "arm": $0.arm, "phase": $0.phase, "rep": $0.rep, "c10": Double($0.c10),
    "gpu_ms": $0.gpuMs, "wall_ms": $0.wallMs,
    "untracked_ms": $0.wallMs - $0.gpuMs, "loss": Double($0.loss),
    "load1": $0.load1, "load5": $0.load5, "epoch": $0.epoch, "env": $0.env,
  ] },
]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent("report-\(label).json"))

print("gate: \(gateFailures.isEmpty ? "PASS" : "FAIL \(gateFailures)")")
print(String(format: "%-10s %5s %9s %9s %9s %9s %6s %6s", "arm", "n", "p50", "min", "p95", "max", "ld1min", "ld1max"))
for arm in arms.map({ $0.0 }) {
  guard let summary = perArm[arm] as? [String: Any] else { continue }
  print(String(format: "%-10s %5d %9.1f %9.1f %9.1f %9.1f %6.2f %6.2f",
    arm, summary["n"] as! Int, summary["p50"] as! Double, summary["min"] as! Double,
    summary["p95"] as! Double, summary["max"] as! Double,
    summary["load1_min"] as! Double, summary["load1_max"] as! Double))
}
print("wrote \(output.appendingPathComponent("report-\(label).json").path)")
exit(0)
