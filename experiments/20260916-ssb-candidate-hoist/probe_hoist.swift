import Foundation
import Metal
import MetalSSBKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// SCRATCH(hoist-probe): decide empirically whether any per-evaluation buffer of
// the 8937-term ARINA phase-loss pipeline is candidate-independent. Runs the
// real loss path twice with two different aberration triples on one engine, and
// digests every buffer the pipeline writes.
//
// usage: probe_hoist <master.h5> <output-dir> <c10> <c12> <phi12> [label]

let args = CommandLine.arguments
guard args.count >= 5 else {
  print("usage: probe_hoist <master.h5> <out-dir> <c10> <c12> <phi12> [label]")
  exit(2)
}
let output = URL(fileURLWithPath: args[2], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let candidateA = (Float(args[3])!, Float(args[4])!, Float(args[5])!)
let candidateB: (Float, Float, Float) = (5.0, 0.5, 0.0)
let label = args.count > 6 ? args[6] : "hoist"

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
  brightfieldRadiusPixels: 53.35992814757164,
  excludedDetectorPixels: [78 * 192 + 74])
try calibration.matchApertureToBrightfieldDisk()
let setup = try calibration.geometry(
  detectorRows: 192, detectorColumns: 192, detectorSum: mean.detectorSum)
precondition(setup.pixels.count == 8937, "expected 8937 active BF terms")
let engine = try MetalSSBEngine(
  device: device, geometry: setup.geometry, cacheBudgetBytes: nil)
let prepareStart = Date()
try engine.prepare(countType: .uint32) { indices, destination, commands in
  try source.encodeDetectorColumns(
    pixels: indices.map { setup.pixels[$0] }, into: destination, commands: commands)
}
let prepareSeconds = Date().timeIntervalSince(prepareStart)
let rotation: Float = 158.88268568029937

// Known-answer check: the device fold must reproduce the host fold bit for bit.
func cpuDigest(_ words: [UInt32]) -> UInt64 {
  var low: UInt32 = 0
  var high: UInt32 = 0
  var start = 0
  while start < words.count {
    let end = min(start + 16, words.count)
    var h: UInt64 = 1469598103934665603
    for index in start..<end {
      h ^= UInt64(words[index])
      h = h &* 1099511628211
      h ^= UInt64(index + 1)
    }
    low = low &+ UInt32(truncatingIfNeeded: h)
    high = high &+ UInt32(truncatingIfNeeded: h >> 32)
    start += 16
  }
  return UInt64(low) | (UInt64(high) << 32)
}
var knownAnswerWords: [UInt32] = (0..<40001).map { UInt32(truncatingIfNeeded: $0 &* 2654435761) }
let gpuKnownAnswer = try engine.probeDigestKnownAnswer(knownAnswerWords)
let cpuKnownAnswer = cpuDigest(knownAnswerWords)
knownAnswerWords[20000] = knownAnswerWords[20000] &+ 1
let mutatedHostDigest = cpuDigest(knownAnswerWords)
let gpuMutated = try engine.probeDigestKnownAnswer(knownAnswerWords)

var reports: [String: Any] = [:]
var losses: [String: Float] = [:]
var timings: [String: Double] = [:]

func evaluate(_ name: String, _ triple: (Float, Float, Float)) throws {
  let start = Date()
  let result = try engine.phaseVariance(
    aberrations: MetalSSBAberrations(
      c10Nanometers: triple.0, c12Nanometers: triple.1, phi12Radians: triple.2),
    rotationDegrees: rotation)
  timings[name] = Date().timeIntervalSince(start)
  losses[name] = result.loss
  reports[name] = engine.probeDumpDigests()
  print("\(name) loss=\(result.loss) wall=\(timings[name]!)s")
}

try evaluate("candidateA", candidateA)
try evaluate("candidateB", candidateB)
let gBytes = 8937 * 512 * 257 * 8
let report: [String: Any] = [
  "label": label,
  "bf_terms": 8937,
  "half_plane_bytes": gBytes / 8937,
  "g_cache_bytes": gBytes,
  "prepare_seconds": prepareSeconds,
  "losses": losses,
  "wall_seconds": timings,
  "known_answer": [
    "gpu": String(gpuKnownAnswer, radix: 16),
    "cpu": String(cpuKnownAnswer, radix: 16),
    "match": gpuKnownAnswer == cpuKnownAnswer,
    "gpu_mutated": String(gpuMutated, radix: 16),
    "cpu_mutated": String(mutatedHostDigest, radix: 16),
    "mutated_match": gpuMutated == mutatedHostDigest,
    "mutated_detected": gpuMutated != gpuKnownAnswer,
  ],
  "digests": reports,
  "allocated_bytes": device.currentAllocatedSize,
]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent("hoist-report.json"))
print("Completed: \(output.path)/hoist-report.json")
