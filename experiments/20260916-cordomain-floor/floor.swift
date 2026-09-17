import Foundation
import Metal
import MetalSSBKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Independent per-evaluation floor check for the cordomain ticket. Loads the
// same 8937-BF ARINA source as probe5, then times only the complete loss
// (correction + column pass + Nyquist residual + row pass + phase moments) on
// a fixed candidate. No optimizer is involved, so this is the raw per-eval
// cost that the 62 s fit is built from.
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
precondition(engine.executedBrightfieldCount == 8937)
let rotation: Float = 158.88268568029937
func params(_ c10: Float) -> MetalSSBAberrations {
  MetalSSBAberrations(c10Nanometers: c10, c12Nanometers: 42.96961, phi12Radians: 0.293384)
}
// Warm once so pipeline state and first-touch are excluded.
_ = try engine.phaseVariance(aberrations: params(0), rotationDegrees: rotation)

var entries: [[String: Any]] = []
for index in 0..<16 {
  let c10: Float = index % 2 == 0 ? 0 : 155.96977
  let wallStart = Date()
  let result = try engine.phaseVariance(aberrations: params(c10), rotationDegrees: rotation)
  let wall = Date().timeIntervalSince(wallStart) * 1000
  let lossValue = Double(result.loss)
  entries.append([
    "index": index,
    "c10_nm": Double(c10),
    "gpu_ms": Double(result.gpuSeconds * 1000),
    "wall_ms": Double(wall),
    "untracked_ms": Double(wall - result.gpuSeconds * 1000),
    "loss": lossValue.isFinite ? lossValue as Any : "nonfinite" as Any,
  ] as [String: Any])
}
let gpu = entries.compactMap { $0["gpu_ms"] as? Double }.sorted()
func percentile(_ values: [Double], _ fraction: Double) -> Double {
  guard !values.isEmpty else { return 0 }
  let position = fraction * Double(values.count - 1)
  let lower = Int(position.rounded(.down))
  let upper = min(lower + 1, values.count - 1)
  let weight = position - Double(lower)
  return values[lower] * (1 - weight) + values[upper] * weight
}
let report: [String: Any] = [
  "device": device.name,
  "bf": 8937,
  "entries": entries,
  "gpu_ms_p50": Double(percentile(gpu, 0.5)),
  "gpu_ms_p95": Double(percentile(gpu, 0.95)),
  "gpu_ms_max": Double(gpu.last ?? 0),
  "gpu_ms_min": Double(gpu.first ?? 0),
  "sampled_allocated_bytes": Double(device.currentAllocatedSize),
]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent("report.json"))
print("loss p50 \(percentile(gpu, 0.5)) ms p95 \(percentile(gpu, 0.95)) ms min \(gpu.first ?? 0) max \(gpu.last ?? 0)")
print("Completed: \(output.path)")
