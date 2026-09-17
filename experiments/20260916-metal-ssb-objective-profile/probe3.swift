import Foundation
import Metal
import MetalSSBKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Minimal loss-only timing probe used for pass-level ablation. Identical input,
// BF selection and controls to probe.swift; only the number of loss calls is
// reduced so one profiling arm costs seconds rather than minutes.
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
let prepareStart = Date()
try engine.prepare(countType: .uint32) { indices, destination, commands in
  try source.encodeDetectorColumns(pixels: indices.map { setup.pixels[$0] }, into: destination, commands: commands)
}
let prepareSeconds = Date().timeIntervalSince(prepareStart)
let rotation: Float = 158.88268568029937
func params(_ c10: Float) -> MetalSSBAberrations {
  MetalSSBAberrations(c10Nanometers: c10, c12Nanometers: 42.96961, phi12Radians: 0.293384)
}
var samples: [[String: Any]] = []
for index in 0..<6 {
  let c10: Float = index == 0 ? 0 : (index % 2 == 0 ? 55 : 155.96977)
  let start = Date()
  let result = try engine.phaseVariance(aberrations: params(c10), rotationDegrees: rotation)
  let wall = Date().timeIntervalSince(start) * 1000
  samples.append(["index": index, "gpu_ms": result.gpuSeconds * 1000,
    "wall_ms": wall, "untracked_ms": wall - result.gpuSeconds * 1000, "loss": result.loss])
}
let report: [String: Any] = ["device": device.name, "bf": 8937, "prepare_seconds": prepareSeconds,
  "samples": samples, "allocated_bytes": device.currentAllocatedSize,
  "profile_skip_column": ProcessInfo.processInfo.environment["SSB_PROFILE_SKIP_COLUMN"] ?? "0",
  "profile_skip_nyquist": ProcessInfo.processInfo.environment["SSB_PROFILE_SKIP_NYQUIST"] ?? "0",
  "profile_skip_rows": ProcessInfo.processInfo.environment["SSB_PROFILE_SKIP_ROWS"] ?? "0"]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent("report.json"))
print("Completed: \(output.path)")
