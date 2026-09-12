import Foundation
import Metal
import MetalSSBKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Real original-file workload. Output buffers are parity evidence, not a load cache.
let args = CommandLine.arguments
let output = URL(fileURLWithPath: args[2], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let device = MTLCreateSystemDefaultDevice()!
let started = Date()
let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: output.appendingPathComponent("index"))
  .prepare(input: URL(fileURLWithPath: args[1]))
let indexed = try Native4DSTEMIndexedSource.open(dataset: catalog.datasets[0])
let source = try MetalCompactH5Loader.load(source: indexed, device: device)
let loadSeconds = Date().timeIntervalSince(started)
let mean = try source.meanDiffractionPattern()
var calibration = MetalSSBCalibration(beamEnergyKeV: 300, semiangleMrad: 30,
  scanStepRowAngstroms: 0.264, scanStepColumnAngstroms: 0.264,
  detectorStepRowMrad: 1, detectorStepColumnMrad: 1,
  centerRow: 94.88451385498047, centerColumn: 96.35952758789062,
  brightfieldRadiusPixels: 53.35992814757164, excludedDetectorPixels: [78 * 192 + 74])
try calibration.matchApertureToBrightfieldDisk()
let setup = try calibration.geometry(detectorRows: 192, detectorColumns: 192, detectorSum: mean.detectorSum)
precondition(setup.pixels.count == 8937)
let engine = try MetalSSBEngine(device: device, geometry: setup.geometry)
let prepareStart = Date()
try engine.prepare(countType: .uint32) { indices, destination, commands in
  try source.encodeDetectorColumns(pixels: indices.map { setup.pixels[$0] }, into: destination, commands: commands)
}
let prepareSeconds = Date().timeIntervalSince(prepareStart)
precondition(engine.executedBrightfieldCount == 8937)
let rotation: Float = 158.88268568029937
var redraws: [[String: Any]] = []
let controls: [Float] = [0, 55, 155.96977]
for repetition in 0..<4 {
  for (index, c10) in controls.enumerated() {
    let parameters = MetalSSBAberrations(c10Nanometers: c10, c12Nanometers: 42.96961, phi12Radians: 0.293384)
    let result = try engine.reconstruct(aberrations: parameters, rotationDegrees: rotation)
    redraws.append(["repetition": repetition, "c10": c10, "wall_seconds": result.wallSeconds, "gpu_seconds": result.gpuSeconds])
    if repetition == 0 {
      try Data(bytes: result.object.contents(), count: result.object.length)
        .write(to: output.appendingPathComponent("object-\(index).c64"))
    }
  }
}
var losses: [[String: Any]] = []
for repetition in 0..<3 {
  for c10 in controls {
    let parameters = MetalSSBAberrations(c10Nanometers: c10, c12Nanometers: 42.96961, phi12Radians: 0.293384)
    let start = Date()
    let result = try engine.phaseVariance(aberrations: parameters, rotationDegrees: rotation)
    losses.append(["repetition": repetition, "c10": c10, "loss": result.loss,
      "wall_seconds": Date().timeIntervalSince(start), "gpu_seconds": result.gpuSeconds])
  }
}
var report: [String: Any] = ["device": device.name, "source": args[1], "bf": 8937,
  "load_seconds": loadSeconds, "prepare_seconds": prepareSeconds,
  "redraws": redraws, "losses": losses, "sampled_allocated_bytes": device.currentAllocatedSize]
if args.count > 3 && args[3] == "fit" {
  let fit = try engine.optimize(start: MetalSSBAberrations(c10Nanometers: 0, c12Nanometers: 50, phi12Radians: 0),
    rotationDegrees: rotation, globalTrials: 200)
  report["fit_seconds"] = fit.elapsedSeconds
  report["fit_loss"] = fit.loss
  report["refinement_evaluations"] = fit.refinementEvaluations
  report["fit_best"] = [fit.best.c10Nanometers, fit.best.c12Nanometers, fit.best.phi12Radians]
}
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent("report.json"))
print("Completed: \(output.path)")
