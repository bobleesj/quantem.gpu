import Foundation
import Metal
import MetalSSBKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Sequence probe: separates pure loss cost from redraw<->loss alternating cost.
// Inputs, BF selection, aberration controls and parity outputs are identical to
// experiments/20260910-ssb-full-bf-scheduling/probe.swift; only the call
// pattern differs, so the difference isolates cache-layout switching.
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
let engine = try MetalSSBEngine(device: device, geometry: setup.geometry, cacheBudgetBytes: nil)
let prepareStart = Date()
try engine.prepare(countType: .uint32) { indices, destination, commands in
  try source.encodeDetectorColumns(pixels: indices.map { setup.pixels[$0] }, into: destination, commands: commands)
}
let prepareSeconds = Date().timeIntervalSince(prepareStart)
precondition(engine.executedBrightfieldCount == 8937)

let rotation: Float = 158.88268568029937
let controls: [Float] = [0, 55, 155.96977]
func params(_ c10: Float) -> MetalSSBAberrations {
  MetalSSBAberrations(c10Nanometers: c10, c12Nanometers: 42.96961, phi12Radians: 0.293384)
}
var phases: [[String: Any]] = []
var callIndex = 0

// Warm both paths once so pipeline compilation and first-touch are excluded.
_ = try engine.reconstruct(aberrations: params(0), rotationDegrees: rotation)
_ = try engine.phaseVariance(aberrations: params(0), rotationDegrees: rotation)

// Phase A: pure loss stream (cache stays in the loss layout).
var pureLoss: [[String: Any]] = []
for repetition in 0..<7 {
  for (index, c10) in controls.enumerated() {
    let wallStart = Date()
    let result = try engine.phaseVariance(aberrations: params(c10), rotationDegrees: rotation)
    let wall = Date().timeIntervalSince(wallStart) * 1000
    pureLoss.append(["repetition": repetition, "control": index, "gpu_ms": result.gpuSeconds * 1000,
      "wall_ms": wall, "untracked_ms": wall - result.gpuSeconds * 1000])
    callIndex += 1
  }
}
phases.append(["phase": "pure_loss", "entries": pureLoss])

// Phase B: alternating redraw then loss (the interactive access pattern).
var alternating: [[String: Any]] = []
for repetition in 0..<5 {
  for (index, c10) in controls.enumerated() {
    let redrawStart = Date()
    let redraw = try engine.reconstruct(aberrations: params(c10), rotationDegrees: rotation)
    let redrawWall = Date().timeIntervalSince(redrawStart) * 1000
    if repetition == 0 {
      try Data(bytes: redraw.object.contents(), count: redraw.object.length)
        .write(to: output.appendingPathComponent("object-\(index).c64"))
    }
    let lossStart = Date()
    let loss = try engine.phaseVariance(aberrations: params(c10), rotationDegrees: rotation)
    let lossWall = Date().timeIntervalSince(lossStart) * 1000
    alternating.append(["repetition": repetition, "control": index,
      "redraw_gpu_ms": redraw.gpuSeconds * 1000, "loss_gpu_ms": loss.gpuSeconds * 1000,
      "redraw_untracked_ms": redrawWall - redraw.gpuSeconds * 1000,
      "loss_untracked_ms": lossWall - loss.gpuSeconds * 1000])
    callIndex += 1
  }
}
phases.append(["phase": "alternating_redraw_loss", "entries": alternating])

// Phase C: pure loss again, to confirm phase A was not drift.
var tailLoss: [[String: Any]] = []
for repetition in 0..<3 {
  for (index, c10) in controls.enumerated() {
    let wallStart = Date()
    let result = try engine.phaseVariance(aberrations: params(c10), rotationDegrees: rotation)
    let wall = Date().timeIntervalSince(wallStart) * 1000
    tailLoss.append(["repetition": repetition, "control": index, "gpu_ms": result.gpuSeconds * 1000,
      "wall_ms": wall, "untracked_ms": wall - result.gpuSeconds * 1000, "loss": result.loss])
  }
}
phases.append(["phase": "pure_loss_tail", "entries": tailLoss])

let report: [String: Any] = ["device": device.name, "source": args[1], "bf": 8937,
  "load_seconds": loadSeconds, "prepare_seconds": prepareSeconds,
  "phases": phases, "sampled_allocated_bytes": device.currentAllocatedSize]
try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent("report.json"))
print("Completed: \(output.path)")
