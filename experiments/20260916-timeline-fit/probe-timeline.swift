import Foundation
import Metal
import MetalSSBKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Timeline verification of the Metal fit. Setup is identical to probe5
// (same input, calibration, BF selection, rotation, start point, seed and
// 200-trial budget); the only addition is a timeline recorder that captures,
// for every objective evaluation, the CPU wall span of each command buffer
// together with its GPU start/end and kernel start/end timestamps.
let args = CommandLine.arguments
let output = URL(fileURLWithPath: args[2], isDirectory: true)
try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
let environment = ProcessInfo.processInfo.environment
let timelineEnabled = environment["SSB_TIMELINE"] != "0"
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

final class TimelineSink {
  var records: [SSBEvaluationTimeline] = []
}
let sink = TimelineSink()
if timelineEnabled {
  let recorder = SSBTimelineRecorder()
  recorder.onEvaluation = { record in sink.records.append(record) }
  engine.timelineRecorder = recorder
}

var firstStamp: Date?
var trialStageEnd: Date?
var lastStage = ""
let progress: (SSBOptimizationProgress) -> Void = { update in
  if firstStamp == nil { firstStamp = Date() }
  if update.stage != lastStage { lastStage = update.stage }
  if update.stage == "tpe" && update.completed == update.total {
    trialStageEnd = Date()
  }
}

let fitWallStart = SSBTimelineRecorder.now()
let fit = try engine.optimize(
  start: MetalSSBAberrations(c10Nanometers: 0, c12Nanometers: 50, phi12Radians: 0),
  rotationDegrees: rotation, globalTrials: 200, progress: progress)
let fitWallEnd = SSBTimelineRecorder.now()
engine.timelineRecorder = nil

let trialStageSeconds = (trialStageEnd ?? Date()).timeIntervalSince(firstStamp ?? Date())
let refinementSeconds = Date().timeIntervalSince(trialStageEnd ?? Date())

if timelineEnabled {
  var lines: [String] = []
  lines.reserveCapacity(sink.records.count)
  for record in sink.records {
    let object: [String: Any] = [
      "index": record.index,
      "wall_start": record.wallStart,
      "wall_end": record.wallEnd,
      "gpu_seconds": record.gpuSeconds,
      "bf": record.brightfieldCount,
      "command_buffers": record.commandBuffers.map { buffer -> [String: Any] in
        [
          "label": buffer.label,
          "wall_start": buffer.wallStart,
          "wall_end": buffer.wallEnd,
          "gpu_start": buffer.gpuStart,
          "gpu_end": buffer.gpuEnd,
          "kernel_start": buffer.kernelStart,
          "kernel_end": buffer.kernelEnd,
        ]
      },
    ]
    let data = try JSONSerialization.data(withJSONObject: object, options: [.sortedKeys])
    lines.append(String(decoding: data, as: UTF8.self))
  }
  try lines.joined(separator: "\n").appending("\n")
    .write(to: output.appendingPathComponent("timeline.jsonl"), atomically: true, encoding: .utf8)
}

var summary: [String: Any] = [
  "device": device.name,
  "bf": 8937,
  "timeline_enabled": timelineEnabled,
  "fit_seconds": fit.elapsedSeconds,
  "fit_wall_seconds": fitWallEnd - fitWallStart,
  "trial_stage_seconds": trialStageSeconds,
  "refinement_stage_seconds": refinementSeconds,
  "refinement_evaluations": fit.refinementEvaluations,
  "fit_loss": fit.loss,
  "fit_best": [fit.best.c10Nanometers, fit.best.c12Nanometers, fit.best.phi12Radians],
  "evaluations_recorded": sink.records.count,
  "peak_allocated_bytes": device.currentAllocatedSize,
]
summary["load_average"] = (try? String(contentsOfFile: "/proc/loadavg", encoding: .utf8)) ?? "n/a"
try JSONSerialization.data(withJSONObject: summary, options: [.prettyPrinted, .sortedKeys])
  .write(to: output.appendingPathComponent(timelineEnabled ? "summary-timeline.json" : "summary-plain.json"))
print(String(format: "fit %.3f s (wall %.3f s, %d evaluations recorded) loss %.17g best [%.15g, %.15g, %.15g]",
  fit.elapsedSeconds, fitWallEnd - fitWallStart, sink.records.count, fit.loss,
  fit.best.c10Nanometers, fit.best.c12Nanometers, fit.best.phi12Radians))
print("Completed: \(output.path)")
