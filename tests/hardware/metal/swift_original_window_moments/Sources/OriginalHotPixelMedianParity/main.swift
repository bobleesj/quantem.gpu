import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Load an original acquisition with one hot-pixel policy through the public
// loader and write what the resident holds: per-pixel totals, every frame's
// exact DPC moments and the columns of the source-flagged pixels. The caller
// compares them with the Python fill of the same counts.
let arguments = CommandLine.arguments
let input = URL(fileURLWithPath: arguments[1])
let cache = URL(fileURLWithPath: arguments[2], isDirectory: true)
let output = URL(fileURLWithPath: arguments[3], isDirectory: true)
guard let policy = NativeHotPixelPolicy(rawValue: arguments[4]),
  let device = MTLCreateSystemDefaultDevice(), let queue = device.makeCommandQueue()
else { fatalError("Usage: <master> <cache> <output> median|preserve on a Metal device") }
let dataset = try Native4DSTEMCatalogBuilder(cacheDirectory: cache).prepare(input: input)
  .datasets[0]
let resident = try MetalCompactH5Loader.load(
  source: Native4DSTEMIndexedSource.open(dataset: dataset), device: device,
  hotPixelPolicy: policy)
defer { resident.releaseResidentStorage() }
let scans = resident.metadata.scanCount
let flagged = dataset.badPixelIndices
let totals = try resident.meanDiffractionPattern().detectorSum
guard let moments = try resident.preparedDPCMomentValues() else {
  fatalError("The resident has no exact DPC moments")
}
let interleaved = moments.total.indices.flatMap {
  [moments.total[$0], moments.detectorRowMoment[$0], moments.detectorColumnMoment[$0]]
}
guard let columns = device.makeBuffer(length: flagged.count * scans * 4, options: .storageModeShared),
  let command = queue.makeCommandBuffer()
else { fatalError("Cannot allocate the detector-column readback") }
try resident.encodeDetectorColumns(pixels: flagged, into: columns, commands: command)
command.commit()
command.waitUntilCompleted()
precondition(command.status == .completed, "Detector-column readback failed")
try totals.withUnsafeBytes { try Data($0).write(to: output.appendingPathComponent("detector_sum.u64")) }
try interleaved.withUnsafeBytes { try Data($0).write(to: output.appendingPathComponent("moments.u64")) }
try Data(bytes: columns.contents(), count: columns.length)
  .write(to: output.appendingPathComponent("columns.u32"))
let summary: [String: Any] = [
  "flagged": flagged, "scans": scans, "excluded": resident.metadata.excludedDetectorPixels,
]
print(String(data: try JSONSerialization.data(withJSONObject: summary, options: [.sortedKeys]),
             encoding: .utf8)!)
