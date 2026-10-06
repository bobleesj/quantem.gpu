import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

// Compare these complete GPU-output fingerprints with a frozen pre-change run.
// No CPU reconstruction, dense resident, source mutation, or large export.
let args = CommandLine.arguments
guard args.count == 2 else { fatalError("usage: EMPADRegionMeanParity source.raw|source.xml|source.qem") }
let device = MTLCreateSystemDefaultDevice()!
let queue = device.makeCommandQueue()!
let source = try NativeEMPADSource.open(URL(fileURLWithPath: args[1]))
var resident = try MetalEMPADResidentSource.load(
  source, device: device, memoryBudgetBytes: UInt64(device.recommendedMaxWorkingSetSize))
defer { resident.releaseResidentStorage() }
if let path = ProcessInfo.processInfo.environment["EMPAD_MEAN_QEM"] {
  try resident.saveQEM(to: URL(fileURLWithPath: path))
  resident.releaseResidentStorage()
  let saved = try NativeEMPADSource.open(URL(fileURLWithPath: path))
  resident = try MetalEMPADResidentSource.load(
    saved, device: device, memoryBudgetBytes: UInt64(device.recommendedMaxWorkingSetSize))
  print("QEM_REOPEN_PASS source_identity=\(resident.sourceIdentitySHA256)")
}
let output = device.makeBuffer(length: 128 * 128 * 4, options: .storageModeShared)!
print("DEVICE \(device.name) shape=\(source.scanRows)x\(source.scanColumns)x128x128 float32 resident_bytes=\(resident.residentBytes)")
let side = min(source.scanRows, source.scanColumns)
var cases: [(String, Range<Int>, Range<Int>, MetalScanRegionShape)] = []
for diameter in [1, 2, 8, 32, 43, 64].filter({ $0 <= side }) {
  for shape in [MetalScanRegionShape.circle, .rectangle] {
    for edge in [false, true] {
      let row = edge ? source.scanRows - diameter : (source.scanRows - diameter) / 2
      let column = edge ? 0 : (source.scanColumns - diameter) / 3
      cases.append(("\(shape)-\(diameter)-\(edge ? "edge" : "center")", row..<row+diameter,
        column..<column+diameter, shape))
    }
  }
}
cases.append(("full", 0..<source.scanRows, 0..<source.scanColumns, .rectangle))
if side >= 12 { cases.append(("nonsquare", 2..<7, 3..<12, .rectangle)) }
for (name, rows, columns, shape) in cases {
  let command = queue.makeCommandBuffer()!
  let started = CFAbsoluteTimeGetCurrent()
  try resident.encodeMeanDiffraction(into: output, command: command, rows: rows, columns: columns, shape: shape)
  let submitted = CFAbsoluteTimeGetCurrent()
  command.commit(); command.waitUntilCompleted()
  guard command.status == .completed else { fatalError("Region command failed: \(String(describing: command.error))") }
  let wall = (CFAbsoluteTimeGetCurrent() - started) * 1000
  let hash = SHA256.hash(data: Data(bytes: output.contents(), count: output.length))
    .map { String(format: "%02x", $0) }.joined()
  print("CASE \(name) sha256=\(hash) wall_ms=\(wall) encode_ms=\((submitted-started)*1000) gpu_ms=\((command.gpuEndTime-command.gpuStartTime)*1000)")
  if ProcessInfo.processInfo.environment["EMPAD_MEAN_VERIFY"] == "1" {
    let reference = queue.makeCommandBuffer()!
    setenv("QGPU_FLOAT_ANS_MEAN_CONTROL", "1", 1)
    try resident.encodeMeanDiffraction(into: output, command: reference, rows: rows, columns: columns, shape: shape)
    unsetenv("QGPU_FLOAT_ANS_MEAN_CONTROL")
    reference.commit(); reference.waitUntilCompleted()
    guard reference.status == .completed else { fatalError("Reference command failed") }
    let expected = SHA256.hash(data: Data(bytes: output.contents(), count: output.length))
      .map { String(format: "%02x", $0) }.joined()
    precondition(hash == expected, "GPU reference mismatch for \(name): \(hash) != \(expected)")
    print("EXACT_GPU_PARITY \(name)")
  }
}
// Same moving, 43-pixel diameter region as the reported interactive workflow.
let diameter = min(43, side)
for shape in [MetalScanRegionShape.circle, .rectangle] {
  var samples: [Double] = []
  var gpu: [Double] = []
  for step in 0..<24 {
    let row = min(source.scanRows-diameter, source.scanRows/3 + step)
    let column = min(source.scanColumns-diameter, source.scanColumns/4 + step)
    let command = queue.makeCommandBuffer()!
    let started = CFAbsoluteTimeGetCurrent()
    try resident.encodeMeanDiffraction(into: output, command: command,
      rows: row..<row+diameter, columns: column..<column+diameter, shape: shape)
    command.commit(); command.waitUntilCompleted()
    guard command.status == .completed else { fatalError("Moving region failed") }
    samples.append((CFAbsoluteTimeGetCurrent()-started)*1000)
    gpu.append((command.gpuEndTime-command.gpuStartTime)*1000)
  }
  samples.sort(); gpu.sort()
  print("COMPUTE_ONLY \(shape) n=24 median_ms=\(samples[12]) p95_ms=\(samples[22]) max_ms=\(samples[23]) gpu_median_ms=\(gpu[12]) allocated_bytes=\(device.currentAllocatedSize)")
}
