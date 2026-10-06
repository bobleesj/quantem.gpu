import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

/// Heads-up profile and parity gate for the restored `.qem` detector path.
///
/// Usage: `metal-qem-detector-update-benchmark <qem file or folder> [tiles] [steps]`
/// Prints per-update wall and GPU milliseconds for the single-acquisition path
/// the app uses while one dataset is on screen, the batched comparison path, and
/// the exact-count parity checks for both.
func percentile(_ values: [Double], _ fraction: Double) -> Double {
  guard !values.isEmpty else { return 0 }
  let sorted = values.sorted()
  return sorted[min(sorted.count - 1, max(0, Int((fraction * Double(sorted.count - 1)).rounded())))]
}

func digest(_ buffer: MTLBuffer) -> String {
  SHA256.hash(data: Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none))
    .map { String(format: "%02x", $0) }.joined()
}

func fixtureURLs(_ path: String, limit: Int) throws -> [URL] {
  var isDirectory: ObjCBool = false
  FileManager.default.fileExists(atPath: path, isDirectory: &isDirectory)
  var urls: [URL]
  if isDirectory.boolValue {
    urls = try FileManager.default.contentsOfDirectory(
      at: URL(fileURLWithPath: path), includingPropertiesForKeys: nil
    ).filter { $0.pathExtension.lowercased() == "qem" }.sorted { $0.path < $1.path }
  } else {
    urls = [URL(fileURLWithPath: path)]
  }
  return Array(urls.prefix(max(1, limit)))
}

func load(_ urls: [URL], device: MTLDevice) throws -> [MetalRuntimeANSResidentSource] {
  let budget = UInt64(ProcessInfo.processInfo.physicalMemory) / 2
  return try urls.map { url in
    let snapshot = try NativeANSSnapshot(url: url)
    return try MetalRuntimeANSResidentSource.load(
      snapshot: snapshot, device: device, maximumAdditionalBytes: budget)
  }
}

/// One moving annular detector, matching the gesture the native drive repeats.
func masks(shape: [Int], valid: [UInt8], steps: Int, inner: Double, outer: Double) -> [[UInt8]] {
  let rows = shape[2]
  let columns = shape[3]
  let centerRow = Double(rows - 1) / 2
  let centerColumn = Double(columns - 1) / 2
  return (0..<steps).map { step in
    // The native drive moves the center 12 degrees per input step (60 steps per
    // revolution) and repeats the revolution; match it so the per-update mask
    // delta is the same size the app produces.
    let angle = Double(step % 60) / 59 * Double.pi * 4
    let row = centerRow + 6 * sin(angle)
    let column = centerColumn + 6 * cos(angle)
    let radius = outer + 3 * sin(2 * angle)
    return (0..<rows * columns).map { pixel in
      let deltaRow = Double(pixel / columns) - row
      let deltaColumn = Double(pixel % columns) - column
      let distance = deltaRow * deltaRow + deltaColumn * deltaColumn
      return valid[pixel] != 0 && distance >= inner * inner && distance <= radius * radius ? 1 : 0
    }
  }
}

let arguments = CommandLine.arguments
guard arguments.count > 1 else {
  FileHandle.standardError.write(Data("usage: <qem file or folder> [tiles] [steps]\n".utf8))
  exit(2)
}
let tileLimit = arguments.count > 2 ? (Int(arguments[2]) ?? 1) : 1
let steps = arguments.count > 3 ? (Int(arguments[3]) ?? 12) : 12
let inner = arguments.count > 4 ? (Double(arguments[4]) ?? 40) : 40
let outer = arguments.count > 5 ? (Double(arguments[5]) ?? 80) : 80
guard tileLimit > 0, steps >= 2, inner >= 0, outer > inner else {
  fatalError("Choose positive tiles, at least two steps, and 0 <= inner < outer radii")
}
guard let device = MTLCreateSystemDefaultDevice() else {
  FileHandle.standardError.write(Data("no Metal device\n".utf8))
  exit(1)
}
let urls = try fixtureURLs(arguments[1], limit: tileLimit)
guard !urls.isEmpty else { fatalError("No .qem acquisitions found at the requested path") }
let sources = try load(urls, device: device)
defer {
  for source in sources { source.releaseResidentStorage() }
}
let shape = sources[0].shape
let gesture = masks(
  shape: shape, valid: sources[0].detectorValidityMask, steps: steps, inner: inner, outer: outer)
print(
  "QEM_UPDATE_FIXTURE files=\(urls.count) shape=\(shape) dtype=\(sources[0].logicalDtype) "
    + "resident_bytes=\(sources.reduce(UInt64(0)) { $0 + $1.residentBytes }) "
    + "steps=\(steps) ring=\(inner)-\(outer)")

if #available(macOS 15.0, *) {
  let single = try MetalRuntimeANSSeries(sources: sources)
  defer { single.release() }
  var singleDigests: [[String]] = []
  var wall: [Double] = []
  var gpu: [Double] = []
  var residuals: [Double] = []
  for (step, mask) in gesture.enumerated() {
    let update = try single.updatePriorityVirtualDetectorBuffer(mask: mask, priorityIndex: 0)
    residuals.append(Double(update.metrics.changedDetectorPixels))
    if step > 0 {
      wall.append(update.metrics.wallMilliseconds)
      gpu.append(update.metrics.gpuMilliseconds)
    }
    var frameDigests = [digest(update.buffer)]
    for index in sources.indices.dropFirst() {
      frameDigests.append(
        digest(
          try single.updatePriorityVirtualDetectorBuffer(
            mask: mask, priorityIndex: index
          ).buffer))
    }
    singleDigests.append(frameDigests)
  }
  print(
    "QEM_UPDATE_SINGLE wall_p50_ms=\(percentile(wall, 0.5)) wall_p95_ms=\(percentile(wall, 0.95)) "
      + "gpu_p50_ms=\(percentile(gpu, 0.5)) gpu_p95_ms=\(percentile(gpu, 0.95)) "
      + "residual_p50=\(percentile(residuals, 0.5)) decoded_pixels=\(Int(residuals.last ?? 0)) scans=\(shape[0] * shape[1]) decodes=\(Int(residuals.last ?? 0) * shape[0] * shape[1]) n=\(wall.count)"
  )

  // Series own their output/delta buffers; sources are read sequentially here.
  // Reuse resident payloads instead of doubling the seven-tilt memory footprint.
  let batched = try MetalRuntimeANSSeries(sources: sources)
  defer { batched.release() }
  var batchWall: [Double] = []
  var batchGPU: [Double] = []
  var batchDigests: [[String]] = []
  for (step, mask) in gesture.enumerated() {
    let update = try batched.updateVirtualDetectorBuffers(mask: mask)
    if step > 0 {
      batchWall.append(update.metrics.wallMilliseconds)
      batchGPU.append(update.metrics.gpuMilliseconds)
    }
    batchDigests.append(update.buffers.map(digest))
  }
  print(
    "QEM_UPDATE_BATCH wall_p50_ms=\(percentile(batchWall, 0.5)) wall_p95_ms=\(percentile(batchWall, 0.95)) "
      + "gpu_p50_ms=\(percentile(batchGPU, 0.5)) gpu_p95_ms=\(percentile(batchGPU, 0.95)) "
      + "tiles=\(sources.count) n=\(batchWall.count)")
  if batchDigests != singleDigests {
    FileHandle.standardError.write(
      Data("PARITY FAILURE: batched images differ from single-source\n".utf8))
    exit(1)
  }
  print("QEM_UPDATE_PARITY batch_matches_single=true")

  let last = gesture[gesture.count - 1]
  let reference = try single.updateVirtualDetectorBuffers(mask: last)
  let scanColumns = shape[1]
  for index in sources.indices {
    let output = reference.buffers[index].contents().assumingMemoryBound(to: UInt32.self)
    for (row, column) in [(shape[0] / 2, scanColumns / 2), (0, 0), (shape[0] - 1, scanColumns - 1)]
    {
      let pattern = try sources[index].extractRawDiffraction(scanRow: row, scanColumn: column)
      var expected: UInt32 = 0
      for pixel in 0..<pattern.count
      where last[pixel] == 1 && sources[index].detectorValidityMask[pixel] == 1 {
        expected += pattern[pixel]
      }
      guard output[row * scanColumns + column] == expected else {
        FileHandle.standardError.write(
          Data(
            "PARITY FAILURE at scan (\(row), \(column)): \(output[row * scanColumns + column]) != \(expected)\n"
              .utf8))
        exit(1)
      }
    }
  }
  print("QEM_UPDATE_PARITY raw_count_reference=true")
} else {
  FileHandle.standardError.write(Data("needs macOS 15\n".utf8))
  exit(1)
}
