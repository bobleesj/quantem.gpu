import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

/// Exact bounded-column timing for the preparation path used by Direct Ptycho.
/// Call with `--columns acquisition.qem selected-pixels.json repetitions`.
/// Every returned count is hashed outside synchronized GPU timing.
func benchmarkColumns(_ arguments: [String]) throws {
  guard arguments.count == 5, let repetitions = Int(arguments[4]), repetitions > 0 else {
    throw NSError(
      domain: "ColumnBenchmark", code: 1,
      userInfo: [
        NSLocalizedDescriptionKey: "Use --columns acquisition.qem selected-pixels.json repetitions"
      ])
  }
  func failure(_ message: String) -> NSError {
    NSError(
      domain: "ColumnBenchmark", code: 4,
      userInfo: [NSLocalizedDescriptionKey: message])
  }
  guard let device = MTLCreateSystemDefaultDevice() else {
    throw failure(
      "A physical Metal device is required. Run this benchmark on an Apple-silicon Mac.")
  }
  let pixels = try JSONDecoder().decode(
    [Int].self, from: Data(contentsOf: URL(fileURLWithPath: arguments[3])))
  let snapshot = try NativeANSSnapshot(url: URL(fileURLWithPath: arguments[2]))
  let source = try MetalRuntimeANSResidentSource.load(snapshot: snapshot, device: device)
  defer { source.releaseResidentStorage() }
  let shape = source.shape
  let plane = shape[0] * shape[1]
  guard !pixels.isEmpty, pixels.allSatisfy({ 0..<shape[2] * shape[3] ~= $0 }) else {
    throw NSError(
      domain: "ColumnBenchmark", code: 2,
      userInfo: [NSLocalizedDescriptionKey: "Select detector pixel indices within the source shape"]
    )
  }
  guard let output = device.makeBuffer(length: 32 * plane * 4, options: .storageModeShared),
    let queue = device.makeCommandQueue()
  else {
    throw failure(
      "Cannot allocate the bounded column workspace. Close inactive datasets and retry.")
  }
  let samples = [0, plane / 2, plane - 1]
  let raw = try samples.map {
    try source.extractRawDiffraction(scanRow: $0 / shape[1], scanColumn: $0 % shape[1])
  }
  let validity = source.detectorValidityMask
  for repetition in 0..<repetitions {
    var wall = 0.0
    var gpu = 0.0
    var hashes: [String] = []
    for first in stride(from: 0, to: pixels.count, by: 32) {
      let selected = Array(pixels[first..<min(first + 32, pixels.count)])
      let begin = CFAbsoluteTimeGetCurrent()
      guard let commands = queue.makeCommandBuffer() else {
        throw failure(
          "Cannot create the column command buffer. Retry after other GPU work finishes.")
      }
      try source.encodeDetectorColumns(
        pixels: selected, into: output, commands: commands,
        scanRows: shape[0], scanColumns: shape[1])
      commands.commit()
      commands.waitUntilCompleted()
      if let error = commands.error { throw error }
      try source.validateDetectorColumnDecoding()
      guard commands.gpuStartTime > 0, commands.gpuEndTime >= commands.gpuStartTime else {
        throw failure(
          "The device supplied no completed GPU timing. This run cannot qualify performance.")
      }
      wall += CFAbsoluteTimeGetCurrent() - begin
      gpu += commands.gpuEndTime - commands.gpuStartTime
      let data = Data(
        bytesNoCopy: output.contents(), count: selected.count * plane * 4, deallocator: .none)
      hashes.append(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
      let actual = output.contents().assumingMemoryBound(to: UInt32.self)
      for (slot, pixel) in selected.enumerated() {
        for (index, scan) in samples.enumerated() {
          let expected = validity[pixel] == 0 ? UInt32(0) : raw[index][pixel]
          guard actual[slot * plane + scan] == expected else {
            throw NSError(
              domain: "ColumnBenchmark", code: 3,
              userInfo: [
                NSLocalizedDescriptionKey:
                  "Column counts differ from the original diffraction pattern"
              ])
          }
        }
      }
    }
    let row: [String: Any] = [
      "repeat": repetition, "device": device.name, "shape": shape,
      "columns": pixels.count, "wall_seconds": wall, "gpu_seconds": gpu,
      "column_batch_sha256": hashes, "raw_sample_parity": true,
      "resident_bytes": source.residentBytes, "allocated_bytes": device.currentAllocatedSize,
    ]
    let data = try JSONSerialization.data(withJSONObject: row, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
  }
}
