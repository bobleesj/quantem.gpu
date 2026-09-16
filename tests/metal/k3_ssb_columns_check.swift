import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

/// Original K3 counts -> compressed columns -> explicitly zero-padded SSB input.
@main struct K3SSBColumnsCheck {
  static func main() throws {
    setbuf(stdout, nil)
    let args = CommandLine.arguments
    guard args.count == 4, let device = MTLCreateSystemDefaultDevice(),
      let queue = device.makeCommandQueue() else { fatalError("original.dm4 copy.qem evidence-directory") }
    let original = try NativeDM4Source(url: URL(fileURLWithPath: args[1]))
    print("original_data_offset=\(original.dataOffset)")
    let source = try MetalRuntimeANSResidentSource.load(
      snapshot: NativeANSSnapshot(url: URL(fileURLWithPath: args[2])), device: device)
    defer { source.releaseResidentStorage() }
    let size = source.shape[0] == 100 ? 128 : 256
    let nativeRows = source.shape[0], nativeCols = source.shape[1]
    let count = nativeRows * nativeCols, detector = source.shape[2] * source.shape[3]
    let selected = [0, 1, detector / 4, detector / 2, detector / 2 + source.shape[3] / 2, detector - 2, detector - 1, 0]
    let output = device.makeBuffer(length: size * size * selected.count * 4, options: .storageModeShared)!
    memset(output.contents(), 0xFF, output.length)
    let started = Date()
    let sums = try source.detectorColumnSums()
    print("detector_sum_seconds=\(Date().timeIntervalSince(started))")
    let command = queue.makeCommandBuffer()!
    try source.encodeDetectorColumns(pixels: selected, into: output, commands: command, scanRows: size, scanColumns: size)
    command.commit(); command.waitUntilCompleted()
    if let error = command.error { throw error }
    try source.validateDetectorColumnDecoding()
    let raw = try Data(contentsOf: original.url, options: .alwaysMapped)
    var mismatches = 0
    let values = output.contents().assumingMemoryBound(to: UInt32.self)
    raw.withUnsafeBytes { bytes in
      let samples = bytes.baseAddress!.advanced(by: original.dataOffset).assumingMemoryBound(to: UInt8.self)
      for (ordinal, pixel) in selected.enumerated() {
        var total: UInt64 = 0
        for row in 0..<size { for col in 0..<size {
          let expected: UInt32 = row < nativeRows && col < nativeCols && source.detectorValidityMask[pixel] != 0
            ? UInt32(samples[(row * nativeCols + col) * detector + pixel]) : 0
          if values[ordinal * size * size + row * size + col] != expected { mismatches += 1 }
          total += UInt64(expected)
        } }
        if total != sums[pixel] { mismatches += 1 }
      }
    }
    guard mismatches == 0 else { fatalError("\(mismatches) column/padding/sum mismatches") }
    let directory = URL(fileURLWithPath: args[3])
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    try sums.withUnsafeBytes { try Data($0).write(to: directory.appendingPathComponent("detector-sums.u64")) }
    print("PASS \(nativeRows)×\(nativeCols) -> \(size)×\(size), \(count) scan positions, original-count parity for \(selected.count) columns including duplicates; all padding zero; selected UInt64 totals exact; dtype=uint8")
  }
}
