import Foundation
import Metal
import Metal4DSTEMStreamingIO

private func usage() -> Never {
  fputs(
    "Usage: metal-ans-file-benchmark SOURCE [--row N] [--column N] "
      + "[--expected-sha256 HEX] [--no-checksums] [--summary]\n",
    stderr)
  exit(2)
}

@main
enum MetalANSFileBenchmark {
  static func main() throws {
    var arguments = Array(CommandLine.arguments.dropFirst())
    guard let sourceArgument = arguments.first else { usage() }
    arguments.removeFirst()
    var row = 0
    var column = 0
    var expectedSHA256: String?
    var verifyChecksums = true
    var summaryOnly = false
    while !arguments.isEmpty {
      let flag = arguments.removeFirst()
      if flag == "--no-checksums" {
        verifyChecksums = false
        continue
      }
      if flag == "--summary" {
        summaryOnly = true
        continue
      }
      guard let value = arguments.first else { usage() }
      arguments.removeFirst()
      switch flag {
      case "--row":
        guard let parsed = Int(value) else { usage() }
        row = parsed
      case "--column":
        guard let parsed = Int(value) else { usage() }
        column = parsed
      case "--expected-sha256":
        expectedSHA256 = value
      default: usage()
      }
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw NSError(domain: "MetalANSFileBenchmark", code: 1, userInfo: [
        NSLocalizedDescriptionKey: "No physical Metal device is available."
      ])
    }
    let started = ContinuousClock.now
    let source = try MetalANSResidentSource(
      sourceURL: URL(fileURLWithPath: sourceArgument), device: device,
      expectedSHA256: expectedSHA256, verifyChecksums: verifyChecksums)
    let loadMilliseconds = milliseconds(from: started)
    let values = try source.extractRawDiffraction(scanRow: row, scanColumn: column)
    var output: [String: Any] = [
      "shape": source.shape,
      "dtype": source.logicalDtype.rawValue,
      "block_frames": source.blockFrames,
      "scale": source.scale,
      "resident_bytes": source.residentBytes,
      "load_ms": loadMilliseconds,
      "checksums_verified": verifyChecksums,
    ]
    if summaryOnly {
      output["selected_count"] = values.count
      output["selected_max"] = values.max() ?? 0
    } else {
      output["selected_values"] = values
    }
    let data = try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys])
    print(String(decoding: data, as: UTF8.self))
    source.releaseResidentStorage()
  }

  private static func milliseconds(from start: ContinuousClock.Instant) -> Double {
    let duration = start.duration(to: .now)
    return Double(duration.components.seconds) * 1_000
      + Double(duration.components.attoseconds) / 1.0e15
  }
}
