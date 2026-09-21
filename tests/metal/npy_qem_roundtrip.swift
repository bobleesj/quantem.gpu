import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

@main struct NumPyQEMRoundtrip {
  static func run(_ url: URL, device: MTLDevice) throws {
    let source = try NativeNPYSource(url: url)
    let bytes = try Data(contentsOf: url, options: .mappedIfSafe)
    let shape = source.shape, pixels = shape[2] * shape[3], scans = shape[0] * shape[1]
    let width = source.dataset.sourceDtype == "uint8" ? 1 : 2
    let expected: [UInt32] = bytes.withUnsafeBytes { raw in
      (0..<scans * pixels).map { index in
        width == 1 ? UInt32(raw[source.dataOffset + index])
          : UInt32(UInt16(littleEndian: raw.loadUnaligned(fromByteOffset: source.dataOffset + index * 2, as: UInt16.self)))
      }
    }
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: folder) }
    let output = folder.appendingPathComponent("copy.qem")
    try MetalQEMExporter.save(.counts(source), to: output, device: device)
    let snapshot = try NativeANSSnapshot(url: output)
    let restored = try MetalRuntimeANSResidentSource.load(snapshot: snapshot, device: device)
    defer { restored.releaseResidentStorage() }
    let original = try MetalRuntimeANSResidentSource.load(array: source, device: device)
    precondition(original.residentBytes == restored.residentBytes,
      "Restoring a saved acquisition must retain the same spatial-index allocation policy")
    original.releaseResidentStorage()
    precondition(restored.shape == shape && restored.dataset.sourceDtype == source.dataset.sourceDtype)
    precondition(restored.dataset.sourceScanCalibration == nil)
    let header = try NativeQEMFile(url: output).header
    let scientific = header["scientific_metadata"] as! [String: Any]
    precondition(scientific["source_metadata"] as? [String: String] == source.dataset.metadata)
    for frame in 0..<scans {
      let actual = try restored.extractRawDiffraction(scanRow: frame / shape[1], scanColumn: frame % shape[1])
      precondition(actual == Array(expected[frame * pixels..<(frame + 1) * pixels]))
    }
    let series = try MetalRuntimeANSSeries(sources: [restored])
    defer { series.release() }
    for phase in 0..<3 {
      let mask: [UInt8] = (0..<pixels).map { UInt8(($0 + phase) % 3 == 0 ? 1 : 0) }
      let result = try series.updatePriorityVirtualDetectorBuffer(mask: mask, priorityIndex: 0)
      let values = result.buffer.contents().assumingMemoryBound(to: UInt32.self)
      for frame in 0..<scans {
        let sum = (0..<pixels).reduce(UInt32(0)) { $0 + (mask[$1] == 1 ? expected[frame * pixels + $1] : 0) }
        precondition(values[frame] == sum)
      }
    }
    print("PASS \(url.lastPathComponent): \(shape), \(source.dataset.sourceDtype), every DP and 3 detector masks exact; metadata preserved; no guessed calibration")
  }
  static func main() throws {
    precondition(CommandLine.arguments.count > 1, "Supply .npy paths and optional --reject= paths")
    guard let device = MTLCreateSystemDefaultDevice() else { fatalError("Metal device unavailable") }
    for path in CommandLine.arguments.dropFirst() {
      if path.hasPrefix("--reject=") {
        let input = String(path.dropFirst("--reject=".count))
        var rejected = false
        do { _ = try NativeNPYSource(url: URL(fileURLWithPath: input)) }
        catch { rejected = true; print("PASS rejected \(URL(fileURLWithPath: input).lastPathComponent): \(error.localizedDescription)") }
        precondition(rejected)
      } else { try run(URL(fileURLWithPath: path), device: device) }
    }
  }
}
