import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

/// Compare the shared conversion service with an independently retained export.
@main struct QEMExportServiceCheck {
  static func main() throws {
    precondition(CommandLine.arguments.count == 3, "Usage: qem-export-service original.xml reference.qem")
    let input = URL(fileURLWithPath: CommandLine.arguments[1])
    let reference = URL(fileURLWithPath: CommandLine.arguments[2])
    let folder = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: folder) }
    let output = folder.appendingPathComponent("copy.qem")
    let device = MTLCreateSystemDefaultDevice()!
    let source = try NativeEMPADSource.open(input)
    try MetalQEMExporter.save(.empad(source, background: nil, alreadyCorrected: false), to: output, device: device)
    let originalHash = SHA256.hash(data: try Data(contentsOf: reference, options: .mappedIfSafe))
    let savedHash = SHA256.hash(data: try Data(contentsOf: output, options: .mappedIfSafe))
    precondition(originalHash == savedHash, "The shared exporter changed the retained file's counts or metadata")
    var rejected = false
    do { try MetalQEMExporter.save(.empad(source, background: nil, alreadyCorrected: false), to: output, device: device) }
    catch { rejected = true }
    precondition(rejected, "An existing output must not be overwritten")
    let cancelled = folder.appendingPathComponent("cancelled.qem")
    do { try MetalQEMExporter.save(.empad(source, background: nil, alreadyCorrected: false), to: cancelled, device: device, shouldCancel: { true }) }
    catch { /* cancellation is expected */ }
    precondition(!FileManager.default.fileExists(atPath: cancelled.path))
    print("PASS: real EMPAD export matches retained file byte-for-byte; no overwrite; cancellation creates no output")
  }
}
