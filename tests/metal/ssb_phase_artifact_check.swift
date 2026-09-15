import Foundation
import MetalSSBKernels

/// Check a real exported phase without constructing a scientific solver.
@main
struct SSBPhaseArtifactCheck {
  static func main() throws {
    let url = URL(fileURLWithPath: CommandLine.arguments[1])
    let saved = try SSBPhaseArtifact.load(from: url)
    let values = saved.phaseValues()
    let restored = values.withUnsafeBytes { Data($0) }
    precondition(SSBPhaseArtifact.digest(restored) == saved.phaseSHA256)
    var rejectedMismatch = false
    do { _ = try SSBPhaseArtifact.load(from: url, matchingSourceIdentity: String(repeating: "0", count: 64)) }
    catch { rejectedMismatch = true }
    precondition(rejectedMismatch)
    let directory = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: directory) }
    let pair = directory.appendingPathComponent("roundtrip.json")
    try saved.savePair(to: pair)
    try saved.savePair(to: pair)
    let roundtrip = try SSBPhaseArtifact.load(from: pair)
    precondition(roundtrip.phase == saved.phase)
    let sameResult = try roundtrip.contentDigest() == saved.contentDigest()
    precondition(sameResult)
    var object = try JSONSerialization.jsonObject(with: JSONEncoder().encode(saved)) as! [String: Any]
    var phase = saved.phase
    phase[0] ^= 1
    object["phase"] = phase.base64EncodedString()
    let corrupt = directory.appendingPathComponent("corrupt.ssbresult")
    try JSONSerialization.data(withJSONObject: object).write(to: corrupt)
    var rejectedCorruption = false
    do { _ = try SSBPhaseArtifact.load(from: corrupt) }
    catch { rejectedCorruption = true }
    precondition(rejectedCorruption)
    var bytes = try Data(contentsOf: pair.deletingPathExtension().appendingPathExtension("npy"))
    bytes[bytes.count - 1] ^= 1
    try bytes.write(to: pair.deletingPathExtension().appendingPathExtension("npy"))
    var refusedOverwrite = false
    do { try saved.savePair(to: pair) } catch { refusedOverwrite = true }
    precondition(refusedOverwrite)
    let preservedBytes = try Data(contentsOf: pair.deletingPathExtension().appendingPathExtension("npy"))
    precondition(preservedBytes == bytes)
    var rejectedPairCorruption = false
    do { _ = try SSBPhaseArtifact.load(from: pair) } catch { rejectedPairCorruption = true }
    precondition(rejectedPairCorruption)
    print("SSB_PHASE_IMPORT_PASS exact_phase_bits=true wrong_source_rejected=true corruption_rejected=true pair_roundtrip=true")
  }
}
