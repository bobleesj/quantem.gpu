import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

/// Exact `.qem` round trips for calibrated scaled-uint16 results (for example merged tilts).
/// Restored float32 bits, regional calibration and provenance must survive unchanged,
/// saving must never re-encode, and every other reader must refuse the codes.
@main
struct ScaledQEMRoundtrip {
  static func require(_ condition: Bool, _ message: String) throws {
    if !condition { throw Native4DSTEMIOError.invalidData(message) }
  }

  static func expectFailure(_ label: String, _ body: () throws -> Void) throws {
    do { try body() } catch {
      print("PASS refused \(label): \(error.localizedDescription)")
      return
    }
    throw Native4DSTEMIOError.invalidData("\(label) was accepted")
  }

  /// Deterministic float32 frames: zeros, negatives, ties, a constant region and large values.
  static func values(region: Int, frames: Int, pixels: Int) -> [Float] {
    var state = UInt64(0x9E37_79B9_7F4A_7C15) &+ UInt64(region)
    return (0..<(frames * pixels)).map { index in
      state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
      let unit = Float(state >> 40) / Float(1 << 24)
      switch region {
      case 0: return index % 7 == 0 ? 0 : unit * 1716.42
      case 1: return (unit - 0.35) * 40
      case 2: return 3.25
      default: return index % 3 == 0 ? 0 : unit * unit * 9.0e4
      }
    }
  }

  static func restored(_ source: MetalPackedSource) throws -> Data {
    let frames = source.shape[0] * source.shape[1]
    let pixels = source.shape[2] * source.shape[3]
    var result = Data()
    for first in stride(from: 0, to: frames, by: 4096) {
      let range = first..<min(frames, first + 4096)
      let buffer = try source.read(range)
      result.append(Data(bytes: buffer.contents(), count: range.count * pixels * 4))
    }
    return result
  }

  static func scientific(_ shape: [Int], sampling: Bool) -> [String: Any] {
    var axes: [[String: Any]] = zip(
      ["scan_row", "scan_column", "detector_row", "detector_column"], shape
    ).map { ["name": $0.0, "size": $0.1] }
    if sampling {
      for index in 0..<2 {
        axes[index]["sampling"] = ["value": 2.5, "unit": "angstrom", "provenance": "test scan"]
        axes[index + 2]["sampling"] = ["value": 0.5, "unit": "mrad", "provenance": "test detector"]
      }
    }
    return [
      "schema": NativeQEMMetadataUnits.schema, "axes": axes,
      "electron_microscope": [String: Any](), "calibration_overrides": [String: Any](),
      "source_metadata": ["derivation": "synthetic regional test"],
      "source_metadata_coverage": "unknown", "source_format": "quantem.scaled-uint16-test",
      "processing": [
        ["operation": "synthetic_regions", "changes_measurements": true],
        ["operation": MetalPackedSource.quantizationOperation, "changes_measurements": true],
      ],
    ]
  }

  static func main() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw Native4DSTEMIOError.invalidData("No Metal device")
    }
    let root = FileManager.default.temporaryDirectory
      .appendingPathComponent("scaled-qem-\(UUID().uuidString)", isDirectory: true)
    try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: root) }

    // Four calibrated regions, one part each, exactly like the native merge writes them.
    let shape = [8, 8, 64, 64]
    let pixels = shape[2] * shape[3]
    let original = try MetalPackedSource(
      shape: shape, precision: try MetalPrecision(device: device))
    for region in 0..<4 {
      let frames = 16
      var data = values(region: region, frames: frames, pixels: pixels)
      let buffer = device.makeBuffer(
        bytes: &data, length: data.count * 4, options: .storageModeShared)!
      let calibration = try MetalPrecision(device: device)
      try calibration.includeRange(buffer, count: data.count)
      try calibration.calibrate(shape: [2, shape[1], shape[2], shape[3]])
      let codes = try calibration.convert(buffer, count: data.count)
      try calibration.finish()
      try original.append(codes, frames: frames, calibration: calibration)
    }
    original.attributes = ["quantem_maped_merge_v1": "{\"version\":1}", "note": "é provenance ✓"]
    let reference = try restored(original)
    let referenceMetadata = original.metadata

    // 1. Save and reopen: identical restored bits, calibration and provenance.
    let first = root.appendingPathComponent("merged.qem")
    try original.saveQEM(to: first, scientificMetadata: scientific(shape, sampling: true))
    try require(MetalPackedSource.isScaledQEM(first), "Saved file is not detected as scaled")
    let reopened = try MetalPackedSource.loadQEM(url: first, device: device)
    try require(try restored(reopened) == reference, "Reopened float32 bits changed")
    try require(reopened.attributes == original.attributes, "Provenance changed")
    try require(
      NSDictionary(dictionary: reopened.qemScientificMetadata ?? [:]).isEqual(
        to: scientific(shape, sampling: true)), "Public metadata changed")
    let before = referenceMetadata["regions"] as! [[String: Any]]
    let after = reopened.metadata["regions"] as! [[String: Any]]
    for (a, b) in zip(before, after) {
      for key in ["scale", "offset", "first_frame", "stop_frame"] {
        try require(
          (a[key] as! NSNumber).doubleValue.bitPattern
            == (b[key] as! NSNumber).doubleValue.bitPattern,
          "Region \(key) changed")
      }
    }
    print("PASS save/reopen restored \(reference.count / 4) float32 values bit for bit")

    // 2. Saving a reopened result writes byte-identical code streams.
    let second = root.appendingPathComponent("merged-again.qem")
    try reopened.saveQEM(to: second, scientificMetadata: scientific(shape, sampling: true))
    let firstFile = try NativeQEMFile(url: first)
    let secondFile = try NativeQEMFile(url: second)
    try require(
      firstFile.header["sha256"] as! [String] == secondFile.header["sha256"] as! [String],
      "Re-saved body checksums differ")
    try require(firstFile.bodyBytes == secondFile.bodyBytes, "Re-saved body size differs")
    print("PASS re-saved body identical (\(firstFile.bodyBytes) bytes)")

    // 3. HDF5 export → reopen (split parts, saved report) → .qem → reopen.
    let h5 = root.appendingPathComponent("merged_master.h5")
    try original.save(to: h5)
    let fromH5 = try MetalPackedSource.load(
      path: h5, device: device, indexDirectory: root.appendingPathComponent("index"))
    let third = root.appendingPathComponent("from-h5.qem")
    try fromH5.saveQEM(to: third, scientificMetadata: scientific(shape, sampling: false))
    let viaH5 = try MetalPackedSource.loadQEM(url: third, device: device)
    try require(try restored(viaH5) == reference, "HDF5 → .qem changed restored bits")
    print("PASS HDF5-reopened result → .qem → reopen bit for bit")

    // 4. Refusals: nothing may be overwritten or written with incomplete metadata.
    try expectFailure("overwrite") {
      try original.saveQEM(to: first, scientificMetadata: scientific(shape, sampling: true))
    }
    try expectFailure("non-.qem destination") {
      try original.saveQEM(
        to: root.appendingPathComponent("x.h5"),
        scientificMetadata: scientific(shape, sampling: true))
    }
    var undeclared = scientific(shape, sampling: true)
    undeclared["processing"] = [["operation": "synthetic_regions", "changes_measurements": true]]
    let missing = root.appendingPathComponent("undeclared.qem")
    try expectFailure("missing quantization record") {
      try original.saveQEM(to: missing, scientificMetadata: undeclared)
    }
    try require(!FileManager.default.fileExists(atPath: missing.path), "Refused save left a file")

    // 5. Tampering and truncation are detected before any GPU use.
    var bytes = try Data(contentsOf: first)
    bytes[firstFile.bodyStart + 17] ^= 0x40
    let corrupted = root.appendingPathComponent("corrupted.qem")
    try bytes.write(to: corrupted)
    try expectFailure("corrupted body") {
      _ = try MetalPackedSource.loadQEM(url: corrupted, device: device)
    }
    let truncated = root.appendingPathComponent("truncated.qem")
    try bytes.prefix(bytes.count - 9).write(to: truncated)
    try expectFailure("truncated file") {
      _ = try MetalPackedSource.loadQEM(url: truncated, device: device)
    }

    // 6. Codes are never readable as counts or float measurements.
    try expectFailure("integer count reader") { _ = try NativeANSSnapshot(url: first) }
    try expectFailure("float32 reader") { _ = try NativeEMPADSource.openQEM(first) }
    let integer = URL(
      fileURLWithPath: CommandLine.arguments.dropFirst().first
        ?? "tests/data/qem-v2/u16-multiple-chunks.qem")
    try require(!MetalPackedSource.isScaledQEM(integer), "Integer .qem detected as scaled")
    try expectFailure("scaled reader on integer counts") {
      _ = try MetalPackedSource.loadQEM(url: integer, device: device)
    }
    print("PASS scaled-uint16 .qem round trip and refusals; device=\(device.name)")
  }
}
