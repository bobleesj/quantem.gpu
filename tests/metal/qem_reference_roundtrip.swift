import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

/// Independent original-count comparison, also usable for Python-written files.
@main struct QEMReferenceRoundtrip {
  static func main() throws {
    let args = CommandLine.arguments
    guard args.count >= 3, let device = MTLCreateSystemDefaultDevice() else {
      fatalError("Usage: qem-reference counts.npy copy.qem [--write]")
    }
    let source = try NativeNPYSource(url: URL(fileURLWithPath: args[1]))
    let output = URL(fileURLWithPath: args[2])
    if args.contains("--write") {
      let overrides: NativeQEMCalibration.Overrides =
        source.dataset.sourceDtype == "uint16"
        ? [
          NativeQEMCalibration.scanRow: .init(
            value: 0.4e-10, unit: "m", evidence: "synthetic reference"),
          NativeQEMCalibration.scanColumn: .init(
            value: 0.6e-10, unit: "m", evidence: "synthetic reference"),
          NativeQEMCalibration.detectorRow: .init(
            value: 0.2, unit: "1/Å", evidence: "synthetic reference"),
          NativeQEMCalibration.detectorColumn: .init(
            value: 0.3, unit: "1/Å", evidence: "synthetic reference"),
          "electron_source/accelerating_voltage": .init(
            value: 200000, unit: "V", evidence: "synthetic reference"),
        ] : [:]
      try MetalQEMExporter.save(
        .counts(source), to: output, device: device,
        calibrationOverrides: overrides)
    }
    let snapshot = try NativeANSSnapshot(url: output)
    if !args.contains("--write") {
      let manifestURL = source.url.deletingLastPathComponent().appendingPathComponent(
        "manifest.json")
      let manifest =
        try JSONSerialization.jsonObject(with: Data(contentsOf: manifestURL)) as! [String: Any]
      let entries = manifest["entries"] as! [[String: Any]]
      let entry = entries.first { $0["name"] as? String == source.dataset.sourceDtype }!
      let expectedMetadata = entry["scientific_metadata"] as! [String: Any]
      precondition(
        NSDictionary(dictionary: snapshot.scientificMetadata).isEqual(to: expectedMetadata),
        "Scientific metadata differs from the frozen reference")
    }
    let resident = try MetalRuntimeANSResidentSource.load(snapshot: snapshot, device: device)
    defer { resident.releaseResidentStorage() }
    precondition(resident.shape == source.shape)
    precondition(resident.dataset.sourceDtype == source.dataset.sourceDtype)
    let original = try Data(contentsOf: source.url)
    let pixels = source.shape[2] * source.shape[3]
    let width = source.dataset.sourceDtype == "uint8" ? 1 : 2
    for frame in 0..<(source.shape[0] * source.shape[1]) {
      let actual = try resident.extractRawDiffraction(
        scanRow: frame / source.shape[1], scanColumn: frame % source.shape[1])
      let expected: [UInt32] = original.withUnsafeBytes { bytes in
        (0..<pixels).map { pixel in
          let offset = source.dataOffset + (frame * pixels + pixel) * width
          return width == 1
            ? UInt32(bytes[offset])
            : UInt32(
              UInt16(littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self)))
        }
      }
      precondition(actual == expected, "Decoded counts differ at scan \(frame)")
    }
    print(
      "PASS native Metal: counts, shape, dtype and reference metadata for \(output.lastPathComponent)"
    )
  }
}
