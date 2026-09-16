import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

@main struct QEMCalibrationRoundtrip {
  static func mutate(_ value: Any, path: ArraySlice<Any>, replacement: Any?) -> Any {
    guard let key = path.first else { return replacement as Any }
    if let index = key as? Int {
      var array = value as! [Any]
      array[index] = mutate(array[index], path: path.dropFirst(), replacement: replacement)
      return array
    }
    var object = value as! [String: Any]
    let name = key as! String
    if path.count == 1 {
      object[name] = replacement
    } else {
      object[name] = mutate(object[name]!, path: path.dropFirst(), replacement: replacement)
    }
    return object
  }

  static func main() throws {
    guard CommandLine.arguments.count == 3, let device = MTLCreateSystemDefaultDevice() else {
      fatalError("Usage: qem-calibration-roundtrip counts.npy new-copy.qem")
    }
    let original = try NativeNPYSource(url: URL(fileURLWithPath: CommandLine.arguments[1]))
    precondition(original.dataset.sourceDtype == "uint16", "This exact-count fixture requires uint16 input")
    let output = URL(fileURLWithPath: CommandLine.arguments[2])
    let fixtures = URL(fileURLWithPath: "tests/data/qem-v2")
    let reference = try NativeQEMFile(url: fixtures.appendingPathComponent("u16-multiple-chunks.qem"))
    let invalidCases = try JSONSerialization.jsonObject(with:
      Data(contentsOf: fixtures.appendingPathComponent("invalid-metadata.json"))) as! [[String: Any]]
    for item in invalidCases {
      let changed = mutate(reference.header["scientific_metadata"]!,
        path: (item["path"] as! [Any])[...], replacement: item["value"]) as! [String: Any]
      var rejected = false
      do { try NativeQEMMetadataUnits.validateScientific(changed) } catch { rejected = true }
      precondition(rejected, "Shared invalid metadata accepted: \(item["name"]!)")
    }
    for invalid in [true, "300", -1.0] as [Any] {
      var rejected = false
      do {
        _ = try NativeQEMMetadataUnits.normalized([
          "schema": NativeQEMMetadataUnits.schema,
          "electron_microscope": ["electron_source/accelerating_voltage": [
            "value": invalid, "unit": "kV",
          ]],
        ])
      } catch { rejected = true }
      precondition(rejected, "Invalid physical quantity accepted")
    }
    let overrides: NativeQEMCalibration.Overrides = [
      NativeQEMCalibration.scanRow: .init(value: 0.45e-10, unit: "m", evidence: "measured scan row"),
      NativeQEMCalibration.scanColumn: .init(value: 0.55e-10, unit: "m", evidence: "measured scan column"),
      NativeQEMCalibration.detectorRow: .init(value: 0.31, unit: "1/Å", evidence: "diffraction standard"),
      NativeQEMCalibration.detectorColumn: .init(value: 0.32, unit: "1/Å", evidence: "diffraction standard"),
      "electron_source/accelerating_voltage": .init(value: 200000, unit: "V", evidence: "microscope setting"),
      "illumination_system/semi_convergence_angle": .init(value: 24.75, unit: "mrad", evidence: "aperture calibration"),
      "scan_controller/regular_scan/dwell_time": .init(value: 50e-6, unit: "s", evidence: "scan timing"),
      "imaging_system/camera_length": .init(value: 0.23, unit: "m", evidence: "camera setting"),
    ]
    var resolvedIdentity: MetalQEMExporter.CalibrationIdentity?
    try MetalQEMExporter.save(.counts(original), to: output, device: device,
      resolveCalibration: { identity in
        precondition(resolvedIdentity == nil, "Resolve once after the normal load")
        resolvedIdentity = identity
        return overrides
      })
    let snapshot = try NativeANSSnapshot(url: output)
    precondition(snapshot.scientificMetadata["schema"] as? String == "quantem.scientific-metadata/2")
    let saved = snapshot.scientificMetadata["calibration_overrides"] as! [String: [String: Any]]
    let expectedUnits = [
      NativeQEMCalibration.scanRow: "angstrom", NativeQEMCalibration.scanColumn: "angstrom",
      NativeQEMCalibration.detectorRow: "1/angstrom", NativeQEMCalibration.detectorColumn: "1/angstrom",
      "electron_source/accelerating_voltage": "kV",
      "illumination_system/semi_convergence_angle": "mrad",
      "scan_controller/regular_scan/dwell_time": "us", "imaging_system/camera_length": "mm",
    ]
    let expectedValues = [
      NativeQEMCalibration.scanRow: 0.45, NativeQEMCalibration.scanColumn: 0.55,
      NativeQEMCalibration.detectorRow: 0.31, NativeQEMCalibration.detectorColumn: 0.32,
      "electron_source/accelerating_voltage": 200.0,
      "illumination_system/semi_convergence_angle": 24.75,
      "scan_controller/regular_scan/dwell_time": 50.0, "imaging_system/camera_length": 230.0,
    ]
    for (path, value) in expectedValues {
      precondition(saved[path]?["unit"] as? String == expectedUnits[path])
      precondition(abs((saved[path]?["value"] as! Double) - value) < abs(value) * 1e-14)
    }
    precondition(resolvedIdentity?.sourceIdentitySHA256 == original.dataset.sourceIdentitySHA256,
      "Resolve calibration using the acquisition identity, not the new file header hash")
    precondition(resolvedIdentity?.originalSourceIdentitySHA256 == original.dataset.sourceIdentitySHA256)
    let restored = try NativeQEMCalibration.read(metadata: snapshot.dataset.metadata ?? [:])
    for (path, expected) in overrides {
      let actual = restored[path]!
      precondition(actual.unit == expected.unit && actual.evidence == expected.evidence)
      precondition(abs(actual.value - expected.value) < abs(expected.value) * 1e-14,
        "Physical calibration must survive without local preferences")
    }
    precondition(snapshot.dataset.sourceScanCalibration == original.dataset.sourceScanCalibration,
      "User calibration must not replace recorded calibration")
    let resident = try MetalRuntimeANSResidentSource.load(snapshot: snapshot, device: device)
    defer { resident.releaseResidentStorage() }
    let expected = try Data(contentsOf: original.url)
    for frame in 0..<(original.dataset.scanRows * original.dataset.scanCols) {
      let raw = try resident.extractRawDiffraction(scanRow: frame / original.dataset.scanCols, scanColumn: frame % original.dataset.scanCols)
      let pixels = original.dataset.detectorRows * original.dataset.detectorCols
      let values = expected.withUnsafeBytes { pointer in
        (0..<pixels).map { UInt32(pointer.loadUnaligned(fromByteOffset: original.dataOffset + (frame * pixels + $0) * 2, as: UInt16.self)) }
      }
      precondition(raw == values, "Saving calibration changed a measured count")
    }
    let preserved = output.deletingPathExtension().appendingPathExtension("preserved.qem")
    let reset = output.deletingPathExtension().appendingPathExtension("reset.qem")
    try resident.saveSnapshot(to: preserved)
    try MetalQEMExporter.save(.counts(original), to: reset, device: device,
      calibrationOverrides: [:], resolveCalibration: { _ in
        preconditionFailure("Explicit queued calibration must take priority over saved edits")
      })
    let preservedSnapshot = try NativeANSSnapshot(url: preserved)
    let resetSnapshot = try NativeANSSnapshot(url: reset)
    let preservedValues = try NativeQEMCalibration.read(metadata: preservedSnapshot.dataset.metadata ?? [:])
    let resetValues = try NativeQEMCalibration.read(metadata: resetSnapshot.dataset.metadata ?? [:])
    precondition(Set(preservedValues.keys) == Set(restored.keys))
    for (path, expected) in restored {
      let actual = preservedValues[path]!
      precondition(actual.unit == expected.unit && actual.evidence == expected.evidence
        && actual.provenance == expected.provenance)
      // Unit conversion can round the final Double bit; measured counts stay exact.
      precondition(abs(actual.value - expected.value) < abs(expected.value) * 1e-14,
        "Resaving changed the physical calibration at \(path)")
    }
    precondition(resetValues.isEmpty)
    let file = try NativeQEMFile(url: output)
    var independent = file.header
    var record = independent["scientific_metadata"] as! [String: Any]
    var axes = record["axes"] as! [[String: Any]]
    for index in 0..<2 {
      axes[index]["sampling"] = ["value": Double(index + 1), "unit": "angstrom",
        "provenance": "synthetic_reference"]
    }
    record["axes"] = axes
    record["electron_microscope"] = ["electron_source/accelerating_voltage": [
      "value": 300.0, "unit": "kV", "provenance": "synthetic_reference",
    ]]
    independent["scientific_metadata"] = record
    let originalBytes = try Data(contentsOf: output)
    for stale in [false, true] {
      independent["metadata"] = stale
        ? ["scan_sampling_A": [8.0, 9.0], "voltage_kV": 80.0] : [:]
      let blob = try JSONSerialization.data(withJSONObject: independent, options: [.sortedKeys])
      var bytes = NativeQEMMetadata.magic
      for size in [blob.count, 56 + blob.count] {
        var value = UInt64(size).littleEndian
        withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
      }
      bytes.append(contentsOf: SHA256.hash(data: blob))
      bytes.append(blob)
      bytes.append(originalBytes.suffix(from: file.bodyStart))
      let path = output.deletingPathExtension().appendingPathExtension("public-\(stale).qem")
      try bytes.write(to: path)
      let checked = try NativeANSSnapshot(url: path)
      precondition(checked.dataset.sourceScanCalibration?.rowSamplingAngstrom == 1)
      precondition(checked.dataset.sourceScanCalibration?.columnSamplingAngstrom == 2)
      precondition(NativeMicroscopeMetadata(metadata: checked.dataset.metadata ?? [:]).beamEnergyKeV == 300)
    }
    var contradictory = record
    var microscope = record["electron_microscope"] as! [String: Any]
    microscope[NativeQEMCalibration.scanRow] = ["value": 99.0, "unit": "angstrom", "provenance": "synthetic_reference"]
    contradictory["electron_microscope"] = microscope
    var rejected = false
    do { try NativeQEMMetadataUnits.validateScientific(contradictory) } catch { rejected = true }
    precondition(rejected, "Contradictory public calibration accepted")
    print("PASS: microscopy units, physical calibration, late-bound identity, every raw DP exact, preserve and explicit clear")
  }
}
