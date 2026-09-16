import Foundation

/// Shared scientific vocabulary, independent of the compressed payload codec.
/// Example: `try NativeQEMMetadata.acquisition(dataset)` before writing a QEM header.
public enum NativeQEMMetadata {
  public static let magic = Data("QEMDATA1".utf8)
  public static let container = "quantem.qem"

  public static func acquisition(_ dataset: Native4DSTEMDataset) throws -> [String: Any] {
    let original = dataset.metadata ?? [:]
    if let saved = original[NativeQEMMetadataUnits.metadataKey],
      let scientific = try JSONSerialization.jsonObject(with: Data(saved.utf8)) as? [String: Any]
    {
      try NativeQEMMetadataUnits.validateScientific(scientific)
      return try NativeQEMMetadataUnits.normalized(scientific)
    }
    let microscope = NativeMicroscopeMetadata(metadata: original)
    var quantities = [String: Any]()
    func quantity(_ path: String, _ value: Double?, _ unit: String) {
      guard let value, value.isFinite, value > 0 else { return }
      quantities[path] = ["value": value, "unit": unit, "provenance": "source_metadata"]
    }
    quantity(
      "electron_source/accelerating_voltage", microscope.beamEnergyKeV.map { $0 * 1000 }, "V")
    quantity(
      "illumination_system/semi_convergence_angle", microscope.semiConvergenceAngleMrad, "mrad")
    quantity(
      "scan_controller/regular_scan/dwell_time", microscope.dwellTimeMicroseconds.map { $0 * 1e-6 },
      "s")
    quantity(
      "imaging_system/camera_length", microscope.cameraLengthMillimeters.map { $0 * 1e-3 }, "m")
    quantity("imaging_system/reciprocal_pixel_size_y", microscope.angularRowMrad, "mrad")
    quantity("imaging_system/reciprocal_pixel_size_x", microscope.angularColumnMrad, "mrad")
    var axes: [[String: Any]] = zip(
      ["scan_row", "scan_column", "detector_row", "detector_column"],
      [dataset.scanRows, dataset.scanCols, dataset.detectorRows, dataset.detectorCols]
    ).map { ["name": $0.0, "size": $0.1] }
    if let scan = dataset.sourceScanCalibration {
      for (axis, step) in [scan.rowSamplingAngstrom, scan.columnSamplingAngstrom].enumerated() {
        axes[axis]["sampling"] =
          [
            "value": step * 1e-10, "unit": "m",
            "provenance": scan.origin.rawValue, "evidence": scan.evidence,
          ] as [String: Any]
      }
      quantity("scan_controller/regular_scan/pixel_size_y", scan.rowSamplingAngstrom * 1e-10, "m")
      quantity(
        "scan_controller/regular_scan/pixel_size_x", scan.columnSamplingAngstrom * 1e-10, "m")
    }
    if let unit = dataset.kPixelUnit, let row = dataset.kPixelSizeRow,
      let col = dataset.kPixelSizeCol,
      row.isFinite, col.isFinite, row > 0, col > 0
    {
      for (axis, step) in [row, col].enumerated() {
        axes[axis + 2]["sampling"] = ["value": step, "unit": unit, "provenance": "source_metadata"]
      }
    }
    return try NativeQEMMetadataUnits.normalized([
      "schema": "quantem.scientific-metadata/1", "axes": axes,
      "electron_microscope": quantities, "source_metadata": original,
      "source_metadata_coverage": "reader-retained",
      "calibration_overrides": [:] as [String: String],
      "processing": [["operation": "lossless_storage", "changes_measurements": false]],
      "source_format": original["sourceFormat"] ?? dataset.schemaIdentity ?? "unknown",
    ])
  }

  /// Reject incompatible envelopes without interpreting codec bytes.
  public static func validate(_ header: [String: Any], shape: [Int]) throws {
    guard header["container"] as? String == container,
      header["container_version"] as? Int == 1,
      header["codec"] as? String == header["profile"] as? String,
      let scientific = header["scientific_metadata"] as? [String: Any],
      let schema = scientific["schema"] as? String,
      [NativeQEMMetadataUnits.legacySchema, NativeQEMMetadataUnits.schema].contains(schema),
      let axes = scientific["axes"] as? [[String: Any]],
      axes.compactMap({ $0["size"] as? Int }) == shape,
      axes.compactMap({ $0["name"] as? String }) == [
        "scan_row", "scan_column", "detector_row", "detector_column",
      ]
    else {
      throw Native4DSTEMIOError.invalidData(
        "Unsupported or inconsistent QEM metadata. Update the reader or re-export the original acquisition."
      )
    }
    if schema == NativeQEMMetadataUnits.schema {
      let canonical = try NativeQEMMetadataUnits.normalized(scientific)
      guard NSDictionary(dictionary: canonical).isEqual(to: scientific) else {
        throw Native4DSTEMIOError.invalidData("Schema-2 QEM quantities require microscopy units.")
      }
    }
    try NativeQEMMetadataUnits.validateScientific(scientific)
    _ = try NativeQEMCalibration.read(scientific: scientific)
  }
}
