import Foundation

/// A metadata attachment and supported calibration values awaiting user review.
/// Example: `try NativeMetadataImport.read(url, scanRows: 256, scanColumns: 256)`.
public struct NativeMetadataImport: Sendable {
  public let document: NativeMetadataDocument
  public let quantities: NativeQEMCalibration.Overrides

  public static func read(_ url: URL, scanRows: Int, scanColumns: Int) throws -> Self {
    let document = try NativeMetadataDocument.read(url)
    let evidence = "Attached \(document.filename); SHA-256 \(document.sha256)"
    let quantities: NativeQEMCalibration.Overrides
    if document.mediaType == "application/xml" {
      quantities = try NativeEMPADSource.metadataQuantities(
        document: document, rows: scanRows, columns: scanColumns, evidence: evidence)
    } else {
      let object = try JSONSerialization.jsonObject(with: Data(document.content.utf8)) as! [String: Any]
      let scientific = object["scientific_metadata"] as? [String: Any] ?? object
      if scientific["schema"] as? String == NativeQEMMetadataUnits.schema
        || scientific["schema"] as? String == NativeQEMMetadataUnits.legacySchema {
        let normalized = try NativeQEMMetadataUnits.normalized(scientific)
        if let axes = normalized["axes"] as? [[String: Any]], axes.count >= 2 {
          guard axes[0]["size"] as? Int == scanRows, axes[1]["size"] as? Int == scanColumns else {
            throw Native4DSTEMIOError.invalidData("Metadata scan dimensions do not match this acquisition.")
          }
        }
        let paths = Set([NativeQEMCalibration.scanRow, NativeQEMCalibration.scanColumn,
          NativeQEMCalibration.detectorRow, NativeQEMCalibration.detectorColumn,
          "electron_source/accelerating_voltage", "illumination_system/semi_convergence_angle",
          "scan_controller/regular_scan/dwell_time", "imaging_system/camera_length"])
        let recorded = normalized["electron_microscope"] as? [String: [String: Any]] ?? [:]
        let converted = try NativeQEMMetadataUnits.calculationOverrides([
          "schema": NativeQEMMetadataUnits.schema,
          "calibration_overrides": recorded.filter { paths.contains($0.key) },
        ])
        quantities = converted.reduce(into: [:]) { result, item in
          if let quantity = item.value as? [String: Any], let value = quantity["value"] as? Double,
            let unit = quantity["unit"] as? String {
            result[item.key] = .init(value: value, unit: unit, evidence: evidence)
          }
        }
      } else {
        // Unknown vendor JSON is preserved, not interpreted through guessed units.
        quantities = [:]
      }
    }
    try NativeQEMCalibration.validate(quantities)
    return Self(document: document, quantities: quantities)
  }
}
