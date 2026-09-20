import Foundation

/// Explicit user quantities, kept separate from recorded microscope metadata.
/// Example: save a value in volts under `electron_source/accelerating_voltage`.
public struct NativeQEMCalibrationQuantity: Codable, Equatable, Hashable, Sendable {
  public let value: Double
  public let unit: String
  public let provenance: String
  public let evidence: String

  public init(value: Double, unit: String, evidence: String) {
    self.value = value
    self.unit = unit
    provenance = "user_override"
    self.evidence = evidence
  }
}

/// Portable overrides use the same microscope paths as recorded quantities.
public enum NativeQEMCalibration {
  public typealias Overrides = [String: NativeQEMCalibrationQuantity]
  public static let metadataKey = "qem_calibration_overrides"
  public static let scanRow = "scan_controller/regular_scan/pixel_size_row"
  public static let scanColumn = "scan_controller/regular_scan/pixel_size_column"
  public static let detectorRow = "imaging_system/reciprocal_pixel_size_row"
  public static let detectorColumn = "imaging_system/reciprocal_pixel_size_column"

  /// Validate scientific units before saving or interpreting a portable override.
  public static func validate(_ overrides: Overrides) throws {
    let units: [String: Set<String>] = [
      "electron_source/accelerating_voltage": ["V"],
      "illumination_system/semi_convergence_angle": ["mrad"],
      "scan_controller/regular_scan/dwell_time": ["s"],
      "imaging_system/camera_length": ["m"],
      scanRow: ["m"], scanColumn: ["m"],
      detectorRow: ["mrad", "1/nm", "1/Å"], detectorColumn: ["mrad", "1/nm", "1/Å"],
    ]
    for (path, quantity) in overrides {
      guard quantity.value.isFinite, quantity.value > 0,
        units[path]?.contains(quantity.unit) == true,
        quantity.provenance == "user_override", !quantity.evidence.isEmpty
      else {
        throw Native4DSTEMIOError.invalidData(
          "Invalid calibration override for \(path); supply a positive value, supported unit and evidence."
        )
      }
    }
    for (row, column) in [(scanRow, scanColumn), (detectorRow, detectorColumn)] {
      guard (overrides[row] == nil) == (overrides[column] == nil),
        overrides[row]?.unit == overrides[column]?.unit
      else {
        throw Native4DSTEMIOError.invalidData(
          "Calibration requires both row and column values in the same units.")
      }
    }
    if let row = overrides[scanRow], let column = overrides[scanColumn] {
      guard (1e-14...1e-6).contains(row.value), (1e-14...1e-6).contains(column.value) else {
        throw Native4DSTEMIOError.invalidData(
          "Scan sampling must be between 0.0001 and 10000 angstrom per pixel.")
      }
    }
  }

  public static func encoded(_ overrides: Overrides) throws -> Data {
    try validate(overrides)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys]
    return try encoder.encode(overrides)
  }

  public static func read(metadata: [String: String]) throws -> Overrides {
    guard let json = metadata[metadataKey] else { return [:] }
    let overrides = try JSONDecoder().decode(Overrides.self, from: Data(json.utf8))
    try validate(overrides)
    return overrides
  }

  public static func read(scientific: [String: Any]) throws -> Overrides {
    let record = try NativeQEMMetadataUnits.calculationOverrides(scientific)
    let overrides = try JSONDecoder().decode(
      Overrides.self, from: JSONSerialization.data(withJSONObject: record))
    try validate(overrides)
    return overrides
  }

  /// Replace overrides without changing the recorded source quantities or axes.
  public static func applying(_ overrides: Overrides, to scientific: [String: Any]) throws
    -> [String: Any]
  {
    var result = scientific
    let record = try JSONSerialization.jsonObject(with: encoded(overrides))
    let normalized = try NativeQEMMetadataUnits.normalized([
      "schema": NativeQEMMetadataUnits.legacySchema, "calibration_overrides": record,
    ])
    result["calibration_overrides"] =
      scientific["schema"] as? String == NativeQEMMetadataUnits.schema
      ? normalized["calibration_overrides"] : record
    return result
  }
}
