import CoreFoundation
import Foundation

/// Microscopy-friendly on-disk units, separate from source tags and calculation APIs.
/// Example: `try NativeQEMMetadataUnits.normalized(scientific)` before export.
public enum NativeQEMMetadataUnits {
  public static let schema = "quantem.scientific-metadata/2"
  public static let legacySchema = "quantem.scientific-metadata/1"
  public static let metadataKey = "qem_scientific_metadata"

  private static func conversion(path: String, unit: String) -> (String, Double)? {
    let target: String
    let factors: [String: Double]
    switch path {
    case "scan", NativeQEMCalibration.scanRow, NativeQEMCalibration.scanColumn:
      target = "angstrom"
      factors = ["m": 1e10, "nm": 10, "angstrom": 1, "Å": 1]
    case "detector", NativeQEMCalibration.detectorRow, NativeQEMCalibration.detectorColumn:
      return [
        "rad": ("mrad", 1000), "mrad": ("mrad", 1),
        "1/nm": ("1/angstrom", 0.1), "1/Å": ("1/angstrom", 1),
        "1/angstrom": ("1/angstrom", 1),
      ][unit]
    case "electron_source/accelerating_voltage":
      target = "kV"
      factors = ["V": 0.001, "kV": 1]
    case "electron_source/beam_energy":
      target = "keV"
      factors = ["eV": 0.001, "keV": 1]
    case "illumination_system/semi_convergence_angle":
      target = "mrad"
      factors = ["rad": 1000, "mrad": 1]
    case "scan_controller/regular_scan/dwell_time":
      target = "us"
      factors = ["s": 1e6, "ms": 1000, "us": 1]
    case "imaging_system/camera_length":
      target = "mm"
      factors = ["m": 1000, "cm": 10, "mm": 1]
    default: return nil
    }
    return factors[unit].map { (target, $0) }
  }

  private static let knownPaths: Set<String> = [
    NativeQEMCalibration.scanRow, NativeQEMCalibration.scanColumn,
    NativeQEMCalibration.detectorRow, NativeQEMCalibration.detectorColumn,
    "electron_source/accelerating_voltage", "electron_source/beam_energy",
    "illumination_system/semi_convergence_angle",
    "scan_controller/regular_scan/dwell_time", "imaging_system/camera_length",
  ]

  private static func normalizedQuantity(_ raw: Any, path: String) throws -> [String: Any] {
    guard var quantity = raw as? [String: Any], let number = quantity["value"] as? NSNumber,
      CFGetTypeID(number) != CFBooleanGetTypeID(),
      let value = quantity["value"] as? Double,
      let unit = quantity["unit"] as? String, value.isFinite, value > 0,
      let (target, factor) = conversion(path: path, unit: unit), (value * factor).isFinite
    else {
      throw Native4DSTEMIOError.invalidData(
        "Invalid QEM quantity at \(path); check its positive value and physical unit.")
    }
    quantity["value"] = value * factor
    quantity["unit"] = target
    return quantity
  }

  /// Preserve original fields and evidence while normalizing known quantities.
  public static func normalized(_ scientific: [String: Any]) throws -> [String: Any] {
    guard let version = scientific["schema"] as? String,
      [schema, legacySchema].contains(version)
    else {
      throw Native4DSTEMIOError.invalidData("Unsupported QEM metadata schema; update the reader.")
    }
    var result = scientific
    if var axes = scientific["axes"] as? [[String: Any]] {
      for index in axes.indices {
        if let sampling = axes[index]["sampling"] {
          let path =
            (axes[index]["name"] as? String)?.hasPrefix("scan_") == true
            ? "scan" : "detector"
          axes[index]["sampling"] = try normalizedQuantity(sampling, path: path)
        }
      }
      result["axes"] = axes
    }
    for section in ["electron_microscope", "calibration_overrides"] {
      guard scientific[section] == nil || scientific[section] is [String: Any] else {
        throw Native4DSTEMIOError.invalidData("QEM \(section) must contain named quantities.")
      }
      if var quantities = scientific[section] as? [String: Any] {
        for (path, quantity) in quantities where knownPaths.contains(path) {
          quantities[path] = try normalizedQuantity(quantity, path: path)
        }
        result[section] = quantities
      }
    }
    result["schema"] = schema
    return result
  }

  /// Quantity names of specification 0.0.1, refused since 0.0.2 names them by row and column.
  static let retiredQuantities: Set<String> = [
    "scan_controller/regular_scan/pixel_size_y", "scan_controller/regular_scan/pixel_size_x",
    "imaging_system/reciprocal_pixel_size_y", "imaging_system/reciprocal_pixel_size_x",
  ]

  /// Validate documented calibration provenance and redundant physical quantities.
  /// Example: `try NativeQEMMetadataUnits.validateScientific(scientific)`.
  public static func validateScientific(_ scientific: [String: Any]) throws {
    _ = try NativeMetadataDocument.read(scientific: scientific)
    let normalized = try normalized(scientific)
    guard scientific["source_metadata"] is [String: Any],
      let coverage = scientific["source_metadata_coverage"] as? String,
      ["reader-retained", "exhaustive", "unknown"].contains(coverage),
      let axes = normalized["axes"] as? [[String: Any]]
    else {
      throw Native4DSTEMIOError.invalidData("QEM requires source metadata and explicit coverage.")
    }
    guard
      normalized["electron_microscope"] == nil
        || normalized["electron_microscope"] is [String: [String: Any]]
    else {
      throw Native4DSTEMIOError.invalidData("QEM microscope fields must be quantities.")
    }
    let quantities = normalized["electron_microscope"] as? [String: [String: Any]] ?? [:]
    for section in ["electron_microscope", "calibration_overrides"] {
      let names = (scientific[section] as? [String: Any] ?? [:]).keys
      if let retired = names.sorted().first(where: { retiredQuantities.contains($0) }) {
        throw Native4DSTEMIOError.invalidData(
          "QEM \(section) uses the x/y names of specification 0.0.1 (\(retired)); "
            + "quantities are named by row and column. Re-export the original acquisition.")
      }
    }
    guard let processing = scientific["processing"] as? [[String: Any]], !processing.isEmpty,
      processing.allSatisfy({ record in
        guard let operation = record["operation"] as? String, !operation.isEmpty,
          let changes = record["changes_measurements"] as? NSNumber
        else { return false }
        return CFGetTypeID(changes) == CFBooleanGetTypeID()
      })
    else {
      throw Native4DSTEMIOError.invalidData(
        "QEM processing must list every operation and whether it changes measurements.")
    }
    func provenance(_ quantity: [String: Any]) throws {
      guard let origin = quantity["provenance"] as? String,
        !origin.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
      else {
        throw Native4DSTEMIOError.invalidData("Calibrated QEM quantities require provenance.")
      }
    }
    let paths = [
      NativeQEMCalibration.scanRow, NativeQEMCalibration.scanColumn,
      NativeQEMCalibration.detectorRow, NativeQEMCalibration.detectorColumn,
    ]
    for (axis, path) in zip(axes, paths) {
      let sampling = axis["sampling"] as? [String: Any]
      if let sampling { try provenance(sampling) }
      if scientific["schema"] as? String == schema, let duplicate = quantities[path] {
        guard let sampling, sampling["unit"] as? String == duplicate["unit"] as? String,
          let value = sampling["value"] as? Double, let other = duplicate["value"] as? Double,
          abs(value - other) <= max(abs(value), abs(other)) * 1e-14
        else {
          throw Native4DSTEMIOError.invalidData("Conflicting QEM axis calibration at \(path).")
        }
      }
    }
    for (_, quantity) in quantities {
      guard let number = quantity["value"] as? NSNumber,
        CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite,
        number.doubleValue > 0, let unit = quantity["unit"] as? String, !unit.isEmpty
      else {
        throw Native4DSTEMIOError.invalidData(
          "QEM microscope fields require a positive value and unit.")
      }
      try provenance(quantity)
    }
  }

  /// Derive recorded calibration without relying on private restoration fields.
  /// Example: `let recorded = try NativeQEMMetadataUnits.recordedMetadata(scientific)`.
  public static func recordedMetadata(_ scientific: [String: Any]) throws -> [String: Any] {
    let normalized = try normalized(scientific)
    let axes = normalized["axes"] as? [[String: Any]] ?? []
    var result: [String: Any] = ["source_metadata": normalized["source_metadata"] ?? [:]]
    for (first, field) in [(0, "scan_sampling_A"), (2, "detector_sampling")] {
      guard axes.count >= first + 2,
        let row = axes[first]["sampling"] as? [String: Any],
        let col = axes[first + 1]["sampling"] as? [String: Any]
      else { continue }
      guard row["unit"] as? String == col["unit"] as? String else {
        throw Native4DSTEMIOError.invalidData("QEM row and column sampling units disagree.")
      }
      result[field] = [row["value"] as! Double, col["value"] as! Double]
      if first == 2 { result["detector_sampling_unit"] = row["unit"] }
    }
    let quantities = normalized["electron_microscope"] as? [String: [String: Any]] ?? [:]
    result["voltage_kV"] = quantities["electron_source/accelerating_voltage"]?["value"]
    return result
  }

  /// Bridge public quantities to native microscope calculations, retaining the public record.
  /// Example: `let metadata = try NativeQEMMetadataUnits.microscopeMetadata(scientific)`.
  public static func microscopeMetadata(_ scientific: [String: Any]) throws -> [String: String] {
    let normalized = try normalized(scientific)
    var result = normalized["source_metadata"] as? [String: String] ?? [:]
    let quantities = normalized["electron_microscope"] as? [String: [String: Any]] ?? [:]
    for (path, quantity) in quantities where knownPaths.contains(path) {
      result["electron_microscope/" + path] = String(quantity["value"] as! Double)
      result["electron_microscope/" + path + "@units"] = quantity["unit"] as? String
    }
    result[metadataKey] = String(
      decoding: try JSONSerialization.data(withJSONObject: scientific), as: UTF8.self)
    return result
  }

  /// Restore schema-2 overrides to the existing native/app calibration API units.
  public static func calculationOverrides(_ scientific: [String: Any]) throws -> [String: Any] {
    guard var overrides = scientific["calibration_overrides"] as? [String: Any] else {
      if scientific["calibration_overrides"] == nil { return [:] }
      throw Native4DSTEMIOError.invalidData("QEM calibration overrides must be named quantities.")
    }
    if scientific["schema"] as? String == schema {
      for (path, raw) in overrides {
        var quantity = try normalizedQuantity(raw, path: path)
        guard let original = raw as? [String: Any],
          quantity["unit"] as? String == original["unit"] as? String
        else {
          throw Native4DSTEMIOError.invalidData("Schema-2 QEM overrides require microscopy units.")
        }
        let unit = quantity["unit"] as! String
        let (target, factor) =
          [
            "angstrom": ("m", 1e-10), "kV": ("V", 1000), "us": ("s", 1e-6),
            "mm": ("m", 0.001), "1/angstrom": ("1/Å", 1), "mrad": ("mrad", 1),
          ][unit] ?? (unit, 1)
        quantity["value"] = (quantity["value"] as! Double) * factor
        quantity["unit"] = target
        overrides[path] = quantity
      }
    }
    return overrides
  }
}
