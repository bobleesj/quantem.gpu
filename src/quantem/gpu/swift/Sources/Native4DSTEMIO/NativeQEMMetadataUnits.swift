import CoreFoundation
import Foundation

/// Microscopy-friendly on-disk units, separate from source tags and calculation APIs.
/// Example: `try NativeQEMMetadataUnits.normalized(scientific)` before export.
public enum NativeQEMMetadataUnits {
  public static let schema = "quantem.scientific-metadata/2"
  public static let legacySchema = "quantem.scientific-metadata/1"

  private static func conversion(path: String, unit: String) -> (String, Double)? {
    let target: String
    let factors: [String: Double]
    switch path {
    case "scan", NativeQEMCalibration.scanRow, NativeQEMCalibration.scanColumn:
      target = "angstrom"
      factors = ["m": 1e10, "nm": 10, "angstrom": 1, "Å": 1]
    case "detector", NativeQEMCalibration.detectorRow, NativeQEMCalibration.detectorColumn:
      return ["rad": ("mrad", 1000), "mrad": ("mrad", 1),
        "1/nm": ("1/angstrom", 0.1), "1/Å": ("1/angstrom", 1),
        "1/angstrom": ("1/angstrom", 1)][unit]
    case "electron_source/accelerating_voltage":
      target = "kV"; factors = ["V": 0.001, "kV": 1]
    case "electron_source/beam_energy":
      target = "keV"; factors = ["eV": 0.001, "keV": 1]
    case "illumination_system/semi_convergence_angle":
      target = "mrad"; factors = ["rad": 1000, "mrad": 1]
    case "scan_controller/regular_scan/dwell_time":
      target = "us"; factors = ["s": 1e6, "ms": 1000, "us": 1]
    case "imaging_system/camera_length":
      target = "mm"; factors = ["m": 1000, "cm": 10, "mm": 1]
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
      [schema, legacySchema].contains(version) else {
      throw Native4DSTEMIOError.invalidData("Unsupported QEM metadata schema; update the reader.")
    }
    var result = scientific
    if var axes = scientific["axes"] as? [[String: Any]] {
      for index in axes.indices {
        if let sampling = axes[index]["sampling"] {
          let path = (axes[index]["name"] as? String)?.hasPrefix("scan_") == true
            ? "scan" : "detector"
          axes[index]["sampling"] = try normalizedQuantity(sampling, path: path)
        }
      }
      result["axes"] = axes
    }
    for section in ["electron_microscope", "calibration_overrides"] {
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
          quantity["unit"] as? String == original["unit"] as? String else {
          throw Native4DSTEMIOError.invalidData("Schema-2 QEM overrides require microscopy units.")
        }
        let unit = quantity["unit"] as! String
        let (target, factor) = [
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
