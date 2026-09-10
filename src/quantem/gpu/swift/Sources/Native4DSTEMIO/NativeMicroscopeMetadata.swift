import Foundation

/// Valid physical quantities from known acquisition metadata paths and units.
/// Unknown, nonfinite and nonpositive measurements remain absent; raw metadata
/// is retained by the catalog for inspection rather than promoted to calibration.
public struct NativeMicroscopeMetadata: Sendable {
  public let beamEnergyKeV: Double?
  public let semiConvergenceAngleMrad: Double?
  public let dwellTimeMicroseconds: Double?
  public let cameraLengthMillimeters: Double?
  public let angularRowMrad: Double?
  public let angularColumnMrad: Double?

  public init(metadata: [String: String]) {
    let root = "electron_microscope/"
    func quantity(_ path: String, factors: [String: Double]) -> Double? {
      guard let text = metadata[path] else { return nil }
      let parts = text.split(whereSeparator: \.isWhitespace)
      guard let first = parts.first, let value = Double(first), value.isFinite, value > 0,
        let unit = metadata[path + "@units"] ?? (parts.count == 2 ? String(parts[1]) : nil),
        let factor = factors[unit], (value * factor).isFinite
      else { return nil }
      return value * factor
    }
    beamEnergyKeV =
      quantity(root + "electron_source/accelerating_voltage", factors: ["V": 0.001, "kV": 1])
      ?? quantity("entry/instrument/detector/incident_energy", factors: ["eV": 0.001, "keV": 1])
    semiConvergenceAngleMrad = quantity(
      root + "illumination_system/semi_convergence_angle", factors: ["rad": 1000, "mrad": 1])
    dwellTimeMicroseconds = quantity(
      root + "scan_controller/regular_scan/dwell_time",
      factors: ["s": 1e6, "ms": 1000, "us": 1, "µs": 1, "μs": 1])
    cameraLengthMillimeters = quantity(
      root + "imaging_system/camera_length", factors: ["m": 1000, "mm": 1, "cm": 10])
    angularRowMrad = quantity(
      root + "imaging_system/reciprocal_pixel_size_y", factors: ["rad": 1000, "mrad": 1])
    angularColumnMrad = quantity(
      root + "imaging_system/reciprocal_pixel_size_x", factors: ["rad": 1000, "mrad": 1])
  }
}
