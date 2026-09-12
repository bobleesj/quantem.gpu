import Foundation

/// Physical input to native SSB. Angles are milliradians; scan steps are Å.
/// Detector sampling is explicit so an assumed aperture cannot silently replace
/// a recorded reciprocal-space calibration.
public struct MetalSSBCalibration: Codable, Equatable, Sendable {
  public var beamEnergyKeV: Double
  public var semiangleMrad: Double
  public var scanStepRowAngstroms: Double
  public var scanStepColumnAngstroms: Double
  public var detectorStepRowMrad: Double
  public var detectorStepColumnMrad: Double
  public var centerRow: Double
  public var centerColumn: Double
  /// Logical detector disk, independent of the calibrated probe aperture.
  /// Nil selects the full nonzero calibrated aperture.
  public var brightfieldRadiusPixels: Double?
  public var excludedDetectorPixels: [Int]?

  /// Interpret the selected circular BF disk as the full illumination aperture.
  /// A semi-angle spans a radius, not a diameter. Explicit sampling overrides
  /// remain possible after this operation; source metadata is never changed.
  public mutating func matchApertureToBrightfieldDisk() throws {
    guard let radius = brightfieldRadiusPixels, radius.isFinite, radius > 0,
      semiangleMrad.isFinite, semiangleMrad > 0 else {
      throw MetalSSBError.invalidGeometry("Enter a positive BF radius and semi-angle before matching the aperture.")
    }
    detectorStepRowMrad = semiangleMrad / radius
    detectorStepColumnMrad = semiangleMrad / radius
  }

  public init(beamEnergyKeV: Double, semiangleMrad: Double,
    scanStepRowAngstroms: Double, scanStepColumnAngstroms: Double,
    detectorStepRowMrad: Double, detectorStepColumnMrad: Double,
    centerRow: Double, centerColumn: Double, brightfieldRadiusPixels: Double? = nil,
    excludedDetectorPixels: [Int]? = nil) {
    self.beamEnergyKeV = beamEnergyKeV
    self.semiangleMrad = semiangleMrad
    self.scanStepRowAngstroms = scanStepRowAngstroms
    self.scanStepColumnAngstroms = scanStepColumnAngstroms
    self.detectorStepRowMrad = detectorStepRowMrad
    self.detectorStepColumnMrad = detectorStepColumnMrad
    self.centerRow = centerRow
    self.centerColumn = centerColumn
    self.brightfieldRadiusPixels = brightfieldRadiusPixels
    self.excludedDetectorPixels = excludedDetectorPixels
  }

  /// Build the full active aperture, preserving detector order and exclusions.
  /// The small detector sum is already reduced from the full original scan.
  public func geometry(detectorRows: Int, detectorColumns: Int,
    detectorSum: [UInt64], excludedPixels: Set<Int> = []) throws
    -> (geometry: MetalSSBGeometry, pixels: [Int]) {
    let positive = [beamEnergyKeV, semiangleMrad, scanStepRowAngstroms,
      scanStepColumnAngstroms, detectorStepRowMrad, detectorStepColumnMrad]
    guard positive.allSatisfy({ $0.isFinite && $0 > 0 }),
      centerRow.isFinite, centerColumn.isFinite,
      detectorRows > 0, detectorColumns > 0,
      centerRow >= 0, centerRow < Double(detectorRows),
      centerColumn >= 0, centerColumn < Double(detectorColumns),
      detectorSum.count == detectorRows * detectorColumns else {
      throw MetalSSBError.invalidGeometry("Enter positive energy, angles and scan sampling, and a detector center inside the image.")
    }
    if let radius = brightfieldRadiusPixels, !radius.isFinite || radius <= 0 {
      throw MetalSSBError.invalidGeometry("Enter a positive bright-field radius in detector pixels.")
    }
    let excludedPixels = excludedPixels.union(excludedDetectorPixels ?? [])
    guard excludedPixels.allSatisfy({ $0 >= 0 && $0 < detectorSum.count }) else {
      throw MetalSSBError.invalidGeometry("Excluded detector coordinates must be inside the detector image.")
    }
    // Relativistic de Broglie wavelength using SI defining constants, then Å.
    let voltage = beamEnergyKeV * 1000
    let h = 6.62607015e-34, mass = 9.1093837139e-31
    let charge = 1.602176634e-19, c = 299792458.0
    let energy = charge * voltage
    let wavelength = h / sqrt(2 * mass * energy * (1 + energy / (2 * mass * c * c))) * 1e10
    let angle = semiangleMrad * 1e-3
    let stepRow = detectorStepRowMrad * 1e-3
    let stepColumn = detectorStepColumnMrad * 1e-3
    var pixels: [Int] = [], row: [Float] = [], column: [Float] = []
    var alpha2: [Float] = [], aperture: [Float] = [], cos2: [Float] = [], sin2: [Float] = []
    var dc = 0.0
    for pixel in detectorSum.indices where !excludedPixels.contains(pixel) {
      if let radius = brightfieldRadiusPixels {
        let dr = Double(pixel / detectorColumns) - centerRow
        let dc = Double(pixel % detectorColumns) - centerColumn
        guard dr * dr + dc * dc <= radius * radius else { continue }
      }
      let kr = (Double(pixel / detectorColumns) - centerRow) * stepRow / wavelength
      let kc = (Double(pixel % detectorColumns) - centerColumn) * stepColumn / wavelength
      let r2 = kr * kr + kc * kc, r = sqrt(r2)
      let edgeWidth = r > 1e-15 ? hypot(kr * stepRow, kc * stepColumn) / r : 0
      let weight = edgeWidth > 1e-15 ? min(1, max(0, (angle - r * wavelength) / edgeWidth + 0.5)) : 1
      guard brightfieldRadiusPixels != nil || weight > 0 else { continue }
      pixels.append(pixel); row.append(Float(kr)); column.append(Float(kc))
      alpha2.append(Float(r2 * wavelength * wavelength)); aperture.append(Float(weight))
      cos2.append(Float(r2 > 1e-30 ? (kr * kr - kc * kc) / r2 : 0))
      sin2.append(Float(r2 > 1e-30 ? 2 * kr * kc / r2 : 0))
      dc += Double(detectorSum[pixel])
    }
    guard !pixels.isEmpty, dc > 0 else {
      throw MetalSSBError.invalidGeometry("The calibrated aperture contains no measured counts. Check the beam center and angular sampling.")
    }
    func frequencies(_ step: Double) -> [Float] {
      (0..<512).map { Float(Double($0 < 256 ? $0 : $0 - 512) / (512 * step)) }
    }
    return (MetalSSBGeometry(brightfieldKX: row, brightfieldKY: column,
      brightfieldAlphaSquared: alpha2, brightfieldAperture: aperture,
      brightfieldCos2Phi: cos2, brightfieldSin2Phi: sin2,
      qxByRow: frequencies(scanStepRowAngstroms), qyByColumn: frequencies(scanStepColumnAngstroms),
      wavelengthAngstroms: Float(wavelength), semiangleRadians: Float(angle),
      angularSamplingYRadians: Float(stepRow), angularSamplingXRadians: Float(stepColumn),
      dcValue: SIMD2(Float(dc / Double(pixels.count)), 0), referenceRotationDegrees: 0), pixels)
  }
}
