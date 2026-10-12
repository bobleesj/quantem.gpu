import Foundation

/// Physical input to native SSB. Angles are milliradians; scan steps are Å.
/// Detector sampling is explicit so an assumed aperture cannot silently replace
/// a recorded reciprocal-space calibration.
///
/// Its JSON form is the `calibration` block of the portable phase record
/// (`quantem.ssb-phase/2`), the keys the Python reader checks: `voltage_kV`,
/// `semiangle_mrad`, `scan_sampling_A`, `detector_sampling_mrad` and
/// `bright_field_center_px` as `[row, col]`, `bright_field_radius_px` and
/// `excluded_detector_pixels` when set.
public struct MetalSSBCalibration: Codable, Equatable, Sendable {
  public var voltageKV: Double
  public var semiangleMrad: Double
  public var scanSamplingRowAngstroms: Double
  public var scanSamplingColumnAngstroms: Double
  public var detectorSamplingRowMrad: Double
  public var detectorSamplingColumnMrad: Double
  public var brightfieldCenterRowPixels: Double
  public var brightfieldCenterColumnPixels: Double
  /// Logical detector disk, independent of the calibrated probe aperture.
  /// Nil selects the full nonzero calibrated aperture.
  public var brightfieldRadiusPixels: Double?
  public var excludedDetectorPixels: [Int]?

  private enum CodingKeys: String, CodingKey {
    case voltageKV = "voltage_kV"
    case semiangleMrad = "semiangle_mrad"
    case scanSampling = "scan_sampling_A"
    case detectorSampling = "detector_sampling_mrad"
    case brightfieldCenter = "bright_field_center_px"
    case brightfieldRadiusPixels = "bright_field_radius_px"
    case excludedDetectorPixels = "excluded_detector_pixels"
  }

  /// Interpret the selected circular BF disk as the full illumination aperture.
  /// A semi-angle spans a radius, not a diameter. Explicit sampling overrides
  /// remain possible after this operation; source metadata is never changed.
  public mutating func matchApertureToBrightfieldDisk() throws {
    guard let radius = brightfieldRadiusPixels, radius.isFinite, radius > 0,
      semiangleMrad.isFinite, semiangleMrad > 0
    else {
      throw MetalSSBError.invalidGeometry(
        "Enter a positive BF radius and semi-angle before matching the aperture.")
    }
    detectorSamplingRowMrad = semiangleMrad / radius
    detectorSamplingColumnMrad = semiangleMrad / radius
  }

  public init(
    voltageKV: Double, semiangleMrad: Double,
    scanSamplingRowAngstroms: Double, scanSamplingColumnAngstroms: Double,
    detectorSamplingRowMrad: Double, detectorSamplingColumnMrad: Double,
    brightfieldCenterRowPixels: Double, brightfieldCenterColumnPixels: Double,
    brightfieldRadiusPixels: Double? = nil, excludedDetectorPixels: [Int]? = nil
  ) {
    self.voltageKV = voltageKV
    self.semiangleMrad = semiangleMrad
    self.scanSamplingRowAngstroms = scanSamplingRowAngstroms
    self.scanSamplingColumnAngstroms = scanSamplingColumnAngstroms
    self.detectorSamplingRowMrad = detectorSamplingRowMrad
    self.detectorSamplingColumnMrad = detectorSamplingColumnMrad
    self.brightfieldCenterRowPixels = brightfieldCenterRowPixels
    self.brightfieldCenterColumnPixels = brightfieldCenterColumnPixels
    self.brightfieldRadiusPixels = brightfieldRadiusPixels
    self.excludedDetectorPixels = excludedDetectorPixels
  }

  /// Read the record's `[row, col]` pairs; a pair of another length is refused, not guessed.
  public init(from decoder: Decoder) throws {
    let container = try decoder.container(keyedBy: CodingKeys.self)
    func pair(_ key: CodingKeys) throws -> (row: Double, col: Double) {
      let values = try container.decode([Double].self, forKey: key)
      guard values.count == 2 else {
        throw DecodingError.dataCorruptedError(
          forKey: key, in: container, debugDescription: "\(key.stringValue) is a [row, col] pair.")
      }
      return (values[0], values[1])
    }
    let scan = try pair(.scanSampling)
    let detector = try pair(.detectorSampling)
    let center = try pair(.brightfieldCenter)
    self.init(
      voltageKV: try container.decode(Double.self, forKey: .voltageKV),
      semiangleMrad: try container.decode(Double.self, forKey: .semiangleMrad),
      scanSamplingRowAngstroms: scan.row, scanSamplingColumnAngstroms: scan.col,
      detectorSamplingRowMrad: detector.row, detectorSamplingColumnMrad: detector.col,
      brightfieldCenterRowPixels: center.row, brightfieldCenterColumnPixels: center.col,
      brightfieldRadiusPixels: try container.decodeIfPresent(
        Double.self, forKey: .brightfieldRadiusPixels),
      excludedDetectorPixels: try container.decodeIfPresent(
        [Int].self, forKey: .excludedDetectorPixels))
  }

  public func encode(to encoder: Encoder) throws {
    var container = encoder.container(keyedBy: CodingKeys.self)
    try container.encode(voltageKV, forKey: .voltageKV)
    try container.encode(semiangleMrad, forKey: .semiangleMrad)
    try container.encode(
      [scanSamplingRowAngstroms, scanSamplingColumnAngstroms], forKey: .scanSampling)
    try container.encode(
      [detectorSamplingRowMrad, detectorSamplingColumnMrad], forKey: .detectorSampling)
    try container.encode(
      [brightfieldCenterRowPixels, brightfieldCenterColumnPixels], forKey: .brightfieldCenter)
    try container.encodeIfPresent(brightfieldRadiusPixels, forKey: .brightfieldRadiusPixels)
    try container.encodeIfPresent(excludedDetectorPixels, forKey: .excludedDetectorPixels)
  }

  /// Build the full active aperture, preserving detector order and exclusions.
  /// The small detector sum is already reduced from the full original scan.
  public func geometry(
    detectorRows: Int, detectorColumns: Int,
    detectorSum: [UInt64], excludedPixels: Set<Int> = [],
    scanRows: Int = 512, scanColumns: Int = 512
  ) throws
    -> (geometry: MetalSSBGeometry, pixels: [Int])
  {
    guard [128, 256, 512].contains(scanRows), scanColumns == scanRows else {
      throw MetalSSBError.invalidGeometry(
        "Native Metal SSB requires a square 128, 256, or 512 scan. Keep the original sampling; do not bin or crop to fit."
      )
    }
    let positive = [
      voltageKV, semiangleMrad, scanSamplingRowAngstroms,
      scanSamplingColumnAngstroms, detectorSamplingRowMrad, detectorSamplingColumnMrad,
    ]
    let centerRow = brightfieldCenterRowPixels
    let centerColumn = brightfieldCenterColumnPixels
    guard positive.allSatisfy({ $0.isFinite && $0 > 0 }),
      centerRow.isFinite, centerColumn.isFinite,
      detectorRows > 0, detectorColumns > 0,
      centerRow >= 0, centerRow < Double(detectorRows),
      centerColumn >= 0, centerColumn < Double(detectorColumns),
      detectorSum.count == detectorRows * detectorColumns
    else {
      throw MetalSSBError.invalidGeometry(
        "Enter positive energy, angles and scan sampling, and a detector center inside the image.")
    }
    if let radius = brightfieldRadiusPixels, !radius.isFinite || radius <= 0 {
      throw MetalSSBError.invalidGeometry(
        "Enter a positive bright-field radius in detector pixels.")
    }
    let excludedPixels = excludedPixels.union(excludedDetectorPixels ?? [])
    guard excludedPixels.allSatisfy({ $0 >= 0 && $0 < detectorSum.count }) else {
      throw MetalSSBError.invalidGeometry(
        "Excluded detector coordinates must be inside the detector image.")
    }
    // Relativistic de Broglie wavelength using SI defining constants, then Å.
    let voltage = voltageKV * 1000
    let h = 6.62607015e-34
    let mass = 9.1093837139e-31
    let charge = 1.602176634e-19
    let c = 299792458.0
    let energy = charge * voltage
    let wavelength = h / sqrt(2 * mass * energy * (1 + energy / (2 * mass * c * c))) * 1e10
    let angle = semiangleMrad * 1e-3
    let stepRow = detectorSamplingRowMrad * 1e-3
    let stepColumn = detectorSamplingColumnMrad * 1e-3
    var pixels: [Int] = []
    var row: [Float] = []
    var column: [Float] = []
    var alpha2: [Float] = []
    var aperture: [Float] = []
    var cos2: [Float] = []
    var sin2: [Float] = []
    var dc = 0.0
    for pixel in detectorSum.indices where !excludedPixels.contains(pixel) {
      if let radius = brightfieldRadiusPixels {
        let dr = Double(pixel / detectorColumns) - centerRow
        let dc = Double(pixel % detectorColumns) - centerColumn
        guard dr * dr + dc * dc <= radius * radius else { continue }
      }
      let kr = (Double(pixel / detectorColumns) - centerRow) * stepRow / wavelength
      let kc = (Double(pixel % detectorColumns) - centerColumn) * stepColumn / wavelength
      let r2 = kr * kr + kc * kc
      let r = sqrt(r2)
      let edgeWidth = r > 1e-15 ? hypot(kr * stepRow, kc * stepColumn) / r : 0
      let weight =
        edgeWidth > 1e-15 ? min(1, max(0, (angle - r * wavelength) / edgeWidth + 0.5)) : 1
      guard brightfieldRadiusPixels != nil || weight > 0 else { continue }
      pixels.append(pixel)
      row.append(Float(kr))
      column.append(Float(kc))
      alpha2.append(Float(r2 * wavelength * wavelength))
      aperture.append(Float(weight))
      cos2.append(Float(r2 > 1e-30 ? (kr * kr - kc * kc) / r2 : 0))
      sin2.append(Float(r2 > 1e-30 ? 2 * kr * kc / r2 : 0))
      dc += Double(detectorSum[pixel])
    }
    guard !pixels.isEmpty, dc > 0 else {
      throw MetalSSBError.invalidGeometry(
        "The calibrated aperture contains no measured counts. Check the beam center and angular sampling."
      )
    }
    func frequencies(_ step: Double, count: Int) -> [Float] {
      (0..<count).map {
        Float(Double($0 < count / 2 ? $0 : $0 - count) / (Double(count) * step))
      }
    }
    return (
      MetalSSBGeometry(
        brightfieldKX: row, brightfieldKY: column,
        brightfieldAlphaSquared: alpha2, brightfieldAperture: aperture,
        brightfieldCos2Phi: cos2, brightfieldSin2Phi: sin2,
        qxByRow: frequencies(scanSamplingRowAngstroms, count: scanRows),
        qyByColumn: frequencies(scanSamplingColumnAngstroms, count: scanColumns),
        wavelengthAngstroms: Float(wavelength), semiangleRadians: Float(angle),
        angularSamplingYRadians: Float(stepRow), angularSamplingXRadians: Float(stepColumn),
        dcValue: SIMD2(Float(dc / Double(pixels.count)), 0), referenceRotationDegrees: 0), pixels
    )
  }
}
