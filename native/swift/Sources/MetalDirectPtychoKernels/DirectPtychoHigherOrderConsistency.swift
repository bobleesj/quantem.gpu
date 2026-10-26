import Foundation

/// The half-band check of a wide fifth-order fit: refit the fifth order at the same limit on each half of a ring split
/// of the band frequencies (alternate |q| rings of width `ringWidthInverseAngstroms`), both halves starting from the
/// same low-order fit, and compare the two fits' probe phase over the aperture. Real higher-order aberrations are the
/// probe's and the same in both halves; terms that fit noise or scattering the model does not describe are not.
///
/// The disagreement is the rms over the aperture, piston removed, of the difference of the two halves' probe phase
/// through fifth order (C10, C12 and C21 through C56; tilt and depth excluded), sampled on 96 x 96 points of the
/// unit disk. The fit is trusted when it is at most `maximumHalfBandRMSRadians`; otherwise the fit shown falls back
/// to the standard ±2 box and its higher orders are marked not trusted (thick crystals give spurious terms at ±2 too,
/// so the flag applies to the ±2 values as well). On measured frames of a thin two-dimensional crystal the halves
/// agreed to 0.16-0.35 rad and on a thick crystal disagreed by 1.33 rad; in simulations the rule passed 14 accurate
/// and 1 wrong of 102 thick nulls and improved 27 of 39 positives beyond the standard box (the wide-box validation of
/// 2026-10-25). Metal, CUDA and MPS compute the same statistic; the record names are Python's.
public struct DirectPtychoHigherOrderConsistency: Codable, Equatable, Sendable {
  /// The ring width of the split (1/A), as the validation's `ring0.05` on 512 x 512 scans.
  public static let ringWidthInverseAngstroms = 0.05
  /// The largest rms aperture-phase disagreement (rad) of a trusted fit.
  public static let maximumHalfBandRMSRadians = 0.5

  /// One half-band refit.
  public struct Half: Codable, Equatable, Sendable {
    /// The ring parity fitted: floor(|q| / width) mod 2.
    public let ringParity: Int
    /// The band frequencies fitted (of the stored half plane).
    public let frequencies: Int
    public let coordinates: [Double]
    public let point: DirectPtychoModelPoint
    public let evaluations: Int
    public let stop: String
    public let seconds: Double

    private enum CodingKeys: String, CodingKey {
      case ringParity = "ring_parity"
      case frequencies, coordinates, point, evaluations, stop, seconds
    }
  }

  /// The limit of the checked fit and its halves.
  public let limit: Double
  public let ringWidthInverseAngstroms: Double
  public let halves: [Half]
  /// rms over the aperture (piston removed) of the two halves' probe-phase difference, rad.
  public let halfBandRMSRadians: Double
  public let maximumHalfBandRMSRadians: Double
  /// The halves agree: the checked fit's higher orders are trusted and it is shown.
  public let trusted: Bool
  /// The limit of the fit shown: `limit` when trusted, else `DirectPtychoFitOptions.standardHigherOrderLimit`.
  public let shownLimit: Double
  /// The checked full-band fit at `limit` (kept here when the fit falls back).
  public let checkedCoordinates: [Double]
  public let checkedPoint: DirectPtychoModelPoint

  /// The record names Python uses: `half_band_rms_rad`, `trusted`, `shown_limit`.
  private enum CodingKeys: String, CodingKey {
    case limit, halves, trusted
    case ringWidthInverseAngstroms = "ring_width_inv_A"
    case halfBandRMSRadians = "half_band_rms_rad"
    case maximumHalfBandRMSRadians = "max_half_band_rms_rad"
    case shownLimit = "shown_limit"
    case checkedCoordinates = "checked_coordinates"
    case checkedPoint = "checked_point"
  }

  /// "halves disagree by 1.33 rad" when not trusted.
  public var reason: String? {
    trusted ? nil : String(format: "halves disagree by %.2f rad", halfBandRMSRadians)
  }

  /// Ring parity of a band frequency q (1/A): floor(|q| / width) mod 2.
  public static func ringParity(_ q: SIMD2<Float>, width: Double = ringWidthInverseAngstroms) -> Int
  {
    let ring = Int(floor(hypot(Double(q.x), Double(q.y)) / width))
    return ((ring % 2) + 2) % 2
  }

  /// rms over the unit aperture (piston removed) of the probe-phase difference of fit coordinates `a` and `b`
  /// (`DirectPtychoFifthOrderCoordinates`), rad: C10 and C12 as (2 pi / lambda) alpha^2 r^2 / 2 (C10 + C12 cos 2(phi - phi12)),
  /// each higher-order term as its edge phase times r^(n+1) (cos m phi, sin m phi).
  public static func apertureDisagreement(
    _ a: [Double], _ b: [Double], wavelengthAngstroms: Double, semiangleRadians: Double,
    points: Int = 96
  ) -> Double {
    let d = zip(a, b).map { $0 - $1 }
    let k = 2 * Double.pi / wavelengthAngstroms * semiangleRadians * semiangleRadians / 2 * 100
    var values: [Double] = []
    values.reserveCapacity(points * points)
    for i in 0..<points {
      let x = (Double(i) + 0.5) / Double(points) * 2 - 1
      for j in 0..<points {
        let y = (Double(j) + 0.5) / Double(points) * 2 - 1
        guard x * x + y * y <= 1 else { continue }
        let r = hypot(x, y)
        let phi = atan2(y, x)
        var value = k * r * r * (d[0] + d[1] * cos(2 * phi) + d[2] * sin(2 * phi))
        var column = 6
        for (n, m) in DirectPtychoFifthOrderCoordinates.orders where n >= 2 {
          let radial = pow(r, Double(n + 1))
          if m == 0 {
            value += d[column] * radial
            column += 1
          } else {
            value +=
              radial * (d[column] * cos(Double(m) * phi) + d[column + 1] * sin(Double(m) * phi))
            column += 2
          }
        }
        values.append(value)
      }
    }
    let mean = values.reduce(0, +) / Double(values.count)
    return (values.reduce(0) { $0 + ($1 - mean) * ($1 - mean) } / Double(values.count)).squareRoot()
  }
}

extension DirectPtychoModelFit {
  /// The half-band check of a fifth-order fit at `limit` with coordinates `checked`, refined from the low-order `start`
  /// on `model` at its rotation: one refit of the same search on each ring half of the band, from `start`.
  public static func higherOrderConsistency(
    start: DirectPtychoModelPoint, checked: [Double], model: MetalDirectPtychoJointModel,
    search: HigherOrderSearch,
    maxEvaluations: Int, limit: Double = DirectPtychoFitOptions.wideHigherOrderLimit,
    ringWidthInverseAngstroms: Double = DirectPtychoHigherOrderConsistency
      .ringWidthInverseAngstroms,
    maximumHalfBandRMSRadians: Double = DirectPtychoHigherOrderConsistency
      .maximumHalfBandRMSRadians,
    progress: (_ completed: Int, _ total: Int) -> Void = { _, _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> DirectPtychoHigherOrderConsistency {
    let parities = model.bandFrequencies.map {
      DirectPtychoHigherOrderConsistency.ringParity($0, width: ringWidthInverseAngstroms)
    }
    var halves: [DirectPtychoHigherOrderConsistency.Half] = []
    progress(0, 2)
    for parity in 0...1 {
      let mask = parities.map { $0 == parity }
      let frequencies = mask.filter { $0 }.count
      guard frequencies > 0 else {
        throw MetalDirectPtychoError.invalidGeometry(
          "The half-band check needs band frequencies in both ring halves. Check the scan sampling."
        )
      }
      let started = Date()
      let refined = try model.withBandMask(mask) {
        try refineFifthOrder(
          start, model: model, search: search, maxEvaluations: maxEvaluations,
          bounds: fifthOrderBounds(higherOrderLimit: limit), isCancelled: isCancelled)
      }
      halves.append(
        .init(
          ringParity: parity, frequencies: frequencies, coordinates: refined.result.coordinates,
          point: refined.point, evaluations: refined.result.evaluations, stop: refined.result.stop,
          seconds: Date().timeIntervalSince(started)))
      progress(parity + 1, 2)
    }
    let disagreement = DirectPtychoHigherOrderConsistency.apertureDisagreement(
      halves[0].coordinates, halves[1].coordinates, wavelengthAngstroms: model.wavelengthAngstroms,
      semiangleRadians: model.semiangleRadians)
    let trusted = disagreement <= maximumHalfBandRMSRadians
    let coordinates = DirectPtychoFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms, semiangleRadians: model.semiangleRadians)
    return DirectPtychoHigherOrderConsistency(
      limit: limit, ringWidthInverseAngstroms: ringWidthInverseAngstroms, halves: halves,
      halfBandRMSRadians: disagreement, maximumHalfBandRMSRadians: maximumHalfBandRMSRadians,
      trusted: trusted,
      shownLimit: trusted ? limit : DirectPtychoFitOptions.standardHigherOrderLimit,
      checkedCoordinates: checked,
      checkedPoint: coordinates.point(checked))
  }
}
