import Foundation

/// G's power over the agreement band and its counted-electron noise (Python `band_power`), from
/// `MetalSSBJointModel.bandPowerSummary()`.
public struct SSBBandPower: Codable, Equatable, Sendable {
  /// P = sum_q w_q sum_k |G(q, k)|^2 over the band frequencies and the active BF pixels: the agreement's upper bound.
  public let power: Double
  /// W = sum_q w_q, with the half-plane weights 1 or 2 (the full-plane band size).
  public let bandWeight: Double
  /// sigma^2 = the mean over k of G(0, k), the counts at k summed over the scan: the Poisson variance of one G(q, k).
  public let noiseVariance: Double
  /// N, the BF pixels of the sums.
  public let brightFieldPixelCount: Int

  public init(power: Double, bandWeight: Double, noiseVariance: Double, brightFieldPixelCount: Int) {
    self.power = power
    self.bandWeight = bandWeight
    self.noiseVariance = noiseVariance
    self.brightFieldPixelCount = brightFieldPixelCount
  }
}

/// Whether the fitted higher orders can be read as the probe's aberrations: Python `higher_order_fit.assess` for
/// counted electrons (docs/performance/ssb/reports/2026-10-21-ssb-higher-order-fit.md).
///
/// - `coherence` rho = (A - F_A) / (P - F_P): the share of G's signal power over the band that the fitted model
///   explains, with A the agreement, P = sum_q w_q sum_k |G(q, k)|^2 (A <= P), and the noise floors F_A = W sigma^2
///   and F_P = W N sigma^2 of the Poisson variance sigma^2 = mean_k G(0, k). It is 1 for a weak-phase object and falls
///   with the dynamical scattering of thick or heavy crystals, which extra free terms then partly fit.
/// - Each term's precision from the covariance 2 sigma^2 (-H)^-1 of all 28 fit coordinates, H the Hessian of A at the
///   fitted point: its standard error (a vector term sqrt(var_x + var_y), a round term |sigma|) and the probe phase it
///   is uncertain by at the aperture edge, (2 pi / lambda) sigma_C alpha^(n+1) / (n+1). nil (infinite) where -H is not
///   positive definite.
/// - `trusted`: coherence >= `coherenceMinimum` and every edge-phase error <= `edgePhaseErrorMaximumRadians`;
///   otherwise `reasons` says why. An untrusted fit keeps its terms.
public struct SSBHigherOrderGuard: Codable, Equatable, Sendable {
  /// On simulated crystals without these aberrations, no fit that returned spurious terms exceeded 0.80.
  public static let coherenceMinimum = 0.9
  /// 3.5 nm of coma or trefoil and 0.15 um of C30 at 30 mrad and 300 kV.
  public static let edgePhaseErrorMaximumRadians = 0.1

  public struct Term: Codable, Equatable, Sendable {
    public let order: Int
    public let symmetry: Int
    /// The standard error of the coefficient (nm); nil where the curvature is not negative definite.
    public let standardErrorNanometers: Double?
    /// The probe phase that standard error is at the aperture edge (rad); nil where it is not determined.
    public let edgePhaseErrorRadians: Double?
    /// Krivanek name, e.g. C21.
    public var name: String { "C\(order)\(symmetry)" }
  }

  /// rho = (A - F_A) / (P - F_P); 0 when the signal P - F_P is not positive.
  public let coherence: Double
  /// The 12 higher-order terms C21 through C56 in fit order.
  public let terms: [Term]
  public let trusted: Bool
  /// Why the guard failed, one sentence per failed condition; empty when trusted.
  public let reasons: [String]
  /// The noise model of the floors and the precision: "poisson counts".
  public let noise: String
  /// A: the fitted agreement (float32 per-frequency powers summed in Double).
  public let agreement: Double
  public let band: SSBBandPower
  /// The thresholds this guard was decided with.
  public let coherenceMinimum: Double
  public let edgePhaseErrorMaximumRadians: Double

  /// The largest edge-phase error over the terms (infinite when one is not determined).
  public var largestEdgePhaseErrorRadians: Double {
    terms.map { $0.edgePhaseErrorRadians ?? .infinity }.max() ?? 0
  }

  /// Decide the guard from the agreement A at the fitted point, the band summary and the Hessian d^2 A / dz^2 in the
  /// fit's 28 dimensionless coordinates, whose higher-order entries are already probe phase (rad) at the aperture edge.
  public static func assess(
    agreement: Double, band: SSBBandPower, hessian: [[Double]],
    wavelengthAngstroms: Double, semiangleRadians: Double,
    coherenceMinimum: Double = SSBHigherOrderGuard.coherenceMinimum,
    edgePhaseErrorMaximumRadians: Double = SSBHigherOrderGuard.edgePhaseErrorMaximumRadians
  ) -> SSBHigherOrderGuard {
    let noise = band.noiseVariance
    let floorAgreement = band.bandWeight * noise
    let signal = band.power - floorAgreement * Double(band.brightFieldPixelCount)
    let coherence = signal > 0 ? (agreement - floorAgreement) / signal : 0
    let variances = coordinateVariances(hessian: hessian, noiseVariance: noise)
    var terms: [Term] = []
    var column = 6
    for (n, m) in SSBFifthOrderCoordinates.orders where n >= 2 {
      let width = m == 0 ? 1 : 2
      // z of a higher-order coordinate is (2 pi / lambda) C alpha^(n+1) / (n+1): its error is the edge phase.
      let edge = variances.map { v in (column..<(column + width)).reduce(0.0) { $0 + max(v[$1], 0) }.squareRoot() }
      let angstroms = edge.map {
        $0 * Double(n + 1) * wavelengthAngstroms / (2 * Double.pi * pow(semiangleRadians, Double(n + 1)))
      }
      terms.append(
        Term(
          order: n, symmetry: m, standardErrorNanometers: angstroms.map { $0 / 10 },
          edgePhaseErrorRadians: edge))
      column += width
    }
    var reasons: [String] = []
    if coherence < coherenceMinimum {
      reasons.append(
        String(
          format:
            "the model explains %.2f of the bright-field signal (below %g): scattering it does not describe, as in a thick or strongly scattering crystal",
          coherence, coherenceMinimum))
    }
    // The first term with the largest error, as Python's max(edge, key=edge.get); undetermined is infinite.
    var worst: Term?
    for term in terms
    where (term.edgePhaseErrorRadians ?? .infinity) > worst.map({ $0.edgePhaseErrorRadians ?? .infinity }) ?? -.infinity {
      worst = term
    }
    if let worst, (worst.edgePhaseErrorRadians ?? .infinity) > edgePhaseErrorMaximumRadians {
      reasons.append(
        String(
          format: "%@ is uncertain by %.2f rad at the aperture edge (above %g): too little signal for its precision",
          worst.name, worst.edgePhaseErrorRadians ?? .infinity, edgePhaseErrorMaximumRadians))
    }
    return SSBHigherOrderGuard(
      coherence: coherence, terms: terms, trusted: reasons.isEmpty, reasons: reasons, noise: "poisson counts",
      agreement: agreement, band: band, coherenceMinimum: coherenceMinimum,
      edgePhaseErrorMaximumRadians: edgePhaseErrorMaximumRadians)
  }

  /// The diagonal of 2 sigma^2 (-H)^-1; nil when -H is not positive definite (not at a maximum) or not finite.
  static func coordinateVariances(hessian: [[Double]], noiseVariance: Double) -> [Double]? {
    let n = hessian.count
    guard n > 0, hessian.allSatisfy({ $0.count == n && $0.allSatisfy(\.isFinite) }) else { return nil }
    let negative = (0..<n).map { i in (0..<n).map { j in -0.5 * (hessian[i][j] + hessian[j][i]) } }
    let (values, vectors) = SSBFifthOrderGradientSearch.symmetricEigen(negative)
    guard let smallest = values.min(), smallest > 0 else { return nil }
    return (0..<n).map { i in
      2 * noiseVariance * (0..<n).reduce(0.0) { $0 + vectors[i][$1] * vectors[i][$1] / values[$1] }
    }
  }
}

extension SSBModelFit {
  /// The guard of a fifth-order fit at dimensionless coordinates `z` with agreement `agreement` (Double-summed), on
  /// `model` at the fit's rotation: the band summary, and the Hessian of the agreement there from forward differences
  /// of exact full-data gradients (`guardHessian`: 29 value-and-gradient evaluations).
  public static func higherOrderGuard(
    model: MetalSSBJointModel, z: [Double], agreement: Double, step: Double = 2e-3,
    progress: (_ completed: Int, _ total: Int) -> Void = { _, _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> SSBHigherOrderGuard {
    let band = try model.bandPowerSummary(isCancelled: isCancelled)
    let hessian = try guardHessian(
      model: model, z: z, step: step, progress: progress, isCancelled: isCancelled)
    return SSBHigherOrderGuard.assess(
      agreement: agreement, band: band, hessian: hessian,
      wavelengthAngstroms: model.wavelengthAngstroms, semiangleRadians: model.semiangleRadians)
  }

  /// d^2 agreement / dz^2 at `z`: symmetrized forward differences of the exact full-data gradients at `z` and at `z`
  /// plus `step` along each coordinate, in batches of 8 candidates so a fit can report progress and stop between
  /// them. Half the evaluations of the central differences of `fifthOrderHessian`; on a measured 512 x 512
  /// acquisition the edge-phase errors of the two agree to 0.3%.
  static func guardHessian(
    model: MetalSSBJointModel, z: [Double], step: Double,
    progress: (_ completed: Int, _ total: Int) -> Void, isCancelled: () -> Bool
  ) throws -> [[Double]] {
    let n = z.count
    let coordinates = SSBFifthOrderCoordinates(
      wavelengthAngstroms: model.wavelengthAngstroms, semiangleRadians: model.semiangleRadians)
    let rows = [z] + (0..<n).map { i -> [Double] in
      var p = z
      p[i] += step
      return p
    }
    var g: [[Double]] = []
    progress(0, rows.count)
    for first in stride(from: 0, to: rows.count, by: 8) {
      if isCancelled() { throw CancellationError() }
      let batch = Array(rows[first..<min(first + 8, rows.count)])
      let values = try model.fifthOrderAgreementsAndGradients(
        batch.map(coordinates.point), packedCoefficients: batch.map(coordinates.pack),
        isCancelled: isCancelled)
      g += values.map { coordinates.gradient(row: $0.gradient) }
      progress(g.count, rows.count)
    }
    var hessian = [[Double]](repeating: [Double](repeating: 0, count: n), count: n)
    for i in 0..<n {
      for j in 0..<n {
        let along = g[1 + j][i] - g[0][i]
        let across = g[1 + i][j] - g[0][j]
        hessian[i][j] = 0.5 * (along + across) / step
      }
    }
    return hessian
  }
}
