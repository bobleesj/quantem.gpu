import Foundation
import Metal

/// The three reconstructions of the fitted optics, as Python's `DirectPtycho.reconstruct(method=...)`.
///
/// `ssb` is conventional single-sideband ptychography (`phase_of_mean`). `phase` solves the linear weak-phase
/// model of every measured scan frequency, all supported aliases jointly, with a damped smoothness prior.
/// `complex` solves the relative log-amplitude and the phase of the same linearized object together.
public enum SSBReconstructionMethod: String, Codable, Sendable, CaseIterable {
  case ssb, phase, complex
}

/// How bright-field pixels are weighted in the regularized normal equations.
public enum SSBDetectorWeighting: String, Codable, Sendable {
  /// Every BF pixel weighs the same (the default).
  case uniform
  /// Each BF pixel weighs 1 / its mean count, floored at 5 % of the median, normalized to mean 1.
  case inverseMean = "inverse_mean"
}

/// Regularization of the `phase` and `complex` direct solves (Python `phase_damping`, `amplitude_damping`,
/// `smoothness_inv_A`, `detector_weighting`). Ignored by `ssb`.
///
/// The prior of each object frequency q is `damping * (1 + (q / q0)^2)^2 * scale`, with q0 =
/// `smoothnessInverseAngstroms` and `scale` the weighted fourth power of the aperture summed over the BF pixels;
/// `smoothnessInverseAngstroms = 0` makes it frequency independent.
public struct SSBDirectRegularization: Codable, Equatable, Sendable {
  /// Damping of the phase, finite and positive.
  public var phaseDamping: Double = 0.03
  /// Damping of the log-amplitude (`complex` only), finite and positive.
  public var amplitudeDamping: Double = 0.03
  /// The prior's frequency scale q0 in inverse Angstrom, finite and nonnegative; 0 = frequency-independent prior.
  public var smoothnessInverseAngstroms: Double = 1
  public var detectorWeighting: SSBDetectorWeighting = .uniform

  public init(
    phaseDamping: Double = 0.03, amplitudeDamping: Double = 0.03,
    smoothnessInverseAngstroms: Double = 1, detectorWeighting: SSBDetectorWeighting = .uniform
  ) {
    self.phaseDamping = phaseDamping
    self.amplitudeDamping = amplitudeDamping
    self.smoothnessInverseAngstroms = smoothnessInverseAngstroms
    self.detectorWeighting = detectorWeighting
  }

  /// The checks of Python `direct_workflow.settings`, with its messages.
  public func validate(for method: SSBReconstructionMethod) throws {
    guard method != .ssb else { return }
    guard phaseDamping.isFinite, phaseDamping > 0 else {
      throw SSBDirectError.invalidRegularization(
        "phase_damping must be finite and positive; got \(phaseDamping).")
    }
    if method == .complex {
      guard amplitudeDamping.isFinite, amplitudeDamping > 0 else {
        throw SSBDirectError.invalidRegularization(
          "amplitude_damping must be finite and positive; got \(amplitudeDamping).")
      }
    }
    guard smoothnessInverseAngstroms.isFinite, smoothnessInverseAngstroms >= 0 else {
      throw SSBDirectError.invalidRegularization(
        "smoothness_inv_A must be nonnegative and finite; use 0 for a flat prior, got \(smoothnessInverseAngstroms)."
      )
    }
  }
}

/// Errors of the regularized direct reconstructions.
public enum SSBDirectError: LocalizedError, Equatable {
  case invalidRegularization(String)
  case unsupportedSampling(Int)
  case notImplemented(SSBReconstructionMethod)
  case tooFewScanPositions
  case noIncidentIntensity
  case noPositiveCounts
  case diverged

  public var errorDescription: String? {
    switch self {
    case .invalidRegularization(let message):
      message
    case .unsupportedSampling(let factor):
      "Regularized direct reconstruction supports output sampling 1x to 4x; got \(factor)x."
    case .notImplemented(let method):
      "The native \(method.rawValue) reconstruction is not implemented yet; use method ssb."
    case .tooFewScanPositions:
      "Direct inversion needs at least 3 scan positions per side; select a larger region."
    case .noIncidentIntensity:
      "No positive incident intensity estimate; check counts and bright-field calibration."
    case .noPositiveCounts:
      "Variance weighting needs positive bright-field counts; use uniform weighting."
    case .diverged:
      "Direct inversion diverged; increase phase_damping or amplitude_damping, or use method phase."
    }
  }
}

extension MetalSSBJointModel {
  /// Reconstruct a model point with one of the three methods.
  ///
  /// `.ssb` is today's `phase_of_mean` path (any regularization is ignored and not recorded); `.phase` and
  /// `.complex` are the regularized direct solves. `.complex` also returns the relative amplitude
  /// |exp(eta + i phi)| = exp(eta) in `amplitude`. The result records `method` and, for the direct solves,
  /// `regularization`.
  public func reconstruct(
    _ point: SSBModelPoint, upsamplingFactor: Int, method: SSBReconstructionMethod,
    regularization: SSBDirectRegularization = .init(), isCancelled: () -> Bool = { false }
  ) throws -> MetalSSBJointPhase {
    switch method {
    case .ssb:
      return try reconstruct(point, upsamplingFactor: upsamplingFactor, isCancelled: isCancelled)
    case .phase, .complex:
      try regularization.validate(for: method)
      guard (1...4).contains(upsamplingFactor) else {
        throw SSBDirectError.unsupportedSampling(upsamplingFactor)
      }
      try point.validate()
      try point.jointParameters.validate()
      if isCancelled() { throw CancellationError() }
      return try reconstructDirect(
        point, upsamplingFactor: upsamplingFactor, complex: method == .complex,
        regularization: regularization,
        isCancelled: isCancelled)
    }
  }
}
