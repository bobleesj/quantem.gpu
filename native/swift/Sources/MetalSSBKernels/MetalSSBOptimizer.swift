import Foundation

/// One low-order point in the engine's units: C10 and C12 in Angstrom, phi12 in radians.
/// Recorded by the retired phase-variance search (`MetalSSBSavedRun.optimization`).
public struct SSBOptimizationPoint: Codable, Equatable, Sendable {
  public var c10Angstroms: Double
  public var c12Angstroms: Double
  public var phi12Radians: Double

  public var phi12Degrees: Double { phi12Radians * 180 / .pi }

  public init(
    c10Angstroms: Double,
    c12Angstroms: Double,
    phi12Radians: Double
  ) {
    self.c10Angstroms = c10Angstroms
    self.c12Angstroms = c12Angstroms
    self.phi12Radians = phi12Radians
  }
}

public struct SSBOptimizationTrial: Codable, Sendable {
  public let point: SSBOptimizationPoint
  public let loss: Double
  public let stage: String
}

/// The record of the retired native phase-variance search (TPE and Nelder-Mead), kept so that
/// runs saved with it (`MetalSSBSavedRun`) still open. No current fit produces it.
public struct SSBOptimizationResult: Codable, Sendable {
  public let best: SSBOptimizationPoint
  public let loss: Double
  public let trials: [SSBOptimizationTrial]
  public let globalTrials: Int
  public let refinementEvaluations: Int
  public let elapsedSeconds: Double
  public var brightfieldSampling: SSBBrightfieldSampling? = nil
}

public enum SSBOptimizationError: LocalizedError {
  case nonFiniteLoss

  public var errorDescription: String? {
    switch self {
    case .nonFiniteLoss:
      "The exact SSB objective returned a non-finite loss."
    }
  }
}
