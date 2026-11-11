import Foundation

/// One low-order point in the engine's units: C10 and C12 in Angstrom, phi12 in radians.
/// Recorded by the retired phase-variance search (`MetalDirectSavedRun.optimization`).
public struct DirectOptimizationPoint: Codable, Equatable, Sendable {
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

public struct DirectOptimizationTrial: Codable, Sendable {
  public let point: DirectOptimizationPoint
  public let loss: Double
  public let stage: String
}

/// The record of the retired native phase-variance search (TPE and Nelder-Mead), kept so that
/// runs saved with it (`MetalDirectSavedRun`) still open. No current fit produces it.
public struct DirectOptimizationResult: Codable, Sendable {
  public let best: DirectOptimizationPoint
  public let loss: Double
  public let trials: [DirectOptimizationTrial]
  public let globalTrials: Int
  public let refinementEvaluations: Int
  public let elapsedSeconds: Double
  public var brightfieldSampling: DirectBrightfieldSampling? = nil
}

public enum DirectOptimizationError: LocalizedError {
  case nonFiniteLoss

  public var errorDescription: String? {
    switch self {
    case .nonFiniteLoss:
      "The exact direct ptychography objective returned a non-finite loss."
    }
  }
}
