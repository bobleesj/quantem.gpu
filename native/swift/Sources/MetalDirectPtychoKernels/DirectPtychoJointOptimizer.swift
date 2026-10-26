import Foundation

public struct DirectPtychoJointTrial: Codable, Sendable {
  public let parameters: DirectPtychoJointParameters
  public let loss: Double
  public let stage: String
}

public struct DirectPtychoJointOptimizationResult: Codable, Sendable {
  public let best: DirectPtychoJointParameters
  public let loss: Double
  public let trials: [DirectPtychoJointTrial]
  public let elapsedSeconds: Double
  public let seed: UInt64
  public let globalTrials: Int
  public let refinementIterationsPerStart: Int
  public let refinementMethod: DirectPtychoJointRefinement?
}

public enum DirectPtychoJointRefinement: String, Codable, Sendable {
  case newton
  case nelderMead = "nelder-mead"
}

// The six-parameter fit is `DirectPtychoModelFit` / `DirectPtychoAgreementSearch`, the Python resident search. These records
// stay so saved joint results written by the earlier native optimizer remain readable.
