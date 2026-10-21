import Foundation

public struct SSBJointTrial: Codable, Sendable {
  public let parameters: SSBJointParameters
  public let loss: Double
  public let stage: String
}

public struct SSBJointOptimizationResult: Codable, Sendable {
  public let best: SSBJointParameters
  public let loss: Double
  public let trials: [SSBJointTrial]
  public let elapsedSeconds: Double
  public let seed: UInt64
  public let globalTrials: Int
  public let refinementIterationsPerStart: Int
  public let refinementMethod: SSBJointRefinement?
}

public enum SSBJointRefinement: String, Codable, Sendable {
  case newton
  case nelderMead = "nelder-mead"
}

// The six-parameter fit is `SSBModelFit` / `SSBAgreementSearch`, the Python resident search. These records
// stay so saved joint results written by the earlier native optimizer remain readable.
