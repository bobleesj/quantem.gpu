import Foundation

/// Reproducible optimization-only sampling of the complete logical BF selection, as the retired
/// phase-variance search recorded it (`SSBOptimizationResult.brightfieldSampling`).
/// Reconstruction keeps the original geometry and all of its BF pixels.
public struct SSBBrightfieldSampling: Codable, Equatable, Sendable {
  public let policy: String
  public let fraction: Double
  public let totalCount: Int
  public let indices: [Int]

  public init(totalCount: Int, fraction: Double = 1) throws {
    guard totalCount > 0, fraction.isFinite, fraction > 0, fraction <= 1 else {
      throw MetalDirectPtychoError.invalidGeometry(
        "BF optimization fraction must be greater than 0 and at most 1, with a nonempty BF selection."
      )
    }
    self.policy = "uniform-without-replacement-seed42-v1"
    self.fraction = fraction
    self.totalCount = totalCount
    let count = min(totalCount, max(2, Int((Double(totalCount) * fraction).rounded())))
    var order = Array(0..<totalCount)
    if count < totalCount {
      var state: UInt64 = 42
      for position in stride(from: totalCount - 1, through: 1, by: -1) {
        state &+= 0x9e37_79b9_7f4a_7c15
        var value = state
        value = (value ^ (value >> 30)) &* 0xbf58_476d_1ce4_e5b9
        value = (value ^ (value >> 27)) &* 0x94d0_49bb_1331_11eb
        value ^= value >> 31
        order.swapAt(position, Int(value % UInt64(position + 1)))
      }
    }
    indices = Array(order.prefix(count)).sorted()
  }
}
