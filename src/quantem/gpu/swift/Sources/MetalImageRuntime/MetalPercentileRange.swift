import Foundation

/// Display percentile policy in fractions (0.05...0.95 means 5...95 percent).
/// This changes display limits only, never scientific samples.
public struct MetalPercentileRange: Equatable, Sendable {
  public let low: Double
  public let high: Double

  public init(low: Double, high: Double) {
    let low = low.isFinite ? min(0.999, max(0, low)) : 0
    self.low = low
    self.high = high.isFinite ? min(1, max(low + 0.001, high)) : 1
  }

  public func replacing(low: Double? = nil, high: Double? = nil) -> Self {
    if let low, high == nil { return Self(low: min(low, self.high - 0.001), high: self.high) }
    return Self(low: low ?? self.low, high: high ?? self.high)
  }

  /// Resolve a policy against this image's histogram, not another image's limits.
  public func window(bins: [UInt32]) -> MetalHistogramContrast {
    if low == 0, high == 1 { return MetalHistogramContrast(low: 0, high: 1) }
    return MetalHistogramContrast.percentileWindow(
      bins: bins, lowerPercentile: low, upperPercentile: high)
  }

  /// Convert a normalized histogram position to a percentile using its CDF.
  /// Positions follow the existing bin-center display convention (N-1 intervals).
  public static func percentile(at position: Double, bins: [UInt32]) -> Double {
    guard position.isFinite, !bins.isEmpty else { return 0 }
    if position <= 0 { return 0 }
    if position >= 1 { return 1 }
    let total = bins.reduce(UInt64(0)) { $0 + UInt64($1) }
    guard total > 0 else { return position }
    let coordinate = position * Double(bins.count - 1)
    let index = Int(coordinate)
    let preceding = bins.prefix(index).reduce(UInt64(0)) { $0 + UInt64($1) }
    return (Double(preceding) + Double(bins[index]) * (coordinate - Double(index))) / Double(total)
  }
}
