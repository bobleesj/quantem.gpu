import Foundation

/// Float shaders use signed log1p of values, unlike shifted unsigned counts.
public enum MetalFloatDisplayMapping {
  public static func rawValue(_ fraction: Double, bounds: SIMD2<Double>, logarithmic: Bool)
    -> Double
  {
    let fraction = min(1, max(0, fraction))
    guard logarithmic else { return bounds.x + fraction * (bounds.y - bounds.x) }
    let low = signedLog(bounds.x)
    let high = signedLog(bounds.y)
    let value = low + fraction * (high - low)
    return value < 0 ? -expm1(-value) : expm1(value)
  }

  public static func fraction(_ value: Double, bounds: SIMD2<Double>, logarithmic: Bool) -> Double {
    if value.isNaN { return 0 }
    if value.isInfinite { return value > 0 ? 1 : 0 }
    guard bounds.y > bounds.x else { return 0.5 }
    let low = logarithmic ? signedLog(bounds.x) : bounds.x
    let high = logarithmic ? signedLog(bounds.y) : bounds.y
    let transformed = logarithmic ? signedLog(value) : value
    return min(1, max(0, (transformed - low) / (high - low)))
  }

  public static func fractions(_ values: SIMD2<Double>, bounds: SIMD2<Double>, logarithmic: Bool)
    -> SIMD2<Double>
  {
    let low = min(0.99, fraction(min(values.x, values.y), bounds: bounds, logarithmic: logarithmic))
    let high = max(
      low + 0.01, fraction(max(values.x, values.y), bounds: bounds, logarithmic: logarithmic))
    return SIMD2(low, min(1, high))
  }

  private static func signedLog(_ value: Double) -> Double {
    value < 0 ? -log1p(-value) : log1p(value)
  }
}

public enum MetalRawContrastMapping {
  public static func fractions(
    for rawRange: SIMD2<Double>,
    minimum: Double,
    maximum: Double,
    scale: MetalHistogramReferenceScale
  ) -> SIMD2<Double> {
    var low = MetalHistogramDisplayContract.normalizedFraction(
      value: min(rawRange.x, rawRange.y),
      minimum: minimum,
      maximum: maximum,
      scale: scale
    )
    var high = MetalHistogramDisplayContract.normalizedFraction(
      value: max(rawRange.x, rawRange.y),
      minimum: minimum,
      maximum: maximum,
      scale: scale
    )
    if high - low < 0.01 {
      if low >= 0.99 {
        low = 0.99
        high = 1
      } else {
        high = min(1, low + 0.01)
      }
    }
    return SIMD2(low, high)
  }
}

/// Display thresholds only; never changes counts, histograms, or normalization.
public enum MetalDisplayLimits {
  /// A nonconstant count image needs at least one count between display limits.
  /// Otherwise sub-count histogram windows render every pixel at the midpoint.
  public static func resolvedInteger(
    _ raw: SIMD2<Double>, bounds: SIMD2<UInt32>
  ) -> SIMD2<UInt32> {
    var low = UInt32(min(Double(bounds.y), max(Double(bounds.x), raw.x)).rounded())
    var high = UInt32(min(Double(bounds.y), max(Double(bounds.x), raw.y)).rounded())
    if low == high, bounds.y > bounds.x {
      if high < bounds.y { high += 1 } else { low -= 1 }
    }
    return SIMD2(low, high)
  }

  public static func integer(_ raw: SIMD2<Double>?) -> SIMD2<UInt32>? {
    guard let raw, raw.x.isFinite, raw.y.isFinite,
      raw.x >= 0, raw.y <= Double(UInt32.max), raw.x < raw.y
    else { return nil }
    let result = SIMD2(UInt32(raw.x.rounded()), UInt32(raw.y.rounded()))
    return result.x < result.y ? result : nil
  }

  public static func floating(_ raw: SIMD2<Double>?) -> SIMD2<Float>? {
    guard let raw, raw.x.isFinite, raw.y.isFinite, raw.x < raw.y else { return nil }
    let result = SIMD2(Float(raw.x), Float(raw.y))
    return result.x.isFinite && result.y.isFinite && result.x < result.y ? result : nil
  }
}
