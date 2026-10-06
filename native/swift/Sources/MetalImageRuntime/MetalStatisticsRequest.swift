import Metal
import MetalDisplayKernels

/// One image and caller-owned outputs for batched statistics encoding.
/// Outputs require 8 range bytes and 256 UInt32 histogram bins. After completion,
/// range contains two values of `scalarType`; float ranges exclude nonfinite values.
/// Keep all buffers alive and do not read or reuse outputs until completion.
public struct MetalStatisticsRequest {
  public enum ScalarType { case uint32, float32 }
  public let values: MTLBuffer
  public let rows: Int
  public let columns: Int
  public let scalarType: ScalarType
  public let scale: MetalDisplayScale
  public let valueRange: MTLBuffer
  public let histogram: MTLBuffer

  public init(
    values: MTLBuffer, rows: Int, columns: Int, scalarType: ScalarType,
    scale: MetalDisplayScale, valueRange: MTLBuffer, histogram: MTLBuffer
  ) {
    self.values = values
    self.rows = rows
    self.columns = columns
    self.scalarType = scalarType
    self.scale = scale
    self.valueRange = valueRange
    self.histogram = histogram
  }
}
