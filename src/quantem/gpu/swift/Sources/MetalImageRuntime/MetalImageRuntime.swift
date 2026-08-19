import Foundation
import Metal
import MetalDisplayKernels

public struct MetalHistogramContrast: Equatable, Sendable {
  public let low: Double
  public let high: Double

  public init(low: Double, high: Double) {
    self.low = low
    self.high = high
  }

  public static func percentileWindow(
    bins: [UInt32],
    lowerPercentile: Double = 0.01,
    upperPercentile: Double = 0.99,
    minimumWidth: Double = 0.01
  ) -> MetalHistogramContrast {
    guard !bins.isEmpty else { return MetalHistogramContrast(low: 0, high: 1) }
    let total = bins.reduce(UInt64(0)) { $0 + UInt64($1) }
    guard total > 0 else { return MetalHistogramContrast(low: 0, high: 1) }
    let lowerTarget = UInt64((Double(total - 1) * lowerPercentile).rounded(.down))
    let upperTarget = UInt64((Double(total - 1) * upperPercentile).rounded(.down))
    let denominator = Double(max(1, bins.count - 1))
    let low = Double(quantileBin(bins: bins, target: lowerTarget)) / denominator
    let high = Double(quantileBin(bins: bins, target: upperTarget)) / denominator
    if high - low >= minimumWidth {
      return MetalHistogramContrast(low: low, high: high)
    }
    let center = (low + high) / 2
    let fittedLow = max(0, min(1 - minimumWidth, center - minimumWidth / 2))
    return MetalHistogramContrast(low: fittedLow, high: fittedLow + minimumWidth)
  }

  private static func quantileBin(bins: [UInt32], target: UInt64) -> Int {
    var cumulative: UInt64 = 0
    for (index, count) in bins.enumerated() {
      cumulative += UInt64(count)
      if cumulative > target { return index }
    }
    return bins.count - 1
  }
}

public enum MetalHistogramReferenceScale: String, Codable, Equatable, Sendable {
  case linear
  case logarithmic
}

public enum MetalHistogramIntervalZone: String, Equatable, Sendable {
  case lowTail = "low_tail"
  case active
  case highTail = "high_tail"
}

public struct MetalHistogramReference: Equatable, Sendable {
  public let bins: [UInt32]
  public let finiteMinimum: Double?
  public let finiteMaximum: Double?
  public let invalidCount: Int

  public init(
    bins: [UInt32],
    finiteMinimum: Double?,
    finiteMaximum: Double?,
    invalidCount: Int
  ) {
    self.bins = bins
    self.finiteMinimum = finiteMinimum
    self.finiteMaximum = finiteMaximum
    self.invalidCount = invalidCount
  }
}

public enum MetalHistogramDisplayContract {
  public static let binCount = 256

  public static func reference(
    values: [Double],
    scale: MetalHistogramReferenceScale
  ) -> MetalHistogramReference {
    let finite = values.filter(\.isFinite)
    let invalidCount = values.count - finite.count
    guard let minimum = finite.min(), let maximum = finite.max() else {
      return MetalHistogramReference(
        bins: [UInt32](repeating: 0, count: binCount),
        finiteMinimum: nil,
        finiteMaximum: nil,
        invalidCount: invalidCount
      )
    }
    var bins = [UInt32](repeating: 0, count: binCount)
    for value in finite {
      let fraction = normalizedFraction(
        value: value,
        minimum: minimum,
        maximum: maximum,
        scale: scale
      )
      let index = min(binCount - 1, max(0, Int((fraction * 255).rounded())))
      bins[index] &+= 1
    }
    return MetalHistogramReference(
      bins: bins,
      finiteMinimum: minimum,
      finiteMaximum: maximum,
      invalidCount: invalidCount
    )
  }

  public static func normalizedFraction(
    value: Double,
    minimum: Double,
    maximum: Double,
    scale: MetalHistogramReferenceScale
  ) -> Double {
    guard value.isFinite, minimum.isFinite, maximum.isFinite else { return 0 }
    let span = max(0, maximum - minimum)
    guard span > 0 else { return 0 }
    let shifted = min(span, max(0, value - minimum))
    switch scale {
    case .linear:
      return shifted / span
    case .logarithmic:
      return log1p(shifted) / log1p(span)
    }
  }

  public static func rawValue(
    fraction: Double,
    minimum: Double,
    maximum: Double,
    scale: MetalHistogramReferenceScale
  ) -> Double? {
    guard minimum.isFinite, maximum.isFinite else { return nil }
    let span = max(0, maximum - minimum)
    guard span > 0 else { return minimum }
    let clamped = min(1, max(0, fraction))
    switch scale {
    case .linear:
      return minimum + clamped * span
    case .logarithmic:
      return minimum + expm1(clamped * log1p(span))
    }
  }

  public static func zone(
    fraction: Double,
    low: Double,
    high: Double
  ) -> MetalHistogramIntervalZone {
    if fraction < low { return .lowTail }
    if fraction > high { return .highTail }
    return .active
  }

  public static func height(count: UInt32, maximumCount: UInt32) -> Double {
    let denominator = max(1, log1p(Double(maximumCount)))
    return log1p(Double(count)) / denominator
  }
}

public enum MetalImageRuntimeError: LocalizedError {
  case allocation(String)
  case missingFunction(String)
  case invalidShape(rows: Int, columns: Int)
  case inputBufferTooSmall(required: Int, actual: Int)
  case commandExecution(String)

  public var errorDescription: String? {
    switch self {
    case .allocation(let purpose):
      "Metal could not allocate the \(purpose)."
    case .missingFunction(let name):
      "MetalImageRuntime is missing the \(name) function."
    case .invalidShape(let rows, let columns):
      "The image shape \(rows)×\(columns) is invalid."
    case .inputBufferTooSmall(let required, let actual):
      "The image requires \(required) bytes, but its buffer contains \(actual)."
    case .commandExecution(let message):
      "Metal image statistics failed: \(message)"
    }
  }
}

public struct MetalUInt32Statistics: @unchecked Sendable {
  public let valueRange: MTLBuffer
  public let histogram: MTLBuffer
  public let minimum: UInt32
  public let maximum: UInt32
  public let bins: [UInt32]
}

public struct MetalFloat32Statistics: @unchecked Sendable {
  public let orderedValueRange: MTLBuffer
  public let histogram: MTLBuffer
  public let minimum: Float
  public let maximum: Float
  public let bins: [UInt32]
}

public final class MetalDisplayStatistics: @unchecked Sendable {
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let rangeUInt32: MTLComputePipelineState
  private let rangeFloat32: MTLComputePipelineState
  private let histogramUInt32: MTLComputePipelineState
  private let histogramFloat32: MTLComputePipelineState
  private let lock = NSLock()

  public init(device: MTLDevice, commandQueue: MTLCommandQueue? = nil) throws {
    self.device = device
    guard let queue = commandQueue ?? device.makeCommandQueue() else {
      throw MetalImageRuntimeError.allocation("statistics command queue")
    }
    self.queue = queue
    let library = try MetalDisplayKernels.makeLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalImageRuntimeError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    rangeUInt32 = try pipeline(MetalDisplayKernels.rangeFunction)
    rangeFloat32 = try pipeline(MetalDisplayKernels.floatRangeFunction)
    histogramUInt32 = try pipeline(MetalDisplayKernels.histogramFunction)
    histogramFloat32 = try pipeline(MetalDisplayKernels.floatHistogramFunction)
  }

  public func analyzeUInt32(
    values: MTLBuffer,
    rows: Int,
    columns: Int,
    scale: MetalDisplayScale
  ) throws -> MetalUInt32Statistics {
    let count = try validate(
      values: values,
      rows: rows,
      columns: columns,
      stride: MemoryLayout<UInt32>.stride
    )
    let range = try makeBuffer(
      length: 2 * MemoryLayout<UInt32>.stride,
      purpose: "UInt32 range buffer"
    )
    let rangePointer = range.contents().bindMemory(to: UInt32.self, capacity: 2)
    rangePointer[0] = .max
    rangePointer[1] = 0
    lock.lock()
    defer { lock.unlock() }
    try runRange(pipeline: rangeUInt32, values: values, range: range, count: count)
    let minimum = rangePointer[0] == .max ? 0 : rangePointer[0]
    let maximum = rangePointer[0] == .max ? 0 : rangePointer[1]
    let histogram = try makeHistogramBuffer()
    var parameters = MetalDisplayParameters(
      rows: rows,
      cols: columns,
      low: minimum,
      high: maximum,
      scale: scale
    )
    try runHistogram(
      pipeline: histogramUInt32,
      values: values,
      histogram: histogram,
      parameters: &parameters,
      count: count
    )
    return MetalUInt32Statistics(
      valueRange: range,
      histogram: histogram,
      minimum: minimum,
      maximum: maximum,
      bins: bins(from: histogram)
    )
  }

  public func analyzeFloat32(
    values: MTLBuffer,
    rows: Int,
    columns: Int,
    scale: MetalDisplayScale
  ) throws -> MetalFloat32Statistics {
    let count = try validate(
      values: values,
      rows: rows,
      columns: columns,
      stride: MemoryLayout<Float>.stride
    )
    let range = try makeBuffer(
      length: 2 * MemoryLayout<UInt32>.stride,
      purpose: "Float32 ordered range buffer"
    )
    let rangePointer = range.contents().bindMemory(to: UInt32.self, capacity: 2)
    rangePointer[0] = .max
    rangePointer[1] = 0
    lock.lock()
    defer { lock.unlock() }
    try runRange(pipeline: rangeFloat32, values: values, range: range, count: count)
    let hasFiniteValues = rangePointer[0] != .max
    let minimum = hasFiniteValues ? decodeOrderedFloat(rangePointer[0]) : 0
    let maximum = hasFiniteValues ? decodeOrderedFloat(rangePointer[1]) : 0
    let histogram = try makeHistogramBuffer()
    var parameters = MetalFloatDisplayParameters(
      rows: rows,
      cols: columns,
      low: minimum,
      high: maximum,
      scale: scale
    )
    try runHistogram(
      pipeline: histogramFloat32,
      values: values,
      histogram: histogram,
      parameters: &parameters,
      count: count
    )
    return MetalFloat32Statistics(
      orderedValueRange: range,
      histogram: histogram,
      minimum: minimum,
      maximum: maximum,
      bins: bins(from: histogram)
    )
  }

  private func validate(
    values: MTLBuffer,
    rows: Int,
    columns: Int,
    stride: Int
  ) throws -> Int {
    guard rows > 0, columns > 0, rows <= Int.max / columns else {
      throw MetalImageRuntimeError.invalidShape(rows: rows, columns: columns)
    }
    let count = rows * columns
    let required = count * stride
    guard values.length >= required else {
      throw MetalImageRuntimeError.inputBufferTooSmall(
        required: required,
        actual: values.length
      )
    }
    return count
  }

  private func makeBuffer(length: Int, purpose: String) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: length, options: .storageModeShared) else {
      throw MetalImageRuntimeError.allocation(purpose)
    }
    return buffer
  }

  private func makeHistogramBuffer() throws -> MTLBuffer {
    let histogram = try makeBuffer(
      length: MetalHistogramDisplayContract.binCount * MemoryLayout<UInt32>.stride,
      purpose: "256-bin histogram"
    )
    memset(histogram.contents(), 0, histogram.length)
    return histogram
  }

  private func runRange(
    pipeline: MTLComputePipelineState,
    values: MTLBuffer,
    range: MTLBuffer,
    count: Int
  ) throws {
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw MetalImageRuntimeError.allocation("range command") }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(values, offset: 0, index: 0)
    encoder.setBuffer(range, offset: 0, index: 1)
    var count32 = UInt32(count)
    encoder.setBytes(&count32, length: MemoryLayout<UInt32>.stride, index: 2)
    dispatch(encoder, pipeline: pipeline, count: count)
    encoder.endEncoding()
    try commitAndWait(command)
  }

  private func runHistogram<Parameters>(
    pipeline: MTLComputePipelineState,
    values: MTLBuffer,
    histogram: MTLBuffer,
    parameters: inout Parameters,
    count: Int
  ) throws {
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw MetalImageRuntimeError.allocation("histogram command") }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(values, offset: 0, index: 0)
    encoder.setBuffer(histogram, offset: 0, index: 1)
    withUnsafeBytes(of: &parameters) { bytes in
      encoder.setBytes(bytes.baseAddress!, length: bytes.count, index: 2)
    }
    dispatch(encoder, pipeline: pipeline, count: count)
    encoder.endEncoding()
    try commitAndWait(command)
  }

  private func dispatch(
    _ encoder: MTLComputeCommandEncoder,
    pipeline: MTLComputePipelineState,
    count: Int
  ) {
    let width = max(1, min(pipeline.maxTotalThreadsPerThreadgroup, 256))
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1)
    )
  }

  private func commitAndWait(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error {
      throw MetalImageRuntimeError.commandExecution(error.localizedDescription)
    }
  }

  private func bins(from histogram: MTLBuffer) -> [UInt32] {
    let pointer = histogram.contents().bindMemory(
      to: UInt32.self,
      capacity: MetalHistogramDisplayContract.binCount
    )
    return Array(
      UnsafeBufferPointer(start: pointer, count: MetalHistogramDisplayContract.binCount)
    )
  }

  private func decodeOrderedFloat(_ ordered: UInt32) -> Float {
    let bits = (ordered & 0x8000_0000) != 0 ? ordered ^ 0x8000_0000 : ~ordered
    return Float(bitPattern: bits)
  }
}

public struct MetalUInt32SurfaceState: @unchecked Sendable {
  public let values: MTLBuffer
  public let statistics: MetalUInt32Statistics
  public let rows: Int
  public let columns: Int
  public private(set) var scale: MetalDisplayScale
  public private(set) var colormap: MetalColormap
  public private(set) var contrastLow: Double
  public private(set) var contrastHigh: Double

  public init(
    values: MTLBuffer,
    statistics: MetalUInt32Statistics,
    rows: Int,
    columns: Int,
    scale: MetalDisplayScale,
    colormap: MetalColormap,
    contrastLow: Double = 0,
    contrastHigh: Double = 1
  ) {
    self.values = values
    self.statistics = statistics
    self.rows = rows
    self.columns = columns
    self.scale = scale
    self.colormap = colormap
    self.contrastLow = contrastLow
    self.contrastHigh = contrastHigh
  }

  @discardableResult
  public mutating func configure(
    scale: MetalDisplayScale,
    colormap: MetalColormap,
    contrastLow: Double,
    contrastHigh: Double
  ) -> Bool {
    let low = min(0.99, max(0, contrastLow))
    let high = min(1, max(low + 0.01, contrastHigh))
    guard
      self.scale != scale || self.colormap != colormap
        || self.contrastLow != low || self.contrastHigh != high
    else { return false }
    self.scale = scale
    self.colormap = colormap
    self.contrastLow = low
    self.contrastHigh = high
    return true
  }

  public func displayParameters() -> MetalDisplayParameters {
    let minimum = statistics.minimum
    let maximum = max(minimum, statistics.maximum)
    let span = Double(maximum - minimum)
    return MetalDisplayParameters(
      rows: rows,
      cols: columns,
      low: minimum
        + UInt32((span * rawDisplayFraction(contrastLow, span: span)).rounded()),
      high: minimum
        + UInt32((span * rawDisplayFraction(contrastHigh, span: span)).rounded()),
      scale: scale
    )
  }

  private func rawDisplayFraction(_ fraction: Double, span: Double) -> Double {
    guard scale == .logarithmic else { return fraction }
    let safeSpan = max(1.0e-20, span)
    return expm1(log1p(safeSpan) * fraction) / safeSpan
  }
}

public struct MetalFloat32SurfaceState: @unchecked Sendable {
  public let values: MTLBuffer
  public let statistics: MetalFloat32Statistics
  public let rows: Int
  public let columns: Int
  public private(set) var scale: MetalDisplayScale
  public private(set) var colormap: MetalColormap
  public private(set) var contrastLow: Double
  public private(set) var contrastHigh: Double

  public init(
    values: MTLBuffer,
    statistics: MetalFloat32Statistics,
    rows: Int,
    columns: Int,
    scale: MetalDisplayScale,
    colormap: MetalColormap,
    contrastLow: Double = 0,
    contrastHigh: Double = 1
  ) {
    self.values = values
    self.statistics = statistics
    self.rows = rows
    self.columns = columns
    self.scale = scale
    self.colormap = colormap
    self.contrastLow = contrastLow
    self.contrastHigh = contrastHigh
  }

  @discardableResult
  public mutating func configure(
    scale: MetalDisplayScale,
    colormap: MetalColormap,
    contrastLow: Double,
    contrastHigh: Double
  ) -> Bool {
    let low = min(0.99, max(0, contrastLow))
    let high = min(1, max(low + 0.01, contrastHigh))
    guard
      self.scale != scale || self.colormap != colormap
        || self.contrastLow != low || self.contrastHigh != high
    else { return false }
    self.scale = scale
    self.colormap = colormap
    self.contrastLow = low
    self.contrastHigh = high
    return true
  }

  public func displayParameters() -> MetalFloatDisplayParameters {
    let span = Double(statistics.maximum - statistics.minimum)
    return MetalFloatDisplayParameters(
      rows: rows,
      cols: columns,
      low: statistics.minimum
        + Float(span * rawDisplayFraction(contrastLow, span: span)),
      high: statistics.minimum
        + Float(span * rawDisplayFraction(contrastHigh, span: span)),
      scale: scale
    )
  }

  private func rawDisplayFraction(_ fraction: Double, span: Double) -> Double {
    guard scale == .logarithmic else { return fraction }
    let safeSpan = max(1.0e-20, span)
    return expm1(log1p(safeSpan) * fraction) / safeSpan
  }
}
