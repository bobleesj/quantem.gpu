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
      // Match the GPU's equal-width, half-open bins; the maximum belongs to
      // the final bin. Rounding to 255 intervals shifts interior samples.
      let index = min(binCount - 1, max(0, Int(fraction * Double(binCount))))
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
    guard span > 0 else { return 0.5 }
    let shifted = min(span, max(0, value - minimum))
    switch scale {
    case .linear:
      return shifted / span
    case .logarithmic:
      let low = signedLog(minimum)
      let high = signedLog(maximum)
      return (signedLog(minimum + shifted) - low) / (high - low)
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
      let low = signedLog(minimum)
      let high = signedLog(maximum)
      let value = low + clamped * (high - low)
      return value < 0 ? -expm1(-value) : expm1(value)
    }
  }

  private static func signedLog(_ value: Double) -> Double {
    value < 0 ? -log1p(-value) : log1p(value)
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
  case invalidStatisticsBuffers

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
    case .invalidStatisticsBuffers:
      "Statistics outputs need separate buffers: 8 range bytes and 1024 histogram bytes per image."
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
  public let valueRange: MTLBuffer
  public let histogram: MTLBuffer
  public let minimum: Float
  public let maximum: Float
  public let bins: [UInt32]
}

public final class MetalDisplayStatistics: @unchecked Sendable {
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let rangeFloat32: MTLComputePipelineState
  private let simdRangeUInt32: MTLComputePipelineState
  private let histogramUInt32FromRange: MTLComputePipelineState
  private let histogramFloat32FromRange: MTLComputePipelineState
  private let finishRangeFloat32: MTLComputePipelineState
  private let copyRangeUInt32: MTLComputePipelineState
  private let histogramPairFromRange: MTLComputePipelineState
  private let lock = NSLock()
  /// Reused statistics buffers for `copyAndAnalyzeUInt32Batch`, keyed by image
  /// and scale count, rotating through `statisticsPoolDepth` sets.
  private var statisticsPool:
    [[Int]: (sets: [[(range: MTLBuffer, histograms: [MTLBuffer])]], next: Int)] = [:]
  /// Results of a batch stay valid for this many later batches of the same shape.
  public static let statisticsPoolDepth = 4

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
    rangeFloat32 = try pipeline(MetalDisplayKernels.floatRangeFunction)
    simdRangeUInt32 = try pipeline(MetalDisplayKernels.simdRangeFunction)
    histogramUInt32FromRange = try pipeline(MetalDisplayKernels.histogramFromRangeFunction)
    histogramFloat32FromRange = try pipeline(MetalDisplayKernels.floatHistogramFromRangeFunction)
    finishRangeFloat32 = try pipeline(MetalDisplayKernels.floatFinishRangeFunction)
    copyRangeUInt32 = try pipeline(MetalDisplayKernels.copyRangeFunction)
    histogramPairFromRange = try pipeline(MetalDisplayKernels.histogramPairFromRangeFunction)
  }

  /// Copy each source image into its destination, then analyze the copies, in
  /// one command buffer with one synchronization point: a pass that copies each
  /// image and reduces its range (one read per image), then one pass per image
  /// that counts up to two histograms in threadgroup memory from the
  /// device-resident range. Ranges and bins equal `analyzeUInt32Batch` on the
  /// destinations: every value gets the same bin; only where counts accumulate
  /// changes. Statistics buffers come from a pool reused every
  /// `statisticsPoolDepth` batches of the same shape, so callers copy a result's
  /// range, histogram and bins before that many further batches (the display
  /// surfaces copy them at publication). The copy runs on this object's queue,
  /// ordered after work committed there earlier, such as renders of a
  /// destination's previous use.
  public func copyAndAnalyzeUInt32Batch(
    sources: [MTLBuffer], destinations: [MTLBuffer], rows: Int, columns: Int,
    scales: [MetalDisplayScale] = [.linear, .logarithmic]
  ) throws -> [[MetalUInt32Statistics]] {
    guard sources.count == destinations.count else {
      throw MetalImageRuntimeError.allocation("one destination per copied image")
    }
    guard !sources.isEmpty else { return [] }
    guard !scales.isEmpty else { return sources.map { _ in [] } }
    guard scales.count <= 2 else {
      throw MetalImageRuntimeError.allocation("at most two histogram scales per batch")
    }
    let counts = try destinations.map {
      try validate(values: $0, rows: rows, columns: columns, stride: 4)
    }
    for (source, destination) in zip(sources, destinations) {
      guard source.length >= rows * columns * 4, source !== destination else {
        throw MetalImageRuntimeError.allocation("distinct copy source of the exact image shape")
      }
    }
    lock.lock()
    defer { lock.unlock() }
    let key = [sources.count, scales.count]
    var pool = statisticsPool[key] ?? (sets: [], next: 0)
    if pool.sets.count < Self.statisticsPoolDepth {
      pool.sets.append(
        try sources.map { _ in
          (
            range: try makeBuffer(length: 8, purpose: "batch UInt32 range"),
            histograms: try scales.map { _ in try makeHistogramBuffer() }
          )
        })
      pool.next = pool.sets.count - 1
    }
    let set = pool.sets[pool.next]
    pool.next = (pool.next + 1) % Self.statisticsPoolDepth
    statisticsPool[key] = pool
    for entry in set {
      let pointer = entry.range.contents().assumingMemoryBound(to: UInt32.self)
      pointer[0] = .max
      pointer[1] = 0
      for histogram in entry.histograms { memset(histogram.contents(), 0, histogram.length) }
    }
    guard let command = queue.makeCommandBuffer(),
      let copyEncoder = command.makeComputeCommandEncoder()
    else {
      throw MetalImageRuntimeError.allocation("copy and statistics command")
    }
    copyEncoder.setComputePipelineState(copyRangeUInt32)
    for index in destinations.indices {
      copyEncoder.setBuffer(sources[index], offset: 0, index: 0)
      copyEncoder.setBuffer(destinations[index], offset: 0, index: 1)
      copyEncoder.setBuffer(set[index].range, offset: 0, index: 2)
      var count = UInt32(counts[index])
      copyEncoder.setBytes(&count, length: 4, index: 3)
      dispatch(copyEncoder, pipeline: copyRangeUInt32, count: counts[index])
    }
    copyEncoder.endEncoding()
    // The encoder boundary orders the copies and ranges before the histograms.
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw MetalImageRuntimeError.allocation("batch histogram encoder")
    }
    encoder.setComputePipelineState(histogramPairFromRange)
    let width = max(1, min(histogramPairFromRange.maxTotalThreadsPerThreadgroup, 256))
    for index in destinations.indices {
      let histograms = set[index].histograms
      var layout = SIMD4<UInt32>(
        UInt32(counts[index]), scales[0].rawValue, scales[scales.count - 1].rawValue,
        UInt32(scales.count))
      encoder.setBuffer(destinations[index], offset: 0, index: 0)
      encoder.setBuffer(histograms[0], offset: 0, index: 1)
      encoder.setBuffer(histograms[histograms.count - 1], offset: 0, index: 2)
      encoder.setBuffer(set[index].range, offset: 0, index: 3)
      encoder.setBytes(&layout, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 4)
      // 64 threadgroups per image; each thread strides over the rest.
      encoder.dispatchThreads(
        MTLSize(width: min(counts[index], width * 64), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    }
    encoder.endEncoding()
    try commitAndWait(command)
    return set.map { entry in
      let pointer = entry.range.contents().assumingMemoryBound(to: UInt32.self)
      let minimum = pointer[0] == .max ? 0 : pointer[0]
      let maximum = pointer[0] == .max ? 0 : pointer[1]
      return entry.histograms.map { histogram in
        MetalUInt32Statistics(
          valueRange: entry.range, histogram: histogram,
          minimum: minimum, maximum: maximum, bins: bins(from: histogram))
      }
    }
  }

  /// Encode mixed-shape integer/float statistics without committing, waiting,
  /// reading back, or allocating image/output buffers. The caller owns submission.
  /// Set `updateRange` to false only when each range is already valid for its image.
  /// Requests must use disjoint outputs on this device, with normal hazard tracking.
  /// Example: `try statistics.encode(images, into: command)` followed by the
  /// caller's rendering work and a single `command.commit()`.
  public func encode(
    _ images: [MetalStatisticsRequest], into command: MTLCommandBuffer,
    updateRange: Bool = true
  ) throws {
    guard !images.isEmpty else { return }
    let counts = try images.map {
      try validate(values: $0.values, rows: $0.rows, columns: $0.columns, stride: 4)
    }
    var outputs = Set<ObjectIdentifier>()
    let inputs = Set(images.map { ObjectIdentifier($0.values) })
    for image in images {
      guard image.valueRange.length >= 8, image.histogram.length >= 1024,
        !inputs.contains(ObjectIdentifier(image.valueRange)),
        !inputs.contains(ObjectIdentifier(image.histogram)),
        outputs.insert(ObjectIdentifier(image.valueRange)).inserted,
        outputs.insert(ObjectIdentifier(image.histogram)).inserted
      else { throw MetalImageRuntimeError.invalidStatisticsBuffers }
    }
    guard let clear = command.makeBlitCommandEncoder() else {
      throw MetalImageRuntimeError.allocation("statistics clear encoder")
    }
    for image in images {
      clear.fill(buffer: image.histogram, range: 0..<1024, value: 0)
      if updateRange {
        clear.fill(buffer: image.valueRange, range: 0..<4, value: 255)
        clear.fill(buffer: image.valueRange, range: 4..<8, value: 0)
      }
    }
    clear.endEncoding()
    if updateRange {
      guard let range = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
        throw MetalImageRuntimeError.allocation("statistics range encoder")
      }
      for (index, image) in images.enumerated() {
        let pipeline = image.scalarType == .uint32 ? simdRangeUInt32 : rangeFloat32
        range.setComputePipelineState(pipeline)
        range.setBuffer(image.values, offset: 0, index: 0)
        range.setBuffer(image.valueRange, offset: 0, index: 1)
        var count = UInt32(counts[index])
        range.setBytes(&count, length: 4, index: 2)
        dispatch(range, pipeline: pipeline, count: counts[index])
      }
      range.endEncoding()
    }
    guard let histogram = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
      throw MetalImageRuntimeError.allocation("statistics histogram encoder")
    }
    for (index, image) in images.enumerated() {
      let pipeline =
        image.scalarType == .uint32 ? histogramUInt32FromRange : histogramFloat32FromRange
      histogram.setComputePipelineState(pipeline)
      histogram.setBuffer(image.values, offset: 0, index: 0)
      histogram.setBuffer(image.histogram, offset: 0, index: 1)
      histogram.setBuffer(image.valueRange, offset: 0, index: 3)
      if image.scalarType == .uint32 {
        var parameters = MetalDisplayParameters(
          rows: image.rows, cols: image.columns, low: 0, high: 0, scale: image.scale)
        histogram.setBytes(
          &parameters, length: MemoryLayout<MetalDisplayParameters>.stride, index: 2)
      } else {
        var parameters = MetalFloatDisplayParameters(
          rows: image.rows, cols: image.columns, low: 0, high: 0, scale: image.scale)
        histogram.setBytes(
          &parameters, length: MemoryLayout<MetalFloatDisplayParameters>.stride, index: 2)
        var ordered = UInt32(updateRange ? 1 : 0)
        histogram.setBytes(&ordered, length: 4, index: 4)
      }
      dispatch(histogram, pipeline: pipeline, count: counts[index])
    }
    histogram.endEncoding()
    if updateRange, images.contains(where: { $0.scalarType == .float32 }) {
      guard let finish = command.makeComputeCommandEncoder() else {
        throw MetalImageRuntimeError.allocation("statistics float range encoder")
      }
      finish.setComputePipelineState(finishRangeFloat32)
      for image in images where image.scalarType == .float32 {
        finish.setBuffer(image.valueRange, offset: 0, index: 0)
        finish.dispatchThreads(
          MTLSize(width: 1, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
      }
      finish.endEncoding()
    }
  }

  public func analyzeUInt32(
    values: MTLBuffer, rows: Int, columns: Int, scale: MetalDisplayScale
  ) throws -> MetalUInt32Statistics {
    try analyzeUInt32Batch(values: [values], rows: rows, columns: columns, scales: [scale])[0][0]
  }

  /// Analyze equal-shaped images and scales in one submission and one completion wait.
  /// Convenience API: allocates outputs. Use encode for reusable output buffers.
  public func analyzeUInt32Batch(
    values: [MTLBuffer], rows: Int, columns: Int,
    scales: [MetalDisplayScale] = [.linear, .logarithmic]
  ) throws -> [[MetalUInt32Statistics]] {
    guard !values.isEmpty else { return [] }
    guard !scales.isEmpty else { return values.map { _ in [] } }
    let ranges = try values.map { _ in try makeBuffer(length: 8, purpose: "image range") }
    let histograms = try values.map { _ in try scales.map { _ in try makeHistogramBuffer() } }
    lock.lock()
    defer { lock.unlock() }
    guard let command = queue.makeCommandBuffer() else {
      throw MetalImageRuntimeError.allocation("statistics command")
    }
    for (scaleIndex, scale) in scales.enumerated() {
      let images = values.indices.map { index in
        MetalStatisticsRequest(
          values: values[index], rows: rows, columns: columns, scalarType: .uint32,
          scale: scale, valueRange: ranges[index], histogram: histograms[index][scaleIndex])
      }
      try encode(images, into: command, updateRange: scaleIndex == 0)
    }
    try commitAndWait(command)
    return values.indices.map { index in
      let range = ranges[index].contents().assumingMemoryBound(to: UInt32.self)
      return histograms[index].map { histogram in
        MetalUInt32Statistics(
          valueRange: ranges[index], histogram: histogram,
          minimum: range[0], maximum: range[1], bins: bins(from: histogram))
      }
    }
  }

  public func analyzeFloat32(
    values: MTLBuffer, rows: Int, columns: Int, scale: MetalDisplayScale
  ) throws -> MetalFloat32Statistics {
    let range = try makeBuffer(length: 8, purpose: "float image range")
    let histogram = try makeHistogramBuffer()
    lock.lock()
    defer { lock.unlock() }
    guard let command = queue.makeCommandBuffer() else {
      throw MetalImageRuntimeError.allocation("statistics command")
    }
    try encode(
      [
        MetalStatisticsRequest(
          values: values, rows: rows, columns: columns, scalarType: .float32,
          scale: scale, valueRange: range, histogram: histogram)
      ], into: command)
    try commitAndWait(command)
    let limits = range.contents().assumingMemoryBound(to: Float.self)
    return MetalFloat32Statistics(
      valueRange: range, histogram: histogram, minimum: limits[0],
      maximum: limits[1], bins: bins(from: histogram))
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
    guard count <= Int(UInt32.max), count <= Int.max / stride else {
      throw MetalImageRuntimeError.invalidShape(rows: rows, columns: columns)
    }
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
    func threshold(_ fraction: Double) -> UInt32 {
      let value = MetalHistogramDisplayContract.rawValue(
        fraction: fraction,
        minimum: Double(minimum), maximum: Double(maximum),
        scale: scale == .logarithmic ? .logarithmic : .linear)!
      return UInt32(min(Double(maximum), max(Double(minimum), value)).rounded())
    }
    return MetalDisplayParameters(
      rows: rows,
      cols: columns,
      low: threshold(contrastLow),
      high: threshold(contrastHigh),
      scale: scale
    )
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
    func threshold(_ fraction: Double) -> Float {
      Float(
        MetalHistogramDisplayContract.rawValue(
          fraction: fraction,
          minimum: Double(statistics.minimum), maximum: Double(statistics.maximum),
          scale: scale == .logarithmic ? .logarithmic : .linear) ?? 0)
    }
    return MetalFloatDisplayParameters(
      rows: rows,
      cols: columns,
      low: threshold(contrastLow),
      high: threshold(contrastHigh),
      scale: scale
    )
  }

}
