import Foundation
import Metal
import MetalCountResources

/// Exact resident count reads shared by scientific consumers of encoded and packed data.
/// Reads return owned, row-major native integer buffers. The caller serializes
/// operations and release; no full acquisition is decoded by this contract.
public protocol MetalResidentCounts: AnyObject {
  var shape: [Int] { get }
  var itemBytes: Int { get }
  var hotPixelIndices: [Int] { get }
  var hotPixelCorrection: String { get }
  var device: MTLDevice { get }
  var readyFrames: Int { get }
  var isReleased: Bool { get }
  var residentBytes: Int { get }
  var representation: Metal4DSTEMResidentRepresentation { get }
  func read(_ frames: Range<Int>) throws -> MTLBuffer
  func encodeRead(_ frames: Range<Int>, into result: MTLBuffer, command: MTLCommandBuffer) throws
  func checkErrors() throws
  func countMeans() throws -> (diffraction: MTLBuffer, brightField: MTLBuffer)
  func releaseResidentStorage()
}

extension MetalResidentCounts {
  /// Read completion is automatic; advanced consumers may compose encodeRead
  /// with subsequent kernels and call checkErrors after their command finishes.
  public func read(_ frames: Range<Int>) throws -> MTLBuffer {
    guard !isReleased, !frames.isEmpty, frames.lowerBound >= 0,
      frames.upperBound <= readyFrames, frames.count <= 8192
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest("Read 1...8192 available resident frames.")
    }
    let bytes = frames.count * shape[2] * shape[3] * itemBytes
    guard bytes <= device.maxBufferLength,
      let result = device.makeBuffer(length: bytes, options: .storageModeShared),
      let queue = device.makeCommandQueue(), let command = queue.makeCommandBuffer()
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "resident count region", bytes: UInt64(bytes))
    }
    try encodeRead(frames, into: result, command: command)
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Resident read failed: \(String(describing: command.error))")
    }
    try checkErrors()
    return result
  }
  public func checkErrors() throws {
    guard !isReleased else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The resident counts were released; load them again.")
    }
  }
}

extension MetalEncodedSource: MetalResidentCounts {
  public var hotPixelCorrection: String { "median" }
  public var representation: Metal4DSTEMResidentRepresentation { .encoded }
  public func countMeans() throws -> (diffraction: MTLBuffer, brightField: MTLBuffer) {
    guard !isReleased, readyFrames == shape[0] * shape[1] else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Complete the resident load before requesting means.")
    }
    return (meanDiffraction, meanBrightField)
  }
}

/// Shared bounded GPU reduction for count sources without loading-time summaries.
/// Integer accumulation precedes the same float32 mean conversion used by encoded loading.
enum ResidentCountMeans {
  static func calculate(_ source: any MetalResidentCounts) throws
    -> (diffraction: MTLBuffer, brightField: MTLBuffer)
  {
    let pixels = source.shape[2] * source.shape[3]
    let frames = source.shape[0] * source.shape[1]
    guard !source.isReleased, source.readyFrames == frames,
      let queue = source.device.makeCommandQueue(),
      let sum = source.device.makeBuffer(length: pixels * 8, options: .storageModeShared),
      let dp = source.device.makeBuffer(length: pixels * 4, options: .storageModeShared),
      let bf = source.device.makeBuffer(length: frames * 4, options: .storageModeShared)
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Load complete counts and free memory before calculating means.")
    }
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try source.device.makeLibrary(
      source: MetalCountResources.source("resident_utilities"), options: options)
    let summary = try source.device.makeComputePipelineState(
      function: library.makeFunction(name: "count_summary")!)
    let bright = try source.device.makeComputePipelineState(
      function: library.makeFunction(name: "count_bright")!)
    let normalize = try source.device.makeComputePipelineState(
      function: library.makeFunction(name: "count_normalize")!)
    for first in stride(from: 0, to: frames, by: 4096) {
      try autoreleasepool {
        let count = min(4096, frames - first)
        let raw = try source.read(first..<(first + count))
        guard let command = queue.makeCommandBuffer() else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            "Could not create count reduction commands.")
        }
        if first == 0 {
          let clear = command.makeBlitCommandEncoder()!
          clear.fill(buffer: sum, range: 0..<sum.length, value: 0)
          clear.endEncoding()
        }
        var parameters = SIMD4<UInt64>(
          UInt64(count), UInt64(pixels), UInt64(source.itemBytes), UInt64(first))
        let a = command.makeComputeCommandEncoder()!
        a.setComputePipelineState(summary)
        a.setBuffer(raw, offset: 0, index: 0)
        a.setBuffer(sum, offset: 0, index: 1)
        a.setBuffer(bf, offset: 0, index: 2)
        a.setBytes(&parameters, length: 32, index: 3)
        a.dispatchThreads(
          MTLSize(width: pixels, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        a.endEncoding()
        let b = command.makeComputeCommandEncoder()!
        b.setComputePipelineState(bright)
        b.setBuffer(raw, offset: 0, index: 0)
        b.setBuffer(bf, offset: 0, index: 1)
        b.setBytes(&parameters, length: 32, index: 2)
        b.dispatchThreadgroups(
          MTLSize(width: count, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
        b.endEncoding()
        if first + count == frames {
          let n = command.makeComputeCommandEncoder()!
          n.setComputePipelineState(normalize)
          n.setBuffer(sum, offset: 0, index: 0)
          n.setBuffer(dp, offset: 0, index: 1)
          var dimensions = SIMD2<UInt32>(UInt32(pixels), UInt32(frames))
          n.setBytes(&dimensions, length: 8, index: 2)
          n.dispatchThreads(
            MTLSize(width: pixels, height: 1, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          n.endEncoding()
        }
        command.commit()
        command.waitUntilCompleted()
        guard command.status == .completed else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            "Count reduction failed: \(String(describing: command.error))")
        }
      }
    }
    return (dp, bf)
  }
}

extension MetalResidentCounts {
  /// Apply the established 3x3 median to marked pixels in bounded decoded reads.
  /// The underlying packed/encoded storage and its caller-owned lifetime are preserved.
  public func correctedHotPixels() throws -> any MetalResidentCounts {
    if hotPixelCorrection == "median" || hotPixelIndices.isEmpty { return self }
    return try MedianCorrectedCounts(self)
  }
}

private final class MedianCorrectedCounts: MetalResidentCounts {
  let source: any MetalResidentCounts
  let hotPixelIndices: [Int]
  let valid: MTLBuffer
  let bad: MTLBuffer
  let pipeline: MTLComputePipelineState
  private var released = false
  private var means: (diffraction: MTLBuffer, brightField: MTLBuffer)?
  var shape: [Int] { source.shape }
  var itemBytes: Int { source.itemBytes }
  var device: MTLDevice { source.device }
  var readyFrames: Int { source.readyFrames }
  var isReleased: Bool { released || source.isReleased }
  var representation: Metal4DSTEMResidentRepresentation { source.representation }
  var hotPixelCorrection: String { "median" }
  var residentBytes: Int {
    source.residentBytes + valid.length + bad.length
      + (means?.diffraction.length ?? 0) + (means?.brightField.length ?? 0)
  }
  init(_ source: any MetalResidentCounts) throws {
    self.source = source
    hotPixelIndices = source.hotPixelIndices
    let pixels = source.shape[2] * source.shape[3]
    guard !source.isReleased, [1, 2, 4].contains(source.itemBytes),
      Set(hotPixelIndices).count == hotPixelIndices.count,
      hotPixelIndices.allSatisfy({ 0..<pixels ~= $0 }),
      let valid = source.device.makeBuffer(length: pixels, options: .storageModeShared),
      let bad = source.device.makeBuffer(
        length: hotPixelIndices.count * 4, options: .storageModeShared)
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Use live integer counts and valid unique detector-mask indices.")
    }
    self.valid = valid
    self.bad = bad
    // Detector masks are small control metadata. Count correction runs on Metal.
    memset(valid.contents(), 1, pixels)
    for (slot, pixel) in hotPixelIndices.enumerated() {
      valid.contents().storeBytes(of: UInt8(0), toByteOffset: pixel, as: UInt8.self)
      bad.contents().storeBytes(of: Int32(pixel), toByteOffset: slot * 4, as: Int32.self)
    }
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try source.device.makeLibrary(
      source: MetalCountResources.source("hot_pixels"), options: options)
    pipeline = try source.device.makeComputePipelineState(
      function: library.makeFunction(name: "hot_median")!)
  }
  func encodeRead(_ frames: Range<Int>, into result: MTLBuffer, command: MTLCommandBuffer) throws {
    try checkErrors()
    try source.encodeRead(frames, into: result, command: command)
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable("Could not encode hot-pixel correction.")
    }
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(result, offset: 0, index: 0)
    encoder.setBuffer(valid, offset: 0, index: 1)
    encoder.setBuffer(bad, offset: 0, index: 2)
    let parameters = [
      hotPixelIndices.count, shape[2], shape[3], frames.count * hotPixelIndices.count, itemBytes,
    ].map(UInt64.init)
    encoder.setBytes(parameters, length: parameters.count * 8, index: 3)
    encoder.dispatchThreads(
      MTLSize(width: frames.count * hotPixelIndices.count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
  }
  func checkErrors() throws {
    guard !isReleased else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Reload released counts before reading corrected pixels.")
    }
    try source.checkErrors()
  }
  func countMeans() throws -> (diffraction: MTLBuffer, brightField: MTLBuffer) {
    try checkErrors()
    if let means { return means }
    let result = try ResidentCountMeans.calculate(self)
    means = result
    return result
  }
  func releaseResidentStorage() {
    released = true
    means = nil
  }
}
