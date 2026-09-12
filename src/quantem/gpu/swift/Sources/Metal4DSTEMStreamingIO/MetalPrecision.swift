import Foundation
import Metal
import MetalCountResources

/// GPU float32 to globally scaled uint16 conversion with complete error accounting.
/// A range pass precedes conversion; only range and error scalars are read by the host.
public final class MetalPrecision {
  public let device: MTLDevice
  public let queue: MTLCommandQueue
  public private(set) var report: [String: Any] = [:]
  private let library: MTLLibrary
  private var pipelines: [String: MTLComputePipelineState] = [:]
  private let rangeState, rangePartials, errorPartials, countPartials, errorState,
    countState: MTLBuffer
  private let partialCount = 65_536
  var exponent = 0
  var coefficients: [Float] = [1, 0, 0, 0]

  public init(device: MTLDevice) throws {
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      throw Self.invalid("An available Metal queue is required.")
    }
    self.queue = queue
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let code = try ["precision", "native_precision", "save_uint16"].map {
      try MetalCountResources.source($0)
    }.joined(separator: "\n")
    library = try device.makeLibrary(source: code, options: options)
    func allocate(_ bytes: Int) throws -> MTLBuffer {
      guard let value = device.makeBuffer(length: bytes, options: .storageModeShared) else {
        throw Self.invalid("Cannot allocate precision workspace.")
      }
      return value
    }
    rangeState = try allocate(16)
    rangePartials = try allocate(partialCount * 16)
    errorPartials = try allocate(partialCount * 16)
    countPartials = try allocate(partialCount * 16)
    errorState = try allocate(16)
    countState = try allocate(32)
    rangeState.contents().storeBytes(
      of: SIMD4<Float>(.infinity, -.infinity, 0, 0), as: SIMD4<Float>.self)
    memset(errorState.contents(), 0, 16)
    memset(countState.contents(), 0, 32)
  }
  /// Extend the global finite range with a float32 region, on the GPU.
  public func includeRange(_ values: MTLBuffer, count: Int) throws {
    try validate(values, count: count, bytes: 4)
    var p = parameters(count)
    p[14] = UInt64(partialCount)
    try precision("precision_range", [values, rangePartials], p: p, count: partialCount)
    try run(
      "native_range_reduce", [rangePartials, rangeState], words: [UInt32(partialCount)], count: 256,
      groupSize: 256, groups: true)
  }
  /// Fix one scale and offset for the entire declared array after its range pass.
  public func calibrate(shape: [Int]) throws {
    guard shape.count == 4, shape.allSatisfy({ $0 > 0 }) else {
      throw Self.invalid("Declare a positive 4D shape before converting scientific storage.")
    }
    let range = rangeState.contents().load(as: SIMD4<Float>.self)
    guard range.x.isFinite, range.y.isFinite, range.z == 0 else {
      throw Self.invalid(
        "Scaled storage requires finite values; preserve nonfinite inputs as float32.")
    }
    guard range.w == 0 else {
      throw Self.invalid(
        "Metal precision conversion cannot preserve float32 subnormal intensities; keep the original float32 file or use CUDA."
      )
    }
    let low = Double(range.x)
    let high = Double(range.y)
    report = [
      "version": 1, "storage": "scaled_uint16", "source_dtype": "float32", "source_shape": shape,
      "intensity_min": low, "intensity_max": high, "scale": high == low ? 1 : (high - low) / 65535,
      "offset": low, "complete": false, "scope": "all saved values",
      "range_scope": "complete source",
      "measurement": "GPU comparison against source", "clipped": 0,
    ]
    configure()
  }
  /// Restore saved scale metadata without another range or conversion pass.
  public func useSavedReport(_ saved: [String: Any]) throws {
    guard saved["storage"] as? String == "scaled_uint16", saved["complete"] as? Bool == true,
      let scale = saved["scale"] as? Double, scale.isFinite, scale > 0,
      let offset = saved["offset"] as? Double, offset.isFinite
    else {
      throw Self.invalid("A complete scaled_uint16 precision report is required.")
    }
    report = saved
    configure()
  }
  private func configure() {
    let magnitude = max(
      abs(report["intensity_min"] as? Double ?? 0), abs(report["intensity_max"] as? Double ?? 0))
    var power: Int32 = 0
    if magnitude != 0 { _ = frexp(magnitude, &power) }
    exponent = -Int(power)
    let scale = ldexp(report["scale"] as! Double, Int32(exponent))
    let offset = ldexp(report["offset"] as! Double, Int32(exponent))
    coefficients = [
      Float(scale), Float(scale - Double(Float(scale))), Float(offset),
      Float(offset - Double(Float(offset))),
    ]
  }
  /// Convert one bounded float32 region and measure its restored-value error on GPU.
  public func convert(_ values: MTLBuffer, count: Int) throws -> MTLBuffer {
    guard !report.isEmpty else {
      throw Self.invalid("Include the complete range and call calibrate before converting.")
    }
    try validate(values, count: count, bytes: 4)
    let codes = try buffer(count * 2)
    var p = parameters(count)
    p[14] = UInt64(partialCount)
    try precision(
      "precision_encode_measure", [values, codes, errorPartials, countPartials], p: p,
      count: partialCount)
    let command = try command()
    let encoder = try encoder(
      command, "native_measure_reduce", [errorPartials, countPartials, errorState, countState])
    var sizes = SIMD2<UInt64>(UInt64(partialCount), UInt64(count))
    encoder.setBytes(&sizes, length: 16, index: 4)
    dispatch(encoder, count: 256, groupSize: 256, groups: true)
    try complete(command)
    return codes
  }
  /// Complete the persisted conversion report using GPU reductions and GPU RMSE.
  @discardableResult public func finish() throws -> [String: Any] {
    let totals = countState.contents().load(as: SIMD4<UInt64>.self)
    guard let shape = report["source_shape"] as? [Int], totals.w == UInt64(shape.reduce(1, *))
    else {
      throw Self.invalid("Every output value must be converted exactly once before finishing.")
    }
    let metrics = try buffer(8)
    try run(
      "native_measure_finish", [errorState, countState, metrics],
      words: [UInt32(bitPattern: Int32(exponent))], count: 1)
    let values = metrics.contents().load(as: SIMD2<Float>.self)
    report["rmse"] = Double(values.x)
    report["max_abs_error"] = Double(values.y)
    report["changed"] = totals.x
    report["positive_to_zero"] = totals.y
    report["overflow"] = totals.z
    report["values"] = totals.w
    report["complete"] = true
    return report
  }
  public func restore(_ codes: MTLBuffer, count: Int) throws -> MTLBuffer {
    guard !report.isEmpty else {
      throw Self.invalid("Load the saved precision report before restoring intensities.")
    }
    try validate(codes, count: count, bytes: 2)
    let output = try buffer(count * 4)
    var p = parameters(count)
    p[3] = 2
    p[13] = 1
    try precision("precision_restore", [codes, output], p: p, count: count)
    return output
  }
  func parameters(_ count: Int) -> [UInt64] {
    var p = [UInt64](repeating: 0, count: 16)
    p[0] = UInt64(count)
    p[4] = 1
    p[5] = UInt64(bitPattern: Int64(exponent))
    return p
  }
  func precision(_ name: String, _ buffers: [MTLBuffer], p: [UInt64], count: Int) throws {
    let command = try command()
    let enc = try encoder(command, name, buffers)
    p.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 6) }
    coefficients.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: 7) }
    dispatch(enc, count: count)
    try complete(command)
  }
  func buffer(_ bytes: Int) throws -> MTLBuffer {
    guard bytes > 0, bytes <= device.maxBufferLength,
      let result = device.makeBuffer(length: bytes, options: .storageModeShared)
    else { throw Self.invalid("Cannot allocate \(bytes) bytes; use a smaller region.") }
    return result
  }
  func validate(_ buffer: MTLBuffer, count: Int, bytes: Int) throws {
    guard count > 0, count <= Int(UInt32.max), buffer.length >= count * bytes,
      buffer.device.registryID == device.registryID
    else {
      throw Self.invalid("Use an in-bounds contiguous region on the selected device.")
    }
  }
  func command() throws -> MTLCommandBuffer {
    guard let result = queue.makeCommandBuffer() else {
      throw Self.invalid("Cannot allocate a command buffer.")
    }
    return result
  }
  func encoder(_ command: MTLCommandBuffer, _ name: String, _ buffers: [MTLBuffer]) throws
    -> MTLComputeCommandEncoder
  {
    if pipelines[name] == nil {
      guard let function = library.makeFunction(name: name) else {
        throw Self.invalid("Missing kernel \(name).")
      }
      pipelines[name] = try device.makeComputePipelineState(function: function)
    }
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw Self.invalid("Cannot allocate a compute encoder.")
    }
    encoder.setComputePipelineState(pipelines[name]!)
    for (index, buffer) in buffers.enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    return encoder
  }
  func dispatch(
    _ encoder: MTLComputeCommandEncoder, count: Int, groupSize: Int = 256, groups: Bool = false
  ) {
    if groups {
      encoder.dispatchThreadgroups(
        MTLSize(width: (count + groupSize - 1) / groupSize, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: groupSize, height: 1, depth: 1))
    } else {
      encoder.dispatchThreads(
        MTLSize(width: count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: groupSize, height: 1, depth: 1))
    }
    encoder.endEncoding()
  }
  func run(
    _ name: String, _ buffers: [MTLBuffer], words: [UInt32], count: Int, groupSize: Int = 256,
    groups: Bool = false
  ) throws {
    let command = try command()
    let enc = try encoder(command, name, buffers)
    words.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) }
    dispatch(enc, count: count, groupSize: groupSize, groups: groups)
    try complete(command)
  }
  func complete(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    if command.status != .completed {
      throw Self.invalid(command.error?.localizedDescription ?? "Metal execution failed.")
    }
  }
  static func invalid(_ value: String) -> Metal4DSTEMStreamingIOError { .invalidRequest(value) }
}
