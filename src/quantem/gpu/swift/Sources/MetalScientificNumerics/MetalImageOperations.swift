import Foundation
import Metal
import Metal4DSTEMStreamingIO
import MetalPerformanceShadersGraph

/// A scientific float32 image or complex64 spectrum with row-column axes.
public final class GPUImage {
  public let rows: Int
  public let columns: Int
  public let isComplex: Bool
  public let buffer: MTLBuffer
  public init(buffer: MTLBuffer, rows: Int, columns: Int, isComplex: Bool = false) {
    self.buffer = buffer
    self.rows = rows
    self.columns = columns
    self.isComplex = isComplex
  }
  /// Explicit bounded readback for tests and inspection, never algorithm execution.
  public func values() -> [Float] {
    Array(
      UnsafeBufferPointer(
        start: buffer.contents().assumingMemoryBound(to: Float.self),
        count: rows * columns * (isComplex ? 2 : 1)))
  }
}

/// Reusable image operations used by native scientific workflows.
/// Fourier transforms use MPSGraph; all image, correlation and shift math stays on Metal.
public final class MetalImageOperations {
  public let device: MTLDevice
  public let queue: MTLCommandQueue
  let library: MTLLibrary
  private var pipelines: [String: MTLComputePipelineState] = [:]
  private struct FFTPlan {
    let graph: MPSGraph
    let input, output: MPSGraphTensor
  }
  private var fftPlans: [String: FFTPlan] = [:]
  public init() throws {
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw Self.invalid("A Metal device is required.")
    }
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      throw Self.invalid("A Metal command queue is required.")
    }
    self.queue = queue
    let url = Bundle.module.url(
      forResource: "images", withExtension: "metal", subdirectory: "Resources")!
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    library = try device.makeLibrary(
      source: String(contentsOf: url, encoding: .utf8), options: options)
  }
  public func image(rows: Int, columns: Int, value: Float = 0) throws -> GPUImage {
    let result = try allocate(rows, columns)
    try run(
      "fill_value", [result.buffer], words: [UInt32(rows * columns)], floats: [value],
      count: rows * columns)
    return result
  }
  public func image(values: [Float], rows: Int, columns: Int) throws -> GPUImage {
    guard values.count == rows * columns else {
      throw Self.invalid("Image values must match its row-column shape.")
    }
    let result = try allocate(rows, columns)
    values.withUnsafeBytes { _ = memcpy(result.buffer.contents(), $0.baseAddress!, $0.count) }
    return result
  }
  public func gaussian(_ image: GPUImage, sigma: Float) throws -> GPUImage {
    if sigma <= 0 { return image }
    let radius = Int(2 * sigma)
    guard image.rows > radius, image.columns > radius else {
      throw Self.invalid("Gaussian reflection padding must be smaller than the image.")
    }
    let a = try allocate(image.rows, image.columns)
    let b = try allocate(image.rows, image.columns)
    for (axis, pair) in [(image, a), (a, b)].enumerated() {
      try run(
        "gaussian", [pair.0.buffer, pair.1.buffer],
        words: [UInt32(image.rows), UInt32(image.columns), UInt32(radius), UInt32(axis)],
        floats: [sigma], count: image.rows * image.columns)
    }
    return b
  }
  public func gradientMagnitude(_ image: GPUImage, sigma: Float) throws -> GPUImage {
    let row = try allocate(image.rows, image.columns)
    let column = try allocate(image.rows, image.columns)
    try run(
      "sobel", [image.buffer, row.buffer, column.buffer],
      words: [UInt32(image.rows), UInt32(image.columns)], count: image.rows * image.columns)
    let a = try gaussian(row, sigma: sigma)
    let b = try gaussian(column, sigma: sigma)
    let result = try allocate(image.rows, image.columns)
    try run(
      "magnitude", [a.buffer, b.buffer, result.buffer], words: [UInt32(image.rows * image.columns)],
      count: image.rows * image.columns)
    return result
  }
  /// Window kind: 0 for ones, 1 for Tukey, 2 for the periodic Hann convention.
  public func window(_ image: GPUImage, kind: Int = 0, edge_blend: Float = 0, padding: Int = 0)
    throws -> GPUImage
  {
    guard padding >= 0, (0...2).contains(kind) else {
      throw Self.invalid("Use nonnegative padding and a supported window.")
    }
    let result = try allocate(image.rows + 2 * padding, image.columns + 2 * padding)
    try run(
      "image_window", [image.buffer, result.buffer],
      words: [UInt32(image.rows), UInt32(image.columns), UInt32(padding), UInt32(kind)],
      floats: [edge_blend], count: result.rows * result.columns)
    return result
  }
  public func shifted(_ image: GPUImage, shifts: GPUImage, index: Int) throws -> GPUImage {
    let result = try allocate(image.rows, image.columns)
    try run(
      "shift_image", [image.buffer, shifts.buffer, result.buffer],
      words: [UInt32(image.rows), UInt32(image.columns), UInt32(index), 0],
      count: image.rows * image.columns)
    return result
  }
  public func centered(_ image: GPUImage, window: GPUImage) throws -> GPUImage {
    let result = try allocate(image.rows, image.columns)
    let mean = try buffer(8)
    try run(
      "window_mean", [image.buffer, window.buffer, mean],
      words: [UInt32(image.rows * image.columns)], count: 256, grouped: true)
    try run(
      "window_center", [image.buffer, window.buffer, mean, result.buffer],
      words: [UInt32(image.rows * image.columns)], count: image.rows * image.columns)
    return result
  }
  public func fourier(_ image: GPUImage, inverse: Bool = false) throws -> GPUImage {
    let key = "\(image.rows),\(image.columns),\(inverse)"
    if fftPlans[key] == nil {
      let graph = MPSGraph()
      graph.options = .none
      let input = graph.placeholder(
        shape: [NSNumber(value: image.rows), NSNumber(value: image.columns)],
        dataType: inverse ? .complexFloat32 : .float32, name: "input")
      let descriptor = MPSGraphFFTDescriptor()
      descriptor.inverse = inverse
      descriptor.scalingMode = inverse ? .size : .none
      let transformed = graph.fastFourierTransform(
        input, axes: [0, 1], descriptor: descriptor, name: "transform")
      let output = inverse ? graph.realPartOfTensor(tensor: transformed, name: "real") : transformed
      fftPlans[key] = FFTPlan(graph: graph, input: input, output: output)
    }
    let plan = fftPlans[key]!
    let result = try allocate(image.rows, image.columns, complex: !inverse)
    let shape = [NSNumber(value: image.rows), NSNumber(value: image.columns)]
    plan.graph.run(
      with: queue,
      feeds: [
        plan.input: MPSGraphTensorData(
          image.buffer, shape: shape, dataType: inverse ? .complexFloat32 : .float32)
      ],
      targetOperations: nil,
      resultsDictionary: [
        plan.output: MPSGraphTensorData(
          result.buffer, shape: shape, dataType: inverse ? .float32 : .complexFloat32)
      ])
    return result
  }
  public func blendSpectrum(
    _ reference: GPUImage, _ next: GPUImage, count: Int, shift: GPUImage? = nil
  ) throws -> GPUImage {
    let result = try allocate(reference.rows, reference.columns, complex: true)
    try run(
      "spectrum_blend",
      [reference.buffer, next.buffer, shift?.buffer ?? reference.buffer, result.buffer],
      words: [
        UInt32(reference.rows), UInt32(reference.columns), UInt32(count), shift == nil ? 0 : 1,
      ], count: reference.rows * reference.columns)
    return result
  }
  public func correlation(_ reference: GPUImage, _ image: GPUImage, upsample_factor: Int = 100)
    throws -> GPUImage
  {
    guard upsample_factor >= 1, upsample_factor <= 1024, reference.isComplex, image.isComplex,
      reference.rows == image.rows, reference.columns == image.columns
    else {
      throw Self.invalid(
        "Correlation needs same-shaped complex spectra and upsample_factor from 1 through 1024.")
    }
    let rows = image.rows
    let cols = image.columns
    let count = rows * cols
    let product = try allocate(rows, cols, complex: true)
    try run(
      "spectrum_product", [reference.buffer, image.buffer, product.buffer], words: [UInt32(count)],
      count: count)
    let real = try fourier(product, inverse: true)
    let shift = try allocate(1, 2)
    try run(
      "peak_fit", [real.buffer, shift.buffer], words: [UInt32(rows), UInt32(cols), 0, 0],
      count: 256, grouped: true)
    if upsample_factor > 2 {
      let width = Int(ceil(1.5 * Double(upsample_factor)))
      let row = try allocate(width, rows, complex: true)
      let col = try allocate(cols, width, complex: true)
      let partial = try allocate(width, cols, complex: true)
      let refined = try allocate(width, width)
      let p = [UInt32(rows), UInt32(cols), UInt32(upsample_factor), UInt32(width)]
      try run(
        "dft_kernels", [shift.buffer, row.buffer, col.buffer], words: p,
        count: max(width * rows, cols * width))
      try run(
        "dft_rows", [product.buffer, row.buffer, partial.buffer], words: p, count: width * cols)
      try run(
        "dft_cols", [partial.buffer, col.buffer, refined.buffer], words: p, count: width * width)
      try run(
        "peak_fit", [refined.buffer, shift.buffer],
        words: [UInt32(width), UInt32(width), UInt32(upsample_factor), 0], count: 256, grouped: true
      )
    }
    let wrapped = try self.image(rows: 1, columns: 2)
    try addShift(wrapped, shift, index: 0, wrapRows: rows, wrapColumns: cols)
    return wrapped
  }
  public func addShift(
    _ shifts: GPUImage, _ value: GPUImage, index: Int, wrapRows: Int = 0, wrapColumns: Int = 0
  ) throws {
    try run(
      "shift_record", [shifts.buffer, value.buffer],
      words: [UInt32(wrapRows), UInt32(wrapColumns), UInt32(index), wrapRows > 0 ? 1 : 0], count: 1)
  }
  public func centerShifts(_ shifts: GPUImage, count: Int? = nil) throws {
    try run("shift_center", [shifts.buffer], words: [UInt32(count ?? shifts.rows)], count: 1)
  }
  public func origin(_ image: GPUImage) throws -> [Int] {
    // The peak index is a two-scalar observation; search and tie handling stay on GPU.
    let result = try buffer(8)
    try run(
      "origin", [image.buffer, result], words: [UInt32(image.rows), UInt32(image.columns)],
      count: 256, grouped: true)
    let values = result.contents().assumingMemoryBound(to: UInt32.self)
    return [Int(values[0]), Int(values[1])]
  }
  public func allocatedBytes() -> Int { device.currentAllocatedSize }
  public func synchronize() throws {
    let command = try makeCommand()
    try complete(command)
  }
  func allocate(_ rows: Int, _ columns: Int, complex: Bool = false) throws -> GPUImage {
    GPUImage(
      buffer: try buffer(rows * columns * (complex ? 8 : 4)), rows: rows, columns: columns,
      isComplex: complex)
  }
  func buffer(_ bytes: Int) throws -> MTLBuffer {
    guard bytes > 0, bytes <= device.maxBufferLength,
      let buffer = device.makeBuffer(length: bytes, options: .storageModeShared)
    else {
      throw Self.invalid("Cannot allocate \(bytes) bytes; reduce the requested region.")
    }
    return buffer
  }
  func makeCommand() throws -> MTLCommandBuffer {
    guard let command = queue.makeCommandBuffer() else {
      throw Self.invalid("Cannot allocate a command buffer.")
    }
    return command
  }
  func encoder(_ command: MTLCommandBuffer, _ name: String, _ buffers: [MTLBuffer]) throws
    -> MTLComputeCommandEncoder
  {
    if pipelines[name] == nil {
      guard let fn = library.makeFunction(name: name) else {
        throw Self.invalid("Missing numeric kernel \(name).")
      }
      pipelines[name] = try device.makeComputePipelineState(function: fn)
    }
    guard let enc = command.makeComputeCommandEncoder() else {
      throw Self.invalid("Cannot allocate a compute encoder.")
    }
    enc.setComputePipelineState(pipelines[name]!)
    for (i, b) in buffers.enumerated() { enc.setBuffer(b, offset: 0, index: i) }
    return enc
  }
  func run(
    _ name: String, _ buffers: [MTLBuffer], words: [UInt32], floats: [Float] = [], count: Int,
    grouped: Bool = false
  ) throws {
    let command = try makeCommand()
    let enc = try encoder(command, name, buffers)
    words.withUnsafeBytes { enc.setBytes($0.baseAddress!, length: $0.count, index: buffers.count) }
    if !floats.isEmpty {
      floats.withUnsafeBytes {
        enc.setBytes($0.baseAddress!, length: $0.count, index: buffers.count + 1)
      }
    }
    if grouped {
      enc.dispatchThreadgroups(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    } else {
      enc.dispatchThreads(
        MTLSize(width: count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    enc.endEncoding()
    try complete(command)
  }
  func complete(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    if command.status != .completed {
      throw Self.invalid(command.error?.localizedDescription ?? "Metal execution failed.")
    }
  }
  static func invalid(_ text: String) -> Metal4DSTEMStreamingIOError { .invalidRequest(text) }
}
