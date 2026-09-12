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
  private struct MatrixPlan {
    let graph: MPSGraph
    let left, right, output: MPSGraphTensor
  }
  private var matrixPlans: [String: MatrixPlan] = [:]
  var convolutionPlans: [String: ImageConvolutionPlan] = [:]
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
  /// Window kind: 0 for ones, 1 for Tukey, 2 for the periodic Hann convention.
  // Parameter labels match the scientific Torch API.
  // swift-format-ignore: AlwaysUseLowerCamelCase
  public func window(_ image: GPUImage, kind: Int = 0, edge_blend: Double = 0, padding: Int = 0)
    throws -> GPUImage
  {
    guard padding >= 0, (0...2).contains(kind) else {
      throw Self.invalid("Use nonnegative padding and a supported window.")
    }
    let result = try allocate(image.rows + 2 * padding, image.columns + 2 * padding)
    let coefficients = [image.rows, image.columns].flatMap { length -> [Float] in
      let alpha = 2 * Double(edge_blend) / Double(length)
      return [
        alpha <= 0 ? 0 : (alpha >= 1 ? 2 : 1), Float(alpha * Double(length - 1)),
        alpha == 0 ? 0 : Float(2 / alpha), Float(2 * Double.pi / Double(length)),
      ]
    }
    try run(
      "image_window", [image.buffer, result.buffer],
      words: [UInt32(image.rows), UInt32(image.columns), UInt32(padding), UInt32(kind)],
      floats: coefficients, count: result.rows * result.columns)
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
    let total = try sum(window)
    let mean = try buffer(4)
    try run(
      "window_mean", [image.buffer, window.buffer, total, mean],
      words: [UInt32(image.rows * image.columns)], count: 1024, grouped: true,
      groupSize: 1024)
    try run(
      "window_center", [image.buffer, window.buffer, mean, result.buffer],
      words: [UInt32(image.rows * image.columns)], count: image.rows * image.columns)
    return result
  }
  func sum(_ image: GPUImage) throws -> MTLBuffer {
    let count = image.rows * image.columns
    var groups = min(512, (count + 8191) / 8192)
    while groups > 1 && count % groups != 0 { groups -= 1 }
    let partials = try buffer(groups * 4)
    let width = min(1024, ((count / groups + 31) / 32) * 32)
    try run(
      "sum_pixels", [image.buffer, partials], words: [UInt32(count / groups)],
      count: groups * width, groupSize: width)
    if groups == 1 { return partials }
    let result = try buffer(4)
    try run(
      "sum_pixels", [partials, result], words: [UInt32(groups)],
      count: 32 * ((groups + 31) / 32), groupSize: 32 * ((groups + 31) / 32))
    return result
  }
  public func fourier(_ image: GPUImage, inverse: Bool = false) throws -> GPUImage {
    let key = "\(image.rows),\(image.columns),\(inverse)"
    if fftPlans[key] == nil {
      let graph = MPSGraph()
      graph.options = .none
      let input = graph.placeholder(
        shape: [NSNumber(value: image.rows), NSNumber(value: image.columns)],
        dataType: .complexFloat32, name: "input")
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
    let inputBuffer: MTLBuffer
    if image.isComplex {
      inputBuffer = image.buffer
    } else {
      inputBuffer = try buffer(image.rows * image.columns * 8)
      try run(
        "complex_image", [image.buffer, inputBuffer],
        words: [UInt32(image.rows * image.columns)], count: image.rows * image.columns)
    }
    let shape = [NSNumber(value: image.rows), NSNumber(value: image.columns)]
    plan.graph.run(
      with: queue,
      feeds: [
        plan.input: MPSGraphTensorData(
          inputBuffer, shape: shape, dataType: .complexFloat32)
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
      ],
      floats: [
        Float(1 / Double(reference.rows)), Float(1 / Double(reference.columns)),
        Float(Double(count) / Double(count + 1)), Float(1 / Double(count + 1)),
      ],
      count: reference.rows * reference.columns)
    return result
  }
  /// Average matching images or spectra with one reduction across inputs.
  public func mean(_ images: [GPUImage]) throws -> GPUImage {
    guard let first = images.first,
      images.allSatisfy({
        $0.rows == first.rows && $0.columns == first.columns && $0.isComplex == first.isComplex
      })
    else { throw Self.invalid("Average at least one image with matching shapes and dtypes.") }
    let elementBytes = first.isComplex ? 8 : 4
    let imageBytes = first.rows * first.columns * elementBytes
    let stacked = try buffer(images.count * imageBytes)
    let command = try makeCommand()
    let copy = command.makeBlitCommandEncoder()!
    for (index, image) in images.enumerated() {
      copy.copy(
        from: image.buffer, sourceOffset: 0, to: stacked,
        destinationOffset: index * imageBytes, size: imageBytes)
    }
    copy.endEncoding()
    try complete(command)
    let result = try allocate(first.rows, first.columns, complex: first.isComplex)
    var lanes = 1
    while lanes < min(32, images.count) { lanes *= 2 }
    let count = imageBytes / 4
    try run(
      "mean_images", [stacked, result.buffer],
      words: [UInt32(images.count), UInt32(count), UInt32(lanes)], count: count * lanes)
    return result
  }
  // Parameter labels match the scientific Torch API.
  // swift-format-ignore: AlwaysUseLowerCamelCase
  public func correlation(_ reference: GPUImage, _ image: GPUImage, upsample_factor: Int = 100)
    throws -> GPUImage
  {
    guard upsample_factor >= 1, reference.isComplex, image.isComplex,
      reference.rows == image.rows, reference.columns == image.columns
    else {
      throw Self.invalid(
        "Correlation needs same-shaped complex spectra and a positive upsample_factor.")
    }
    let refinedWidth = ceil(1.5 * Double(upsample_factor))
    guard refinedWidth * refinedWidth * 4 <= Double(device.maxBufferLength),
      refinedWidth * Double(max(image.rows, image.columns)) * 8 <= Double(device.maxBufferLength)
    else {
      throw Self.invalid(
        "The requested upsample_factor exceeds the device buffer limit; reduce it.")
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
      let p = [UInt32(rows), UInt32(cols), UInt32(upsample_factor), UInt32(width)]
      try run(
        "dft_kernels", [shift.buffer, row.buffer, col.buffer], words: p,
        floats: [
          Float(-2 * Double.pi / (Double(rows) * Double(upsample_factor))),
          Float(-2 * Double.pi / (Double(cols) * Double(upsample_factor))),
        ],
        count: max(width * rows, cols * width))
      let partial = try matrixProduct(row, product, conjugateRight: true)
      let refined = try matrixProduct(partial, col, realOutput: true)
      try run(
        "peak_fit", [refined.buffer, shift.buffer],
        words: [UInt32(width), UInt32(width), UInt32(upsample_factor), 0], count: 256, grouped: true
      )
    }
    let wrapped = try self.image(rows: 1, columns: 2)
    try addShift(wrapped, shift, index: 0, wrapRows: rows, wrapColumns: cols)
    return wrapped
  }
  private func matrixProduct(
    _ left: GPUImage, _ right: GPUImage, conjugateRight: Bool = false, realOutput: Bool = false
  ) throws -> GPUImage {
    let key = "\(left.rows),\(left.columns),\(right.columns),\(conjugateRight),\(realOutput)"
    if matrixPlans[key] == nil {
      let graph = MPSGraph()
      graph.options = .none
      let a = graph.placeholder(
        shape: [NSNumber(value: left.rows), NSNumber(value: left.columns)],
        dataType: .complexFloat32, name: "left")
      let b = graph.placeholder(
        shape: [NSNumber(value: right.rows), NSNumber(value: right.columns)],
        dataType: .complexFloat32, name: "right")
      let result = graph.matrixMultiplication(
        primary: a, secondary: conjugateRight ? graph.conjugate(tensor: b, name: nil) : b,
        name: nil)
      let output = realOutput ? graph.realPartOfTensor(tensor: result, name: nil) : result
      matrixPlans[key] = MatrixPlan(graph: graph, left: a, right: b, output: output)
    }
    let plan = matrixPlans[key]!
    let result = try allocate(left.rows, right.columns, complex: !realOutput)
    plan.graph.run(
      with: queue,
      feeds: [
        plan.left: MPSGraphTensorData(
          left.buffer,
          shape: [
            NSNumber(value: left.rows), NSNumber(value: left.columns),
          ], dataType: .complexFloat32),
        plan.right: MPSGraphTensorData(
          right.buffer,
          shape: [
            NSNumber(value: right.rows), NSNumber(value: right.columns),
          ], dataType: .complexFloat32),
      ],
      targetOperations: nil,
      resultsDictionary: [
        plan.output: MPSGraphTensorData(
          result.buffer,
          shape: [
            NSNumber(value: result.rows), NSNumber(value: result.columns),
          ],
          dataType: realOutput ? .float32 : .complexFloat32)
      ])
    return result
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
    grouped: Bool = false, groupSize: Int = 256
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
        threadsPerThreadgroup: MTLSize(width: groupSize, height: 1, depth: 1))
    } else {
      enc.dispatchThreads(
        MTLSize(width: count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: groupSize, height: 1, depth: 1))
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
