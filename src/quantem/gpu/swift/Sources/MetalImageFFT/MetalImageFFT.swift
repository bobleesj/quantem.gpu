import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Scalar source formats accepted by ``MetalImageFFT``.
public enum MetalImageFFTScalarType: Sendable, Equatable {
  case uint8
  case uint16
  case uint32
  case float32

  fileprivate var byteStride: Int {
    switch self {
    case .uint8: MemoryLayout<UInt8>.stride
    case .uint16: MemoryLayout<UInt16>.stride
    case .uint32: MemoryLayout<UInt32>.stride
    case .float32: MemoryLayout<Float>.stride
    }
  }
}

/// A GPU-resident centered log-magnitude Fourier image.
public struct MetalImageFFTResult: @unchecked Sendable {
  public let buffer: MTLBuffer
  public let rows: Int
  public let columns: Int
  public let minimum: Float
  public let maximum: Float
}

/// Errors raised by ``MetalImageFFT``.
public enum MetalImageFFTError: LocalizedError {
  case missingResource(String)
  case missingFunction(String)
  case libraryCompilation(String)
  case invalidShape(rows: Int, columns: Int)
  case inputBufferTooSmall(required: Int, actual: Int)
  case allocation(String)
  case commandExecution(String)

  public var errorDescription: String? {
    switch self {
    case .missingResource(let name):
      "MetalImageFFT is missing \(name)."
    case .missingFunction(let name):
      "MetalImageFFT is missing the \(name) function."
    case .libraryCompilation(let message):
      "MetalImageFFT kernel compilation failed: \(message)"
    case .invalidShape(let rows, let columns):
      "The FFT image shape \(rows)×\(columns) is invalid."
    case .inputBufferTooSmall(let required, let actual):
      "The FFT input needs \(required) bytes, but the buffer contains \(actual)."
    case .allocation(let purpose):
      "Metal could not allocate the \(purpose)."
    case .commandExecution(let message):
      "The Metal FFT failed: \(message)"
    }
  }
}

/// A reusable, GPU-only two-dimensional FFT for scientific image inspection.
///
/// The returned image is `fftshift(log1p(abs(fft2(source))))` in row-major
/// `Float` storage. Forward transforms use Apple's MPSGraph FFT so warm
/// 256–2048 scan images stay inside a 60–120 Hz frame. Reciprocal coordinates
/// come from the image dimensions and the caller's spatial calibration.
public final class MetalImageFFT: @unchecked Sendable {
  private struct ShapeKey: Hashable {
    let rows: Int
    let columns: Int
  }

  private struct GraphPlan {
    let graph: MPSGraph
    let input: MPSGraphTensor
    let output: MPSGraphTensor
    let maximum: MPSGraphTensor
  }

  private struct Workspace {
    let packed: MTLBuffer
    let maximum: MTLBuffer
  }

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let packUInt8: MTLComputePipelineState
  private let packUInt16: MTLComputePipelineState
  private let packUInt32: MTLComputePipelineState
  private let packFloat32: MTLComputePipelineState
  private let lock = NSLock()
  private let executionLock = NSLock()
  private var plans: [ShapeKey: GraphPlan] = [:]
  private var workspaces: [ShapeKey: Workspace] = [:]

  public init(device: MTLDevice) throws {
    self.device = device
    guard let queue = device.makeCommandQueue() else {
      throw MetalImageFFTError.allocation("FFT command queue")
    }
    self.queue = queue
    let library = try Self.makeLibrary(device: device)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalImageFFTError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    packUInt8 = try pipeline("image_fft_pack_u8")
    packUInt16 = try pipeline("image_fft_pack_u16")
    packUInt32 = try pipeline("image_fft_pack_u32")
    packFloat32 = try pipeline("image_fft_pack_f32")
  }

  /// Compile and cache the MPSGraph for a scan shape before the first display.
  public func prewarm(rows: Int, columns: Int) throws {
    let key = ShapeKey(rows: rows, columns: columns)
    _ = try lockedPlan(for: key)
    _ = try lockedWorkspace(for: key, count: rows * columns)
  }

  /// Compute a centered log-magnitude FFT without a CPU numerical fallback.
  ///
  /// - Parameters:
  ///   - source: Row-major scalar values already resident in a Metal buffer.
  ///   - rows: Number of image rows.
  ///   - columns: Number of image columns.
  ///   - scalarType: Scalar representation stored in `source`.
  ///   - output: Optional destination. When the buffer is large enough it is
  ///     reused so a live FFT panel can update in place.
  /// - Returns: A GPU-resident Float32 image and its GPU-computed display range.
  public func logMagnitude(
    source: MTLBuffer,
    rows: Int,
    columns: Int,
    scalarType: MetalImageFFTScalarType,
    output: MTLBuffer? = nil
  ) throws -> MetalImageFFTResult {
    guard rows > 0, columns > 0,
      rows <= Int(UInt32.max), columns <= Int(UInt32.max),
      rows <= (1 << 30), columns <= (1 << 30),
      rows <= Int.max / columns
    else { throw MetalImageFFTError.invalidShape(rows: rows, columns: columns) }
    let count = rows * columns
    guard count <= Int.max / scalarType.byteStride else {
      throw MetalImageFFTError.invalidShape(rows: rows, columns: columns)
    }
    let requiredBytes = count * scalarType.byteStride
    guard source.length >= requiredBytes else {
      throw MetalImageFFTError.inputBufferTooSmall(
        required: requiredBytes,
        actual: source.length
      )
    }
    let outputBytes = count * MemoryLayout<Float>.stride
    let resultBuffer: MTLBuffer
    if let output, output.length >= outputBytes {
      resultBuffer = output
    } else if let allocated = device.makeBuffer(
      length: outputBytes,
      options: .storageModeShared
    ) {
      resultBuffer = allocated
    } else {
      throw MetalImageFFTError.allocation("FFT image workspace")
    }

    let key = ShapeKey(rows: rows, columns: columns)
    let plan = try lockedPlan(for: key)
    let workspace = try lockedWorkspace(for: key, count: count)

    // Plans and scratch buffers are shared between callers for warm reuse.
    // Keep one transform in flight so a concurrent same-shape request cannot
    // replace the packed input or maximum before the first graph consumes it.
    executionLock.lock()
    defer { executionLock.unlock() }

    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw MetalImageFFTError.allocation("FFT image workspace") }
    var count32 = UInt32(count)
    encoder.setComputePipelineState(packPipeline(for: scalarType))
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(workspace.packed, offset: 0, index: 1)
    encoder.setBytes(&count32, length: MemoryLayout<UInt32>.stride, index: 2)
    encoder.dispatchThreads(
      MTLSize(width: count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: min(256, count), height: 1, depth: 1)
    )
    encoder.endEncoding()
    command.commit()

    let shape: [NSNumber] = [NSNumber(value: rows), NSNumber(value: columns)]
    let inputData = MPSGraphTensorData(
      workspace.packed,
      shape: shape,
      dataType: .float32
    )
    let outputData = MPSGraphTensorData(
      resultBuffer,
      shape: shape,
      dataType: .float32
    )
    let maximumData = MPSGraphTensorData(
      workspace.maximum,
      shape: [1],
      dataType: .float32
    )
    plan.graph.run(
      with: queue,
      feeds: [plan.input: inputData],
      targetOperations: nil,
      resultsDictionary: [
        plan.output: outputData,
        plan.maximum: maximumData,
      ]
    )

    let maximum = workspace.maximum.contents().assumingMemoryBound(to: Float.self).pointee
    return MetalImageFFTResult(
      buffer: resultBuffer,
      rows: rows,
      columns: columns,
      minimum: 0,
      maximum: maximum.isFinite ? maximum : 0
    )
  }

  private func packPipeline(
    for scalarType: MetalImageFFTScalarType
  ) -> MTLComputePipelineState {
    switch scalarType {
    case .uint8: packUInt8
    case .uint16: packUInt16
    case .uint32: packUInt32
    case .float32: packFloat32
    }
  }

  private func lockedPlan(for key: ShapeKey) throws -> GraphPlan {
    lock.lock()
    defer { lock.unlock() }
    if let plan = plans[key] { return plan }
    let plan = try makePlan(rows: key.rows, columns: key.columns)
    plans[key] = plan
    return plan
  }

  private func lockedWorkspace(for key: ShapeKey, count: Int) throws -> Workspace {
    lock.lock()
    defer { lock.unlock() }
    if let workspace = workspaces[key] { return workspace }
    guard
      let packed = device.makeBuffer(
        length: count * MemoryLayout<Float>.stride,
        options: .storageModePrivate
      ),
      let maximum = device.makeBuffer(
        length: MemoryLayout<Float>.stride,
        options: .storageModeShared
      )
    else { throw MetalImageFFTError.allocation("FFT image workspace") }
    let workspace = Workspace(packed: packed, maximum: maximum)
    workspaces[key] = workspace
    return workspace
  }

  private func makePlan(rows: Int, columns: Int) throws -> GraphPlan {
    let graph = MPSGraph()
    graph.options = .none
    let shape: [NSNumber] = [NSNumber(value: rows), NSNumber(value: columns)]
    let input = graph.placeholder(shape: shape, dataType: .float32, name: "real")
    let descriptor = MPSGraphFFTDescriptor()
    descriptor.inverse = false
    descriptor.scalingMode = .none
    let transformed = graph.fastFourierTransform(
      input,
      axes: [0, 1],
      descriptor: descriptor,
      name: "fft2"
    )
    let real = graph.realPartOfTensor(tensor: transformed, name: "fft_real")
    let imaginary = graph.imaginaryPartOfTensor(tensor: transformed, name: "fft_imag")
    let magnitude = graph.squareRoot(
      with: graph.addition(
        graph.square(with: real, name: nil),
        graph.square(with: imaginary, name: nil),
        name: nil
      ),
      name: "abs"
    )
    let one = graph.constant(1.0, dataType: .float32)
    let logMagnitude = graph.logarithm(
      with: graph.addition(magnitude, one, name: nil),
      name: "log1p"
    )
    let shifted = Self.fftshift(graph, logMagnitude, rows: rows, columns: columns)
    let maximum = graph.reductionMaximum(
      with: shifted,
      axes: [0, 1],
      name: "max"
    )
    let shapedType = MPSGraphShapedType(shape: shape, dataType: .float32)
    _ = graph.compile(
      with: MPSGraphDevice(mtlDevice: device),
      feeds: [input: shapedType],
      targetTensors: [shifted, maximum],
      targetOperations: nil,
      compilationDescriptor: nil
    )
    return GraphPlan(graph: graph, input: input, output: shifted, maximum: maximum)
  }

  private static func fftshift(
    _ graph: MPSGraph,
    _ tensor: MPSGraphTensor,
    rows: Int,
    columns: Int
  ) -> MPSGraphTensor {
    var shifted = tensor
    let rowShift = rows / 2
    if rowShift > 0 {
      let top = graph.sliceTensor(
        shifted,
        dimension: 0,
        start: rows - rowShift,
        length: rowShift,
        name: nil
      )
      let bottom = graph.sliceTensor(
        shifted,
        dimension: 0,
        start: 0,
        length: rows - rowShift,
        name: nil
      )
      shifted = graph.concatTensors([top, bottom], dimension: 0, name: "fftshift_rows")
    }
    let columnShift = columns / 2
    if columnShift > 0 {
      let left = graph.sliceTensor(
        shifted,
        dimension: 1,
        start: columns - columnShift,
        length: columnShift,
        name: nil
      )
      let right = graph.sliceTensor(
        shifted,
        dimension: 1,
        start: 0,
        length: columns - columnShift,
        name: nil
      )
      shifted = graph.concatTensors([left, right], dimension: 1, name: "fftshift_cols")
    }
    return shifted
  }

  private static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
    let packagedURL = Bundle.main.resourceURL?
      .appendingPathComponent("MetalKernels_MetalImageFFT.bundle", isDirectory: true)
      .appendingPathComponent("Resources", isDirectory: true)
      .appendingPathComponent("fft.metal")
    let url =
      packagedURL.flatMap {
        FileManager.default.fileExists(atPath: $0.path) ? $0 : nil
      } ?? Bundle.module.url(
        forResource: "fft",
        withExtension: "metal",
        subdirectory: "Resources"
      ) ?? Bundle.module.url(forResource: "fft", withExtension: "metal")
    guard let url else { throw MetalImageFFTError.missingResource("fft.metal") }
    do {
      return try device.makeLibrary(
        source: String(contentsOf: url, encoding: .utf8),
        options: nil
      )
    } catch {
      throw MetalImageFFTError.libraryCompilation(error.localizedDescription)
    }
  }
}
