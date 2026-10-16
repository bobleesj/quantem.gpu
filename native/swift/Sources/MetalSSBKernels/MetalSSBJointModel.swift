import Foundation
import Metal
import MetalPerformanceShadersGraph

/// The depth-averaged SSB transfer model. C10 is at the specimen midplane;
/// defocus f = -C10. Tilt components are in the scan (row, column) frame.
public struct SSBJointParameters: Codable, Equatable, Sendable {
  public var aberrations: MetalSSBAberrations
  public var depthAngstroms: Float
  public var tiltRowMrad: Float
  public var tiltColumnMrad: Float

  public init(
    aberrations: MetalSSBAberrations, depthAngstroms: Float = 200,
    tiltRowMrad: Float = 0, tiltColumnMrad: Float = 0
  ) {
    self.aberrations = aberrations
    self.depthAngstroms = depthAngstroms
    self.tiltRowMrad = tiltRowMrad
    self.tiltColumnMrad = tiltColumnMrad
  }

  var packed: [Float] {
    [
      aberrations.c10Angstroms, aberrations.c12Angstroms,
      Float(cos(2 * Double(aberrations.phi12Radians))),
      Float(sin(2 * Double(aberrations.phi12Radians))),
      depthAngstroms, Float(Double(tiltRowMrad) * 0.001), Float(Double(tiltColumnMrad) * 0.001),
    ]
  }

  func validate() throws {
    guard packed.allSatisfy(\.isFinite), depthAngstroms >= 0,
      !(aberrations.higherOrder ?? []).contains(where: { $0.magnitudeAngstroms != 0 })
    else {
      throw MetalSSBError.invalidGeometry(
        "Joint SSB needs finite C10/C12, nonnegative depth and tilt in mrad. Higher-order terms are not supported by this model."
      )
    }
  }
}

/// GPU phase in radians, using the mean of individual bright-field phases.
/// Upsampling unfolds the measured Fourier aliases; it is not image interpolation.
public struct MetalSSBJointPhase: @unchecked Sendable {
  public let phase: MTLBuffer
  public let rows: Int
  public let columns: Int
  public let upsamplingFactor: Int
  public let wallSeconds: Double
  public let parameters: SSBJointParameters
  public let scanDetectorRotationDegrees: Float
  public let provenance: MetalSSBProvenance
}

/// Serial native Metal implementation of QuantEM's joint tilt/depth SSB model.
/// Search uses the original scan grid and all selected BF pixels. The band is
/// an explicit part of the least-squares objective, in inverse Angstrom.
public final class MetalSSBJointModel {
  typealias SpectrumVisitor = (MTLBuffer, Int, Bool, Int, Int) throws -> Void
  typealias VisitSpectra = (@escaping SpectrumVisitor) throws -> Void

  private struct Grid {
    var n, outputN, count, bfOffset, totalBF, bandCount, sourceHalf, chunks: UInt32
    var wl, semiangle, angularRow, angularColumn, dqRow, dqColumn, dcReal, dcImag, factor: Float
  }
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let visit: VisitSpectra
  private let validatePreparation: () throws -> Void
  private let geometry: MetalSSBGeometry
  private let provenance: MetalSSBProvenance
  public let scanDetectorRotationDegrees: Float
  public let bandInverseAngstroms: ClosedRange<Float>
  public let bandCount: Int
  private var grid: Grid
  private let kBuffer, qBuffer, indexBuffer, weights, bandG, partial, power: MTLBuffer
  private let gather, objective, reduce, correct, addPhase, normalize: MTLComputePipelineState

  init(
    device: MTLDevice, geometry: MetalSSBGeometry, activeIndices: [Int], rotation: Float,
    band: ClosedRange<Float>, provenance: MetalSSBProvenance,
    visit: @escaping VisitSpectra, validatePreparation: @escaping () throws -> Void,
    isCancelled: () -> Bool
  ) throws {
    guard rotation.isFinite, band.lowerBound >= 0, band.upperBound > band.lowerBound else {
      throw MetalSSBError.invalidGeometry(
        "Choose a finite rotation and a nonempty positive frequency band.")
    }
    self.device = device
    self.geometry = geometry
    self.provenance = provenance
    self.visit = visit
    self.validatePreparation = validatePreparation
    scanDetectorRotationDegrees = rotation
    bandInverseAngstroms = band
    guard let queue = device.makeCommandQueue() else { throw MetalSSBError.commandQueue }
    self.queue = queue
    let packaged = Bundle.main.resourceURL?.appendingPathComponent(
      "MetalKernels_MetalSSBKernels.bundle/Resources/ssb_joint.metal")
    let url =
      packaged.flatMap { FileManager.default.fileExists(atPath: $0.path) ? $0 : nil }
      ?? Bundle.module.url(
        forResource: "ssb_joint", withExtension: "metal", subdirectory: "Resources")
    guard let url else { throw MetalSSBError.missingResource("ssb_joint.metal") }
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try device.makeLibrary(
      source: String(contentsOf: url, encoding: .utf8), options: options)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalSSBError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    gather = try pipeline("joint_gather")
    objective = try pipeline("joint_objective")
    reduce = try pipeline("joint_power")
    correct = try pipeline("joint_correct")
    addPhase = try pipeline("joint_phase_add")
    normalize = try pipeline("joint_phase_normalize")
    let n = geometry.qxByRow.count
    var indices: [SIMD2<UInt32>] = []
    var qs: [SIMD2<Float>] = []
    var ws: [Float] = []
    for r in 0..<n {
      for c in 0...(n / 2) {
        let q = SIMD2(geometry.qxByRow[r], geometry.qyByColumn[c])
        let radius = sqrt(q.x * q.x + q.y * q.y)
        if radius > band.lowerBound && radius < band.upperBound {
          indices.append(SIMD2(UInt32(r), UInt32(c)))
          qs.append(q)
          ws.append(c == 0 || c == n / 2 ? 1 : 2)
        }
      }
    }
    guard !indices.isEmpty else {
      throw MetalSSBError.invalidGeometry(
        "No scan frequencies lie in the joint search band. Check scan spacing or choose a wider band."
      )
    }
    bandCount = indices.count
    // Match the reference's scalar angle conversion before Float32 vector math.
    let delta = -(Double(rotation) - Double(geometry.referenceRotationDegrees)) * Double.pi / 180
    let cosine = Float(cos(delta))
    let sine = Float(sin(delta))
    let ks = activeIndices.map { i -> SIMD2<Float> in
      let x = geometry.brightfieldKX[i]
      let y = geometry.brightfieldKY[i]
      return SIMD2(x * cosine + y * sine, -x * sine + y * cosine)
    }
    kBuffer = try Self.buffer(device, ks)
    qBuffer = try Self.buffer(device, qs)
    indexBuffer = try Self.buffer(device, indices)
    weights = try Self.buffer(device, ws)
    bandG = try Self.allocate(device, activeIndices.count * bandCount * 8)
    let chunks = (activeIndices.count + 255) / 256
    partial = try Self.allocate(device, chunks * bandCount * 16)
    power = try Self.allocate(device, bandCount * 4, shared: true)
    grid = Grid(
      n: UInt32(n), outputN: UInt32(n), count: 0, bfOffset: 0,
      totalBF: UInt32(activeIndices.count), bandCount: UInt32(bandCount), sourceHalf: 1,
      chunks: UInt32(chunks),
      wl: geometry.wavelengthAngstroms, semiangle: geometry.semiangleRadians,
      angularRow: geometry.angularSamplingYRadians, angularColumn: geometry.angularSamplingXRadians,
      dqRow: geometry.qxByRow[1], dqColumn: geometry.qyByColumn[1],
      dcReal: geometry.dcValue.x, dcImag: geometry.dcValue.y,
      factor: geometry.phaseFactor ?? Float.pi / geometry.wavelengthAngstroms)
    try withoutActuallyEscaping(isCancelled) { cancelled in
      try visit { [self] source, offset, half, bfOffset, count in
        if cancelled() { throw CancellationError() }
        var p = grid
        p.sourceHalf = half ? 1 : 0
        p.bfOffset = UInt32(bfOffset)
        p.count = UInt32(count)
        let commands = try command()
        let encoder = try self.encoder(commands, gather)
        encoder.setBuffer(source, offset: offset, index: 0)
        encoder.setBuffer(bandG, offset: 0, index: 1)
        encoder.setBuffer(indexBuffer, offset: 0, index: 2)
        encoder.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 3)
        dispatch(encoder, width: bandCount, height: count)
        try finish(commands)
      }
    }
  }

  /// Negative explained Fourier power. Lower is better; it is not a confidence
  /// score or a certificate of the physical scan-rotation branch.
  public func loss(_ parameters: SSBJointParameters) throws -> Double {
    try validatePreparation()
    try parameters.validate()
    var p = grid
    let t = parameters.packed
    let commands = try command()
    let encoder = try self.encoder(commands, objective)
    for (index, buffer) in [bandG, kBuffer, qBuffer, partial].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 4)
    encoder.setBytes(t, length: t.count * 4, index: 5)
    dispatch(encoder, width: bandCount, height: Int(p.chunks))
    let reduction = try self.encoder(commands, reduce)
    for (index, buffer) in [partial, weights, power].enumerated() {
      reduction.setBuffer(buffer, offset: 0, index: index)
    }
    reduction.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 3)
    dispatch(reduction, width: bandCount)
    try finish(commands)
    // Match the reference's Float64 final accumulation of GPU Float32 powers.
    let values = power.contents().assumingMemoryBound(to: Float.self)
    var sum = 0.0
    for i in 0..<bandCount { sum += Double(values[i]) }
    guard sum.isFinite else { throw SSBOptimizationError.nonFiniteLoss }
    return -sum
  }

  /// Reconstruct with all BF pixels. Cancellation is checked between GPU batches.
  public func reconstruct(
    _ parameters: SSBJointParameters, upsamplingFactor: Int = 1,
    isCancelled: () -> Bool = { false }
  ) throws -> MetalSSBJointPhase {
    try validatePreparation()
    try parameters.validate()
    guard [1, 2, 3, 4, 8].contains(upsamplingFactor) else {
      throw MetalSSBError.invalidGeometry("Use an SSB sampling factor of 1, 2, 3, 4 or 8.")
    }
    let started = Date()
    let n = Int(grid.n) * upsamplingFactor
    let plane = n * n
    let phase = try Self.allocate(device, plane * 4, shared: true)
    let corrected = try Self.allocate(device, 8 * plane * 8)
    let object = try Self.allocate(device, 8 * plane * 8)
    let clear = try command()
    guard let blit = clear.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
    blit.fill(buffer: phase, range: 0..<phase.length, value: 0)
    blit.endEncoding()
    try finish(clear)
    let t = parameters.packed
    var plans: [Int: (MPSGraph, MPSGraphTensor, MPSGraphTensor)] = [:]
    try withoutActuallyEscaping(isCancelled) { cancelled in
      try visit { [self] source, offset, half, bfOffset, count in
        if cancelled() { throw CancellationError() }
        var p = grid
        p.outputN = UInt32(n)
        p.sourceHalf = half ? 1 : 0
        p.count = UInt32(count)
        p.bfOffset = UInt32(bfOffset)
        let commands = try command()
        let encoder = try self.encoder(commands, correct)
        encoder.setBuffer(source, offset: offset, index: 0)
        encoder.setBuffer(kBuffer, offset: 0, index: 1)
        encoder.setBuffer(corrected, offset: 0, index: 2)
        encoder.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 3)
        encoder.setBytes(t, length: t.count * 4, index: 4)
        dispatch(encoder, width: plane, height: count)
        try finish(commands)
        let shape: [NSNumber] = [NSNumber(value: count), NSNumber(value: n), NSNumber(value: n)]
        if plans[count] == nil {
          let graph = MPSGraph()
          graph.options = .none
          let input = graph.placeholder(shape: shape, dataType: .complexFloat32, name: "corrected")
          let descriptor = MPSGraphFFTDescriptor()
          descriptor.inverse = true
          descriptor.scalingMode = .size
          let output = graph.fastFourierTransform(
            input, axes: [1, 2], descriptor: descriptor, name: "ifft2")
          plans[count] = (graph, input, output)
        }
        let (graph, input, output) = plans[count]!
        graph.run(
          with: queue,
          feeds: [input: MPSGraphTensorData(corrected, shape: shape, dataType: .complexFloat32)],
          targetOperations: nil,
          resultsDictionary: [
            output: MPSGraphTensorData(object, shape: shape, dataType: .complexFloat32)
          ])
        let sumCommands = try command()
        let sum = try self.encoder(sumCommands, addPhase)
        sum.setBuffer(object, offset: 0, index: 0)
        sum.setBuffer(phase, offset: 0, index: 1)
        sum.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 2)
        dispatch(sum, width: plane)
        try finish(sumCommands)
      }
    }
    let final = try command()
    let encoder = try self.encoder(final, normalize)
    var count = UInt32(plane)
    var divisor = Float(geometry.logicalBrightfieldCount)
    encoder.setBuffer(phase, offset: 0, index: 0)
    encoder.setBytes(&count, length: 4, index: 1)
    encoder.setBytes(&divisor, length: 4, index: 2)
    dispatch(encoder, width: plane)
    try finish(final)
    return MetalSSBJointPhase(
      phase: phase, rows: n, columns: n, upsamplingFactor: upsamplingFactor,
      wallSeconds: Date().timeIntervalSince(started), parameters: parameters,
      scanDetectorRotationDegrees: scanDetectorRotationDegrees, provenance: provenance)
  }

  private func command() throws -> MTLCommandBuffer {
    guard let command = queue.makeCommandBuffer() else { throw MetalSSBError.commandQueue }
    return command
  }
  private func encoder(_ command: MTLCommandBuffer, _ pipeline: MTLComputePipelineState) throws
    -> MTLComputeCommandEncoder
  {
    guard let encoder = command.makeComputeCommandEncoder() else {
      throw MetalSSBError.commandQueue
    }
    encoder.setComputePipelineState(pipeline)
    return encoder
  }
  private func dispatch(_ encoder: MTLComputeCommandEncoder, width: Int, height: Int = 1) {
    encoder.dispatchThreads(
      MTLSize(width: width, height: height, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
  }
  private func finish(_ command: MTLCommandBuffer) throws {
    command.commit()
    command.waitUntilCompleted()
    if let error = command.error {
      throw MetalSSBError.commandExecution(error.localizedDescription)
    }
  }
  private static func allocate(_ device: MTLDevice, _ length: Int, shared: Bool = false) throws
    -> MTLBuffer
  {
    guard
      let buffer = device.makeBuffer(
        length: max(4, length), options: shared ? .storageModeShared : .storageModePrivate)
    else {
      throw MetalSSBError.allocation(
        "joint SSB workspace (\(length) bytes). Close inactive datasets and retry")
    }
    return buffer
  }
  private static func buffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
    guard
      let result = values.withUnsafeBytes({ bytes in
        device.makeBuffer(
          bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
      })
    else {
      throw MetalSSBError.allocation("joint SSB geometry")
    }
    return result
  }
}
