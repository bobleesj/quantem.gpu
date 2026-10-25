import Foundation
import Metal
import MetalPerformanceShadersGraph

/// The transfer-model point in the Python engine's units and double precision: C10, C12 and the depth spread
/// in Angstrom, phi12 in radians, the sample tilt in mrad (row, column; scan frame).
public struct SSBModelPoint: Codable, Equatable, Sendable {
  public var c10Angstroms: Double
  public var c12Angstroms: Double
  public var phi12Radians: Double
  public var sampleTiltMrad: [Double]
  public var depthSpreadAngstroms: Double
  public var higherOrder: [SSBModelHigherOrder]

  public init(
    c10Angstroms: Double, c12Angstroms: Double, phi12Radians: Double,
    sampleTiltMrad: [Double] = [0, 0], depthSpreadAngstroms: Double = 0,
    higherOrder: [SSBModelHigherOrder] = []
  ) {
    self.c10Angstroms = c10Angstroms
    self.c12Angstroms = c12Angstroms
    self.phi12Radians = phi12Radians
    self.sampleTiltMrad = sampleTiltMrad
    self.depthSpreadAngstroms = depthSpreadAngstroms
    self.higherOrder = higherOrder
  }

  /// From public units: C10 and C12 in nm, phi12 in degrees, the depth spread in nm.
  public init(
    c10Nanometers: Double, c12Nanometers: Double, phi12Degrees: Double,
    sampleTiltMrad: [Double] = [0, 0], depthSpreadNanometers: Double = 0,
    higherOrder: [SSBModelHigherOrder] = []
  ) {
    self.init(
      c10Angstroms: c10Nanometers * 10, c12Angstroms: c12Nanometers * 10,
      phi12Radians: phi12Degrees * .pi / 180, sampleTiltMrad: sampleTiltMrad,
      depthSpreadAngstroms: depthSpreadNanometers * 10, higherOrder: higherOrder)
  }

  public init(_ parameters: SSBJointParameters) {
    self.init(
      c10Angstroms: Double(parameters.aberrations.c10Angstroms),
      c12Angstroms: Double(parameters.aberrations.c12Angstroms),
      phi12Radians: Double(parameters.aberrations.phi12Radians),
      sampleTiltMrad: [Double(parameters.tiltRowMrad), Double(parameters.tiltColumnMrad)],
      depthSpreadAngstroms: Double(parameters.depthAngstroms),
      higherOrder: (parameters.aberrations.higherOrder ?? []).map(SSBModelHigherOrder.init))
  }

  private enum CodingKeys: String, CodingKey {
    case c10Angstroms, c12Angstroms, phi12Radians, sampleTiltMrad, depthSpreadAngstroms, higherOrder
  }

  public init(from decoder: Decoder) throws {
    let values = try decoder.container(keyedBy: CodingKeys.self)
    self.init(
      c10Angstroms: try values.decode(Double.self, forKey: .c10Angstroms),
      c12Angstroms: try values.decode(Double.self, forKey: .c12Angstroms),
      phi12Radians: try values.decode(Double.self, forKey: .phi12Radians),
      sampleTiltMrad: try values.decode([Double].self, forKey: .sampleTiltMrad),
      depthSpreadAngstroms: try values.decode(Double.self, forKey: .depthSpreadAngstroms),
      higherOrder: try values.decodeIfPresent([SSBModelHigherOrder].self, forKey: .higherOrder)
        ?? [])
  }

  public var hasHigherOrder: Bool { higherOrder.contains { $0.magnitudeAngstroms != 0 } }

  public var c10Nanometers: Double { c10Angstroms / 10 }
  public var c12Nanometers: Double { c12Angstroms / 10 }
  public var phi12Degrees: Double { phi12Radians * 180 / .pi }
  public var depthSpreadNanometers: Double { depthSpreadAngstroms / 10 }

  /// The float32 parameters the joint model's kernels read.
  public var jointParameters: SSBJointParameters {
    SSBJointParameters(
      aberrations: .init(
        c10Angstroms: Float(c10Angstroms), c12Angstroms: Float(c12Angstroms),
        phi12Radians: Float(phi12Radians),
        higherOrder: higherOrder.isEmpty ? nil : higherOrder.map(\.metal)),
      depthAngstroms: Float(depthSpreadAngstroms), tiltRowMrad: Float(sampleTiltMrad[0]),
      tiltColumnMrad: Float(sampleTiltMrad[1]))
  }

  /// One fit trial, as the resident search's `_trial` encodes it: C10, C12, cos 2 phi12, sin 2 phi12, depth
  /// spread, tilt row and column in rad, each rounded once from double precision.
  var trialRow: [Float] {
    [
      Float(c10Angstroms), Float(c12Angstroms), Float(cos(2 * phi12Radians)),
      Float(sin(2 * phi12Radians)), Float(depthSpreadAngstroms), Float(sampleTiltMrad[0] / 1000),
      Float(sampleTiltMrad[1] / 1000),
    ]
  }

  func validate() throws {
    try SSBModelHigherOrder.validate(higherOrder)
    guard sampleTiltMrad.count == 2,
      ([c10Angstroms, c12Angstroms, phi12Radians, depthSpreadAngstroms] + sampleTiltMrad)
        .allSatisfy(
          \.isFinite), depthSpreadAngstroms >= 0
    else {
      throw MetalSSBError.invalidGeometry(
        "The SSB model needs finite C10, C12, phi12, a (row, column) tilt and a nonnegative depth spread."
      )
    }
  }
}

/// Coherent reconstruction on the native or a finer output grid (`ssb_model.metal`).
final class SSBCoherentReconstruction {
  struct Grid {
    var sourceN, outputN, gCols, count, bfOffset, sourceHalf, rows, cols: UInt32
    var chunk, groups, groupOffset, depth: UInt32
  }
  /// BF pixels per culling group; it divides every cache chunk and streamed batch offset.
  static let chunk = 8
  private struct Plan {
    let qRow, qCol, rowIndices, colIndices: MTLBuffer
    let rows, cols: Int
    let half: Bool
    /// Largest |output frequency index| with support on each axis.
    let rowLimit, colLimit: Int
  }
  struct RedrawGrid {
    var sourceN, outputN, gCols, count, bfOffset, sourceHalf, rowLimit, colLimit: UInt32
    var support2: Float
    var pixels: UInt32
  }

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let geometry: MetalSSBGeometry
  private let kBuffer: MTLBuffer
  private let kHost: [SIMD2<Float>]
  private let bounds: MTLBuffer
  private let probe: MTLBuffer
  private let probePipeline, coherentPipeline, mirrorPipeline, originPipeline,
    phasePipeline: MTLComputePipelineState
  private let higherProbePipeline, higherCoherentPipeline: MTLComputePipelineState
  private var plans: [Int: Plan] = [:]
  private var fft: [Int: (MPSGraph, MPSGraphTensor, MPSGraphTensor)] = [:]
  private let separable: SSBSeparableReconstruction
  private var workspaces: [String: MTLBuffer] = [:]
  private func workspace(_ name: String, _ length: Int) throws -> MTLBuffer {
    if let buffer = workspaces[name], buffer.length == length { return buffer }
    let buffer = try Self.allocate(device, length)
    workspaces[name] = buffer
    return buffer
  }
  /// 8x only: the separable kernels (default) or the per-term kernel.
  var useSeparable = true
  /// Tests only: evaluate 1x-4x with the earlier separable/per-term kernels as an independent parity reference.
  /// Reconstruction never falls back to them; 1x-4x always uses `ssb_redraw.metal`.
  var referenceKernels = false
  private let library: MTLLibrary
  private let redrawMirrorPipeline, redrawTaylorPipeline: MTLComputePipelineState
  private var redrawPipelines: [String: MTLComputePipelineState] = [:]
  /// The largest BF pixel frequency magnitude.
  private let maximumK: Float
  /// Source digests of the redraw and the earlier coherent kernels, from the bundled `.metal` text.
  private let redrawSource, perTermSource: String
  /// The kernel that produced the latest reconstruction.
  private(set) var lastKernel: MetalKernelIdentity?

  init(
    device: MTLDevice, queue: MTLCommandQueue, library: MTLLibrary, geometry: MetalSSBGeometry,
    kBuffer: MTLBuffer, k: [SIMD2<Float>]
  ) throws {
    self.device = device
    self.queue = queue
    self.geometry = geometry
    self.kBuffer = kBuffer
    kHost = k
    self.library = library
    maximumK = k.map { hypotf($0.x, $0.y) }.max() ?? 0
    redrawSource = try SSBKernelRegistry.sourceSHA256(SSBKernelRegistry.redraw)
    perTermSource = try SSBKernelRegistry.sourceSHA256(SSBKernelRegistry.perTerm)
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalSSBError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    higherProbePipeline = try pipeline("fifth_model_probe")
    higherCoherentPipeline = try pipeline("fifth_model_coherent")
    probePipeline = try pipeline("model_probe")
    coherentPipeline = try pipeline("model_coherent")
    mirrorPipeline = try pipeline("model_mirror")
    originPipeline = try pipeline("model_set_origin")
    phasePipeline = try pipeline("model_wave_phase")
    redrawMirrorPipeline = try pipeline("redraw_mirror")
    redrawTaylorPipeline = try pipeline("redraw_taylor")
    // Group bounds (row min, row max, column min, column max, radius^2), as _alias_group_bounds_kernel.
    let radius = Self.supportRadius(geometry)
    var values: [Float] = []
    for first in stride(from: 0, to: k.count, by: Self.chunk) {
      let group = k[first..<min(first + Self.chunk, k.count)]
      values += [
        group.map(\.x).min()!, group.map(\.x).max()!, group.map(\.y).min()!, group.map(\.y).max()!,
        radius * radius,
      ]
    }
    bounds = try Self.buffer(device, values)
    probe = try Self.allocate(device, k.count * 5 * 4, shared: true)
    // The aperture at each k does not depend on the aberrations: compute it once for the separable tables.
    var params = Self.parameters(
      geometry, SSBModelPoint(c10Angstroms: 0, c12Angstroms: 0, phi12Radians: 0))
    var count = UInt32(k.count)
    guard let commands = queue.makeCommandBuffer(),
      let encoder = commands.makeComputeCommandEncoder()
    else {
      throw MetalSSBError.commandQueue
    }
    encoder.setComputePipelineState(probePipeline)
    encoder.setBuffer(kBuffer, offset: 0, index: 0)
    encoder.setBytes(&params, length: params.count * 4, index: 1)
    encoder.setBuffer(probe, offset: 0, index: 2)
    encoder.setBytes(&count, length: 4, index: 3)
    encoder.dispatchThreads(
      MTLSize(width: k.count, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
    commands.commit()
    commands.waitUntilCompleted()
    separable = try SSBSeparableReconstruction(
      device: device, library: library, geometry: geometry, k: k)
    try separable.prepare(probe: probe)
  }

  /// The soft aperture's outer radius in inverse Angstrom, rounded outwards (float32, as the kernels).
  static func supportRadius(_ geometry: MetalSSBGeometry) -> Float {
    var radius =
      (geometry.semiangleRadians
        + 0.5 * max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians))
      / geometry.wavelengthAngstroms
    radius *= 1 + 8 * 1.1920928955078125e-7
    return radius
  }

  static func parameters(_ geometry: MetalSSBGeometry, _ point: SSBModelPoint) -> [Float] {
    [
      geometry.wavelengthAngstroms, geometry.semiangleRadians, geometry.angularSamplingYRadians,
      geometry.angularSamplingXRadians, Float(point.c10Angstroms), Float(point.c12Angstroms),
      Float(cos(2 * point.phi12Radians)), Float(sin(2 * point.phi12Radians)),
      geometry.phaseFactor ?? Float(Double.pi / Double(geometry.wavelengthAngstroms)),
      Float(point.depthSpreadAngstroms), Float(point.sampleTiltMrad[0] * 1e-3),
      Float(point.sampleTiltMrad[1] * 1e-3),
    ]
  }

  /// Output frequencies, retained support rows and columns, and whether the half spectrum suffices
  /// (`_coherent_support`, `_coherent_columns`).
  private func plan(factor: Int, higher: Bool) throws -> Plan {
    let key = higher ? -factor : factor
    if let plan = plans[key] { return plan }
    let source = geometry.qxByRow.count
    let n = source * factor
    func frequencies(_ step: Float) -> [Float] {
      (0..<n).map { (index: Int) -> Float in
        let integer: Int = index < (n - 1) / 2 + 1 ? index : index - n
        let normalized: Double = Double(integer) * (1.0 / Double(n))
        let scaled: Double = normalized * Double(n)
        return Float(scaled * Double(step))
      }
    }
    let rowStep = geometry.qxByRow[1]
    let colStep = geometry.qyByColumn[1]
    var radius = Double(
      (Double(geometry.semiangleRadians)
        + 0.5 * Double(max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians)))
        / Double(geometry.wavelengthAngstroms))
    radius *= 1 + 8 * Double(Float.ulpOfOne)
    let rowSupport = Double(kHost.map { abs($0.x) }.max() ?? 0) + radius
    let colSupport = Double(kHost.map { abs($0.y) }.max() ?? 0) + radius
    let half =
      !higher && factor > 1 && Double(n / 2) * Double(abs(rowStep)) > rowSupport
      && Double(n / 2) * Double(abs(colStep)) > colSupport
    let columns = half ? n / 2 + 1 : n
    func limit(_ value: Double, _ step: Float) -> Double {
      (value / Double(abs(step))).rounded(.up) + 1
    }
    func support(_ value: Double, _ step: Float, _ count: Int) -> [UInt32] {
      let limit = limit(value, step)
      let indices: [Int] = (0..<count).filter { (index: Int) -> Bool in
        let integer: Int = index < (n - 1) / 2 + 1 ? index : index - n
        return Double(abs(integer)) <= limit
      }
      return indices.map { UInt32($0) }
    }
    let rows = support(rowSupport, rowStep, n)
    let cols = support(colSupport, colStep, columns)
    let plan = Plan(
      qRow: try Self.buffer(device, frequencies(rowStep)),
      qCol: try Self.buffer(device, frequencies(colStep)),
      rowIndices: try Self.buffer(device, rows), colIndices: try Self.buffer(device, cols),
      rows: rows.count, cols: cols.count, half: half,
      rowLimit: Int(limit(rowSupport, rowStep)), colLimit: Int(limit(colSupport, colStep)))
    plans[key] = plan
    return plan
  }

  /// The coherent (`phase_of_mean`) phase.
  func reconstruct(
    _ point: SSBModelPoint, factor: Int,
    visit: (Int, @escaping MetalSSBJointModel.SpectrumVisitor) throws -> Void,
    isCancelled: () -> Bool
  ) throws -> (phase: MTLBuffer, rows: Int) {
    // 1x-4x: the interactive redraw kernel, the only production path. 8x (an explicit manual override above the
    // automatic 4x cap) is the one case on the earlier separable / per-term kernels.
    if factor <= 4 && !referenceKernels {
      return try reconstructRedraw(point, factor: factor, visit: visit, isCancelled: isCancelled)
    }
    precondition(factor == 8 || referenceKernels, "1x-4x reconstruction uses the redraw kernel only")
    let higher = point.hasHigherOrder
    let useSeparable = self.useSeparable && !higher
    lastKernel = MetalKernelIdentity(
      name: SSBKernelRegistry.perTerm.name, version: SSBKernelRegistry.perTerm.version,
      sourceSHA256: perTermSource,
      variant: "factor=\(factor) \(useSeparable ? "separable" : "per-term")")
    let plan = try plan(factor: factor, higher: higher)
    let coefficients = higher ? try Self.buffer(device, point.reconstructionCoefficients) : nil
    let selectedProbe = higher ? higherProbePipeline : probePipeline
    let selectedCoherent = higher ? higherCoherentPipeline : coherentPipeline
    let source = geometry.qxByRow.count
    let n = source * factor
    var params = Self.parameters(geometry, point)
    let total = try workspace("total", n * n * 8)
    let commands = try command()
    if !useSeparable {
      guard let blit = commands.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
      blit.fill(buffer: total, range: 0..<(n * n * 8), value: 0)
      blit.endEncoding()
    }
    var count = UInt32(kHost.count)
    let probeEncoder = try encoder(commands, selectedProbe)
    probeEncoder.setBuffer(kBuffer, offset: 0, index: 0)
    probeEncoder.setBytes(&params, length: params.count * 4, index: 1)
    probeEncoder.setBuffer(probe, offset: 0, index: 2)
    probeEncoder.setBytes(&count, length: 4, index: 3)
    if let coefficients { probeEncoder.setBuffer(coefficients, offset: 0, index: 4) }
    dispatch(probeEncoder, width: kHost.count)
    if useSeparable {
      do {
        try separable.encode(
          params: params, factor: factor, probe: probe, full: total, commands: commands,
          visit: visit,
          isCancelled: isCancelled)
        return try finishPhase(total: total, n: n, commands: commands, mirror: false)
      } catch is SSBSeparableReconstruction.NotCached {
        // A streamed (uncached) spectrum: the per-term kernel handles every layout.
      }
    }
    var pending: MTLCommandBuffer? = try useSeparable ? command() : commands
    if useSeparable, let first = pending {
      guard let blit = first.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
      blit.fill(buffer: total, range: 0..<(n * n * 8), value: 0)
      blit.endEncoding()
      let probeAgain = try encoder(first, selectedProbe)
      probeAgain.setBuffer(kBuffer, offset: 0, index: 0)
      probeAgain.setBytes(&params, length: params.count * 4, index: 1)
      probeAgain.setBuffer(probe, offset: 0, index: 2)
      probeAgain.setBytes(&count, length: 4, index: 3)
      if let coefficients { probeAgain.setBuffer(coefficients, offset: 0, index: 4) }
      dispatch(probeAgain, width: kHost.count)
    }
    try withoutActuallyEscaping(isCancelled) { cancelled in
      try visit(512) { [self] spectra, offset, half, bfOffset, batch in
        if cancelled() { throw CancellationError() }
        let commands = try pending ?? command()
        var grid = Grid(
          sourceN: UInt32(source), outputN: UInt32(n),
          gCols: UInt32(half ? source / 2 + 1 : source),
          count: UInt32(batch), bfOffset: UInt32(bfOffset), sourceHalf: half ? 1 : 0,
          rows: UInt32(plan.rows), cols: UInt32(plan.cols), chunk: UInt32(Self.chunk),
          groups: UInt32((batch + Self.chunk - 1) / Self.chunk),
          groupOffset: UInt32(bfOffset / Self.chunk), depth: point.depthSpreadAngstroms != 0 ? 1 : 0
        )
        let compute = try encoder(commands, selectedCoherent)
        compute.setBuffer(spectra, offset: offset, index: 0)
        compute.setBuffer(plan.qRow, offset: 0, index: 1)
        compute.setBuffer(plan.qCol, offset: 0, index: 2)
        compute.setBuffer(kBuffer, offset: 0, index: 3)
        compute.setBytes(&params, length: params.count * 4, index: 4)
        compute.setBuffer(probe, offset: 0, index: 5)
        compute.setBuffer(bounds, offset: 0, index: 6)
        compute.setBuffer(plan.rowIndices, offset: 0, index: 7)
        compute.setBuffer(plan.colIndices, offset: 0, index: 8)
        compute.setBuffer(total, offset: 0, index: 9)
        compute.setBytes(&grid, length: MemoryLayout<Grid>.stride, index: 10)
        if let coefficients { compute.setBuffer(coefficients, offset: 0, index: 11) }
        compute.dispatchThreads(
          MTLSize(width: plan.cols, height: plan.rows, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
        compute.endEncoding()
        if half {
          pending = commands
        } else {
          // A streamed batch reuses the engine's spectrum workspace: finish before the next batch.
          try finish(commands)
          pending = nil
        }
      }
    }
    return try finishPhase(total: total, n: n, commands: pending ?? command(), mirror: plan.half)
  }

  private func redrawPipeline(factor: Int, degree: Int, depth: Bool) throws -> MTLComputePipelineState {
    let key = "\(factor)-\(degree)-\(depth)"
    if let pipeline = redrawPipelines[key] { return pipeline }
    let constants = MTLFunctionConstantValues()
    var f = UInt32(factor), d = UInt32(degree), thick = depth
    constants.setConstantValue(&f, type: .uint, index: 0)
    constants.setConstantValue(&d, type: .uint, index: 1)
    constants.setConstantValue(&thick, type: .bool, index: 2)
    let function = try library.makeFunction(name: "redraw_coherent", constantValues: constants)
    let pipeline = try device.makeComputePipelineState(function: function)
    redrawPipelines[key] = pipeline
    return pipeline
  }

  /// The interactive redraw (`ssb_redraw.metal`): the source half, every alias per thread, then mirrored.
  private func reconstructRedraw(
    _ point: SSBModelPoint, factor: Int,
    visit: (Int, @escaping MetalSSBJointModel.SpectrumVisitor) throws -> Void,
    isCancelled: () -> Bool
  ) throws -> (phase: MTLBuffer, rows: Int) {
    let higher = point.hasHigherOrder
    let plan = try plan(factor: factor, higher: true)
    var params = Self.parameters(geometry, point)
    let packed = higher ? point.reconstructionCoefficients : nil
    var polynomial =
      packed.map { SSBRedrawPolynomial.full(coefficients: $0, wavelength: params[0]) }
      ?? SSBRedrawPolynomial.primary(params)
    let pipeline = try redrawPipeline(
      factor: factor, degree: higher ? 6 : 2, depth: point.depthSpreadAngstroms != 0)
    lastKernel = MetalKernelIdentity(
      name: SSBKernelRegistry.redraw.name, version: SSBKernelRegistry.redraw.version,
      sourceSHA256: redrawSource,
      variant: "factor=\(factor) degree=\(higher ? 6 : 2) depth=\(point.depthSpreadAngstroms != 0 ? "on" : "off")")
    let source = geometry.qxByRow.count
    let n = source * factor
    let total = try workspace("total", n * n * 8)
    let commands = try command()
    guard let blit = commands.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
    blit.fill(buffer: total, range: 0..<(n * n * 8), value: 0)
    blit.endEncoding()
    var count = UInt32(kHost.count)
    // The probe geometry of every BF pixel and chi expanded about it (7 x 7 Taylor coefficients each), read by
    // every batch below.
    let taylor = try workspace("taylor", kHost.count * 49 * 4)
    let taylorEncoder = try encoder(commands, redrawTaylorPipeline)
    taylorEncoder.setBuffer(kBuffer, offset: 0, index: 0)
    taylorEncoder.setBytes(&params, length: params.count * 4, index: 1)
    taylorEncoder.setBytes(&polynomial, length: polynomial.count * 4, index: 2)
    taylorEncoder.setBuffer(probe, offset: 0, index: 3)
    taylorEncoder.setBuffer(taylor, offset: 0, index: 4)
    taylorEncoder.setBytes(&count, length: 4, index: 5)
    dispatch(taylorEncoder, width: kHost.count)
    // Source rows above N/2 are mirrored, except the output Nyquist column, which has no partner on the grid:
    // those rows run only when that column has support.
    let nyquist = abs(n / 2 < (n - 1) / 2 + 1 ? n / 2 : n / 2 - n) <= plan.colLimit
    let radius = Double(maximumK) + Double(Self.supportRadius(geometry))
    let rowFloats = 4 + 2 * (higher ? 8 : 4)
    var pending: MTLCommandBuffer? = commands
    try withoutActuallyEscaping(isCancelled) { cancelled in
      try visit(512) { [self] spectra, offset, half, bfOffset, batch in
        if cancelled() { throw CancellationError() }
        let commands = try pending ?? command()
        var grid = RedrawGrid(
          sourceN: UInt32(source), outputN: UInt32(n),
          gCols: UInt32(half ? source / 2 + 1 : source), count: UInt32(batch),
          bfOffset: UInt32(bfOffset), sourceHalf: half ? 1 : 0, rowLimit: UInt32(plan.rowLimit),
          colLimit: UInt32(plan.colLimit), support2: Float(radius * radius * (1 + 1e-5)),
          pixels: UInt32(kHost.count))
        let compute = try encoder(commands, pipeline)
        compute.setBuffer(spectra, offset: offset, index: 0)
        compute.setBuffer(kBuffer, offset: 0, index: 1)
        compute.setBytes(&params, length: params.count * 4, index: 2)
        compute.setBuffer(probe, offset: 0, index: 3)
        compute.setBuffer(taylor, offset: 0, index: 4)
        compute.setBuffer(total, offset: 0, index: 5)
        compute.setBytes(&grid, length: MemoryLayout<RedrawGrid>.stride, index: 6)
        compute.setBuffer(plan.qRow, offset: 0, index: 7)
        compute.setBuffer(plan.qCol, offset: 0, index: 8)
        // Per batch of 32 BF pixels: pixel values, the coefficient rows of each alias row, and per SIMD group
        // and alias the column interval.
        compute.setThreadgroupMemoryLength(Self.redrawBatch * 16, index: 0)
        compute.setThreadgroupMemoryLength(Self.redrawBatch * factor * rowFloats * 4, index: 1)
        compute.setThreadgroupMemoryLength(Self.redrawWidth / 32 * factor * factor * 16, index: 2)
        compute.dispatchThreadgroups(
          MTLSize(
            width: source / Self.redrawWidth, height: nyquist ? source : source / 2 + 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: Self.redrawWidth, height: 1, depth: 1))
        compute.endEncoding()
        if half {
          pending = commands
        } else {
          // A streamed batch reuses the engine's spectrum workspace: finish before the next batch.
          try finish(commands)
          pending = nil
        }
      }
    }
    let final = try pending ?? command()
    var size = SIMD2<UInt32>(UInt32(n), UInt32(source))
    let mirror = try encoder(final, redrawMirrorPipeline)
    mirror.setBuffer(total, offset: 0, index: 0)
    mirror.setBytes(&size, length: 8, index: 1)
    mirror.dispatchThreads(
      MTLSize(width: n, height: n, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
    mirror.endEncoding()
    return try finishPhase(total: total, n: n, commands: final, mirror: false)
  }
  /// `ssb_redraw.metal` launch constants: BF pixels per shared batch and threads per threadgroup.
  static let redrawBatch = 32
  static let redrawWidth = 128

  /// Mirror (per-term half spectra), set the common origin, inverse-transform and take the phase.
  private func finishPhase(
    total: MTLBuffer, n: Int, commands final: MTLCommandBuffer, mirror half: Bool
  ) throws
    -> (phase: MTLBuffer, rows: Int)
  {
    if half {
      var size = UInt32(n)
      let mirror = try encoder(final, mirrorPipeline)
      mirror.setBuffer(total, offset: 0, index: 0)
      mirror.setBytes(&size, length: 4, index: 1)
      mirror.dispatchThreads(
        MTLSize(width: n - n / 2 - 1, height: n, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1))
      mirror.endEncoding()
    }
    // Every logical BF pixel, stored or not, carries the common real DC at the origin.
    var origin = SIMD2<Float>(
      Float(Double(geometry.logicalBrightfieldCount) * Double(geometry.dcValue.x)),
      Float(Double(geometry.logicalBrightfieldCount) * Double(geometry.dcValue.y)))
    let originEncoder = try encoder(final, originPipeline)
    originEncoder.setBuffer(total, offset: 0, index: 0)
    originEncoder.setBytes(&origin, length: 8, index: 1)
    dispatch(originEncoder, width: 1)
    let wave = try workspace("wave", n * n * 8)
    let shape: [NSNumber] = [NSNumber(value: n), NSNumber(value: n)]
    if fft[n] == nil {
      let graph = MPSGraph()
      graph.options = .none
      let input = graph.placeholder(shape: shape, dataType: .complexFloat32, name: "spectrum")
      let descriptor = MPSGraphFFTDescriptor()
      descriptor.inverse = true
      descriptor.scalingMode = .size
      let output = graph.fastFourierTransform(
        input, axes: [0, 1], descriptor: descriptor, name: "ifft2")
      fft[n] = (graph, input, output)
    }
    let (graph, input, output) = fft[n]!
    let scheduled = MPSCommandBuffer(commandBuffer: final)
    let execution = MPSGraphExecutionDescriptor()
    execution.waitUntilCompleted = false
    graph.encode(
      to: scheduled,
      feeds: [input: MPSGraphTensorData(total, shape: shape, dataType: .complexFloat32)],
      targetOperations: nil,
      resultsDictionary: [
        output: MPSGraphTensorData(wave, shape: shape, dataType: .complexFloat32)
      ],
      executionDescriptor: execution)
    let phase = try Self.allocate(device, n * n * 4, shared: true)
    var planeCount = UInt32(n * n)
    let phaseEncoder = try encoder(scheduled, phasePipeline)
    phaseEncoder.setBuffer(wave, offset: 0, index: 0)
    phaseEncoder.setBuffer(phase, offset: 0, index: 1)
    phaseEncoder.setBytes(&planeCount, length: 4, index: 2)
    dispatch(phaseEncoder, width: n * n)
    try finish(scheduled)
    return (phase, n)
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
  private func dispatch(_ encoder: MTLComputeCommandEncoder, width: Int) {
    encoder.dispatchThreads(
      MTLSize(width: width, height: 1, depth: 1),
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
  static func allocate(_ device: MTLDevice, _ length: Int, shared: Bool = false) throws -> MTLBuffer
  {
    guard
      let buffer = device.makeBuffer(
        length: max(4, length), options: shared ? .storageModeShared : .storageModePrivate)
    else {
      throw MetalSSBError.allocation(
        "SSB reconstruction workspace (\(length) bytes). Close inactive datasets and retry")
    }
    return buffer
  }
  static func buffer<T>(_ device: MTLDevice, _ values: [T]) throws -> MTLBuffer {
    guard
      let result = values.withUnsafeBytes({ bytes in
        device.makeBuffer(
          bytes: bytes.baseAddress!, length: max(4, bytes.count), options: .storageModeShared)
      })
    else { throw MetalSSBError.allocation("SSB reconstruction geometry") }
    return result
  }
}
