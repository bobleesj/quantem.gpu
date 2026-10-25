import Foundation
import Metal

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
    try SSBModelHigherOrder.validate((aberrations.higherOrder ?? []).map(SSBModelHigherOrder.init))
    guard packed.allSatisfy(\.isFinite), depthAngstroms >= 0
    else {
      throw MetalSSBError.invalidGeometry(
        "Joint SSB needs finite C10/C12, nonnegative depth and tilt in mrad."
      )
    }
  }
}

/// GPU phase in radians of one of the three reconstructions (`method`): SSB (`phase_of_mean`) or the regularized
/// direct `phase` / `complex` solves. Upsampling unfolds the measured Fourier aliases; it is not image interpolation.
public struct MetalSSBJointPhase: @unchecked Sendable {
  public let phase: MTLBuffer
  public let rows: Int
  public let columns: Int
  public let upsamplingFactor: Int
  public let wallSeconds: Double
  public let parameters: SSBJointParameters
  public let scanDetectorRotationDegrees: Double
  public let provenance: MetalSSBProvenance
  /// Exact Double point used for this phase, independently of the Float UI bridge.
  public var point: SSBModelPoint? = nil
  /// Method `.complex` only: the relative amplitude |exp(eta + i phi)| = exp(eta) of the regularized solve (float32,
  /// rows x columns), as the CUDA result's `amplitude`. Nil for `.ssb` and `.phase`.
  public var amplitude: MTLBuffer? = nil
  /// How the output factor was chosen: `.auto` (from the calibration, at most 4x) or an explicit
  /// `.factor(n)` override. Nil when the caller passed a bare `upsamplingFactor`.
  public var outputSampling: SSBOutputSampling? = nil
  /// The registered GPU kernel (name, version, source fingerprint, variant) that produced this result.
  public var kernel: MetalKernelIdentity? = nil
  /// The reconstruction method: `.ssb` (`phase_of_mean`), or the direct `.phase` / `.complex` solves.
  public var method: SSBReconstructionMethod = .ssb
  /// The regularization of a `.phase` or `.complex` result; nil for `.ssb`.
  public var regularization: SSBDirectRegularization? = nil
  /// Set when this is a drag preview (`dragPreview`), evaluated from a drag window rather than solved at its own
  /// optics; nil for every exact result. Previews are for display while dragging, never saved or exported.
  public var preview: SSBDragPreview? = nil
}

/// Serial native Metal implementation of QuantEM's joint tilt/depth SSB model.
/// Search uses the original scan grid and all selected BF pixels. The band is
/// an explicit part of the least-squares objective, in inverse Angstrom.
public final class MetalSSBJointModel {
  typealias SpectrumVisitor = (MTLBuffer, Int, Bool, Int, Int) throws -> Void
  typealias VisitSpectra = (Int, @escaping SpectrumVisitor) throws -> Void

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
  /// The scan-detector rotation the model's BF coordinates are rotated to; `retarget` changes it.
  public private(set) var scanDetectorRotationDegrees: Double
  public let bandInverseAngstroms: ClosedRange<Float>
  public let bandCount: Int
  private var grid: Grid
  private let qBuffer, indexBuffer, weights, bandG: MTLBuffer
  /// G(0, k) of each active BF pixel, the counts at k summed over the scan (the Poisson variance of G(q, k)).
  private let zeroFrequency: MTLBuffer
  /// sum_k |G(q, k)|^2 of each band frequency, computed on the first `bandPowerSummary()` call.
  private var frequencyPower: [Double]?
  private var kBuffer: MTLBuffer
  private let activeIndices: [Int]
  private let library: MTLLibrary
  private let gather: MTLComputePipelineState
  private let reduceBatch: MTLComputePipelineState
  /// Whether reconstructions use the separable kernels (default) or the per-term reference kernel.
  var separableReconstruction: Bool {
    get { coherent.useSeparable }
    set { coherent.useSeparable = newValue }
  }
  /// Tests only: false evaluates 1x-4x with the earlier kernels as a parity reference. Production
  /// reconstruction at 1x-4x always uses the redraw kernel; there is no runtime fallback.
  var redrawReconstruction: Bool {
    get { !coherent.referenceKernels }
    set { coherent.referenceKernels = !newValue }
  }
  private var angleSumPipelines: [Int: MTLComputePipelineState] = [:]
  private var neighbourBuffer: MTLBuffer
  private var rowStep = SIMD2<Float>(0, 0)
  private let makeProbeGeometry: MTLComputePipelineState
  private let probeGeometry: MTLBuffer
  private var batchPartial: MTLBuffer?
  private var batchPower: MTLBuffer?
  private var coherent: SSBCoherentReconstruction
  private var fifthOrder: SSBFifthOrderAgreement?
  /// GPU execution time (command-buffer GPU start to end) of the latest `agreements` or
  /// `fifthOrderAgreements` call, for benchmarks; excludes host encoding and the host band reduction.
  public private(set) var lastAgreementGPUSeconds = 0.0
  private var lowOrder: SSBLowOrderDerivatives?
  /// GPU execution time (command-buffer GPU start to end, summed over its command buffers) of the latest
  /// `lowOrderDerivatives` call, for benchmarks.
  public private(set) var lastLowOrderGPUSeconds = 0.0
  /// The value stage's part of `lastLowOrderGPUSeconds` (zero when the call had no separate value stage).
  public private(set) var lastLowOrderValueGPUSeconds = 0.0
  private var fifthOrderGradient: SSBFifthOrderGradient?
  /// The regularized direct solver (`phase`, `complex`), built on first use; it keeps the layout and the prepared
  /// normal equations of the current optics.
  private var directSolver: SSBDirectSolver?

  /// The number of active bright-field pixels of the objective.
  var brightFieldPixelCount: Int { activeIndices.count }

  public var wavelengthAngstroms: Double {
    geometry.modelWavelengthAngstroms ?? Double(geometry.wavelengthAngstroms)
  }
  public var semiangleRadians: Double {
    geometry.modelSemiangleRadians ?? Double(geometry.semiangleRadians)
  }

  init(
    device: MTLDevice, geometry: MetalSSBGeometry, activeIndices: [Int], rotation: Double,
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
    self.activeIndices = activeIndices
    scanDetectorRotationDegrees = rotation
    bandInverseAngstroms = band
    guard let queue = device.makeCommandQueue() else { throw MetalSSBError.commandQueue }
    self.queue = queue
    let options = MTLCompileOptions()
    options.fastMathEnabled = false
    let library = try device.makeLibrary(
      source: MetalSSBShaderSource.load([
        "ssb_fft", "ssb_joint", "ssb_model", "ssb_separable", "ssb_fifth_order",
        "ssb_redraw", "ssb_fifth_agreement", "ssb_fifth_order_gradient",
        "ssb_joint_agreement", "ssb_low_order_derivatives", "ssb_direct",
      ]),
      options: options)
    self.library = library
    func pipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw MetalSSBError.missingFunction(name)
      }
      return try device.makeComputePipelineState(function: function)
    }
    gather = try pipeline("joint_gather")
    reduceBatch = try pipeline("joint_power_batch")
    makeProbeGeometry = try pipeline("joint_probe_geometry")
    let n = geometry.qxByRow.count
    var indices: [SIMD2<UInt32>] = []
    var qs: [SIMD2<Float>] = []
    var ws: [Float] = []
    for r in 0..<n {
      for c in 0...(n / 2) {
        let q = SIMD2(geometry.qxByRow[r], geometry.qyByColumn[c])
        // NumPy's float32 hypot, as the MPS band.
        let radius = hypotf(q.x, q.y)
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
    let ks = Self.rotated(geometry: geometry, activeIndices: activeIndices, rotation: rotation)
    kBuffer = try Self.buffer(device, ks)
    let steps = Self.rowSteps(ks)
    neighbourBuffer = try Self.buffer(device, steps.neighbour)
    rowStep = steps.step
    coherent = try SSBCoherentReconstruction(
      device: device, queue: queue, library: library, geometry: geometry, kBuffer: kBuffer, k: ks)
    probeGeometry = try Self.allocate(device, activeIndices.count * 16)
    qBuffer = try Self.buffer(device, qs)
    indexBuffer = try Self.buffer(device, indices)
    weights = try Self.buffer(device, ws)
    bandG = try Self.allocate(device, activeIndices.count * bandCount * 8)
    zeroFrequency = try Self.allocate(device, activeIndices.count * 8, shared: true)
    let chunks = (activeIndices.count + 255) / 256
    grid = Grid(
      n: UInt32(n), outputN: UInt32(n), count: 0, bfOffset: 0,
      totalBF: UInt32(activeIndices.count), bandCount: UInt32(bandCount), sourceHalf: 1,
      chunks: UInt32(chunks),
      wl: geometry.wavelengthAngstroms, semiangle: geometry.semiangleRadians,
      angularRow: geometry.angularSamplingYRadians, angularColumn: geometry.angularSamplingXRadians,
      dqRow: geometry.qxByRow[1], dqColumn: geometry.qyByColumn[1],
      dcReal: geometry.dcValue.x, dcImag: geometry.dcValue.y,
      factor: geometry.phaseFactor ?? Float.pi / geometry.wavelengthAngstroms)
    try encodeProbeGeometry()
    try withoutActuallyEscaping(isCancelled) { cancelled in
      try visit(8) { [self] source, offset, half, bfOffset, count in
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
        // G(0, k) is the first element of each spectrum, Hermitian half or full plane.
        guard let blit = commands.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
        let spectrumBytes = (half ? n * (n / 2 + 1) : n * n) * 8
        for i in 0..<count {
          blit.copy(
            from: source, sourceOffset: offset + i * spectrumBytes, to: zeroFrequency,
            destinationOffset: (bfOffset + i) * 8, size: 8)
        }
        blit.endEncoding()
        try finish(commands)
      }
    }
  }

  /// BF coordinates rotated by `rotation`, as the MPS preparation: the angle in double precision, the
  /// rotation in float32 (`kx cos + ky sin`, `-kx sin + ky cos`).
  static func rotated(geometry: MetalSSBGeometry, activeIndices: [Int], rotation: Double) -> [SIMD2<
    Float
  >] {
    let delta = (-(rotation - Double(geometry.referenceRotationDegrees))) * (Double.pi / 180)
    let cosine = Float(cos(delta))
    let sine = Float(sin(delta))
    return activeIndices.map { i -> SIMD2<Float> in
      let x = geometry.brightfieldKX[i]
      let y = geometry.brightfieldKY[i]
      return SIMD2(x * cosine + y * sine, -x * sine + y * cosine)
    }
  }

  /// 1 for a stored BF pixel whose predecessor is its left neighbour in the detector row, and the row step
  /// (Python MPS `resident_fit._row_steps`, from the rotated float32 coordinates in double precision).
  static func rowSteps(_ k: [SIMD2<Float>]) -> (neighbour: [Float], step: SIMD2<Float>) {
    var neighbour = [Float](repeating: 0, count: k.count)
    guard k.count > 1 else { return (neighbour, .zero) }
    let differences = (1..<k.count).map {
      (Double(k[$0].x) - Double(k[$0 - 1].x), Double(k[$0].y) - Double(k[$0 - 1].y))
    }
    let lengths = differences.map { hypot($0.0, $0.1) }
    let shortest = lengths.min()!
    let short = zip(differences, lengths).filter { $0.1 <= 1.5 * shortest }.map(\.0)
    func median(_ values: [Double]) -> Double {
      let sorted = values.sorted()
      let middle = sorted.count / 2
      return sorted.count % 2 == 1 ? sorted[middle] : 0.5 * (sorted[middle - 1] + sorted[middle])
    }
    let guess = (median(short.map(\.0)), median(short.map(\.1)))
    let scale = hypot(guess.0, guess.1)
    var sum = (0.0, 0.0)
    var count = 0
    for (index, difference) in differences.enumerated()
    where hypot(difference.0 - guess.0, difference.1 - guess.1) <= 1e-3 * scale {
      neighbour[index + 1] = 1
      sum.0 += difference.0
      sum.1 += difference.1
      count += 1
    }
    guard count > 0 else { return (neighbour, .zero) }
    return (neighbour, SIMD2(Float(sum.0 / Double(count)), Float(sum.1 / Double(count))))
  }

  private func encodeProbeGeometry() throws {
    let commands = try command()
    let encoder = try encoder(commands, makeProbeGeometry)
    encoder.setBuffer(kBuffer, offset: 0, index: 0)
    encoder.setBuffer(probeGeometry, offset: 0, index: 1)
    var probeGrid = grid
    encoder.setBytes(&probeGrid, length: MemoryLayout<Grid>.stride, index: 2)
    dispatch(encoder, width: activeIndices.count)
    try finish(commands)
  }

  /// Rotate the model to another scan-detector rotation (the other 180-degree branch, or a rotation
  /// scan) without gathering the measured band evidence again.
  public func retarget(scanDetectorRotationDegrees rotation: Double) throws {
    try validatePreparation()
    guard rotation.isFinite else {
      throw MetalSSBError.invalidGeometry("Choose a finite rotation.")
    }
    if rotation == scanDetectorRotationDegrees { return }
    let ks = Self.rotated(geometry: geometry, activeIndices: activeIndices, rotation: rotation)
    kBuffer = try Self.buffer(device, ks)
    let steps = Self.rowSteps(ks)
    neighbourBuffer = try Self.buffer(device, steps.neighbour)
    rowStep = steps.step
    coherent = try SSBCoherentReconstruction(
      device: device, queue: queue, library: library, geometry: geometry, kBuffer: kBuffer, k: ks)
    try encodeProbeGeometry()
    scanDetectorRotationDegrees = rotation
  }

  /// Negative explained Fourier power. Lower is better; it is not a confidence
  /// score or a certificate of the physical scan-rotation branch.
  public func loss(_ parameters: SSBJointParameters) throws -> Double {
    try losses([parameters])[0]
  }

  /// Evaluate independent candidates together on the same measured BF evidence.
  /// Geometry and input reads are shared across up to eight candidates. The
  /// precision and accumulation order of each scalar objective are retained.
  public func losses(
    _ parameters: [SSBJointParameters], isCancelled: () -> Bool = { false }
  ) throws -> [Double] {
    for value in parameters { try value.validate() }
    return try agreements(parameters.map(SSBModelPoint.init), isCancelled: isCancelled).map { -$0 }
  }

  /// The least-squares agreement of the transfer model with the measured double-overlap data, larger is
  /// better: per band frequency |sum_k G conj(gamma)|^2 / sum_k |gamma|^2, summed with the half-plane weights
  /// (Python `thick_fit_batch`, `resident_fit.evaluate`).
  public func agreements(_ points: [SSBModelPoint], isCancelled: () -> Bool = { false }) throws
    -> [Double]
  {
    for point in points { try point.validate() }
    if points.contains(where: \.hasHigherOrder) {
      return try fifthOrderAgreements(points, isCancelled: isCancelled)
    }
    return try agreements(rows: points.map(\.trialRow), isCancelled: isCancelled)
  }

  /// Joint fifth-order agreement, including at zero higher-order coefficients. The optimizer may supply
  /// exact Cartesian (14×2) packs to avoid a polar roundtrip; these include the 1/(n+1) prefactor.
  /// Scientific geometry, BF sums and weighted reduction remain float32 on Metal.
  public func fifthOrderAgreements(
    _ points: [SSBModelPoint], packedCoefficients: [[Float]]? = nil,
    isCancelled: () -> Bool = { false }
  ) throws -> [Double] {
    try validatePreparation()
    for point in points { try point.validate() }
    let coefficients = packedCoefficients ?? points.map(\.reconstructionCoefficients)
    guard coefficients.count == points.count,
      coefficients.allSatisfy({ $0.count == 28 && $0.allSatisfy(\.isFinite) })
    else {
      throw MetalSSBError.invalidGeometry(
        "Supply 28 finite Cartesian coefficients per SSB candidate.")
    }
    if isCancelled() { throw CancellationError() }
    if points.isEmpty { return [] }
    let rows = try fifthOrderRows(points, coefficients)
    let scores = try fifthOrder!.evaluate(
      rows: rows, geometry: geometry, k: kBuffer, q: qBuffer, band: bandG, weights: weights,
      probeGeometry: probeGeometry, count: activeIndices.count, bandCount: bandCount,
      isCancelled: isCancelled)
    lastAgreementGPUSeconds = fifthOrder!.lastGPUSeconds
    return scores
  }

  /// The kernel rows of `fifthOrderAgreements`: depth, tilt row and column (rad), the 28 packed coefficients.
  private func fifthOrderRows(_ points: [SSBModelPoint], _ coefficients: [[Float]]) throws -> [[Float]] {
    if fifthOrder == nil {
      let q = Array(
        UnsafeBufferPointer(
          start: qBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self), count: bandCount))
      fifthOrder = try SSBFifthOrderAgreement(device: device, queue: queue, library: library, q: q)
    }
    return zip(points, coefficients).map { point, packed in
      [
        Float(point.depthSpreadAngstroms), Float(point.sampleTiltMrad[0] * 1e-3),
        Float(point.sampleTiltMrad[1] * 1e-3),
      ] + packed
    }
  }

  /// The fifth-order agreement of each candidate (bit-identical to `fifthOrderAgreements`) and its exact gradient
  /// with respect to the objective row [depth (A), tilt row (rad), tilt column (rad), 28 packed coefficients (A)].
  /// `agreement64` sums the same float32 per-frequency powers in Double (finer than the float32 total).
  /// Per-term arithmetic is float32 on Metal; the gradient partials are reduced in Double.
  /// `gradients: false` returns the agreements alone (empty gradients) and keeps their projections, so
  /// `fifthOrderRetainedGradients` of the same candidates adds only the gradient pass.
  /// A `kStride` above 1 uses every kStride-th BF pixel only (unscaled): a steering model, never the objective.
  func fifthOrderAgreementsAndGradients(
    _ points: [SSBModelPoint], packedCoefficients: [[Float]], kStride: Int = 1,
    gradients: Bool = true, isCancelled: () -> Bool = { false }
  ) throws -> [(agreement: Double, agreement64: Double, gradient: [Double])] {
    try validatePreparation()
    for point in points { try point.validate() }
    guard packedCoefficients.count == points.count,
      packedCoefficients.allSatisfy({ $0.count == 28 && $0.allSatisfy(\.isFinite) })
    else {
      throw MetalSSBError.invalidGeometry(
        "Supply 28 finite Cartesian coefficients per SSB candidate.")
    }
    if isCancelled() { throw CancellationError() }
    if points.isEmpty { return [] }
    let rows = try fifthOrderRows(points, packedCoefficients)
    if fifthOrderGradient == nil {
      fifthOrderGradient = try SSBFifthOrderGradient(
        device: device, queue: queue, library: library, objective: fifthOrder!)
    }
    let values = try fifthOrderGradient!.evaluate(
      rows: rows, geometry: geometry, k: kBuffer, q: qBuffer, band: bandG, weights: weights,
      probeGeometry: probeGeometry, count: activeIndices.count, bandCount: bandCount,
      kStride: kStride, gradients: gradients, isCancelled: isCancelled)
    lastAgreementGPUSeconds = fifthOrderGradient!.lastGPUSeconds
    return values.map { ($0.agreement, $0.agreement64, $0.gradient) }
  }

  /// The agreements and gradients of the candidates of the last value-only
  /// `fifthOrderAgreementsAndGradients(gradients: false)`, reusing its objective pass; other candidates are
  /// evaluated in full.
  func fifthOrderRetainedGradients(
    _ points: [SSBModelPoint], packedCoefficients: [[Float]], kStride: Int = 1,
    isCancelled: () -> Bool = { false }
  ) throws -> [(agreement: Double, agreement64: Double, gradient: [Double])] {
    try validatePreparation()
    guard let fifthOrderGradient else {
      return try fifthOrderAgreementsAndGradients(
        points, packedCoefficients: packedCoefficients, kStride: kStride, isCancelled: isCancelled)
    }
    let rows = try fifthOrderRows(points, packedCoefficients)
    let values = try fifthOrderGradient.retainedGradients(
      rows: rows, geometry: geometry, k: kBuffer, q: qBuffer, band: bandG, weights: weights,
      probeGeometry: probeGeometry, count: activeIndices.count, bandCount: bandCount,
      kStride: kStride, isCancelled: isCancelled)
    lastAgreementGPUSeconds = fifthOrderGradient.lastGPUSeconds
    return values.map { ($0.agreement, $0.agreement64, $0.gradient) }
  }

  /// The low-order agreement with its exact gradient and Hessian at each row of Newton coordinates
  /// (C10, C12 cos 2 phi12, C12 sin 2 phi12 [A], tilt row, tilt column [mrad], depth spread [A]).
  func lowOrderDerivatives(
    _ rows: [[Double]], twoStage: Bool = false, isCancelled: () -> Bool = { false }
  ) throws -> [SSBLowOrderDerivatives.Result] {
    let (evaluator, inputs) = try lowOrderEvaluator()
    let results = try evaluator.evaluate(
      rows: rows, inputs: inputs, twoStage: twoStage, isCancelled: isCancelled)
    lastLowOrderGPUSeconds = evaluator.lastGPUSeconds
    lastLowOrderValueGPUSeconds = evaluator.lastValueGPUSeconds
    return results
  }

  /// The first stage of `lowOrderDerivatives`: the agreement at each row of Newton coordinates, keeping the sums
  /// that `lowOrderDerivatives(at:)` needs for any of those rows (`withDerivatives`: forming every row's
  /// derivatives in the same pass).
  func lowOrderValues(
    _ rows: [[Double]], withDerivatives: Bool = false, isCancelled: () -> Bool = { false }
  ) throws -> [Double] {
    let (evaluator, inputs) = try lowOrderEvaluator()
    let values = try evaluator.values(
      rows: rows, inputs: inputs, withDerivatives: withDerivatives, isCancelled: isCancelled)
    lastLowOrderGPUSeconds = evaluator.lastGPUSeconds
    return values
  }

  /// The second stage: value, gradient and Hessian at the given rows of the latest `lowOrderValues` call.
  func lowOrderDerivatives(at indices: [Int], isCancelled: () -> Bool = { false }) throws
    -> [SSBLowOrderDerivatives.Result]
  {
    let (evaluator, _) = try lowOrderEvaluator()
    let results = try evaluator.derivatives(at: indices, isCancelled: isCancelled)
    lastLowOrderGPUSeconds = evaluator.lastGPUSeconds
    return results
  }

  private func lowOrderEvaluator() throws -> (SSBLowOrderDerivatives, SSBLowOrderDerivatives.Inputs) {
    try validatePreparation()
    if lowOrder == nil {
      lowOrder = try SSBLowOrderDerivatives(device: device, queue: queue, library: library)
    }
    var p = grid
    var extra = angleSumExtra(first: 0)
    let inputs = SSBLowOrderDerivatives.Inputs(
      grid: withUnsafeBytes(of: &p) { Array($0) }, extra: withUnsafeBytes(of: &extra) { Array($0) },
      bandCount: bandCount, count: activeIndices.count, chunks: Int(grid.chunks), band: bandG,
      k: kBuffer, q: qBuffer, weights: weights, probeGeometry: probeGeometry,
      neighbour: neighbourBuffer, rowStep: rowStep)
    return (lowOrder!, inputs)
  }

  /// Benchmarks: `lowOrderDerivatives` (the fused pass, or `twoStage`) as the search's derivative type.
  public func lowOrderAgreementDerivatives(_ rows: [[Double]], twoStage: Bool = false) throws
    -> [SSBAgreementSearch.Derivatives]
  {
    try lowOrderDerivatives(rows, twoStage: twoStage).map {
      SSBAgreementSearch.Derivatives(value: $0.value, gradient: $0.gradient, hessian: $0.hessian)
    }
  }

  /// The exact float32 inputs of the agreement objective at the current rotation, for an independent
  /// reference implementation: rotated BF coordinates `k` (1/A), band frequencies `q` (1/A), half-plane
  /// weights, per-BF probe geometry (alpha^2, cos 2phi, sin 2phi, aperture) and the gathered evidence
  /// `G[bf * bandCount + band]`. The physical constants are `wavelengthAngstroms`, `semiangleRadians`, the
  /// angular sampling and `phaseFactor` of the geometry.
  public func objectiveArrays() throws -> (
    k: [SIMD2<Float>], q: [SIMD2<Float>], weights: [Float], probe: [SIMD4<Float>],
    evidence: [SIMD2<Float>], constants: [String: Double]
  ) {
    try validatePreparation()
    func read<T>(_ buffer: MTLBuffer, count: Int, as type: T.Type) throws -> [T] {
      let bytes = count * MemoryLayout<T>.stride
      let shared = try Self.allocate(device, bytes, shared: true)
      let commands = try command()
      guard let blit = commands.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
      blit.copy(from: buffer, sourceOffset: 0, to: shared, destinationOffset: 0, size: bytes)
      blit.endEncoding()
      try finish(commands)
      return Array(
        UnsafeBufferPointer(start: shared.contents().assumingMemoryBound(to: T.self), count: count))
    }
    let count = activeIndices.count
    return (
      k: try read(kBuffer, count: count, as: SIMD2<Float>.self),
      q: try read(qBuffer, count: bandCount, as: SIMD2<Float>.self),
      weights: try read(weights, count: bandCount, as: Float.self),
      probe: try read(probeGeometry, count: count, as: SIMD4<Float>.self),
      evidence: try read(bandG, count: count * bandCount, as: SIMD2<Float>.self),
      constants: [
        "wavelength_A": Double(geometry.wavelengthAngstroms),
        "semiangle_rad": Double(geometry.semiangleRadians),
        "angular_row_rad": Double(geometry.angularSamplingYRadians),
        "angular_column_rad": Double(geometry.angularSamplingXRadians),
        "factor": Double(geometry.phaseFactor ?? Float.pi / geometry.wavelengthAngstroms),
        "model_wavelength_A": wavelengthAngstroms, "model_semiangle_rad": semiangleRadians,
        "rotation_deg": scanDetectorRotationDegrees,
        "bf_count": Double(count), "band_count": Double(bandCount),
      ]
    )
  }

  /// G's power over the agreement band and its noise (Python `band_power`): P = sum_q w_q sum_k |G(q, k)|^2 (the
  /// agreement's upper bound) over the active BF pixels, the band weight sum_q w_q, and the noise variance of one
  /// G(q, k) for counted electrons, the mean over k of G(0, k). The per-frequency sums are computed once per model,
  /// in Double from the float32 evidence; the weights are read at each call.
  public func bandPowerSummary(isCancelled: () -> Bool = { false }) throws -> SSBBandPower {
    try validatePreparation()
    let count = activeIndices.count
    if frequencyPower == nil {
      let bands = bandCount
      var power = [Double](repeating: 0, count: bands)
      let rowBytes = bands * 8
      let rowsPerCopy = max(1, min(count, (256 << 20) / rowBytes))
      let staging = try Self.allocate(device, rowsPerCopy * rowBytes, shared: true)
      let workers = max(1, min(ProcessInfo.processInfo.activeProcessorCount, 16, bands))
      for first in stride(from: 0, to: count, by: rowsPerCopy) {
        if isCancelled() { throw CancellationError() }
        let rows = min(rowsPerCopy, count - first)
        let commands = try command()
        guard let blit = commands.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
        blit.copy(
          from: bandG, sourceOffset: first * rowBytes, to: staging, destinationOffset: 0,
          size: rows * rowBytes)
        blit.endEncoding()
        try finish(commands)
        let g = UnsafePointer(staging.contents().assumingMemoryBound(to: SIMD2<Float>.self))
        power.withUnsafeMutableBufferPointer { sums in
          let out = sums.baseAddress!
          // Each worker owns a range of band frequencies, so no partial sums need combining.
          DispatchQueue.concurrentPerform(iterations: workers) { worker in
            let lower = bands * worker / workers
            let upper = bands * (worker + 1) / workers
            for row in 0..<rows {
              let base = g + row * bands
              for i in lower..<upper {
                let v = base[i]
                out[i] += Double(v.x) * Double(v.x) + Double(v.y) * Double(v.y)
              }
            }
          }
        }
      }
      frequencyPower = power
    }
    let w = weights.contents().assumingMemoryBound(to: Float.self)
    var total = 0.0
    var weight = 0.0
    for (i, p) in frequencyPower!.enumerated() {
      total += Double(w[i]) * p
      weight += Double(w[i])
    }
    let dc = zeroFrequency.contents().assumingMemoryBound(to: SIMD2<Float>.self)
    var counts = 0.0
    for i in 0..<count { counts += Double(dc[i].x) }
    return SSBBandPower(
      power: total, bandWeight: weight, noiseVariance: counts / Double(count),
      brightFieldPixelCount: count)
  }

  /// The band frequencies q (1/A, row and column) of the agreement, in the order of its weights.
  var bandFrequencies: [SIMD2<Float>] {
    Array(UnsafeBufferPointer(start: qBuffer.contents().assumingMemoryBound(to: SIMD2<Float>.self), count: bandCount))
  }

  /// Evaluate `body` with the per-frequency agreement weights multiplied by `mask` (0 or 1 per band frequency), as
  /// for a fit on part of the band; the weights are restored afterwards, also when `body` throws.
  func withBandMask<T>(_ mask: [Bool], _ body: () throws -> T) throws -> T {
    guard mask.count == bandCount else {
      throw MetalSSBError.invalidGeometry("Supply one mask entry per band frequency.")
    }
    let target = weights.contents().assumingMemoryBound(to: Float.self)
    let original = Array(UnsafeBufferPointer(start: target, count: bandCount))
    for i in 0..<bandCount where !mask[i] { target[i] = 0 }
    defer { for i in 0..<bandCount { target[i] = original[i] } }
    return try body()
  }

  func agreements(rows: [[Float]], isCancelled: () -> Bool) throws -> [Double] {
    try validatePreparation()
    if rows.isEmpty { return [] }
    if isCancelled() { throw CancellationError() }
    let group = 8
    if batchPartial == nil {
      batchPartial = try Self.allocate(device, Int(grid.chunks) * bandCount * 16 * group)
    }
    guard let batchPartial else { throw MetalSSBError.commandQueue }
    if (batchPower?.length ?? 0) < rows.count * bandCount * 16 {
      batchPower = try Self.allocate(device, rows.count * bandCount * 16, shared: true)
    }
    guard let batchPower else { throw MetalSSBError.commandQueue }
    // Every candidate's partial sums and band sums in one command buffer, one wait.
    let commands = try command()
    for first in stride(from: 0, to: rows.count, by: group) {
      let candidates = rows[first..<min(first + group, rows.count)]
      let packed = candidates.flatMap { $0 }
      var p = grid
      var batch = UInt32(candidates.count)
      // Four candidates per thread, then one dispatch for the remaining one to three.
      let trials = try Self.buffer(device, packed)
      let full = candidates.count / 4 * 4
      for (perThread, offset, groups) in [(4, 0, full / 4), (candidates.count - full, full, 1)]
      where perThread > 0 && groups > 0 {
        var step = rowStep
        var extra = angleSumExtra(first: offset)
        let compute = try encoder(commands, try angleSumPipeline(perThread))
        for (index, buffer) in [bandG, kBuffer, qBuffer, batchPartial].enumerated() {
          compute.setBuffer(buffer, offset: 0, index: index)
        }
        compute.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 4)
        compute.setBuffer(trials, offset: 0, index: 5)
        compute.setBytes(&batch, length: 4, index: 6)
        compute.setBuffer(probeGeometry, offset: 0, index: 7)
        compute.setBuffer(neighbourBuffer, offset: 0, index: 8)
        compute.setBytes(&step, length: 8, index: 9)
        compute.setBytes(&extra, length: MemoryLayout<AngleSumExtra>.stride, index: 10)
        compute.dispatchThreads(
          MTLSize(width: bandCount, height: Int(p.chunks), depth: groups),
          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        compute.endEncoding()
      }
      let reduction = try encoder(commands, reduceBatch)
      reduction.setBuffer(batchPartial, offset: 0, index: 0)
      reduction.setBuffer(batchPower, offset: first * bandCount * 16, index: 1)
      reduction.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 2)
      reduction.setBytes(&batch, length: 4, index: 3)
      dispatch(reduction, width: bandCount, height: Int(batch))
    }
    try finish(commands)
    lastAgreementGPUSeconds = commands.gpuEndTime - commands.gpuStartTime
    // The band power in double precision from the float32 BF sums, as the MPS band reduction.
    let sums = batchPower.contents().assumingMemoryBound(to: SIMD4<Float>.self)
    let weight = weights.contents().assumingMemoryBound(to: Float.self)
    var result: [Double] = []
    result.reserveCapacity(rows.count)
    for candidate in rows.indices {
      var total = 0.0
      for i in 0..<bandCount {
        let value = sums[candidate * bandCount + i]
        let denominator = Double(value.z)
        if denominator > 0 {
          let re = Double(value.x)
          let im = Double(value.y)
          total += Double(weight[i]) * (re * re + im * im) / denominator
        }
      }
      guard total.isFinite else { throw SSBOptimizationError.nonFiniteLoss }
      result.append(total)
    }
    return result
  }

  private struct AngleSumExtra {
    var inner2, outer2: Float
    var first: UInt32
  }

  /// The radii where the soft aperture of `joint_geometry_xy` is exactly 1 / 0 (worst-case anisotropic sampling),
  /// moved inward / outward by 1e-4 in r^2, far beyond the float32 rounding of the edge expression.
  private func angleSumExtra(first: Int) -> AngleSumExtra {
    let wl = Double(geometry.wavelengthAngstroms), semi = Double(geometry.semiangleRadians)
    let angular = Double(max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians))
    let inner = (semi - 0.5 * angular) / wl, outer = (semi + 0.5 * angular) / wl
    return AngleSumExtra(
      inner2: inner > 0 ? Float(inner * inner * (1 - 1e-4)) : -1,
      outer2: Float(outer * outer * (1 + 1e-4)), first: UInt32(first))
  }

  private func angleSumPipeline(_ perThread: Int) throws -> MTLComputePipelineState {
    if let existing = angleSumPipelines[perThread] { return existing }
    let constants = MTLFunctionConstantValues()
    var value = UInt32(perThread)
    constants.setConstantValue(&value, type: .uint, index: 50)
    let function = try library.makeFunction(
      name: "joint_agreement_angle_sums", constantValues: constants)
    let pipeline = try device.makeComputePipelineState(function: function)
    angleSumPipelines[perThread] = pipeline
    return pipeline
  }

  /// SSB (`phase_of_mean`) with all BF pixels: the corrected waves of every bright-field pixel summed before one
  /// phase, as Python's `SSB.reconstruct` and `SSB.preview`. Cancellation is checked between GPU batches.
  public func reconstruct(
    _ parameters: SSBJointParameters, upsamplingFactor: Int = 1, isCancelled: () -> Bool = { false }
  ) throws -> MetalSSBJointPhase {
    try reconstruct(SSBModelPoint(parameters), upsamplingFactor: upsamplingFactor, isCancelled: isCancelled)
  }

  /// SSB with Auto or an explicit output sampling, recorded in the result's `outputSampling` so a manual 8x is
  /// never mistaken for an automatic choice.
  public func reconstruct(
    _ point: SSBModelPoint, outputSampling: SSBOutputSampling, calibration: MetalSSBCalibration,
    isCancelled: () -> Bool = { false }
  ) throws -> MetalSSBJointPhase {
    var result = try reconstruct(
      point, upsamplingFactor: outputSampling.resolve(calibration), isCancelled: isCancelled)
    result.outputSampling = outputSampling
    return result
  }

  /// SSB of a double-precision model point (a fit's result) with all BF pixels.
  public func reconstruct(
    _ point: SSBModelPoint, upsamplingFactor: Int = 1, isCancelled: () -> Bool = { false }
  ) throws -> MetalSSBJointPhase {
    try validatePreparation()
    try point.validate()
    let parameters = point.jointParameters
    try parameters.validate()
    guard [1, 2, 3, 4, 8].contains(upsamplingFactor) else {
      throw MetalSSBError.invalidGeometry("Use an SSB sampling factor of 1, 2, 3, 4 or 8.")
    }
    if isCancelled() { throw CancellationError() }
    let started = Date()
    let (phase, n) = try coherent.reconstruct(
      point, factor: upsamplingFactor, visit: visit, isCancelled: isCancelled)
    return MetalSSBJointPhase(
      phase: phase, rows: n, columns: n, upsamplingFactor: upsamplingFactor,
      wallSeconds: Date().timeIntervalSince(started), parameters: parameters,
      scanDetectorRotationDegrees: scanDetectorRotationDegrees, provenance: provenance,
      point: point, kernel: coherent.lastKernel)
  }

  /// `.phase` / `.complex` of `reconstruct(_:upsamplingFactor:method:regularization:isCancelled:)`.
  func reconstructDirect(
    _ point: SSBModelPoint, upsamplingFactor: Int, complex: Bool, regularization: SSBDirectRegularization,
    isCancelled: () -> Bool
  ) throws -> MetalSSBJointPhase {
    try validatePreparation()
    let started = Date()
    if directSolver == nil {
      directSolver = try SSBDirectSolver(
        device: device, queue: queue, library: library, geometry: geometry, activeIndices: activeIndices)
    }
    let solver = directSolver!
    let (phase, amplitude, n) = try solver.reconstruct(
      point, factor: upsamplingFactor, complex: complex, regularization: regularization,
      rotation: scanDetectorRotationDegrees, k: kBuffer, visit: visit, isCancelled: isCancelled)
    var result = MetalSSBJointPhase(
      phase: phase, rows: n, columns: n, upsamplingFactor: upsamplingFactor,
      wallSeconds: Date().timeIntervalSince(started), parameters: point.jointParameters,
      scanDetectorRotationDegrees: scanDetectorRotationDegrees, provenance: provenance,
      point: point, amplitude: amplitude,
      kernel: MetalKernelIdentity(
        name: SSBKernelRegistry.direct.name, version: SSBKernelRegistry.direct.version,
        sourceSHA256: solver.sourceSHA256,
        variant: "\(complex ? "complex" : "phase") factor=\(upsamplingFactor) depth=\(point.depthSpreadAngstroms != 0 ? "on" : "off")"))
    result.method = complex ? .complex : .phase
    result.regularization = regularization
    return result
  }

  // MARK: - Drag previews of the regularized reconstructions

  /// Add up to `nodes` exact preparations to the drag window of `.phase` or `.complex` along `control` that covers
  /// `point` (starting a new one when none does), and return its state.
  ///
  /// A window is a Chebyshev series, in one control with every other input held, of the prepared normal
  /// equations (Python's CUDA `DragWindow`): it needs order + 1 exact preparations (29 for a magnitude, tilt or
  /// depth, 50 for an azimuth), each as costly as an optics change of `reconstruct`. Build one node per call
  /// between interactive frames, or several while idle. A complex window also serves `.phase`. Windows whose held
  /// inputs no longer match `point` are dropped; at most three tiles per control and a quarter of the device's
  /// recommended working set are kept.
  public func buildDragWindow(
    along control: SSBDragControl, at point: SSBModelPoint, upsamplingFactor: Int, method: SSBReconstructionMethod,
    regularization: SSBDirectRegularization = .init(), nodes: Int = 1, isCancelled: () -> Bool = { false }
  ) throws -> SSBDragWindowState {
    let solver = try directSolver(method: method, regularization: regularization, factor: upsamplingFactor, point)
    return try solver.buildWindow(
      along: control, at: point, factor: upsamplingFactor, complex: method == .complex,
      regularization: regularization, rotation: scanDetectorRotationDegrees, k: kBuffer, nodes: nodes, visit: visit,
      isCancelled: isCancelled)
  }

  /// The state of the drag window along `control` covering `point`, or nil when there is none.
  public func dragWindowState(
    along control: SSBDragControl, at point: SSBModelPoint, upsamplingFactor: Int, method: SSBReconstructionMethod,
    regularization: SSBDirectRegularization = .init()
  ) -> SSBDragWindowState? {
    guard method != .ssb else { return nil }
    return directSolver?.windowState(
      along: control, at: point, factor: upsamplingFactor, complex: method == .complex,
      regularization: regularization, rotation: scanDetectorRotationDegrees)
  }

  /// A drag preview of `.phase` or `.complex` at `point` from a complete window along `control`, or nil when no
  /// window covers it (then reconstruct exactly, and build the window).
  ///
  /// Inside its window a preview agrees with `reconstruct` at the same point to the float32 level (the series is
  /// truncated at 1e-10 of its terms' bound; coefficients are rounded to float32), but it is not the exact solve:
  /// the result's `preview` is set, and it must not be saved or exported. Settle on `reconstruct` when the drag ends.
  public func dragPreview(
    _ point: SSBModelPoint, along control: SSBDragControl, upsamplingFactor: Int, method: SSBReconstructionMethod,
    regularization: SSBDirectRegularization = .init()
  ) throws -> MetalSSBJointPhase? {
    guard method != .ssb, let solver = directSolver else { return nil }
    try regularization.validate(for: method)
    try point.validate()
    try validatePreparation()
    let started = Date()
    guard
      let (phase, amplitude, n, window) = try solver.preview(
        point, along: control, factor: upsamplingFactor, complex: method == .complex,
        regularization: regularization, rotation: scanDetectorRotationDegrees)
    else { return nil }
    var result = MetalSSBJointPhase(
      phase: phase, rows: n, columns: n, upsamplingFactor: upsamplingFactor,
      wallSeconds: Date().timeIntervalSince(started), parameters: point.jointParameters,
      scanDetectorRotationDegrees: scanDetectorRotationDegrees, provenance: provenance,
      point: point, amplitude: method == .complex ? amplitude : nil,
      kernel: MetalKernelIdentity(
        name: SSBKernelRegistry.direct.name, version: SSBKernelRegistry.direct.version,
        sourceSHA256: solver.sourceSHA256,
        variant: "\(method == .complex ? "complex" : "phase") factor=\(upsamplingFactor) drag-window order=\(window.order)"))
    result.method = method
    result.regularization = regularization
    result.preview = window
    return result
  }

  /// Release every drag window (their memory).
  public func releaseDragWindows() { directSolver?.releaseWindows() }

  private func directSolver(
    method: SSBReconstructionMethod, regularization: SSBDirectRegularization, factor: Int, _ point: SSBModelPoint
  ) throws -> SSBDirectSolver {
    guard method != .ssb else {
      throw MetalSSBError.invalidGeometry("Drag windows serve the regularized methods phase and complex.")
    }
    try regularization.validate(for: method)
    guard (1...4).contains(factor) else { throw SSBDirectError.unsupportedSampling(factor) }
    try point.validate()
    try validatePreparation()
    if directSolver == nil {
      directSolver = try SSBDirectSolver(
        device: device, queue: queue, library: library, geometry: geometry, activeIndices: activeIndices)
    }
    return directSolver!
  }

  /// Stage times of the latest `.phase` / `.complex` reconstruction (benchmarks).
  @_spi(SSBBenchmark) public var lastDirectTiming: SSBDirectTiming? { directSolver?.lastTiming }

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
