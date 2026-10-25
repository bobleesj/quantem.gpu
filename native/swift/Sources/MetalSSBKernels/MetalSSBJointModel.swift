import Foundation
import Metal
import MetalPerformanceShaders
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
    try SSBModelHigherOrder.validate((aberrations.higherOrder ?? []).map(SSBModelHigherOrder.init))
    guard packed.allSatisfy(\.isFinite), depthAngstroms >= 0
    else {
      throw MetalSSBError.invalidGeometry(
        "Joint SSB needs finite C10/C12, nonnegative depth and tilt in mrad."
      )
    }
  }
}

/// GPU phase in radians, using the selected SSB phase estimator.
/// Upsampling unfolds the measured Fourier aliases; it is not image interpolation.
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
  /// `complexWave` only: the object wave's amplitude (float32, rows x columns), |inverse FFT of the summed
  /// corrected spectrum| / BF count, as Python's `object_wave`. Nil for the phase estimators.
  public var amplitude: MTLBuffer? = nil
  /// The estimator that produced this result.
  public var estimator: SSBPhaseEstimator = .phaseOfMean
  /// How the output factor was chosen: `.auto` (from the calibration, at most 4x) or an explicit
  /// `.factor(n)` override. Nil when the caller passed a bare `upsamplingFactor`.
  public var outputSampling: SSBOutputSampling? = nil
  /// The registered GPU kernel (name, version, source fingerprint, variant) that produced this result.
  public var kernel: MetalKernelIdentity? = nil
}

/// How the agreement objective evaluates its phases.
public enum SSBAgreementKernel: String, Codable, Sendable {
  /// Angle sums advanced along detector rows (the Python MPS fit's kernel), the default.
  case angleSums = "angle_sums"
  /// Two precise sincos and two sines per (band frequency, BF pixel, candidate): the reference definition.
  case perTerm = "per_term"
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
  private var kBuffer: MTLBuffer
  private let activeIndices: [Int]
  private let library: MTLLibrary
  private let gather, correct, addPhase, normalize: MTLComputePipelineState
  private let fusedColumn, fusedRow: MTLComputePipelineState
  private let fusedTwiddle: MTLBuffer
  private(set) var supportsFused1536: Bool
  private let objectiveBatch, reduceBatch, objectiveAngleSums: MTLComputePipelineState
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
  /// The objective's phase evaluation; both give the same agreement to about 1e-6 relative.
  public var agreementKernel: SSBAgreementKernel = .angleSums
  private var neighbourBuffer: MTLBuffer
  private var rowStep = SIMD2<Float>(0, 0)
  private let makeProbeGeometry, trialProbe: MTLComputePipelineState
  private let probeGeometry, probeChi: MTLBuffer
  private var batchPartial: MTLBuffer?
  private var batchPower: MTLBuffer?
  private var coherent: SSBCoherentReconstruction
  private var fifthOrder: SSBFifthOrderAgreement?
  private var fifthOrderGradient: SSBFifthOrderGradient?

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
        "ssb_fft", "ssb_joint", "ssb_joint_fft", "ssb_model", "ssb_separable", "ssb_fifth_order",
        "ssb_fifth_order_gradient", "ssb_redraw",
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
    objectiveBatch = try pipeline("joint_objective_batch")
    reduceBatch = try pipeline("joint_power_batch")
    objectiveAngleSums = try pipeline("joint_objective_angle_sums")
    makeProbeGeometry = try pipeline("joint_probe_geometry")
    trialProbe = try pipeline("joint_trial_probe")
    fusedColumn = try pipeline("joint_ifft1536_half_columns")
    fusedRow = try pipeline("joint_ifft1536_half_rows_phase")
    let twiddles = (0..<2048).map { i -> SIMD2<Float> in
      let angle =
        i < 512
        ? -2 * Double.pi * Double(i) / 512
        : 2 * Double.pi * Double(i - 512) / 1536
      return SIMD2(Float(cos(angle)), Float(sin(angle)))
    }
    fusedTwiddle = try Self.buffer(device, twiddles)
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
    supportsFused1536 = Self.fused1536(geometry: geometry, k: ks)
    kBuffer = try Self.buffer(device, ks)
    let steps = Self.rowSteps(ks)
    neighbourBuffer = try Self.buffer(device, steps.neighbour)
    rowStep = steps.step
    coherent = try SSBCoherentReconstruction(
      device: device, queue: queue, library: library, geometry: geometry, kBuffer: kBuffer, k: ks)
    probeGeometry = try Self.allocate(device, activeIndices.count * 16)
    probeChi = try Self.allocate(device, activeIndices.count * 8 * 4)
    qBuffer = try Self.buffer(device, qs)
    indexBuffer = try Self.buffer(device, indices)
    weights = try Self.buffer(device, ws)
    bandG = try Self.allocate(device, activeIndices.count * bandCount * 8)
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

  /// The real-field 1536 path needs no Nyquist correction when the calibrated aperture support ends
  /// strictly before both unfolded-grid boundaries; a margin covers float32 coordinate roundoff.
  static func fused1536(geometry: MetalSSBGeometry, k ks: [SIMD2<Float>]) -> Bool {
    let radius =
      (geometry.semiangleRadians
        + 0.5 * max(geometry.angularSamplingYRadians, geometry.angularSamplingXRadians))
      / geometry.wavelengthAngstroms
    let rowSupport = ((ks.map { abs($0.x) }.max() ?? 0) + radius) * 1.00001
    let columnSupport = ((ks.map { abs($0.y) }.max() ?? 0) + radius) * 1.00001
    return geometry.qxByRow.count == 512
      && 768 * abs(geometry.qxByRow[1]) > rowSupport
      && 768 * abs(geometry.qyByColumn[1]) > columnSupport
      && geometry.dcValue.x > 0
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
    supportsFused1536 = Self.fused1536(geometry: geometry, k: ks)
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
    if fifthOrder == nil {
      fifthOrder = try SSBFifthOrderAgreement(device: device, queue: queue, library: library)
    }
    let rows = zip(points, coefficients).map { point, packed in
      [
        Float(point.depthSpreadAngstroms), Float(point.sampleTiltMrad[0] * 1e-3),
        Float(point.sampleTiltMrad[1] * 1e-3),
      ] + packed
    }
    return try fifthOrder!.evaluate(
      rows: rows, geometry: geometry, k: kBuffer, q: qBuffer, band: bandG, weights: weights,
      probeGeometry: probeGeometry, count: activeIndices.count, bandCount: bandCount,
      isCancelled: isCancelled)
  }

  /// The fifth-order agreement of each candidate (bit-identical to `fifthOrderAgreements`) and its exact gradient
  /// with respect to the objective row [depth (A), tilt row (rad), tilt column (rad), 28 packed coefficients (A)].
  /// `agreement64` sums the same float32 per-frequency powers in Double (finer than the float32 total).
  /// Per-term arithmetic is float32 on Metal; the gradient partials are reduced in Double.
  /// A `qStride` above 1 returns the agreement over every qStride-th band frequency and its gradient, both scaled
  /// by the ratio of the full to the subsampled band weight: a cheap steering model, never the objective.
  /// `gradients: false` returns the agreements alone (empty gradients) and keeps their projections, so
  /// `fifthOrderRetainedGradients` of the same candidates adds only the gradient pass.
  /// A `kStride` above 1 uses every kStride-th BF pixel only (unscaled): a steering model, never the objective.
  func fifthOrderAgreementsAndGradients(
    _ points: [SSBModelPoint], packedCoefficients: [[Float]], qStride: Int = 1, kStride: Int = 1,
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
    if fifthOrderGradient == nil {
      fifthOrderGradient = try SSBFifthOrderGradient(device: device, queue: queue, library: library)
    }
    let rows = zip(points, packedCoefficients).map { point, packed in
      [
        Float(point.depthSpreadAngstroms), Float(point.sampleTiltMrad[0] * 1e-3),
        Float(point.sampleTiltMrad[1] * 1e-3),
      ] + packed
    }
    let ratio = bandWeightRatio(qStride: qStride)
    return try fifthOrderGradient!.evaluate(
      rows: rows, geometry: geometry, k: kBuffer, q: qBuffer, band: bandG, weights: weights,
      probeGeometry: probeGeometry, count: activeIndices.count, bandCount: bandCount,
      qStride: qStride, kStride: kStride, gradients: gradients, isCancelled: isCancelled
    ).map { ($0.agreement * ratio, $0.agreement64 * ratio, $0.gradient.map { $0 * ratio }) }
  }

  /// The agreements and gradients of the candidates of the last value-only
  /// `fifthOrderAgreementsAndGradients(gradients: false)`, reusing its objective pass; other candidates are
  /// evaluated in full.
  func fifthOrderRetainedGradients(
    _ points: [SSBModelPoint], packedCoefficients: [[Float]], qStride: Int = 1, kStride: Int = 1,
    isCancelled: () -> Bool = { false }
  ) throws -> [(agreement: Double, agreement64: Double, gradient: [Double])] {
    try validatePreparation()
    guard let fifthOrderGradient else {
      return try fifthOrderAgreementsAndGradients(
        points, packedCoefficients: packedCoefficients, qStride: qStride, kStride: kStride,
        isCancelled: isCancelled)
    }
    let rows = zip(points, packedCoefficients).map { point, packed in
      [
        Float(point.depthSpreadAngstroms), Float(point.sampleTiltMrad[0] * 1e-3),
        Float(point.sampleTiltMrad[1] * 1e-3),
      ] + packed
    }
    let ratio = bandWeightRatio(qStride: qStride)
    return try fifthOrderGradient.retainedGradients(
      rows: rows, geometry: geometry, k: kBuffer, q: qBuffer, band: bandG, weights: weights,
      probeGeometry: probeGeometry, count: activeIndices.count, bandCount: bandCount,
      qStride: qStride, kStride: kStride, isCancelled: isCancelled
    ).map { ($0.agreement * ratio, $0.agreement64 * ratio, $0.gradient.map { $0 * ratio }) }
  }

  /// The full band weight over the weight of every qStride-th band frequency.
  private func bandWeightRatio(qStride: Int) -> Double {
    guard qStride > 1 else { return 1 }
    let w = UnsafeBufferPointer(
      start: weights.contents().assumingMemoryBound(to: Float.self), count: bandCount)
    let sub = stride(from: 0, to: bandCount, by: qStride).reduce(0.0) { $0 + Double(w[$1]) }
    return w.reduce(0.0) { $0 + Double($1) } / sub
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
      if agreementKernel == .angleSums {
        let trials = try Self.buffer(device, packed)
        var step = rowStep
        let compute = try encoder(commands, objectiveAngleSums)
        for (index, buffer) in [bandG, kBuffer, qBuffer, batchPartial].enumerated() {
          compute.setBuffer(buffer, offset: 0, index: index)
        }
        compute.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 4)
        compute.setBuffer(trials, offset: 0, index: 5)
        compute.setBytes(&batch, length: 4, index: 6)
        compute.setBuffer(probeGeometry, offset: 0, index: 7)
        compute.setBuffer(neighbourBuffer, offset: 0, index: 8)
        compute.setBytes(&step, length: 8, index: 9)
        compute.dispatchThreads(
          MTLSize(width: bandCount, height: Int(p.chunks), depth: (candidates.count + 3) / 4),
          threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1))
        compute.endEncoding()
      } else {
        try encodeProbe(commands, parameters: packed, batch: batch)
        let compute = try encoder(commands, objectiveBatch)
        for (index, buffer) in [bandG, kBuffer, qBuffer, batchPartial].enumerated() {
          compute.setBuffer(buffer, offset: 0, index: index)
        }
        compute.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 4)
        compute.setBytes(packed, length: packed.count * 4, index: 5)
        compute.setBytes(&batch, length: 4, index: 6)
        compute.setBuffer(probeGeometry, offset: 0, index: 7)
        compute.setBuffer(probeChi, offset: 0, index: 8)
        dispatch(compute, width: bandCount, height: Int(p.chunks))
      }
      let reduction = try encoder(commands, reduceBatch)
      reduction.setBuffer(batchPartial, offset: 0, index: 0)
      reduction.setBuffer(batchPower, offset: first * bandCount * 16, index: 1)
      reduction.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 2)
      reduction.setBytes(&batch, length: 4, index: 3)
      dispatch(reduction, width: bandCount, height: Int(batch))
    }
    try finish(commands)
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

  private func encodeProbe(_ commands: MTLCommandBuffer, parameters: [Float], batch: UInt32) throws
  {
    var p = grid
    var count = batch
    let compute = try encoder(commands, trialProbe)
    compute.setBuffer(probeGeometry, offset: 0, index: 0)
    compute.setBuffer(probeChi, offset: 0, index: 1)
    compute.setBytes(parameters, length: parameters.count * 4, index: 2)
    compute.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 3)
    compute.setBytes(&count, length: 4, index: 4)
    dispatch(compute, width: Int(p.totalBF))
  }

  /// Reconstruct with all BF pixels. Cancellation is checked between GPU batches.
  ///
  /// `phaseOfMean` (the default) sums the corrected waves before one phase, as Python's `SSB.reconstruct`
  /// and `SSB.preview`; `meanPhase` averages the per-pixel phases; `complexWave` returns the summed wave's
  /// phase and amplitude (1x, C10/C12, no tilt or depth spread).
  public func reconstruct(
    _ parameters: SSBJointParameters, upsamplingFactor: Int = 1,
    estimator: SSBPhaseEstimator = .phaseOfMean,
    isCancelled: () -> Bool = { false }
  ) throws -> MetalSSBJointPhase {
    try reconstruct(
      SSBModelPoint(parameters), upsamplingFactor: upsamplingFactor, estimator: estimator,
      isCancelled: isCancelled)
  }

  /// Reconstruct with Auto or an explicit output sampling, recorded in the result's `outputSampling` so a
  /// manual 8x is never mistaken for an automatic choice.
  public func reconstruct(
    _ point: SSBModelPoint, outputSampling: SSBOutputSampling, calibration: MetalSSBCalibration,
    estimator: SSBPhaseEstimator = .phaseOfMean,
    isCancelled: () -> Bool = { false }
  ) throws -> MetalSSBJointPhase {
    var result = try reconstruct(
      point, upsamplingFactor: outputSampling.resolve(calibration), estimator: estimator,
      isCancelled: isCancelled)
    result.outputSampling = outputSampling
    return result
  }

  /// Reconstruct a double-precision model point (a fit's result) with all BF pixels.
  public func reconstruct(
    _ point: SSBModelPoint, upsamplingFactor: Int = 1,
    estimator: SSBPhaseEstimator = .phaseOfMean,
    isCancelled: () -> Bool = { false }
  ) throws -> MetalSSBJointPhase {
    try validatePreparation()
    try point.validate()
    let parameters = point.jointParameters
    try parameters.validate()
    guard [1, 2, 3, 4, 8].contains(upsamplingFactor) else {
      throw MetalSSBError.invalidGeometry("Use an SSB sampling factor of 1, 2, 3, 4 or 8.")
    }
    if isCancelled() { throw CancellationError() }
    if estimator == .complexWave {
      guard !point.hasHigherOrder else {
        throw MetalSSBError.invalidGeometry(
          "Higher-order reconstruction supports phase_of_mean or mean_phase; choose either phase estimator.")
      }
      guard upsamplingFactor == 1, point.depthSpreadAngstroms == 0,
        point.sampleTiltMrad.allSatisfy({ $0 == 0 })
      else {
        throw MetalSSBError.invalidGeometry(
          "complex_wave requires upsampling_factor=1 and no sample tilt/depth; choose a phase estimator instead.")
      }
    }
    if estimator == .phaseOfMean || estimator == .complexWave {
      let started = Date()
      let (phase, n, amplitude) = try coherent.reconstruct(
        point, factor: upsamplingFactor, amplitude: estimator == .complexWave, visit: visit,
        isCancelled: isCancelled)
      return MetalSSBJointPhase(
        phase: phase, rows: n, columns: n, upsamplingFactor: upsamplingFactor,
        wallSeconds: Date().timeIntervalSince(started), parameters: parameters,
        scanDetectorRotationDegrees: scanDetectorRotationDegrees, provenance: provenance,
        point: point, amplitude: amplitude, estimator: estimator, kernel: coherent.lastKernel)
    }
    guard !point.hasHigherOrder else {
      throw MetalSSBError.invalidGeometry(
        "Higher-order joint SSB uses phaseOfMean; choose that estimator to retain every fitted term."
      )
    }
    let meanPhaseKernel = MetalKernelIdentity(
      name: SSBKernelRegistry.meanPhase.name, version: SSBKernelRegistry.meanPhase.version,
      sourceSHA256: try SSBKernelRegistry.sourceSHA256(SSBKernelRegistry.meanPhase),
      variant: upsamplingFactor == 3 && supportsFused1536 ? "factor=3 fused1536" : "factor=\(upsamplingFactor) general")
    if upsamplingFactor == 3 && supportsFused1536 {
      var result = try reconstructFused1536(parameters, isCancelled: isCancelled)
      result.point = point
      result.estimator = .meanPhase
      result.kernel = meanPhaseKernel
      return result
    }
    var result = try reconstructGeneral(
      parameters, upsamplingFactor: upsamplingFactor,
      isCancelled: isCancelled)
    result.point = point
    result.estimator = .meanPhase
    result.kernel = meanPhaseKernel
    return result
  }

  // General complex FFT path retained for other sizes and nonzero boundary
  // support. It is also the independent GPU control for the specialized path.
  func reconstructGeneral(
    _ parameters: SSBJointParameters, upsamplingFactor: Int,
    isCancelled: () -> Bool = { false }
  ) throws -> MetalSSBJointPhase {
    let started = Date()
    let n = Int(grid.n) * upsamplingFactor
    let plane = n * n
    let phase = try Self.allocate(device, plane * 4, shared: true)
    let batchSize = 8
    let corrected = try Self.allocate(device, batchSize * plane * 8)
    let object = try Self.allocate(device, batchSize * plane * 8)
    let clear = try command()
    guard let blit = clear.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
    blit.fill(buffer: phase, range: 0..<phase.length, value: 0)
    blit.endEncoding()
    try finish(clear)
    let t = parameters.packed
    var plans: [Int: (MPSGraph, MPSGraphTensor, MPSGraphTensor)] = [:]
    let execution = MPSGraphExecutionDescriptor()
    execution.waitUntilCompleted = false
    try withoutActuallyEscaping(isCancelled) { cancelled in
      try visit(batchSize) { [self] source, offset, half, bfOffset, count in
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
        let scheduled = MPSCommandBuffer(commandBuffer: commands)
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
        graph.encode(
          to: scheduled,
          feeds: [input: MPSGraphTensorData(corrected, shape: shape, dataType: .complexFloat32)],
          targetOperations: nil,
          resultsDictionary: [
            output: MPSGraphTensorData(object, shape: shape, dataType: .complexFloat32)
          ], executionDescriptor: execution)
        let sum = try self.encoder(scheduled, addPhase)
        sum.setBuffer(object, offset: 0, index: 0)
        sum.setBuffer(phase, offset: 0, index: 1)
        sum.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 2)
        dispatch(sum, width: plane)
        try finish(scheduled)
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

  private func reconstructFused1536(
    _ parameters: SSBJointParameters, isCancelled: () -> Bool
  ) throws -> MetalSSBJointPhase {
    let started = Date()
    let n = 1536
    let plane = 1536 * 1536
    let batchSize = 16
    let phase = try Self.allocate(device, plane * 4, shared: true)
    let intermediate = try Self.allocate(device, batchSize * n * (n / 2 + 1) * 8)
    let clear = try command()
    guard let blit = clear.makeBlitCommandEncoder() else { throw MetalSSBError.commandQueue }
    blit.fill(buffer: phase, range: 0..<phase.length, value: 0)
    blit.endEncoding()
    try finish(clear)
    let t = parameters.packed
    try withoutActuallyEscaping(isCancelled) { cancelled in
      try visit(batchSize) { [self] source, offset, half, bfOffset, count in
        if cancelled() { throw CancellationError() }
        var p = grid
        p.outputN = UInt32(n)
        p.sourceHalf = half ? 1 : 0
        p.count = UInt32(count)
        p.bfOffset = UInt32(bfOffset)
        let commands = try command()
        let column = try encoder(commands, fusedColumn)
        column.setBuffer(source, offset: offset, index: 0)
        column.setBuffer(kBuffer, offset: 0, index: 1)
        column.setBuffer(intermediate, offset: 0, index: 2)
        column.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 3)
        column.setBytes(t, length: t.count * 4, index: 4)
        column.setBuffer(fusedTwiddle, offset: 0, index: 5)
        column.setBuffer(probeGeometry, offset: 0, index: 6)
        column.dispatchThreadgroups(
          MTLSize(width: n / 2 + 1, height: count, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1))
        column.endEncoding()
        let row = try encoder(commands, fusedRow)
        row.setBuffer(intermediate, offset: 0, index: 0)
        row.setBuffer(phase, offset: 0, index: 1)
        row.setBuffer(fusedTwiddle, offset: 0, index: 2)
        row.setBytes(&p, length: MemoryLayout<Grid>.stride, index: 3)
        row.setBuffer(kBuffer, offset: 0, index: 4)
        row.dispatchThreadgroups(
          MTLSize(width: n, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 192, height: 1, depth: 1))
        row.endEncoding()
        // This also protects the engine's reusable streamed-spectrum buffer.
        // The completed phase stays owned by this result across later calls.
        try finish(commands)
      }
    }
    let final = try command()
    let norm = try encoder(final, normalize)
    var count = UInt32(plane)
    var divisor = Float(geometry.logicalBrightfieldCount)
    norm.setBuffer(phase, offset: 0, index: 0)
    norm.setBytes(&count, length: 4, index: 1)
    norm.setBytes(&divisor, length: 4, index: 2)
    dispatch(norm, width: plane)
    try finish(final)
    return MetalSSBJointPhase(
      phase: phase, rows: n, columns: n, upsamplingFactor: 3,
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
