import Foundation
import Metal

/// Aberrations used by the native single-sideband reconstruction.
public struct MetalSSBAberrations: Equatable, Sendable {
  public var c10Nanometers: Float
  public var c12Nanometers: Float
  public var phi12Radians: Float

  public init(
    c10Nanometers: Float,
    c12Nanometers: Float,
    phi12Radians: Float
  ) {
    self.c10Nanometers = c10Nanometers
    self.c12Nanometers = c12Nanometers
    self.phi12Radians = phi12Radians
  }
}

/// Calibrated reciprocal-space geometry for a native 512 by 512 SSB session.
///
/// `qxByRow` is indexed by scan Fourier row and `qyByColumn` by scan Fourier
/// column. Bright-field arrays remain in their logical source order. The
/// engine may skip only entries whose aperture is mathematically zero; the
/// logical count is always retained for normalization.
public struct MetalSSBGeometry: Sendable {
  public let brightfieldKX: [Float]
  public let brightfieldKY: [Float]
  public let brightfieldAlphaSquared: [Float]
  public let brightfieldAperture: [Float]
  public let brightfieldCos2Phi: [Float]
  public let brightfieldSin2Phi: [Float]
  public let qxByRow: [Float]
  public let qyByColumn: [Float]
  public let wavelengthAngstroms: Float
  public let semiangleRadians: Float
  public let angularSamplingYRadians: Float
  public let angularSamplingXRadians: Float
  public let dcValue: SIMD2<Float>
  public let referenceRotationDegrees: Float

  public var logicalBrightfieldCount: Int { brightfieldKX.count }

  public init(
    brightfieldKX: [Float],
    brightfieldKY: [Float],
    brightfieldAlphaSquared: [Float],
    brightfieldAperture: [Float],
    brightfieldCos2Phi: [Float],
    brightfieldSin2Phi: [Float],
    qxByRow: [Float],
    qyByColumn: [Float],
    wavelengthAngstroms: Float,
    semiangleRadians: Float,
    angularSamplingYRadians: Float,
    angularSamplingXRadians: Float,
    dcValue: SIMD2<Float>,
    referenceRotationDegrees: Float
  ) {
    self.brightfieldKX = brightfieldKX
    self.brightfieldKY = brightfieldKY
    self.brightfieldAlphaSquared = brightfieldAlphaSquared
    self.brightfieldAperture = brightfieldAperture
    self.brightfieldCos2Phi = brightfieldCos2Phi
    self.brightfieldSin2Phi = brightfieldSin2Phi
    self.qxByRow = qxByRow
    self.qyByColumn = qyByColumn
    self.wavelengthAngstroms = wavelengthAngstroms
    self.semiangleRadians = semiangleRadians
    self.angularSamplingYRadians = angularSamplingYRadians
    self.angularSamplingXRadians = angularSamplingXRadians
    self.dcValue = dcValue
    self.referenceRotationDegrees = referenceRotationDegrees
  }
}

/// Scientific provenance attached to every native Metal SSB result.
public struct MetalSSBProvenance: Equatable, Sendable {
  public let scanRows: Int
  public let scanColumns: Int
  public let sourceDType: String
  public let computeDType: String
  public let logicalBrightfieldCount: Int
  public let executedBrightfieldCount: Int
  public let zeroApertureBrightfieldCount: Int
  public let cachedBrightfieldCount: Int
  public let streamedBrightfieldCount: Int
  public let cacheBytes: Int
  public let scanBin: Int
  public let scanCrop: String
  public let brightfieldSelection: String
}

/// One exact native Metal SSB reconstruction.
///
/// Both buffers contain row-major `complex64` values with shape 512 by 512.
/// They are owned by this result and remain valid across later engine calls.
public struct MetalSSBResult: @unchecked Sendable {
  public let object: MTLBuffer
  public let fourierSum: MTLBuffer
  public let wallSeconds: Double
  public let gpuSeconds: Double
  public let provenance: MetalSSBProvenance
}

/// Exact full-objective phase-variance measurement.
public struct MetalSSBPhaseVarianceResult: Equatable, Sendable {
  public let loss: Float
  public let wallSeconds: Double
  public let gpuSeconds: Double
  public let provenance: MetalSSBProvenance
}

/// Errors raised by the native Metal SSB engine.
public enum MetalSSBError: LocalizedError {
  case missingResource(String)
  case missingFunction(String)
  case libraryCompilation(String)
  case commandQueue
  case invalidGeometry(String)
  case inputBufferTooSmall(required: Int, actual: Int)
  case allocation(String)
  case notPrepared
  case tooManyCacheChunks(required: Int, supported: Int)
  case commandExecution(String)

  public var errorDescription: String? {
    switch self {
    case .missingResource(let name):
      "MetalSSBKernels is missing \(name)."
    case .missingFunction(let name):
      "MetalSSBKernels is missing the \(name) function."
    case .libraryCompilation(let message):
      "Metal SSB kernel compilation failed: \(message)"
    case .commandQueue:
      "Metal could not create the SSB command queue."
    case .invalidGeometry(let message):
      "The SSB geometry is invalid: \(message)"
    case .inputBufferTooSmall(let required, let actual):
      "The SSB bright-field input needs \(required) bytes, but the buffer contains \(actual)."
    case .allocation(let purpose):
      "Metal could not allocate the SSB \(purpose)."
    case .notPrepared:
      "Prepare the SSB bright-field input before reconstructing or fitting."
    case .tooManyCacheChunks(let required, let supported):
      "The SSB cache needs \(required) chunks, but this kernel supports \(supported)."
    case .commandExecution(let message):
      "The Metal SSB command failed: \(message)"
    }
  }
}

/// GPU-resident native 512 by 512 single-sideband reconstruction and fitting.
///
/// The engine consumes lossless plane-major `uint8` bright-field columns. It
/// never crops or bins scan positions. It keeps every logical bright-field
/// term in the normalization and skips only terms proven to have zero aperture.
public final class MetalSSBEngine {
  private static let size = 512
  private static let plane = size * size
  private static let halfColumns = size / 2 + 1
  private static let halfPlane = size * halfColumns
  private static let batchCapacity = 32
  private static let cacheChunkCapacity = 512
  private static let maximumCacheChunks = 12
  private static let fftThreads = 64

  private struct FFTParams {
    var n: UInt32
    var log2n: UInt32
    var batch: UInt32
    var inverse: UInt32
  }

  private struct SSBParams {
    var n: UInt32
    var batch: UInt32
    var logicalBF: UInt32
    var bfOffset: UInt32
    var wavelength: Float
    var semiangle: Float
    var angularY: Float
    var angularX: Float
    var c10: Float
    var c12: Float
    var cos2Phi12: Float
    var sin2Phi12: Float
    var factor: Float
    var dcReal: Float
    var dcImaginary: Float
    var apertureInnerK2: Float
    var apertureOuterK2: Float
  }

  private struct HalfExtractParams {
    var sourceN: UInt32
    var batch: UInt32
  }

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let geometry: MetalSSBGeometry
  private let cacheBudgetBytes: Int?
  private let activeBrightfieldIndices: [Int]

  private let convertPipeline: MTLComputePipelineState
  private let fftPipeline: MTLComputePipelineState
  private let transposePipeline: MTLComputePipelineState
  private let fullAccumulatePipeline: MTLComputePipelineState
  private let halfAccumulatePipeline: MTLComputePipelineState
  private let finalizePipeline: MTLComputePipelineState
  private let halfExtractPipeline: MTLComputePipelineState
  private let halfLossCorrectionPipeline: MTLComputePipelineState
  private let fullLossCorrectionPipeline: MTLComputePipelineState
  private let phaseMomentPipeline: MTLComputePipelineState
  private let chiTrigPipeline: MTLComputePipelineState
  private let crossTrigPipeline: MTLComputePipelineState
  private let fusedLossRowPipeline: MTLComputePipelineState
  private let fusedLossMomentPipeline: MTLComputePipelineState
  private let halfToColumnMajorPipeline: MTLComputePipelineState
  private let halfToRowMajorPipeline: MTLComputePipelineState

  private let rawBuffer: MTLBuffer
  private let fftA: MTLBuffer
  private let fftB: MTLBuffer
  private let accumulator: MTLBuffer
  private let finalTemporary: MTLBuffer
  private let twiddleBuffer: MTLBuffer
  private let qxBuffer: MTLBuffer
  private let qyBuffer: MTLBuffer
  private let phaseSumBuffer: MTLBuffer
  private let phaseSumSquaredBuffer: MTLBuffer
  private var activeGeometryBuffer: MTLBuffer
  private var activeTrigBuffer: MTLBuffer
  private var activeGeometryRotation: Float?

  private var sourceBrightfield: MTLBuffer?
  private var cacheBuffers: [MTLBuffer] = []
  private var cacheCounts: [Int] = []
  private var cachedBrightfieldCount = 0
  private var cacheIsColumnMajor = false
  private var chiTrigBuffer: MTLBuffer?
  private var crossTrigBuffer: MTLBuffer?
  private var prepared = false

  /// Create a reusable SSB engine.
  ///
  /// - Parameters:
  ///   - device: Apple Metal device that owns all source and result buffers.
  ///   - geometry: Exact full-BF calibration in logical source order.
  ///   - cacheBudgetBytes: Maximum bytes retained for the Hermitian `G(k)`
  ///     cache. `nil` requests a complete cache. A smaller budget preserves
  ///     exactness by streaming the uncached tail from the source buffer.
  public init(
    device: MTLDevice,
    geometry: MetalSSBGeometry,
    cacheBudgetBytes: Int? = nil
  ) throws {
    try Self.validate(geometry)
    if let cacheBudgetBytes, cacheBudgetBytes < 0 {
      throw MetalSSBError.invalidGeometry(
        "cacheBudgetBytes must be nonnegative"
      )
    }
    self.device = device
    self.geometry = geometry
    self.cacheBudgetBytes = cacheBudgetBytes
    guard let queue = device.makeCommandQueue() else {
      throw MetalSSBError.commandQueue
    }
    self.queue = queue

    let activeIndices = Self.activeIndices(geometry)
    guard !activeIndices.isEmpty else {
      throw MetalSSBError.invalidGeometry(
        "no bright-field term has nonzero aperture"
      )
    }
    activeBrightfieldIndices = activeIndices
    let library = try Self.makeLibrary(device: device)
    convertPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "uint8_to_complex"
    )
    fftPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "fft_rows_radix8"
    )
    transposePipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "transpose_complex32"
    )
    fullAccumulatePipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_gamma_accumulate"
    )
    halfAccumulatePipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_gamma_accumulate_half"
    )
    finalizePipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_finalize_fourier_sum"
    )
    halfExtractPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "extract_hermitian_half"
    )
    halfLossCorrectionPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_correct_half_for_phase_loss"
    )
    fullLossCorrectionPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_correct_full_for_phase_loss"
    )
    phaseMomentPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_accumulate_phase_moments"
    )
    chiTrigPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_precompute_chi_trig512"
    )
    crossTrigPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_precompute_cross_trig512"
    )
    fusedLossRowPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_correct_half_column_ifft512_hermitian"
    )
    fusedLossMomentPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_ifft512_rows_hermitian_phase_moments"
    )
    halfToColumnMajorPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_transpose_half_to_column_major"
    )
    halfToRowMajorPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "ssb_transpose_half_to_row_major"
    )

    let rawBytes = Self.batchCapacity * Self.plane
    let complexBytes = rawBytes * MemoryLayout<SIMD2<Float>>.stride
    rawBuffer = try Self.allocate(
      device: device,
      length: rawBytes,
      options: .storageModePrivate,
      purpose: "uint8 batch"
    )
    fftA = try Self.allocate(
      device: device,
      length: complexBytes,
      options: .storageModePrivate,
      purpose: "FFT workspace A"
    )
    fftB = try Self.allocate(
      device: device,
      length: complexBytes,
      options: .storageModePrivate,
      purpose: "FFT workspace B"
    )
    let planeBytes = Self.plane * MemoryLayout<SIMD2<Float>>.stride
    accumulator = try Self.allocate(
      device: device,
      length: planeBytes,
      options: .storageModePrivate,
      purpose: "Fourier accumulator"
    )
    finalTemporary = try Self.allocate(
      device: device,
      length: planeBytes,
      options: .storageModePrivate,
      purpose: "inverse FFT workspace"
    )
    phaseSumBuffer = try Self.allocate(
      device: device,
      length: Self.plane * MemoryLayout<Float>.stride,
      options: .storageModeShared,
      purpose: "phase sum"
    )
    phaseSumSquaredBuffer = try Self.allocate(
      device: device,
      length: Self.plane * MemoryLayout<Float>.stride,
      options: .storageModeShared,
      purpose: "phase squared sum"
    )

    var twiddle: [SIMD2<Float>] = []
    twiddle.reserveCapacity(Self.size)
    for index in 0..<Self.size {
      let angle = -2 * Float.pi * Float(index) / Float(Self.size)
      twiddle.append(SIMD2<Float>(cos(angle), sin(angle)))
    }
    twiddleBuffer = try Self.makeBuffer(
      device: device,
      values: twiddle,
      purpose: "FFT twiddle table"
    )
    qxBuffer = try Self.makeBuffer(
      device: device,
      values: geometry.qxByRow,
      purpose: "scan-row reciprocal coordinates"
    )
    qyBuffer = try Self.makeBuffer(
      device: device,
      values: geometry.qyByColumn,
      purpose: "scan-column reciprocal coordinates"
    )

    let placeholderGeometry = [SIMD4<Float>](
      repeating: .zero,
      count: activeIndices.count
    )
    let placeholderTrig = [SIMD2<Float>](
      repeating: .zero,
      count: activeIndices.count
    )
    activeGeometryBuffer = try Self.makeBuffer(
      device: device,
      values: placeholderGeometry,
      purpose: "active BF geometry"
    )
    activeTrigBuffer = try Self.makeBuffer(
      device: device,
      values: placeholderTrig,
      purpose: "active BF angle table"
    )
    try rebuildGeometry(rotationDegrees: geometry.referenceRotationDegrees)
  }

  /// Number of exact nonzero-aperture terms executed by this session.
  public var executedBrightfieldCount: Int {
    activeBrightfieldIndices.count
  }

  /// Number of logical bright-field terms retained in normalization.
  public var logicalBrightfieldCount: Int {
    geometry.logicalBrightfieldCount
  }

  /// Prepare lossless plane-major `uint8` bright-field columns.
  ///
  /// The source shape is `[logicalBrightfieldCount, 512, 512]`. Preparation
  /// builds as much of the exact Hermitian `G(k)` cache as the configured
  /// budget admits and keeps the source buffer for any exact streamed tail.
  public func prepare(brightfield: MTLBuffer) throws {
    let requiredBytes = geometry.logicalBrightfieldCount * Self.plane
    guard brightfield.length >= requiredBytes else {
      throw MetalSSBError.inputBufferTooSmall(
        required: requiredBytes,
        actual: brightfield.length
      )
    }
    sourceBrightfield = brightfield
    cacheBuffers.removeAll(keepingCapacity: false)
    cacheCounts.removeAll(keepingCapacity: false)
    cachedBrightfieldCount = 0
    cacheIsColumnMajor = false
    chiTrigBuffer = nil
    crossTrigBuffer = nil
    prepared = false

    let bytesPerBrightfield =
      Self.halfPlane
      * MemoryLayout<SIMD2<Float>>.stride
    let activeCount = activeBrightfieldIndices.count
    let requestedCount: Int
    if let cacheBudgetBytes {
      let admitted = max(0, cacheBudgetBytes / bytesPerBrightfield)
      requestedCount = min(activeCount, admitted)
    } else {
      requestedCount = activeCount
    }
    let cachedCount =
      requestedCount == activeCount
      ? activeCount
      : (requestedCount / Self.batchCapacity) * Self.batchCapacity
    let chunks =
      cachedCount == 0
      ? 0
      : (cachedCount + Self.cacheChunkCapacity - 1)
        / Self.cacheChunkCapacity
    guard chunks <= Self.maximumCacheChunks else {
      throw MetalSSBError.tooManyCacheChunks(
        required: chunks,
        supported: Self.maximumCacheChunks
      )
    }
    if cachedCount > 0 {
      chiTrigBuffer = try Self.allocate(
        device: device,
        length: Self.plane * MemoryLayout<SIMD2<Float>>.stride,
        options: .storageModePrivate,
        purpose: "exact phase-loss chi table"
      )
      crossTrigBuffer = try Self.allocate(
        device: device,
        length: activeCount * 2 * Self.size
          * MemoryLayout<SIMD2<Float>>.stride,
        options: .storageModePrivate,
        purpose: "exact phase-loss cross table"
      )
    }

    let halfBytes = Self.halfPlane * MemoryLayout<SIMD2<Float>>.stride
    for chunk in 0..<chunks {
      let count = min(
        Self.cacheChunkCapacity,
        cachedCount - chunk * Self.cacheChunkCapacity
      )
      cacheBuffers.append(
        try Self.allocate(
          device: device,
          length: count * halfBytes,
          options: .storageModePrivate,
          purpose: "Hermitian G(k) cache chunk \(chunk)"
        ))
      cacheCounts.append(count)
    }

    for offset in stride(
      from: 0,
      to: cachedCount,
      by: Self.batchCapacity
    ) {
      let batch = min(Self.batchCapacity, cachedCount - offset)
      guard let commands = queue.makeCommandBuffer(),
        let blit = commands.makeBlitCommandEncoder()
      else { throw MetalSSBError.commandQueue }
      for local in 0..<batch {
        let logical = activeBrightfieldIndices[offset + local]
        blit.copy(
          from: brightfield,
          sourceOffset: logical * Self.plane,
          to: rawBuffer,
          destinationOffset: local * Self.plane,
          size: Self.plane
        )
      }
      blit.endEncoding()
      encodeForwardFFT(commands, batch: batch)

      let cacheIndex = offset / Self.cacheChunkCapacity
      let cacheOffset = offset - cacheIndex * Self.cacheChunkCapacity
      var extract = HalfExtractParams(
        sourceN: UInt32(Self.size),
        batch: UInt32(batch)
      )
      guard let encoder = commands.makeComputeCommandEncoder() else {
        throw MetalSSBError.commandQueue
      }
      encoder.setComputePipelineState(halfExtractPipeline)
      encoder.setBuffer(fftA, offset: 0, index: 0)
      encoder.setBuffer(
        cacheBuffers[cacheIndex],
        offset: cacheOffset * halfBytes,
        index: 1
      )
      encoder.setBytes(
        &extract,
        length: MemoryLayout<HalfExtractParams>.stride,
        index: 2
      )
      encoder.dispatchThreads(
        MTLSize(width: batch * Self.halfPlane, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
      )
      encoder.endEncoding()
      try commitAndWait(commands)
    }
    cachedBrightfieldCount = cachedCount
    prepared = true
  }

  /// Reconstruct one exact SSB object from the prepared full-BF session.
  public func reconstruct(
    aberrations: MetalSSBAberrations,
    rotationDegrees: Float? = nil
  ) throws -> MetalSSBResult {
    guard prepared, let sourceBrightfield else {
      throw MetalSSBError.notPrepared
    }
    try setCacheColumnMajor(false)
    let rotation = rotationDegrees ?? geometry.referenceRotationDegrees
    try rebuildGeometry(rotationDegrees: rotation)
    let started = Date()
    var gpuSeconds = 0.0

    guard let clearCommands = queue.makeCommandBuffer(),
      let clear = clearCommands.makeBlitCommandEncoder()
    else { throw MetalSSBError.commandQueue }
    clear.fill(buffer: accumulator, range: 0..<accumulator.length, value: 0)
    clear.endEncoding()
    if cachedBrightfieldCount > 0 {
      let cachedParams = parameters(
        batch: cachedBrightfieldCount,
        offset: 0,
        aberrations: aberrations
      )
      try encodeCachedAccumulator(
        clearCommands,
        sources: cacheBuffers,
        params: cachedParams
      )
    }
    try commitAndWait(clearCommands)
    gpuSeconds += gpuDuration(clearCommands)

    for offset in stride(
      from: cachedBrightfieldCount,
      to: activeBrightfieldIndices.count,
      by: Self.batchCapacity
    ) {
      let batch = min(
        Self.batchCapacity,
        activeBrightfieldIndices.count - offset
      )
      guard let commands = queue.makeCommandBuffer(),
        let blit = commands.makeBlitCommandEncoder()
      else { throw MetalSSBError.commandQueue }
      for local in 0..<batch {
        let logical = activeBrightfieldIndices[offset + local]
        blit.copy(
          from: sourceBrightfield,
          sourceOffset: logical * Self.plane,
          to: rawBuffer,
          destinationOffset: local * Self.plane,
          size: Self.plane
        )
      }
      blit.endEncoding()
      encodeForwardFFT(commands, batch: batch)
      encodeFullAccumulator(
        commands,
        source: fftA,
        params: parameters(
          batch: batch,
          offset: offset,
          aberrations: aberrations
        )
      )
      try commitAndWait(commands)
      gpuSeconds += gpuDuration(commands)
    }

    let resultBytes = Self.plane * MemoryLayout<SIMD2<Float>>.stride
    let object = try Self.allocate(
      device: device,
      length: resultBytes,
      options: .storageModeShared,
      purpose: "object result"
    )
    let fourier = try Self.allocate(
      device: device,
      length: resultBytes,
      options: .storageModeShared,
      purpose: "Fourier result"
    )
    guard let finalCommands = queue.makeCommandBuffer() else {
      throw MetalSSBError.commandQueue
    }
    encodeFinalize(
      finalCommands,
      params: parameters(batch: 1, offset: 0, aberrations: aberrations)
    )
    guard let preserve = finalCommands.makeBlitCommandEncoder() else {
      throw MetalSSBError.commandQueue
    }
    preserve.copy(
      from: accumulator,
      sourceOffset: 0,
      to: fourier,
      destinationOffset: 0,
      size: resultBytes
    )
    preserve.endEncoding()
    encodeFFT(
      finalCommands,
      input: accumulator,
      temporary: finalTemporary,
      output: object,
      params: FFTParams(
        n: UInt32(Self.size),
        log2n: 9,
        batch: 1,
        inverse: 1
      )
    )
    try commitAndWait(finalCommands)
    gpuSeconds += gpuDuration(finalCommands)
    return MetalSSBResult(
      object: object,
      fourierSum: fourier,
      wallSeconds: Date().timeIntervalSince(started),
      gpuSeconds: gpuSeconds,
      provenance: currentProvenance()
    )
  }

  /// Evaluate the exact full-BF native phase-variance objective.
  public func phaseVariance(
    aberrations: MetalSSBAberrations,
    rotationDegrees: Float? = nil
  ) throws -> MetalSSBPhaseVarianceResult {
    guard prepared, let sourceBrightfield else {
      throw MetalSSBError.notPrepared
    }
    try setCacheColumnMajor(true)
    let rotation = rotationDegrees ?? geometry.referenceRotationDegrees
    try rebuildGeometry(rotationDegrees: rotation)
    let started = Date()
    var gpuSeconds = 0.0

    guard let clearCommands = queue.makeCommandBuffer(),
      let clear = clearCommands.makeBlitCommandEncoder()
    else { throw MetalSSBError.commandQueue }
    clear.fill(
      buffer: phaseSumBuffer,
      range: 0..<phaseSumBuffer.length,
      value: 0
    )
    clear.fill(
      buffer: phaseSumSquaredBuffer,
      range: 0..<phaseSumSquaredBuffer.length,
      value: 0
    )
    clear.endEncoding()
    if cachedBrightfieldCount > 0 {
      guard let chiTrigBuffer, let crossTrigBuffer else {
        throw MetalSSBError.allocation("exact phase-loss tables")
      }
      let baseParams = parameters(
        batch: activeBrightfieldIndices.count,
        offset: 0,
        aberrations: aberrations
      )
      encodeChiTrig(
        clearCommands,
        params: baseParams,
        output: chiTrigBuffer
      )
      encodeCrossTrig(
        clearCommands,
        params: baseParams,
        output: crossTrigBuffer
      )
    }
    try commitAndWait(clearCommands)
    gpuSeconds += gpuDuration(clearCommands)

    let halfBytes = Self.halfPlane * MemoryLayout<SIMD2<Float>>.stride
    var globalOffset = 0
    for (cache, cacheCount) in zip(cacheBuffers, cacheCounts) {
      guard let commands = queue.makeCommandBuffer() else {
        throw MetalSSBError.commandQueue
      }
      for localOffset in stride(
        from: 0,
        to: cacheCount,
        by: Self.batchCapacity
      ) {
        let batch = min(Self.batchCapacity, cacheCount - localOffset)
        let params = parameters(
          batch: batch,
          offset: globalOffset + localOffset,
          aberrations: aberrations
        )
        try encodeFusedCachedPhaseLoss(
          commands,
          source: cache,
          sourceOffset: localOffset * halfBytes,
          batch: batch,
          params: params
        )
      }
      try commitAndWait(commands)
      gpuSeconds += gpuDuration(commands)
      globalOffset += cacheCount
    }

    for offset in stride(
      from: globalOffset,
      to: activeBrightfieldIndices.count,
      by: Self.batchCapacity
    ) {
      let batch = min(
        Self.batchCapacity,
        activeBrightfieldIndices.count - offset
      )
      guard let commands = queue.makeCommandBuffer(),
        let blit = commands.makeBlitCommandEncoder()
      else { throw MetalSSBError.commandQueue }
      for local in 0..<batch {
        let logical = activeBrightfieldIndices[offset + local]
        blit.copy(
          from: sourceBrightfield,
          sourceOffset: logical * Self.plane,
          to: rawBuffer,
          destinationOffset: local * Self.plane,
          size: Self.plane
        )
      }
      blit.endEncoding()
      encodeForwardFFT(commands, batch: batch)
      let params = parameters(
        batch: batch,
        offset: offset,
        aberrations: aberrations
      )
      encodeLossCorrection(
        commands,
        source: fftA,
        sourceOffset: 0,
        sourceIsHermitianHalf: false,
        params: params
      )
      encodeFFT(
        commands,
        input: fftB,
        temporary: fftA,
        output: fftB,
        params: FFTParams(
          n: UInt32(Self.size),
          log2n: 9,
          batch: UInt32(batch),
          inverse: 1
        )
      )
      encodePhaseMoments(commands, batch: batch)
      try commitAndWait(commands)
      gpuSeconds += gpuDuration(commands)
    }

    let sums = phaseSumBuffer.contents().bindMemory(
      to: Float.self,
      capacity: Self.plane
    )
    let squared = phaseSumSquaredBuffer.contents().bindMemory(
      to: Float.self,
      capacity: Self.plane
    )
    let logical = Double(geometry.logicalBrightfieldCount)
    let pixels = Double(Self.plane)
    var sumOfSquares = 0.0
    var squareOfMeans = 0.0
    for pixel in 0..<Self.plane {
      sumOfSquares += Double(squared[pixel])
      let mean = Double(sums[pixel]) / logical
      squareOfMeans += mean * mean
    }
    let loss = sumOfSquares / (logical * pixels) - squareOfMeans / pixels
    return MetalSSBPhaseVarianceResult(
      loss: Float(loss),
      wallSeconds: Date().timeIntervalSince(started),
      gpuSeconds: gpuSeconds,
      provenance: currentProvenance()
    )
  }

  /// Run the native deterministic 200-trial search and Nelder-Mead refinement.
  public func optimize(
    start: MetalSSBAberrations,
    rotationDegrees: Float? = nil,
    globalTrials: Int = 200,
    seed: UInt64 = 42,
    progress: (SSBOptimizationProgress) -> Void = { _ in },
    isCancelled: () -> Bool = { false }
  ) throws -> SSBOptimizationResult {
    let rotation = rotationDegrees ?? geometry.referenceRotationDegrees
    return try SSBOptimizer(globalTrials: globalTrials, seed: seed).run(
      start: SSBOptimizationPoint(
        c10Nanometers: Double(start.c10Nanometers),
        c12Nanometers: Double(start.c12Nanometers),
        phi12Radians: Double(start.phi12Radians)
      ),
      evaluate: { point in
        Double(
          try self.phaseVariance(
            aberrations: MetalSSBAberrations(
              c10Nanometers: Float(point.c10Nanometers),
              c12Nanometers: Float(point.c12Nanometers),
              phi12Radians: Float(point.phi12Radians)
            ),
            rotationDegrees: rotation
          ).loss)
      },
      progress: progress,
      isCancelled: isCancelled
    )
  }

  private func parameters(
    batch: Int,
    offset: Int,
    aberrations: MetalSSBAberrations
  ) -> SSBParams {
    let isotropicEdge = 0.5 * geometry.angularSamplingXRadians
    let innerK =
      max(0, geometry.semiangleRadians - isotropicEdge)
      / geometry.wavelengthAngstroms
    let outerK =
      (geometry.semiangleRadians + isotropicEdge)
      / geometry.wavelengthAngstroms
    return SSBParams(
      n: UInt32(Self.size),
      batch: UInt32(batch),
      logicalBF: UInt32(geometry.logicalBrightfieldCount),
      bfOffset: UInt32(offset),
      wavelength: geometry.wavelengthAngstroms,
      semiangle: geometry.semiangleRadians,
      angularY: geometry.angularSamplingYRadians,
      angularX: geometry.angularSamplingXRadians,
      c10: aberrations.c10Nanometers,
      c12: aberrations.c12Nanometers,
      cos2Phi12: cos(2 * aberrations.phi12Radians),
      sin2Phi12: sin(2 * aberrations.phi12Radians),
      factor: Float.pi / geometry.wavelengthAngstroms,
      dcReal: geometry.dcValue.x,
      dcImaginary: geometry.dcValue.y,
      apertureInnerK2: innerK * innerK,
      apertureOuterK2: outerK * outerK
    )
  }

  private func currentProvenance() -> MetalSSBProvenance {
    MetalSSBProvenance(
      scanRows: Self.size,
      scanColumns: Self.size,
      sourceDType: "uint8",
      computeDType: "float32/complex64",
      logicalBrightfieldCount: geometry.logicalBrightfieldCount,
      executedBrightfieldCount: activeBrightfieldIndices.count,
      zeroApertureBrightfieldCount:
        geometry.logicalBrightfieldCount - activeBrightfieldIndices.count,
      cachedBrightfieldCount: cachedBrightfieldCount,
      streamedBrightfieldCount:
        activeBrightfieldIndices.count - cachedBrightfieldCount,
      cacheBytes: cacheBuffers.reduce(0) { $0 + $1.length },
      scanBin: 1,
      scanCrop: "none",
      brightfieldSelection: "exact-zero-aperture-pruning"
    )
  }

  private func setCacheColumnMajor(_ columnMajor: Bool) throws {
    guard !cacheBuffers.isEmpty, cacheIsColumnMajor != columnMajor else {
      return
    }
    let pipeline =
      columnMajor
      ? halfToColumnMajorPipeline : halfToRowMajorPipeline
    let halfBytes = Self.halfPlane * MemoryLayout<SIMD2<Float>>.stride
    for (cache, cacheCount) in zip(cacheBuffers, cacheCounts) {
      guard let commands = queue.makeCommandBuffer() else {
        throw MetalSSBError.commandQueue
      }
      for offset in stride(
        from: 0,
        to: cacheCount,
        by: Self.batchCapacity
      ) {
        let batch = min(Self.batchCapacity, cacheCount - offset)
        var mutableBatch = UInt32(batch)
        guard let encoder = commands.makeComputeCommandEncoder() else {
          throw MetalSSBError.commandQueue
        }
        encoder.setComputePipelineState(pipeline)
        encoder.setBuffer(
          cache,
          offset: offset * halfBytes,
          index: 0
        )
        encoder.setBuffer(fftA, offset: 0, index: 1)
        encoder.setBytes(
          &mutableBatch,
          length: MemoryLayout<UInt32>.stride,
          index: 2
        )
        encoder.dispatchThreads(
          MTLSize(width: batch * Self.halfPlane, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.endEncoding()
        guard let blit = commands.makeBlitCommandEncoder() else {
          throw MetalSSBError.commandQueue
        }
        blit.copy(
          from: fftA,
          sourceOffset: 0,
          to: cache,
          destinationOffset: offset * halfBytes,
          size: batch * halfBytes
        )
        blit.endEncoding()
      }
      try commitAndWait(commands)
    }
    cacheIsColumnMajor = columnMajor
  }

  private func rebuildGeometry(rotationDegrees: Float) throws {
    if activeGeometryRotation?.bitPattern == rotationDegrees.bitPattern {
      return
    }
    var geometryValues: [SIMD4<Float>] = []
    var trigValues: [SIMD2<Float>] = []
    geometryValues.reserveCapacity(activeBrightfieldIndices.count)
    trigValues.reserveCapacity(activeBrightfieldIndices.count)
    let delta =
      (rotationDegrees - geometry.referenceRotationDegrees)
      * Float.pi / 180
    let cosine = cos(-delta)
    let sine = sin(-delta)
    for logical in activeBrightfieldIndices {
      if abs(delta) < 1e-7 {
        geometryValues.append(
          SIMD4<Float>(
            geometry.brightfieldKX[logical],
            geometry.brightfieldKY[logical],
            geometry.brightfieldAlphaSquared[logical],
            geometry.brightfieldAperture[logical]
          ))
        trigValues.append(
          SIMD2<Float>(
            geometry.brightfieldCos2Phi[logical],
            geometry.brightfieldSin2Phi[logical]
          ))
        continue
      }
      let baseX = geometry.brightfieldKX[logical]
      let baseY = geometry.brightfieldKY[logical]
      let x = baseX * cosine + baseY * sine
      let y = -baseX * sine + baseY * cosine
      let x2 = x * x
      let y2 = y * y
      let radiusSquared = x2 + y2
      let radius = sqrt(radiusSquared)
      let alpha = radius * geometry.wavelengthAngstroms
      let alphaSquared = alpha * alpha
      let inverseRadiusSquared: Float =
        radiusSquared > 1e-30
        ? 1 / radiusSquared : 0
      let cos2 = (x2 - y2) * inverseRadiusSquared
      let sin2 = 2 * x * y * inverseRadiusSquared
      let denominatorNumerator =
        pow(x * geometry.angularSamplingYRadians, 2)
        + pow(y * geometry.angularSamplingXRadians, 2)
      let denominator: Float =
        radius > 1e-15
        ? sqrt(denominatorNumerator) / radius : 0
      let edge: Float =
        denominator > 1e-15
        ? (geometry.semiangleRadians - alpha) / denominator + 0.5
        : 1
      geometryValues.append(
        SIMD4<Float>(
          x,
          y,
          alphaSquared,
          min(max(edge, 0), 1)
        ))
      trigValues.append(SIMD2<Float>(cos2, sin2))
    }
    activeGeometryBuffer = try Self.makeBuffer(
      device: device,
      values: geometryValues,
      purpose: "active BF geometry"
    )
    activeTrigBuffer = try Self.makeBuffer(
      device: device,
      values: trigValues,
      purpose: "active BF angle table"
    )
    activeGeometryRotation = rotationDegrees
  }

  private func encodeForwardFFT(
    _ commands: MTLCommandBuffer,
    batch: Int
  ) {
    var count = UInt32(batch * Self.plane)
    let convert = commands.makeComputeCommandEncoder()!
    convert.setComputePipelineState(convertPipeline)
    convert.setBuffer(rawBuffer, offset: 0, index: 0)
    convert.setBuffer(fftA, offset: 0, index: 1)
    convert.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)
    convert.dispatchThreads(
      MTLSize(width: Int(count), height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    convert.endEncoding()
    encodeFFT(
      commands,
      input: fftA,
      temporary: fftB,
      output: fftA,
      params: FFTParams(
        n: UInt32(Self.size),
        log2n: 9,
        batch: UInt32(batch),
        inverse: 0
      )
    )
  }

  private func encodeFFT(
    _ commands: MTLCommandBuffer,
    input: MTLBuffer,
    temporary: MTLBuffer,
    output: MTLBuffer,
    params: FFTParams
  ) {
    let n = Int(params.n)
    let batch = Int(params.batch)
    let rowGroups = MTLSize(width: n * batch, height: 1, depth: 1)
    let rowThreads = MTLSize(width: Self.fftThreads, height: 1, depth: 1)
    let transposeCount = (n + 31) / 32
    let transposeGroups = MTLSize(
      width: transposeCount,
      height: transposeCount,
      depth: batch
    )
    let transposeThreads = MTLSize(width: 32, height: 8, depth: 1)
    var mutable = params

    let firstRows = commands.makeComputeCommandEncoder()!
    firstRows.setComputePipelineState(fftPipeline)
    firstRows.setBuffer(input, offset: 0, index: 0)
    firstRows.setBuffer(temporary, offset: 0, index: 1)
    firstRows.setBuffer(twiddleBuffer, offset: 0, index: 2)
    firstRows.setBytes(
      &mutable,
      length: MemoryLayout<FFTParams>.stride,
      index: 3
    )
    firstRows.setThreadgroupMemoryLength(
      n * MemoryLayout<SIMD2<Float>>.stride,
      index: 0
    )
    firstRows.dispatchThreadgroups(
      rowGroups,
      threadsPerThreadgroup: rowThreads
    )
    firstRows.endEncoding()

    let firstTranspose = commands.makeComputeCommandEncoder()!
    firstTranspose.setComputePipelineState(transposePipeline)
    firstTranspose.setBuffer(temporary, offset: 0, index: 0)
    firstTranspose.setBuffer(output, offset: 0, index: 1)
    firstTranspose.setBytes(
      &mutable,
      length: MemoryLayout<FFTParams>.stride,
      index: 2
    )
    firstTranspose.dispatchThreadgroups(
      transposeGroups,
      threadsPerThreadgroup: transposeThreads
    )
    firstTranspose.endEncoding()

    let secondRows = commands.makeComputeCommandEncoder()!
    secondRows.setComputePipelineState(fftPipeline)
    secondRows.setBuffer(output, offset: 0, index: 0)
    secondRows.setBuffer(temporary, offset: 0, index: 1)
    secondRows.setBuffer(twiddleBuffer, offset: 0, index: 2)
    secondRows.setBytes(
      &mutable,
      length: MemoryLayout<FFTParams>.stride,
      index: 3
    )
    secondRows.setThreadgroupMemoryLength(
      n * MemoryLayout<SIMD2<Float>>.stride,
      index: 0
    )
    secondRows.dispatchThreadgroups(
      rowGroups,
      threadsPerThreadgroup: rowThreads
    )
    secondRows.endEncoding()

    let secondTranspose = commands.makeComputeCommandEncoder()!
    secondTranspose.setComputePipelineState(transposePipeline)
    secondTranspose.setBuffer(temporary, offset: 0, index: 0)
    secondTranspose.setBuffer(output, offset: 0, index: 1)
    secondTranspose.setBytes(
      &mutable,
      length: MemoryLayout<FFTParams>.stride,
      index: 2
    )
    secondTranspose.dispatchThreadgroups(
      transposeGroups,
      threadsPerThreadgroup: transposeThreads
    )
    secondTranspose.endEncoding()
  }

  private func encodeFullAccumulator(
    _ commands: MTLCommandBuffer,
    source: MTLBuffer,
    params: SSBParams
  ) {
    var mutable = params
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(fullAccumulatePipeline)
    encoder.setBuffer(source, offset: 0, index: 0)
    encoder.setBuffer(activeGeometryBuffer, offset: 0, index: 1)
    encoder.setBuffer(activeTrigBuffer, offset: 0, index: 2)
    encoder.setBuffer(qxBuffer, offset: 0, index: 3)
    encoder.setBuffer(qyBuffer, offset: 0, index: 4)
    encoder.setBuffer(accumulator, offset: 0, index: 5)
    encoder.setBytes(
      &mutable,
      length: MemoryLayout<SSBParams>.stride,
      index: 6
    )
    encoder.dispatchThreads(
      MTLSize(width: Self.plane, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
  }

  private func encodeCachedAccumulator(
    _ commands: MTLCommandBuffer,
    sources: [MTLBuffer],
    params: SSBParams
  ) throws {
    guard let fallback = sources.first else { return }
    guard sources.count <= Self.maximumCacheChunks else {
      throw MetalSSBError.tooManyCacheChunks(
        required: sources.count,
        supported: Self.maximumCacheChunks
      )
    }
    var mutable = params
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(halfAccumulatePipeline)
    for index in 0..<Self.maximumCacheChunks {
      encoder.setBuffer(
        index < sources.count ? sources[index] : fallback,
        offset: 0,
        index: index
      )
    }
    encoder.setBuffer(activeGeometryBuffer, offset: 0, index: 12)
    encoder.setBuffer(qxBuffer, offset: 0, index: 13)
    encoder.setBuffer(qyBuffer, offset: 0, index: 14)
    encoder.setBuffer(accumulator, offset: 0, index: 15)
    encoder.setBytes(
      &mutable,
      length: MemoryLayout<SSBParams>.stride,
      index: 16
    )
    encoder.dispatchThreads(
      MTLSize(width: Self.halfPlane, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
  }

  private func encodeFinalize(
    _ commands: MTLCommandBuffer,
    params: SSBParams
  ) {
    var mutable = params
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(finalizePipeline)
    encoder.setBuffer(accumulator, offset: 0, index: 0)
    encoder.setBytes(
      &mutable,
      length: MemoryLayout<SSBParams>.stride,
      index: 1
    )
    encoder.dispatchThreads(
      MTLSize(width: Self.plane, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
  }

  private func encodeChiTrig(
    _ commands: MTLCommandBuffer,
    params: SSBParams,
    output: MTLBuffer
  ) {
    var mutable = params
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(chiTrigPipeline)
    encoder.setBuffer(qxBuffer, offset: 0, index: 0)
    encoder.setBuffer(qyBuffer, offset: 0, index: 1)
    encoder.setBuffer(output, offset: 0, index: 2)
    encoder.setBytes(
      &mutable,
      length: MemoryLayout<SSBParams>.stride,
      index: 3
    )
    encoder.dispatchThreads(
      MTLSize(width: Self.plane, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
  }

  private func encodeCrossTrig(
    _ commands: MTLCommandBuffer,
    params: SSBParams,
    output: MTLBuffer
  ) {
    var mutable = params
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(crossTrigPipeline)
    encoder.setBuffer(activeGeometryBuffer, offset: 0, index: 0)
    encoder.setBuffer(qxBuffer, offset: 0, index: 1)
    encoder.setBuffer(qyBuffer, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)
    encoder.setBytes(
      &mutable,
      length: MemoryLayout<SSBParams>.stride,
      index: 4
    )
    encoder.dispatchThreads(
      MTLSize(
        width: activeBrightfieldIndices.count * 2 * Self.size,
        height: 1,
        depth: 1
      ),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
  }

  private func encodeFusedCachedPhaseLoss(
    _ commands: MTLCommandBuffer,
    source: MTLBuffer,
    sourceOffset: Int,
    batch: Int,
    params: SSBParams
  ) throws {
    guard let chiTrigBuffer, let crossTrigBuffer else {
      throw MetalSSBError.allocation("exact phase-loss tables")
    }
    var mutable = params
    let rows = commands.makeComputeCommandEncoder()!
    rows.setComputePipelineState(fusedLossRowPipeline)
    rows.setBuffer(source, offset: sourceOffset, index: 0)
    rows.setBuffer(activeGeometryBuffer, offset: 0, index: 1)
    rows.setBuffer(qxBuffer, offset: 0, index: 2)
    rows.setBuffer(qyBuffer, offset: 0, index: 3)
    rows.setBuffer(twiddleBuffer, offset: 0, index: 4)
    rows.setBuffer(fftA, offset: 0, index: 5)
    rows.setBytes(
      &mutable,
      length: MemoryLayout<SSBParams>.stride,
      index: 6
    )
    rows.setBuffer(chiTrigBuffer, offset: 0, index: 7)
    rows.setBuffer(crossTrigBuffer, offset: 0, index: 8)
    rows.dispatchThreadgroups(
      MTLSize(width: Self.halfColumns, height: batch, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
    )
    rows.endEncoding()

    var mutableBatch = UInt32(batch)
    var dc = paramsDC(params)
    let columns = commands.makeComputeCommandEncoder()!
    columns.setComputePipelineState(fusedLossMomentPipeline)
    columns.setBuffer(fftA, offset: 0, index: 0)
    columns.setBuffer(twiddleBuffer, offset: 0, index: 1)
    columns.setBuffer(phaseSumBuffer, offset: 0, index: 2)
    columns.setBuffer(phaseSumSquaredBuffer, offset: 0, index: 3)
    columns.setBytes(
      &mutableBatch,
      length: MemoryLayout<UInt32>.stride,
      index: 4
    )
    columns.setBytes(
      &dc,
      length: MemoryLayout<SIMD2<Float>>.stride,
      index: 5
    )
    columns.dispatchThreadgroups(
      MTLSize(width: Self.size, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
    )
    columns.endEncoding()
  }

  private func paramsDC(_ params: SSBParams) -> SIMD2<Float> {
    SIMD2<Float>(params.dcReal, params.dcImaginary)
  }

  private func encodeLossCorrection(
    _ commands: MTLCommandBuffer,
    source: MTLBuffer,
    sourceOffset: Int,
    sourceIsHermitianHalf: Bool,
    params: SSBParams
  ) {
    var mutable = params
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(
      sourceIsHermitianHalf
        ? halfLossCorrectionPipeline
        : fullLossCorrectionPipeline
    )
    encoder.setBuffer(source, offset: sourceOffset, index: 0)
    encoder.setBuffer(activeGeometryBuffer, offset: 0, index: 1)
    encoder.setBuffer(qxBuffer, offset: 0, index: 2)
    encoder.setBuffer(qyBuffer, offset: 0, index: 3)
    encoder.setBuffer(fftB, offset: 0, index: 4)
    encoder.setBytes(
      &mutable,
      length: MemoryLayout<SSBParams>.stride,
      index: 5
    )
    encoder.dispatchThreads(
      MTLSize(
        width: Int(params.batch) * Self.plane,
        height: 1,
        depth: 1
      ),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
  }

  private func encodePhaseMoments(
    _ commands: MTLCommandBuffer,
    batch: Int
  ) {
    var mutableBatch = UInt32(batch)
    let encoder = commands.makeComputeCommandEncoder()!
    encoder.setComputePipelineState(phaseMomentPipeline)
    encoder.setBuffer(fftB, offset: 0, index: 0)
    encoder.setBuffer(phaseSumBuffer, offset: 0, index: 1)
    encoder.setBuffer(phaseSumSquaredBuffer, offset: 0, index: 2)
    encoder.setBytes(
      &mutableBatch,
      length: MemoryLayout<UInt32>.stride,
      index: 3
    )
    encoder.dispatchThreads(
      MTLSize(width: Self.plane, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
  }

  private func commitAndWait(_ commands: MTLCommandBuffer) throws {
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error {
      throw MetalSSBError.commandExecution(error.localizedDescription)
    }
  }

  private func gpuDuration(_ commands: MTLCommandBuffer) -> Double {
    guard commands.gpuEndTime > commands.gpuStartTime else { return 0 }
    return commands.gpuEndTime - commands.gpuStartTime
  }

  private static func validate(_ geometry: MetalSSBGeometry) throws {
    let count = geometry.logicalBrightfieldCount
    guard count > 0 else {
      throw MetalSSBError.invalidGeometry("logical BF count must be positive")
    }
    let arrays = [
      geometry.brightfieldKY,
      geometry.brightfieldAlphaSquared,
      geometry.brightfieldAperture,
      geometry.brightfieldCos2Phi,
      geometry.brightfieldSin2Phi,
    ]
    guard arrays.allSatisfy({ $0.count == count }) else {
      throw MetalSSBError.invalidGeometry(
        "all bright-field arrays must have \(count) values"
      )
    }
    guard geometry.qxByRow.count == size,
      geometry.qyByColumn.count == size
    else {
      throw MetalSSBError.invalidGeometry(
        "qxByRow and qyByColumn must each contain 512 values"
      )
    }
    let allArrays =
      [geometry.brightfieldKX] + arrays
      + [geometry.qxByRow, geometry.qyByColumn]
    guard
      allArrays.allSatisfy({ values in
        values.allSatisfy { $0.isFinite }
      })
    else {
      throw MetalSSBError.invalidGeometry(
        "coordinate and aperture arrays must be finite"
      )
    }
    guard
      geometry.brightfieldAperture.allSatisfy({
        $0 >= 0 && $0 <= 1
      })
    else {
      throw MetalSSBError.invalidGeometry(
        "bright-field aperture values must be within 0...1"
      )
    }
    guard geometry.wavelengthAngstroms.isFinite,
      geometry.wavelengthAngstroms > 0,
      geometry.semiangleRadians.isFinite,
      geometry.semiangleRadians > 0,
      geometry.angularSamplingYRadians.isFinite,
      geometry.angularSamplingYRadians > 0,
      geometry.angularSamplingXRadians.isFinite,
      geometry.angularSamplingXRadians > 0,
      geometry.dcValue.x.isFinite,
      geometry.dcValue.y.isFinite,
      geometry.referenceRotationDegrees.isFinite
    else {
      throw MetalSSBError.invalidGeometry(
        "physical calibration scalars must be finite and positive"
      )
    }
  }

  private static func activeIndices(
    _ geometry: MetalSSBGeometry
  ) -> [Int] {
    let maximumEdgeWidth = max(
      geometry.angularSamplingYRadians,
      geometry.angularSamplingXRadians
    )
    let outerAlpha = geometry.semiangleRadians + 0.5 * maximumEdgeWidth
    return geometry.brightfieldKX.indices.filter { index in
      hypot(
        geometry.brightfieldKX[index],
        geometry.brightfieldKY[index]
      ) * geometry.wavelengthAngstroms <= outerAlpha
    }
  }

  private static func allocate(
    device: MTLDevice,
    length: Int,
    options: MTLResourceOptions,
    purpose: String
  ) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: length, options: options) else {
      throw MetalSSBError.allocation(purpose)
    }
    return buffer
  }

  private static func makeBuffer<T>(
    device: MTLDevice,
    values: [T],
    purpose: String
  ) throws -> MTLBuffer {
    guard
      let buffer = values.withUnsafeBytes({ bytes in
        bytes.baseAddress.flatMap {
          device.makeBuffer(
            bytes: $0,
            length: bytes.count,
            options: .storageModeShared
          )
        }
      })
    else {
      throw MetalSSBError.allocation(purpose)
    }
    return buffer
  }

  private static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
    let packagedURL = Bundle.main.resourceURL?
      .appendingPathComponent(
        "MetalKernels_MetalSSBKernels.bundle",
        isDirectory: true
      )
      .appendingPathComponent("Resources", isDirectory: true)
      .appendingPathComponent("ssb.metal")
    let url =
      packagedURL.flatMap {
        FileManager.default.fileExists(atPath: $0.path) ? $0 : nil
      } ?? Bundle.module.url(
        forResource: "ssb",
        withExtension: "metal",
        subdirectory: "Resources"
      ) ?? Bundle.module.url(forResource: "ssb", withExtension: "metal")
    guard let url else { throw MetalSSBError.missingResource("ssb.metal") }
    do {
      return try device.makeLibrary(
        source: String(contentsOf: url, encoding: .utf8),
        options: nil
      )
    } catch {
      throw MetalSSBError.libraryCompilation(error.localizedDescription)
    }
  }

  private static func makePipeline(
    device: MTLDevice,
    library: MTLLibrary,
    name: String
  ) throws -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else {
      throw MetalSSBError.missingFunction(name)
    }
    do {
      return try device.makeComputePipelineState(function: function)
    } catch {
      throw MetalSSBError.libraryCompilation(
        "pipeline creation failed for \(name)"
      )
    }
  }

}
