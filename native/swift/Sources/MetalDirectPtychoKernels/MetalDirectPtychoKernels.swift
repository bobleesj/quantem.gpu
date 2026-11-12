import Foundation
import Metal
import MetalPerformanceShadersGraph

/// Unsigned counts or signed float32 measurements read without source quantization.
public enum MetalDirectPtychoCountType: Int, Codable, Sendable {
  case uint8 = 1
  case uint16 = 2
  case uint32 = 4
  case float32 = 5

  public var byteWidth: Int { self == .float32 ? 4 : rawValue }
}

/// Aberrations used by the native direct-ptychography reconstruction.
///
/// The kernels evaluate chi = (pi / lambda[A]) alpha^2 (C10 + C12 cos 2(phi - phi12)) with no
/// conversion, so C10 and C12 are Angstrom and phi12 radians. A stored record states them as
/// `C10_nm`, `C12_nm` and `phi12_deg`; the conversion happens once, where the record is written.
public struct MetalDirectPtychoAberrations: Codable, Equatable, Sendable {
  public var c10Angstroms: Float
  public var c12Angstroms: Float
  public var phi12Radians: Float
  public var higherOrder: [MetalDirectPtychoHigherOrder]?

  public init(
    c10Angstroms: Float,
    c12Angstroms: Float,
    phi12Radians: Float,
    higherOrder: [MetalDirectPtychoHigherOrder]? = nil
  ) {
    self.c10Angstroms = c10Angstroms
    self.c12Angstroms = c12Angstroms
    self.phi12Radians = phi12Radians
    self.higherOrder = higherOrder
  }
}

/// Calibrated reciprocal-space geometry for a native square direct ptychography session.
///
/// `qxByRow` is indexed by scan Fourier row and `qyByColumn` by scan Fourier
/// column. Bright-field arrays remain in their logical source order. The
/// engine may skip only entries whose aperture is mathematically zero; the
/// logical count is always retained for normalization.
public struct MetalDirectPtychoGeometry: Codable, Sendable {
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
  /// pi / wavelength, rounded once from the calibration in Double precision.
  public let phaseFactor: Float?
  /// Original calibration precision for dimensionless higher-order search coordinates.
  public let modelWavelengthAngstroms: Double?
  public let modelSemiangleRadians: Double?

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
    referenceRotationDegrees: Float,
    phaseFactor: Float? = nil,
    modelWavelengthAngstroms: Double? = nil, modelSemiangleRadians: Double? = nil
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
    self.phaseFactor = phaseFactor
    self.modelWavelengthAngstroms = modelWavelengthAngstroms
    self.modelSemiangleRadians = modelSemiangleRadians
  }
}

/// Scientific provenance attached to every native Metal direct ptychography result.
public struct MetalDirectPtychoProvenance: Codable, Equatable, Sendable {
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
  /// Exact-grid preparation kernel, when the large irregular-grid path is used.
  public var fourierKernel: MetalKernelIdentity? = nil
}

/// A native Metal direct ptychography reconstruction saved by an earlier version (`MetalDirectSavedRun`).
///
/// Both buffers contain row-major `complex64` values with the native scan shape
/// recorded in `provenance` (128, 256, or 512 pixels per side).
/// They are owned by this result. The engine that produced them, the non-model
/// reconstruction, is retired; `MetalDirectPtychoEngine.phase(of:)` still converts a restored
/// object to its phase.
public struct MetalDirectPtychoResult: @unchecked Sendable {
  public let object: MTLBuffer
  public let fourierSum: MTLBuffer
  public let wallSeconds: Double
  public let gpuSeconds: Double
  public let provenance: MetalDirectPtychoProvenance
}

/// Errors raised by the native Metal direct ptychography engine.
public enum MetalDirectPtychoError: LocalizedError {
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
      "MetalDirectPtychoKernels is missing \(name)."
    case .missingFunction(let name):
      "MetalDirectPtychoKernels is missing the \(name) function."
    case .libraryCompilation(let message):
      "Metal direct ptychography kernel compilation failed: \(message)"
    case .commandQueue:
      "Metal could not create the direct ptychography command queue."
    case .invalidGeometry(let message):
      "The direct ptychography geometry is invalid: \(message)"
    case .inputBufferTooSmall(let required, let actual):
      "The direct ptychography bright-field input needs \(required) bytes, but the buffer contains \(actual)."
    case .allocation(let purpose):
      "Metal could not allocate the direct ptychography \(purpose)."
    case .notPrepared:
      "Prepare the direct ptychography bright-field input before reconstructing or fitting."
    case .tooManyCacheChunks(let required, let supported):
      "The direct ptychography cache needs \(required) chunks, but this kernel supports \(supported)."
    case .commandExecution(let message):
      "The Metal direct ptychography command failed: \(message)"
    }
  }
}

/// GPU-resident preparation of native 128, 256, or 512 square direct ptychography evidence.
///
/// The engine consumes plane-major unsigned bright-field counts without narrowing,
/// Fourier-transforms every bright-field plane and keeps as many Hermitian `G(k)`
/// spectra as its budget admits. It never crops or bins scan positions; every
/// logical bright-field term stays in the normalization and only terms proven to
/// have zero aperture are skipped. `jointModel` reconstructs and fits on these
/// spectra; the engine's own non-model reconstruction and phase-variance search are
/// retired.
public final class MetalDirectPtychoEngine {
  private let size: Int
  private let columns: Int
  private let nativeSquareFFT: Bool
  private var accurateRectangularFFT: MetalDirectPtychoBluesteinFFT?
  private var forwardPlans: [Int: (MPSGraph, MPSGraphTensor, MPSGraphTensor)] = [:]
  private let plane: Int
  private let halfPlane: Int
  private static let batchCapacity = 32
  private var cacheChunkCapacity = 512
  private static let maximumCacheChunks = 12
  private static let fftThreads = 64

  private struct FFTParams {
    var n: UInt32
    var log2n: UInt32
    var batch: UInt32
    var inverse: UInt32
  }

  private struct HalfExtractParams {
    var sourceN: UInt32
    var batch: UInt32
  }

  private let device: MTLDevice
  private var objectPhasePipeline: MTLComputePipelineState?
  private let queue: MTLCommandQueue
  private let geometry: MetalDirectPtychoGeometry
  private let cacheBudgetBytes: Int?
  private let activeBrightfieldIndices: [Int]
  /// The logical BF pixels outside the aperture's support (no transfer, never cached), ascending.
  private let inactiveBrightfieldIndices: [Int]

  private let convertPipeline: MTLComputePipelineState
  private let fftLibrary: MTLLibrary
  private let fftPipeline: MTLComputePipelineState
  private var countsFFTPipeline: MTLComputePipelineState?
  private let fusedCounts = ProcessInfo.processInfo.environment["QUANTEM_METAL_FUSED_COUNTS"] != "0"
  private let transposePipeline: MTLComputePipelineState
  private let halfExtractPipeline: MTLComputePipelineState
  private let transposeHalfPipeline: MTLComputePipelineState
  private var fusedHalf: Bool { nativeSquareFFT && ProcessInfo.processInfo.environment["QUANTEM_METAL_FUSED_HALF"] != "0" }

  private var rawBuffer: MTLBuffer
  private var sourceCountType: MetalDirectPtychoCountType = .uint8
  private var fftA: MTLBuffer
  private var fftB: MTLBuffer
  private let twiddleBuffer: MTLBuffer

  private var encodeBrightfield: (([Int], MTLBuffer, MTLCommandBuffer) throws -> Void)?
  private var cacheBuffers: [MTLBuffer] = []
  private var cachedBrightfieldCount = 0
  private var prepared = false
  private var preparationGeneration: UInt64 = 0

  /// Create a reusable direct ptychography engine.
  ///
  /// - Parameters:
  ///   - device: Apple Metal device that owns all source and result buffers.
  ///   - geometry: Exact full-BF calibration in logical source order.
  ///   - cacheBudgetBytes: Maximum bytes retained for the Hermitian `G(k)`
  ///     cache. `nil` requests a complete cache. A smaller budget preserves
  ///     exactness by streaming the uncached tail from the source buffer.
  public init(
    device: MTLDevice,
    geometry: MetalDirectPtychoGeometry,
    cacheBudgetBytes: Int? = nil
  ) throws {
    try Self.validate(geometry)
    let size = geometry.qxByRow.count
    let columns = geometry.qyByColumn.count
    let plane = size * columns
    self.columns = columns
    self.nativeSquareFFT = size == columns && [128, 256, 512].contains(size)
    self.size = size
    self.plane = plane
    self.halfPlane = size * (columns / 2 + 1)
    if let cacheBudgetBytes, cacheBudgetBytes < 0 {
      throw MetalDirectPtychoError.invalidGeometry(
        "cacheBudgetBytes must be nonnegative"
      )
    }
    self.device = device
    self.geometry = geometry
    self.cacheBudgetBytes = cacheBudgetBytes
    guard let queue = device.makeCommandQueue() else {
      throw MetalDirectPtychoError.commandQueue
    }
    self.queue = queue
    if MetalDirectPtychoBluesteinFFT.isRequired(rows: size, columns: columns) {
      accurateRectangularFFT = try MetalDirectPtychoBluesteinFFT(
        device: device, queue: queue, rows: size, columns: columns)
    }

    let activeIndices = Self.activeIndices(geometry)
    guard !activeIndices.isEmpty else {
      throw MetalDirectPtychoError.invalidGeometry(
        "no bright-field term has nonzero aperture"
      )
    }
    activeBrightfieldIndices = activeIndices
    let active = Set(activeIndices)
    inactiveBrightfieldIndices = (0..<geometry.logicalBrightfieldCount).filter {
      !active.contains($0)
    }
    let library = try Self.makeLibrary(device: device)
    fftLibrary = library
    convertPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "counts_to_complex"
    )
    fftPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: size == 512 ? "fft_rows_radix8" : "fft_rows_small_mixed_radix"
    )
    transposePipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "transpose_complex32"
    )
    halfExtractPipeline = try Self.makePipeline(
      device: device,
      library: library,
      name: "extract_rectangular_hermitian_half"
    )

    transposeHalfPipeline = try Self.makePipeline(
      device: device, library: library, name: "transpose_hermitian_half32")

    let rawBytes = Self.batchCapacity * plane
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

    var twiddle: [SIMD2<Float>] = []
    twiddle.reserveCapacity(size)
    for index in 0..<size {
      let angle = -2 * Float.pi * Float(index) / Float(size)
      twiddle.append(SIMD2<Float>(cos(angle), sin(angle)))
    }
    twiddleBuffer = try Self.makeBuffer(
      device: device,
      values: twiddle,
      purpose: "FFT twiddle table"
    )
  }

  /// Number of exact nonzero-aperture terms executed by this session.
  public var executedBrightfieldCount: Int {
    activeBrightfieldIndices.count
  }

  /// Number of logical bright-field terms retained in normalization.
  public var logicalBrightfieldCount: Int {
    geometry.logicalBrightfieldCount
  }

  /// Prepare plane-major bright-field counts without clipping or integer narrowing.
  ///
  /// The source shape is `[logicalBrightfieldCount, scanRows, scanColumns]`. Preparation
  /// builds as much of the exact Hermitian `G(k)` cache as the configured
  /// budget admits and keeps the source buffer for any exact streamed tail.
  public func prepare(
    brightfield: MTLBuffer, countType: MetalDirectPtychoCountType = .uint8
  ) throws {
    let requiredBytes = geometry.logicalBrightfieldCount * plane * countType.byteWidth
    guard brightfield.length >= requiredBytes else {
      throw MetalDirectPtychoError.inputBufferTooSmall(
        required: requiredBytes,
        actual: brightfield.length
      )
    }
    try prepare(countType: countType) { [plane] indices, destination, commands in
      guard let blit = commands.makeBlitCommandEncoder() else {
        throw MetalDirectPtychoError.commandQueue
      }
      for (local, logical) in indices.enumerated() {
        blit.copy(
          from: brightfield,
          sourceOffset: logical * plane * countType.byteWidth,
          to: destination, destinationOffset: local * plane * countType.byteWidth,
          size: plane * countType.byteWidth)
      }
      blit.endEncoding()
    }
  }

  /// Prepare directly from a resident source, without a dense bright-field copy.
  ///
  /// The encoder writes the requested logical detector columns in plane-major
  /// order into the bounded destination on the supplied command buffer. It
  /// must not commit that buffer. Source and engine calls must be serialized.
  /// The closure is retained for exact streaming when the Fourier cache is full.
  public func prepare(
    countType: MetalDirectPtychoCountType,
    progress: (Int, Int) -> Void = { _, _ in },
    encodeBrightfield: @escaping ([Int], MTLBuffer, MTLCommandBuffer) throws -> Void
  ) throws {
    preparationGeneration &+= 1
    prepared = false
    self.encodeBrightfield = encodeBrightfield
    try resizeStaging(batch: Self.batchCapacity, countType: countType)
    sourceCountType = countType
    if fusedCounts && nativeSquareFFT && size == 512 {
      let constants = MTLFunctionConstantValues()
      var width = UInt32(countType.rawValue)
      constants.setConstantValue(&width, type: .uint, index: 43)
      let function = try fftLibrary.makeFunction(
        name: "fft_rows_radix8", constantValues: constants)
      countsFFTPipeline = try device.makeComputePipelineState(function: function)
    }
    cacheBuffers.removeAll(keepingCapacity: false)
    cachedBrightfieldCount = 0

    let bytesPerBrightfield =
      halfPlane
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
    cacheChunkCapacity = 512
    while cachedCount > cacheChunkCapacity * Self.maximumCacheChunks {
      cacheChunkCapacity *= 2
    }
    let chunks =
      cachedCount == 0
      ? 0
      : (cachedCount + cacheChunkCapacity - 1)
        / cacheChunkCapacity
    guard chunks <= Self.maximumCacheChunks else {
      throw MetalDirectPtychoError.tooManyCacheChunks(
        required: chunks,
        supported: Self.maximumCacheChunks
      )
    }

    let halfBytes = halfPlane * MemoryLayout<SIMD2<Float>>.stride
    for chunk in 0..<chunks {
      let count = min(
        cacheChunkCapacity,
        cachedCount - chunk * cacheChunkCapacity
      )
      cacheBuffers.append(
        try Self.allocate(
          device: device,
          length: count * halfBytes,
          options: .storageModePrivate,
          purpose: "Hermitian G(k) cache chunk \(chunk)"
        ))
    }

    progress(0, cachedCount)
    for offset in stride(
      from: 0,
      to: cachedCount,
      by: Self.batchCapacity
    ) {
      let batch = min(Self.batchCapacity, cachedCount - offset)
      let commands = try fftCommand()
      try encodeBrightfield(
        Array(activeBrightfieldIndices[offset..<(offset + batch)]), rawBuffer, commands)
      encodeForwardFFT(commands, batch: batch, transposeOutput: !fusedHalf)

      let cacheIndex = offset / cacheChunkCapacity
      let cacheOffset = offset - cacheIndex * cacheChunkCapacity
      var extract = SIMD4<UInt32>(UInt32(size), UInt32(columns), UInt32(batch), 0)
      guard let encoder = commands.makeComputeCommandEncoder() else {
        throw MetalDirectPtychoError.commandQueue
      }
      encoder.setComputePipelineState(fusedHalf ? transposeHalfPipeline : halfExtractPipeline)
      encoder.setBuffer(fusedHalf ? fftB : fftA, offset: 0, index: 0)
      encoder.setBuffer(
        cacheBuffers[cacheIndex],
        offset: cacheOffset * halfBytes,
        index: 1
      )
      if fusedHalf {
        var params = FFTParams(
          n: UInt32(size), log2n: UInt32(size.trailingZeroBitCount),
          batch: UInt32(batch), inverse: 0)
        encoder.setBytes(&params, length: MemoryLayout<FFTParams>.stride, index: 2)
        encoder.dispatchThreadgroups(
          MTLSize(width: (size + 31) / 32, height: (size / 2 + 32) / 32, depth: batch),
          threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1))
      } else {
        encoder.setBytes(&extract, length: MemoryLayout<SIMD4<UInt32>>.stride, index: 2)
        encoder.dispatchThreads(
          MTLSize(width: batch * halfPlane, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      }
      encoder.endEncoding()
      try commitAndWait(commands)
      progress(offset + batch, cachedCount)
    }
    cachedBrightfieldCount = cachedCount
    if ProcessInfo.processInfo.environment["QUANTEM_METAL_TRIM_FFT_WORKSPACE"] != "0" {
      // Subsequent uncached and inactive evidence is streamed in batches of at most eight.
      // The preparation commands have completed; no live FFT can reference the old buffers.
      try resizeStaging(batch: 8, countType: countType)
    }
    prepared = true
  }

  /// Temporary Fourier workspace, separate from the exact retained spectra.
  var preparationWorkspaceBytes: Int { rawBuffer.length + fftA.length + fftB.length }

  private func resizeStaging(batch: Int, countType: MetalDirectPtychoCountType) throws {
    let rawBytes = batch * plane * countType.byteWidth
    let complexBytes = batch * plane * MemoryLayout<SIMD2<Float>>.stride
    if rawBuffer.length != rawBytes {
      rawBuffer = try Self.allocate(device: device, length: rawBytes,
        options: .storageModePrivate, purpose: "source count batch")
    }
    if fftA.length != complexBytes {
      fftA = try Self.allocate(device: device, length: complexBytes,
        options: .storageModePrivate, purpose: "FFT workspace A")
    }
    if fftB.length != complexBytes {
      fftB = try Self.allocate(device: device, length: complexBytes,
        options: .storageModePrivate, purpose: "FFT workspace B")
    }
  }

  /// Build the joint tilt/depth model from this session's exact prepared spectra.
  /// Calls must be serialized with all other operations on this engine.
  public func jointModel(
    scanDetectorRotationDegrees: Double,
    bandInverseAngstroms: ClosedRange<Float> = 0.2...0.9,
    isCancelled: () -> Bool = { false }
  ) throws -> MetalDirectPtychoJointModel {
    guard prepared else { throw MetalDirectPtychoError.notPrepared }
    let generation = preparationGeneration
    return try MetalDirectPtychoJointModel(
      device: device, geometry: geometry,
      activeIndices: activeBrightfieldIndices, rotation: scanDetectorRotationDegrees,
      spectrumStorage: cachedBrightfieldCount == activeBrightfieldIndices.count
        && device.argumentBuffersSupport == .tier2
        && ProcessInfo.processInfo.environment["QUANTEM_METAL_INDEXED_BAND"] == "1"
        ? DirectPtychoSpectrumStorage(buffers: cacheBuffers, chunkCapacity: cacheChunkCapacity,
            planeStride: halfPlane) : nil,
      band: bandInverseAngstroms, provenance: currentProvenance(),
      visit: { [self] batchSize, body in try visitSpectra(batchSize: batchSize, body) },
      visitInactive: { [self] batchSize, body in
        try visitInactiveSpectra(batchSize: batchSize, body)
      },
      validatePreparation: { [self] in
        guard prepared, preparationGeneration == generation else {
          throw MetalDirectPtychoError.invalidGeometry(
            "The direct ptychography source was prepared again. Create a new joint model before evaluating or reconstructing."
          )
        }
      }, isCancelled: isCancelled)
  }

  // A narrow internal bridge: cached Hermitian spectra and streamed full spectra
  // are consumed on the engine queue, never copied through CPU arrays.
  private func visitSpectra(batchSize: Int, _ body: MetalDirectPtychoJointModel.SpectrumVisitor)
    throws
  {
    guard prepared, let encodeBrightfield else { throw MetalDirectPtychoError.notPrepared }
    let strideBytes = halfPlane * MemoryLayout<SIMD2<Float>>.stride
    var offset = 0
    while offset < cachedBrightfieldCount {
      let within = offset % cacheChunkCapacity
      let count = min(batchSize, cachedBrightfieldCount - offset, cacheChunkCapacity - within)
      let chunk = offset / cacheChunkCapacity
      try body(cacheBuffers[chunk], within * strideBytes, true, offset, count)
      offset += count
    }
    for offset in stride(from: cachedBrightfieldCount, to: activeBrightfieldIndices.count, by: 8) {
      let count = min(8, activeBrightfieldIndices.count - offset)
      let commands = try fftCommand()
      try encodeBrightfield(
        Array(activeBrightfieldIndices[offset..<(offset + count)]), rawBuffer, commands)
      encodeForwardFFT(commands, batch: count)
      try commitAndWait(commands)
      try body(fftA, 0, false, offset, count)
    }
  }

  /// The full spectra of the BF pixels outside the aperture's support, in `inactiveBrightfieldIndices` order and
  /// batches of at most 8, streamed like the uncached tail: the regularized solves' inverse-mean weights need their
  /// scan means (Python weights every selected BF pixel).
  private func visitInactiveSpectra(
    batchSize: Int, _ body: MetalDirectPtychoJointModel.SpectrumVisitor
  )
    throws
  {
    guard prepared, let encodeBrightfield else { throw MetalDirectPtychoError.notPrepared }
    for offset in stride(
      from: 0, to: inactiveBrightfieldIndices.count, by: min(8, max(1, batchSize)))
    {
      let count = min(8, max(1, batchSize), inactiveBrightfieldIndices.count - offset)
      let commands = try fftCommand()
      try encodeBrightfield(
        Array(inactiveBrightfieldIndices[offset..<(offset + count)]), rawBuffer, commands)
      encodeForwardFFT(commands, batch: count)
      try commitAndWait(commands)
      try body(fftA, 0, false, offset, count)
    }
  }

  /// Convert a saved complex object (`MetalDirectSavedRun.reconstruction`) to its scalar phase, in
  /// radians, on Metal. The engine's geometry must have the saved run's scan size.
  public func phase(of result: MetalDirectPtychoResult) throws -> MTLBuffer {
    guard result.provenance.scanRows == size, result.provenance.scanColumns == columns,
      result.object.length >= plane * MemoryLayout<SIMD2<Float>>.stride
    else {
      throw MetalDirectPtychoError.invalidGeometry(
        "This result is \(result.provenance.scanRows) by \(result.provenance.scanColumns), but the engine is \(size) by \(columns). Restore it with matching scan geometry before converting phase."
      )
    }
    if objectPhasePipeline == nil {
      objectPhasePipeline = try Self.makePipeline(
        device: device,
        library: Self.makeLibrary(device: device), name: "direct_object_phase")
    }
    let output = try Self.allocate(
      device: device, length: plane * 4,
      options: .storageModeShared, purpose: "object phase")
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder(),
      let objectPhasePipeline
    else { throw MetalDirectPtychoError.commandQueue }
    encoder.setComputePipelineState(objectPhasePipeline)
    encoder.setBuffer(result.object, offset: 0, index: 0)
    encoder.setBuffer(output, offset: 0, index: 1)
    encoder.dispatchThreads(
      MTLSize(width: plane, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    encoder.endEncoding()
    try commitAndWait(command)
    return output
  }

  private func currentProvenance() -> MetalDirectPtychoProvenance {
    let logical = geometry.logicalBrightfieldCount
    let executed = activeBrightfieldIndices.count
    return MetalDirectPtychoProvenance(
      scanRows: size,
      scanColumns: columns,
      sourceDType: String(describing: sourceCountType),
      computeDType: "float32/complex64",
      logicalBrightfieldCount: logical,
      executedBrightfieldCount: executed,
      zeroApertureBrightfieldCount:
        logical - executed,
      cachedBrightfieldCount: cachedBrightfieldCount,
      streamedBrightfieldCount:
        executed - cachedBrightfieldCount,
      cacheBytes: cacheBuffers.reduce(0) { $0 + $1.length },
      scanBin: 1,
      scanCrop: "none",
      brightfieldSelection: "exact-zero-aperture-pruning",
      fourierKernel: accurateRectangularFFT?.kernel
    )
  }

  /// MPSGraph transforms the exact native grid; no scan padding changes the model.
  private func fftCommand() throws -> MTLCommandBuffer {
    guard let command = queue.makeCommandBuffer() else { throw MetalDirectPtychoError.commandQueue }
    return nativeSquareFFT || accurateRectangularFFT != nil
      ? command : MPSCommandBuffer(commandBuffer: command)
  }

  private func encodeRectangularFFT(_ commands: MPSCommandBuffer, batch: Int) {
    let shape = [NSNumber(value: batch), NSNumber(value: size), NSNumber(value: columns)]
    if forwardPlans[batch] == nil {
      let graph = MPSGraph()
      graph.options = .none
      let input = graph.placeholder(shape: shape, dataType: .complexFloat32, name: "scan")
      let descriptor = MPSGraphFFTDescriptor()
      descriptor.scalingMode = .none
      let output = graph.fastFourierTransform(input, axes: [1, 2], descriptor: descriptor, name: "fft2")
      forwardPlans[batch] = (graph, input, output)
    }
    let (graph, input, output) = forwardPlans[batch]!
    let execution = MPSGraphExecutionDescriptor()
    execution.waitUntilCompleted = false
    graph.encode(to: commands,
      feeds: [input: MPSGraphTensorData(fftA, shape: shape, dataType: .complexFloat32)],
      targetOperations: nil,
      resultsDictionary: [output: MPSGraphTensorData(fftB, shape: shape, dataType: .complexFloat32)],
      executionDescriptor: execution)
    let blit = commands.makeBlitCommandEncoder()!
    blit.copy(from: fftB, sourceOffset: 0, to: fftA, destinationOffset: 0, size: batch * plane * 8)
    blit.endEncoding()
  }

  private func encodeForwardFFT(
    _ commands: MTLCommandBuffer,
    batch: Int,
    transposeOutput: Bool = true
  ) {
    if let countsFFTPipeline {
      encodeFFT(
        commands, input: rawBuffer, temporary: fftB, output: fftA,
        params: FFTParams(
          n: UInt32(size), log2n: UInt32(size.trailingZeroBitCount),
          batch: UInt32(batch), inverse: 0),
        transposeOutput: transposeOutput, firstPipeline: countsFFTPipeline)
      return
    }
    var count = UInt32(batch * plane)
    let convert = commands.makeComputeCommandEncoder()!
    convert.setComputePipelineState(convertPipeline)
    convert.setBuffer(rawBuffer, offset: 0, index: 0)
    convert.setBuffer(fftA, offset: 0, index: 1)
    convert.setBytes(&count, length: MemoryLayout<UInt32>.stride, index: 2)
    var byteWidth = UInt32(sourceCountType.rawValue)
    convert.setBytes(&byteWidth, length: MemoryLayout<UInt32>.stride, index: 3)
    convert.dispatchThreads(
      MTLSize(width: Int(count), height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    convert.endEncoding()
    if let accurateRectangularFFT {
      accurateRectangularFFT.encode(command: commands, input: fftA, temporary: fftB, batch: batch)
      return
    }
    if !nativeSquareFFT {
      encodeRectangularFFT(commands as! MPSCommandBuffer, batch: batch)
      return
    }
    encodeFFT(
      commands,
      input: fftA,
      temporary: fftB,
      output: fftA,
      params: FFTParams(
        n: UInt32(size),
        log2n: UInt32(size.trailingZeroBitCount),
        batch: UInt32(batch),
        inverse: 0
      ),
      transposeOutput: transposeOutput
    )
  }

  private func encodeFFT(
    _ commands: MTLCommandBuffer,
    input: MTLBuffer,
    temporary: MTLBuffer,
    output: MTLBuffer,
    params: FFTParams,
    transposeOutput: Bool = true,
    firstPipeline: MTLComputePipelineState? = nil
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
    firstRows.setComputePipelineState(firstPipeline ?? fftPipeline)
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

    guard transposeOutput else { return }
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

  private func commitAndWait(_ commands: MTLCommandBuffer) throws {
    commands.commit()
    commands.waitUntilCompleted()
    if let error = commands.error {
      throw MetalDirectPtychoError.commandExecution(error.localizedDescription)
    }
  }

  private static func validate(_ geometry: MetalDirectPtychoGeometry) throws {
    let count = geometry.logicalBrightfieldCount
    guard count > 0 else {
      throw MetalDirectPtychoError.invalidGeometry("logical BF count must be positive")
    }
    let arrays = [
      geometry.brightfieldKY,
      geometry.brightfieldAlphaSquared,
      geometry.brightfieldAperture,
      geometry.brightfieldCos2Phi,
      geometry.brightfieldSin2Phi,
    ]
    guard arrays.allSatisfy({ $0.count == count }) else {
      throw MetalDirectPtychoError.invalidGeometry(
        "all bright-field arrays must have \(count) values"
      )
    }
    guard (3...512).contains(geometry.qxByRow.count),
      (3...512).contains(geometry.qyByColumn.count)
    else {
      throw MetalDirectPtychoError.invalidGeometry(
        "Native Direct Ptycho supports 3–512 scan positions on each axis, including rectangular scans; retain the original sampling."
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
      throw MetalDirectPtychoError.invalidGeometry(
        "coordinate and aperture arrays must be finite"
      )
    }
    guard
      geometry.brightfieldAperture.allSatisfy({
        $0 >= 0 && $0 <= 1
      })
    else {
      throw MetalDirectPtychoError.invalidGeometry(
        "bright-field aperture values must be within 0...1"
      )
    }
    guard geometry.wavelengthAngstroms.isFinite,
      geometry.wavelengthAngstroms > 0,
      geometry.phaseFactor.map({ $0.isFinite && $0 > 0 }) ?? true,
      geometry.modelWavelengthAngstroms.map({
        $0.isFinite && $0 > 0 && Float($0) == geometry.wavelengthAngstroms
      }) ?? true,
      geometry.modelSemiangleRadians.map({
        $0.isFinite && $0 > 0 && Float($0) == geometry.semiangleRadians
      }) ?? true,
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
      throw MetalDirectPtychoError.invalidGeometry(
        "physical calibration scalars must be finite and positive"
      )
    }
  }

  private static func activeIndices(
    _ geometry: MetalDirectPtychoGeometry
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
      throw MetalDirectPtychoError.allocation(purpose)
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
      throw MetalDirectPtychoError.allocation(purpose)
    }
    return buffer
  }

  private static func makeLibrary(device: MTLDevice) throws -> MTLLibrary {
    do {
      return try device.makeLibrary(
        source: MetalDirectPtychoShaderSource.load(["direct_fft", "direct"]),
        options: nil
      )
    } catch {
      throw MetalDirectPtychoError.libraryCompilation(error.localizedDescription)
    }
  }

  private static func makePipeline(
    device: MTLDevice,
    library: MTLLibrary,
    name: String
  ) throws -> MTLComputePipelineState {
    let function: MTLFunction
    if name == "fft_rows_radix8" {
      function = try library.makeFunction(name: name, constantValues: MTLFunctionConstantValues())
    } else {
      guard let found = library.makeFunction(name: name) else {
        throw MetalDirectPtychoError.missingFunction(name)
      }
      function = found
    }
    do {
      return try device.makeComputePipelineState(function: function)
    } catch {
      throw MetalDirectPtychoError.libraryCompilation(
        "pipeline creation failed for \(name)"
      )
    }
  }

}
