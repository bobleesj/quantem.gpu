import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMKernels
import Native4DSTEMIO

/// What a scheduled polar-index preparation did, so a caller can charge memory
/// exactly for the state the resident ended in.
@_spi(PairedRuntimeTANSPrototype)
public enum ResidentDetectorIndexPreparationOutcome: Sendable, Equatable {
  /// The index is installed and costs `addedBytes` more device memory.
  case installed(addedBytes: UInt64, buildSeconds: Double)
  /// An index was already present; nothing was allocated.
  case alreadyPrepared
  /// The build completed while the resident or its index was being torn down.
  case superseded
  case cancelled
  /// The build failed; the resident keeps serving the exact un-indexed path.
  case failed(String)
}

/// Exact original-HDF5 resident backed by the paired runtime tANS ABI.
public final class MetalPairedRuntimeTANSResidentSource: @unchecked Sendable {
  private let configuration: PairedRuntimeConfiguration
  private let sourceDataset: Native4DSTEMDataset
  private let sourceLibrary: MTLLibrary
  public var interactionMode: MetalResidentInteractionMode? { configuration.mode }
  private var residentIndexPrepared = false
  /// Bumped whenever the resident or its index is torn down. A build that
  /// started under an older generation refuses to install its result, which is
  /// what makes an index build safe to run with `stateLock` released.
  private var residentIndexGeneration: UInt64 = 0
  /// Background queue for index builds that must not extend a load or a frame.
  private static let residentDetectorIndexQueue = DispatchQueue(
    label: "quantem.gpu.paired-runtime-polar-index", qos: .utility)
  private func runtimeOption(_ name: String) -> String? {
    if residentIndexPrepared {
      if name == "QGPU_PAIRED_RUNTIME_POLAR_INDEX" { return "1" }
      if name == "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT" { return "scan512" }
    }
    return configuration.value(name)
  }
  private func runtimeOptionIsExplicit(_ name: String) -> Bool { configuration.isExplicit(name) }
  public let shape: [Int]
  public let logicalDtype: Metal4DSTEMIntegerDType
  public let sourceIdentitySHA256: String
  @_spi(PairedRuntimeTANSPrototype) public let compactOffsetsEnabled: Bool
  @_spi(PairedRuntimeTANSPrototype) public let macroLookaheadBits: Int
  public private(set) var loadMetrics: MetalPairedRuntimeTANSBuildMetrics
  public let dpcMoments: MetalCompactH5ExactDPCMoments
  private var released = false
  /// Stream rank of each detector pixel within a packet, or nil when streams follow pixel order.
  private let streamRankOfPixel: [UInt32]?
  /// Detector pixel stored at each stream rank, or nil when streams follow pixel order.
  private let pixelOfStreamRank: [UInt32]?
  @_spi(PairedRuntimeTANSPrototype) public var streamOrder: String {
    streamRankOfPixel == nil ? "pixel" : "radial1"
  }
  public var isReleased: Bool {
    metadataLock.lock()
    defer { metadataLock.unlock() }
    return cachedReleased
  }

  private var payload: MTLBuffer!
  private var offsets: MTLBuffer!
  private var modes: MTLBuffer!
  private var decodingTable: MTLBuffer!
  private var macroDecodingTable: MTLBuffer?
  private var polarIndex: MetalPairedRuntimeTANSPolarIndex?
  private struct DiagnosticScratch {
    let jointPlanEnabled: Bool
    let previous: [UInt8]
    let target: [UInt8]
    let plan: PairedRuntimeTANSPolarPlan
    let output: MTLBuffer
    let status: MTLBuffer
    let selected: MTLBuffer
    let coefficients: MTLBuffer
    let indexInputs: MetalPairedRuntimeTANSPolarIndex.PreparedInputs?
    var byteCount: Int {
      output.length + status.length + selected.length + coefficients.length
        + (indexInputs?.byteCount ?? 0)
    }
  }
  private var diagnosticScratch: DiagnosticScratch?
  private let stateLock = NSLock()
  private let metadataLock = NSLock()
  private var cachedReleased = false
  private var cachedResidentBytes: UInt64 = 0
  private var cachedPolarFieldCount = 0
  private var cachedPolarResidualCount = 0
  private var cachedHistoryHit = false
  private var cachedHistoryBase = false
  private var cachedUpdateProfile: [String: Double] = [:]
  private var cachedPolarIndexBytes: UInt64 = 0
  private var cachedPolarIndexBuildMilliseconds: Double = 0
  private var polarFieldCount = 0
  private var polarResidualCount = 0
  @_spi(PairedRuntimeTANSPrototype) public var lastPolarFieldCount: Int {
    metadataLock.lock()
    defer { metadataLock.unlock() }
    return cachedPolarFieldCount
  }
  @_spi(PairedRuntimeTANSPrototype) public var lastPolarResidualCount: Int {
    metadataLock.lock()
    defer { metadataLock.unlock() }
    return cachedPolarResidualCount
  }
  @_spi(PairedRuntimeTANSPrototype) public var polarIndexBytes: UInt64 {
    metadataLock.lock()
    defer { metadataLock.unlock() }
    return cachedPolarIndexBytes
  }
  @_spi(PairedRuntimeTANSPrototype) public var polarIndexBuildMilliseconds: Double {
    metadataLock.lock()
    defer { metadataLock.unlock() }
    return cachedPolarIndexBuildMilliseconds
  }
  @_spi(PairedRuntimeTANSPrototype) public var polarScan512QueryPipelinePrepared: Bool {
    polarIndex?.scan512QueryPipelinePrepared ?? false
  }
  @_spi(PairedRuntimeTANSPrototype) public var compactOffsetBytes: Int {
    compactOffsetsEnabled ? (offsets?.length ?? 0) : 0
  }
  @_spi(PairedRuntimeTANSPrototype) public var macroTableBytes: Int {
    macroDecodingTable?.length ?? 0
  }
  @_spi(PairedRuntimeTANSPrototype) public var lastHistoryHit: Bool {
    metadataLock.lock()
    defer { metadataLock.unlock() }
    return cachedHistoryHit
  }
  @_spi(PairedRuntimeTANSPrototype) public var lastHistoryBase: Bool {
    metadataLock.lock()
    defer { metadataLock.unlock() }
    return cachedHistoryBase
  }
  @_spi(PairedRuntimeTANSPrototype) public var lastUpdateProfile: [String: Double] {
    metadataLock.lock()
    defer { metadataLock.unlock() }
    return cachedUpdateProfile
  }
  private let queue: MTLCommandQueue
  private let detectorProfiler: PairedRuntimeDetectorProfiler?
  private let selectedDPPipeline: MTLComputePipelineState
  private let detectorPipeline: MTLComputePipelineState
  private let detectorPacketOwner2Pipeline: MTLComputePipelineState
  private let detectorSIMDEntropyFastPathPipeline: MTLComputePipelineState?
  private let detectorBranchlessPopPipeline: MTLComputePipelineState?
  private let detectorRefillPipelines: [Int: MTLComputePipelineState]
  private let detectorPhasedReadersPipeline: MTLComputePipelineState?
  private let detectorPairUnrollPipelines: [Int: MTLComputePipelineState]
  private let detectorDecodeChecksumPipeline: MTLComputePipelineState?
  private let detectorLazyRefillPipeline: MTLComputePipelineState?
  private let detectorTrustedTablePipeline: MTLComputePipelineState?
  private let detectorTrustedPlainSumsPipeline: MTLComputePipelineState?
  private let detectorTrustedRegisterSumsPipeline: MTLComputePipelineState?
  private let detectorTrustedTableSplit4Pipeline: MTLComputePipelineState?
  private let detectorTrustedTableSplit8Pipeline: MTLComputePipelineState?
  private let detectorVectorPairReductionPipeline: MTLComputePipelineState?
  private let detectorTrustedVectorPairReductionPipeline: MTLComputePipelineState?
  private let detectorTrustedWindowReaderPipeline: MTLComputePipelineState?
  private let detectorTrustedWindowReaderPlainPipeline: MTLComputePipelineState?
  private let detectorWindowDiagPipelines: [Int: MTLComputePipelineState]
  private let detectorTrustedSetupWindowPipeline: MTLComputePipelineState?
  private let detectorTrustedSetupHeaderDiagPipeline: MTLComputePipelineState?
  private let detectorTrustedSetupEventRowsPipeline: MTLComputePipelineState?
  private let detectorTrustedSetupAdjacentPipeline: MTLComputePipelineState?
  private let detectorTrustedSetupFlatEventsPipeline: MTLComputePipelineState?
  private let detectorTrustedSetupCompactPairsPipeline: MTLComputePipelineState?
  private let detectorTrustedSetupCompactQuadsPipeline: MTLComputePipelineState?
  private let detectorPacketOwner4Pipeline: MTLComputePipelineState
  private let detectorSplitPipelines: [Int: MTLComputePipelineState]
  private let detectorReuseWordPipeline: MTLComputePipelineState?
  private let detectorRegisterSumsPipeline: MTLComputePipelineState?
  private let detectorPlainSumsPipeline: MTLComputePipelineState?
  private let detectorMacroPipeline: MTLComputePipelineState?
  private let detectorReader32Pipeline: MTLComputePipelineState?
  private let detectorCooperativePipeline: MTLComputePipelineState?
  private let detectorSparseScatterPipeline: MTLComputePipelineState?
  private let detectorDenseCompactionPipeline: MTLComputePipelineState?
  private let detectorPartialsPipeline: MTLComputePipelineState
  private let detectorPartialStoresPipeline: MTLComputePipelineState?
  private let detectorFinishPipeline: MTLComputePipelineState
  private var failure: MTLBuffer!
  private var diffraction: MTLBuffer!
  private var detectorProduct: MTLBuffer!
  private var pendingCopyDestination: MTLBuffer?
  private let historyEnabled: Bool
  private var historyProduct: MTLBuffer?
  private var historyMask: [UInt8]?
  private var historyValid = false
  private var historyPolarFieldCount = 0
  private var historyPolarResidualCount = 0
  private var detectorPartials: MTLBuffer?
  private var detectorMask: [UInt8]
  // Block stride state: rows in 512-scan block b last applied blockPhaseMasks[b % blockStride].
  private var blockStride = 1
  private var blockPhaseMasks: [[UInt8]] = []
  private var activePacketStride = 1
  private var activePacketPhase = 0
  private let validPixels: [UInt8]

  public var residentBytes: UInt64 {
    metadataLock.lock()
    defer { metadataLock.unlock() }
    return cachedResidentBytes
  }

  /// Binary detector-validity mask used by derived products; raw DPs stay untouched.
  public var detectorValidityMask: [UInt8] {
    validPixels
  }

  /// Decode and encode one complete native acquisition directly into paired tANS.
  public static func supports(source: Native4DSTEMIndexedSource) -> Bool {
    source.dataset.scanRows == PairedRuntimeTANSRecordABI.scanRows
      && source.dataset.scanCols == PairedRuntimeTANSRecordABI.scanColumns
      && source.dataset.detectorRows == PairedRuntimeTANSRecordABI.detectorRows
      && source.dataset.detectorCols == PairedRuntimeTANSRecordABI.detectorColumns
      && source.logicalFrameCount
        == PairedRuntimeTANSRecordABI.recordScans
        * PairedRuntimeTANSRecordABI.recordsPerAcquisition
      && (source.sourceBytesPerValue == 1 || source.sourceBytesPerValue == 2)
  }

  /// Decode and encode one complete native acquisition directly into paired tANS.
  public static func load(
    source: Native4DSTEMIndexedSource, device: MTLDevice,
    maximumAdditionalBytes: UInt64? = nil,
    interaction: MetalResidentInteractionMode? = nil,
    shouldCancel: () -> Bool = { false },
    progress: (Int, Int) -> Void = { _, _ in }
  ) throws -> MetalPairedRuntimeTANSResidentSource {
    let started = CFAbsoluteTimeGetCurrent()
    let configuration = PairedRuntimeConfiguration(mode: interaction)
    let allocatedBefore = UInt64(device.currentAllocatedSize)
    let allocationLimit = min(
      device.recommendedMaxWorkingSetSize,
      maximumAdditionalBytes.map {
        allocatedBefore.addingReportingOverflow($0).overflow
          ? UInt64.max : allocatedBefore + $0
      } ?? UInt64.max)
    let result = try MetalPairedRuntimeTANSHDF5Builder.build(
      source: source, device: device, maximumAdditionalBytes: maximumAdditionalBytes,
      configuration: configuration,
      shouldCancel: shouldCancel, progress: progress)
    // Consolidation temporarily retains the encoded records and their compact
    // copy. Check that peak before allocating, not after publishing the resident.
    let allocated = UInt64(device.currentAllocatedSize)
    let consolidationReserve = result.metrics.residentBytes + (UInt64(32) << 20)
    guard allocated <= allocationLimit, consolidationReserve <= allocationLimit - allocated else {
      throw Self.invalid(
        "Resident consolidation exceeds the available memory budget. Use Normal or select fewer tilts."
      )
    }
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    return try MetalPairedRuntimeTANSResidentSource(
      dataset: source.dataset, provider: result.provider, streamPixels: result.streamPixels,
      dpcMoments: result.dpcMoments,
      metrics: result.metrics,
      loadStarted: started, device: device, configuration: configuration,
      allocationLimit: allocationLimit, shouldCancel: shouldCancel)
  }

  init(
    dataset: Native4DSTEMDataset, provider: PairedRuntimeTANSRecordProvider,
    streamPixels: [UInt32]?,
    dpcMoments: MetalCompactH5ExactDPCMoments,
    metrics: MetalPairedRuntimeTANSBuildMetrics, loadStarted: Double, device: MTLDevice,
    configuration: PairedRuntimeConfiguration, allocationLimit: UInt64,
    shouldCancel: () -> Bool
  ) throws {
    self.configuration = configuration
    func runtimeOption(_ name: String) -> String? { configuration.value(name) }
    func runtimeOptionIsExplicit(_ name: String) -> Bool { configuration.isExplicit(name) }
    guard let identity = dataset.sourceIdentitySHA256,
      let queue = device.makeCommandQueue()
    else { throw Self.invalid("Paired-runtime resident initialization failed") }
    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: device)
    sourceDataset = dataset
    sourceLibrary = library
    var initialStreamRanks: [UInt32]? = nil
    if let streamPixels {
      var ranks = [UInt32](repeating: UInt32.max, count: streamPixels.count)
      for (rank, pixel) in streamPixels.enumerated() {
        guard Int(pixel) < ranks.count, ranks[Int(pixel)] == UInt32.max else {
          throw Self.invalid("Paired-runtime stream order is not a permutation")
        }
        ranks[Int(pixel)] = UInt32(rank)
      }
      initialStreamRanks = ranks
    }
    streamRankOfPixel = initialStreamRanks
    pixelOfStreamRank = streamPixels
    let compactOffsetSetting = runtimeOption("QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS") == "1"
    compactOffsetsEnabled = compactOffsetSetting
    let macroEnabledAtInitialization = runtimeOption("QGPU_PAIRED_RUNTIME_MACRO") == "1"
    let macroLookaheadText = runtimeOption("QGPU_PAIRED_RUNTIME_MACRO_LOOKAHEAD_BITS") ?? "4"
    guard let preparedMacroLookaheadBits = Int(macroLookaheadText),
      preparedMacroLookaheadBits == 2 || preparedMacroLookaheadBits == 4
    else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_MACRO_LOOKAHEAD_BITS must be 2 or 4")
    }
    macroLookaheadBits = macroEnabledAtInitialization ? preparedMacroLookaheadBits : 0
    let makeFunctionConstants: () -> MTLFunctionConstantValues = {
      let values = MTLFunctionConstantValues()
      var enabled = compactOffsetSetting
      values.setConstantValue(
        &enabled, type: .bool,
        index: Metal4DSTEMKernels.pairedRuntimeTANSCompactOffsetsFunctionConstantIndex)
      return values
    }
    let defaults = makeFunctionConstants()
    let selectedDP = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSSelectedDPFunction,
      constantValues: defaults)
    let detector = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwnerFunction,
      constantValues: defaults)
    let detectorPartials = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPartialsFunction,
      constantValues: defaults)
    let detectorFinish = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorFinishFunction,
      constantValues: defaults)
    self.queue = queue
    detectorProfiler = PairedRuntimeDetectorProfiler.makeIfRequested(device: device)
    selectedDPPipeline = try device.makeComputePipelineState(function: selectedDP)
    detectorPipeline = try device.makeComputePipelineState(function: detector)
    var twoStreams = UInt32(2)
    let twoConstants = makeFunctionConstants()
    twoConstants.setConstantValue(&twoStreams, type: .uint, index: 0)
    let detectorPacketOwner2 = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: twoConstants)
    detectorPacketOwner2Pipeline = try device.makeComputePipelineState(
      function: detectorPacketOwner2)
    let prepareSIMDEntropyFastPathValue =
      runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_SIMD_ENTROPY_FAST_PATH") ?? "0"
    guard prepareSIMDEntropyFastPathValue == "0" || prepareSIMDEntropyFastPathValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_PREPARE_SIMD_ENTROPY_FAST_PATH must be 0 or 1")
    }
    if prepareSIMDEntropyFastPathValue == "1" {
      let constants = makeFunctionConstants()
      var enabled = true
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(
        &enabled, type: .bool,
        index: Metal4DSTEMKernels.pairedRuntimeTANSSIMDEntropyFastPathFunctionConstantIndex)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorSIMDEntropyFastPathPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorSIMDEntropyFastPathPipeline = nil
    }
    if runtimeOption("QGPU_PREPARE_BRANCHLESS_POP") == "1" {
      let constants = makeFunctionConstants()
      var enabled = true
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(&enabled, type: .bool, index: 16)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorBranchlessPopPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorBranchlessPopPipeline = nil
    }
    var refillPipelines: [Int: MTLComputePipelineState] = [:]
    if runtimeOption("QGPU_PREPARE_REFILL_THRESHOLDS") == "1" {
      for threshold in [16, 24] {
        let constants = makeFunctionConstants()
        var refillThreshold = UInt32(threshold)
        constants.setConstantValue(&twoStreams, type: .uint, index: 0)
        constants.setConstantValue(&refillThreshold, type: .uint, index: 17)
        let function = try library.makeFunction(
          name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
          constantValues: constants)
        refillPipelines[threshold] = try device.makeComputePipelineState(function: function)
      }
    }
    detectorRefillPipelines = refillPipelines
    if runtimeOption("QGPU_PREPARE_PHASED_READERS") == "1" {
      let constants = makeFunctionConstants()
      var enabled = true
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(&enabled, type: .bool, index: 18)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorPhasedReadersPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorPhasedReadersPipeline = nil
    }
    var pairUnrollPipelines: [Int: MTLComputePipelineState] = [:]
    if runtimeOption("QGPU_PREPARE_PAIR_UNROLL") == "1" {
      for factor in [2, 4, 8] {
        let constants = makeFunctionConstants()
        var pairUnroll = UInt32(factor)
        constants.setConstantValue(&twoStreams, type: .uint, index: 0)
        constants.setConstantValue(&pairUnroll, type: .uint, index: 19)
        let function = try library.makeFunction(
          name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
          constantValues: constants)
        pairUnrollPipelines[factor] = try device.makeComputePipelineState(function: function)
      }
    }
    detectorPairUnrollPipelines = pairUnrollPipelines
    if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_DECODE_CHECKSUM") == "1" {
      var checksum = true
      let checksumConstants = makeFunctionConstants()
      checksumConstants.setConstantValue(&twoStreams, type: .uint, index: 0)
      checksumConstants.setConstantValue(
        &checksum, type: .bool, index: 14)  // FC14: diagnostic decode checksum
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: checksumConstants)
      detectorDecodeChecksumPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorDecodeChecksumPipeline = nil
    }
    if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_LAZY_REFILL") == "1" {
      var lazyRefill = true
      let constants = makeFunctionConstants()
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(&lazyRefill, type: .bool, index: 15)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorLazyRefillPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorLazyRefillPipeline = nil
    }
    if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE") == "1" {
      // The provider is an internal builder product whose table is uploaded
      // from this deterministic factory; prove every transition before using
      // the FC13 fast path. Arbitrary length-valid tables are not trusted.
      try Self.validateTrustedTable(
        provider: provider, queue: queue, device: device)
      var trusted = true
      let constants = makeFunctionConstants()
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(&trusted, type: .bool, index: 13)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorTrustedTablePipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorTrustedTablePipeline = nil
    }
    let prepareWindowReaderValue = runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_WINDOW_READER") ?? "0"
    guard prepareWindowReaderValue == "0" || prepareWindowReaderValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_PREPARE_WINDOW_READER must be 0 or 1")
    }
    if prepareWindowReaderValue == "1" {
      // FC28 relies on the FC13 proof (bits <= 10, base + 2^bits <= 1024) for
      // its 64-bit window coverage and table bounds, so it is prepared only
      // after the trusted table above has been validated.
      guard detectorTrustedTablePipeline != nil else {
        throw Self.invalid(
          "Prepare the window-reader pipeline only with QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE=1"
        )
      }
      let cadenceText = runtimeOption("QGPU_PAIRED_RUNTIME_WINDOW_READER_CADENCE") ?? "3"
      guard var cadence = UInt32(cadenceText), (1...3).contains(cadence) else {
        throw Self.invalid("QGPU_PAIRED_RUNTIME_WINDOW_READER_CADENCE must be 1, 2, or 3")
      }
      var trusted = true
      var windowReader = true
      let constants = makeFunctionConstants()
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(&trusted, type: .bool, index: 13)
      constants.setConstantValue(
        &windowReader, type: .bool,
        index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderFunctionConstantIndex)
      constants.setConstantValue(
        &cadence, type: .uint,
        index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderCadenceFunctionConstantIndex)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorTrustedWindowReaderPipeline = try device.makeComputePipelineState(function: function)
      if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_WINDOW_READER_PLAIN") == "1" {
        // FC28 + FC12: lane-0 plain partial adds instead of atomics.
        var plain = true
        constants.setConstantValue(&plain, type: .bool, index: 12)
        let plainFunction = try library.makeFunction(
          name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
          constantValues: constants)
        detectorTrustedWindowReaderPlainPipeline = try device.makeComputePipelineState(
          function: plainFunction)
      } else {
        detectorTrustedWindowReaderPlainPipeline = nil
      }
      var diagnostics: [Int: MTLComputePipelineState] = [:]
      if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_WINDOW_DIAG") == "1" {
        for level in [1, 2, 3, 4, 6] {
          let diagnostic = makeFunctionConstants()
          var trustedDiag = true
          var windowDiag = true
          var cadenceDiag = cadence
          var levelValue = UInt32(level)
          diagnostic.setConstantValue(&twoStreams, type: .uint, index: 0)
          diagnostic.setConstantValue(&trustedDiag, type: .bool, index: 13)
          diagnostic.setConstantValue(
            &windowDiag, type: .bool,
            index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderFunctionConstantIndex)
          diagnostic.setConstantValue(
            &cadenceDiag, type: .uint,
            index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderCadenceFunctionConstantIndex)
          diagnostic.setConstantValue(&levelValue, type: .uint, index: 40)
          let function = try library.makeFunction(
            name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
            constantValues: diagnostic)
          diagnostics[level] = try device.makeComputePipelineState(function: function)
        }
      }
      detectorWindowDiagPipelines = diagnostics
      if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_SETUP") == "1" {
        let trustedSetupConstants = makeFunctionConstants()
        var trustedValue = true
        var windowValue = true
        var cadenceValue = cadence
        var trustedSetupValue = true
        trustedSetupConstants.setConstantValue(&twoStreams, type: .uint, index: 0)
        trustedSetupConstants.setConstantValue(&trustedValue, type: .bool, index: 13)
        trustedSetupConstants.setConstantValue(
          &windowValue, type: .bool,
          index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderFunctionConstantIndex)
        trustedSetupConstants.setConstantValue(
          &cadenceValue, type: .uint,
          index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderCadenceFunctionConstantIndex)
        trustedSetupConstants.setConstantValue(&trustedSetupValue, type: .bool, index: 41)
        let trustedSetupFunction = try library.makeFunction(
          name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
          constantValues: trustedSetupConstants)
        detectorTrustedSetupWindowPipeline = try device.makeComputePipelineState(
          function: trustedSetupFunction)
        if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_COMPACT_PAIRS") == "1" {
          let pairConstants = makeFunctionConstants()
          var pairTrusted = true
          var pairWindow = true
          var pairCadence = cadence
          var pairTrustedSetup = true
          var compactPairs = true
          pairConstants.setConstantValue(&twoStreams, type: .uint, index: 0)
          pairConstants.setConstantValue(&pairTrusted, type: .bool, index: 13)
          pairConstants.setConstantValue(
            &pairWindow, type: .bool,
            index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderFunctionConstantIndex)
          pairConstants.setConstantValue(
            &pairCadence, type: .uint,
            index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderCadenceFunctionConstantIndex)
          pairConstants.setConstantValue(&pairTrustedSetup, type: .bool, index: 41)
          pairConstants.setConstantValue(&compactPairs, type: .bool, index: 46)
          detectorTrustedSetupCompactPairsPipeline = try device.makeComputePipelineState(
            function: try library.makeFunction(
              name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
              constantValues: pairConstants))
          var compactQuads = true
          pairConstants.setConstantValue(&compactQuads, type: .bool, index: 47)
          detectorTrustedSetupCompactQuadsPipeline = try device.makeComputePipelineState(
            function: try library.makeFunction(
              name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
              constantValues: pairConstants))
        } else {
          detectorTrustedSetupCompactPairsPipeline = nil
          detectorTrustedSetupCompactQuadsPipeline = nil
        }
        if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_FLAT_EVENTS") == "1" {
          let flatConstants = makeFunctionConstants()
          var flatTrusted = true
          var flatWindow = true
          var flatCadence = cadence
          var flatTrustedSetup = true
          var flatEvents = true
          flatConstants.setConstantValue(&twoStreams, type: .uint, index: 0)
          flatConstants.setConstantValue(&flatTrusted, type: .bool, index: 13)
          flatConstants.setConstantValue(
            &flatWindow, type: .bool,
            index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderFunctionConstantIndex)
          flatConstants.setConstantValue(
            &flatCadence, type: .uint,
            index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderCadenceFunctionConstantIndex)
          flatConstants.setConstantValue(&flatTrustedSetup, type: .bool, index: 41)
          flatConstants.setConstantValue(&flatEvents, type: .bool, index: 44)
          detectorTrustedSetupFlatEventsPipeline = try device.makeComputePipelineState(
            function: try library.makeFunction(
              name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
              constantValues: flatConstants))
        } else {
          detectorTrustedSetupFlatEventsPipeline = nil
        }
        if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_ADJACENT_LANE_STREAMS") == "1" {
          let adjacentConstants = makeFunctionConstants()
          var adjacentTrusted = true
          var adjacentWindow = true
          var adjacentCadence = cadence
          var adjacentTrustedSetup = true
          var adjacentStreams = true
          adjacentConstants.setConstantValue(&twoStreams, type: .uint, index: 0)
          adjacentConstants.setConstantValue(&adjacentTrusted, type: .bool, index: 13)
          adjacentConstants.setConstantValue(
            &adjacentWindow, type: .bool,
            index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderFunctionConstantIndex)
          adjacentConstants.setConstantValue(
            &adjacentCadence, type: .uint,
            index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderCadenceFunctionConstantIndex)
          adjacentConstants.setConstantValue(&adjacentTrustedSetup, type: .bool, index: 41)
          adjacentConstants.setConstantValue(&adjacentStreams, type: .bool, index: 43)
          detectorTrustedSetupAdjacentPipeline = try device.makeComputePipelineState(
            function: try library.makeFunction(
              name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
              constantValues: adjacentConstants))
        } else {
          detectorTrustedSetupAdjacentPipeline = nil
        }
        if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_EVENT_ROWS") == "1" {
          let eventRowsConstants = makeFunctionConstants()
          var eventTrusted = true
          var eventWindow = true
          var eventCadence = cadence
          var eventTrustedSetup = true
          var eventRows = true
          eventRowsConstants.setConstantValue(&twoStreams, type: .uint, index: 0)
          eventRowsConstants.setConstantValue(&eventTrusted, type: .bool, index: 13)
          eventRowsConstants.setConstantValue(
            &eventWindow, type: .bool,
            index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderFunctionConstantIndex)
          eventRowsConstants.setConstantValue(
            &eventCadence, type: .uint,
            index: Metal4DSTEMKernels.pairedRuntimeTANSWindowReaderCadenceFunctionConstantIndex)
          eventRowsConstants.setConstantValue(&eventTrustedSetup, type: .bool, index: 41)
          eventRowsConstants.setConstantValue(&eventRows, type: .bool, index: 30)
          detectorTrustedSetupEventRowsPipeline = try device.makeComputePipelineState(
            function: try library.makeFunction(
              name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
              constantValues: eventRowsConstants))
        } else {
          detectorTrustedSetupEventRowsPipeline = nil
        }
        if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_WINDOW_DIAG") == "1" {
          var headerDiagLevel = UInt32(7)
          trustedSetupConstants.setConstantValue(&headerDiagLevel, type: .uint, index: 40)
          let headerDiagFunction = try library.makeFunction(
            name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
            constantValues: trustedSetupConstants)
          detectorTrustedSetupHeaderDiagPipeline = try device.makeComputePipelineState(
            function: headerDiagFunction)
        } else {
          detectorTrustedSetupHeaderDiagPipeline = nil
        }
      } else {
        detectorTrustedSetupWindowPipeline = nil
        detectorTrustedSetupHeaderDiagPipeline = nil
        detectorTrustedSetupEventRowsPipeline = nil
        detectorTrustedSetupAdjacentPipeline = nil
        detectorTrustedSetupFlatEventsPipeline = nil
        detectorTrustedSetupCompactPairsPipeline = nil
        detectorTrustedSetupCompactQuadsPipeline = nil
      }
    } else {
      detectorTrustedSetupWindowPipeline = nil
      detectorTrustedSetupHeaderDiagPipeline = nil
      detectorTrustedSetupEventRowsPipeline = nil
      detectorTrustedSetupAdjacentPipeline = nil
      detectorTrustedSetupFlatEventsPipeline = nil
      detectorTrustedSetupCompactPairsPipeline = nil
      detectorTrustedSetupCompactQuadsPipeline = nil
      detectorTrustedWindowReaderPipeline = nil
      detectorTrustedWindowReaderPlainPipeline = nil
      detectorWindowDiagPipelines = [:]
    }
    // Compositions of the validated trusted table with the plain-sums and
    // register-sums reductions (function constants 12 and 11).
    if detectorTrustedTablePipeline != nil,
      runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_PLAIN_SUMS") == "1"
    {
      var trusted = true
      var plain = true
      let constants = makeFunctionConstants()
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(&trusted, type: .bool, index: 13)
      constants.setConstantValue(&plain, type: .bool, index: 12)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorTrustedPlainSumsPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorTrustedPlainSumsPipeline = nil
    }
    if detectorTrustedTablePipeline != nil,
      runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_REGISTER_SUMS") == "1"
    {
      var trusted = true
      var register = true
      let constants = makeFunctionConstants()
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(&trusted, type: .bool, index: 13)
      constants.setConstantValue(&register, type: .bool, index: 11)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorTrustedRegisterSumsPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorTrustedRegisterSumsPipeline = nil
    }
    let prepareTrustedTableSplit4Value =
      runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE_PACKET_SPLIT4") ?? "0"
    guard prepareTrustedTableSplit4Value == "0" || prepareTrustedTableSplit4Value == "1" else {
      throw Self.invalid(
        "QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE_PACKET_SPLIT4 must be 0 or 1")
    }
    guard
      prepareTrustedTableSplit4Value == "0"
        || (runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE") == "1"
          && runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_PACKET_SPLITS") == "1")
    else {
      throw Self.invalid(
        "Prepare the trusted-table split-4 pipeline only with trusted-table and packet-split preparation enabled"
      )
    }
    if prepareTrustedTableSplit4Value == "1" {
      // This composition is intentionally explicit and experimental. Validate
      // the factory-produced table before compiling a detector pipeline with
      // both FC9 (packet split) and FC13 (trusted state bound) enabled.
      try Self.validateTrustedTable(provider: provider, queue: queue, device: device)
      var split4 = UInt32(4)
      var trusted = true
      let constants = makeFunctionConstants()
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(&split4, type: .uint, index: 9)
      constants.setConstantValue(&trusted, type: .bool, index: 13)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorTrustedTableSplit4Pipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorTrustedTableSplit4Pipeline = nil
    }
    let prepareTrustedTableSplit8Value =
      runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE_PACKET_SPLIT8") ?? "0"
    guard prepareTrustedTableSplit8Value == "0" || prepareTrustedTableSplit8Value == "1" else {
      throw Self.invalid(
        "QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE_PACKET_SPLIT8 must be 0 or 1")
    }
    guard
      prepareTrustedTableSplit8Value == "0"
        || (runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE") == "1"
          && runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_PACKET_SPLITS") == "1")
    else {
      throw Self.invalid(
        "Prepare the trusted-table split-8 pipeline only with trusted-table and packet-split preparation enabled"
      )
    }
    if prepareTrustedTableSplit8Value == "1" {
      // Keep this composition explicit and experimental. The trusted-table
      // fast path is used only with the factory-validated transition table.
      try Self.validateTrustedTable(provider: provider, queue: queue, device: device)
      var split8 = UInt32(8)
      var trusted = true
      let constants = makeFunctionConstants()
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(&split8, type: .uint, index: 9)
      constants.setConstantValue(&trusted, type: .bool, index: 13)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorTrustedTableSplit8Pipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorTrustedTableSplit8Pipeline = nil
    }
    let prepareVectorPairReductionValue =
      runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_VECTOR_PAIR_REDUCTION") ?? "0"
    guard prepareVectorPairReductionValue == "0" || prepareVectorPairReductionValue == "1" else {
      throw Self.invalid(
        "QGPU_PAIRED_RUNTIME_PREPARE_VECTOR_PAIR_REDUCTION must be 0 or 1")
    }
    if prepareVectorPairReductionValue == "1" {
      let constants = makeFunctionConstants()
      var vectorReduction = true
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(
        &vectorReduction, type: .bool,
        index: Metal4DSTEMKernels.pairedRuntimeTANSVectorPairReductionFunctionConstantIndex)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorVectorPairReductionPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorVectorPairReductionPipeline = nil
    }
    if prepareVectorPairReductionValue == "1"
      && runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE") == "1"
    {
      var trusted = true
      var vectorReduction = true
      let constants = makeFunctionConstants()
      constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      constants.setConstantValue(&trusted, type: .bool, index: 13)
      constants.setConstantValue(
        &vectorReduction, type: .bool,
        index: Metal4DSTEMKernels.pairedRuntimeTANSVectorPairReductionFunctionConstantIndex)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorTrustedVectorPairReductionPipeline = try device.makeComputePipelineState(
        function: function)
    } else {
      detectorTrustedVectorPairReductionPipeline = nil
    }
    if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_REUSE_WORD") == "1" {
      let constants = makeFunctionConstants()
      var streams: UInt32 = 2
      var enabled = true
      constants.setConstantValue(&streams, type: .uint, index: 0)
      constants.setConstantValue(&enabled, type: .bool, index: 10)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorReuseWordPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorReuseWordPipeline = nil
    }
    if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_REGISTER_SUMS") == "1" {
      let constants = makeFunctionConstants()
      var streams: UInt32 = 2
      var enabled = true
      constants.setConstantValue(&streams, type: .uint, index: 0)
      constants.setConstantValue(&enabled, type: .bool, index: 11)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorRegisterSumsPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorRegisterSumsPipeline = nil
    }
    if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_PLAIN_SUMS") == "1" {
      let constants = makeFunctionConstants()
      var streams: UInt32 = 2
      var enabled = true
      constants.setConstantValue(&streams, type: .uint, index: 0)
      constants.setConstantValue(&enabled, type: .bool, index: 12)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: constants)
      detectorPlainSumsPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorPlainSumsPipeline = nil
    }
    var splitPipelines: [Int: MTLComputePipelineState] = [:]
    if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_PACKET_SPLITS") == "1" {
      for count in [2, 4, 8] {
        var split = UInt32(count)
        let constants = makeFunctionConstants()
        constants.setConstantValue(&twoStreams, type: .uint, index: 0)
        constants.setConstantValue(&split, type: .uint, index: 9)
        let function = try library.makeFunction(
          name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
          constantValues: constants)
        splitPipelines[count] = try device.makeComputePipelineState(function: function)
      }
    }
    detectorSplitPipelines = splitPipelines
    if runtimeOption("QGPU_PAIRED_RUNTIME_READER32") == "1" {
      var reader32Enabled = true
      let reader32Constants = makeFunctionConstants()
      reader32Constants.setConstantValue(&twoStreams, type: .uint, index: 0)
      reader32Constants.setConstantValue(&reader32Enabled, type: .bool, index: 3)
      let detectorReader32 = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: reader32Constants)
      detectorReader32Pipeline = try device.makeComputePipelineState(function: detectorReader32)
    } else {
      detectorReader32Pipeline = nil
    }
    if runtimeOption("QGPU_PAIRED_RUNTIME_COOPERATIVE") == "1" {
      let cooperativeConstants = makeFunctionConstants()
      let detectorCooperative = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorCooperativeFunction,
        constantValues: cooperativeConstants)
      detectorCooperativePipeline = try device.makeComputePipelineState(
        function: detectorCooperative)
    } else {
      detectorCooperativePipeline = nil
    }
    if runtimeOption("QGPU_PAIRED_RUNTIME_MACRO") == "1" {
      var macroEnabled = true
      var macroLookaheadBits = UInt32(preparedMacroLookaheadBits)
      let macroConstants = makeFunctionConstants()
      macroConstants.setConstantValue(&twoStreams, type: .uint, index: 0)
      macroConstants.setConstantValue(&macroEnabled, type: .bool, index: 2)
      macroConstants.setConstantValue(&macroLookaheadBits, type: .uint, index: 22)
      let detectorMacro = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
        constantValues: macroConstants)
      detectorMacroPipeline = try device.makeComputePipelineState(function: detectorMacro)
      let ordinary = try PairedRuntimeTANSTables.build().packedDecoding
      let interleaved = try PairedRuntimeTANSMacroTable.build(
        decoding: ordinary, lookaheadBits: preparedMacroLookaheadBits)
      macroDecodingTable = try Self.upload(
        interleaved, device: device, label: "paired-runtime macro decoding")
    } else {
      detectorMacroPipeline = nil
      macroDecodingTable = nil
    }
    var fourStreams = UInt32(4)
    let fourConstants = makeFunctionConstants()
    fourConstants.setConstantValue(&fourStreams, type: .uint, index: 0)
    let detectorPacketOwner4 = try library.makeFunction(
      name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPacketOwner2Function,
      constantValues: fourConstants)
    detectorPacketOwner4Pipeline = try device.makeComputePipelineState(
      function: detectorPacketOwner4)
    if runtimeOption("QGPU_PAIRED_RUNTIME_SPARSE_SPLIT") == "1" {
      let detectorSparseScatter = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorSparseScatterFunction,
        constantValues: makeFunctionConstants())
      detectorSparseScatterPipeline = try device.makeComputePipelineState(
        function: detectorSparseScatter)
    } else {
      detectorSparseScatterPipeline = nil
    }
    if runtimeOption("QGPU_PAIRED_RUNTIME_DENSE_COMPACTION") == "1" {
      var plainScratch =
        runtimeOption("QGPU_PAIRED_RUNTIME_PLAIN_SCRATCH") == "1"
      let denseConstants = makeFunctionConstants()
      denseConstants.setConstantValue(&plainScratch, type: .bool, index: 1)
      let detectorDenseCompaction = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorDenseCompactionFunction,
        constantValues: denseConstants)
      detectorDenseCompactionPipeline = try device.makeComputePipelineState(
        function: detectorDenseCompaction)
    } else {
      detectorDenseCompactionPipeline = nil
    }
    detectorPartialsPipeline = try device.makeComputePipelineState(function: detectorPartials)
    if runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_PARTIAL_STORES") == "1" {
      var enabled = true
      let constants = makeFunctionConstants()
      constants.setConstantValue(&enabled, type: .bool, index: 8)
      let function = try library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSDetectorPartialsFunction,
        constantValues: constants)
      detectorPartialStoresPipeline = try device.makeComputePipelineState(function: function)
    } else {
      detectorPartialStoresPipeline = nil
    }
    detectorFinishPipeline = try device.makeComputePipelineState(function: detectorFinish)
    shape = [
      dataset.scanRows, dataset.scanCols,
      dataset.detectorRows, dataset.detectorCols,
    ]
    logicalDtype = provider.descriptor.logicalDtype
    sourceIdentitySHA256 = identity
    self.dpcMoments = dpcMoments
    let consolidationStarted = CFAbsoluteTimeGetCurrent()
    let consolidated = try Self.consolidate(
      provider: provider, library: library, queue: queue,
      compactOffsetsEnabled: compactOffsetSetting)
    payload = consolidated.payload
    offsets = consolidated.offsets
    modes = consolidated.modes
    decodingTable = provider.decodingTable
    loadMetrics = MetalPairedRuntimeTANSBuildMetrics(
      totalSeconds: CFAbsoluteTimeGetCurrent() - loadStarted,
      fusedDecodeAndSizeSeconds: metrics.fusedDecodeAndSizeSeconds,
      provisionalCPUPrefixSeconds: metrics.provisionalCPUPrefixSeconds,
      compactSeconds: metrics.compactSeconds,
      consolidationSeconds: CFAbsoluteTimeGetCurrent() - consolidationStarted,
      residentBytes: consolidated.residentBytes + UInt64(provider.decodingTable.length))
    let pixels = shape[2] * shape[3]
    failure = try Self.buffer(
      device: device, bytes: 4, options: .storageModeShared,
      label: "paired-runtime query failure")
    diffraction = try Self.buffer(
      device: device, bytes: pixels * 4, options: .storageModeShared,
      label: "paired-runtime diffraction")
    detectorProduct = try Self.buffer(
      device: device, bytes: shape[0] * shape[1] * 4, options: .storageModeShared,
      label: "paired-runtime detector product")
    let historySetting = runtimeOption("QGPU_PAIRED_RUNTIME_HISTORY") ?? "0"
    guard historySetting == "0" || historySetting == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_HISTORY must be 0 or 1")
    }
    historyEnabled = historySetting == "1"
    if historyEnabled {
      historyProduct = try Self.buffer(
        device: device, bytes: shape[0] * shape[1] * 4, options: .storageModeShared,
        label: "paired-runtime detector history")
      memset(historyProduct!.contents(), 0, historyProduct!.length)
      historyMask = [UInt8](repeating: 0, count: pixels)
    } else {
      historyProduct = nil
      historyMask = nil
    }
    detectorMask = [UInt8](repeating: 0, count: pixels)
    var valid = [UInt8](repeating: 1, count: pixels)
    for pixel in dataset.badPixelIndices where valid.indices.contains(pixel) {
      valid[pixel] = 0
    }
    validPixels = valid
    let polarSetting = runtimeOption("QGPU_PAIRED_RUNTIME_POLAR_INDEX") ?? "0"
    guard polarSetting == "0" || polarSetting == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_POLAR_INDEX must be 0 or 1")
    }
    let scan512QueryPrepareSetting =
      runtimeOption("QGPU_PAIRED_RUNTIME_PREPARE_POLAR_QUERY_SCAN512") ?? "0"
    guard scan512QueryPrepareSetting == "0" || scan512QueryPrepareSetting == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_PREPARE_POLAR_QUERY_SCAN512 must be 0 or 1")
    }
    guard scan512QueryPrepareSetting == "0" || polarSetting == "1" else {
      throw Self.invalid(
        "Prepare the scan512 polar-query pipeline only when the polar index is enabled")
    }
    if polarSetting == "1" {
      let leafSetting = runtimeOption("QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS") ?? "64"
      guard let leafPixels = Int(leafSetting), [16, 32, 64].contains(leafPixels) else {
        throw Self.invalid("QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS must be 16, 32, or 64")
      }
      var layoutKind = runtimeOption("QGPU_PAIRED_RUNTIME_POLAR_LAYOUT") ?? "polar"
      if PairedRuntimeTANSPolarPlan.isPaddedLayout(layoutKind), leafPixels != 16,
        !runtimeOptionIsExplicit("QGPU_PAIRED_RUNTIME_POLAR_LAYOUT")
      {
        // The defaulted fine-core layout is defined for 16-pixel leaves only.
        layoutKind = "radial1"
      }
      guard
        ["polar", "radial1", "radialhalf", "radial1core4", "radial1fine4"].contains(layoutKind),
        !PairedRuntimeTANSPolarPlan.isPaddedLayout(layoutKind) || leafPixels == 16
      else {
        throw Self.invalid(
          "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT must be polar, radial1, radialhalf, or radial1core4 (16-pixel leaves)"
        )
      }
      polarIndex = try autoreleasepool {
        try MetalPairedRuntimeTANSPolarIndex(
          device: device, library: library, queue: queue, payload: payload,
          offsets: offsets, modes: modes, decoding: decodingTable,
          validPixels: valid, packets: shape[0] * shape[1] / 512,
          leafPixels: leafPixels, layoutKind: layoutKind,
          streamRankOfPixel: initialStreamRanks,
          compactOffsetsEnabled: compactOffsetSetting,
          prepareScan512QueryPipeline: scan512QueryPrepareSetting == "1",
          allocationLimit: allocationLimit, shouldCancel: shouldCancel)
      }
      let previous = loadMetrics
      loadMetrics = MetalPairedRuntimeTANSBuildMetrics(
        totalSeconds: CFAbsoluteTimeGetCurrent() - loadStarted,
        fusedDecodeAndSizeSeconds: previous.fusedDecodeAndSizeSeconds,
        provisionalCPUPrefixSeconds: previous.provisionalCPUPrefixSeconds,
        compactSeconds: previous.compactSeconds,
        consolidationSeconds: previous.consolidationSeconds,
        residentBytes: previous.residentBytes)
    }
    refreshMetadataSnapshot()
  }

  /// Build only a fine detector index from existing encoded buffers, without file IO.
  /// This experiment preserves Normal encoding; it is not the full Fast profile.
  /// Failure or cancellation leaves the resident and its current image unchanged.
  ///
  /// The build runs with `stateLock` released: the lock is only taken to
  /// snapshot the resident state and again to install the finished index. That
  /// matters because `stateLock` also serializes `extractRawDiffraction` and
  /// every virtual-detector update, so holding it for the ~0.9 s build would
  /// freeze all interaction for the duration of the build.
  @_spi(PairedRuntimeTANSPrototype)
  public func prepareResidentDetectorIndex(
    maximumAdditionalBytes: UInt64, shouldCancel: () -> Bool = { false }
  ) throws {
    guard
      let request = try makeResidentDetectorIndexRequest(
        maximumAdditionalBytes: maximumAdditionalBytes)
    else { return }
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    let candidate = try buildResidentDetectorIndex(request, shouldCancel: shouldCancel)
    if shouldCancel() { throw Metal4DSTEMStreamingIOError.cancelled }
    guard installResidentDetectorIndex(candidate, request: request) else {
      throw Self.invalid(
        "The paired-runtime ANS source changed while its polar index was built")
    }
  }

  /// Build the same index without blocking the caller.
  ///
  /// The load path uses this so opening a folder never waits for an
  /// interaction accelerator. Until the index is installed, every query takes
  /// the shipped un-indexed path, which is already exact; the completion runs
  /// on the private build queue and reports which of those two states a caller
  /// should charge memory for.
  @_spi(PairedRuntimeTANSPrototype)
  public func scheduleResidentDetectorIndexPreparation(
    maximumAdditionalBytes: UInt64,
    shouldCancel: @escaping @Sendable () -> Bool = { false },
    completion: @escaping @Sendable (ResidentDetectorIndexPreparationOutcome) -> Void = { _ in }
  ) {
    let request: ResidentDetectorIndexBuildRequest
    do {
      guard
        let snapshot = try makeResidentDetectorIndexRequest(
          maximumAdditionalBytes: maximumAdditionalBytes)
      else {
        completion(.alreadyPrepared)
        return
      }
      request = snapshot
    } catch {
      completion(.failed(String(describing: error)))
      return
    }
    Self.residentDetectorIndexQueue.async { [self] in
      if shouldCancel() {
        completion(.cancelled)
        return
      }
      let allocatedBefore = UInt64(request.device.currentAllocatedSize)
      let started = CFAbsoluteTimeGetCurrent()
      let candidate: MetalPairedRuntimeTANSPolarIndex
      do {
        // The build's packing chunks each end in `waitUntilCompleted`. On the
        // resident's own queue an interaction command lands behind whichever
        // chunk is in flight, which measured as a 107 ms hitch. A dedicated
        // queue lets Metal schedule the two independently, and every buffer the
        // index keeps is queue-agnostic.
        candidate = try buildResidentDetectorIndex(
          request, queue: request.device.makeCommandQueue(),
          shouldCancel: shouldCancel)
      } catch {
        completion(.failed(String(describing: error)))
        return
      }
      let seconds = CFAbsoluteTimeGetCurrent() - started
      if shouldCancel() {
        completion(.cancelled)
        return
      }
      let allocatedAfter = UInt64(request.device.currentAllocatedSize)
      let added = allocatedAfter > allocatedBefore ? allocatedAfter - allocatedBefore : 0
      guard installResidentDetectorIndex(candidate, request: request) else {
        completion(.superseded)
        return
      }
      completion(.installed(addedBytes: added, buildSeconds: seconds))
    }
  }

  /// True once an index has been installed, whether built inline or scheduled.
  @_spi(PairedRuntimeTANSPrototype)
  public var residentDetectorIndexPrepared: Bool {
    stateLock.lock()
    defer { stateLock.unlock() }
    return residentIndexPrepared
  }

  /// Every input a polar-index build reads after `stateLock` is released.
  ///
  /// The buffer references are strong, so the build keeps its own inputs alive
  /// even if the resident is released while it runs. `generation` is rechecked
  /// before the result is installed, so a build that outlives its resident, or
  /// one that races an explicit index release, is dropped instead of installed.
  private struct ResidentDetectorIndexBuildRequest: @unchecked Sendable {
    let generation: UInt64
    let device: MTLDevice
    let queue: MTLCommandQueue
    let library: MTLLibrary
    let payload: MTLBuffer
    let offsets: MTLBuffer
    let modes: MTLBuffer
    let decoding: MTLBuffer
    let validPixels: [UInt8]
    let packets: Int
    let streamRankOfPixel: [UInt32]?
    let compactOffsetsEnabled: Bool
    let allocationLimit: UInt64
  }

  /// Validate and snapshot under `stateLock`, then release it before any build
  /// work. Returns `nil` when an index already exists, which keeps the shipped
  /// silent no-op. Reuses the library the load already compiled instead of
  /// recompiling it under the lock.
  private func makeResidentDetectorIndexRequest(
    maximumAdditionalBytes: UInt64
  ) throws -> ResidentDetectorIndexBuildRequest? {
    stateLock.lock()
    defer { stateLock.unlock() }
    try requireLive()
    if residentIndexPrepared { return nil }
    guard configuration.mode == .normal, polarIndex == nil else {
      throw Self.invalid(
        "Index-only preparation requires a Normal resident without an existing index")
    }
    let device = queue.device
    let before = UInt64(device.currentAllocatedSize)
    let limit = before.addingReportingOverflow(maximumAdditionalBytes)
    guard !limit.overflow else {
      throw Self.invalid("Additional index budget exceeds addressable memory")
    }
    return ResidentDetectorIndexBuildRequest(
      generation: residentIndexGeneration,
      device: device, queue: queue, library: sourceLibrary,
      payload: payload, offsets: offsets, modes: modes, decoding: decodingTable,
      validPixels: validPixels, packets: shape[0] * shape[1] / 512,
      streamRankOfPixel: streamRankOfPixel,
      compactOffsetsEnabled: compactOffsetsEnabled,
      allocationLimit: min(limit.partialValue, device.recommendedMaxWorkingSetSize))
  }

  /// The expensive half. Runs with no `stateLock` held.
  private func buildResidentDetectorIndex(
    _ request: ResidentDetectorIndexBuildRequest, queue: MTLCommandQueue? = nil,
    shouldCancel: () -> Bool
  ) throws -> MetalPairedRuntimeTANSPolarIndex {
    try autoreleasepool {
      try MetalPairedRuntimeTANSPolarIndex(
        device: request.device, library: request.library, queue: queue ?? request.queue,
        payload: request.payload, offsets: request.offsets, modes: request.modes,
        decoding: request.decoding, validPixels: request.validPixels,
        packets: request.packets, leafPixels: 16, layoutKind: "radial1fine4",
        streamRankOfPixel: request.streamRankOfPixel,
        compactOffsetsEnabled: request.compactOffsetsEnabled,
        prepareScan512QueryPipeline: true,
        allocationLimit: request.allocationLimit,
        shouldCancel: shouldCancel)
    }
  }

  /// Install a finished index under `stateLock`, rejecting stale generations.
  private func installResidentDetectorIndex(
    _ candidate: MetalPairedRuntimeTANSPolarIndex,
    request: ResidentDetectorIndexBuildRequest
  ) -> Bool {
    stateLock.lock()
    defer { stateLock.unlock() }
    guard !released, residentIndexGeneration == request.generation,
      !residentIndexPrepared, polarIndex == nil,
      let livePayload = payload, livePayload === request.payload
    else { return false }
    polarIndex = candidate
    residentIndexPrepared = true
    refreshMetadataSnapshot()
    return true
  }

  /// Drop only the optional resident-built index; encoded counts remain allocated.
  @_spi(PairedRuntimeTANSPrototype)
  public func releaseResidentDetectorIndex() {
    stateLock.lock()
    defer { stateLock.unlock() }
    // Invalidate any build that is still running, including one scheduled by
    // `scheduleResidentDetectorIndexPreparation`, so it cannot reinstall.
    residentIndexGeneration &+= 1
    guard residentIndexPrepared else { return }
    polarIndex = nil
    residentIndexPrepared = false
    refreshMetadataSnapshot()
  }

  /// Map detector pixel ids to the stream addresses used by the resident layout.
  private func streamAddresses(_ pixels: [UInt32]) -> [UInt32] {
    guard let streamRankOfPixel else { return pixels }
    return pixels.map { $0 < UInt32(streamRankOfPixel.count) ? streamRankOfPixel[Int($0)] : $0 }
  }

  /// Convert retained ANS counts to the existing packed layout without file IO.
  /// The original remains valid until explicitly released by the caller.
  @_spi(PairedRuntimeTANSPrototype)
  public func makePackedResident(
    maximumAdditionalBytes: UInt64, staging: Bool = false,
    shouldCancel: () -> Bool = { false }
  ) throws -> MetalCompactH5ResidentSource {
    stateLock.lock()
    defer { stateLock.unlock() }
    try requireLive()
    return try PairedRuntimePackedConversion.convert(
      dataset: sourceDataset, moments: dpcMoments, payload: payload, offsets: offsets,
      modes: modes, table: decodingTable, pixelOfRank: pixelOfStreamRank,
      compactOffsets: compactOffsetsEnabled, library: sourceLibrary, queue: queue,
      maximumAdditionalBytes: maximumAdditionalBytes, staging: staging, shouldCancel: shouldCancel)
  }

  /// Return one exact diffraction pattern, widened only for the public API.
  public func extractRawDiffraction(scanRow: Int, scanColumn: Int) throws -> [UInt32] {
    stateLock.lock()
    defer { stateLock.unlock() }
    try requireLive()
    guard (0..<shape[0]).contains(scanRow), (0..<shape[1]).contains(scanColumn) else {
      throw Self.invalid("Choose an in-bounds scan row and column")
    }
    let scan = scanRow * shape[1] + scanColumn
    memset(failure.contents(), 0, 4)
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw Self.invalid("Metal could not encode a paired-runtime diffraction query") }
    var parameters: [UInt32] = [
      UInt32(shape[2] * shape[3]),
      UInt32(shape[0] * shape[1] / PairedRuntimeTANSRecordABI.streamScans),
      UInt32(scan), UInt32(payload.length),
    ]
    encoder.setComputePipelineState(selectedDPPipeline)
    for (index, buffer) in [
      payload, offsets, modes, decodingTable,
      diffraction, failure,
    ].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBytes(&parameters, length: parameters.count * 4, index: 6)
    let width = selectedDPPipeline.threadExecutionWidth
    encoder.dispatchThreads(
      MTLSize(width: shape[2] * shape[3], height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(selectedDPPipeline.maxTotalThreadsPerThreadgroup, width * 4),
        height: 1, depth: 1))
    encoder.endEncoding()
    try finish(command, operation: "diffraction")
    let stored = UnsafeBufferPointer(
      start: diffraction.contents().assumingMemoryBound(to: UInt32.self),
      count: shape[2] * shape[3])
    guard let pixelOfStreamRank else { return Array(stored) }
    var values = [UInt32](repeating: 0, count: stored.count)
    for (rank, pixel) in pixelOfStreamRank.enumerated() { values[Int(pixel)] = stored[rank] }
    return values
  }

  private struct UpdateProfile {
    let requestEntry: Double
    let preparationStart: Double
    var planningStart: Double = 0
    var planningEnd: Double = 0
    var preparationEnd: Double = 0
    var commitHost: Double = 0
    var waitCompletionHost: Double = 0
    var readbackStart: Double = 0
    var readbackEnd: Double = 0
    var gpuStart: Double = 0
    var gpuEnd: Double = 0
    var profileResolveMilliseconds: Double = 0
    var stageResult: PairedRuntimeDetectorProfiler.StageResult?

    var values: [String: Double] {
      var result: [String: Double] = [
        "cpu_request_entry_seconds": requestEntry,
        "cpu_preparation_start_seconds": preparationStart,
        "cpu_planning_start_seconds": planningStart,
        "cpu_planning_end_seconds": planningEnd,
        "cpu_preparation_end_seconds": preparationEnd,
        "cpu_planning_milliseconds": max(0, planningEnd - planningStart) * 1_000,
        "cpu_preparation_milliseconds": max(0, preparationEnd - preparationStart) * 1_000,
        "host_commit_seconds": commitHost,
        "host_wait_completion_seconds": waitCompletionHost,
        "host_readback_milliseconds": max(0, readbackEnd - readbackStart) * 1_000,
        "host_profile_resolve_milliseconds": profileResolveMilliseconds,
        "command_gpu_start_seconds": gpuStart,
        "command_gpu_end_seconds": gpuEnd,
        "command_gpu_milliseconds": max(0, gpuEnd - gpuStart) * 1_000,
      ]
      if let stageResult {
        result["stage_profile_valid"] = stageResult.valid ? 1 : 0
        result["stage_command_milliseconds"] = stageResult.commandNanoseconds / 1_000_000
        result["stage_encoder_union_milliseconds"] =
          stageResult.encoderUnionNanoseconds / 1_000_000
        if let scale = stageResult.calibratedNanosecondsPerTick {
          result["stage_calibrated_nanoseconds_per_tick"] = scale
        }
        for (name, nanoseconds) in stageResult.stageNanoseconds {
          result["stage_\(name)_milliseconds"] = nanoseconds / 1_000_000
        }
        for (name, seconds) in stageResult.stageTimeline {
          result[name] = seconds
        }
      }
      return result
    }
  }

  private struct PendingDetectorUpdate {
    let command: MTLCommandBuffer?
    let mask: [UInt8]
    let previousMask: [UInt8]
    let started: Double
    let changedCount: Int
    let operation: String
    let candidateProduct: MTLBuffer?
    let historyHit: Bool
    let historyBase: Bool
    let previousHistoryValid: Bool
    let previousPolarFieldCount: Int
    let previousPolarResidualCount: Int
    let profile: UpdateProfile?
    let stageSession: PairedRuntimeDetectorProfiler.Session?
    let keepAlive: [MTLBuffer]
  }

  public typealias DetectorUpdateResult = (
    values: [UInt32], wallMilliseconds: Double, gpuMilliseconds: Double,
    changedDetectorPixels: Int
  )
  @_spi(PairedRuntimeTANSPrototype)
  public typealias DetectorDecodeChecksumResult = (
    values: [UInt32], wallMilliseconds: Double, gpuMilliseconds: Double
  )

  @_spi(FourWayCheckpointPrototype)
  public struct FourWayCheckpointDiagnosticResult: Sendable {
    public let outcome: String
    public let sourceIdentitySHA256: String
    public let selectedStreamIndices: [UInt32]
    public let residualDetectorPixelCount: Int
    public let residualStreamCount: Int
    public let coverageFraction: Double
    public let modeHistogram: [Int]
    public let selectedEntropyCount: Int
    public let selectedFallbackCount: Int
    public let selectedUnsupportedCount: Int
    public let compactOffsetsEnabled: Bool
    public let compactOffsetBytes: Int
    public let offsetBufferBytes: Int
    public let residentBytes: UInt64
    public let modeInspectionScratchBytes: Int
    public let checkpointAllocationBytes: Int
    public let segmentScratchBytes: Int
    public let diagnosticAllocationBytes: Int
    public let allocatedSizeBefore: Int
    public let allocatedSizeAfter: Int
    public let runtimeCompileMilliseconds: Double
    public let modeInspectionWallMilliseconds: Double
    public let modeInspectionGPUMilliseconds: Double
    public let captureWallMilliseconds: Double?
    public let captureGPUMilliseconds: Double?
    public let segmentWallMilliseconds: Double?
    public let segmentGPUMilliseconds: Double?
    /// Capture bins are status values 0–6 followed by an unknown/unwritten bin.
    public let captureStatusCounts: [Int]?
    /// Segment bins are status values 0–5 followed by an unknown/unwritten bin.
    public let segmentStatusCounts: [Int]?
    public let decodedValues: [UInt16]?
    public let decodedValueOrder: String
  }

  /// Update one exact virtual-detector image using only changed mask pixels.
  public func updateVirtualDetector(mask: [UInt8]) throws -> DetectorUpdateResult {
    stateLock.lock()
    defer { stateLock.unlock() }
    if blockStride > 1 { return try settleBlockStride(to: mask) }
    return try finishDetectorUpdate(try prepareDetectorUpdate(mask: mask))
  }

  /// Same exact update, but the image is copied once into `destination`
  /// (at least scan positions x 4 bytes of UInt32) and `values` is empty.
  @_spi(PairedRuntimeTANSPrototype)
  public func updateVirtualDetector(mask: [UInt8], into destination: MTLBuffer) throws
    -> DetectorUpdateResult
  {
    stateLock.lock()
    defer { stateLock.unlock() }
    try requireLive()
    guard destination.storageMode == .shared,
      destination.length >= shape[0] * shape[1] * MemoryLayout<UInt32>.stride
    else {
      throw Self.invalid(
        "The detector destination must be a shared buffer at least as large as the scan image")
    }
    pendingCopyDestination = destination
    defer { pendingCopyDestination = nil }
    if blockStride > 1 { return try settleBlockStride(to: mask) }
    return try finishDetectorUpdate(try prepareDetectorUpdate(mask: mask))
  }

  /// Exact motion update of every `stride`-th 512-scan block starting at `phase`.
  /// Rows in the updated blocks become exact for `mask`; other rows keep the last
  /// mask their block phase received. `stride == 1` settles every phase to `mask`.
  /// The returned values mix rows from different recent masks while `stride > 1`;
  /// callers must label them as a motion preview until they settle with stride 1.
  @_spi(PairedRuntimeTANSPrototype)
  public func updateVirtualDetector(
    mask: [UInt8], blockStride stride: Int, phase: Int
  ) throws -> DetectorUpdateResult {
    stateLock.lock()
    defer { stateLock.unlock() }
    guard [1, 2, 4, 8].contains(stride), phase >= 0, phase < stride else {
      throw Self.invalid("Block stride must be 1, 2, 4 or 8 with 0 <= phase < stride")
    }
    guard
      stride == 1
        || (!historyEnabled
          && runtimeOption("QGPU_PAIRED_RUNTIME_FROM_ZERO") != "1")
    else { throw Self.invalid("Block stride cannot be combined with history or from_zero") }
    if stride == 1 {
      if blockStride > 1 { return try settleBlockStride(to: mask) }
      return try finishDetectorUpdate(try prepareDetectorUpdate(mask: mask))
    }
    if blockStride != stride {
      if blockStride > 1 { _ = try settleBlockStride(to: detectorMask) }
      blockPhaseMasks = Array(repeating: detectorMask, count: stride)
      blockStride = stride
    }
    detectorMask = blockPhaseMasks[phase]
    activePacketStride = stride
    activePacketPhase = phase
    defer {
      activePacketStride = 1
      activePacketPhase = 0
    }
    let result = try finishDetectorUpdate(try prepareDetectorUpdate(mask: mask))
    blockPhaseMasks[phase] = mask
    return result
  }

  /// Bring every block phase to `mask` with one exact update per stale phase.
  private func settleBlockStride(to mask: [UInt8]) throws -> DetectorUpdateResult {
    let stride = blockStride
    var last: DetectorUpdateResult?
    for phase in 0..<stride where blockPhaseMasks[phase] != mask {
      detectorMask = blockPhaseMasks[phase]
      activePacketStride = stride
      activePacketPhase = phase
      defer {
        activePacketStride = 1
        activePacketPhase = 0
      }
      last = try finishDetectorUpdate(try prepareDetectorUpdate(mask: mask))
      blockPhaseMasks[phase] = mask
    }
    blockStride = 1
    blockPhaseMasks = []
    detectorMask = mask
    if let last { return last }
    return (detectorValues(), 0, 0, 0)
  }

  /// Run the FC14 decode-coverage diagnostic without changing detector state.
  /// The result contains one wraparound UInt32 checksum for each 512-scan
  /// packet; it is not an image and must not be used as a parity or speed path.
  @_spi(PairedRuntimeTANSPrototype)
  public func detectorDecodeChecksum(mask: [UInt8]) throws -> DetectorDecodeChecksumResult {
    stateLock.lock()
    defer { stateLock.unlock() }
    try requireLive()
    guard let pipeline = detectorDecodeChecksumPipeline else {
      throw Self.invalid(
        "Prepare the FC14 decode-checksum pipeline with "
          + "QGPU_PAIRED_RUNTIME_PREPARE_DECODE_CHECKSUM=1 before loading the resident")
    }
    let pixels = shape[2] * shape[3]
    guard mask.count == pixels, mask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
      throw Self.invalid("The decode-checksum mask must contain one binary value per pixel")
    }
    var selected: [UInt32] = []
    selected.reserveCapacity(mask.reduce(0) { $0 + ($1 == 0 ? 0 : 1) })
    for pixel in mask.indices where mask[pixel] != 0 && validPixels[pixel] != 0 {
      selected.append(UInt32(pixel))
    }
    let packets = shape[0] * shape[1] / PairedRuntimeTANSRecordABI.streamScans
    let selectedBuffer = try Self.upload(
      selected.isEmpty ? [0] : streamAddresses(selected), device: detectorProduct.device,
      label: "paired-runtime decode-checksum selected pixels")
    let coefficients = try Self.upload(
      [Int32](repeating: 1, count: max(1, selected.count)), device: detectorProduct.device,
      label: "paired-runtime decode-checksum coefficients")
    let checksums = try Self.buffer(
      device: detectorProduct.device, bytes: packets * MemoryLayout<UInt32>.stride,
      options: .storageModeShared, label: "paired-runtime decode-checksum output")
    let localFailure = try Self.buffer(
      device: detectorProduct.device, bytes: MemoryLayout<UInt32>.stride,
      options: .storageModeShared, label: "paired-runtime decode-checksum failure")
    memset(checksums.contents(), 0, checksums.length)
    memset(localFailure.contents(), 0, localFailure.length)
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw Self.invalid("Metal could not encode the FC14 decode checksum") }
    var parameters: [UInt32] = [
      UInt32(pixels), UInt32(packets), UInt32(selected.count), 0,
      UInt32(payload.length), 1, 1, 0,
    ]
    encoder.setComputePipelineState(pipeline)
    for (index, buffer) in [
      payload, offsets, modes, decodingTable, selectedBuffer, coefficients,
      checksums, localFailure,
    ].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBytes(&parameters, length: parameters.count * MemoryLayout<UInt32>.stride, index: 8)
    encoder.dispatchThreadgroups(
      MTLSize(width: (packets + 3) / 4, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    encoder.endEncoding()
    let started = CFAbsoluteTimeGetCurrent()
    command.commit()
    command.waitUntilCompleted()
    let code = localFailure.contents().load(as: UInt32.self)
    guard command.status == .completed, command.error == nil, code == 0 else {
      throw Self.invalid(
        "Paired-runtime FC14 decode checksum failed with code \(code): "
          + (command.error?.localizedDescription ?? "invalid stream"))
    }
    let values = Array(
      UnsafeBufferPointer(
        start: checksums.contents().assumingMemoryBound(to: UInt32.self), count: packets))
    return (
      values, (CFAbsoluteTimeGetCurrent() - started) * 1_000,
      max(0, command.gpuEndTime - command.gpuStartTime) * 1_000
    )
  }

  /// Runtime-compile and dispatch the isolated four-way checkpoint prototype.
  /// This diagnostic decodes only the caller-selected entropy streams; it does
  /// not update a virtual detector or establish full-map parity/performance.
  @_spi(FourWayCheckpointPrototype)
  public func diagnoseFourWayCheckpoint(
    selectedStreamIndices: [UInt32], residualDetectorPixels: [UInt32],
    shaderSourceURL: URL
  ) throws -> FourWayCheckpointDiagnosticResult {
    stateLock.lock()
    defer { stateLock.unlock() }
    try requireLive()

    let pixels = shape[2] * shape[3]
    let scanCount = shape[0] * shape[1]
    guard pixels > 0, scanCount > 0,
      scanCount % PairedRuntimeTANSRecordABI.streamScans == 0
    else { throw Self.invalid("Four-way diagnostics require complete 512-scan streams") }
    let packets = scanCount / PairedRuntimeTANSRecordABI.streamScans
    let (streamCount, streamCountOverflow) = pixels.multipliedReportingOverflow(by: packets)
    let (residualStreamCount, residualCountOverflow) =
      residualDetectorPixels.count.multipliedReportingOverflow(by: packets)
    guard !streamCountOverflow, !residualCountOverflow,
      streamCount <= Int(UInt32.max), payload.length <= Int(UInt32.max),
      !selectedStreamIndices.isEmpty, selectedStreamIndices.count <= 4_096,
      !residualDetectorPixels.isEmpty
    else {
      throw Self.invalid(
        "Four-way diagnostics need 1–4096 selected streams in a nonempty residual set")
    }
    let residualSet = Set(residualDetectorPixels)
    guard residualSet.count == residualDetectorPixels.count,
      residualDetectorPixels.allSatisfy({
        Int($0) < pixels && validPixels[Int($0)] != 0
      })
    else {
      throw Self.invalid(
        "Residual detector pixels must be unique, in bounds, and valid for this resident")
    }
    guard streamRankOfPixel == nil else {
      throw Self.invalid("The four-way checkpoint diagnostic requires pixel stream order")
    }
    let selectedSet = Set(selectedStreamIndices)
    guard selectedSet.count == selectedStreamIndices.count,
      selectedStreamIndices.allSatisfy({
        Int($0) < streamCount && residualSet.contains($0 % UInt32(pixels))
      })
    else {
      throw Self.invalid(
        "Selected streams must be unique, in bounds, and belong to the supplied residual set")
    }

    let device = queue.device
    guard let source = try? String(contentsOf: shaderSourceURL, encoding: .utf8) else {
      throw Self.invalid("Could not read four-way Metal source at \(shaderSourceURL.path)")
    }
    let allocatedBefore = device.currentAllocatedSize
    let compileStarted = CFAbsoluteTimeGetCurrent()
    let compileOptions = MTLCompileOptions()
    compileOptions.fastMathEnabled = false
    let library: MTLLibrary
    do {
      library = try device.makeLibrary(source: source, options: compileOptions)
    } catch {
      throw Self.invalid("Four-way runtime MSL compilation failed: \(error.localizedDescription)")
    }
    func makePipeline(_ name: String) throws -> MTLComputePipelineState {
      guard let function = library.makeFunction(name: name) else {
        throw Self.invalid("Four-way Metal function \(name) is missing")
      }
      return try device.makeComputePipelineState(function: function)
    }
    let inspectPipeline: MTLComputePipelineState
    let capturePipeline: MTLComputePipelineState
    let segmentPipeline: MTLComputePipelineState
    do {
      inspectPipeline = try makePipeline("fw_inspect_selected_modes")
      capturePipeline = try makePipeline("fw_capture_selected_checkpoints")
      segmentPipeline = try makePipeline("fw_decode_fourway_segment_values")
    } catch {
      throw Self.invalid("Four-way Metal pipeline creation failed: \(error.localizedDescription)")
    }
    let compileMilliseconds = (CFAbsoluteTimeGetCurrent() - compileStarted) * 1_000

    func dispatch(
      pipeline: MTLComputePipelineState, buffers: [MTLBuffer],
      parameters: inout [UInt32], threadCount: Int, stage: String
    ) throws -> (wallMilliseconds: Double, gpuMilliseconds: Double) {
      guard threadCount > 0, let command = queue.makeCommandBuffer(),
        let encoder = command.makeComputeCommandEncoder()
      else { throw Self.invalid("Metal could not encode four-way \(stage)") }
      encoder.setComputePipelineState(pipeline)
      for (index, buffer) in buffers.enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBytes(
        &parameters, length: parameters.count * MemoryLayout<UInt32>.stride,
        index: buffers.count)
      let width = max(1, min(256, pipeline.maxTotalThreadsPerThreadgroup))
      encoder.dispatchThreads(
        MTLSize(width: threadCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
      encoder.endEncoding()
      let started = CFAbsoluteTimeGetCurrent()
      command.commit()
      command.waitUntilCompleted()
      let wallMilliseconds = (CFAbsoluteTimeGetCurrent() - started) * 1_000
      guard command.status == .completed, command.error == nil else {
        throw Self.invalid(
          "Four-way \(stage) dispatch failed: "
            + (command.error?.localizedDescription ?? "unknown Metal failure"))
      }
      return (
        wallMilliseconds,
        max(0, command.gpuEndTime - command.gpuStartTime) * 1_000
      )
    }

    let selectedBuffer = try Self.upload(
      selectedStreamIndices, device: device,
      label: "four-way diagnostic selected streams")
    let selectedModes = try Self.buffer(
      device: device, bytes: selectedStreamIndices.count, options: .storageModeShared,
      label: "four-way diagnostic selected modes")
    memset(selectedModes.contents(), 0, selectedModes.length)
    var inspectParameters: [UInt32] = [UInt32(selectedStreamIndices.count)]
    let inspectTiming = try dispatch(
      pipeline: inspectPipeline, buffers: [modes, selectedBuffer, selectedModes],
      parameters: &inspectParameters, threadCount: selectedStreamIndices.count,
      stage: "mode inspection")
    let inspectedModes = Array(
      UnsafeBufferPointer(
        start: selectedModes.contents().assumingMemoryBound(to: UInt8.self),
        count: selectedStreamIndices.count))
    var modeHistogram = [Int](repeating: 0, count: 256)
    for mode in inspectedModes { modeHistogram[Int(mode)] += 1 }
    let entropyCount = modeHistogram[64...95].reduce(0, +)
    let fallbackCount = modeHistogram[252...255].reduce(0, +)
    let unsupportedCount = selectedStreamIndices.count - entropyCount - fallbackCount
    let modeInspectionScratchBytes = selectedBuffer.length + selectedModes.length
    let allocatedAfterInspection = device.currentAllocatedSize
    guard fallbackCount == 0, unsupportedCount == 0 else {
      return FourWayCheckpointDiagnosticResult(
        outcome: "rejected-non-entropy-modes", sourceIdentitySHA256: sourceIdentitySHA256,
        selectedStreamIndices: selectedStreamIndices,
        residualDetectorPixelCount: residualDetectorPixels.count,
        residualStreamCount: residualStreamCount,
        coverageFraction: Double(selectedStreamIndices.count) / Double(residualStreamCount),
        modeHistogram: modeHistogram, selectedEntropyCount: entropyCount,
        selectedFallbackCount: fallbackCount, selectedUnsupportedCount: unsupportedCount,
        compactOffsetsEnabled: compactOffsetsEnabled,
        compactOffsetBytes: compactOffsetsEnabled ? offsets.length : 0,
        offsetBufferBytes: offsets.length,
        residentBytes: residentBytes, modeInspectionScratchBytes: modeInspectionScratchBytes,
        checkpointAllocationBytes: 0, segmentScratchBytes: 0,
        diagnosticAllocationBytes: modeInspectionScratchBytes,
        allocatedSizeBefore: allocatedBefore, allocatedSizeAfter: allocatedAfterInspection,
        runtimeCompileMilliseconds: compileMilliseconds,
        modeInspectionWallMilliseconds: inspectTiming.wallMilliseconds,
        modeInspectionGPUMilliseconds: inspectTiming.gpuMilliseconds,
        captureWallMilliseconds: nil, captureGPUMilliseconds: nil,
        segmentWallMilliseconds: nil, segmentGPUMilliseconds: nil,
        captureStatusCounts: nil, segmentStatusCounts: nil, decodedValues: nil,
        decodedValueOrder: "none; non-entropy selection rejected before capture")
    }

    let checkpointBytes = selectedStreamIndices.count * 9
    let captureStatusBytes = selectedStreamIndices.count * MemoryLayout<UInt32>.stride
    let checkpoints = try Self.buffer(
      device: device, bytes: checkpointBytes, options: .storageModeShared,
      label: "four-way diagnostic checkpoints")
    let captureStatus = try Self.buffer(
      device: device, bytes: captureStatusBytes, options: .storageModeShared,
      label: "four-way diagnostic capture status")
    memset(checkpoints.contents(), 0, checkpoints.length)
    memset(captureStatus.contents(), 0xff, captureStatus.length)
    var captureParameters: [UInt32] = [
      UInt32(streamCount), UInt32(selectedStreamIndices.count), UInt32(payload.length),
      compactOffsetsEnabled ? 1 : 0, UInt32(PairedRuntimeTANSRecordABI.streamScans),
    ]
    let captureTiming = try dispatch(
      pipeline: capturePipeline,
      buffers: [
        payload, offsets, modes, decodingTable, selectedBuffer, checkpoints, captureStatus,
      ],
      parameters: &captureParameters, threadCount: selectedStreamIndices.count,
      stage: "checkpoint capture")
    let captureStatuses = Array(
      UnsafeBufferPointer(
        start: captureStatus.contents().assumingMemoryBound(to: UInt32.self),
        count: selectedStreamIndices.count))
    var captureHistogram = [Int](repeating: 0, count: 8)
    for status in captureStatuses {
      let index =
        Int(status) < captureHistogram.count - 1
        ? Int(status) : captureHistogram.count - 1
      captureHistogram[index] += 1
    }
    let allocatedAfterCapture = device.currentAllocatedSize
    let captureAllocationBytes = checkpoints.length + captureStatus.length
    if captureHistogram[1] != selectedStreamIndices.count {
      return FourWayCheckpointDiagnosticResult(
        outcome: "checkpoint-capture-error", sourceIdentitySHA256: sourceIdentitySHA256,
        selectedStreamIndices: selectedStreamIndices,
        residualDetectorPixelCount: residualDetectorPixels.count,
        residualStreamCount: residualStreamCount,
        coverageFraction: Double(selectedStreamIndices.count) / Double(residualStreamCount),
        modeHistogram: modeHistogram, selectedEntropyCount: entropyCount,
        selectedFallbackCount: fallbackCount, selectedUnsupportedCount: unsupportedCount,
        compactOffsetsEnabled: compactOffsetsEnabled,
        compactOffsetBytes: compactOffsetsEnabled ? offsets.length : 0,
        offsetBufferBytes: offsets.length,
        residentBytes: residentBytes, modeInspectionScratchBytes: modeInspectionScratchBytes,
        checkpointAllocationBytes: checkpoints.length, segmentScratchBytes: 0,
        diagnosticAllocationBytes: modeInspectionScratchBytes + captureAllocationBytes,
        allocatedSizeBefore: allocatedBefore, allocatedSizeAfter: allocatedAfterCapture,
        runtimeCompileMilliseconds: compileMilliseconds,
        modeInspectionWallMilliseconds: inspectTiming.wallMilliseconds,
        modeInspectionGPUMilliseconds: inspectTiming.gpuMilliseconds,
        captureWallMilliseconds: captureTiming.wallMilliseconds,
        captureGPUMilliseconds: captureTiming.gpuMilliseconds,
        segmentWallMilliseconds: nil, segmentGPUMilliseconds: nil,
        captureStatusCounts: captureHistogram, segmentStatusCounts: nil,
        decodedValues: nil,
        decodedValueOrder: "none; checkpoint capture did not validate all selected streams")
    }

    let segmentCount = selectedStreamIndices.count * 4
    let (decodedValueCount, decodedCountOverflow) =
      selectedStreamIndices.count.multipliedReportingOverflow(
        by: PairedRuntimeTANSRecordABI.streamScans)
    let (decodedBytes, decodedBytesOverflow) =
      decodedValueCount.multipliedReportingOverflow(by: MemoryLayout<UInt16>.stride)
    guard !decodedCountOverflow, !decodedBytesOverflow,
      decodedBytes <= device.maxBufferLength
    else { throw Self.invalid("Four-way exact-value output exceeds Metal buffer limits") }
    let decodedValuesBuffer = try Self.buffer(
      device: device, bytes: decodedBytes, options: .storageModeShared,
      label: "four-way diagnostic decoded UInt16 values")
    let terminalStates = try Self.buffer(
      device: device, bytes: segmentCount * MemoryLayout<UInt32>.stride,
      options: .storageModeShared, label: "four-way diagnostic terminal states")
    let terminalUnread = try Self.buffer(
      device: device, bytes: segmentCount * MemoryLayout<UInt32>.stride,
      options: .storageModeShared, label: "four-way diagnostic terminal bits")
    let segmentStatus = try Self.buffer(
      device: device, bytes: segmentCount * MemoryLayout<UInt32>.stride,
      options: .storageModeShared, label: "four-way diagnostic segment status")
    memset(decodedValuesBuffer.contents(), 0, decodedValuesBuffer.length)
    memset(terminalStates.contents(), 0, terminalStates.length)
    memset(terminalUnread.contents(), 0, terminalUnread.length)
    memset(segmentStatus.contents(), 0xff, segmentStatus.length)
    var segmentParameters: [UInt32] = captureParameters
    let segmentTiming = try dispatch(
      pipeline: segmentPipeline,
      buffers: [
        payload, offsets, modes, decodingTable, selectedBuffer, checkpoints,
        captureStatus, decodedValuesBuffer, terminalStates, terminalUnread, segmentStatus,
      ],
      parameters: &segmentParameters, threadCount: segmentCount, stage: "four-way segment decode")
    let segmentStatuses = Array(
      UnsafeBufferPointer(
        start: segmentStatus.contents().assumingMemoryBound(to: UInt32.self),
        count: segmentCount))
    var segmentHistogram = [Int](repeating: 0, count: 7)
    for status in segmentStatuses {
      let index =
        Int(status) < segmentHistogram.count - 1
        ? Int(status) : segmentHistogram.count - 1
      segmentHistogram[index] += 1
    }
    let segmentScratchBytes =
      decodedValuesBuffer.length + terminalStates.length
      + terminalUnread.length + segmentStatus.length
    let allocatedAfterSegments = device.currentAllocatedSize
    let diagnosticAllocationBytes =
      modeInspectionScratchBytes + captureAllocationBytes
      + segmentScratchBytes
    guard segmentHistogram[1] == segmentCount else {
      return FourWayCheckpointDiagnosticResult(
        outcome: "segmented-decode-error", sourceIdentitySHA256: sourceIdentitySHA256,
        selectedStreamIndices: selectedStreamIndices,
        residualDetectorPixelCount: residualDetectorPixels.count,
        residualStreamCount: residualStreamCount,
        coverageFraction: Double(selectedStreamIndices.count) / Double(residualStreamCount),
        modeHistogram: modeHistogram, selectedEntropyCount: entropyCount,
        selectedFallbackCount: fallbackCount, selectedUnsupportedCount: unsupportedCount,
        compactOffsetsEnabled: compactOffsetsEnabled,
        compactOffsetBytes: compactOffsetsEnabled ? offsets.length : 0,
        offsetBufferBytes: offsets.length,
        residentBytes: residentBytes, modeInspectionScratchBytes: modeInspectionScratchBytes,
        checkpointAllocationBytes: checkpoints.length, segmentScratchBytes: segmentScratchBytes,
        diagnosticAllocationBytes: diagnosticAllocationBytes,
        allocatedSizeBefore: allocatedBefore, allocatedSizeAfter: allocatedAfterSegments,
        runtimeCompileMilliseconds: compileMilliseconds,
        modeInspectionWallMilliseconds: inspectTiming.wallMilliseconds,
        modeInspectionGPUMilliseconds: inspectTiming.gpuMilliseconds,
        captureWallMilliseconds: captureTiming.wallMilliseconds,
        captureGPUMilliseconds: captureTiming.gpuMilliseconds,
        segmentWallMilliseconds: segmentTiming.wallMilliseconds,
        segmentGPUMilliseconds: segmentTiming.gpuMilliseconds,
        captureStatusCounts: captureHistogram, segmentStatusCounts: segmentHistogram,
        decodedValues: nil, decodedValueOrder: "none; one or more segment validations failed")
    }
    let decodedValues = Array(
      UnsafeBufferPointer(
        start: decodedValuesBuffer.contents().assumingMemoryBound(to: UInt16.self),
        count: decodedValueCount))
    return FourWayCheckpointDiagnosticResult(
      outcome: "decoded-entropy-subset-parity-unchecked",
      sourceIdentitySHA256: sourceIdentitySHA256, selectedStreamIndices: selectedStreamIndices,
      residualDetectorPixelCount: residualDetectorPixels.count,
      residualStreamCount: residualStreamCount,
      coverageFraction: Double(selectedStreamIndices.count) / Double(residualStreamCount),
      modeHistogram: modeHistogram, selectedEntropyCount: entropyCount,
      selectedFallbackCount: fallbackCount, selectedUnsupportedCount: unsupportedCount,
      compactOffsetsEnabled: compactOffsetsEnabled,
      compactOffsetBytes: compactOffsetsEnabled ? offsets.length : 0,
      offsetBufferBytes: offsets.length,
      residentBytes: residentBytes, modeInspectionScratchBytes: modeInspectionScratchBytes,
      checkpointAllocationBytes: checkpoints.length, segmentScratchBytes: segmentScratchBytes,
      diagnosticAllocationBytes: diagnosticAllocationBytes,
      allocatedSizeBefore: allocatedBefore, allocatedSizeAfter: allocatedAfterSegments,
      runtimeCompileMilliseconds: compileMilliseconds,
      modeInspectionWallMilliseconds: inspectTiming.wallMilliseconds,
      modeInspectionGPUMilliseconds: inspectTiming.gpuMilliseconds,
      captureWallMilliseconds: captureTiming.wallMilliseconds,
      captureGPUMilliseconds: captureTiming.gpuMilliseconds,
      segmentWallMilliseconds: segmentTiming.wallMilliseconds,
      segmentGPUMilliseconds: segmentTiming.gpuMilliseconds,
      captureStatusCounts: captureHistogram, segmentStatusCounts: segmentHistogram,
      decodedValues: decodedValues,
      decodedValueOrder: "selectedStreamIndices order; 512 UInt16 scans per stream")
  }

  /// Diagnostic signed contribution, independent of the current image/history.
  /// Scratch byte counts sum requested Metal buffer lengths, excluding host arrays
  /// and driver overhead. Reuse retains one exact mask pair until replacement/release.
  @_spi(PairedRuntimeTANSPrototype)
  public func isolateDetectorStage(
    previous: [UInt8], target: [UInt8], stage: String,
    branchlessPop: Bool = false, reuseScratch: Bool = false, refillThreshold: Int = 32,
    phasedReaders: Bool = false, pairUnroll: Int = 1
  )
    throws -> (
      values: [UInt32], gpuMilliseconds: Double, fields: Int, residuals: Int,
      allocatedScratchBytes: Int, preparedScratchBytes: Int, reusedScratch: Bool
    )
  {
    stateLock.lock()
    defer {
      refreshMetadataSnapshot()
      stateLock.unlock()
    }
    try requireLive()
    guard let polarIndex, ["index", "residual", "combined"].contains(stage),
      previous.count == validPixels.count, target.count == validPixels.count,
      previous.allSatisfy({ $0 <= 1 }), target.allSatisfy({ $0 <= 1 })
    else {
      throw Self.invalid("Stage isolation requires a prepared index and binary detector masks")
    }
    guard [16, 24, 32].contains(refillThreshold), !branchlessPop || refillThreshold == 32 else {
      throw Self.invalid("Use refillThreshold 16, 24, or 32; branchlessPop requires 32")
    }
    guard !phasedReaders || (!branchlessPop && refillThreshold == 32) else {
      throw Self.invalid("phasedReaders requires branchlessPop=false and refillThreshold=32")
    }
    guard [1, 2, 4, 8].contains(pairUnroll),
      pairUnroll == 1 || (!branchlessPop && refillThreshold == 32 && !phasedReaders)
    else {
      throw Self.invalid(
        "Use pairUnroll 1, 2, 4, or 8; unrolling requires branchlessPop=false, "
          + "refillThreshold=32, and phasedReaders=false")
    }
    let scratch: DiagnosticScratch
    let reused: Bool
    let jointPlanEnabled = runtimeOption("QGPU_PAIRED_RUNTIME_JOINT_PLAN") == "1"
    if reuseScratch, let cached = diagnosticScratch,
      cached.jointPlanEnabled == jointPlanEnabled,
      cached.previous == previous, cached.target == target
    {
      scratch = cached
      reused = true
    } else {
      // Drop the previous diagnostic allocation before preparing another mask pair.
      if reuseScratch { diagnosticScratch = nil }
      let delta = target.indices.map {
        validPixels[$0] == 0 ? Int32(0) : Int32(target[$0]) - Int32(previous[$0])
      }
      let plan = PairedRuntimeTANSPolarPlan.make(
        delta: delta, validPixels: validPixels, detectorRows: shape[2],
        detectorColumns: shape[3], leafPixels: polarIndex.leafPixels,
        layoutKind: polarIndex.layoutKind)
      let device = queue.device
      let output = try Self.buffer(
        device: device, bytes: detectorProduct.length,
        options: .storageModeShared, label: "diagnostic stage contribution")
      let status = try Self.buffer(
        device: device, bytes: 4,
        options: .storageModeShared, label: "diagnostic stage status")
      let selected = try Self.upload(
        plan.residualPixels.isEmpty ? [0] : streamAddresses(plan.residualPixels),
        device: device, label: "diagnostic residual pixels")
      let coefficients = try Self.upload(
        plan.residualCoefficients.isEmpty ? [0] : plan.residualCoefficients,
        device: device, label: "diagnostic residual coefficients")
      let indexInputs =
        reuseScratch || stage != "residual"
        ? try polarIndex.prepareInputs(plan: plan) : nil
      scratch = DiagnosticScratch(
        jointPlanEnabled: jointPlanEnabled,
        previous: previous, target: target, plan: plan,
        output: output, status: status, selected: selected, coefficients: coefficients,
        indexInputs: indexInputs)
      reused = false
      if reuseScratch {
        diagnosticScratch = scratch
      }
    }
    let plan = scratch.plan
    let output = scratch.output
    let status = scratch.status
    let selected = scratch.selected
    let coefficients = scratch.coefficients
    memset(output.contents(), 0, output.length)
    memset(status.contents(), 0, status.length)
    guard let command = queue.makeCommandBuffer() else {
      throw Self.invalid("No diagnostic command")
    }
    if stage != "residual" {
      try polarIndex.encode(
        inputs: scratch.indexInputs!, output: output,
        failure: status, command: command)
    }
    if stage != "index" && !plan.residualPixels.isEmpty {
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw Self.invalid("No diagnostic encoder")
      }
      let packets = shape[0] * shape[1] / 512
      var parameters: [UInt32] = [
        UInt32(validPixels.count), UInt32(packets),
        UInt32(plan.residualPixels.count), 0, UInt32(payload.length), 1, 1, 0,
      ]
      if branchlessPop {
        guard let pipeline = detectorBranchlessPopPipeline else {
          throw Self.invalid("Prepare branchless-pop diagnostic before loading")
        }
        encoder.setComputePipelineState(pipeline)
      } else if refillThreshold != 32 {
        guard let pipeline = detectorRefillPipelines[refillThreshold] else {
          throw Self.invalid("Set QGPU_PREPARE_REFILL_THRESHOLDS=1 before loading")
        }
        encoder.setComputePipelineState(pipeline)
      } else if phasedReaders {
        guard let pipeline = detectorPhasedReadersPipeline else {
          throw Self.invalid("Set QGPU_PREPARE_PHASED_READERS=1 before loading")
        }
        encoder.setComputePipelineState(pipeline)
      } else if pairUnroll != 1 {
        guard let pipeline = detectorPairUnrollPipelines[pairUnroll] else {
          throw Self.invalid("Set QGPU_PREPARE_PAIR_UNROLL=1 before loading")
        }
        encoder.setComputePipelineState(pipeline)
      } else {
        encoder.setComputePipelineState(detectorPacketOwner2Pipeline)
      }
      for (index, buffer) in [
        payload, offsets, modes, decodingTable, selected,
        coefficients, output, status,
      ].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBytes(&parameters, length: parameters.count * 4, index: 8)
      encoder.dispatchThreadgroups(
        MTLSize(width: (packets + 3) / 4, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.endEncoding()
    }
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed, status.contents().load(as: UInt32.self) == 0 else {
      throw Self.invalid("Stage isolation GPU failure")
    }
    return (
      Array(
        UnsafeBufferPointer(
          start: output.contents().assumingMemoryBound(to: UInt32.self),
          count: shape[0] * shape[1])), (command.gpuEndTime - command.gpuStartTime) * 1000,
      plan.selectedFields.count, plan.residualPixels.count,
      reused ? 0 : scratch.byteCount, scratch.byteCount, reused
    )
  }

  /// Submit one detector update for every unique resident, then complete them.
  @_spi(PairedRuntimeTANSPrototype)
  public static func updateVirtualDetectors(
    sources: [MetalPairedRuntimeTANSResidentSource], mask: [UInt8]
  ) throws -> [DetectorUpdateResult] {
    try updateVirtualDetectors(
      sources: sources, masks: Array(repeating: mask, count: sources.count))
  }

  /// Submit source-specific detector masks for every unique resident.
  @_spi(PairedRuntimeTANSPrototype)
  public static func updateVirtualDetectors(
    sources: [MetalPairedRuntimeTANSResidentSource], masks: [[UInt8]]
  ) throws -> [DetectorUpdateResult] {
    guard sources.count == masks.count else {
      throw Self.invalid("A paired-runtime detector batch requires one mask per resident")
    }
    var ordered = Array(sources.enumerated())
    guard Set(sources.map { ObjectIdentifier($0) }).count == sources.count else {
      throw Self.invalid("A paired-runtime detector batch cannot contain duplicate residents")
    }
    ordered.sort {
      UInt(bitPattern: Unmanaged.passUnretained($0.element).toOpaque())
        < UInt(bitPattern: Unmanaged.passUnretained($1.element).toOpaque())
    }
    for entry in ordered { entry.element.stateLock.lock() }
    defer { for entry in ordered.reversed() { entry.element.stateLock.unlock() } }
    for entry in ordered where entry.element.blockStride > 1 {
      _ = try entry.element.settleBlockStride(to: masks[entry.offset])
    }

    var pending:
      [(
        offset: Int, resident: MetalPairedRuntimeTANSResidentSource,
        update: PendingDetectorUpdate
      )] = []
    do {
      for entry in ordered {
        pending.append(
          (
            entry.offset, entry.element,
            try entry.element.prepareDetectorUpdate(mask: masks[entry.offset])
          ))
      }
    } catch {
      for item in pending { item.resident.rollbackPreparedDetectorUpdate(item.update) }
      throw error
    }

    // Commit every resident before waiting on any one resident.
    var commitTimes: [ObjectIdentifier: Double] = [:]
    for item in pending {
      let commitTime = item.update.profile == nil ? 0 : ProcessInfo.processInfo.systemUptime
      if item.update.command != nil { item.update.stageSession?.willCommit() }
      item.update.command?.commit()
      if commitTime != 0 { commitTimes[ObjectIdentifier(item.resident)] = commitTime }
    }
    var results = [DetectorUpdateResult?](repeating: nil, count: sources.count)
    var firstError: Error?
    for item in pending {
      do {
        results[item.offset] = try item.resident.finishDetectorUpdate(
          item.update, commit: false, commitHost: commitTimes[ObjectIdentifier(item.resident)])
      } catch {
        firstError = firstError ?? error
      }
    }
    if let firstError { throw firstError }
    return results.compactMap { $0 }
  }

  /// Return process-wide polar-plan cache counters for benchmark diagnostics.
  @_spi(PairedRuntimeTANSPrototype)
  public static func polarPlanCacheProfileSnapshot() -> [String: Double] {
    PairedRuntimeTANSPolarPlan.cacheProfileSnapshot()
  }

  private func prepareDetectorUpdate(mask: [UInt8]) throws -> PendingDetectorUpdate {
    let profileEnabled = runtimeOption("QGPU_PAIRED_RUNTIME_PROFILE") == "1"
    let requestEntry = profileEnabled ? ProcessInfo.processInfo.systemUptime : 0
    let preparationStart = profileEnabled ? ProcessInfo.processInfo.systemUptime : 0
    var profile =
      profileEnabled
      ? UpdateProfile(requestEntry: requestEntry, preparationStart: preparationStart)
      : nil
    let stageSession = profileEnabled ? detectorProfiler?.makeSession() : nil
    try requireLive()
    let historyActive =
      historyEnabled
      && runtimeOption("QGPU_PAIRED_RUNTIME_HISTORY") == "1"
    let previousHistoryValid = historyValid
    let previousPolarFields = polarFieldCount
    let previousPolarResiduals = polarResidualCount
    var preparationSucceeded = false
    defer {
      if historyActive && !preparationSucceeded {
        historyValid = previousHistoryValid
        polarFieldCount = previousPolarFields
        polarResidualCount = previousPolarResiduals
        refreshMetadataSnapshot()
      }
    }
    if !historyActive { historyValid = false }
    metadataLock.lock()
    cachedHistoryHit = false
    cachedHistoryBase = false
    metadataLock.unlock()
    if profileEnabled {
      metadataLock.lock()
      cachedUpdateProfile = [:]
      metadataLock.unlock()
    }
    let pixels = shape[2] * shape[3]
    if profileEnabled { profile?.planningStart = ProcessInfo.processInfo.systemUptime }
    guard mask.count == pixels,
      mask.withUnsafeBufferPointer({ values in !values.contains(where: { $0 > 1 }) })
    else {
      throw Self.invalid("The paired-runtime detector mask must contain one binary value per pixel")
    }
    // Opt-in raw-compute mode: every update clears its output and reconstructs
    // the requested mask from the compressed resident. It never adds a signed
    // change to the previous detector image, even when the mask is unchanged.
    let fromZeroValue = runtimeOption("QGPU_PAIRED_RUNTIME_FROM_ZERO") ?? "0"
    guard fromZeroValue == "0" || fromZeroValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_FROM_ZERO must be 0 or 1")
    }
    let fromZero = fromZeroValue == "1"
    guard !fromZero || !historyActive else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_FROM_ZERO=1 cannot be combined with detector history")
    }
    var selected: [UInt32] = []
    var coefficients: [Int32] = []
    selected.reserveCapacity(pixels)
    coefficients.reserveCapacity(pixels)
    if fromZero {
      for pixel in 0..<pixels where mask[pixel] != 0 {
        selected.append(UInt32(pixel))
        coefficients.append(1)
      }
    } else {
      // Read the stored mask through local buffers: per-element access to the class
      // property inside this 36,864-pixel loop dominated host preparation.
      let currentMask = detectorMask
      mask.withUnsafeBufferPointer { next in
        currentMask.withUnsafeBufferPointer { current in
          for pixel in 0..<pixels where next[pixel] != current[pixel] {
            selected.append(UInt32(pixel))
            coefficients.append(next[pixel] == 1 ? 1 : -1)
          }
        }
      }
    }
    guard !selected.isEmpty || fromZero else {
      if profileEnabled {
        let completed = ProcessInfo.processInfo.systemUptime
        profile?.planningEnd = completed
        profile?.preparationEnd = completed
      }
      preparationSucceeded = true
      return PendingDetectorUpdate(
        command: nil, mask: mask, previousMask: detectorMask, started: 0,
        changedCount: 0, operation: "detector", candidateProduct: nil,
        historyHit: false, historyBase: false, previousHistoryValid: previousHistoryValid,
        previousPolarFieldCount: previousPolarFields,
        previousPolarResidualCount: previousPolarResiduals, profile: profile,
        stageSession: stageSession, keepAlive: [])
    }
    let changedCount = selected.count
    let previousMask = detectorMask
    if historyActive, historyValid, historyMask == mask {
      if profileEnabled {
        let completed = ProcessInfo.processInfo.systemUptime
        profile?.planningEnd = completed
        profile?.preparationEnd = completed
      }
      preparationSucceeded = true
      return PendingDetectorUpdate(
        command: nil, mask: mask, previousMask: previousMask, started: 0,
        changedCount: changedCount, operation: "detector history", candidateProduct: nil,
        historyHit: true, historyBase: false, previousHistoryValid: previousHistoryValid,
        previousPolarFieldCount: previousPolarFields,
        previousPolarResidualCount: previousPolarResiduals, profile: profile,
        stageSession: stageSession, keepAlive: [])
    }
    var polarPlan: PairedRuntimeTANSPolarPlan?
    var startFromZero = false
    var usingHistoryBase = false
    let chooseBaseValue = runtimeOption("QGPU_PAIRED_RUNTIME_CHOOSE_BASE") ?? "0"
    guard chooseBaseValue == "0" || chooseBaseValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_CHOOSE_BASE must be 0 or 1")
    }
    let historyBaseValue = runtimeOption("QGPU_PAIRED_RUNTIME_HISTORY_BASE") ?? "0"
    guard historyBaseValue == "0" || historyBaseValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_HISTORY_BASE must be 0 or 1")
    }
    let requestHistoryBase = historyBaseValue == "1"
    guard !requestHistoryBase || historyActive else {
      throw Self.invalid(
        "QGPU_PAIRED_RUNTIME_HISTORY_BASE=1 requires QGPU_PAIRED_RUNTIME_HISTORY=1 and startup history storage"
      )
    }
    // The public low-level mask API can still include invalid detector pixels.
    // Its exact raw-mask semantics use the direct path; indexed fields exclude
    // invalid pixels and are used only when both masks follow that policy.
    let indexRequested =
      polarIndex != nil
      && runtimeOption("QGPU_PAIRED_RUNTIME_POLAR_INDEX") == "1"
    let currentMaskForIndex = detectorMask
    let currentIndexAllowed =
      indexRequested
      && !validPixels.withUnsafeBufferPointer { valid in
        mask.withUnsafeBufferPointer { next in
          currentMaskForIndex.withUnsafeBufferPointer { current in
            (0..<pixels).contains(where: { valid[$0] == 0 && (next[$0] != 0 || current[$0] != 0) })
          }
        }
      }
    var polarQueryVariantValue =
      runtimeOption("QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT") ?? "packet-groups"
    // A defaulted scan512 query quietly uses the ordinary query when this request
    // cannot use the prepared index (index off, invalid pixels in the mask, or a
    // non-512 scan); an explicitly requested variant still fails loudly.
    if !runtimeOptionIsExplicit("QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT"),
      polarQueryVariantValue == "scan512",
      !(polarIndex?.queryPipelinePrepared(for: .scan512) == true && currentIndexAllowed
        && shape[0] == 512 && shape[1] == 512)
    {
      polarQueryVariantValue = "packet-groups"
    }
    guard
      let polarQueryVariant = MetalPairedRuntimeTANSPolarIndex.QueryVariant(
        rawValue: polarQueryVariantValue)
    else {
      throw Self.invalid(
        "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT must be packet-groups, scan512, "
          + "scan512-stripe2, scan512-stripe4, scan512-stripe8, scan512-field4, "
          + "scan512-contiguous-quad, or packet-major")
    }
    if polarQueryVariant.requiresPreparedQueryPipeline {
      let baselineSettings: [(String, String)] = [
        ("QGPU_PAIRED_RUNTIME_DETECTOR_KERNEL", "packet-owner2"),
        ("QGPU_PAIRED_RUNTIME_STREAMS_PER_LANE", "2"),
        ("QGPU_PAIRED_RUNTIME_SPARSE_SPLIT", "0"),
        ("QGPU_PAIRED_RUNTIME_DENSE_COMPACTION", "0"),
        ("QGPU_PAIRED_RUNTIME_MACRO", "0"),
        ("QGPU_PAIRED_RUNTIME_COOPERATIVE", "0"),
        ("QGPU_PAIRED_RUNTIME_READER32", "0"),
        ("QGPU_PAIRED_RUNTIME_REUSE_WORD", "0"),
        ("QGPU_PAIRED_RUNTIME_LAZY_REFILL", "0"),
        ("QGPU_PAIRED_RUNTIME_SIMD_ENTROPY_FAST_PATH", "0"),
        ("QGPU_PAIRED_RUNTIME_JOINT_PLAN", "0"),
        ("QGPU_PAIRED_RUNTIME_HISTORY", "0"),
        ("QGPU_PAIRED_RUNTIME_HISTORY_BASE", "0"),
        ("QGPU_PAIRED_RUNTIME_CHOOSE_BASE", "0"),
      ]
      // scan512 changes polar-query planning, while trusted-table changes only
      // the already-validated tANS state-bound check in the detector pipeline.
      // Keep the other specializations isolated but allow these independent
      // stages to compose; the detector path retains its own compatibility
      // checks below.
      let packetSplitsSetting = runtimeOption("QGPU_PAIRED_RUNTIME_PACKET_SPLITS") ?? "1"
      let trustedTableSetting = runtimeOption("QGPU_PAIRED_RUNTIME_TRUSTED_TABLE") ?? "0"
      let split4TrustedTableComposition =
        polarQueryVariant == .scan512
        && packetSplitsSetting == "4" && trustedTableSetting == "1"
        && detectorTrustedTableSplit4Pipeline != nil
      let split8TrustedTableComposition =
        polarQueryVariant == .scan512
        && packetSplitsSetting == "8" && trustedTableSetting == "1"
        && detectorTrustedTableSplit8Pipeline != nil
      guard polarIndex?.queryPipelinePrepared(for: polarQueryVariant) == true,
        currentIndexAllowed, shape[0] == 512, shape[1] == 512,
        packetSplitsSetting == "1" || split4TrustedTableComposition
          || split8TrustedTableComposition,
        baselineSettings.allSatisfy({
          (runtimeOption($0.0) ?? $0.1) == $0.1
        })
      else {
        throw Self.invalid(
          "The polar-query variant requires its prepared pipeline, a prepared "
            + "index, and a 512×512 scan, "
            + "and the indexed packet-owner2 baseline (2 streams/lane, "
            + "with only explicitly prepared scan512/trusted-table/split-4 or split-8 compositions allowed; "
            + "joint plan/history and other detector specializations off)")
      }
    }
    let historyIndexAllowed =
      currentIndexAllowed && historyValid
      && historyMask != nil
      && !validPixels.indices.contains(where: {
        validPixels[$0] == 0 && (mask[$0] != 0 || historyMask![$0] != 0)
      })
    if let polarIndex, currentIndexAllowed {
      var delta = [Int32](repeating: 0, count: pixels)
      for (pixel, coefficient) in zip(selected, coefficients) {
        delta[Int(pixel)] = coefficient
      }
      let previousPlan = PairedRuntimeTANSPolarPlan.make(
        delta: delta, validPixels: validPixels,
        detectorRows: shape[2], detectorColumns: shape[3], leafPixels: polarIndex.leafPixels,
        layoutKind: polarIndex.layoutKind)
      var plan = previousPlan
      var selectedCost = previousPlan.estimatedCost
      if fromZero {
        // `delta` above already holds the absolute target mask in this mode.
        startFromZero = true
      }
      if chooseBaseValue == "1" && !fromZero {
        let absolute = mask.map(Int32.init)
        let zeroPlan = PairedRuntimeTANSPolarPlan.make(
          delta: absolute, validPixels: validPixels,
          detectorRows: shape[2], detectorColumns: shape[3], leafPixels: polarIndex.leafPixels,
          layoutKind: polarIndex.layoutKind)
        // Charge one field-equivalent for clearing the complete output. A tie
        // stays on the previous base, avoiding an unnecessary full-buffer write.
        if zeroPlan.estimatedCost + 1 < previousPlan.estimatedCost {
          plan = zeroPlan
          startFromZero = true
          selectedCost = zeroPlan.estimatedCost + 1  // account for clearing the candidate output
        }
      }
      if requestHistoryBase, historyIndexAllowed, !fromZero {
        var historyDelta = [Int32](repeating: 0, count: pixels)
        for pixel in 0..<pixels where mask[pixel] != historyMask![pixel] {
          historyDelta[pixel] = mask[pixel] == 1 ? 1 : -1
        }
        let historyPlan = PairedRuntimeTANSPolarPlan.make(
          delta: historyDelta, validPixels: validPixels,
          detectorRows: shape[2], detectorColumns: shape[3], leafPixels: polarIndex.leafPixels,
          layoutKind: polarIndex.layoutKind)
        let historyCost = historyPlan.estimatedCost
        let currentCost = previousPlan.estimatedCost
        let paretoDominates = {
          (
            candidate: PairedRuntimeTANSPolarPlan,
            baseline: PairedRuntimeTANSPolarPlan
          ) -> Bool in
          let residualNoMore = candidate.residualPixels.count <= baseline.residualPixels.count
          let fieldsNoMore = candidate.selectedFields.count <= baseline.selectedFields.count
          let oneStrict =
            candidate.residualPixels.count < baseline.residualPixels.count
            || candidate.selectedFields.count < baseline.selectedFields.count
          return residualNoMore && fieldsNoMore && oneStrict
        }
        // The planner's scalar estimate can trade many residual pixels for a
        // few fields too aggressively. Require a Pareto improvement over both
        // the current plan and a cheaper zero-based plan, if one was selected.
        if historyCost < currentCost && historyCost < selectedCost
          && paretoDominates(historyPlan, previousPlan)
          && paretoDominates(historyPlan, plan)
        {
          plan = historyPlan
          startFromZero = false
          usingHistoryBase = true
          selectedCost = historyCost
        }
      }
      if plan.usedIndex {
        polarPlan = plan
        selected = plan.residualPixels
        coefficients = plan.residualCoefficients
      } else if startFromZero || usingHistoryBase {
        selected = plan.residualPixels
        coefficients = plan.residualCoefficients
      }
    } else if fromZero {
      startFromZero = true
    } else if requestHistoryBase, historyValid, let historyMask {
      var historySelected: [UInt32] = []
      var historyCoefficients: [Int32] = []
      historySelected.reserveCapacity(changedCount)
      historyCoefficients.reserveCapacity(changedCount)
      for pixel in 0..<pixels where mask[pixel] != historyMask[pixel] {
        historySelected.append(UInt32(pixel))
        historyCoefficients.append(mask[pixel] == 1 ? 1 : -1)
      }
      // Raw mode charges one work item per changed detector pixel. Ties stay
      // on the current product so the history path is strictly opt-in.
      if historySelected.count < selected.count {
        selected = historySelected
        coefficients = historyCoefficients
        usingHistoryBase = true
      }
    }
    if profileEnabled { profile?.planningEnd = ProcessInfo.processInfo.systemUptime }
    // Opt-in exact reordering of residual pixels. The detector kernels run lanes
    // in lockstep SIMD groups; row-major order mixes dense small-radius pixels
    // with sparse large-radius pixels in the same group, so sparse lanes wait
    // for dense ones. Sorting by detector radius groups similar work. Each pixel
    // keeps its own coefficient, so the summed image is unchanged.
    let residualOrderValue = runtimeOption("QGPU_PAIRED_RUNTIME_RESIDUAL_ORDER") ?? "index"
    guard ["index", "radius", "rank"].contains(residualOrderValue) else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_RESIDUAL_ORDER must be index, radius, or rank")
    }
    if residualOrderValue == "rank", selected.count > 1 {
      // Stream-rank order follows the payload layout, so a chunk's streams are adjacent.
      guard
        let layout = PairedRuntimeTANSPolarPlan.indexLayout(leafPixels: 16, layoutKind: "radial1")
      else { throw Self.invalid("Rank residual order requires the radial1 layout") }
      let rankOf: [UInt32]
      if let streamRankOfPixel {
        rankOf = streamRankOfPixel
      } else {
        var ranks = [UInt32](repeating: 0, count: shape[2] * shape[3])
        for (rank, pixel) in layout.permutation.enumerated() where pixel >= 0 {
          ranks[Int(pixel)] = UInt32(rank)
        }
        rankOf = ranks
      }
      let order = selected.indices.sorted { rankOf[Int(selected[$0])] < rankOf[Int(selected[$1])] }
      selected = order.map { selected[$0] }
      coefficients = order.map { coefficients[$0] }
    }
    if residualOrderValue == "radius", selected.count > 1 {
      let columns = shape[3]
      let centerRow = shape[2] / 2
      let centerColumn = columns / 2
      let order = selected.indices.sorted { left, right in
        let a = Int(selected[left])
        let b = Int(selected[right])
        let ay = a / columns - centerRow
        let ax = a % columns - centerColumn
        let by = b / columns - centerRow
        let bx = b % columns - centerColumn
        let ra = ay * ay + ax * ax
        let rb = by * by + bx * bx
        return ra == rb ? a < b : ra < rb
      }
      selected = order.map { selected[$0] }
      coefficients = order.map { coefficients[$0] }
    }
    polarFieldCount = polarPlan?.selectedFields.count ?? 0
    polarResidualCount = selected.count
    refreshMetadataSnapshot()
    let selectedBuffer = try Self.upload(
      selected.isEmpty ? [0] : streamAddresses(selected),
      device: detectorProduct.device, label: "paired-runtime selected pixels")
    let coefficientBuffer = try Self.upload(
      coefficients.isEmpty ? [0] : coefficients,
      device: detectorProduct.device, label: "paired-runtime coefficients")
    let outputProduct: MTLBuffer
    if historyActive {
      guard let historyProduct else {
        throw Self.invalid("Paired-runtime history storage is unavailable")
      }
      outputProduct = historyProduct
    } else {
      outputProduct = detectorProduct
    }
    memset(failure.contents(), 0, 4)
    guard let command = queue.makeCommandBuffer() else {
      throw Self.invalid("Metal could not encode a paired-runtime detector query")
    }
    let started = CFAbsoluteTimeGetCurrent()
    if historyActive && !startFromZero && !usingHistoryBase {
      guard let blit = command.makeBlitCommandEncoder() else {
        throw Self.invalid("Metal could not copy the paired-runtime detector history")
      }
      blit.copy(
        from: detectorProduct, sourceOffset: 0, to: outputProduct, destinationOffset: 0,
        size: detectorProduct.length)
      blit.endEncoding()
    }
    if startFromZero {
      guard let blit = command.makeBlitCommandEncoder() else {
        throw Self.invalid("Metal could not clear a zero-based paired-runtime detector query")
      }
      blit.fill(buffer: outputProduct, range: 0..<outputProduct.length, value: 0)
      blit.endEncoding()
    }
    if let polarPlan, let polarIndex {
      try polarIndex.encode(
        plan: polarPlan, output: outputProduct, failure: failure, command: command,
        variant: polarQueryVariant, profiler: stageSession,
        packetStride: activePacketStride, packetPhase: activePacketPhase)
    }
    let packets = shape[0] * shape[1] / PairedRuntimeTANSRecordABI.streamScans
    if (polarPlan != nil || startFromZero) && selected.isEmpty {
      if profileEnabled { profile?.preparationEnd = ProcessInfo.processInfo.systemUptime }
      preparationSucceeded = true
      return PendingDetectorUpdate(
        command: command, mask: mask, previousMask: previousMask, started: started,
        changedCount: changedCount, operation: "polar-only detector",
        candidateProduct: historyActive ? outputProduct : nil, historyHit: false,
        historyBase: usingHistoryBase,
        previousHistoryValid: previousHistoryValid,
        previousPolarFieldCount: previousPolarFields,
        previousPolarResidualCount: previousPolarResiduals,
        profile: profile,
        stageSession: stageSession,
        keepAlive: [selectedBuffer, coefficientBuffer])
    }

    let detectorKernel = runtimeOption("QGPU_PAIRED_RUNTIME_DETECTOR_KERNEL") ?? "packet-owner2"
    guard ["packet-owner2", "partials", "adaptive-partials"].contains(detectorKernel) else {
      throw Self.invalid(
        "QGPU_PAIRED_RUNTIME_DETECTOR_KERNEL must be packet-owner2, partials, or adaptive-partials")
    }
    let simdEntropyFastPathValue =
      runtimeOption("QGPU_PAIRED_RUNTIME_SIMD_ENTROPY_FAST_PATH") ?? "0"
    guard simdEntropyFastPathValue == "0" || simdEntropyFastPathValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_SIMD_ENTROPY_FAST_PATH must be 0 or 1")
    }
    let simdEntropyFastPath = simdEntropyFastPathValue == "1"
    guard !simdEntropyFastPath || detectorKernel == "packet-owner2" else {
      throw Self.invalid("SIMD entropy fast path requires packet-owner2")
    }
    let lazyRefillValue = runtimeOption("QGPU_PAIRED_RUNTIME_LAZY_REFILL") ?? "0"
    guard lazyRefillValue == "0" || lazyRefillValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_LAZY_REFILL must be 0 or 1")
    }
    let lazyRefill = lazyRefillValue == "1"
    guard !lazyRefill || detectorKernel == "packet-owner2" else {
      throw Self.invalid("Lazy refill requires the ordinary packet-owner2 kernel")
    }
    let trustedTableValue = runtimeOption("QGPU_PAIRED_RUNTIME_TRUSTED_TABLE") ?? "0"
    guard trustedTableValue == "0" || trustedTableValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_TRUSTED_TABLE must be 0 or 1")
    }
    let trustedTable = trustedTableValue == "1"
    guard !trustedTable || detectorKernel == "packet-owner2" else {
      throw Self.invalid("Trusted table mode requires the packet-owner2 kernel")
    }
    let vectorPairReductionValue = runtimeOption("QGPU_PAIRED_RUNTIME_VECTOR_PAIR_REDUCTION") ?? "0"
    guard vectorPairReductionValue == "0" || vectorPairReductionValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_VECTOR_PAIR_REDUCTION must be 0 or 1")
    }
    let vectorPairReduction = vectorPairReductionValue == "1"
    let windowReaderValue = runtimeOption("QGPU_PAIRED_RUNTIME_WINDOW_READER") ?? "0"
    guard windowReaderValue == "0" || windowReaderValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_WINDOW_READER must be 0 or 1")
    }
    let windowReader = windowReaderValue == "1"
    guard !windowReader || (detectorKernel == "packet-owner2" && trustedTable) else {
      throw Self.invalid("The window reader requires the trusted-table packet-owner2 kernel")
    }
    let partialMaximumGroupsValue = runtimeOption("QGPU_PAIRED_RUNTIME_PARTIAL_MAX_GROUPS") ?? "8"
    guard let partialMaximumGroups = Int(partialMaximumGroupsValue),
      [8, 16, 32].contains(partialMaximumGroups)
    else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_PARTIAL_MAX_GROUPS must be 8, 16, or 32")
    }
    let adaptiveGroups = (selected.count + 63) / 64
    let requestedPartialBytes = adaptiveGroups * packets * 512 * MemoryLayout<UInt32>.stride
    if activePacketStride > 1
      && (detectorKernel == "partials" || detectorKernel == "adaptive-partials")
    {
      throw Self.invalid("Block stride requires the ordinary two-stream packet-owner2 kernel")
    }
    if requestedPartialBytes <= 128 * 1024 * 1024
      && (detectorKernel == "partials"
        || (detectorKernel == "adaptive-partials" && !selected.isEmpty
          && adaptiveGroups <= partialMaximumGroups))
    {
      // The partial kernel gives each 64-pixel block its own SIMD group and
      // removes the packet-owner atomics from the hot loop. Scratch is lazy
      // and reused, so ordinary loads keep their compact resident footprint.
      let groups = adaptiveGroups
      let partialBytes = groups * packets * 512 * MemoryLayout<UInt32>.stride
      if detectorPartials == nil || detectorPartials!.length < partialBytes {
        guard
          let next = detectorProduct.device.makeBuffer(
            length: max(partialBytes, MemoryLayout<UInt32>.stride), options: .storageModeShared)
        else { throw Self.invalid("Metal could not allocate exact detector partials") }
        next.label = "paired-runtime detector partials"
        detectorPartials = next
        refreshMetadataSnapshot()
      }
      guard let partials = detectorPartials,
        let encoder = stageSession?.makeComputeEncoder(
          commandBuffer: command, stage: "residual_partials")
          ?? command.makeComputeCommandEncoder()
      else { throw Self.invalid("Metal could not encode exact detector partials") }
      // This scratch uses shared storage and is consumed only after the
      // command is committed. Clearing it here avoids an additional blit
      // encoder, which older AGX drivers can reject before a compute encoder.
      let usePartialStores = runtimeOption("QGPU_PAIRED_RUNTIME_PARTIAL_STORES") == "1"
      if usePartialStores && detectorPartialStoresPipeline == nil {
        throw Self.invalid("Prepare the partial-store pipeline before loading the resident")
      }
      if !usePartialStores { memset(partials.contents(), 0, partialBytes) }
      var partialParameters: [UInt32] = [
        UInt32(pixels), UInt32(packets), UInt32(selected.count), UInt32(groups),
        UInt32(payload.length),
      ]
      // Keep both dispatches in one compute encoder. AGX can reject adjacent
      // compute encoders in one command buffer while coalescing work; changing
      // the pipeline and bindings is equivalent and avoids that driver path.
      encoder.setComputePipelineState(
        usePartialStores ? detectorPartialStoresPipeline! : detectorPartialsPipeline)
      for (index, buffer) in [
        payload, offsets, modes, decodingTable, selectedBuffer, coefficientBuffer,
        partials, failure,
      ].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBytes(
        &partialParameters, length: partialParameters.count * MemoryLayout<UInt32>.stride,
        index: 8)
      encoder.dispatchThreadgroups(
        MTLSize(width: groups, height: packets, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
      // Keep both dispatches in their original encoder while profiling.
      let finishEncoder = encoder
      var finishParameters: [UInt32] = [UInt32(packets), UInt32(groups)]
      finishEncoder.setComputePipelineState(detectorFinishPipeline)
      finishEncoder.setBuffer(partials, offset: 0, index: 0)
      finishEncoder.setBuffer(outputProduct, offset: 0, index: 1)
      finishEncoder.setBytes(
        &finishParameters, length: finishParameters.count * MemoryLayout<UInt32>.stride,
        index: 2)
      let finishWidth = detectorFinishPipeline.threadExecutionWidth
      finishEncoder.dispatchThreads(
        MTLSize(width: packets * 512, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(
          width: min(detectorFinishPipeline.maxTotalThreadsPerThreadgroup, finishWidth * 4),
          height: 1, depth: 1))
      finishEncoder.endEncoding()
      if profileEnabled { profile?.preparationEnd = ProcessInfo.processInfo.systemUptime }
      preparationSucceeded = true
      return PendingDetectorUpdate(
        command: command, mask: mask, previousMask: previousMask, started: started,
        changedCount: changedCount, operation: "detector partials",
        candidateProduct: historyActive ? outputProduct : nil, historyHit: false,
        historyBase: usingHistoryBase,
        previousHistoryValid: previousHistoryValid,
        previousPolarFieldCount: previousPolarFields,
        previousPolarResidualCount: previousPolarResiduals,
        profile: profile,
        stageSession: stageSession,
        keepAlive: [selectedBuffer, coefficientBuffer])
    }

    guard
      let encoder = stageSession?.makeComputeEncoder(
        commandBuffer: command, stage: "residual") ?? command.makeComputeCommandEncoder()
    else {
      throw Self.invalid("Metal could not encode a paired-runtime detector update")
    }
    let streamsPerLane = runtimeOption("QGPU_PAIRED_RUNTIME_STREAMS_PER_LANE") ?? "2"
    guard streamsPerLane == "1" || streamsPerLane == "2" || streamsPerLane == "4" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_STREAMS_PER_LANE must be 1, 2, or 4")
    }
    let sparseSplitValue = runtimeOption("QGPU_PAIRED_RUNTIME_SPARSE_SPLIT") ?? "0"
    guard sparseSplitValue == "0" || sparseSplitValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_SPARSE_SPLIT must be 0 or 1")
    }
    let sparseSplit = sparseSplitValue == "1"
    let denseCompactionValue = runtimeOption("QGPU_PAIRED_RUNTIME_DENSE_COMPACTION") ?? "0"
    guard denseCompactionValue == "0" || denseCompactionValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_DENSE_COMPACTION must be 0 or 1")
    }
    let denseCompaction = denseCompactionValue == "1"
    guard !denseCompaction || sparseSplit else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_DENSE_COMPACTION=1 requires sparse split")
    }
    let plainScratchValue = runtimeOption("QGPU_PAIRED_RUNTIME_PLAIN_SCRATCH") ?? "0"
    guard plainScratchValue == "0" || plainScratchValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_PLAIN_SCRATCH must be 0 or 1")
    }
    let macroValue = runtimeOption("QGPU_PAIRED_RUNTIME_MACRO") ?? "0"
    guard macroValue == "0" || macroValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_MACRO must be 0 or 1")
    }
    let macro = macroValue == "1"
    let macroLookaheadValue = runtimeOption("QGPU_PAIRED_RUNTIME_MACRO_LOOKAHEAD_BITS") ?? "4"
    guard let macroLookahead = Int(macroLookaheadValue), macroLookahead == 2 || macroLookahead == 4
    else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_MACRO_LOOKAHEAD_BITS must be 2 or 4")
    }
    guard !macro || macroLookaheadBits == macroLookahead else {
      throw Self.invalid(
        "Macro lookahead width changed after resident initialization; rebuild residents with "
          + "the requested table width")
    }
    guard !macro || !denseCompaction else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_MACRO=1 requires dense compaction off")
    }
    let cooperativeValue = runtimeOption("QGPU_PAIRED_RUNTIME_COOPERATIVE") ?? "0"
    guard cooperativeValue == "0" || cooperativeValue == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_COOPERATIVE must be 0 or 1")
    }
    let cooperative = cooperativeValue == "1"
    guard !cooperative || (sparseSplit && !macro && !denseCompaction) else {
      throw Self.invalid(
        "QGPU_PAIRED_RUNTIME_COOPERATIVE=1 requires sparse split and disables macro/dense")
    }
    let reader32Value = runtimeOption("QGPU_PAIRED_RUNTIME_READER32") ?? "0"
    guard reader32Value == "0" || reader32Value == "1" else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_READER32 must be 0 or 1")
    }
    let reader32 = reader32Value == "1"
    guard !reader32 || (!cooperative && !macro && !denseCompaction) else {
      throw Self.invalid("QGPU_PAIRED_RUNTIME_READER32=1 requires cooperative/macro/dense off")
    }
    guard let packetSplits = Int(runtimeOption("QGPU_PAIRED_RUNTIME_PACKET_SPLITS") ?? "1"),
      [1, 2, 4, 8].contains(packetSplits)
    else { throw Self.invalid("Packet splits must be 1, 2, 4, or 8") }
    guard
      packetSplits == 1
        || (streamsPerLane == "2" && !reader32 && !cooperative
          && !macro && !denseCompaction && !sparseSplit
          && detectorSplitPipelines[packetSplits] != nil)
    else {
      throw Self.invalid("Prepare packet-split pipelines and disable other packet specializations")
    }
    var parameters: [UInt32] = [
      UInt32(pixels), UInt32(packets), UInt32(selected.count), 0,
      UInt32(payload.length), sparseSplit ? 0 : 1,
      UInt32(activePacketStride), UInt32(activePacketPhase),
    ]
    guard
      activePacketStride == 1
        || (detectorKernel == "packet-owner2" && streamsPerLane == "2" && packetSplits == 1
          && !cooperative && !sparseSplit && !denseCompaction && !macro)
    else {
      throw Self.invalid("Block stride requires the ordinary two-stream packet-owner2 kernel")
    }
    let logicalPackets = (packets - activePacketPhase + activePacketStride - 1) / activePacketStride
    let reuseWord = runtimeOption("QGPU_PAIRED_RUNTIME_REUSE_WORD") == "1"
    let registerSums = runtimeOption("QGPU_PAIRED_RUNTIME_REGISTER_SUMS") == "1"
    let plainSums = runtimeOption("QGPU_PAIRED_RUNTIME_PLAIN_SUMS") == "1"
    guard
      !plainSums
        || (streamsPerLane == "2" && packetSplits == 1 && !reader32
          && !cooperative && !macro && !denseCompaction && !sparseSplit && !reuseWord
          && !registerSums)
    else { throw Self.invalid("Plain sums require the ordinary two-stream packet kernel") }
    guard
      !registerSums
        || (streamsPerLane == "2" && packetSplits == 1 && !reader32
          && !cooperative && !macro && !denseCompaction && !sparseSplit && !reuseWord)
    else { throw Self.invalid("Register sums require the ordinary two-stream packet kernel") }
    guard
      !reuseWord
        || (streamsPerLane == "2" && packetSplits == 1 && !reader32
          && !cooperative && !macro && !denseCompaction && !sparseSplit)
    else { throw Self.invalid("Word reuse requires the ordinary two-stream packet kernel") }
    guard
      !lazyRefill
        || (streamsPerLane == "2" && packetSplits == 1 && !reader32
          && !cooperative && !macro && !denseCompaction && !sparseSplit && !reuseWord
          && !registerSums && !plainSums && !trustedTable && detectorLazyRefillPipeline != nil)
    else {
      throw Self.invalid(
        "Lazy refill requires its prepared ordinary two-stream packet-owner2 pipeline")
    }
    guard
      !trustedTable
        || (streamsPerLane == "2"
          && (packetSplits == 1
            || (packetSplits == 4 && detectorTrustedTableSplit4Pipeline != nil)
            || (packetSplits == 8 && detectorTrustedTableSplit8Pipeline != nil))
          && !reader32
          && !cooperative && !macro && !denseCompaction && !sparseSplit && !reuseWord
          && (!registerSums || (packetSplits == 1 && detectorTrustedRegisterSumsPipeline != nil))
          && (!plainSums || (packetSplits == 1 && detectorTrustedPlainSumsPipeline != nil))
          && !(registerSums && plainSums))
    else {
      throw Self.invalid(
        "Trusted table requires the ordinary two-stream packet kernel or its prepared split-4/split-8, plain-sums or register-sums composition"
      )
    }
    guard
      !simdEntropyFastPath
        || (streamsPerLane == "2" && packetSplits == 1
          && !reader32 && !cooperative && !macro && !denseCompaction && !sparseSplit
          && !reuseWord && !registerSums && !plainSums && !lazyRefill && !trustedTable
          && runtimeOption("QGPU_PAIRED_RUNTIME_JOINT_PLAN") != "1"
          && runtimeOption("QGPU_PAIRED_RUNTIME_HISTORY") != "1")
    else {
      throw Self.invalid(
        "SIMD entropy fast path is an isolated packet-owner2 experiment; disable other specializations"
      )
    }
    guard
      !vectorPairReduction
        || (detectorKernel == "packet-owner2"
          && streamsPerLane == "2" && packetSplits == 1 && !reader32 && !cooperative
          && !macro && !denseCompaction && !sparseSplit && !reuseWord && !registerSums
          && !plainSums && !lazyRefill && !simdEntropyFastPath
          && runtimeOption("QGPU_PAIRED_RUNTIME_JOINT_PLAN") != "1"
          && runtimeOption("QGPU_PAIRED_RUNTIME_HISTORY") != "1")
    else {
      throw Self.invalid(
        "Packed pair reduction requires the ordinary indexed two-stream packet-owner2 path")
    }
    guard
      !windowReader
        || (detectorTrustedWindowReaderPipeline != nil
          && streamsPerLane == "2" && packetSplits == 1 && !reader32 && !cooperative
          && !macro && !denseCompaction && !sparseSplit && !reuseWord && !registerSums
          && (!plainSums || detectorTrustedWindowReaderPlainPipeline != nil)
          && !lazyRefill && !simdEntropyFastPath && !vectorPairReduction)
    else {
      throw Self.invalid(
        "The window reader requires its prepared trusted-table two-stream packet-owner2 "
          + "pipeline with other detector specializations disabled")
    }
    if vectorPairReduction && trustedTable {
      guard let detectorTrustedVectorPairReductionPipeline else {
        throw Self.invalid(
          "Prepare the trusted-table vector-pair reduction pipeline before testing it")
      }
      encoder.setComputePipelineState(detectorTrustedVectorPairReductionPipeline)
    } else if vectorPairReduction {
      guard let detectorVectorPairReductionPipeline else {
        throw Self.invalid("Prepare the vector-pair reduction pipeline before testing it")
      }
      encoder.setComputePipelineState(detectorVectorPairReductionPipeline)
    } else if simdEntropyFastPath {
      guard let detectorSIMDEntropyFastPathPipeline else {
        throw Self.invalid(
          "Prepare the SIMD entropy fast-path pipeline before loading the resident")
      }
      encoder.setComputePipelineState(detectorSIMDEntropyFastPathPipeline)
    } else if windowReader {
      let diagLevel = Int(runtimeOption("QGPU_PAIRED_RUNTIME_WINDOW_DIAG") ?? "0") ?? 0
      // Defaulted trusted setup applies only once the validated index exists.
      let trustedSetup =
        runtimeOption("QGPU_PAIRED_RUNTIME_TRUSTED_SETUP") == "1"
        && (polarIndex != nil || runtimeOptionIsExplicit("QGPU_PAIRED_RUNTIME_TRUSTED_SETUP"))
      if trustedSetup {
        // Only valid after the load-time exact index build decoded and validated every stream.
        let eventRowsRequested = runtimeOption("QGPU_PAIRED_RUNTIME_EVENT_ROWS") == "1"
        let adjacentRequested = runtimeOption("QGPU_PAIRED_RUNTIME_ADJACENT_LANE_STREAMS") == "1"
        let flatEventsRequested = runtimeOption("QGPU_PAIRED_RUNTIME_FLAT_EVENTS") == "1"
        let compactPairsRequested = runtimeOption("QGPU_PAIRED_RUNTIME_COMPACT_PAIRS") == "1"
        // Four bytes per trip builds on two per trip; an explicit COMPACT_PAIRS=0 disables both.
        let compactQuadsRequested =
          compactPairsRequested
          && runtimeOption("QGPU_PAIRED_RUNTIME_COMPACT_QUADS") == "1"
        let trustedPipeline =
          diagLevel == 7
          ? detectorTrustedSetupHeaderDiagPipeline
          : eventRowsRequested
            ? detectorTrustedSetupEventRowsPipeline
            : adjacentRequested
              ? detectorTrustedSetupAdjacentPipeline
              : flatEventsRequested
                ? detectorTrustedSetupFlatEventsPipeline
                : compactQuadsRequested
                  ? detectorTrustedSetupCompactQuadsPipeline
                  : compactPairsRequested
                    ? detectorTrustedSetupCompactPairsPipeline : detectorTrustedSetupWindowPipeline
        guard let trustedSetupPipeline = trustedPipeline, polarIndex != nil,
          diagLevel == 0 || diagLevel == 7, !plainSums
        else {
          throw Self.invalid(
            "Trusted setup requires its prepared pipeline, the validated polar index, and no other diagnostics"
          )
        }
        encoder.setComputePipelineState(trustedSetupPipeline)
      } else if diagLevel != 0 {
        guard let diagnostic = detectorWindowDiagPipelines[diagLevel], !plainSums else {
          throw Self.invalid("Prepare the window diagnostic pipelines (levels 1, 2, 3, 4, 6) first")
        }
        encoder.setComputePipelineState(diagnostic)
      } else {
        encoder.setComputePipelineState(
          plainSums
            ? detectorTrustedWindowReaderPlainPipeline! : detectorTrustedWindowReaderPipeline!)
      }
    } else if trustedTable && plainSums {
      encoder.setComputePipelineState(detectorTrustedPlainSumsPipeline!)
    } else if trustedTable && registerSums {
      encoder.setComputePipelineState(detectorTrustedRegisterSumsPipeline!)
    } else if trustedTable && packetSplits == 4 {
      guard let detectorTrustedTableSplit4Pipeline else {
        throw Self.invalid(
          "Prepare the trusted-table split-4 pipeline before testing the composition")
      }
      encoder.setComputePipelineState(detectorTrustedTableSplit4Pipeline)
    } else if trustedTable && packetSplits == 8 {
      guard let detectorTrustedTableSplit8Pipeline else {
        throw Self.invalid(
          "Prepare the trusted-table split-8 pipeline before testing the composition")
      }
      encoder.setComputePipelineState(detectorTrustedTableSplit8Pipeline)
    } else if trustedTable {
      guard let detectorTrustedTablePipeline else {
        throw Self.invalid("Prepare the trusted-table pipeline before testing it")
      }
      encoder.setComputePipelineState(detectorTrustedTablePipeline)
    } else if plainSums {
      guard let detectorPlainSumsPipeline else {
        throw Self.invalid("Prepare the plain-sums pipeline before testing it")
      }
      encoder.setComputePipelineState(detectorPlainSumsPipeline)
    } else if registerSums {
      guard let detectorRegisterSumsPipeline else {
        throw Self.invalid("Prepare the register-sums pipeline before testing it")
      }
      encoder.setComputePipelineState(detectorRegisterSumsPipeline)
    } else if reuseWord {
      guard let detectorReuseWordPipeline else {
        throw Self.invalid("Prepare the word-reuse pipeline before testing it")
      }
      encoder.setComputePipelineState(detectorReuseWordPipeline)
    } else if lazyRefill {
      guard let detectorLazyRefillPipeline else {
        throw Self.invalid("Prepare the lazy-refill pipeline before testing it")
      }
      encoder.setComputePipelineState(detectorLazyRefillPipeline)
    } else if packetSplits > 1 {
      encoder.setComputePipelineState(detectorSplitPipelines[packetSplits]!)
    } else if reader32 {
      guard let detectorReader32Pipeline else {
        throw Self.invalid("Paired-runtime reader32 pipeline was not prepared")
      }
      encoder.setComputePipelineState(detectorReader32Pipeline)
    } else if cooperative {
      guard let detectorCooperativePipeline else {
        throw Self.invalid("Paired-runtime cooperative pipeline was not prepared")
      }
      encoder.setComputePipelineState(detectorCooperativePipeline)
    } else if macro {
      guard let detectorMacroPipeline, macroDecodingTable != nil else {
        throw Self.invalid("Paired-runtime macro pipeline was not prepared")
      }
      encoder.setComputePipelineState(detectorMacroPipeline)
    } else if denseCompaction {
      guard let detectorDenseCompactionPipeline else {
        throw Self.invalid("Paired-runtime dense-compaction pipeline was not prepared")
      }
      encoder.setComputePipelineState(detectorDenseCompactionPipeline)
    } else {
      encoder.setComputePipelineState(
        streamsPerLane == "4"
          ? detectorPacketOwner4Pipeline
          : streamsPerLane == "2" ? detectorPacketOwner2Pipeline : detectorPipeline)
    }
    let activeDecoding = macro ? macroDecodingTable! : decodingTable!
    for (index, buffer) in [
      payload, offsets, modes, activeDecoding, selectedBuffer, coefficientBuffer,
      outputProduct, failure,
    ].enumerated() {
      encoder.setBuffer(buffer, offset: 0, index: index)
    }
    encoder.setBytes(&parameters, length: parameters.count * 4, index: 8)
    encoder.dispatchThreadgroups(
      MTLSize(
        width: cooperative ? packets : ((logicalPackets + 3) / 4) * packetSplits, height: 1,
        depth: 1),
      threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    if sparseSplit {
      guard let detectorSparseScatterPipeline else {
        throw Self.invalid("Paired-runtime sparse-scatter pipeline was not prepared")
      }
      var sparseParameters: [UInt32] = [
        UInt32(pixels), UInt32(packets), UInt32(selected.count), UInt32(payload.length),
      ]
      encoder.setComputePipelineState(detectorSparseScatterPipeline)
      for (index, buffer) in [
        payload, offsets, modes, selectedBuffer, coefficientBuffer,
        outputProduct, failure,
      ].enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index)
      }
      encoder.setBytes(&sparseParameters, length: sparseParameters.count * 4, index: 7)
      let sparseWidth = detectorSparseScatterPipeline.threadExecutionWidth
      encoder.dispatchThreads(
        MTLSize(width: packets * selected.count, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(
          width: min(
            detectorSparseScatterPipeline.maxTotalThreadsPerThreadgroup, sparseWidth * 4),
          height: 1, depth: 1))
    }
    encoder.endEncoding()
    if profileEnabled { profile?.preparationEnd = ProcessInfo.processInfo.systemUptime }
    preparationSucceeded = true
    return PendingDetectorUpdate(
      command: command, mask: mask, previousMask: previousMask, started: started,
      changedCount: changedCount, operation: "detector",
      candidateProduct: historyActive ? outputProduct : nil, historyHit: false,
      historyBase: usingHistoryBase,
      previousHistoryValid: previousHistoryValid,
      previousPolarFieldCount: previousPolarFields,
      previousPolarResidualCount: previousPolarResiduals,
      profile: profile,
      stageSession: stageSession,
      keepAlive: [selectedBuffer, coefficientBuffer])
  }

  private func finishDetectorUpdate(
    _ pending: PendingDetectorUpdate, commit shouldCommit: Bool = true,
    commitHost suppliedCommitHost: Double? = nil
  ) throws -> DetectorUpdateResult {
    var profile = pending.profile
    guard let command = pending.command else {
      if pending.historyHit {
        let oldProduct = detectorProduct
        detectorProduct = historyProduct!
        historyProduct = oldProduct
        let oldMask = detectorMask
        detectorMask = historyMask!
        historyMask = oldMask
        let oldFields = polarFieldCount
        polarFieldCount = historyPolarFieldCount
        historyPolarFieldCount = oldFields
        let oldResiduals = polarResidualCount
        polarResidualCount = historyPolarResidualCount
        historyPolarResidualCount = oldResiduals
        refreshMetadataSnapshot()
        metadataLock.lock()
        cachedHistoryHit = true
        cachedHistoryBase = false
        metadataLock.unlock()
      }
      if profile != nil {
        profile?.readbackStart = ProcessInfo.processInfo.systemUptime
        let values = detectorValues()
        profile?.readbackEnd = ProcessInfo.processInfo.systemUptime
        publishUpdateProfile(profile)
        return (values, 0, 0, pending.changedCount)
      }
      return (detectorValues(), 0, 0, pending.changedCount)
    }
    do {
      if shouldCommit {
        let commitHost = profile == nil ? 0 : ProcessInfo.processInfo.systemUptime
        pending.stageSession?.willCommit()
        command.commit()
        if commitHost != 0 { profile?.commitHost = commitHost }
      } else if let suppliedCommitHost {
        profile?.commitHost = suppliedCommitHost
      }
      let waitCompletionHost: Double
      command.waitUntilCompleted()
      waitCompletionHost = profile == nil ? 0 : ProcessInfo.processInfo.systemUptime
      if waitCompletionHost != 0 { profile?.waitCompletionHost = waitCompletionHost }
      let code = failure.contents().load(as: UInt32.self)
      guard command.status == .completed, command.error == nil, code == 0 else {
        throw Self.invalid(
          "Paired-runtime \(pending.operation) failed with code \(code): "
            + (command.error?.localizedDescription ?? "invalid stream"))
      }
    } catch {
      if pending.candidateProduct != nil {
        memset(pending.candidateProduct!.contents(), 0, pending.candidateProduct!.length)
        polarFieldCount = pending.previousPolarFieldCount
        polarResidualCount = pending.previousPolarResidualCount
        historyValid = false
        refreshMetadataSnapshot()
      } else {
        resetDetectorStateAfterFailedUpdate()
      }
      throw error
    }
    if let stageSession = pending.stageSession {
      let resolveStart = ProcessInfo.processInfo.systemUptime
      profile?.stageResult = stageSession.record(command: command)
      profile?.profileResolveMilliseconds =
        (ProcessInfo.processInfo.systemUptime - resolveStart) * 1_000
    }
    if let candidate = pending.candidateProduct {
      let oldProduct = detectorProduct
      detectorProduct = candidate
      historyProduct = oldProduct
      historyMask = pending.previousMask
      historyPolarFieldCount = pending.previousPolarFieldCount
      historyPolarResidualCount = pending.previousPolarResidualCount
      historyValid = true
    }
    detectorMask = pending.mask
    metadataLock.lock()
    cachedHistoryBase = pending.historyBase
    metadataLock.unlock()
    let values: [UInt32]
    if profile != nil {
      profile?.readbackStart = ProcessInfo.processInfo.systemUptime
      values = detectorValues()
      profile?.readbackEnd = ProcessInfo.processInfo.systemUptime
      profile?.gpuStart = command.gpuStartTime
      profile?.gpuEnd = command.gpuEndTime
      publishUpdateProfile(profile)
    } else {
      values = detectorValues()
    }
    return (
      values, (CFAbsoluteTimeGetCurrent() - pending.started) * 1_000,
      max(0, command.gpuEndTime - command.gpuStartTime) * 1_000,
      pending.changedCount
    )
  }

  /// Restore host metadata when a batch fails before command submission.
  /// No detector buffer has been modified at this point.
  private func rollbackPreparedDetectorUpdate(_ pending: PendingDetectorUpdate) {
    historyValid = pending.previousHistoryValid
    polarFieldCount = pending.previousPolarFieldCount
    polarResidualCount = pending.previousPolarResidualCount
    refreshMetadataSnapshot()
  }

  private func publishUpdateProfile(_ profile: UpdateProfile?) {
    guard let profile else { return }
    metadataLock.lock()
    cachedUpdateProfile = profile.values
    metadataLock.unlock()
  }

  /// Count stored modes touched by a detector mask without copying resident metadata.
  @_spi(PairedRuntimeTANSPrototype)
  public func detectorStreamModeCounts(mask: [UInt8]) throws -> [UInt32] {
    stateLock.lock()
    defer { stateLock.unlock() }
    try requireLive()
    let pixels = shape[2] * shape[3]
    guard mask.count == pixels, mask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
      throw Self.invalid("The paired-runtime mode-profile mask must be binary")
    }
    let selected = mask.indices.compactMap { mask[$0] == 1 ? UInt32($0) : nil }
    guard !selected.isEmpty else { return [UInt32](repeating: 0, count: 256) }

    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: queue.device)
    guard let function = library.makeFunction(name: "paired_runtime_tans_detector_mode_histogram")
    else { throw Self.invalid("Paired-runtime mode-histogram kernel is missing") }
    let pipeline = try queue.device.makeComputePipelineState(function: function)
    let selectedBuffer = try Self.upload(
      streamAddresses(selected), device: queue.device, label: "paired-runtime mode-profile pixels")
    let histogram = try Self.buffer(
      device: queue.device, bytes: 256 * MemoryLayout<UInt32>.stride,
      options: .storageModeShared, label: "paired-runtime mode histogram")
    memset(histogram.contents(), 0, histogram.length)
    memset(failure.contents(), 0, MemoryLayout<UInt32>.stride)
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw Self.invalid("Metal could not encode paired-runtime mode profiling") }
    var parameters: [UInt32] = [
      UInt32(pixels),
      UInt32(shape[0] * shape[1] / PairedRuntimeTANSRecordABI.streamScans),
      UInt32(selected.count),
    ]
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(modes, offset: 0, index: 0)
    encoder.setBuffer(selectedBuffer, offset: 0, index: 1)
    encoder.setBuffer(histogram, offset: 0, index: 2)
    encoder.setBytes(&parameters, length: parameters.count * 4, index: 3)
    let jobs = Int(parameters[1]) * selected.count
    let width = pipeline.threadExecutionWidth
    encoder.dispatchThreads(
      MTLSize(width: jobs, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(
        width: min(pipeline.maxTotalThreadsPerThreadgroup, width * 4),
        height: 1, depth: 1))
    encoder.endEncoding()
    try finish(command, operation: "mode profiling")
    let counts = Array(
      UnsafeBufferPointer(
        start: histogram.contents().assumingMemoryBound(to: UInt32.self), count: 256))
    let expected = UInt64(selected.count) * UInt64(parameters[1])
    guard counts.reduce(UInt64(0), { $0 + UInt64($1) }) == expected else {
      throw Self.invalid("Paired-runtime mode histogram did not count every selected stream")
    }
    return counts
  }

  /// Count entropy-coded streams using a threadgroup reduction, not a hot global
  /// histogram. This benchmark diagnostic is sized for a full detector mask.
  @_spi(PairedRuntimeTANSPrototype)
  public func detectorEntropyModeCount(mask: [UInt8]) throws -> (
    count: UInt32, metalAllocatedBytes: Int
  ) {
    stateLock.lock()
    defer { stateLock.unlock() }
    try requireLive()
    let pixels = shape[2] * shape[3]
    guard mask.count == pixels, mask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
      throw Self.invalid("The paired-runtime entropy-mode mask must be binary")
    }
    let selected = mask.indices.compactMap { mask[$0] == 1 ? UInt32($0) : nil }
    guard !selected.isEmpty else { return (0, queue.device.currentAllocatedSize) }

    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: queue.device)
    guard
      let function = library.makeFunction(name: "paired_runtime_tans_detector_entropy_mode_count")
    else { throw Self.invalid("Paired-runtime entropy-mode-count kernel is missing") }
    let pipeline = try queue.device.makeComputePipelineState(function: function)
    let selectedBuffer = try Self.upload(
      streamAddresses(selected), device: queue.device, label: "paired-runtime entropy-mode pixels")
    let result = try Self.buffer(
      device: queue.device, bytes: MemoryLayout<UInt32>.stride,
      options: .storageModeShared, label: "paired-runtime entropy-mode count")
    memset(result.contents(), 0, result.length)
    let threadgroupWidth = 128
    guard pipeline.maxTotalThreadsPerThreadgroup >= threadgroupWidth else {
      throw Self.invalid("Entropy-mode census requires a 128-thread Metal threadgroup")
    }
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw Self.invalid("Metal could not encode entropy-mode census") }
    var parameters: [UInt32] = [
      UInt32(pixels),
      UInt32(shape[0] * shape[1] / PairedRuntimeTANSRecordABI.streamScans),
      UInt32(selected.count),
    ]
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(modes, offset: 0, index: 0)
    encoder.setBuffer(selectedBuffer, offset: 0, index: 1)
    encoder.setBuffer(result, offset: 0, index: 2)
    encoder.setBytes(&parameters, length: parameters.count * MemoryLayout<UInt32>.stride, index: 3)
    let diagnosticAllocationBytes = queue.device.currentAllocatedSize
    let jobs = Int(parameters[1]) * selected.count
    let groups = (jobs + threadgroupWidth - 1) / threadgroupWidth
    encoder.dispatchThreadgroups(
      MTLSize(width: groups, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: threadgroupWidth, height: 1, depth: 1))
    encoder.endEncoding()
    try finish(command, operation: "entropy-mode census")
    let entropyStreams = result.contents().assumingMemoryBound(to: UInt32.self).pointee
    guard UInt64(entropyStreams) <= UInt64(selected.count) * UInt64(parameters[1]) else {
      throw Self.invalid("Entropy-mode census returned more streams than were selected")
    }
    return (entropyStreams, diagnosticAllocationBytes)
  }

  /// Count entropy-only decoder chunks and SIMD halves for one exact mask delta.
  ///
  /// This benchmark diagnostic builds the same polar plan as
  /// `isolateDetectorStage`, reads the private tANS modes only on Metal, and
  /// returns small counters. Partial chunks are reported separately and are not
  /// part of their respective complete-chunk denominators.
  @_spi(PairedRuntimeTANSPrototype)
  public func detectorEntropyChunkCensus(previous: [UInt8], target: [UInt8]) throws -> (
    residualStreams: Int, packets: Int, fullChunksPerPacket: Int, tailStreamsPerPacket: Int,
    allEntropyFullChunks: UInt32, mixedFullChunks: UInt32,
    allEntropyTailChunks: UInt32, mixedTailChunks: UInt32,
    simdFullChunksPerPacket: Int, simdTailStreamsPerPacket: Int,
    allEntropySIMDFullChunks: UInt32, mixedSIMDFullChunks: UInt32,
    allEntropySIMDTailChunks: UInt32, mixedSIMDTailChunks: UInt32,
    residentBytesBefore: UInt64, residentBytesAfter: UInt64,
    metalAllocatedBytesDuringCensus: Int,
    sourceIdentitySHA256: String
  ) {
    stateLock.lock()
    defer { stateLock.unlock() }
    try requireLive()
    guard let polarIndex,
      previous.count == validPixels.count, target.count == validPixels.count,
      previous.allSatisfy({ $0 <= 1 }), target.allSatisfy({ $0 <= 1 })
    else {
      throw Self.invalid(
        "Entropy-chunk census requires a prepared polar index and binary detector masks")
    }

    let identity = sourceIdentitySHA256
    let residentBytesBefore = residentBytes
    let delta = target.indices.map {
      validPixels[$0] == 0 ? Int32(0) : Int32(target[$0]) - Int32(previous[$0])
    }
    let plan = PairedRuntimeTANSPolarPlan.make(
      delta: delta, validPixels: validPixels, detectorRows: shape[2],
      detectorColumns: shape[3], leafPixels: polarIndex.leafPixels,
      layoutKind: polarIndex.layoutKind)
    let chunkWidth = 64
    let fullChunks = plan.residualPixels.count / chunkWidth
    let tailStreams = plan.residualPixels.count % chunkWidth
    let packets = shape[0] * shape[1] / PairedRuntimeTANSRecordABI.streamScans
    guard packets > 0,
      shape[0] * shape[1] % PairedRuntimeTANSRecordABI.streamScans == 0,
      !plan.residualPixels.isEmpty
    else { throw Self.invalid("Entropy-chunk census has no complete packet/residual work") }

    let selected = try Self.upload(
      streamAddresses(plan.residualPixels), device: queue.device,
      label: "paired-runtime entropy-census residual pixels")
    let counts = try Self.buffer(
      device: queue.device, bytes: 9 * MemoryLayout<UInt32>.stride,
      options: .storageModeShared, label: "paired-runtime entropy-census counts")
    memset(counts.contents(), 0, counts.length)
    let library = try Metal4DSTEMKernels.makePairedRuntimeTANSLibrary(device: queue.device)
    guard
      let function = library.makeFunction(
        name: Metal4DSTEMKernels.pairedRuntimeTANSEntropyChunkCensusFunction)
    else { throw Self.invalid("Paired-runtime entropy-chunk census kernel is missing") }
    let pipeline = try queue.device.makeComputePipelineState(function: function)
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else { throw Self.invalid("Metal could not encode entropy-chunk census") }
    var parameters: [UInt32] = [
      UInt32(validPixels.count), UInt32(packets), UInt32(plan.residualPixels.count),
      UInt32(chunkWidth), UInt32(fullChunks), UInt32(tailStreams),
    ]
    encoder.setComputePipelineState(pipeline)
    encoder.setBuffer(modes, offset: 0, index: 0)
    encoder.setBuffer(selected, offset: 0, index: 1)
    encoder.setBuffer(counts, offset: 0, index: 2)
    encoder.setBytes(&parameters, length: parameters.count * MemoryLayout<UInt32>.stride, index: 3)
    let chunksPerPacket = fullChunks + (tailStreams == 0 ? 0 : 1)
    let jobs = packets * chunksPerPacket
    let width = min(pipeline.maxTotalThreadsPerThreadgroup, pipeline.threadExecutionWidth * 4)
    encoder.dispatchThreads(
      MTLSize(width: jobs, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: width, height: 1, depth: 1))
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed, command.error == nil else {
      throw Self.invalid(
        "Entropy-chunk census failed: \(command.error?.localizedDescription ?? "incomplete command")"
      )
    }
    let values = Array(
      UnsafeBufferPointer(start: counts.contents().assumingMemoryBound(to: UInt32.self), count: 9))
    let simdFullChunks = plan.residualPixels.count / 32
    let simdTailStreams = plan.residualPixels.count % 32
    guard UInt64(values[0]) + UInt64(values[1]) == UInt64(fullChunks * packets),
      UInt64(values[2]) + UInt64(values[3]) == UInt64((tailStreams == 0 ? 0 : packets)),
      UInt64(values[4]) + UInt64(values[5]) == UInt64(simdFullChunks * packets),
      UInt64(values[6]) + UInt64(values[7]) == UInt64((simdTailStreams == 0 ? 0 : packets)),
      values[8] == 0
    else { throw Self.invalid("Entropy-chunk census counters are inconsistent") }
    let residentBytesAfter = residentBytes
    guard sourceIdentitySHA256 == identity, residentBytesAfter == residentBytesBefore else {
      throw Self.invalid("Entropy-chunk census changed resident identity or resident bytes")
    }
    return (
      plan.residualPixels.count, packets, fullChunks, tailStreams,
      values[0], values[1], values[2], values[3], simdFullChunks, simdTailStreams,
      values[4], values[5], values[6], values[7],
      residentBytesBefore, residentBytesAfter, queue.device.currentAllocatedSize, identity
    )
  }

  public func releaseResidentStorage() {
    stateLock.lock()
    defer { stateLock.unlock() }
    // Invalidate any index build still running against these buffers.
    residentIndexGeneration &+= 1
    payload = nil
    offsets = nil
    modes = nil
    decodingTable = nil
    macroDecodingTable = nil
    polarIndex = nil
    diagnosticScratch = nil
    failure = nil
    diffraction = nil
    detectorProduct = nil
    historyProduct = nil
    historyMask = nil
    historyValid = false
    detectorPartials = nil
    detectorMask.removeAll(keepingCapacity: false)
    metadataLock.lock()
    cachedHistoryHit = false
    cachedHistoryBase = false
    cachedUpdateProfile = [:]
    metadataLock.unlock()
    released = true
    refreshMetadataSnapshot()
  }

  private func requireLive() throws {
    guard !released, payload != nil, offsets != nil, modes != nil,
      decodingTable != nil, failure != nil, diffraction != nil, detectorProduct != nil,
      !historyEnabled || historyProduct != nil
    else { throw Self.invalid("The paired-runtime ANS source was released; load it again") }
  }

  private func finish(_ command: MTLCommandBuffer, operation: String) throws {
    command.commit()
    command.waitUntilCompleted()
    let code = failure.contents().load(as: UInt32.self)
    guard command.status == .completed, command.error == nil, code == 0 else {
      throw Self.invalid(
        "Paired-runtime \(operation) failed with code \(code): "
          + (command.error?.localizedDescription ?? "invalid stream"))
    }
  }

  private func detectorValues() -> [UInt32] {
    if let destination = pendingCopyDestination {
      // One copy straight into the caller's display buffer; no intermediate array.
      memcpy(
        destination.contents(), detectorProduct.contents(),
        shape[0] * shape[1] * MemoryLayout<UInt32>.stride)
      return []
    }
    return Array(
      UnsafeBufferPointer(
        start: detectorProduct.contents().assumingMemoryBound(to: UInt32.self),
        count: shape[0] * shape[1]))
  }

  /// Restore the baseline without reimplementing detector science on the CPU.
  /// The caller holds `stateLock`, and the failed command has finished.
  private func resetDetectorStateAfterFailedUpdate() {
    memset(detectorProduct.contents(), 0, detectorProduct.length)
    detectorMask = [UInt8](repeating: 0, count: validPixels.count)
    polarFieldCount = 0
    polarResidualCount = 0
    refreshMetadataSnapshot()
  }

  /// Publish UI-facing metadata without extending the serialized operation lock.
  /// Callers mutate resident state only while holding `stateLock` or during initialization.
  private func refreshMetadataSnapshot() {
    let indexBytes = polarIndex?.residentBytes ?? 0
    let products =
      released
      ? 0
      : failure.length + diffraction.length + detectorProduct.length
        + (historyProduct?.length ?? 0)
    let scratch =
      released
      ? 0
      : (detectorPartials?.length ?? 0) + (macroDecodingTable?.length ?? 0)
        + (diagnosticScratch?.byteCount ?? 0)
    let totalBytes =
      released
      ? 0
      : loadMetrics.residentBytes + UInt64(products) + UInt64(scratch) + indexBytes
    metadataLock.lock()
    cachedReleased = released
    cachedResidentBytes = totalBytes
    cachedPolarFieldCount = polarFieldCount
    cachedPolarResidualCount = polarResidualCount
    cachedPolarIndexBytes = indexBytes
    cachedPolarIndexBuildMilliseconds = polarIndex?.buildMilliseconds ?? 0
    metadataLock.unlock()
  }

  private static func upload<T>(_ values: [T], device: MTLDevice, label: String) throws
    -> MTLBuffer
  {
    let buffer = values.withUnsafeBytes {
      device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
    }
    guard let buffer else { throw invalid("Metal could not allocate \(label)") }
    buffer.label = label
    return buffer
  }

  private static func consolidate(
    provider: PairedRuntimeTANSRecordProvider, library: MTLLibrary, queue: MTLCommandQueue,
    compactOffsetsEnabled: Bool
  ) throws -> (payload: MTLBuffer, offsets: MTLBuffer, modes: MTLBuffer, residentBytes: UInt64) {
    let streamCount = provider.descriptor.streamsPerRecord * provider.records.count
    let recordStreamCount = provider.descriptor.streamsPerRecord
    let payloadBytes = provider.receipt.recordExtents.reduce(0) {
      $0 + $1.terminalPayloadBytes
    }
    guard streamCount > 0, streamCount <= Int(UInt32.max), recordStreamCount > 0,
      payloadBytes <= Int(UInt32.max)
    else {
      throw invalid("One paired-runtime acquisition exceeds the UInt32 payload ABI")
    }
    let pipeline: MTLComputePipelineState
    let compactFailure: MTLBuffer?
    let offsetBytes: Int
    if compactOffsetsEnabled {
      guard streamCount & 31 == 0, recordStreamCount & 31 == 0,
        let rebase = library.makeFunction(
          name: Metal4DSTEMKernels.pairedRuntimeTANSRebaseBlock32OffsetsFunction)
      else {
        throw invalid("Block-32 offset consolidation requires aligned stream counts")
      }
      pipeline = try queue.device.makeComputePipelineState(function: rebase)
      let baseCount = streamCount / 32 + 1
      let (baseBytes, baseOverflow) = baseCount.multipliedReportingOverflow(by: 4)
      let (startsCount, startsCountOverflow) = streamCount.addingReportingOverflow(1)
      let (startsBytes, startsOverflow) = startsCount.multipliedReportingOverflow(by: 2)
      let (packedBytes, packedOverflow) = baseBytes.addingReportingOverflow(startsBytes)
      guard !baseOverflow, !startsCountOverflow, !startsOverflow, !packedOverflow,
        packedBytes <= queue.device.maxBufferLength
      else { throw invalid("Compact paired-runtime offsets exceed the Metal buffer limit") }
      offsetBytes = packedBytes
      compactFailure = try buffer(
        device: queue.device, bytes: 4, options: .storageModeShared,
        label: "paired-runtime compact offset validation")
      memset(compactFailure!.contents(), 0, 4)
    } else {
      guard
        let rebase = library.makeFunction(
          name: Metal4DSTEMKernels.pairedRuntimeTANSRebaseOffsetsFunction)
      else { throw invalid("Paired-runtime offset consolidation kernel is missing") }
      pipeline = try queue.device.makeComputePipelineState(function: rebase)
      offsetBytes = (streamCount + 1) * 4
      compactFailure = nil
    }
    let payload = try buffer(
      device: queue.device, bytes: max(8, ((payloadBytes + 3) & ~3) + 8),
      options: .storageModePrivate, label: "paired-runtime acquisition payload")
    let offsets = try buffer(
      device: queue.device, bytes: offsetBytes,
      options: .storageModePrivate,
      label: compactOffsetsEnabled
        ? "paired-runtime acquisition compact offsets" : "paired-runtime acquisition offsets")
    let modes = try buffer(
      device: queue.device, bytes: streamCount,
      options: .storageModePrivate, label: "paired-runtime acquisition modes")
    guard let command = queue.makeCommandBuffer() else {
      throw invalid("Metal could not consolidate paired-runtime records")
    }
    var payloadFirst = 0
    for (recordIndex, record) in provider.records.enumerated() {
      let extent = provider.receipt.recordExtents[recordIndex]
      guard let blit = command.makeBlitCommandEncoder() else {
        throw invalid("Metal could not copy paired-runtime records")
      }
      if extent.terminalPayloadBytes > 0 {
        blit.copy(
          from: record.payload, sourceOffset: 0, to: payload,
          destinationOffset: payloadFirst, size: extent.terminalPayloadBytes)
      }
      blit.copy(
        from: record.modes, sourceOffset: 0, to: modes,
        destinationOffset: recordIndex * provider.descriptor.streamsPerRecord,
        size: provider.descriptor.streamsPerRecord)
      blit.endEncoding()
      guard let encoder = command.makeComputeCommandEncoder() else {
        throw invalid("Metal could not rebase paired-runtime record offsets")
      }
      encoder.setComputePipelineState(pipeline)
      encoder.setBuffer(record.offsets, offset: 0, index: 0)
      encoder.setBuffer(offsets, offset: 0, index: 1)
      if compactOffsetsEnabled {
        guard let compactFailure,
          payloadFirst <= Int(UInt32.max),
          extent.terminalPayloadBytes <= Int(UInt32.max)
        else { throw invalid("Compact paired-runtime offset parameters exceed UInt32") }
        var parameters: [UInt32] = [
          UInt32(recordStreamCount), UInt32(recordIndex * recordStreamCount),
          UInt32(payloadFirst), UInt32(streamCount), UInt32(extent.terminalPayloadBytes),
        ]
        encoder.setBuffer(compactFailure, offset: 0, index: 2)
        encoder.setBytes(&parameters, length: parameters.count * 4, index: 3)
      } else {
        var parameters: [UInt32] = [
          UInt32(recordStreamCount), UInt32(recordIndex * recordStreamCount),
          UInt32(payloadFirst),
        ]
        encoder.setBytes(&parameters, length: parameters.count * 4, index: 2)
      }
      encoder.dispatchThreads(
        MTLSize(width: recordStreamCount + 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
      encoder.endEncoding()
      payloadFirst += extent.terminalPayloadBytes
    }
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed, command.error == nil else {
      throw invalid(
        "Paired-runtime record consolidation failed: "
          + (command.error?.localizedDescription ?? "unknown Metal failure"))
    }
    if let compactFailure {
      let failureCode = compactFailure.contents().load(as: UInt32.self)
      guard failureCode == 0 else {
        throw invalid("Compact paired-runtime offset validation failed (code \(failureCode))")
      }
    }
    return (
      payload, offsets, modes,
      UInt64(payload.length + offsets.length + modes.length)
    )
  }

  private static func buffer(
    device: MTLDevice, bytes: Int, options: MTLResourceOptions, label: String
  ) throws -> MTLBuffer {
    guard let buffer = device.makeBuffer(length: bytes, options: options) else {
      throw invalid("Metal could not allocate \(label)")
    }
    buffer.label = label
    return buffer
  }

  private static func invalid(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest(message)
  }

  private static let trustedTableExpected: Result<[UInt32], PairedRuntimeTANSTableError> = {
    do {
      let tables = try PairedRuntimeTANSTables.build()
      return .success(tables.packedDecoding)
    } catch let error as PairedRuntimeTANSTableError {
      return .failure(error)
    } catch {
      return .failure(.invalidExtent)
    }
  }()

  private static func validateTrustedTable(
    provider: PairedRuntimeTANSRecordProvider, queue: MTLCommandQueue, device: MTLDevice
  ) throws {
    let expected: [UInt32]
    switch trustedTableExpected {
    case .success(let values): expected = values
    case .failure(let error): throw error
    }
    guard provider.decodingTable.length == expected.count * MemoryLayout<UInt32>.stride,
      let staging = device.makeBuffer(
        length: expected.count * MemoryLayout<UInt32>.stride, options: .storageModeShared),
      let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
    else { throw invalid("Trusted paired-runtime table validation could not allocate readback") }
    blit.copy(
      from: provider.decodingTable, sourceOffset: 0, to: staging, destinationOffset: 0,
      size: provider.decodingTable.length)
    blit.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed, command.error == nil else {
      throw invalid("Trusted paired-runtime table readback failed")
    }
    let actual = Array(
      UnsafeBufferPointer(
        start: staging.contents().assumingMemoryBound(to: UInt32.self), count: expected.count))
    guard actual == expected else {
      let mismatch = actual.indices.first { actual[$0] != expected[$0] } ?? expected.count
      throw invalid("Trusted paired-runtime table differs from deterministic entry \(mismatch)")
    }
    for (entry, code) in actual.enumerated() {
      let bits = (code >> 12) & 15
      let base = code >> 16
      guard bits <= 10,
        base <= UInt32(PairedRuntimeTANSTables.stateCount) - (UInt32(1) << bits)
      else { throw invalid("Trusted paired-runtime table entry \(entry) exceeds state bounds") }
    }
  }
}

/// Bounded concurrent loader for an exact paired-runtime tilt series.
public enum MetalPairedRuntimeTANSSeriesLoader {
  // MTLDevice resource creation is thread safe. Older SDKs omit its Sendable
  // conformance; each task still owns its source and newly allocated buffers.
  private struct SharedDevice: @unchecked Sendable {
    let value: MTLDevice
  }

  public static func load(
    sources: [Native4DSTEMIndexedSource], device: MTLDevice,
    maximumConcurrentLoads: Int = 3,
    maximumAdditionalBytesPerLoad: UInt64? = nil
  ) async throws -> [MetalPairedRuntimeTANSResidentSource] {
    guard !sources.isEmpty else { return [] }
    guard maximumConcurrentLoads > 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Paired-runtime series loading requires at least one concurrent load slot")
    }
    var next = 0
    var completed = 0
    let sharedDevice = SharedDevice(value: device)
    var ordered = [MetalPairedRuntimeTANSResidentSource?](
      repeating: nil, count: sources.count)
    return try await withThrowingTaskGroup(
      of: (Int, MetalPairedRuntimeTANSResidentSource).self
    ) { group in
      func submit(_ index: Int) {
        let source = sources[index]
        group.addTask {
          let resident = try MetalPairedRuntimeTANSResidentSource.load(
            source: source, device: sharedDevice.value,
            maximumAdditionalBytes: maximumAdditionalBytesPerLoad)
          return (index, resident)
        }
      }
      while next < min(maximumConcurrentLoads, sources.count) {
        submit(next)
        next += 1
      }
      while let (index, resident) = try await group.next() {
        ordered[index] = resident
        completed += 1
        if next < sources.count {
          submit(next)
          next += 1
        }
      }
      guard completed == sources.count else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Paired-runtime series loading ended before every acquisition was resident")
      }
      return ordered.compactMap { $0 }
    }
  }
}
