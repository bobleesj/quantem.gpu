import CryptoKit
import Foundation
import Metal
@_spi(EntropySeriesPrototype) import Metal4DSTEMKernels

/// Unqualified archive-compatibility prototype, intentionally package-internal.
///
/// Selected acquisitions retain exact entropy streams, not a dense 4D tensor.
/// Authentication is sealed-record integrity, not independent scientific parity.
/// Do not expose through a consumer receipt until full validation gates pass.
final class MetalTANSResidentSeries {
  let acquisitionIndices: [Int]
  let shape: [Int]
  /// SHA-256 of the archive manifest; derived caches are keyed by it.
  let archiveCheckpointSHA256: String
  private(set) var readAndAuthenticationSeconds = 0.0
  private(set) var privateUploadSeconds = 0.0
  private(set) var loadSeconds = 0.0
  private(set) var readMetrics = TANSArchive.ReadMetrics()
  let transferConcurrency: Int
  let sourceReadPolicy: TANSArchive.SourceReadPolicy
  private(set) var stagingBytes = 0
  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let pipeline: MTLComputePipelineState
  private let word32Pipeline: MTLComputePipelineState
  var useWord32Query = false
  var queryThreadgroupWidth = 128
  private let auditPipeline: MTLComputePipelineState
  private let displayPipeline: MTLComputePipelineState
  private let detectorPipeline: MTLComputePipelineState
  private let detectorFinishPipeline: MTLComputePipelineState
  private let detectorBatchFunction: MTLFunction
  private let detectorBatchPipeline: MTLComputePipelineState
  private var interleavedDetectorPipelines: [Int: MTLComputePipelineState] = [:]
  private let tansLibrary: MTLLibrary
  private var sharedModelDetectorPipelines: [Int: MTLComputePipelineState] = [:]
  // Explicit experiment only. Zero retains the frozen ordinary kernel.
  var experimentalDetectorStreamsPerLane = 0
  // One-factor occupancy experiment; nil preserves the frozen launch topology.
  var experimentalSharedModelThreadgroupWidth: Int?
  // Isolated packet-ILP experiment; one preserves the frozen decoder.
  var experimentalSharedPacketsPerLane = 1
  // Isolated halfword-refill experiment; false preserves the frozen decoder.
  var experimentalSharedRefill16 = false
  var experimentalSignedPairReduction = false
  var experimentalPairLoopUnroll = 1
  var experimentalCompilerThreadgroupLimit = 0
  var experimentalStagedDetectorReduction = false
  var experimentalPrefetchDecoderEntry = false
  var experimentalDeferredReductionPairs = 1
  var experimentalDecoderBitExtract = false
  var experimentalPacketMajorGrid = false
  var experimentalPairLookupBits = 0
  private var detectorInitPipeline: MTLComputePipelineState?
  // Packet-owner exact kernel: one SIMD group per (record, 512-scan packet)
  // decodes every dense model group of the record, adds its sparse events and
  // writes each output scan once, with no device atomics. Same streams,
  // symbols and integer contributions as the shared-model kernel; only the
  // addition order changes (exact in uint32). Cross-checked bit for bit
  // against the shared-model kernel on all 66 acquisitions (lockstep test,
  // QUANTEM_TANS_LOCKSTEP_CROSSCHECK=1). One switch, QUANTEM_TANS_PACKET_OWNER:
  // unset, every query made through this property's default (index and atlas
  // builds, the fixture suites) uses the shared-model kernel and the app's
  // interactive queries (`configureInteractiveGrouping`) use this one; `1`
  // selects it for every query, e.g. to run the fixture exactness suites
  // through it; `0` keeps the shared-model kernel for the app's queries too.
  static let defaultPacketOwner =
    ProcessInfo.processInfo.environment["QUANTEM_TANS_PACKET_OWNER"] == "1"
  var experimentalPacketOwnerKernel = MetalTANSResidentSeries.defaultPacketOwner
  private var packetOwnerPipeline: MTLComputePipelineState?
  // Rank bytes of every resident record: one byte per 32-event flag word,
  // derived once from the immutable flags on the GPU (about 96 MB for 66
  // acquisitions), so a sparse event's count rank needs no popcount loop.
  private var ownerRankBytes: (buffer: MTLBuffer, offsets: [Int])?
  private func ownerRankByteTable() throws -> (buffer: MTLBuffer, offsets: [Int]) {
    if let ownerRankBytes, ownerRankBytes.offsets.count == records.count { return ownerRankBytes }
    let sparseOffsets = try chunks.map { chunk -> Int in
      guard let component = chunk.components.first(where: { $0.name == "sparse" }) else {
        throw TANSArchive.invalid("Missing sparse component")
      }
      return component.offset
    }
    guard let headerFunction = tansLibrary.makeFunction(name: "tans_owner_sparse_header"),
      let buildFunction = tansLibrary.makeFunction(name: "tans_owner_rank_bytes_build")
    else { throw TANSArchive.invalid("Missing owner rank-byte kernels") }
    let headerPipeline = try device.makeComputePipelineState(function: headerFunction)
    let buildPipeline = try device.makeComputePipelineState(function: buildFunction)
    guard
      let words = device.makeBuffer(length: max(4, records.count * 4), options: .storageModeShared),
      let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
    else { throw TANSArchive.invalid("Cannot encode owner rank-byte header pass") }
    encoder.setComputePipelineState(headerPipeline)
    for index in records.indices {
      encoder.setBuffer(
        records[index], offset: recordBaseOffsets[index] + sparseOffsets[index], index: 0)
      encoder.setBuffer(words, offset: index * 4, index: 1)
      encoder.dispatchThreads(
        MTLSize(width: 1, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 1, height: 1, depth: 1))
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("Owner rank-byte header pass failed")
    }
    let counts = words.contents().bindMemory(to: UInt32.self, capacity: records.count)
    var offsets: [Int] = []
    var total = 0
    for index in records.indices {
      // Flag words cannot exceed the authenticated sparse component.
      let component = chunks[index].components.first { $0.name == "sparse" }!
      let count = Int(counts[index])
      guard count <= component.nbytes / 4 else {
        throw TANSArchive.invalid("Sparse flag words exceed the sparse component")
      }
      offsets.append(total)
      total += (count + 15) / 16 * 16
    }
    guard let bytes = device.makeBuffer(length: max(16, total), options: .storageModePrivate),
      let build = queue.makeCommandBuffer(), let buildEncoder = build.makeComputeCommandEncoder()
    else { throw TANSArchive.invalid("Cannot allocate owner rank bytes") }
    bytes.label = "Owner sparse rank bytes (derived from immutable flags)"
    buildEncoder.setComputePipelineState(buildPipeline)
    for index in records.indices where counts[index] > 0 {
      buildEncoder.setBuffer(
        records[index], offset: recordBaseOffsets[index] + sparseOffsets[index], index: 0)
      buildEncoder.setBuffer(bytes, offset: offsets[index], index: 1)
      buildEncoder.dispatchThreads(
        MTLSize(width: Int(counts[index]), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
    }
    buildEncoder.endEncoding()
    build.commit()
    build.waitUntilCompleted()
    guard build.status == .completed else {
      throw TANSArchive.invalid("Owner rank-byte build failed")
    }
    ownerRankBytes = (bytes, offsets)
    return (bytes, offsets)
  }
  private var packedOwnerDecoding: MTLBuffer?
  /// Packed owner decode table, built once from the validated codebook:
  /// a[5:0] | bits[9:6] | escape[10] | b[21:16] | next base[31:22]. Escape
  /// entries keep their bits and base with a = b = 0. Metadata only.
  private func packedOwnerDecodingTable() throws -> MTLBuffer {
    if let packedOwnerDecoding { return packedOwnerDecoding }
    let source = globals[0]
    let count = source.length / 4
    let codes = source.contents().bindMemory(to: UInt32.self, capacity: count)
    var packed = [UInt32](repeating: 0, count: count)
    for index in 0..<count {
      let code = UInt32(littleEndian: codes[index])
      let pair = code & 4095
      let escape: UInt32 = pair == 4095 ? 1 : 0
      let a: UInt32 = escape == 1 ? 0 : (pair & 63)
      let b: UInt32 = escape == 1 ? 0 : (pair >> 6)
      let bits: UInt32 = (code >> 12) & 15
      let base: UInt32 = code >> 16
      guard bits <= 10, base < 1024 else {
        throw TANSArchive.invalid("tANS entry does not fit the packed owner decode layout")
      }
      packed[index] = a | (bits << 6) | (escape << 10) | (b << 16) | (base << 22)
    }
    guard
      let buffer = packed.withUnsafeBytes({
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    else { throw TANSArchive.invalid("Cannot allocate the packed owner decode table") }
    buffer.label = "Packed owner tANS decoding"
    packedOwnerDecoding = buffer
    return buffer
  }
  private var pairLookup: MTLBuffer?
  private var preparedPairLookupBits = 0
  var experimentalPairLookupBytes: Int { pairLookup?.length ?? 0 }
  // One-factor read-only lookup experiment; false keeps the shared table.
  var experimentalDirectDecodingTable = false
  // Exact deterministic zero transitions, using the existing shared table only.
  var experimentalZeroRunDecoding = false
  var experimentalPreparedZeroRuns = false
  private var prepareZeroRunsPipeline: MTLComputePipelineState?
  var experimentalZeroBitArithmetic = false
  // Keep full homogeneous groups; combine only their otherwise-padded tails.
  var experimentalMixedModelTails = false
  var experimentalMixedTailSavingsDivisor = 0
  var experimentalSeparateMixedDispatches = false
  var experimentalMixedOnlySpecialization = false
  var experimentalUseTileIndex = false
  var experimentalPlanAfterIndex = false
  /// Planner price of one exact tile add in dense-column units. The planner only
  /// ranks exact decompositions, so this never changes a count.
  var experimentalTileCost = 0.5
  private var exactTileIndex: TANSExactTileIndex?
  private let detectorColumnCost: [Double]?
  var experimentalTileIndexBytes: Int { exactTileIndex?.residentBytes ?? 0 }
  private(set) var lastDetectorTileFields = 0
  /// Exact annulus atlas experiment: packed complete images of stored masks,
  /// a third exact base beside the previous image and a recompute. A field is
  /// the exact image of its own mask bytes, so any request can start from it
  /// with a -1...1 residual; nothing depends on geometry.
  var experimentalUseAtlas = false
  private var experimentalAtlas: TANSExactTileIndex?
  private var experimentalAtlasMasks: [[UInt8]] = []
  private var experimentalAtlasCentroids: [(row: Double, col: Double)] = []
  private var experimentalAtlasBudget: UInt64 = 0
  /// When set, queries write these outputs (indexed by retained acquisition)
  /// and commit no seed or ring cursor, so an atlas build never touches an
  /// image the caller may be displaying or a seed its next frame continues from.
  private var detachedDetectorOutputs: [MTLBuffer]?
  var experimentalAtlasBytes: Int { experimentalAtlas?.residentBytes ?? 0 }
  var experimentalAtlasFieldCount: Int { experimentalAtlasMasks.count }
  private(set) var lastDetectorAtlasField: Int?
  /// Host seconds spent choosing the exact decomposition of the last query.
  private(set) var lastDetectorPlanSeconds = 0.0
  /// 1 where the detector pixel is a dense (tANS) column, 0 where it is a
  /// sparse-event column. Read-only planning metadata; consumers use it to
  /// predict decode cost (dense columns dominate), never to change counts.
  var detectorDenseMask: [UInt8] { cacheMapValues.map { $0 < 0 ? 1 : 0 } }
  private let detectorPartialBatchPipeline: MTLComputePipelineState
  private let detectorFinishBatchPipeline: MTLComputePipelineState
  private let detectorSparsePipeline: MTLComputePipelineState
  private let detectorSparsePrefixPipeline: MTLComputePipelineState
  var experimentalSparsePrefixCarry = false
  // Keep identical 64-record encoder grids, but amortize command submission.
  var experimentalSingleCommandEncoders = false
  var experimentalSingleComputePass = false
  var experimentalMetal4Submission = false
  private var metal4QueryContext: AnyObject?

  func prepareExperimentalMetal4Submission() throws {
    guard #available(macOS 26.0, iOS 26.0, *) else {
      throw TANSArchive.invalid("Explicit entropy submission requires macOS/iOS26")
    }
    if metal4QueryContext == nil {
      metal4QueryContext = try TANSMetal4Query(
        device: device, library: tansLibrary, sources: records + globals)
    }
  }
  private let cacheMapValues: [Int32]
  let validDetectorMask: [UInt8]
  var useBatchedDetector = true
  // Benchmark-only alternative. Exact, but measured slower for live deltas
  // because its partial write/read traffic outweighs avoided atomics.
  var usePartialDetector = false
  private(set) var lastDetectorGPUSeconds = 0.0
  private(set) var lastDetectorCommandTiming: [String: Double] = [:]
  private var submissionProbeMode: TANSSubmissionProbeMode?
  private var submissionProbePipeline: MTLComputePipelineState?
  private var submissionProbeWords: [UInt32] = []
  // Opt-in diagnostic only. File IO/counter instrumentation invalidates speed claims.
  var experimentalMemoryAuditURL: URL?
  // Diagnostic only: immutable source records are already covered by the set.
  var experimentalUseResidentSetForSourceReads = false
  // Submission-overlap experiment; zero retains one complete-series command.
  var experimentalDetectorRecordsPerCommand = 0
  // Disjoint acquisition shards may use separate submission queues. Default off.
  var experimentalDetectorQueueCount = 1
  private var experimentalExtraDetectorQueues: [MTLCommandQueue] = []
  // Independent exact atomic contributions may overlap within one encoder.
  // Initialization/index additions stay in earlier ordered encoders.
  var experimentalConcurrentDetectorDispatches = false
  // Track only the complete 2D outputs actually written by a command shard.
  var experimentalNarrowOutputDeclarations = false
  // Keep the first retained command plus all invocation resources alive
  // through completion; later submissions may omit redundant ARC ownership.
  var experimentalInvocationOwnedSubmissions = false
  // Test-only failure injection; nil never interrupts an invocation.
  var experimentalFailAfterSubmittedCommands: Int?
  private(set) var lastDetectorScratchBytes = 0
  private(set) var lastDetectorModelGroups = 0
  private(set) var lastDetectorMixedModelGroups = 0
  private(set) var lastDetectorPaddedModelLanes = 0
  private(set) var lastDetectorDecodedColumns = 0
  private(set) var lastDetectorUsedPrevious = false
  /// One exact delta seed per retained acquisition: the binary mask of its
  /// last completed image and that image. Seeds are independent, so a
  /// single-acquisition interactive query and a later catch-up over the other
  /// acquisitions never invalidate each other. Dropping a seed is always
  /// exact; it only removes an accelerator, never a count.
  struct DetectorSeed {
    let mask: [UInt8]
    let maskID: UInt64
    let image: MTLBuffer
  }
  private var detectorSeeds: [Int: DetectorSeed] = [:]
  /// Three complete 512x512 UInt32 images per retained acquisition. A query
  /// writes the slot after the current seed, so the buffers returned by the
  /// previous two queries for that acquisition are never written by this one.
  /// Seeds and ring cursors advance when a query is submitted; a query that
  /// fails restores them (and those of every later query, which read its
  /// output as their seed) before its failure is reported.
  static let detectorImageRingSlots = 3
  /// Pipelined queries in flight at once. Two keep a failed query's prior
  /// seed slot unwritten until the failure is seen, so rollback is exact: the
  /// next two slots of an acquisition belong to the two in-flight queries.
  static let maximumDetectorQueriesInFlight = detectorImageRingSlots - 1
  private var detectorImageRings: [Int: [MTLBuffer]] = [:]
  private var detectorRingCursors: [Int: Int] = [:]
  private var nextDetectorMaskID: UInt64 = 1
  /// Submitted queries not yet finished by their caller, oldest first. All use
  /// the one detector queue, so each reads its seed after the earlier query
  /// that wrote it (default tracked hazards on every output image).
  private var inFlightDetectorQueries: [PendingDetectorQuery] = []
  private var nextDetectorQuerySequence: UInt64 = 1
  var detectorQueriesInFlight: Int { inFlightDetectorQueries.count }
  /// Test-only failure injection: a query whose sequence is listed is treated
  /// as failed on the GPU when it is verified, after its commands completed.
  var experimentalFailedDetectorQuerySequences: Set<UInt64> = []

  /// Test-only: the exact mask an acquisition's next query starts from.
  func experimentalDetectorSeedMask(acquisition: Int) -> [UInt8]? {
    acquisitionIndices.firstIndex(of: acquisition).flatMap { detectorSeeds[$0]?.mask }
  }

  /// Exact encoding of one seed group, recorded at submit. Plan diagnostics
  /// are captured here because a later submission overwrites the `last*`
  /// properties before this one is finished.
  struct SubmittedDetectorGroup {
    let acquisitions: [Int]
    let images: [MTLBuffer]
    let columns: Int
    let tiles: Int
    let usedPrevious: Bool
    let planSeconds: Double
    let atlasField: Int?
    let scratchBytes: Int
    let modelGroups: Int
    let mixedModelGroups: Int
    let paddedModelLanes: Int
    let batch: DetectorBatchSubmission
  }

  /// Committed commands of one encoded group and the step that waits for
  /// them, checks their status and returns GPU seconds with command timing.
  struct DetectorBatchSubmission {
    let commands: [MTLCommandBuffer]
    let finish: () throws -> (seconds: Double, timing: [String: Double])
  }

  /// A submitted exact detector query. Its images are complete only once
  /// `finishDetectorImages` returns them.
  final class PendingDetectorQuery {
    let sequence: UInt64
    let outputAcquisitionIndices: [Int]
    let queriesInFlightAtSubmit: Int
    let completion = TANSDetectorCompletion()
    var groups: [SubmittedDetectorGroup] = []
    /// Seed and ring cursor of every acquisition this query advanced, as
    /// they were before it, in commit order.
    var priors: [(retained: Int, seed: DetectorSeed?, cursor: Int?)] = []
    var verified = false
    var failure: Error?

    init(sequence: UInt64, outputAcquisitionIndices: [Int], queriesInFlightAtSubmit: Int) {
      self.sequence = sequence
      self.outputAcquisitionIndices = outputAcquisitionIndices
      self.queriesInFlightAtSubmit = queriesInFlightAtSubmit
    }

    var commands: [MTLCommandBuffer] { groups.flatMap(\.batch.commands) }
  }

  /// Result of a finished query: complete images in requested order and the
  /// diagnostics of that query (not of a later submission).
  struct DetectorQueryResult {
    let images: [MTLBuffer]
    let gpuSeconds: Double
    let decodedColumns: Int
    let usedPrevious: Bool
    let atlasField: Int?
    let timing: [String: Double]
  }
  private let chunks: [TANSArchive.Chunk]
  /// Disjoint-index writes from `DispatchQueue.concurrentPerform`: each
  /// iteration owns exactly one element, so no two iterations alias.
  private struct UncheckedSendableBuffer<Element>: @unchecked Sendable {
    let buffer: UnsafeMutableBufferPointer<Element>
    init(_ buffer: UnsafeMutableBufferPointer<Element>) { self.buffer = buffer }
  }
  /// One exact dispatch table per (record set, output image set). Its bindings
  /// are immutable metadata: the four entropy component slices of every chunk
  /// and the output slice each chunk writes. Only the output image identities
  /// change between interactive queries (the three-slot ring rotates), so a
  /// small cache turns a per-query rebuild of 1,056 argument-buffer entries
  /// into a lookup. Nothing about counts, geometry or exactness depends on it.
  private struct DetectorTableKey: Hashable {
    let recordIndices: [Int]
    let images: [ObjectIdentifier]
  }
  private var detectorTables: [DetectorTableKey: (table: MTLBuffer, offsets: MTLBuffer)] = [:]
  private var detectorTableOrder: [DetectorTableKey] = []
  private static let detectorTableCacheLimit = 8
  /// Component byte offsets per record, resolved once instead of by name search
  /// on every query. Order: dense, dense_offsets, sparse, sparse_offsets.
  private var detectorComponentOffsets: [[Int]] = []
  private var records: [MTLBuffer] = []
  private var recordBaseOffsets: [Int] = []
  private var acquisitionStorage: [MTLBuffer] = []
  var experimentalEncodedAllocationCount: Int {
    acquisitionStorage.isEmpty ? records.count : acquisitionStorage.count
  }
  var experimentalUntrackedRecordCount: Int {
    records.filter { $0.hazardTrackingMode == .untracked }.count
  }
  private var recordHeaps: [MTLHeap] = []
  private var recordHeapIndices: [Int] = []
  var experimentalDeclareEncodedHeaps = false
  private(set) var experimentalRecordHeapBytes = 0
  var experimentalRecordHeapCount: Int { recordHeaps.count }
  private var globals: [MTLBuffer] = []
  // Availability-erased so the unchanged baseline still supports macOS14.
  private var experimentalResidencyHold: AnyObject?
  private(set) var experimentalResidencySetBytes: UInt64 = 0
  private(set) var experimentalResidencyAllocationCount = 0
  private(set) var experimentalResidencyAttachedToQueue = false
  private var output: MTLBuffer?
  private let retainedColumns: UInt32
  private let sparseColumns: UInt32
  private(set) var isReleased = false
  private(set) var lastQueryGPUSeconds = 0.0

  var residentBytes: Int {
    (acquisitionStorage.isEmpty ? records : acquisitionStorage).reduce(0) { $0 + $1.length }
      + globals.reduce(0) { $0 + $1.length }
      + (output?.length ?? 0)
      + detectorImageRings.values.reduce(0) { $0 + $1.reduce(0) { $0 + $1.length } }
      + experimentalTileIndexBytes + experimentalPairLookupBytes + experimentalAtlasBytes
      + (ownerRankBytes?.buffer.length ?? 0) + (packedOwnerDecoding?.length ?? 0)
  }

  init(
    directory: URL, acquisitions: [Int], device: MTLDevice, maximumAdditionalBytes: UInt64,
    transferConcurrency: Int = 4,
    sourceReadPolicy: TANSArchive.SourceReadPolicy = .avoidCaching,
    experimentalRecordsPerHeap: Int = 0,
    experimentalUntrackedEncodedSource: Bool = false,
    experimentalCoalescedEncodedSource: Bool = false,
    experimentalSharedEncodedSource: Bool = false
  )
    throws
  {
    guard (1...4).contains(transferConcurrency), [0, 16, 64].contains(experimentalRecordsPerHeap),
      !experimentalUntrackedEncodedSource || experimentalRecordsPerHeap == 0,
      !experimentalCoalescedEncodedSource
        || (!experimentalUntrackedEncodedSource && experimentalRecordsPerHeap == 0),
      !experimentalSharedEncodedSource
        || (experimentalCoalescedEncodedSource && device.hasUnifiedMemory)
    else {
      throw TANSArchive.invalid("Use one through four bounded encoded-record transfer workers")
    }
    self.transferConcurrency = transferConcurrency
    self.sourceReadPolicy = sourceReadPolicy
    let start = ProcessInfo.processInfo.systemUptime
    let archive = try TANSArchive(directory: directory, acquisitions: acquisitions)
    self.device = device
    acquisitionIndices = acquisitions
    shape = [acquisitions.count, 512, 512, 192, 192]
    archiveCheckpointSHA256 = archive.checkpointSHA256
    chunks = archive.chunks
    guard let queue = device.makeCommandQueue() else {
      throw TANSArchive.invalid("Cannot create a Metal command queue")
    }
    self.queue = queue
    let library = try Metal4DSTEMKernels.makeTANSLibrary(device: device)
    tansLibrary = library
    guard let function = library.makeFunction(name: "tans_diffraction") else {
      throw TANSArchive.invalid("Missing tANS kernel")
    }
    pipeline = try device.makeComputePipelineState(function: function)
    guard let word32 = library.makeFunction(name: "tans_diffraction_word32") else {
      throw TANSArchive.invalid("Missing exact word32 tANS query")
    }
    word32Pipeline = try device.makeComputePipelineState(function: word32)
    guard let auditFunction = library.makeFunction(name: "tans_audit_packet") else {
      throw TANSArchive.invalid("Missing bounded tANS audit kernel")
    }
    auditPipeline = try device.makeComputePipelineState(function: auditFunction)
    guard let displayFunction = library.makeFunction(name: "tans_display_counts") else {
      throw TANSArchive.invalid("Missing exact diffraction display kernel")
    }
    displayPipeline = try device.makeComputePipelineState(function: displayFunction)
    guard let detectorFunction = library.makeFunction(name: "tans_detector_partials"),
      let finishFunction = library.makeFunction(name: "tans_detector_finish")
    else {
      throw TANSArchive.invalid("Missing exact tANS detector kernels")
    }
    detectorPipeline = try device.makeComputePipelineState(function: detectorFunction)
    detectorFinishPipeline = try device.makeComputePipelineState(function: finishFunction)
    guard let batchFunction = library.makeFunction(name: "tans_detector_batch") else {
      throw TANSArchive.invalid("Missing batched exact detector kernel")
    }
    detectorBatchFunction = batchFunction
    detectorBatchPipeline = try device.makeComputePipelineState(function: batchFunction)
    guard let partialBatchFunction = library.makeFunction(name: "tans_detector_partial_batch"),
      let finishBatchFunction = library.makeFunction(name: "tans_detector_finish_batch")
    else {
      throw TANSArchive.invalid("Missing non-atomic batched detector kernels")
    }
    detectorPartialBatchPipeline = try device.makeComputePipelineState(
      function: partialBatchFunction)
    detectorFinishBatchPipeline = try device.makeComputePipelineState(function: finishBatchFunction)
    let baselineSparseConstants = MTLFunctionConstantValues()
    var baselinePrefixCarry = false
    baselineSparseConstants.setConstantValue(&baselinePrefixCarry, type: .bool, index: 22)
    let sparseFunction = try library.makeFunction(
      name: "tans_detector_sparse_batch", constantValues: baselineSparseConstants)
    detectorSparsePipeline = try device.makeComputePipelineState(function: sparseFunction)
    let sparseConstants = MTLFunctionConstantValues()
    var prefixCarry = true
    sparseConstants.setConstantValue(&prefixCarry, type: .bool, index: 22)
    detectorSparsePrefixPipeline = try device.makeComputePipelineState(
      function: library.makeFunction(
        name: "tans_detector_sparse_batch", constantValues: sparseConstants))
    let mapData = archive.arrays["planner__cache_map"]!
    validDetectorMask = Array(archive.arrays["planner__valid"]!)
    detectorColumnCost = archive.arrays["planner__column_cost"]?.withUnsafeBytes { raw in
      (0..<36864).map { raw.loadUnaligned(fromByteOffset: $0 * 8, as: Double.self) }
    }
    let cacheMap: [Int32] = mapData.withUnsafeBytes { bytes in
      (0..<36864).map {
        Int32(littleEndian: bytes.loadUnaligned(fromByteOffset: $0 * 4, as: Int32.self))
      }
    }
    let selectedSparse = cacheMap.filter { $0 >= 0 }.sorted()
    cacheMapValues = cacheMap
    guard cacheMap.allSatisfy({ $0 >= -1 }),
      selectedSparse == (0..<selectedSparse.count).map(Int32.init)
    else {
      throw TANSArchive.invalid("Sparse detector mapping is not a permutation")
    }
    sparseColumns = UInt32(selectedSparse.count)
    retainedColumns = UInt32(36864 - selectedSparse.count)
    var rank = UInt32(0)
    let ranks: [UInt32] = cacheMap.map { value in
      if value >= 0 { return UInt32.max }
      defer { rank += 1 }
      return rank
    }
    let modelCount = archive.arrays["codec__decoding"]!.count / 4096
    try archive.arrays["codec__decoding"]!.withUnsafeBytes { bytes in
      for offset in stride(from: 0, to: bytes.count, by: 4) {
        let code = UInt32(
          littleEndian: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
        let bits = (code >> 12) & 15
        guard bits <= 10, (code >> 16) + ((1 << bits) - 1) < 1024 else {
          throw TANSArchive.invalid("tANS transition leaves its1024-state model")
        }
      }
    }
    guard archive.arrays["model_ids"]!.allSatisfy({ $0 == 255 || Int($0) < modelCount }) else {
      throw TANSArchive.invalid("Detector references an absent tANS model")
    }
    let lengths = chunks.map(\.recordBytes)
    let coalescedPlan =
      try experimentalCoalescedEncodedSource
      ? TANSCoalescedRecordPlan(
        lengths: lengths, maxBufferLength: device.maxBufferLength,
        maximumBytes: maximumAdditionalBytes) : nil
    recordBaseOffsets = coalescedPlan?.offsets ?? Array(repeating: 0, count: chunks.count)
    let heapPlan =
      try experimentalRecordsPerHeap == 0
      ? nil
      : TANSRecordHeapPlan(
        lengths: lengths, recordsPerHeap: experimentalRecordsPerHeap, device: device)
    recordHeapIndices = heapPlan?.heapIndices ?? []
    let globalBytes = archive.arrays.values.reduce(0) { $0 + $1.count } + ranks.count * 4
    let outputBytes = acquisitions.count * 36864 * 2
    let stageBytes = lengths.max() ?? 0
    stagingBytes = transferConcurrency * stageBytes
    // One reusable encoded-record stage per worker; read/authenticate in place.
    // No intermediate Data copy or decoded 4D storage is allocated.
    var required = UInt64(
      globalBytes + outputBytes + transferConcurrency * stageBytes)
    for length in lengths {
      let next = required.addingReportingOverflow(UInt64(length))
      guard !next.overflow, length <= device.maxBufferLength else {
        throw TANSArchive.invalid("Encoded record exceeds Metal limits")
      }
      required = next.partialValue
    }
    if let heapPlan {
      let logicalBytes = lengths.reduce(0, +)
      let overhead = heapPlan.allocationBytes - logicalBytes
      guard overhead >= 0 else { throw TANSArchive.invalid("Encoded heap admission underflow") }
      let padded = required.addingReportingOverflow(UInt64(overhead))
      guard !padded.overflow else { throw TANSArchive.invalid("Encoded heap admission overflow") }
      required = padded.partialValue
    }
    guard required <= maximumAdditionalBytes else {
      throw TANSArchive.invalid(
        "Encoded series needs \(required) additional bytes, budget is \(maximumAdditionalBytes); no fallback or binning"
      )
    }
    func shared(_ data: Data, label: String) throws -> MTLBuffer {
      let buffer = data.withUnsafeBytes { bytes in
        device.makeBuffer(
          bytes: bytes.baseAddress!, length: bytes.count, options: .storageModeShared)
      }
      guard let buffer else { throw TANSArchive.invalid("Cannot allocate \(label)") }
      buffer.label = label
      return buffer
    }
    globals = try ["codec__decoding", "model_ids", "planner__cache_map"].map {
      try shared(archive.arrays[$0]!, label: $0)
    }
    globals.append(
      try ranks.withUnsafeBytes { try shared(Data($0), label: "retained detector ranks") })
    guard let output = device.makeBuffer(length: outputBytes, options: .storageModeShared)
    else {
      throw TANSArchive.invalid("Cannot allocate bounded transfer and diffraction buffers")
    }
    self.output = output
    output.label = "Exact tANS selected diffraction series"
    let window = try TANSUploadWindow(
      archive: archive, device: device, queue: queue, stageBytes: stageBytes,
      concurrency: transferConcurrency, readPolicy: sourceReadPolicy,
      untrackedEncodedSource: experimentalUntrackedEncodedSource,
      destinationStorageMode: experimentalSharedEncodedSource ? .shared : .private)
    var readSeconds = 0.0
    var uploadSeconds = 0.0
    for first in stride(from: 0, to: chunks.count, by: transferConcurrency) {
      let end = min(chunks.count, first + transferConcurrency)
      var destinations: [MTLBuffer]?
      var destinationOffsets: [Int]?
      if let coalescedPlan {
        destinations = try (first..<end).map { index in
          let allocation = coalescedPlan.allocationIndices[index]
          if allocation == acquisitionStorage.count {
            guard
              let buffer = device.makeBuffer(
                length: coalescedPlan.allocationSizes[allocation],
                options: experimentalSharedEncodedSource
                  ? [.storageModeShared, .hazardTrackingModeTracked]
                  : [.storageModePrivate, .hazardTrackingModeTracked])
            else {
              throw TANSArchive.invalid("Cannot allocate exact coalesced source; no fallback")
            }
            buffer.label = "Immutable encoded acquisition \(allocation), sixteen source records"
            acquisitionStorage.append(buffer)
          }
          return acquisitionStorage[allocation]
        }
        destinationOffsets = Array(coalescedPlan.offsets[first..<end])
      } else if let heapPlan {
        destinations = try (first..<end).map { index in
          let heapIndex = heapPlan.heapIndices[index]
          if heapIndex == recordHeaps.count {
            let descriptor = MTLHeapDescriptor()
            descriptor.type = .placement
            descriptor.storageMode = .private
            descriptor.hazardTrackingMode = .tracked
            descriptor.size = heapPlan.heapSizes[heapIndex]
            guard let heap = device.makeHeap(descriptor: descriptor) else {
              throw TANSArchive.invalid("Cannot allocate complete encoded heap; no fallback")
            }
            heap.label = "Exact encoded records heap \(heapIndex)"
            recordHeaps.append(heap)
            experimentalRecordHeapBytes += heap.size
          }
          guard
            let buffer = recordHeaps[heapIndex].makeBuffer(
              length: chunks[index].recordBytes,
              options: [.storageModePrivate, .hazardTrackingModeTracked],
              offset: heapPlan.offsets[index])
          else { throw TANSArchive.invalid("Cannot allocate disjoint encoded record in heap") }
          return buffer
        }
      }
      // Join every worker/GPU command before failure; retain source record order.
      for loaded in try window.load(
        Array(chunks[first..<end]), destinations: destinations,
        destinationOffsets: destinationOffsets)
      {
        readSeconds += loaded.readSeconds
        uploadSeconds += loaded.uploadSeconds
        readMetrics.ioSeconds += loaded.metrics.ioSeconds
        readMetrics.hashSeconds += loaded.metrics.hashSeconds
        readMetrics.validationSeconds += loaded.metrics.validationSeconds
        readMetrics.sourceBytesRead += loaded.metrics.sourceBytesRead
        readMetrics.authenticatedRecords += loaded.metrics.authenticatedRecords
        records.append(loaded.resident)
      }
    }
    readAndAuthenticationSeconds = readSeconds
    privateUploadSeconds = uploadSeconds
    loadSeconds = ProcessInfo.processInfo.systemUptime - start
  }

  /// Full uint16 detector counts at one native scan location, in selected acquisition order.
  /// The returned array is an independent small 2D-per-acquisition snapshot.
  /// Retained positions of a requested acquisition subset, in requested order.
  private func retainedIndices(for selectedAcquisitions: [Int]?) throws -> [Int] {
    let requested = selectedAcquisitions ?? acquisitionIndices
    guard !requested.isEmpty, Set(requested).count == requested.count else {
      throw TANSArchive.invalid("Select unique retained acquisition indices")
    }
    return try requested.map { acquisition in
      guard let retained = acquisitionIndices.firstIndex(of: acquisition) else {
        throw TANSArchive.invalid("Selected acquisition is not retained")
      }
      return retained
    }
  }

  private func diffractionBuffer(
    scanRow: Int, scanColumn: Int, selectedAcquisitions: [Int]? = nil
  ) throws -> MTLBuffer {
    guard !isReleased, let output else {
      throw TANSArchive.invalid("tANS series has been released")
    }
    guard (0..<512).contains(scanRow), (0..<512).contains(scanColumn) else {
      throw TANSArchive.invalid("Scan row and column must be in 0..<512")
    }
    let position = scanRow * 512 + scanColumn
    let retained = try retainedIndices(for: selectedAcquisitions)
    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
    else {
      throw TANSArchive.invalid("Cannot encode exact tANS diffraction")
    }
    encoder.setComputePipelineState(useWord32Query ? word32Pipeline : pipeline)
    for index in retained {
      let acquisition = acquisitionIndices[index]
      let localChunk = position / 16384
      let recordIndex = index * 16 + localChunk
      let chunk = chunks[recordIndex]
      for (binding, name) in ["dense", "dense_offsets", "sparse", "sparse_offsets"].enumerated() {
        guard let component = chunk.components.first(where: { $0.name == name }),
          component.dtype == "<u4"
        else {
          encoder.endEncoding()
          throw TANSArchive.invalid("Missing authenticated entropy component \(name)")
        }
        encoder.setBuffer(
          records[recordIndex], offset: recordBaseOffsets[recordIndex] + component.offset,
          index: binding)
      }
      for (index, buffer) in globals.enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: index + 4)
      }
      encoder.setBuffer(output, offset: index * 36864 * 2, index: 8)
      var parameters: [UInt32] = [
        UInt32(position % 16384), retainedColumns, sparseColumns,
        UInt32((acquisition * 4 + localChunk / 4) * 36864),
      ]
      encoder.setBytes(&parameters, length: 16, index: 9)
      encoder.dispatchThreads(
        MTLSize(width: 36864, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: queryThreadgroupWidth, height: 1, depth: 1))
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("Exact tANS query failed: \(String(describing: command.error))")
    }
    lastQueryGPUSeconds = command.gpuEndTime - command.gpuStartTime
    return output
  }

  func extractDiffraction(scanRow: Int, scanColumn: Int) throws -> [UInt16] {
    let output = try diffractionBuffer(scanRow: scanRow, scanColumn: scanColumn)
    return Array(
      UnsafeBufferPointer(
        start: output.contents().bindMemory(
          to: UInt16.self, capacity: acquisitionIndices.count * 36864),
        count: acquisitionIndices.count * 36864))
  }

  /// Exact widening for existing UInt32 Metal image renderers. No image download.
  /// A subset decodes only the selected acquisitions, in requested order; the
  /// shared UInt16 scratch keeps each acquisition at its retained offset.
  func diffractionImages(
    scanRow: Int, scanColumn: Int, selectedAcquisitions: [Int]? = nil
  ) throws -> [MTLBuffer] {
    let source = try diffractionBuffer(
      scanRow: scanRow, scanColumn: scanColumn, selectedAcquisitions: selectedAcquisitions)
    let retained = try retainedIndices(for: selectedAcquisitions)
    guard let command = queue.makeCommandBuffer(), let encoder = command.makeComputeCommandEncoder()
    else { throw TANSArchive.invalid("Cannot prepare exact diffraction display") }
    encoder.setComputePipelineState(displayPipeline)
    var images: [MTLBuffer] = []
    for index in retained {
      guard let image = device.makeBuffer(length: 36864 * 4, options: .storageModeShared) else {
        encoder.endEncoding()
        throw TANSArchive.invalid("Cannot allocate a diffraction image")
      }
      images.append(image)
      encoder.setBuffer(source, offset: index * 36864 * 2, index: 0)
      encoder.setBuffer(image, offset: 0, index: 1)
      encoder.dispatchThreads(
        MTLSize(width: 36864, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
    encoder.endEncoding()
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("Diffraction display failed")
    }
    return images
  }

  /// Exact binary-mask sum for every scan in every retained acquisition.
  /// The caller supplies any validity mask explicitly. Raw source is unchanged.
  /// Scratch contains only 32-column reduction partials, not raw detector data.
  /// Returned count buffers are immutable. Reuse only a completed result from
  /// this source instance; failures never advance the retained mask or image.
  func detectorImages(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool = false
  ) throws -> [MTLBuffer] {
    try detectorImages(
      mask: mask, maximumAdditionalBytes: maximumAdditionalBytes,
      rebase: rebase, selectedAcquisitions: nil)
  }

  /// Exact detector images for either the full resident series or a selected
  /// acquisition. Every requested image is complete at return. Drain temporary
  /// Objective-C command objects per request so a sustained worker does not
  /// retain old GPU publications until its thread's outer autorelease pool.
  func detectorImages(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool = false, selectedAcquisitions: [Int]?
  ) throws -> [MTLBuffer] {
    try autoreleasepool {
      try detectorImagesNow(
        mask: mask, maximumAdditionalBytes: maximumAdditionalBytes,
        rebase: rebase, selectedAcquisitions: selectedAcquisitions)
    }
  }

  /// The synchronous query: submit and finish at once. A synchronous query
  /// never overlaps pipelined ones: they complete and are verified first, so
  /// no later failure can roll back past it. One that writes the output ring
  /// is refused while a pipelined query is unfinished: its caller has not
  /// read that query's images yet, and ring writes outside the in-flight
  /// limit could rewrite them before it does.
  private func detectorImagesNow(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool, selectedAcquisitions: [Int]?
  ) throws -> [MTLBuffer] {
    if detachedDetectorOutputs == nil { try requireNoUnfinishedDetectorQueries() }
    drainDetectorQueries()
    let pending = try submitDetectorImagesNow(
      mask: mask, maximumAdditionalBytes: maximumAdditionalBytes,
      rebase: rebase, selectedAcquisitions: selectedAcquisitions)
    let images = try finishDetectorImagesNow(pending).images
    // As before the split, a cancelled caller receives no images; the
    // completed query's seeds are exact and stay.
    try Task.checkCancellation()
    return images
  }

  /// Pipelined form of `detectorImages`: plan, encode and commit, then return
  /// without waiting. Seeds and ring cursors advance now, so the next query
  /// can be planned and submitted before this one completes; the single
  /// detector queue orders its seed reads after these writes. At most
  /// `maximumDetectorQueriesInFlight` submissions may be unfinished. Every
  /// submission must be finished with `finishDetectorImages`, which reports a
  /// failure of it or of an earlier query it was seeded from.
  func submitDetectorImages(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool = false, selectedAcquisitions: [Int]?
  ) throws -> PendingDetectorQuery {
    guard inFlightDetectorQueries.count < Self.maximumDetectorQueriesInFlight else {
      throw TANSArchive.invalid(
        "At most \(Self.maximumDetectorQueriesInFlight) exact detector queries may be in flight; finish one first"
      )
    }
    guard experimentalDetectorQueueCount == 1, detachedDetectorOutputs == nil else {
      throw TANSArchive.invalid("Pipelined detector queries require the single ordered queue")
    }
    return try autoreleasepool {
      try submitDetectorImagesNow(
        mask: mask, maximumAdditionalBytes: maximumAdditionalBytes,
        rebase: rebase, selectedAcquisitions: selectedAcquisitions)
    }
  }

  /// Wait for a submitted query and every earlier one, verify them in order
  /// and return its complete images. A failure restores the seeds of the
  /// failed query and of every later one, which then also report failure.
  func finishDetectorImages(_ pending: PendingDetectorQuery) throws -> DetectorQueryResult {
    try autoreleasepool { try finishDetectorImagesNow(pending) }
  }

  /// Run reduced-work diagnostics through ordinary request preparation.
  /// Probe completion deliberately throws before replacing any scientific output.
  func measureSubmissionProbe(
    mask: [UInt8], mode: TANSSubmissionProbeMode,
    maximumAdditionalBytes: UInt64, rebase: Bool = false
  ) throws -> TANSSubmissionProbeResult {
    guard inFlightDetectorQueries.isEmpty, submissionProbeMode == nil, useBatchedDetector,
      !usePartialDetector,
      experimentalDetectorStreamsPerLane == 32,
      experimentalDetectorRecordsPerCommand == 64, experimentalDetectorQueueCount == 1,
      !experimentalMetal4Submission, !experimentalSingleCommandEncoders,
      !experimentalSingleComputePass,
      experimentalMemoryAuditURL == nil, experimentalFailAfterSubmittedCommands == nil
    else {
      throw TANSArchive.invalid("Submission probe requires the isolated exact grouped64 baseline")
    }
    if submissionProbePipeline == nil {
      guard let function = tansLibrary.makeFunction(name: "tans_submission_probe") else {
        throw TANSArchive.invalid("Missing isolated submission probe kernel")
      }
      submissionProbePipeline = try device.makeComputePipelineState(function: function)
    }
    submissionProbeWords = []
    submissionProbeMode = mode
    defer { submissionProbeMode = nil }
    do {
      _ = try detectorImages(
        mask: mask, maximumAdditionalBytes: maximumAdditionalBytes, rebase: rebase)
    } catch is TANSSubmissionProbeCompleted {
      return TANSSubmissionProbeResult(
        timing: lastDetectorCommandTiming, records: submissionProbeWords)
    }
    throw TANSArchive.invalid("Diagnostic unexpectedly returned scientific output")
  }

  /// Keep mask decomposition and base selection in one immutable result.
  /// The caller must seed from exactly the base used for these coefficients.
  private func detectorPlan(
    mask: [UInt8], previous: [UInt8]?, preferPrevious: Bool, allowBaseChoice: Bool
  ) throws -> (residual: [Int32], tiles: [(Int, Int32)], usePrevious: Bool, atlasField: Int?) {
    let seeded = preferPrevious && previous != nil
    if experimentalUseTileIndex {
      guard useBatchedDetector, let index = exactTileIndex, let cost = detectorColumnCost else {
        throw TANSArchive.invalid("Prepare the exact tile index before enabling its batched query")
      }
      var chosen: (residual: [Int32], tiles: [(Int, Int32)], usePrevious: Bool)
      if experimentalPlanAfterIndex, allowBaseChoice, let previous {
        chosen = index.planChoosingBase(
          mask: mask, previous: previous, preferPrevious: preferPrevious,
          valid: validDetectorMask, cost: cost, tileCost: experimentalTileCost)
      } else {
        var residual = mask.indices.map {
          seeded ? Int32(mask[$0]) - Int32(previous![$0]) : Int32(mask[$0])
        }
        let tiles = index.plan(
          coefficients: &residual, valid: validDetectorMask, cost: cost,
          tileCost: experimentalTileCost)
        chosen = (residual, tiles, seeded)
      }
      if experimentalUseAtlas,
        let atlas = atlasPlan(
          mask: mask, cost: cost, below: detectorPlanScore(chosen.residual, chosen.tiles, cost))
      {
        return (atlas.residual, atlas.tiles, false, atlas.field)
      }
      return (chosen.residual, chosen.tiles, chosen.usePrevious, nil)
    }
    guard !experimentalUseAtlas else {
      throw TANSArchive.invalid("The exact atlas base requires the tile-index planner")
    }
    let residual = mask.indices.map {
      seeded ? Int32(mask[$0]) - Int32(previous![$0]) : Int32(mask[$0])
    }
    return (residual, [], seeded, nil)
  }

  /// The planner's price of an exact decomposition, as in `planChoosingBase`.
  private func detectorPlanScore(
    _ residual: [Int32], _ tiles: [(Int, Int32)], _ cost: [Double]
  ) -> Double {
    var total = Double(tiles.count) * experimentalTileCost
    for pixel in residual.indices where residual[pixel] != 0 { total += cost[pixel] }
    return total
  }

  /// Cheapest exact atlas base for `mask`, or nil when none beats `bound`.
  /// Stored masks are ranked by centroid distance, the nearest few by
  /// differing pixels (a word-wise XOR count), and the best is priced by
  /// planner cost. Ranking only chooses a base: the residual against that
  /// base's stored bytes is always -1...1. Tile substitution is not tried on
  /// atlas residuals; they are thin crescents that an 8x8 tile does not fit.
  private func atlasPlan(
    mask: [UInt8], cost: [Double], below bound: Double
  ) -> (residual: [Int32], tiles: [(Int, Int32)], field: Int, score: Double)? {
    guard experimentalAtlas != nil, !experimentalAtlasMasks.isEmpty else { return nil }
    let centroid = Self.maskCentroid(mask)
    let byCentroid = experimentalAtlasCentroids.enumerated().map { field, stored in
      let dr = stored.row - centroid.row
      let dc = stored.col - centroid.col
      return (field: field, distance: dr * dr + dc * dc)
    }.sorted { $0.distance < $1.distance }
    let ranked = mask.withUnsafeBytes { request in
      byCentroid.prefix(8).map { candidate in
        experimentalAtlasMasks[candidate.field].withUnsafeBytes { stored in
          var differing = 0
          for offset in stride(from: 0, to: 36864, by: 8) {
            differing +=
              (request.loadUnaligned(fromByteOffset: offset, as: UInt64.self)
              ^ stored.loadUnaligned(fromByteOffset: offset, as: UInt64.self)).nonzeroBitCount
          }
          return (field: candidate.field, differing: differing)
        }
      }
    }.sorted { $0.differing < $1.differing }
    var best = -1
    var bestCost = Double.infinity
    for candidate in ranked.prefix(4) {
      let stored = experimentalAtlasMasks[candidate.field]
      var priced = 0.0
      for pixel in 0..<36864 where mask[pixel] != stored[pixel] { priced += cost[pixel] }
      if priced < bestCost {
        bestCost = priced
        best = candidate.field
      }
    }
    // The field add is priced like one tile add.
    let score = experimentalTileCost + bestCost
    guard best >= 0, score < bound else { return nil }
    let stored = experimentalAtlasMasks[best]
    return (mask.indices.map { Int32(mask[$0]) - Int32(stored[$0]) }, [], best, score)
  }

  private static func maskCentroid(_ mask: [UInt8]) -> (row: Double, col: Double) {
    var row = 0
    var col = 0
    var count = 0
    for q in 0..<36864 where mask[q] != 0 {
      row += q / 192
      col += q % 192
      count += 1
    }
    return count == 0 ? (0, 0) : (Double(row) / Double(count), Double(col) / Double(count))
  }

  /// Requested acquisitions that share one exact seed mask, so one residual
  /// decomposition serves all of them in a single batch.
  private struct DetectorGroup {
    var retained: [Int]
    var acquisitions: [Int]
    var previous: [UInt8]?
    var seedImages: [MTLBuffer]
  }

  /// The ring slot a query may write for this acquisition: the slot after the
  /// current seed image. Allocation is lazy and bounded to three complete
  /// images per retained acquisition; the cursor advances when the query is
  /// submitted and returns to its prior value if the query fails.
  private func detectorOutputSlot(retained: Int) throws -> (image: MTLBuffer, slot: Int) {
    if detectorImageRings[retained] == nil {
      var ring: [MTLBuffer] = []
      for _ in 0..<Self.detectorImageRingSlots {
        guard let image = device.makeBuffer(length: 512 * 512 * 4, options: .storageModeShared)
        else {
          throw TANSArchive.invalid("Cannot allocate complete exact virtual images")
        }
        ring.append(image)
      }
      detectorImageRings[retained] = ring
    }
    let slot = (detectorRingCursors[retained, default: -1] + 1) % Self.detectorImageRingSlots
    return (detectorImageRings[retained]![slot], slot)
  }

  private func submitDetectorImagesNow(
    mask: [UInt8], maximumAdditionalBytes: UInt64,
    rebase: Bool, selectedAcquisitions: [Int]?
  ) throws -> PendingDetectorQuery {
    guard !isReleased, mask.count == 36864, mask.allSatisfy({ $0 <= 1 }) else {
      throw TANSArchive.invalid("Provide a binary 192x192 detector mask on a live entropy series")
    }
    let outputAcquisitionIndices = selectedAcquisitions ?? acquisitionIndices
    guard !outputAcquisitionIndices.isEmpty,
      Set(outputAcquisitionIndices).count == outputAcquisitionIndices.count,
      outputAcquisitionIndices.allSatisfy({ acquisitionIndices.contains($0) })
    else {
      throw TANSArchive.invalid("Select unique retained detector acquisition indices")
    }
    let fullCount = mask.reduce(0) { $0 + Int($1) }
    // Group the requested acquisitions by identical exact seed. Seeds are per
    // acquisition, so an interactive single-acquisition query and a later
    // catch-up over the others each continue from their own completed image.
    var groups: [DetectorGroup] = []
    var groupIndexByMaskID: [UInt64: Int] = [:]
    // Seeds written by separate calls carry different ids even when their
    // masks are byte-identical (a catch-up pass refreshes acquisitions one call
    // at a time). Matching by content keeps such acquisitions in one batch; an
    // id seen once is resolved once, so the common single-id case never hashes.
    var groupIndicesByContent: [Int: [Int]] = [:]
    var fullGroupIndex: Int?
    for acquisition in outputAcquisitionIndices {
      guard let retained = acquisitionIndices.firstIndex(of: acquisition) else {
        throw TANSArchive.invalid("Selected detector acquisition is not retained")
      }
      if !rebase, let seed = detectorSeeds[retained] {
        var index = groupIndexByMaskID[seed.maskID]
        if index == nil {
          var hasher = Hasher()
          hasher.combine(seed.mask)
          let contentKey = hasher.finalize()
          // A hash match is only a candidate; the full byte comparison decides.
          index = groupIndicesByContent[contentKey]?.first { groups[$0].previous == seed.mask }
          if index == nil {
            index = groups.count
            groups.append(
              DetectorGroup(
                retained: [], acquisitions: [], previous: seed.mask, seedImages: []))
            groupIndicesByContent[contentKey, default: []].append(index!)
          }
          groupIndexByMaskID[seed.maskID] = index
        }
        groups[index!].retained.append(retained)
        groups[index!].acquisitions.append(acquisition)
        groups[index!].seedImages.append(seed.image)
      } else if let index = fullGroupIndex {
        groups[index].retained.append(retained)
        groups[index].acquisitions.append(acquisition)
      } else {
        fullGroupIndex = groups.count
        groups.append(
          DetectorGroup(
            retained: [retained], acquisitions: [acquisition], previous: nil, seedImages: []))
      }
    }
    let maskID = nextDetectorMaskID
    nextDetectorMaskID += 1
    let pending = PendingDetectorQuery(
      sequence: nextDetectorQuerySequence, outputAcquisitionIndices: outputAcquisitionIndices,
      queriesInFlightAtSubmit: inFlightDetectorQueries.count)
    nextDetectorQuerySequence += 1
    // Finishing verifies every earlier query too, so its completion includes
    // theirs: a caller that awaits it never blocks in the finish step, even
    // when the GPU completes an independent later query first.
    for earlier in inFlightDetectorQueries { pending.completion.track(earlier.completion) }
    do {
      for group in groups {
        // Encode the group before touching `pending.groups`: the call records
        // seed priors on `pending` and may throw, so no access to `pending`
        // is held open across it.
        let submitted = try runDetectorGroup(
          mask: mask, fullCount: fullCount, group: group,
          maximumAdditionalBytes: maximumAdditionalBytes, maskID: maskID, pending: pending)
        pending.groups.append(submitted)
      }
    } catch {
      // Earlier groups of this query are already committed: let them finish
      // before their temporaries are released, then restore their seeds.
      for command in pending.commands { command.waitUntilCompleted() }
      rollBack(pending)
      pending.completion.seal()
      throw error
    }
    pending.completion.seal()
    inFlightDetectorQueries.append(pending)
    return pending
  }

  /// Complete a submitted query: wait for it and every earlier query, verify
  /// their commands in submission order, then report this query's images and
  /// diagnostics exactly as the synchronous query does.
  private func finishDetectorImagesNow(_ pending: PendingDetectorQuery) throws
    -> DetectorQueryResult
  {
    guard !isReleased else { throw TANSArchive.invalid("tANS series has been released") }
    guard let position = inFlightDetectorQueries.firstIndex(where: { $0 === pending }) else {
      throw TANSArchive.invalid("Exact detector query was already finished or never submitted")
    }
    verifyDetectorQueries(through: position)
    var outcomes: [(seconds: Double, timing: [String: Double])] = []
    if pending.failure == nil {
      do {
        for group in pending.groups { outcomes.append(try group.batch.finish()) }
      } catch {
        poisonDetectorQueries(from: position, error: error)
      }
    }
    inFlightDetectorQueries.remove(at: position)
    if let failure = pending.failure { throw failure }
    var imagesByAcquisition: [Int: MTLBuffer] = [:]
    var totalColumns = 0
    var totalTiles = 0
    var usedPreviousEverywhere = true
    var gpuSeconds = 0.0
    var planSeconds = 0.0
    var atlasFields: [Int?] = []
    var timing: [String: Double] = [:]
    for (ordinal, (group, outcome)) in zip(pending.groups, outcomes).enumerated() {
      for (acquisition, image) in zip(group.acquisitions, group.images) {
        imagesByAcquisition[acquisition] = image
      }
      totalColumns += group.columns
      totalTiles += group.tiles
      usedPreviousEverywhere = usedPreviousEverywhere && group.usedPrevious
      gpuSeconds += outcome.seconds
      planSeconds += group.planSeconds
      atlasFields.append(group.atlasField)
      if ordinal == 0 {
        timing = outcome.timing
      } else {
        for key in [
          "encode_ms", "commit_ms", "wait_ms", "gpu_intervals_sum_ms",
          "first_command_to_last_gpu_end_ms", "submitted_commands",
        ] {
          timing[key] = (timing[key] ?? 0) + (outcome.timing[key] ?? 0)
        }
      }
    }
    timing["seed_groups"] = Double(pending.groups.count)
    // Groups can start from different bases. Report the first group's, which
    // holds the first requested acquisition (the inspected one for interactive
    // frames), and how many groups started from an atlas image.
    lastDetectorAtlasField = atlasFields.first ?? nil
    timing["atlas_field"] = Double(lastDetectorAtlasField ?? -1)
    timing["atlas_groups"] = Double(atlasFields.compactMap { $0 }.count)
    timing["plan_ms"] = planSeconds * 1000
    timing["queries_in_flight_at_submit"] = Double(pending.queriesInFlightAtSubmit)
    lastDetectorCommandTiming = timing
    lastDetectorDecodedColumns = totalColumns
    lastDetectorTileFields = totalTiles
    lastDetectorUsedPrevious = usedPreviousEverywhere
    lastDetectorGPUSeconds = gpuSeconds
    if let last = pending.groups.last {
      lastDetectorPlanSeconds = last.planSeconds
      lastDetectorScratchBytes = last.scratchBytes
      lastDetectorModelGroups = last.modelGroups
      lastDetectorMixedModelGroups = last.mixedModelGroups
      lastDetectorPaddedModelLanes = last.paddedModelLanes
    }
    return DetectorQueryResult(
      images: pending.outputAcquisitionIndices.map { imagesByAcquisition[$0]! },
      gpuSeconds: gpuSeconds, decodedColumns: totalColumns,
      usedPrevious: usedPreviousEverywhere, atlasField: lastDetectorAtlasField, timing: timing)
  }

  /// Ring writes outside the pipelined in-flight limit (synchronous queries,
  /// index construction) wait until every submission has been finished.
  private func requireNoUnfinishedDetectorQueries() throws {
    guard inFlightDetectorQueries.isEmpty else {
      throw TANSArchive.invalid(
        "Finish the \(inFlightDetectorQueries.count) submitted exact detector queries first; this call would rewrite output images their caller has not read"
      )
    }
  }

  /// Wait for every in-flight query and verify it. Nothing is removed: each
  /// is still finished by its own caller.
  private func drainDetectorQueries() {
    guard !inFlightDetectorQueries.isEmpty else { return }
    verifyDetectorQueries(through: inFlightDetectorQueries.count - 1)
  }

  /// Wait for the in-flight queries up to `position` and check their commands
  /// in submission order. The first failed query is rolled back together with
  /// every later one; verified queries can no longer fail.
  private func verifyDetectorQueries(through position: Int) {
    let queries = inFlightDetectorQueries[...position]
    for query in queries {
      for command in query.commands { command.waitUntilCompleted() }
    }
    for (index, query) in zip(queries.indices, queries)
    where !query.verified && query.failure == nil {
      let failed = query.commands.first { $0.status != .completed }
      if failed != nil || experimentalFailedDetectorQuerySequences.contains(query.sequence) {
        let reason =
          failed.map { String(describing: $0.error) }
          ?? "injected failure of query \(query.sequence)"
        poisonDetectorQueries(
          from: index, error: TANSArchive.invalid("Batched exact detector query failed: \(reason)"))
        return
      }
      query.verified = true
    }
  }

  /// Restore the seeds and ring cursors of the query at `index` and of every
  /// later query (their seeds are its images), newest first, and mark them
  /// failed. Images returned by finished queries are never touched.
  private func poisonDetectorQueries(from index: Int, error: Error) {
    let failed = inFlightDetectorQueries[index]
    for query in inFlightDetectorQueries[index...].reversed() {
      rollBack(query)
      if query.failure == nil {
        query.failure =
          query === failed
          ? error
          : TANSArchive.invalid(
            "Exact detector query \(query.sequence) was seeded from failed query \(failed.sequence); prior images preserved"
          )
      }
    }
  }

  private func rollBack(_ query: PendingDetectorQuery) {
    for prior in query.priors.reversed() {
      detectorSeeds[prior.retained] = prior.seed
      detectorRingCursors[prior.retained] = prior.cursor
    }
    query.priors.removeAll()
  }

  /// Encode and commit one seed group. Its seeds and ring cursors advance at
  /// submit; their prior values are recorded in `pending` for rollback.
  private func runDetectorGroup(
    mask: [UInt8], fullCount: Int, group: DetectorGroup,
    maximumAdditionalBytes: UInt64, maskID: UInt64, pending: PendingDetectorQuery
  ) throws -> SubmittedDetectorGroup {
    let outputAcquisitionIndices = group.acquisitions
    let difference = group.previous.map { old in
      zip(mask, old).reduce(0) { $0 + ($1.0 != $1.1 ? 1 : 0) }
    }
    let preferPrevious = difference.map { $0 < fullCount } ?? false
    let planStarted = ProcessInfo.processInfo.systemUptime
    let plan = try detectorPlan(
      mask: mask, previous: group.previous, preferPrevious: preferPrevious,
      allowBaseChoice: group.previous != nil)
    lastDetectorPlanSeconds = ProcessInfo.processInfo.systemUptime - planStarted
    let residual = plan.residual
    let selectedTiles = plan.tiles
    let usePrevious = plan.usePrevious
    let atlasField = plan.atlasField
    lastDetectorTileFields = selectedTiles.count
    lastDetectorAtlasField = atlasField
    let unsorted = mask.indices.filter { residual[$0] != 0 }
    let dense = unsorted.filter { cacheMapValues[$0] < 0 }
    let sparse = unsorted.filter { cacheMapValues[$0] >= 0 }
    let selected = (useBatchedDetector ? dense + sparse : unsorted).map { UInt32($0) }
    let signs: [Int32] = selected.map { residual[Int($0)] }
    let groups = (selected.count + 31) / 32
    // The batched kernel is also valid for a selected acquisition. Its record
    // table is narrowed to the requested acquisition's 16 entropy chunks, so
    // one dispatch can cover the complete selected image without changing the
    // exact integer/delta contract.
    let useBatchForRequest = useBatchedDetector
    // A full-series delta has one disjoint partial range per chunk/group. It
    // avoids atomics when that exact scratch fits the caller's budget; large
    // masks (notably full ADF) retain the bounded atomic fallback.
    let outputBytes = outputAcquisitionIndices.count * 512 * 512 * 4
    let selectionBytes = max(4, selected.count * 4)
    let indirectBytes =
      (useBatchForRequest ? chunks.count * 44 : 0)
      + (selectedTiles.isEmpty
        ? 0
        : (exactTileIndex!.fields.count * 64 + selectedTiles.count * 8 + outputAcquisitionIndices
          .count * 12))
      + (atlasField == nil
        ? 0 : 64 + 8 + outputAcquisitionIndices.count * 12)
    let overheadBytes = outputBytes + selectionBytes * 2 + indirectBytes
    let batchRecordCount = outputAcquisitionIndices.count * 16
    let partialGroups = (dense.count + 31) / 32
    let partialBytesPerRecord = partialGroups * 32 * 512 * 4
    let availableScratchBytes =
      maximumAdditionalBytes > UInt64(overheadBytes)
      ? maximumAdditionalBytes - UInt64(overheadBytes) : 0
    let partialRecordBudget =
      partialBytesPerRecord > 0
      ? Int(min(availableScratchBytes, UInt64(device.maxBufferLength))) / partialBytesPerRecord : 0
    let usePartialBatch =
      usePartialDetector && useBatchForRequest
      && outputAcquisitionIndices.count == acquisitionIndices.count
      && dense.count > 0 && partialRecordBudget > 0
    let partialBatchBytes =
      usePartialBatch
      ? min(batchRecordCount, partialRecordBudget) * partialBytesPerRecord : 0
    let scratchBytes =
      usePartialBatch
      ? partialBatchBytes
      : (useBatchForRequest ? 4 : max(4, groups * 32 * 512 * 4))
    guard
      UInt64(scratchBytes + outputBytes + selectionBytes * 2 + indirectBytes)
        <= maximumAdditionalBytes,
      scratchBytes <= device.maxBufferLength
    else {
      throw TANSArchive.invalid(
        "Insufficient budget for exact detector partials and 2D outputs; prior output preserved")
    }
    guard let scratch = device.makeBuffer(length: scratchBytes, options: .storageModePrivate),
      let selection = device.makeBuffer(length: selectionBytes, options: .storageModeShared),
      let coefficients = device.makeBuffer(length: selectionBytes, options: .storageModeShared)
    else {
      throw TANSArchive.invalid("Cannot allocate exact detector partials")
    }
    // Plain copies, without closures. The two former `withUnsafeBytes`
    // closures were identical, the release optimizer merged them into one
    // function that did not preserve the Swift error register, and the
    // caller adopted that register as its error value: a query failed with
    // an error that was never thrown (Swift 6.3, release build).
    if !selected.isEmpty {
      selection.contents().copyMemory(
        from: selected, byteCount: selected.count * MemoryLayout<UInt32>.stride)
    }
    if !signs.isEmpty {
      coefficients.contents().copyMemory(
        from: signs, byteCount: signs.count * MemoryLayout<Int32>.stride)
    }
    let slots = try group.retained.map { retained -> (image: MTLBuffer, slot: Int) in
      if let detached = detachedDetectorOutputs { return (detached[retained], -1) }
      return try detectorOutputSlot(retained: retained)
    }
    let images = slots.map(\.image)
    lastDetectorScratchBytes = scratchBytes + selectionBytes * 2 + indirectBytes
    lastDetectorDecodedColumns = selected.count
    lastDetectorUsedPrevious = usePrevious
    lastDetectorGPUSeconds = 0
    func commitSeeds() {
      guard detachedDetectorOutputs == nil else { return }
      for (index, retained) in group.retained.enumerated() {
        pending.priors.append((retained, detectorSeeds[retained], detectorRingCursors[retained]))
        detectorSeeds[retained] = DetectorSeed(mask: mask, maskID: maskID, image: images[index])
        detectorRingCursors[retained] = slots[index].slot
      }
    }
    // The next query plans from this group's mask and reads its images as
    // seeds; both are fixed at commit, so the seeds advance now.
    func submitted(_ batch: DetectorBatchSubmission) throws -> SubmittedDetectorGroup {
      do {
        try Task.checkCancellation()
      } catch {
        for command in batch.commands { command.waitUntilCompleted() }
        throw error
      }
      commitSeeds()
      return SubmittedDetectorGroup(
        acquisitions: outputAcquisitionIndices, images: images, columns: selected.count,
        tiles: selectedTiles.count, usedPrevious: usePrevious, planSeconds: lastDetectorPlanSeconds,
        atlasField: atlasField, scratchBytes: lastDetectorScratchBytes,
        modelGroups: lastDetectorModelGroups, mixedModelGroups: lastDetectorMixedModelGroups,
        paddedModelLanes: lastDetectorPaddedModelLanes, batch: batch)
    }
    if useBatchForRequest {
      // The batched kernel is also valid for a selected acquisition. Its record
      // table is narrowed to the requested acquisitions' entropy chunks, so
      // one dispatch covers each complete selected image without changing the
      // exact integer/delta contract. Chunk scan offsets stay distinct for
      // subsets as well; a fused sum across chunks is not a valid image.
      let batch = try detectorBatch(
        images: images, seedImages: usePrevious ? group.seedImages : [], selection: selection,
        coefficients: coefficients, selectedCount: selected.count,
        denseCount: dense.count, seed: usePrevious, partials: usePartialBatch ? scratch : nil,
        outputAcquisitionIndices: outputAcquisitionIndices, selectedTiles: selectedTiles,
        atlasField: atlasField,
        metadataBudget: maximumAdditionalBytes - UInt64(scratchBytes + overheadBytes),
        completion: pending.completion)
      return try submitted(batch)
    }
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw TANSArchive.invalid("Cannot encode exact detector query")
    }
    // One serial compute encoder replaces 2,112 encoders and 66 CPU/GPU waits.
    // Explicit barriers protect the bounded scratch between producer/consumer
    // dispatches; every final 2D output is separate from the prior publication.
    defer { if command.status == .notEnqueued { encoder.endEncoding() } }
    for (index, acquisition) in outputAcquisitionIndices.enumerated() {
      try Task.checkCancellation()
      guard let retainedIndex = acquisitionIndices.firstIndex(of: acquisition) else {
        throw TANSArchive.invalid("Selected detector acquisition is not retained")
      }
      for localChunk in 0..<16 {
        let recordIndex = retainedIndex * 16 + localChunk
        let chunk = chunks[recordIndex]
        if groups > 0 {
          encoder.setComputePipelineState(detectorPipeline)
          for (binding, name) in ["dense", "dense_offsets", "sparse", "sparse_offsets"].enumerated()
          {
            guard let component = chunk.components.first(where: { $0.name == name }) else {
              throw TANSArchive.invalid("Missing authenticated detector source")
            }
            encoder.setBuffer(
              records[recordIndex], offset: recordBaseOffsets[recordIndex] + component.offset,
              index: binding)
          }
          for (binding, buffer) in globals.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: binding + 4)
          }
          encoder.setBuffer(scratch, offset: 0, index: 8)
          var parameters: [UInt32] = [
            retainedColumns, sparseColumns,
            UInt32((acquisition * 4 + localChunk / 4) * 36864), UInt32(selected.count),
            UInt32(groups),
          ]
          encoder.setBytes(&parameters, length: 20, index: 9)
          encoder.setBuffer(selection, offset: 0, index: 10)
          encoder.setBuffer(coefficients, offset: 0, index: 11)
          encoder.dispatchThreadgroups(
            MTLSize(width: groups, height: 32, depth: 1),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
          encoder.memoryBarrier(scope: .buffers)
        }
        encoder.setComputePipelineState(detectorFinishPipeline)
        encoder.setBuffer(scratch, offset: 0, index: 0)
        encoder.setBuffer(images[index], offset: localChunk * 16384 * 4, index: 1)
        var count = UInt32(groups)
        encoder.setBytes(&count, length: 4, index: 2)
        if usePrevious {
          guard index < group.seedImages.count else {
            throw TANSArchive.invalid("Previous detector image set does not match selection")
          }
          encoder.setBuffer(group.seedImages[index], offset: localChunk * 16384 * 4, index: 3)
        } else {
          encoder.setBuffer(images[index], offset: localChunk * 16384 * 4, index: 3)
        }
        var seed: UInt32 = usePrevious ? 1 : 0
        encoder.setBytes(&seed, length: 4, index: 4)
        encoder.dispatchThreads(
          MTLSize(width: 16384, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
        encoder.memoryBarrier(scope: .buffers)
      }
    }
    encoder.endEncoding()
    pending.completion.track(command)
    command.commit()
    command.waitUntilCompleted()
    guard command.status == .completed else {
      throw TANSArchive.invalid("Exact detector query failed: \(String(describing:command.error))")
    }
    let seconds = command.gpuEndTime - command.gpuStartTime
    lastDetectorGPUSeconds = seconds
    // This reference path stays synchronous and records no command timing.
    let timing = lastDetectorCommandTiming
    return try submitted(
      DetectorBatchSubmission(commands: [command], finish: { (seconds, timing) }))
  }

  private func detectorBatch(
    images: [MTLBuffer], seedImages: [MTLBuffer], selection: MTLBuffer, coefficients: MTLBuffer,
    selectedCount: Int, denseCount: Int, seed: Bool,
    partials: MTLBuffer?,
    outputAcquisitionIndices: [Int], selectedTiles: [(Int, Int32)], atlasField: Int? = nil,
    metadataBudget: UInt64, completion: TANSDetectorCompletion
  ) throws -> DetectorBatchSubmission {
    let encodeStarted = ProcessInfo.processInfo.systemUptime
    guard !seed || seedImages.count == images.count else {
      throw TANSArchive.invalid("Previous detector image set does not match selection")
    }
    guard
      !experimentalSingleComputePass
        || (experimentalDetectorQueueCount == 1 && experimentalDetectorRecordsPerCommand == 64
          && partials == nil && experimentalMemoryAuditURL == nil
          && !experimentalSingleCommandEncoders && !experimentalMetal4Submission
          && !experimentalConcurrentDetectorDispatches && !experimentalNarrowOutputDeclarations
          && submissionProbeMode == nil
          && experimentalFailAfterSubmittedCommands == nil)
    else {
      throw TANSArchive.invalid(
        "Single compute pass requires the isolated serial grouped64 scientific query")
    }
    guard
      !experimentalSingleCommandEncoders
        || (experimentalDetectorQueueCount == 1 && experimentalDetectorRecordsPerCommand == 64
          && partials == nil && experimentalMemoryAuditURL == nil)
    else {
      throw TANSArchive.invalid(
        "Single-command encoder experiment requires one queue and 64-record grids without diagnostic counters"
      )
    }
    let auditSamples: MTLCounterSampleBuffer?
    if experimentalMemoryAuditURL != nil {
      guard device.supportsCounterSampling(.atStageBoundary),
        let counters = device.counterSets?.first(where: { $0.name == "timestamp" })
      else { throw TANSArchive.invalid("Audit requires public stage timestamp counters") }
      let descriptor = MTLCounterSampleBufferDescriptor()
      descriptor.counterSet = counters
      descriptor.storageMode = .shared
      descriptor.sampleCount = 256
      auditSamples = try device.makeCounterSampleBuffer(descriptor: descriptor)
    } else {
      auditSamples = nil
    }
    var auditStageCount = 0
    var auditCommits: [Double] = []
    guard [1, 4].contains(experimentalDetectorQueueCount),
      experimentalDetectorQueueCount == 1
        || (experimentalDetectorRecordsPerCommand == 64 && partials == nil
          && !acquisitionStorage.isEmpty && experimentalNarrowOutputDeclarations
          && experimentalResidencyHold == nil && !experimentalInvocationOwnedSubmissions)
    else {
      throw TANSArchive.invalid(
        "Parallel detector queues require disjoint coalesced64-record shards")
    }
    if experimentalDetectorQueueCount == 4 && experimentalExtraDetectorQueues.isEmpty {
      experimentalExtraDetectorQueues = try (0..<3).map { _ in
        guard let extra = device.makeCommandQueue() else {
          throw TANSArchive.invalid("Cannot allocate exact detector submission queue")
        }
        return extra
      }
    }
    guard [0, 16, 64, 256].contains(experimentalDetectorRecordsPerCommand),
      experimentalDetectorRecordsPerCommand == 0
        || (experimentalDetectorStreamsPerLane == 32 && partials == nil)
    else {
      throw TANSArchive.invalid("Submission overlap requires shared mode32 without partial scratch")
    }
    guard
      !experimentalUseResidentSetForSourceReads
        || (experimentalResidencyHold != nil && experimentalResidencyAttachedToQueue)
    else {
      throw TANSArchive.invalid("Explicit source residency requires a live residency set")
    }
    guard device.argumentBuffersSupport == .tier2 else {
      throw TANSArchive.invalid("Batched entropy query requires Metal argument-buffer tier 2")
    }
    guard
      [0, 1, 2, 4, 8, 32, 64, 65, 66, 67, 68, 69, 70].contains(experimentalDetectorStreamsPerLane)
    else {
      throw TANSArchive.invalid(
        "Experimental detector mode must be 0, 1, 2, 4, 8, 32 or a registered CUDA-followup mode 64...70"
      )
    }
    let streams = experimentalDetectorStreamsPerLane
    let useShared = streams >= 32
    let pairReduction = [64, 66, 67, 68, 69, 70].contains(streams)
    let sharedThreads =
      experimentalSharedModelThreadgroupWidth
      ?? ([67, 70].contains(streams) ? 512 : 128)
    guard [32, 64, 128, 256, 512].contains(sharedThreads) else {
      throw TANSArchive.invalid("Shared-model width must be 32, 64, 128, 256 or 512 threads")
    }
    let packets = experimentalSharedPacketsPerLane
    guard [0, 4, 6].contains(experimentalPairLookupBits),
      experimentalPairLookupBits == 0
        || (streams == 32 && packets == 1 && sharedThreads == 128
          && preparedPairLookupBits == experimentalPairLookupBits && pairLookup != nil
          && !experimentalSharedRefill16 && !experimentalDirectDecodingTable
          && !experimentalZeroRunDecoding && !experimentalZeroBitArithmetic
          && !experimentalPreparedZeroRuns && !experimentalSignedPairReduction
          && !experimentalPrefetchDecoderEntry && !experimentalDecoderBitExtract
          && !experimentalStagedDetectorReduction && !experimentalPacketMajorGrid
          && experimentalDeferredReductionPairs == 1 && experimentalPairLoopUnroll == 1)
    else { throw TANSArchive.invalid("Pair lookup requires prepared exact metadata and shared32") }
    guard
      !experimentalPacketMajorGrid
        || (streams == 32 && packets == 1 && !experimentalDecoderBitExtract
          && !experimentalPrefetchDecoderEntry && experimentalDeferredReductionPairs == 1
          && experimentalPairLoopUnroll == 1 && !experimentalStagedDetectorReduction)
    else { throw TANSArchive.invalid("Packet-major grid requires the frozen shared32 decoder") }
    guard
      !experimentalDecoderBitExtract
        || (streams == 32 && packets == 1 && experimentalPairLoopUnroll == 1
          && experimentalDeferredReductionPairs == 1 && !experimentalPrefetchDecoderEntry
          && !experimentalStagedDetectorReduction && !experimentalSharedRefill16
          && !experimentalZeroRunDecoding && !experimentalZeroBitArithmetic
          && !experimentalDirectDecodingTable)
    else { throw TANSArchive.invalid("Bit extraction requires the frozen exact shared32 decoder") }
    guard [1, 2, 4, 8].contains(experimentalDeferredReductionPairs),
      experimentalDeferredReductionPairs == 1
        || (streams == 32 && packets == 1 && experimentalPairLoopUnroll == 1
          && !experimentalStagedDetectorReduction && !experimentalPrefetchDecoderEntry)
    else { throw TANSArchive.invalid("Deferred reduction requires shared32 and2,4or8 exact pairs") }
    guard
      !experimentalPrefetchDecoderEntry
        || (streams == 32 && packets == 1 && !experimentalSharedRefill16
          && !experimentalDirectDecodingTable && !experimentalZeroRunDecoding
          && !experimentalZeroBitArithmetic && !experimentalStagedDetectorReduction)
    else { throw TANSArchive.invalid("Entry prefetch requires the frozen exact shared32 decoder") }
    guard
      !experimentalStagedDetectorReduction
        || (streams == 32 && sharedThreads == 128 && packets == 1)
    else {
      throw TANSArchive.invalid("Staged reduction requires shared32 with128threads and one packet")
    }
    guard [0, 128, 256, 512].contains(experimentalCompilerThreadgroupLimit),
      experimentalCompilerThreadgroupLimit == 0
        || (streams == 32 && sharedThreads <= experimentalCompilerThreadgroupLimit && packets == 1)
    else {
      throw TANSArchive.invalid("Compiler thread limit must cover the unchanged shared launch")
    }
    guard [1, 2, 4, 8].contains(experimentalPairLoopUnroll),
      experimentalPairLoopUnroll == 1 || (streams == 32 && packets == 1)
    else {
      throw TANSArchive.invalid("Pair-loop unrolling requires shared mode32 and factor1,2,4or8")
    }
    guard [1, 2, 4].contains(packets), packets == 1 || streams == 32,
      sharedThreads * packets <= 1024
    else {
      throw TANSArchive.invalid("Packet ILP requires shared mode32 and one, two or four packets")
    }
    guard !experimentalSharedRefill16 || (streams == 32 && packets == 1) else {
      throw TANSArchive.invalid("Halfword refill requires shared mode32 and one packet")
    }
    guard
      !experimentalDirectDecodingTable
        || (streams == 32 && packets == 1 && !experimentalSharedRefill16)
    else {
      throw TANSArchive.invalid("Direct table lookup requires the unchanged shared mode32 decoder")
    }
    guard
      !experimentalZeroRunDecoding
        || (streams == 32 && packets == 1 && !experimentalSharedRefill16
          && !experimentalDirectDecodingTable)
    else {
      throw TANSArchive.invalid("Zero-run decoding requires the unchanged shared mode32 table")
    }
    guard
      !experimentalZeroBitArithmetic
        || (streams == 32 && packets == 1 && !experimentalSharedRefill16
          && !experimentalDirectDecodingTable && !experimentalZeroRunDecoding)
    else {
      throw TANSArchive.invalid("Zero-bit arithmetic requires the unchanged shared mode32 table")
    }
    guard
      !experimentalPreparedZeroRuns
        || (experimentalZeroRunDecoding && streams == 32 && packets == 1 && partials == nil)
    else {
      throw TANSArchive.invalid("Prepared zero-run metadata requires exact shared32 zero decoding")
    }
    guard
      !experimentalMixedModelTails
        || (streams == 32 && packets == 1 && !experimentalSharedRefill16
          && !experimentalDirectDecodingTable
          && (!experimentalZeroRunDecoding || experimentalPreparedZeroRuns)
          && !experimentalZeroBitArithmetic
          && [0, 8].contains(experimentalMixedTailSavingsDivisor))
    else {
      throw TANSArchive.invalid("Mixed model tails require the unchanged shared mode32 decoder")
    }
    guard !experimentalSeparateMixedDispatches || experimentalMixedModelTails else {
      throw TANSArchive.invalid("Separated mixed dispatch requires explicit mixed-tail grouping")
    }
    guard !experimentalMixedOnlySpecialization || experimentalSeparateMixedDispatches else {
      throw TANSArchive.invalid("Mixed-only specialization requires separated mixed dispatch")
    }
    guard !experimentalSignedPairReduction || (streams == 32 && packets == 1) else {
      throw TANSArchive.invalid("Signed pair reduction requires shared mode32 and one packet")
    }
    // The plain shared32 decoder reduces with the exact packed transpose
    // butterfly; the frozen decoder experiments keep the per-pair reductions.
    let transposeReduction =
      streams == 32 && packets == 1 && !pairReduction
      && !experimentalSharedRefill16 && !experimentalDirectDecodingTable
      && !experimentalZeroRunDecoding && !experimentalZeroBitArithmetic
      && !experimentalSignedPairReduction && experimentalPairLookupBits == 0
      && !experimentalStagedDetectorReduction && experimentalDeferredReductionPairs == 1
      && experimentalPairLoopUnroll == 1 && !experimentalDecoderBitExtract
    // Compact work-list grids (one threadgroup row per real (record, model
    // group) instead of max-groups padding per record) apply to the separated
    // shared32 dispatches.
    let compactGrid =
      streams == 32 && packets == 1 && experimentalSeparateMixedDispatches
      && !experimentalPacketMajorGrid && submissionProbeMode == nil
    // Key bits: 0-2 packets, 3-15 threads, 16-22 streams, 23 narrow mixed P=2,
    // 24-56 the frozen decoder experiments, 57 compact grid, 59 narrow mixed
    // P=4, 60 transpose reduction.
    let sharedPipelineKey =
      streams * 65536 + sharedThreads * 8 + packets
      + (experimentalSharedRefill16 ? 1 << 24 : 0)
      + (experimentalDirectDecodingTable ? 1 << 25 : 0)
      + (experimentalZeroRunDecoding ? 1 << 26 : 0)
      + (experimentalZeroBitArithmetic ? 1 << 27 : 0)
      + (experimentalMixedModelTails ? 1 << 28 : 0)
      + (experimentalMixedOnlySpecialization ? 1 << 29 : 0)
      + (experimentalSignedPairReduction ? 1 << 30 : 0)
      + (experimentalPreparedZeroRuns ? 1 << 31 : 0)
      + (experimentalPairLoopUnroll << 32)
      + (experimentalCompilerThreadgroupLimit << 36)
      + (experimentalStagedDetectorReduction ? 1 << 46 : 0)
      + (experimentalPrefetchDecoderEntry ? 1 << 47 : 0)
      + (experimentalDeferredReductionPairs << 48)
      + (experimentalDecoderBitExtract ? 1 << 52 : 0)
      + (experimentalPacketMajorGrid ? 1 << 53 : 0)
      + (experimentalPairLookupBits << 54)
      + (compactGrid ? 1 << 57 : 0)
      + (transposeReduction ? 1 << 60 : 0)
    let homogeneousPipelineKey = sharedPipelineKey & ~((1 << 28) | (1 << 29))
    // Mixed groups whose active lanes fit 8 or 16 slots (tiny residuals, the
    // last tail per record) decode 4 or 2 packets per SIMD group. Host-verified
    // per group: six-bit lanes only, coefficients +-1.
    let narrowMixed =
      compactGrid && experimentalMixedOnlySpecialization && transposeReduction
      && (1024 / sharedThreads) % 4 == 0
    let narrowPipelineKeys = [2: sharedPipelineKey | (1 << 23), 4: sharedPipelineKey | (1 << 59)]
    let neededSharedKeys =
      (experimentalSeparateMixedDispatches
        ? [sharedPipelineKey, homogeneousPipelineKey] : [sharedPipelineKey])
      + (narrowMixed ? [narrowPipelineKeys[2]!, narrowPipelineKeys[4]!] : [])
    for pipelineKey in neededSharedKeys
    where useShared && sharedModelDetectorPipelines[pipelineKey] == nil {
      let constants = MTLFunctionConstantValues()
      var pair = pairReduction
      var word32 = [65, 66, 67].contains(streams)
      var threads = UInt32(sharedThreads)
      constants.setConstantValue(&pair, type: .bool, index: 2)
      constants.setConstantValue(&word32, type: .bool, index: 3)
      constants.setConstantValue(&threads, type: .uint, index: 4)
      var packetCount = UInt32(packets)
      constants.setConstantValue(&packetCount, type: .uint, index: 5)
      var pairs: UInt32 = streams >= 69 ? 4 : 1
      constants.setConstantValue(&pairs, type: .uint, index: 6)
      var refill16 = experimentalSharedRefill16
      constants.setConstantValue(&refill16, type: .bool, index: 7)
      var directTable = experimentalDirectDecodingTable
      constants.setConstantValue(&directTable, type: .bool, index: 8)
      var zeroRuns = experimentalZeroRunDecoding
      constants.setConstantValue(&zeroRuns, type: .bool, index: 9)
      var zeroArithmetic = experimentalZeroBitArithmetic
      constants.setConstantValue(&zeroArithmetic, type: .bool, index: 10)
      var mixedTails = (pipelineKey & (1 << 28)) != 0
      constants.setConstantValue(&mixedTails, type: .bool, index: 11)
      var mixedOnly = (pipelineKey & (1 << 29)) != 0
      constants.setConstantValue(&mixedOnly, type: .bool, index: 12)
      var signedPair = experimentalSignedPairReduction
      constants.setConstantValue(&signedPair, type: .bool, index: 13)
      var preparedZeroRuns = experimentalPreparedZeroRuns
      constants.setConstantValue(&preparedZeroRuns, type: .bool, index: 14)
      var pairUnroll = UInt32(experimentalPairLoopUnroll)
      constants.setConstantValue(&pairUnroll, type: .uint, index: 15)
      var staged = experimentalStagedDetectorReduction
      guard !staged || (streams == 32 && sharedThreads == 128 && packets == 1) else {
        throw TANSArchive.invalid("Staged reduction requires exact mode32 and128threads")
      }
      constants.setConstantValue(&staged, type: .bool, index: 16)
      var prefetch = experimentalPrefetchDecoderEntry
      constants.setConstantValue(&prefetch, type: .bool, index: 17)
      var deferredPairs = UInt32(experimentalDeferredReductionPairs)
      constants.setConstantValue(&deferredPairs, type: .uint, index: 18)
      var bitExtract = experimentalDecoderBitExtract
      constants.setConstantValue(&bitExtract, type: .bool, index: 19)
      var packetMajor = experimentalPacketMajorGrid
      constants.setConstantValue(&packetMajor, type: .bool, index: 20)
      var pairLookupBits = UInt32(experimentalPairLookupBits)
      constants.setConstantValue(&pairLookupBits, type: .uint, index: 21)
      var transpose = (pipelineKey & (1 << 60)) != 0
      constants.setConstantValue(&transpose, type: .bool, index: 23)
      var workList = (pipelineKey & (1 << 57)) != 0
      constants.setConstantValue(&workList, type: .bool, index: 32)
      var lanePackets = UInt32(
        (pipelineKey & (1 << 59)) != 0 ? 4 : ((pipelineKey & (1 << 23)) != 0 ? 2 : 1))
      constants.setConstantValue(&lanePackets, type: .uint, index: 33)
      let name =
        packets > 1
        ? "tans_detector_shared_packet_ilp_batch"
        : (streams >= 68 ? "tans_detector_cuda_funnel_batch" : "tans_detector_shared_model_batch")
      let function = try tansLibrary.makeFunction(name: name, constantValues: constants)
      if experimentalCompilerThreadgroupLimit > 0 {
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.maxTotalThreadsPerThreadgroup = experimentalCompilerThreadgroupLimit
        let compiled = try device.makeComputePipelineState(
          descriptor: descriptor, options: [], reflection: nil)
        guard compiled.maxTotalThreadsPerThreadgroup == experimentalCompilerThreadgroupLimit else {
          throw TANSArchive.invalid("Compiler did not honor the exact experiment thread limit")
        }
        sharedModelDetectorPipelines[pipelineKey] = compiled
        print(
          "TANS_COMPILED_LIMIT requested=\(experimentalCompilerThreadgroupLimit) actual=\(compiled.maxTotalThreadsPerThreadgroup) threads=\(sharedThreads) mixed_only=\(mixedOnly)"
        )
      } else {
        sharedModelDetectorPipelines[pipelineKey] = try device.makeComputePipelineState(
          function: function)
      }
    }
    if streams > 0 && !useShared && interleavedDetectorPipelines[streams] == nil {
      let values = MTLFunctionConstantValues()
      var count = UInt32(streams == 8 ? 1 : streams)
      var coalesced = streams == 8
      values.setConstantValue(&count, type: .uint, index: 0)
      values.setConstantValue(&coalesced, type: .bool, index: 1)
      let function = try tansLibrary.makeFunction(
        name: "tans_detector_interleaved_batch", constantValues: values)
      interleavedDetectorPipelines[streams] = try device.makeComputePipelineState(
        function: function)
    }
    // Packet-owner kernel replaces the homogeneous, mixed and sparse dispatches
    // of the plain shared32 transpose decoder; everything else keeps its path.
    let usePacketOwner =
      experimentalPacketOwnerKernel && useShared && streams == 32 && packets == 1
      && transposeReduction && partials == nil && submissionProbeMode == nil
      && !experimentalMetal4Submission && !experimentalPreparedZeroRuns && denseCount > 0
    // The packed decode table is built (once, then cached) only when the
    // packet-owner kernel is dispatched.
    var packedTable: MTLBuffer?
    if usePacketOwner {
      packedTable = try packedOwnerDecodingTable()
      if packetOwnerPipeline == nil {
        guard let function = tansLibrary.makeFunction(name: "tans_detector_packet_owner_batch")
        else { throw TANSArchive.invalid("Missing packet-owner detector kernel") }
        packetOwnerPipeline = try device.makeComputePipelineState(function: function)
      }
    }
    let arguments = detectorBatchFunction.makeArgumentEncoder(bufferIndex: 0)
    // Bound metadata arithmetic before constructing the dispatch table. An
    // invalid source/selection must throw, never trap in optimized flatMap.
    guard records.count % 16 == 0, acquisitionIndices.count == records.count / 16,
      outputAcquisitionIndices.count <= acquisitionIndices.count
    else { throw TANSArchive.invalid("Inconsistent retained entropy record geometry") }
    var recordIndices: [Int] = []
    recordIndices.reserveCapacity(outputAcquisitionIndices.count * 16)
    for acquisition in outputAcquisitionIndices {
      guard let retainedIndex = acquisitionIndices.firstIndex(of: acquisition),
        retainedIndex < records.count / 16
      else { throw TANSArchive.invalid("Requested acquisition has no complete entropy records") }
      for chunk in 0..<16 { recordIndices.append(retainedIndex * 16 + chunk) }
    }
    let tableKey = DetectorTableKey(
      recordIndices: recordIndices, images: images.map(ObjectIdentifier.init))
    // Keys name output buffers by identity, which is unique only while those
    // buffers live. Ring slots live until release; detached atlas outputs are
    // freed after every append, so a later append could reuse their identities
    // and hit a table that encodes freed GPU addresses. Never cache those.
    let cacheable = detachedDetectorOutputs == nil
    let cachedTable = cacheable ? detectorTables[tableKey] : nil
    guard recordIndices.count == outputAcquisitionIndices.count * 16,
      arguments.encodedLength == 40,
      let table = cachedTable?.table
        ?? device.makeBuffer(length: recordIndices.count * 40, options: .storageModeShared),
      let modelOffsets = cachedTable?.offsets
        ?? device.makeBuffer(length: recordIndices.count * 4, options: .storageModeShared),
      let firstCommand = queue.makeCommandBuffer()
    else {
      throw TANSArchive.invalid("Cannot allocate exact detector dispatch table")
    }
    // Fused initialization: the exact base (seed or zero) plus any tile/atlas
    // fields is written by one dispatch at the head of the decode pass (no
    // blit encoder, no separate read-modify-write pass). The blit form stays
    // for the diagnostic and experiment submissions that split the pass.
    let fusedInitialization =
      partials == nil && submissionProbeMode == nil
      && experimentalDetectorQueueCount == 1 && !experimentalMetal4Submission
      && !experimentalPreparedZeroRuns && auditSamples == nil && !experimentalSingleComputePass
      && !(selectedTiles.isEmpty == false && atlasField != nil)
    // Owner-written base: with no tile or atlas field the packet-owner kernel,
    // the only writer of every output scan, stores seed (2) or zero (1) + sums
    // and the initialization pass is skipped. The same uint32 values.
    let ownerBaseMode: UInt32 =
      usePacketOwner && fusedInitialization && selectedTiles.isEmpty && atlasField == nil
      ? (seed ? 2 : 1) : 0
    var ownerBaseTable: MTLBuffer?
    if ownerBaseMode == 2 {
      // Record order is requested-acquisition order, sixteen chunks per image.
      var addresses: [UInt64] = []
      addresses.reserveCapacity(recordIndices.count)
      for index in recordIndices.indices {
        let imageIndex = index / 16
        guard imageIndex < seedImages.count,
          chunks[recordIndices[index]].acquisition == outputAcquisitionIndices[imageIndex]
        else { throw TANSArchive.invalid("Owner base table does not match the output images") }
        addresses.append(seedImages[imageIndex].gpuAddress + UInt64((index % 16) * 16384 * 4))
      }
      ownerBaseTable = addresses.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      }
      guard ownerBaseTable != nil else {
        throw TANSArchive.invalid("Cannot allocate the exact owner base table")
      }
    }
    // Rank-byte addresses per dispatch record.
    var ownerRankTable: (table: MTLBuffer, bytes: MTLBuffer)?
    if usePacketOwner {
      let rankBytes = try ownerRankByteTable()
      let addresses = recordIndices.map {
        rankBytes.buffer.gpuAddress + UInt64(rankBytes.offsets[$0])
      }
      guard
        let table = addresses.withUnsafeBytes({
          device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
        })
      else { throw TANSArchive.invalid("Cannot allocate the owner rank-byte table") }
      ownerRankTable = (table, rankBytes.buffer)
    }
    // Cache slots of the selected sparse columns (the planner's cache map
    // applied here instead of per event on the GPU).
    var ownerSparseSlots: MTLBuffer?
    if usePacketOwner && selectedCount > denseCount {
      let columns = selection.contents().bindMemory(to: UInt32.self, capacity: selectedCount)
      let slots = try (denseCount..<selectedCount).map { index -> UInt32 in
        let column = Int(columns[index])
        guard column < cacheMapValues.count, cacheMapValues[column] >= 0 else {
          throw TANSArchive.invalid("Selected sparse column has no sparse cache slot")
        }
        return UInt32(cacheMapValues[column])
      }
      ownerSparseSlots = slots.withUnsafeBytes {
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      }
      guard ownerSparseSlots != nil else {
        throw TANSArchive.invalid("Cannot allocate the sparse cache-slot list")
      }
    }
    let blit: MTLBlitCommandEncoder?
    if fusedInitialization {
      blit = nil
    } else {
      guard let encoder = firstCommand.makeBlitCommandEncoder() else {
        throw TANSArchive.invalid("Cannot encode exact detector initialization")
      }
      blit = encoder
    }
    var command = firstCommand
    var preparedDecoding: MTLBuffer?
    var remainingMetadataBudget = metadataBudget
    let probeOutput: MTLBuffer?
    if submissionProbeMode != nil {
      let bytes = recordIndices.count * 16
      guard UInt64(bytes) <= remainingMetadataBudget,
        let output = device.makeBuffer(length: bytes, options: .storageModeShared)
      else { throw TANSArchive.invalid("Submission probe exceeds its explicit scratch budget") }
      probeOutput = output
      remainingMetadataBudget -= UInt64(bytes)
      lastDetectorScratchBytes += bytes
      blit!.fill(buffer: output, range: 0..<bytes, value: 0)
    } else {
      probeOutput = nil
    }
    var submittedCommands: [MTLCommandBuffer] = []
    let initialized: MTLEvent?
    if experimentalDetectorQueueCount > 1 {
      guard let event = device.makeEvent() else {
        throw TANSArchive.invalid("Cannot synchronize exact detector shard initialization")
      }
      initialized = event
    } else {
      initialized = nil
    }
    // An encoding/allocation/cancellation error after an early submission must
    // not release the invocation's destination while Metal is still writing it.
    // Once the last command is committed the finish step owns that wait.
    var handedOff = false
    defer {
      if !handedOff { for submitted in submittedCommands { submitted.waitUntilCompleted() } }
      withExtendedLifetime(
        (
          self, firstCommand, images, seedImages, selection, coefficients, table, modelOffsets,
          initialized
        )
      ) {}
    }
    if let blit {
      for (index, image) in images.enumerated() {
        if seed {
          blit.copy(
            from: seedImages[index], sourceOffset: 0,
            to: image, destinationOffset: 0, size: image.length)
        } else {
          blit.fill(buffer: image, range: 0..<image.length, value: 0)
        }
      }
      blit.endEncoding()
    }
    if experimentalPreparedZeroRuns && denseCount > 0 {
      let bytes = globals[0].length
      guard UInt64(bytes) <= remainingMetadataBudget else {
        throw TANSArchive.invalid("Exact prepared decoder table exceeds query budget; no fallback")
      }
      if prepareZeroRunsPipeline == nil {
        guard let function = tansLibrary.makeFunction(name: "tans_prepare_zero_runs") else {
          throw TANSArchive.invalid("Missing exact zero-transition preparation kernel")
        }
        prepareZeroRunsPipeline = try device.makeComputePipelineState(function: function)
      }
      guard let table = device.makeBuffer(length: bytes, options: .storageModePrivate),
        let encoder = command.makeComputeCommandEncoder()
      else {
        throw TANSArchive.invalid("Cannot prepare bounded exact decoder lookup metadata")
      }
      var count = UInt32(bytes / 4)
      encoder.setComputePipelineState(prepareZeroRunsPipeline!)
      encoder.setBuffer(globals[0], offset: 0, index: 0)
      encoder.setBuffer(table, offset: 0, index: 1)
      encoder.setBytes(&count, length: 4, index: 2)
      encoder.dispatchThreads(
        MTLSize(width: Int(count), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.endEncoding()
      preparedDecoding = table
      remainingMetadataBudget -= UInt64(bytes)
      lastDetectorScratchBytes += bytes
    }
    // Open fused-initialization pass; the first decode chunk continues in it.
    var prefixEncoder: MTLComputeCommandEncoder?
    defer { prefixEncoder?.endEncoding() }
    let retainedSlots = outputAcquisitionIndices.map { acquisitionIndices.firstIndex(of: $0)! }
    if fusedInitialization {
      let initDescriptor = MTLComputePassDescriptor()
      initDescriptor.dispatchType = experimentalConcurrentDetectorDispatches ? .concurrent : .serial
      guard let encoder = command.makeComputeCommandEncoder(descriptor: initDescriptor) else {
        throw TANSArchive.invalid("Cannot encode exact detector initialization")
      }
      prefixEncoder = encoder
      if !selectedTiles.isEmpty {
        try exactTileIndex!.encodeAddFromBase(
          encoder: encoder, images: images, bases: seed ? seedImages : nil,
          acquisitions: retainedSlots, selected: selectedTiles)
      } else if let atlasField {
        // The atlas image replaces the base: it is written onto zero.
        guard !seed, let atlas = experimentalAtlas else {
          throw TANSArchive.invalid("An exact atlas base must start from zero-filled outputs")
        }
        try atlas.encodeAddFromBase(
          encoder: encoder, images: images, bases: nil, acquisitions: retainedSlots,
          selected: [(atlasField, 1)], bindSelectedOnly: true)
      } else if ownerBaseMode == 0 {
        try encodeDetectorInitialization(
          encoder: encoder, images: images, bases: seed ? seedImages : nil)
      }
      // Every later dispatch adds exact contributions onto initialized outputs
      // (with an owner-written base the owner stores every scan itself).
      encoder.memoryBarrier(scope: .buffers)
    } else {
      if !selectedTiles.isEmpty {
        try exactTileIndex!.encodeAdd(
          command: command, images: images, acquisitions: retainedSlots,
          selected: selectedTiles)
      }
      if let atlasField {
        // The atlas image replaces the base: it is added onto zero-filled outputs.
        guard !seed, let atlas = experimentalAtlas else {
          throw TANSArchive.invalid("An exact atlas base must start from zero-filled outputs")
        }
        try atlas.encodeAdd(
          command: command, images: images, acquisitions: retainedSlots,
          selected: [(atlasField, 1)], bindSelectedOnly: true)
      }
    }
    let offsets = modelOffsets.contents().assumingMemoryBound(to: UInt32.self)
    if cachedTable == nil {
      if detectorComponentOffsets.count != chunks.count {
        // Resolve the four component slices of every chunk once. Names are
        // archive metadata; the resolved offsets are identical every query.
        detectorComponentOffsets = try chunks.map { chunk in
          try ["dense", "dense_offsets", "sparse", "sparse_offsets"].map { name in
            guard let component = chunk.components.first(where: { $0.name == name }) else {
              throw TANSArchive.invalid("Missing exact batched entropy component")
            }
            return component.offset
          }
        }
      }
      var imageIndexByAcquisition: [Int: Int] = [:]
      for (index, acquisition) in outputAcquisitionIndices.enumerated() {
        imageIndexByAcquisition[acquisition] = index
      }
      for (index, recordIndex) in recordIndices.enumerated() {
        let chunk = chunks[recordIndex]
        arguments.setArgumentBuffer(table, offset: index * 40)
        for binding in 0..<4 {
          arguments.setBuffer(
            records[recordIndex],
            offset: recordBaseOffsets[recordIndex] + detectorComponentOffsets[recordIndex][binding],
            index: binding)
        }
        guard let imageIndex = imageIndexByAcquisition[chunk.acquisition] else {
          throw TANSArchive.invalid("Batched detector record is not in the requested output set")
        }
        arguments.setBuffer(images[imageIndex], offset: (index % 16) * 16384 * 4, index: 4)
        offsets[index] = UInt32((chunk.acquisition * 4 + (index % 16) / 4) * 36864)
      }
      if cacheable {
        detectorTables[tableKey] = (table, modelOffsets)
        detectorTableOrder.append(tableKey)
        while detectorTableOrder.count > Self.detectorTableCacheLimit {
          detectorTables.removeValue(forKey: detectorTableOrder.removeFirst())
        }
      }
    }
    var groupedBuffers: [MTLBuffer] = []
    var maximumModelGroups = 0
    var maximumHomogeneousGroups = 0
    var maximumMixedGroups = 0
    lastDetectorModelGroups = 0
    lastDetectorMixedModelGroups = 0
    lastDetectorPaddedModelLanes = 0
    // Compact grids: per command chunk, the (start, count) range of its work
    // items in the uploaded lists. Items are (chunk-local record << 16 | group).
    var homogeneousWork: MTLBuffer?
    var mixedWork: MTLBuffer?
    var homogeneousWorkRanges: [(start: Int, count: Int)] = []
    var mixedWorkRanges: [(start: Int, count: Int)] = []
    // Narrow mixed lists, keyed by packets per SIMD group (2 or 4).
    var narrowWork: [Int: MTLBuffer] = [:]
    var narrowWorkRanges: [Int: [(start: Int, count: Int)]] = [:]
    var launchStatistics: [String: Double] = [:]
    let commandRecordStride =
      experimentalDetectorRecordsPerCommand > 0
      ? experimentalDetectorRecordsPerCommand : recordIndices.count
    if useShared && denseCount > 0 {
      // Group only immutable metadata on CPU. Scientific counts stay encoded
      // on GPU; no decoding, output computation, or representation duplication.
      let models = globals[1].contents().assumingMemoryBound(to: UInt8.self)
      let input = selection.contents().assumingMemoryBound(to: UInt32.self)
      let signs = coefficients.contents().assumingMemoryBound(to: Int32.self)
      var descriptors: [UInt32] = []
      var groupedSelection: [UInt32] = []
      var groupedSigns: [Int32] = []
      var starts: [UInt32] = []
      var counts: [UInt32] = []
      var homogeneousCounts: [UInt32] = []
      var mixedStarts: [UInt32] = []
      var mixedCounts: [UInt32] = []
      var contexts: [UInt32: (UInt32, UInt32, Int)] = [:]
      // Every distinct model context groups the same selected columns
      // independently, so the groupings are built concurrently and then
      // concatenated in first-appearance order. The concatenation reproduces
      // the serial layout exactly: a context's first group index is the number
      // of groups emitted by the contexts before it, and its selection base is
      // 32 times that. Counts, coefficients and lane assignment are unchanged.
      var distinctContexts: [UInt32] = []
      var seenContexts = Set<UInt32>()
      for index in recordIndices.indices where !seenContexts.contains(offsets[index]) {
        seenContexts.insert(offsets[index])
        distinctContexts.append(offsets[index])
      }
      let groupingInputs = TANSGroupingInputs(
        models: models, input: input, signs: signs, denseCount: denseCount,
        pairReduction: pairReduction, modelKeyCount: pairReduction ? 512 : 256,
        mixedTails: experimentalMixedModelTails,
        savingsDivisor: experimentalMixedTailSavingsDivisor)
      var groupings = [TANSContextGrouping](
        repeating: TANSContextGrouping(), count: distinctContexts.count)
      groupings.withUnsafeMutableBufferPointer { buffer in
        let sendable = UncheckedSendableBuffer(buffer)
        let contexts = distinctContexts
        DispatchQueue.concurrentPerform(iterations: contexts.count) { index in
          sendable.buffer[index] = tansBuildContextGrouping(
            context: contexts[index], inputs: groupingInputs)
        }
      }
      for (ordinal, context) in distinctContexts.enumerated() {
        let grouping = groupings[ordinal]
        let start = UInt32(descriptors.count / 2)
        let selectionBase = UInt32(groupedSelection.count)
        descriptors.reserveCapacity(descriptors.count + grouping.descriptors.count)
        for pair in stride(from: 0, to: grouping.descriptors.count, by: 2) {
          descriptors.append(grouping.descriptors[pair])
          descriptors.append(grouping.descriptors[pair + 1] + selectionBase)
        }
        groupedSelection.append(contentsOf: grouping.selection)
        groupedSigns.append(contentsOf: grouping.signs)
        contexts[context] = (
          start, UInt32(descriptors.count / 2) - start, grouping.tailGroups
        )
      }
      for index in recordIndices.indices {
        let context = offsets[index]
        let range = contexts[context]!
        starts.append(range.0)
        counts.append(range.1)
        maximumModelGroups = max(maximumModelGroups, Int(range.1))
        if experimentalSeparateMixedDispatches {
          let homogeneous = range.1 - UInt32(range.2)
          homogeneousCounts.append(homogeneous)
          mixedStarts.append(range.0 + homogeneous)
          mixedCounts.append(UInt32(range.2))
          maximumHomogeneousGroups = max(maximumHomogeneousGroups, Int(homogeneous))
          maximumMixedGroups = max(maximumMixedGroups, range.2)
        }
        lastDetectorModelGroups += Int(range.1)
        lastDetectorMixedModelGroups += range.2
      }
      lastDetectorPaddedModelLanes = lastDetectorModelGroups * 32 - denseCount * recordIndices.count
      let bytes =
        (descriptors.count + groupedSelection.count + groupedSigns.count + starts.count
          + counts.count + homogeneousCounts.count + mixedStarts.count + mixedCounts.count) * 4
      guard UInt64(bytes) <= remainingMetadataBudget else {
        throw TANSArchive.invalid("Exact model-group metadata exceeds caller budget; no fallback")
      }
      func upload<T>(_ values: [T]) throws -> MTLBuffer {
        try values.withUnsafeBytes { bytes in
          guard
            let buffer = device.makeBuffer(
              bytes: bytes.baseAddress!, length: bytes.count,
              options: .storageModeShared)
          else {
            throw TANSArchive.invalid("Cannot allocate exact model-group metadata")
          }
          return buffer
        }
      }
      groupedBuffers = try [
        upload(descriptors), upload(groupedSelection), upload(groupedSigns),
        upload(starts), upload(counts),
      ]
      if experimentalSeparateMixedDispatches {
        groupedBuffers += try [upload(homogeneousCounts), upload(mixedStarts), upload(mixedCounts)]
      }
      // Grid statistics describe the separated shared-model dispatches; the
      // packet-owner kernel dispatches none of them.
      if experimentalSeparateMixedDispatches && !usePacketOwner {
        let homogeneousTotal = homogeneousCounts.reduce(0) { $0 + Int($1) }
        let mixedTotal = mixedCounts.reduce(0) { $0 + Int($1) }
        launchStatistics = [
          "launch_hom_total": Double(homogeneousTotal),
          "launch_hom_max": Double(maximumHomogeneousGroups),
          "launch_mix_total": Double(mixedTotal),
          "launch_mix_max": Double(maximumMixedGroups),
          "launch_records": Double(recordIndices.count),
        ]
      }
      // The packet-owner kernel walks the grouped descriptors itself and needs
      // no work lists.
      if compactGrid && !usePacketOwner {
        // Packets per SIMD group for one mixed group: 4 or 2 when every active
        // lane sits in the first 8 or 16 slots, is a six-bit tANS column and
        // has coefficient +-1 (the kernel's narrow reduction bound); else 1.
        func lanePackets(record: Int, group: UInt32) -> Int {
          guard narrowMixed else { return 1 }
          let descriptor = Int(mixedStarts[record] + group)
          let base = Int(descriptors[2 * descriptor + 1])
          var span = 0
          for lane in 0..<32 {
            let q = groupedSelection[base + lane]
            if q == UInt32.max { continue }
            let sign = groupedSigns[base + lane]
            if models[Int(offsets[record]) + Int(q)] == 255 || !(sign == 1 || sign == -1) {
              return 1
            }
            span = lane + 1
          }
          return span <= 8 ? 4 : (span <= 16 ? 2 : 1)
        }
        var homogeneousItems: [UInt32] = []
        var mixedItems: [UInt32] = []
        var narrowItems: [Int: [UInt32]] = [2: [], 4: []]
        for chunkStart in stride(from: 0, to: recordIndices.count, by: commandRecordStride) {
          let chunkEnd = min(recordIndices.count, chunkStart + commandRecordStride)
          let homogeneousStart = homogeneousItems.count
          let mixedStart = mixedItems.count
          let narrowStarts = narrowItems.mapValues(\.count)
          for record in chunkStart..<chunkEnd {
            let local = UInt32(record - chunkStart) << 16
            guard homogeneousCounts[record] < 65536, mixedCounts[record] < 65536,
              record - chunkStart < 65536
            else { throw TANSArchive.invalid("Compact detector work item exceeds 16 bits") }
            for group in 0..<homogeneousCounts[record] { homogeneousItems.append(local | group) }
            for group in 0..<mixedCounts[record] {
              let packetsPerSIMD = lanePackets(record: record, group: group)
              if packetsPerSIMD == 1 {
                mixedItems.append(local | group)
              } else {
                narrowItems[packetsPerSIMD]!.append(local | group)
              }
            }
          }
          homogeneousWorkRanges.append(
            (homogeneousStart, homogeneousItems.count - homogeneousStart))
          mixedWorkRanges.append((mixedStart, mixedItems.count - mixedStart))
          for (packetsPerSIMD, start) in narrowStarts {
            narrowWorkRanges[packetsPerSIMD, default: []].append(
              (start, narrowItems[packetsPerSIMD]!.count - start))
          }
        }
        homogeneousWork = try upload(homogeneousItems.isEmpty ? [UInt32(0)] : homogeneousItems)
        mixedWork = try upload(mixedItems.isEmpty ? [UInt32(0)] : mixedItems)
        for (packetsPerSIMD, items) in narrowItems where !items.isEmpty {
          narrowWork[packetsPerSIMD] = try upload(items)
        }
        launchStatistics["launch_mix_p2"] = Double(narrowItems[2]!.count)
        launchStatistics["launch_mix_p4"] = Double(narrowItems[4]!.count)
        lastDetectorScratchBytes +=
          (homogeneousItems.count + mixedItems.count + narrowItems[2]!.count
            + narrowItems[4]!.count) * 4
      }
      lastDetectorScratchBytes += bytes
    }
    if experimentalMetal4Submission && selectedCount > 0 {
      guard #available(macOS 26.0, iOS 26.0, *),
        let context = metal4QueryContext as? TANSMetal4Query,
        streams == 32 && sharedThreads == 128 && packets == 1 && partials == nil,
        experimentalMixedModelTails && experimentalSeparateMixedDispatches
          && experimentalMixedOnlySpecialization && experimentalMixedTailSavingsDivisor == 8,
        experimentalDetectorRecordsPerCommand == 64 && experimentalDetectorQueueCount == 1,
        !experimentalSingleCommandEncoders && !experimentalSparsePrefixCarry,
        !experimentalDirectDecodingTable && !experimentalZeroBitArithmetic,
        !experimentalPreparedZeroRuns && !experimentalZeroRunDecoding
          && !experimentalSignedPairReduction && !experimentalStagedDetectorReduction,
        !experimentalPrefetchDecoderEntry && !experimentalDecoderBitExtract
          && !experimentalPacketMajorGrid && !experimentalSharedRefill16,
        experimentalDeferredReductionPairs == 1 && experimentalPairLoopUnroll == 1
          && experimentalPairLookupBits == 0 && experimentalMemoryAuditURL == nil,
        experimentalFailAfterSubmittedCommands == nil
      else {
        throw TANSArchive.invalid(
          "Metal4 submission requires the frozen grouped64 baseline and prepared context")
      }
      try Task.checkCancellation()
      completion.track(command)
      command.commit()
      submittedCommands.append(command)
      command.waitUntilCompleted()
      guard command.status == .completed else {
        throw TANSArchive.invalid("Exact initialization failed before explicit entropy dispatch")
      }
      let initializationDone = ProcessInfo.processInfo.systemUptime
      let result = try context.run(
        table: table, globals: globals, selection: selection, coefficients: coefficients,
        modelOffsets: modelOffsets, grouped: groupedBuffers, images: images,
        records: recordIndices.count,
        denseCount: denseCount, sparseCount: selectedCount - denseCount,
        retainedColumns: retainedColumns, sparseColumns: sparseColumns,
        homogeneousGroups: maximumHomogeneousGroups, mixedGroups: maximumMixedGroups)
      lastDetectorCommandTiming = [
        "metal4_submission": 1, "submitted_commands": 2,
        "initialization_wall_ms": (initializationDone - encodeStarted) * 1000,
        "metal4_submit_return_ms": (ProcessInfo.processInfo.systemUptime - initializationDone)
          * 1000,
        "gpu_intervals_sum_ms":
          (command.gpuEndTime - command.gpuStartTime + result.end - result.start) * 1000,
        "first_command_to_last_gpu_end_ms": (result.end - command.gpuStartTime) * 1000,
        "encoded_allocations": Double(experimentalEncodedAllocationCount),
        "sparse_prefix_carry": 0, "single_command_encoders": 0,
      ]
      // Explicit Metal4 submission completes inside `run`; nothing is deferred.
      let seconds = result.end - command.gpuStartTime
      let timing = lastDetectorCommandTiming
      return DetectorBatchSubmission(commands: submittedCommands, finish: { (seconds, timing) })
    }
    if selectedCount > 0 {
      if experimentalFailAfterSubmittedCommands == 0 {
        throw TANSArchive.invalid("Injected detector submission failure before commit")
      }
      let denseGroups = (denseCount + 31) / 32
      let sparseCount = selectedCount - denseCount
      if let partials, denseCount > 0 {
        // Apple GPU maxBufferLength is commonly much smaller than the full
        // 66-acquisition partial volume. Reuse one bounded scratch shard and
        // order producer/finish encoders per shard; no atomic accumulation is
        // needed, and every source record still participates exactly once.
        let bytesPerRecord = max(1, denseGroups * 32 * 512 * 4)
        let recordsPerShard = max(1, min(recordIndices.count, partials.length / bytesPerRecord))
        var shardStart = 0
        while shardStart < recordIndices.count {
          let shardCount = min(recordsPerShard, recordIndices.count - shardStart)
          guard let encoder = command.makeComputeCommandEncoder() else {
            throw TANSArchive.invalid("Cannot encode sharded detector partials")
          }
          try declareEncodedReads(
            recordIndices[shardStart..<(shardStart + shardCount)], encoder: encoder)
          encoder.useResources(images, usage: [.read, .write])
          encoder.setComputePipelineState(detectorPartialBatchPipeline)
          encoder.setBuffer(table, offset: shardStart * 40, index: 0)
          for (binding, buffer) in globals.enumerated() {
            encoder.setBuffer(buffer, offset: 0, index: binding + 1)
          }
          encoder.setBuffer(selection, offset: 0, index: 5)
          encoder.setBuffer(coefficients, offset: 0, index: 6)
          var parameters: [UInt32] = [
            retainedColumns, sparseColumns, 0,
            UInt32(denseCount), UInt32(denseGroups),
          ]
          encoder.setBytes(&parameters, length: 20, index: 7)
          encoder.setBuffer(modelOffsets, offset: shardStart * 4, index: 8)
          encoder.setBuffer(partials, offset: 0, index: 9)
          encoder.dispatchThreadgroups(
            MTLSize(width: denseGroups, height: 32, depth: shardCount),
            threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
          if sparseCount > 0 {
            encoder.setComputePipelineState(
              experimentalSparsePrefixCarry ? detectorSparsePrefixPipeline : detectorSparsePipeline)
            encoder.setBuffer(table, offset: shardStart * 40, index: 0)
            encoder.setBuffer(globals[2], offset: 0, index: 1)
            encoder.setBuffer(selection, offset: denseCount * 4, index: 2)
            encoder.setBuffer(coefficients, offset: denseCount * 4, index: 3)
            var sparseParameters: [UInt32] = [UInt32(sparseCount), sparseColumns]
            encoder.setBytes(&sparseParameters, length: 8, index: 4)
            encoder.dispatchThreads(
              MTLSize(width: sparseCount * 32, height: shardCount, depth: 1),
              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          }
          encoder.endEncoding()
          guard let finish = command.makeComputeCommandEncoder() else {
            throw TANSArchive.invalid("Cannot encode sharded detector finish")
          }
          finish.useResources(images, usage: [.read, .write])
          finish.setComputePipelineState(detectorFinishBatchPipeline)
          finish.setBuffer(partials, offset: 0, index: 0)
          finish.setBuffer(table, offset: shardStart * 40, index: 1)
          var groupCount = UInt32(denseGroups)
          finish.setBytes(&groupCount, length: 4, index: 2)
          finish.dispatchThreads(
            MTLSize(width: 16384, height: 1, depth: shardCount),
            threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
          finish.endEncoding()
          shardStart += shardCount
        }
      } else {
        let recordsPerCommand =
          experimentalDetectorRecordsPerCommand > 0
          ? experimentalDetectorRecordsPerCommand : recordIndices.count
        // Initial fill/copy and exact index additions produce every image.
        // All other queues wait for this prefix before touching their disjoint
        // complete acquisition images. Never publish a partially completed set.
        if let initialized { command.encodeSignalEvent(initialized, value: 1) }
        var sharedPass: MTLComputeCommandEncoder?
        // On cancellation/encoding failure, close an uncommitted pass before
        // temporary buffers leave scope. No partial image is published.
        defer { sharedPass?.endEncoding() }
        for recordStart in stride(from: 0, to: recordIndices.count, by: recordsPerCommand) {
          try Task.checkCancellation()
          let recordCount = min(recordsPerCommand, recordIndices.count - recordStart)
          let chunkOrdinal = recordStart / recordsPerCommand
          let encoder: MTLComputeCommandEncoder
          if let pass = sharedPass {
            encoder = pass
          } else if recordStart == 0, let prefix = prefixEncoder {
            // Continue the fused-initialization pass; its barrier orders it.
            encoder = prefix
            prefixEncoder = nil
          } else {
            let computeDescriptor = MTLComputePassDescriptor()
            computeDescriptor.dispatchType =
              experimentalConcurrentDetectorDispatches ? .concurrent : .serial
            if let auditSamples {
              guard auditStageCount * 2 + 1 < auditSamples.sampleCount else {
                throw TANSArchive.invalid("Audit stage sample budget exceeded")
              }
              let attachment = computeDescriptor.sampleBufferAttachments[0]!
              attachment.sampleBuffer = auditSamples
              attachment.startOfEncoderSampleIndex = auditStageCount * 2
              attachment.endOfEncoderSampleIndex = auditStageCount * 2 + 1
              auditStageCount += 1
            }
            guard
              let pass = command.makeComputeCommandEncoder(descriptor: computeDescriptor)
            else {
              throw TANSArchive.invalid("Cannot encode batched entropy detector")
            }
            encoder = pass
            if experimentalSingleComputePass { sharedPass = pass }
          }
          // Uploads completed before publication; these records remain immutable
          // until explicit release. Output/index hazard tracking remains enabled.
          if !experimentalSingleComputePass || recordStart == 0 {
            try declareEncodedReads(
              experimentalSingleComputePass
                ? recordIndices[...]
                : recordIndices[recordStart..<(recordStart + recordCount)], encoder: encoder)
            if experimentalNarrowOutputDeclarations {
              // Dispatch-table order is requested acquisition order, sixteen
              // complete scan chunks per image, including reordered subsets.
              let firstImage = recordStart / 16
              let lastImage = (recordStart + recordCount - 1) / 16
              encoder.useResources(Array(images[firstImage...lastImage]), usage: [.read, .write])
            } else {
              encoder.useResources(images, usage: [.read, .write])
            }
          }
          if let probeMode = submissionProbeMode, let probeOutput {
            // Same source declarations, table, initialization and 17 commands.
            // This reduced grid is a floor diagnostic, not the scientific launch.
            encoder.useResources(
              globals + groupedBuffers + [table, modelOffsets, selection, coefficients],
              usage: .read)
            encoder.setComputePipelineState(submissionProbePipeline!)
            encoder.setBuffer(table, offset: recordStart * 40, index: 0)
            encoder.setBuffer(globals[3], offset: 0, index: 1)
            encoder.setBuffer(selection, offset: 0, index: 2)
            encoder.setBuffer(probeOutput, offset: 0, index: 3)
            var parameters: [UInt32] = [
              retainedColumns, UInt32(denseCount), UInt32(recordCount),
              UInt32(recordStart), probeMode.rawValue,
            ]
            encoder.setBytes(&parameters, length: 20, index: 4)
            encoder.dispatchThreads(
              MTLSize(
                width: probeMode == .bindings ? 1 : max(1, denseCount * 32),
                height: recordCount, depth: 1),
              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          } else if usePacketOwner {
            // One SIMD group per (record, packet) owns its 512 scans: dense model
            // groups, then sparse events, then one store (or add) per scan.
            // Four SIMD groups (packets of one record) per threadgroup, 512
            // per-scan sum words each; must equal tans_owner_simds (tans.metal).
            let ownerSimds = 4
            let ownerThreadgroupWords = ownerSimds * 512
            encoder.setComputePipelineState(packetOwnerPipeline!)
            encoder.setBuffer(table, offset: recordStart * 40, index: 0)
            for (binding, buffer) in globals.enumerated() {
              encoder.setBuffer(buffer, offset: 0, index: binding + 1)
            }
            var ownerQuery: [UInt32] = [
              retainedColumns, sparseColumns, UInt32(sparseCount), ownerBaseMode,
            ]
            encoder.setBytes(&ownerQuery, length: 16, index: 7)
            encoder.setBuffer(modelOffsets, offset: recordStart * 4, index: 8)
            for (index, buffer) in groupedBuffers.prefix(5).enumerated() {
              encoder.setBuffer(
                buffer, offset: index >= 3 ? recordStart * 4 : 0, index: index + 9)
            }
            // With no sparse column the slot and coefficient bindings are
            // declared but never read.
            encoder.setBuffer(ownerSparseSlots ?? selection, offset: 0, index: 14)
            encoder.setBuffer(
              coefficients, offset: sparseCount > 0 ? denseCount * 4 : 0, index: 15)
            encoder.setBuffer(packedTable!, offset: 0, index: 16)
            if let ownerBaseTable {
              encoder.setBuffer(ownerBaseTable, offset: recordStart * 8, index: 17)
              encoder.useResources(seedImages, usage: .read)
            } else {
              // Declared by the kernel; not read unless the base mode is 2.
              encoder.setBuffer(table, offset: 0, index: 17)
            }
            encoder.setBuffer(ownerRankTable!.table, offset: recordStart * 8, index: 18)
            encoder.useResource(ownerRankTable!.bytes, usage: .read)
            encoder.setThreadgroupMemoryLength(ownerThreadgroupWords * 4, index: 0)
            encoder.dispatchThreadgroups(
              MTLSize(width: 32 / ownerSimds, height: recordCount, depth: 1),
              threadsPerThreadgroup: MTLSize(width: ownerSimds * 32, height: 1, depth: 1))
          } else if denseCount > 0 {
            let selectedPipeline =
              useShared
              ? sharedModelDetectorPipelines[sharedPipelineKey]!
              : (streams > 0 ? interleavedDetectorPipelines[streams]! : detectorBatchPipeline)
            let streamsPerLane = streams == 8 ? 1 : max(1, streams)
            let dispatchGroups = (denseCount + 32 * streamsPerLane - 1) / (32 * streamsPerLane)
            encoder.setComputePipelineState(selectedPipeline)
            encoder.setBuffer(table, offset: recordStart * 40, index: 0)
            for (binding, buffer) in globals.enumerated() {
              encoder.setBuffer(buffer, offset: 0, index: binding + 1)
            }
            if let preparedDecoding { encoder.setBuffer(preparedDecoding, offset: 0, index: 1) }
            encoder.setBuffer(selection, offset: 0, index: 5)
            encoder.setBuffer(coefficients, offset: 0, index: 6)
            var parameters: [UInt32] = [
              retainedColumns, sparseColumns, 0, UInt32(denseCount), UInt32(denseGroups),
            ]
            encoder.setBytes(&parameters, length: 20, index: 7)
            encoder.setBuffer(modelOffsets, offset: recordStart * 4, index: 8)
            if useShared {
              if experimentalPairLookupBits > 0 {
                encoder.setBuffer(pairLookup, offset: 0, index: 14)
              }
              for (index, buffer) in groupedBuffers.prefix(5).enumerated() {
                encoder.setBuffer(
                  buffer, offset: index >= 3 ? recordStart * 4 : 0, index: index + 9)
              }
            }
            if useShared {
              if experimentalSeparateMixedDispatches {
                // Full groups execute the original specialization. Only mixed
                // tails pay per-lane model-table lookup; both add exact integer
                // contributions to the same destination in this command.
                if compactGrid {
                  let range = homogeneousWorkRanges[chunkOrdinal]
                  if range.count > 0 {
                    encoder.setComputePipelineState(
                      sharedModelDetectorPipelines[homogeneousPipelineKey]!)
                    encoder.setBuffer(groupedBuffers[5], offset: recordStart * 4, index: 13)
                    encoder.setBuffer(homogeneousWork, offset: range.start * 4, index: 15)
                    encoder.dispatchThreadgroups(
                      MTLSize(width: range.count, height: 1024 / sharedThreads, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: sharedThreads, height: 1, depth: 1))
                  }
                  // (packets per SIMD group, pipeline, work list, range) for the
                  // full mixed list and the narrow 2- and 4-packet lists.
                  var mixedDispatches: [(Int, MTLComputePipelineState, MTLBuffer, Int, Int)] = []
                  let mixedRange = mixedWorkRanges[chunkOrdinal]
                  if mixedRange.count > 0, let mixedWork {
                    mixedDispatches.append(
                      (1, selectedPipeline, mixedWork, mixedRange.start, mixedRange.count))
                  }
                  for packetsPerSIMD in [2, 4] {
                    guard let work = narrowWork[packetsPerSIMD],
                      let ranges = narrowWorkRanges[packetsPerSIMD], chunkOrdinal < ranges.count,
                      ranges[chunkOrdinal].count > 0,
                      let narrowPipeline =
                        sharedModelDetectorPipelines[narrowPipelineKeys[packetsPerSIMD]!]
                    else { continue }
                    mixedDispatches.append(
                      (
                        packetsPerSIMD, narrowPipeline, work, ranges[chunkOrdinal].start,
                        ranges[chunkOrdinal].count
                      ))
                  }
                  for (packetsPerSIMD, pipeline, work, start, count) in mixedDispatches {
                    encoder.setComputePipelineState(pipeline)
                    encoder.setBuffer(groupedBuffers[6], offset: recordStart * 4, index: 12)
                    encoder.setBuffer(groupedBuffers[7], offset: recordStart * 4, index: 13)
                    encoder.setBuffer(work, offset: start * 4, index: 15)
                    encoder.dispatchThreadgroups(
                      MTLSize(
                        width: count, height: 1024 / sharedThreads / packetsPerSIMD, depth: 1),
                      threadsPerThreadgroup: MTLSize(width: sharedThreads, height: 1, depth: 1))
                  }
                } else {
                  if maximumHomogeneousGroups > 0 {
                    encoder.setComputePipelineState(
                      sharedModelDetectorPipelines[homogeneousPipelineKey]!)
                    encoder.setBuffer(groupedBuffers[5], offset: recordStart * 4, index: 13)
                    encoder.dispatchThreadgroups(
                      MTLSize(
                        width: experimentalPacketMajorGrid
                          ? 1024 / sharedThreads : maximumHomogeneousGroups,
                        height: experimentalPacketMajorGrid
                          ? maximumHomogeneousGroups : 1024 / sharedThreads,
                        depth: recordCount),
                      threadsPerThreadgroup: MTLSize(width: sharedThreads, height: 1, depth: 1))
                  }
                  if maximumMixedGroups > 0 {
                    encoder.setComputePipelineState(selectedPipeline)
                    encoder.setBuffer(groupedBuffers[6], offset: recordStart * 4, index: 12)
                    encoder.setBuffer(groupedBuffers[7], offset: recordStart * 4, index: 13)
                    encoder.dispatchThreadgroups(
                      MTLSize(
                        width: experimentalPacketMajorGrid
                          ? 1024 / sharedThreads : maximumMixedGroups,
                        height: experimentalPacketMajorGrid
                          ? maximumMixedGroups : 1024 / sharedThreads,
                        depth: recordCount),
                      threadsPerThreadgroup: MTLSize(width: sharedThreads, height: 1, depth: 1))
                  }
                }
              } else {
                encoder.dispatchThreadgroups(
                  MTLSize(
                    width: experimentalPacketMajorGrid
                      ? 1024 / (sharedThreads * packets) : maximumModelGroups,
                    height: experimentalPacketMajorGrid
                      ? maximumModelGroups : 1024 / (sharedThreads * packets),
                    depth: recordCount),
                  threadsPerThreadgroup: MTLSize(width: sharedThreads, height: 1, depth: 1))
              }
            } else {
              encoder.dispatchThreadgroups(
                MTLSize(width: dispatchGroups, height: 32, depth: recordCount),
                threadsPerThreadgroup: MTLSize(width: 32, height: 1, depth: 1))
            }
          }
          if sparseCount > 0 && submissionProbeMode == nil && !usePacketOwner {
            encoder.setComputePipelineState(
              experimentalSparsePrefixCarry ? detectorSparsePrefixPipeline : detectorSparsePipeline)
            encoder.setBuffer(table, offset: recordStart * 40, index: 0)
            encoder.setBuffer(globals[2], offset: 0, index: 1)
            encoder.setBuffer(selection, offset: denseCount * 4, index: 2)
            encoder.setBuffer(coefficients, offset: denseCount * 4, index: 3)
            var parameters: [UInt32] = [UInt32(sparseCount), sparseColumns]
            encoder.setBytes(&parameters, length: 8, index: 4)
            encoder.dispatchThreads(
              MTLSize(width: sparseCount * 32, height: recordCount, depth: 1),
              threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
          }
          if !experimentalSingleComputePass { encoder.endEncoding() }
          if recordStart + recordCount < recordIndices.count
            && !experimentalSingleCommandEncoders && !experimentalSingleComputePass
          {
            try Task.checkCancellation()
            if auditSamples != nil { auditCommits.append(ProcessInfo.processInfo.systemUptime) }
            completion.track(command)
            command.commit()
            submittedCommands.append(command)
            if experimentalFailAfterSubmittedCommands == submittedCommands.count {
              throw TANSArchive.invalid("Injected detector submission failure after commit")
            }
            let nextQueue: MTLCommandQueue
            if experimentalDetectorQueueCount > 1 {
              let queueIndex = submittedCommands.count % experimentalDetectorQueueCount
              nextQueue = queueIndex == 0 ? queue : experimentalExtraDetectorQueues[queueIndex - 1]
            } else {
              nextQueue = queue
            }
            let nextCommand =
              experimentalInvocationOwnedSubmissions
              ? nextQueue.makeCommandBufferWithUnretainedReferences()
              : nextQueue.makeCommandBuffer()
            guard let next = nextCommand else {
              throw TANSArchive.invalid("Cannot allocate subsequent exact detector submission")
            }
            command = next
            if let initialized { command.encodeWaitForEvent(initialized, value: 1) }
          }
        }
        sharedPass?.endEncoding()
        sharedPass = nil
      }
    }
    // A fused initialization with nothing left to decode still closes its pass.
    if let prefix = prefixEncoder {
      prefix.endEncoding()
      prefixEncoder = nil
    }
    try Task.checkCancellation()
    let commitStarted = ProcessInfo.processInfo.systemUptime
    if auditSamples != nil { auditCommits.append(commitStarted) }
    completion.track(command)
    command.commit()
    submittedCommands.append(command)
    let commitReturned = ProcessInfo.processInfo.systemUptime
    // From here the finish step owns the wait. A pipelined query runs it when
    // its caller finishes the query; diagnostics that sample the GPU or throw
    // a probe result run it now, as the synchronous query always did. Values
    // a later submission overwrites are captured first.
    handedOff = true
    let committed = submittedCommands
    let finalCommand = command
    let modelGroups = lastDetectorModelGroups
    let mixedModelGroups = lastDetectorMixedModelGroups
    let paddedModelLanes = lastDetectorPaddedModelLanes
    let queryScratchBytes = lastDetectorScratchBytes
    // Keep every argument-table target and query temporary alive until the
    // query is finished, not only the buffers the commands bind directly.
    // Explicit, not left to what the closure happens to capture: later
    // commands can hold unretained references (invocation-owned submissions).
    let retainedUntilFinish = (
      firstCommand, table, modelOffsets, selection, coefficients, initialized, ownerBaseTable,
      ownerRankTable?.table, ownerSparseSlots, preparedDecoding, homogeneousWork, mixedWork,
      probeOutput, images, seedImages, groupedBuffers, Array(narrowWork.values), partials
    )
    let finishBatch: () throws -> (seconds: Double, timing: [String: Double]) = { [self] in
      for submitted in committed { submitted.waitUntilCompleted() }
      let waitReturned = ProcessInfo.processInfo.systemUptime
      withExtendedLifetime(retainedUntilFinish) {}
      guard committed.allSatisfy({ $0.status == .completed }) else {
        throw TANSArchive.invalid(
          "Batched exact detector query failed: \(String(describing: committed.first(where: { $0.status != .completed })?.error))"
        )
      }
      let firstGPUStart = committed.map(\.gpuStartTime).min()!
      let lastGPUEnd = committed.map(\.gpuEndTime).max()!
      lastDetectorCommandTiming = [
        "encode_ms": (commitStarted - encodeStarted) * 1000,
        "commit_ms": (commitReturned - commitStarted) * 1000,
        "wait_ms": (waitReturned - commitReturned) * 1000,
        "driver_schedule_ms": (finalCommand.kernelEndTime - finalCommand.kernelStartTime) * 1000,
        "commit_to_gpu_start_ms": (finalCommand.gpuStartTime - commitStarted) * 1000,
        "submitted_commands": Double(committed.count),
        "sparse_prefix_carry": experimentalSparsePrefixCarry ? 1 : 0,
        "single_command_encoders": experimentalSingleCommandEncoders ? 1 : 0,
        "single_compute_pass": experimentalSingleComputePass ? 1 : 0,
        "packet_owner": usePacketOwner ? 1 : 0,
        "submission_queues": Double(experimentalDetectorQueueCount),
        "encoded_allocations": Double(experimentalEncodedAllocationCount),
        "concurrent_dispatches": experimentalConcurrentDetectorDispatches ? 1 : 0,
        "staged_reduction": experimentalStagedDetectorReduction ? 1 : 0,
        "prefetch_decoder_entry": experimentalPrefetchDecoderEntry ? 1 : 0,
        "deferred_reduction_pairs": Double(experimentalDeferredReductionPairs),
        "decoder_bit_extract": experimentalDecoderBitExtract ? 1 : 0,
        "packet_major_grid": experimentalPacketMajorGrid ? 1 : 0,
        "pair_lookup_bits": Double(experimentalPairLookupBits),
        "pair_lookup_bytes": Double(experimentalPairLookupBytes),
        "narrow_output_declarations": experimentalNarrowOutputDeclarations ? 1 : 0,
        "invocation_owned_submissions": experimentalInvocationOwnedSubmissions ? 1 : 0,
        "unretained_commands": Double(committed.filter { !$0.retainedReferences }.count),
        "gpu_intervals_sum_ms": committed.reduce(0) {
          $0 + ($1.gpuEndTime - $1.gpuStartTime) * 1000
        },
        "first_command_to_last_gpu_end_ms": (lastGPUEnd - firstGPUStart) * 1000,
        "model_groups": Double(modelGroups),
        "mixed_model_groups": Double(mixedModelGroups),
        "padded_model_lanes": Double(paddedModelLanes),
        "dense_columns": Double(denseCount),
        "sparse_columns": Double(selectedCount - denseCount),
        "fused_initialization": fusedInitialization ? 1 : 0,
        "compact_grid": compactGrid && !usePacketOwner ? 1 : 0,
      ]
      lastDetectorCommandTiming.merge(launchStatistics) { $1 }
      if let probeMode = submissionProbeMode, let probeOutput {
        submissionProbeWords = Array(
          UnsafeBufferPointer(
            start: probeOutput.contents().assumingMemoryBound(to: UInt32.self),
            count: recordIndices.count * 4))
        lastDetectorCommandTiming["probe_mode"] = Double(probeMode.rawValue)
        lastDetectorCommandTiming["probe_scratch_bytes"] = Double(probeOutput.length)
        lastDetectorCommandTiming["dense_stream_count"] = Double(
          denseCount * 32 * recordIndices.count)
        throw TANSSubmissionProbeCompleted()
      }
      if let auditURL = experimentalMemoryAuditURL, let auditSamples {
        guard let resolved = try auditSamples.resolveCounterRange(0..<(auditStageCount * 2)),
          resolved.count == auditStageCount * 16
        else {
          throw TANSArchive.invalid("Audit timestamp resolution failed")
        }
        let timestamps = tansCounterWords(resolved)
        guard !timestamps.contains(UInt64.max), !timestamps.contains(0),
          auditCommits.count == committed.count
        else {
          throw TANSArchive.invalid("Audit contains invalid timestamps or command identity")
        }
        func words(_ buffer: MTLBuffer, count: Int? = nil) -> [UInt32] {
          Array(
            UnsafeBufferPointer(
              start: buffer.contents().assumingMemoryBound(to: UInt32.self),
              count: count ?? buffer.length / 4))
        }
        var seen = Set<ObjectIdentifier>()
        let uniqueSources = records.filter { seen.insert(ObjectIdentifier($0)).inserted }
        let sourceIDs = Dictionary(
          uniqueKeysWithValues: uniqueSources.enumerated().map {
            (ObjectIdentifier($0.element), $0.offset)
          })
        let plan: [String: Any] = [
          "scope": "Diagnostic: stage timestamps are not cache-hit counters or native FPS",
          "record_count": records.count, "unique_source_allocations": uniqueSources.count,
          "unique_source_bytes": uniqueSources.reduce(0) { $0 + $1.length },
          "record_references_bytes_not_allocation": records.reduce(0) { $0 + $1.length },
          "source_lengths": uniqueSources.map(\.length),
          "record_source_ids": records.map { sourceIDs[ObjectIdentifier($0)]! },
          "record_base_offsets": recordBaseOffsets,
          "global_buffers": globals.map { ["label": $0.label ?? "", "bytes": $0.length] },
          "selected_DP_bytes": output?.length ?? 0,
          "previous_2D_output_bytes": Double(
            detectorImageRings.values.reduce(0) { $0 + $1.reduce(0) { $0 + $1.length } }),
          "new_2D_output_bytes": images.reduce(0) { $0 + $1.length },
          "previous_new_aliases": Double(
            images.filter { b in detectorSeeds.values.contains { $0.image === b } }.count),
          "output_seed_copy": seed, "partials_bytes": partials?.length ?? 0,
          "query_metadata_bytes": queryScratchBytes,
          "allocated_bytes": device.currentAllocatedSize,
          "index_fields": exactTileIndex?.fields.map { f in
            [
              "row": f.tile.row, "col": f.tile.col, "side": f.tile.side,
              "width": Int(f.width), "words": Int(f.words), "payload_bytes": f.payload.length,
              "block_bytes": f.blocks?.length ?? 0,
            ]
          } ?? [],
          "selected_tiles": selectedTiles.map { [$0.0, Int($0.1)] },
          "dense_count": denseCount, "retained_columns": retainedColumns,
          "sparse_columns": sparseColumns,
          "selected": words(selection, count: selectedCount),
          "coefficient_bits": words(coefficients, count: selectedCount),
          "group_buffers": groupedBuffers.map { words($0) },
          "model_groups": modelGroups, "mixed_groups": mixedModelGroups,
          "padded_lanes": paddedModelLanes,
          "pipeline": sharedModelDetectorPipelines[sharedPipelineKey].map {
            [
              "thread_execution_width": $0.threadExecutionWidth,
              "max_threads": $0.maxTotalThreadsPerThreadgroup,
              "static_threadgroup_bytes": $0.staticThreadgroupMemoryLength,
            ]
          } ?? [:],
          "shared_threads": sharedThreads,
          "stage_timestamps_ns": timestamps,
          "commands": committed.enumerated().map { i, c in
            [
              "commit_cpu_s": auditCommits[i], "kernel_start_s": c.kernelStartTime,
              "kernel_end_s": c.kernelEndTime, "gpu_start_s": c.gpuStartTime,
              "gpu_end_s": c.gpuEndTime,
            ]
          },
        ]
        try JSONSerialization.data(withJSONObject: plan, options: [.sortedKeys]).write(to: auditURL)
      }
      return (lastGPUEnd - firstGPUStart, lastDetectorCommandTiming)
    }
    guard auditSamples == nil, submissionProbeMode == nil else {
      let outcome = try finishBatch()
      return DetectorBatchSubmission(commands: committed, finish: { outcome })
    }
    return DetectorBatchSubmission(commands: committed, finish: finishBatch)
  }

  /// Exact output initialization in an open compute pass: each image becomes
  /// its seed (bases) or zero. Later dispatches must be ordered after it.
  private func encodeDetectorInitialization(
    encoder: MTLComputeCommandEncoder, images: [MTLBuffer], bases: [MTLBuffer]?
  ) throws {
    guard bases == nil || bases!.count == images.count,
      images.allSatisfy({ $0.length == 512 * 512 * 4 })
    else { throw TANSArchive.invalid("Exact initialization needs one complete base per image") }
    if detectorInitPipeline == nil {
      guard let function = tansLibrary.makeFunction(name: "tans_detector_init") else {
        throw TANSArchive.invalid("Missing exact detector initialization kernel")
      }
      detectorInitPipeline = try device.makeComputePipelineState(function: function)
    }
    let addresses = images.map(\.gpuAddress) + (bases ?? images).map(\.gpuAddress)
    guard
      let table = addresses.withUnsafeBytes({
        device.makeBuffer(bytes: $0.baseAddress!, length: $0.count, options: .storageModeShared)
      })
    else { throw TANSArchive.invalid("Cannot allocate exact initialization table") }
    encoder.setComputePipelineState(detectorInitPipeline!)
    encoder.useResource(table, usage: .read)
    encoder.useResources(images, usage: [.read, .write])
    if let bases { encoder.useResources(bases, usage: .read) }
    encoder.setBuffer(table, offset: 0, index: 0)
    encoder.setBuffer(table, offset: images.count * 8, index: 1)
    var hasBase: UInt32 = bases == nil ? 0 : 1
    encoder.setBytes(&hasBase, length: 4, index: 2)
    encoder.dispatchThreads(
      MTLSize(width: 65536, height: images.count, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1))
  }

  private func declareEncodedReads(_ indices: ArraySlice<Int>, encoder: MTLComputeCommandEncoder)
    throws
  {
    if experimentalDeclareEncodedHeaps {
      guard !experimentalUseResidentSetForSourceReads, recordHeapIndices.count == records.count,
        !recordHeaps.isEmpty
      else {
        throw TANSArchive.invalid("Heap declarations require live tracked encoded heaps only")
      }
      let selected = Set(indices.map { recordHeapIndices[$0] }).sorted().map { recordHeaps[$0] }
      encoder.useHeaps(selected)
    } else if !experimentalUseResidentSetForSourceReads && !acquisitionStorage.isEmpty {
      var seen = Set<ObjectIdentifier>()
      let selected = indices.compactMap { index -> MTLBuffer? in
        let buffer = records[index]
        return seen.insert(ObjectIdentifier(buffer)).inserted ? buffer : nil
      }
      encoder.useResources(selected, usage: .read)
    } else if !experimentalUseResidentSetForSourceReads {
      encoder.useResources(indices.map { records[$0] }, usage: .read)
    }
  }

  /// Build exact two-symbol codebook metadata within the frozen index budget.
  /// Failed preparation cannot replace the previous metadata.
  func prepareExperimentalPairLookup(bits: Int, maximumMetadataBytes: UInt64) throws {
    guard !isReleased, experimentalResidencyHold == nil,
      maximumMetadataBytes <= 2_108_620_800,
      UInt64(experimentalTileIndexBytes) <= maximumMetadataBytes
    else { throw TANSArchive.invalid("Pair lookup needs a live source and explicit index budget") }
    let prepared = try TANSPairLookup.make(
      device: device, queue: queue, library: tansLibrary, decoding: globals[0], bits: bits,
      maximumBytes: maximumMetadataBytes - UInt64(experimentalTileIndexBytes))
    pairLookup = prepared
    preparedPairLookupBits = bits
  }

  /// Explicit experiment preparation. The original source is never expanded;
  /// each temporary full 2D tile product is packed and independently checked.
  func prepareExperimentalTileIndex(
    maximumIndexBytes: UInt64, blockedPacking: Bool = false,
    layout: TANSExactTileIndex.Layout = .centerFine
  ) throws {
    guard experimentalResidencyHold == nil else {
      throw TANSArchive.invalid("End the experimental residency hold before rebuilding the index")
    }
    guard detectorColumnCost != nil else {
      throw TANSArchive.invalid(
        "Exact tile-index experiment requires authenticated detector planning costs")
    }
    guard !isReleased, maximumIndexBytes <= 8 << 30,
      maximumIndexBytes >= UInt64(experimentalPairLookupBytes),
      UInt64(device.currentAllocatedSize) + maximumIndexBytes + (1 << 30)
        <= device.recommendedMaxWorkingSetSize
    else {
      throw TANSArchive.invalid(
        "Exact tile index requires its explicit <=8GiB budget and safe headroom")
    }
    let priorEnabled = experimentalUseTileIndex
    let priorMode = experimentalDetectorStreamsPerLane
    let priorAtlas = experimentalUseAtlas
    // Index construction issues unseeded tile queries over every acquisition
    // and rotates their output rings. Dropping every exact seed is always
    // exact: the next query per acquisition simply starts from a recompute.
    // It would rewrite images of unfinished pipelined queries, so it waits
    // until every submission has been finished.
    try requireNoUnfinishedDetectorQueries()
    detectorSeeds.removeAll()
    defer {
      detectorSeeds.removeAll()
      experimentalUseTileIndex = priorEnabled
      experimentalDetectorStreamsPerLane = priorMode
      experimentalUseAtlas = priorAtlas
    }
    // Build queries plan without the index; atlas fields stay valid (they are
    // complete exact images) and are used again once the build finishes.
    experimentalUseTileIndex = false
    experimentalUseAtlas = false
    experimentalDetectorStreamsPerLane = 32
    let index = try TANSExactTileIndex(
      device: device, queue: queue, library: tansLibrary,
      blockedPacking: blockedPacking)
    for (ordinal, tile) in TANSExactTileIndex.tiles(for: layout).enumerated() {
      try Task.checkCancellation()
      try autoreleasepool {
        var mask = [UInt8](repeating: 0, count: 36864)
        for q in tile.pixels { mask[q] = validDetectorMask[q] }
        let images = try detectorImages(mask: mask, maximumAdditionalBytes: 1 << 30, rebase: true)
        try index.append(
          tile: tile, images: images,
          maximumBytes: maximumIndexBytes - UInt64(experimentalPairLookupBytes))
      }
      if ordinal % 10 == 9 {
        print(
          "TILE_INDEX_BUILD fields=\(ordinal+1) bytes=\(index.residentBytes) roundtrip_exact=true")
        fflush(stdout)
      }
    }
    exactTileIndex = index
  }

  /// Start an empty exact atlas with its own explicit byte budget. Fields are
  /// added one at a time by `appendExperimentalAtlasField(mask:)`, so a caller
  /// can build them between interactive queries. Any prior atlas is dropped.
  func beginExperimentalAtlas(maximumBytes: UInt64) throws {
    guard experimentalResidencyHold == nil else {
      throw TANSArchive.invalid("End the experimental residency hold before building the atlas")
    }
    // In-flight queries may read the atlas being dropped.
    drainDetectorQueries()
    experimentalAtlas = nil
    experimentalAtlasMasks = []
    experimentalAtlasCentroids = []
    guard !isReleased, exactTileIndex != nil, useBatchedDetector,
      UInt64(device.currentAllocatedSize) + maximumBytes + (1 << 30)
        <= device.recommendedMaxWorkingSetSize
    else {
      throw TANSArchive.invalid(
        "Exact atlas requires the prepared tile index, an explicit budget and safe headroom")
    }
    experimentalAtlas = try TANSExactTileIndex(
      device: device, queue: queue, library: tansLibrary, blockedPacking: true)
    experimentalAtlasBudget = maximumBytes
  }

  /// Add the exact complete image of `mask` for every retained acquisition.
  /// It comes from an unseeded tile-index query written to temporary outputs,
  /// so no displayed image, seed or ring cursor changes, and the atlas is off
  /// while it runs, so no field is derived from another field. The packed
  /// field is audited bit for bit against that query's output.
  func appendExperimentalAtlasField(mask: [UInt8]) throws {
    guard let atlas = experimentalAtlas, !isReleased, experimentalResidencyHold == nil,
      mask.count == 36864, mask.allSatisfy({ $0 <= 1 })
    else {
      throw TANSArchive.invalid("Begin the exact atlas and provide a binary 192x192 mask")
    }
    let priorUse = experimentalUseAtlas
    let priorIndex = experimentalUseTileIndex
    let priorMode = experimentalDetectorStreamsPerLane
    defer {
      detachedDetectorOutputs = nil
      experimentalUseAtlas = priorUse
      experimentalUseTileIndex = priorIndex
      experimentalDetectorStreamsPerLane = priorMode
    }
    experimentalUseAtlas = false
    experimentalUseTileIndex = true
    experimentalDetectorStreamsPerLane = 32
    try autoreleasepool {
      detachedDetectorOutputs = try acquisitionIndices.map { _ in
        guard let image = device.makeBuffer(length: 512 * 512 * 4, options: .storageModeShared)
        else {
          throw TANSArchive.invalid("Cannot allocate temporary exact atlas images")
        }
        return image
      }
      let images = try detectorImages(mask: mask, maximumAdditionalBytes: 2 << 30, rebase: true)
      // An empty tile keeps atlas fields out of any rectangle planning.
      try atlas.append(
        tile: TANSExactTileIndex.Tile(row: 0, col: 0, side: 0), images: images,
        maximumBytes: experimentalAtlasBudget)
    }
    experimentalAtlasMasks.append(mask)
    experimentalAtlasCentroids.append(Self.maskCentroid(mask))
  }

  /// Build a complete atlas in one call (benchmarks).
  func prepareExperimentalAtlas(masks: [[UInt8]], maximumBytes: UInt64) throws {
    try beginExperimentalAtlas(maximumBytes: maximumBytes)
    for (ordinal, mask) in masks.enumerated() {
      try Task.checkCancellation()
      try appendExperimentalAtlasField(mask: mask)
      if ordinal % 10 == 9 {
        print(
          "ATLAS_BUILD fields=\(ordinal+1) bytes=\(experimentalAtlasBytes) roundtrip_exact=true")
        fflush(stdout)
      }
    }
  }

  /// On-disk form of a prepared exact tile index. Only the packed 2D tile sums
  /// are stored; the encoded source is never duplicated. The cache is keyed by
  /// the archive manifest digest, the retained acquisitions and the layout, and
  /// every field is digested so a corrupt or foreign file cannot be restored.
  struct TileIndexCacheManifest: Codable, Equatable {
    struct Field: Codable, Equatable {
      let row: Int
      let col: Int
      let side: Int
      let width: UInt32
      let words: UInt32
      let offset: Int
      let bytes: Int
      let sha256: String
    }
    var format = "quantem-exact-tile-index-cache-v1"
    let checkpointSHA256: String
    let acquisitions: [Int]
    let layout: String
    let imageCount: Int
    let fields: [Field]
  }
  static let tileIndexCacheManifestName = "exact-tile-index.json"
  static let tileIndexCachePayloadName = "exact-tile-index.bin"

  /// Write the prepared exact tile index into `directory` (created if needed).
  /// The payload is written first and the manifest last, both through
  /// temporary files, so a partial write is never a valid cache.
  func exportExperimentalTileIndex(
    to directory: URL, layout: TANSExactTileIndex.Layout = .centerFine
  )
    throws
  {
    guard !isReleased, let index = exactTileIndex, !index.blockedPacking else {
      throw TANSArchive.invalid("Export requires a prepared, unblocked exact tile index")
    }
    let tiles = TANSExactTileIndex.tiles(for: layout)
    guard index.fields.count == tiles.count,
      zip(index.fields, tiles).allSatisfy({
        $0.0.tile.row == $0.1.row && $0.0.tile.col == $0.1.col && $0.0.tile.side == $0.1.side
      })
    else { throw TANSArchive.invalid("Prepared index does not match the declared layout") }
    try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    let payloadURL = directory.appendingPathComponent(Self.tileIndexCachePayloadName)
    let manifestURL = directory.appendingPathComponent(Self.tileIndexCacheManifestName)
    let temporaryPayload = payloadURL.appendingPathExtension("partial")
    FileManager.default.createFile(atPath: temporaryPayload.path, contents: nil)
    let handle = try FileHandle(forWritingTo: temporaryPayload)
    var entries: [TileIndexCacheManifest.Field] = []
    var offset = 0
    do {
      for (ordinal, field) in index.fields.enumerated() {
        let bytes = try index.exportField(ordinal)
        try handle.write(contentsOf: bytes)
        entries.append(
          TileIndexCacheManifest.Field(
            row: field.tile.row, col: field.tile.col, side: field.tile.side,
            width: field.width, words: field.words, offset: offset, bytes: bytes.count,
            sha256: SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()))
        offset += bytes.count
      }
      try handle.synchronize()
      try handle.close()
    } catch {
      try? handle.close()
      try? FileManager.default.removeItem(at: temporaryPayload)
      throw error
    }
    let manifest = TileIndexCacheManifest(
      checkpointSHA256: archiveCheckpointSHA256, acquisitions: acquisitionIndices,
      layout: layout.rawValue, imageCount: acquisitionIndices.count, fields: entries)
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    let manifestData = try encoder.encode(manifest)
    _ = try? FileManager.default.removeItem(at: payloadURL)
    try FileManager.default.moveItem(at: temporaryPayload, to: payloadURL)
    let temporaryManifest = manifestURL.appendingPathExtension("partial")
    try manifestData.write(to: temporaryManifest, options: .atomic)
    _ = try? FileManager.default.removeItem(at: manifestURL)
    try FileManager.default.moveItem(at: temporaryManifest, to: manifestURL)
  }

  /// Restore a previously exported exact tile index for this exact archive and
  /// acquisition selection. Every field digest is verified before any GPU
  /// allocation is retained; any mismatch throws and leaves the series without
  /// an index, so the caller falls back to building one.
  func importExperimentalTileIndex(
    from directory: URL, maximumIndexBytes: UInt64,
    layout: TANSExactTileIndex.Layout = .centerFine
  ) throws {
    guard experimentalResidencyHold == nil else {
      throw TANSArchive.invalid("End the experimental residency hold before restoring an index")
    }
    guard !isReleased, maximumIndexBytes <= 8 << 30,
      UInt64(device.currentAllocatedSize) + maximumIndexBytes + (1 << 30)
        <= device.recommendedMaxWorkingSetSize
    else {
      throw TANSArchive.invalid(
        "Exact tile index requires its explicit <=8GiB budget and safe headroom")
    }
    let decoder = JSONDecoder()
    let manifest = try decoder.decode(
      TileIndexCacheManifest.self,
      from: Data(contentsOf: directory.appendingPathComponent(Self.tileIndexCacheManifestName)))
    let tiles = TANSExactTileIndex.tiles(for: layout)
    guard manifest.format == "quantem-exact-tile-index-cache-v1",
      manifest.checkpointSHA256 == archiveCheckpointSHA256,
      manifest.acquisitions == acquisitionIndices,
      manifest.layout == layout.rawValue,
      manifest.imageCount == acquisitionIndices.count,
      manifest.fields.count == tiles.count,
      zip(manifest.fields, tiles).allSatisfy({
        $0.0.row == $0.1.row && $0.0.col == $0.1.col && $0.0.side == $0.1.side
      })
    else {
      throw TANSArchive.invalid(
        "Exact tile index cache does not describe this archive, selection and layout")
    }
    let payload = try Data(
      contentsOf: directory.appendingPathComponent(Self.tileIndexCachePayloadName),
      options: .alwaysMapped)
    let total = manifest.fields.reduce(0) { $0 + $1.bytes }
    guard payload.count == total, UInt64(total) <= maximumIndexBytes else {
      throw TANSArchive.invalid("Exact tile index cache payload size disagrees with its manifest")
    }
    let index = try TANSExactTileIndex(device: device, queue: queue, library: tansLibrary)
    for (entry, tile) in zip(manifest.fields, tiles) {
      guard entry.offset >= 0, entry.bytes > 0, entry.offset + entry.bytes <= payload.count else {
        throw TANSArchive.invalid("Exact tile index cache field range is invalid")
      }
      let bytes = payload.subdata(in: entry.offset..<(entry.offset + entry.bytes))
      guard SHA256.hash(data: bytes).map({ String(format: "%02x", $0) }).joined() == entry.sha256
      else {
        throw TANSArchive.invalid("Exact tile index cache field digest mismatch")
      }
      try index.restore(
        tile: tile, width: entry.width, words: entry.words, payload: bytes,
        imageCount: acquisitionIndices.count, maximumBytes: maximumIndexBytes)
    }
    drainDetectorQueries()
    detectorSeeds.removeAll()
    exactTileIndex = index
  }

  /// Isolated architecture trials release one index before preparing another.
  /// Never release source counts or a caller's prior scientific publication.
  func discardExperimentalTileIndex() throws {
    guard experimentalResidencyHold == nil else {
      throw TANSArchive.invalid("End residency hold before discarding the tile index")
    }
    exactTileIndex = nil
    experimentalUseTileIndex = false
  }

  /// Caller serializes queries and release. Pipelined queries still on the
  /// GPU read these allocations: release waits for them, and finishing one
  /// afterwards reports that the series was released.
  func releaseResidentStorage() {
    for query in inFlightDetectorQueries {
      for command in query.commands { command.waitUntilCompleted() }
    }
    inFlightDetectorQueries.removeAll()
    metal4QueryContext = nil
    endExperimentalResidencyHold()
    experimentalExtraDetectorQueues.removeAll()
    records.removeAll()
    recordBaseOffsets.removeAll()
    ownerRankBytes = nil
    packedOwnerDecoding = nil
    acquisitionStorage.removeAll()
    recordHeaps.removeAll()
    recordHeapIndices.removeAll()
    experimentalRecordHeapBytes = 0
    globals.removeAll()
    pairLookup = nil
    preparedPairLookupBits = 0
    experimentalPairLookupBits = 0
    exactTileIndex = nil
    output = nil
    detectorSeeds.removeAll()
    detectorImageRings.removeAll()
    detectorRingCursors.removeAll()
    detectorTables.removeAll()
    detectorTableOrder.removeAll()
    experimentalAtlas = nil
    experimentalAtlasMasks = []
    experimentalAtlasCentroids = []
    isReleased = true
  }

  /// Explicit diagnostic only: retain the existing encoded/index allocations
  /// across idle gaps, without repacking or creating another scientific buffer.
  func beginExperimentalResidencyHold(attachToQueue: Bool = true) throws {
    guard !isReleased, experimentalResidencyHold == nil, exactTileIndex != nil
    else {
      throw TANSArchive.invalid("Prepare one live exact index before requesting its residency")
    }
    guard #available(macOS 15.0, *) else {
      throw TANSArchive.invalid("Explicit residency sets require macOS15 or newer")
    }
    // Coalesced records alias their acquisition allocation sixteen times.
    // Admit and hold the actual allocations once, not the record views.
    let encodedAllocations = acquisitionStorage.isEmpty ? records : acquisitionStorage
    let buffers =
      encodedAllocations + globals + exactTileIndex!.fields.map(\.payload)
      + exactTileIndex!.fields.compactMap(\.blocks)
      + (experimentalAtlas?.fields.map(\.payload) ?? [])
      + (experimentalAtlas?.fields.compactMap(\.blocks) ?? [])
    let bytes = buffers.reduce(UInt64(0)) { $0 + UInt64($1.allocatedSize) }
    guard bytes <= device.recommendedMaxWorkingSetSize,
      UInt64(device.currentAllocatedSize) <= ProcessInfo.processInfo.physicalMemory * 4 / 5
    else {
      throw TANSArchive.invalid("Existing exact allocations exceed the explicit residency budget")
    }
    let descriptor = MTLResidencySetDescriptor()
    descriptor.label = "Experimental exact entropy source and tile index hold"
    descriptor.initialCapacity = buffers.count
    let residency = try device.makeResidencySet(descriptor: descriptor)
    for buffer in buffers { residency.addAllocation(buffer) }
    residency.commit()
    if attachToQueue { queue.addResidencySet(residency) }
    residency.requestResidency()
    experimentalResidencySetBytes = residency.allocatedSize
    experimentalResidencyAllocationCount = residency.allocationCount
    experimentalResidencyAttachedToQueue = attachToQueue
    experimentalResidencyHold = residency
  }

  /// Diagnostic: keep only the encoded records of a few acquisitions (plus
  /// globals and the exact index) resident across idle gaps. Unlike the
  /// all-series hold, this covers about one to two GiB per acquisition.
  func beginExperimentalAcquisitionResidencyHold(acquisitions: [Int]) throws {
    guard !isReleased, experimentalResidencyHold == nil, acquisitionStorage.isEmpty else {
      throw TANSArchive.invalid(
        "Acquisition residency hold requires record-backed encoded storage without another hold")
    }
    guard #available(macOS 15.0, *) else {
      throw TANSArchive.invalid("Explicit residency sets require macOS15 or newer")
    }
    let retained = try retainedIndices(for: acquisitions)
    var buffers: [MTLBuffer] = globals
    for index in retained { buffers += records[(index * 16)..<(index * 16 + 16)] }
    if let index = exactTileIndex {
      buffers += index.fields.map(\.payload) + index.fields.compactMap(\.blocks)
    }
    for index in retained { buffers += detectorImageRings[index] ?? [] }
    let descriptor = MTLResidencySetDescriptor()
    descriptor.label = "Experimental per-acquisition entropy residency hold"
    descriptor.initialCapacity = buffers.count
    let residency = try device.makeResidencySet(descriptor: descriptor)
    for buffer in buffers { residency.addAllocation(buffer) }
    residency.commit()
    queue.addResidencySet(residency)
    residency.requestResidency()
    experimentalResidencySetBytes = residency.allocatedSize
    experimentalResidencyAllocationCount = residency.allocationCount
    experimentalResidencyAttachedToQueue = true
    experimentalResidencyHold = residency
  }

  func endExperimentalResidencyHold() {
    if #available(macOS 15.0, *), let residency = experimentalResidencyHold as? MTLResidencySet {
      if experimentalResidencyAttachedToQueue { queue.removeResidencySet(residency) }
      residency.endResidency()
      residency.removeAllAllocations()
      residency.commit()
    }
    experimentalResidencyHold = nil
    experimentalResidencySetBytes = 0
    experimentalResidencyAllocationCount = 0
    experimentalResidencyAttachedToQueue = false
  }

  /// Expensive independent-parity diagnostic, not a rendering or resident path.
  /// Streams one36MiB decoded packet through SHA256, then reuses the same scratch.
  /// No full decoded acquisition is retained and audit time is not load timing.
  func auditFullCountSHA256(acquisitionIndex: Int) throws -> String {
    guard !isReleased, let selection = acquisitionIndices.firstIndex(of: acquisitionIndex) else {
      throw TANSArchive.invalid("Select a retained acquisition before auditing")
    }
    guard let scratch = device.makeBuffer(length: 512 * 36864 * 2, options: .storageModeShared)
    else {
      throw TANSArchive.invalid("Cannot allocate bounded36MiB parity scratch")
    }
    scratch.label = "Bounded tANS full-count audit scratch, not resident source"
    var digest = SHA256()
    for packet in 0..<512 {
      let localChunk = packet / 32
      let recordIndex = selection * 16 + localChunk
      let chunk = chunks[recordIndex]
      guard let command = queue.makeCommandBuffer(),
        let encoder = command.makeComputeCommandEncoder()
      else {
        throw TANSArchive.invalid("Cannot encode bounded tANS audit")
      }
      encoder.setComputePipelineState(auditPipeline)
      for (binding, name) in ["dense", "dense_offsets", "sparse", "sparse_offsets"].enumerated() {
        guard let component = chunk.components.first(where: { $0.name == name }),
          component.dtype == "<u4"
        else {
          encoder.endEncoding()
          throw TANSArchive.invalid("Missing authenticated entropy component \(name)")
        }
        encoder.setBuffer(
          records[recordIndex], offset: recordBaseOffsets[recordIndex] + component.offset,
          index: binding)
      }
      for (binding, buffer) in globals.enumerated() {
        encoder.setBuffer(buffer, offset: 0, index: binding + 4)
      }
      encoder.setBuffer(scratch, offset: 0, index: 8)
      var parameters: [UInt32] = [
        UInt32((packet % 32) * 512), retainedColumns, sparseColumns,
        UInt32((acquisitionIndex * 4 + localChunk / 4) * 36864),
      ]
      encoder.setBytes(&parameters, length: 16, index: 9)
      encoder.dispatchThreads(
        MTLSize(width: 36864, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed else {
        throw TANSArchive.invalid("Bounded tANS audit failed")
      }
      digest.update(
        bufferPointer: UnsafeRawBufferPointer(start: scratch.contents(), count: scratch.length))
    }
    return digest.finalize().map { String(format: "%02x", $0) }.joined()
  }
}

/// Resolved 64-bit counter samples, copied without a closure. The throwing
/// finish step used two identical `withUnsafeBytes` closures here, the shape
/// the release optimizer merged into an error-register-clobbering function
/// in `runDetectorGroup` (Swift 6.3); one plain copy leaves nothing to merge.
func tansCounterWords(_ data: Data) -> [UInt64] {
  var words = [UInt64](repeating: 0, count: data.count / MemoryLayout<UInt64>.stride)
  (data as NSData).getBytes(&words, length: words.count * MemoryLayout<UInt64>.stride)
  return words
}

/// Completion of every command one detector query committed, and of the
/// earlier queries it depends on. A handler is attached to each command
/// before its commit, the only time Metal accepts one, so callers are
/// notified without blocking a thread in a wait.
final class TANSDetectorCompletion: @unchecked Sendable {
  private let lock = NSLock()
  private var outstanding = 0
  private var sealed = false
  private var handlers: [@Sendable () -> Void] = []

  /// Call before `command.commit()`.
  func track(_ command: MTLCommandBuffer) {
    lock.lock()
    outstanding += 1
    lock.unlock()
    command.addCompletedHandler { [self] _ in commandCompleted() }
  }

  /// Also wait for `earlier` (another query's completion). Call before `seal()`.
  func track(_ earlier: TANSDetectorCompletion) {
    lock.lock()
    outstanding += 1
    lock.unlock()
    earlier.notify { [self] in commandCompleted() }
  }

  /// No further commands will be tracked.
  func seal() {
    lock.lock()
    sealed = true
    let ready = takeReadyHandlers()
    lock.unlock()
    for handler in ready { handler() }
  }

  var isComplete: Bool {
    lock.lock()
    defer { lock.unlock() }
    return sealed && outstanding == 0
  }

  /// Runs `handler` once, when the query is sealed and every tracked command
  /// has completed (successfully or not); at once if that is already so.
  func notify(_ handler: @escaping @Sendable () -> Void) {
    lock.lock()
    if sealed && outstanding == 0 {
      lock.unlock()
      handler()
      return
    }
    handlers.append(handler)
    lock.unlock()
  }

  private func commandCompleted() {
    lock.lock()
    outstanding -= 1
    let ready = takeReadyHandlers()
    lock.unlock()
    for handler in ready { handler() }
  }

  /// Caller holds the lock.
  private func takeReadyHandlers() -> [@Sendable () -> Void] {
    guard sealed && outstanding == 0 else { return [] }
    defer { handlers.removeAll() }
    return handlers
  }
}

/// Immutable grouping inputs shared by every model context of one query. The
/// pointers address query-local shared buffers that no iteration writes, and
/// each iteration returns its own value, so concurrent reads are safe.
private struct TANSGroupingInputs: @unchecked Sendable {
  let models: UnsafeMutablePointer<UInt8>
  let input: UnsafeMutablePointer<UInt32>
  let signs: UnsafeMutablePointer<Int32>
  let denseCount: Int
  let pairReduction: Bool
  let modelKeyCount: Int
  let mixedTails: Bool
  let savingsDivisor: Int
}

/// The exact 32-lane model grouping of one context: descriptors, the selected
/// detector columns per lane, their signed coefficients, and how many of the
/// emitted groups are mixed-model tails. Selection indices are context-local;
/// the caller shifts them when concatenating contexts in order.
private struct TANSContextGrouping {
  var descriptors: [UInt32] = []
  var selection: [UInt32] = []
  var signs: [Int32] = []
  var tailGroups = 0
}

/// Pure function of immutable metadata: no counts are decoded or approximated,
/// and the result depends only on `context` and `inputs`.
private func tansBuildContextGrouping(
  context: UInt32, inputs: TANSGroupingInputs
) -> TANSContextGrouping {
  var local = TANSContextGrouping()
  var descriptors: [UInt32] = []
  var groupedSelection: [UInt32] = []
  var groupedSigns: [Int32] = []
  var byModel: [[Int]] = Array(repeating: [], count: inputs.modelKeyCount)
  for i in 0..<inputs.denseCount {
    let model = Int(inputs.models[Int(context) + Int(inputs.input[i])])
    let key = inputs.pairReduction ? model * 2 + (inputs.signs[i] < 0 ? 1 : 0) : model
    byModel[key].append(i)
  }
  var tails: [Int] = []
  // Avoid changing already-efficient homogeneous work. This bound is
  // based only on immutable grouping metadata, never count values or
  // a scientist-selected product/geometry name.
  let originalGroups = byModel.reduce(0) { $0 + ($1.count + 31) / 32 }
  let fullGroups = byModel.reduce(0) { $0 + $1.count / 32 }
  let tailColumns = byModel.reduce(0) { $0 + $1.count % 32 }
  let compactGroups = fullGroups + (tailColumns + 31) / 32
  let useMixedTails =
    inputs.mixedTails
    && (inputs.savingsDivisor == 0
      || (originalGroups - compactGroups) * inputs.savingsDivisor
        >= originalGroups)
  for key in byModel.indices {
    let model = inputs.pairReduction ? key / 2 : key
    let members = byModel[key]
    let homogeneousCount =
      useMixedTails ? members.count / 32 * 32 : members.count
    for begin in stride(from: 0, to: homogeneousCount, by: 32) {
      descriptors += [UInt32(model), UInt32(groupedSelection.count)]
      for lane in 0..<32 {
        if begin + lane < members.count {
          let i = members[begin + lane]
          groupedSelection.append(inputs.input[i])
          groupedSigns.append(inputs.signs[i])
        } else {
          groupedSelection.append(UInt32.max)
          groupedSigns.append(0)
        }
      }
    }
    if useMixedTails {
      tails.append(contentsOf: members[homogeneousCount...])
    }
  }
  // Descriptor256 is explicit mixed metadata, never a decoder-model ID.
  for begin in stride(from: 0, to: tails.count, by: 32) {
    descriptors += [256, UInt32(groupedSelection.count)]
    for lane in 0..<32 {
      if begin + lane < tails.count {
        let i = tails[begin + lane]
        groupedSelection.append(inputs.input[i])
        groupedSigns.append(inputs.signs[i])
      } else {
        groupedSelection.append(UInt32.max)
        groupedSigns.append(0)
      }
    }
  }
  local.descriptors = descriptors
  local.selection = groupedSelection
  local.signs = groupedSigns
  local.tailGroups = (tails.count + 31) / 32
  return local
}
