import CoreFoundation
import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMKernels

// Alternate kernel policies exist only in instrumented benchmark builds.
// Ordinary applications use the qualified policy without shell configuration.
private func compactKernelOption(_ name: String, byDefault defaultValue: Bool) -> Bool {
  #if QGPU_PACKING_DIAGNOSTICS
    switch ProcessInfo.processInfo.environment["COMPACT_" + name] {
    case "1": return true
    case "0": return false
    default: return defaultValue
    }
  #else
    return defaultValue
  #endif
}

/// Source-bound detector geometry carried by the compact HDF5 manifest.
public struct MetalCompactH5DetectorCalibration: Equatable, Sendable {
  public let detectorCenterRow: Double
  public let detectorCenterColumn: Double
  public let brightFieldRadius: Double
  public let dpcRotationDegrees: Double?
  public let dpcComponentOrderExchanged: Bool?
  public let method: String
}

/// Source-bound exact detector moments stored in one contiguous HDF5 range.
public struct MetalCompactH5PreparedDPCMoments: Equatable, Sendable {
  public let fileOffset: UInt64
  public let fileBytes: UInt64
  public let sha256: String
  /// Nil for in-memory original-HDF5 residents, which do not hash a dense tensor.
  public let workingLogicalSHA256: String?
  public let workingDtype: String
  public let detectorMaskSHA256: String
  public let scanCount: Int
  public let selectedDetectorPixels: Int
  public let detectorColumns: Int
  public let totalBound: UInt64
  public let rowMomentBound: UInt64
  public let columnMomentBound: UInt64
  public let narrowInteger: Bool
  public let narrowProducts: Bool

  /// Backward-compatible name for QGIX-v3 uint8 moment receipts.
  @available(*, deprecated, message: "Use workingLogicalSHA256.")
  public var workingUInt8SHA256: String? { workingLogicalSHA256 }
}

/// One exact source-bound canonical virtual-detector product.
public struct MetalCompactH5PreparedDetectorProduct: Equatable, Sendable {
  public let name: String
  public let centerRow: Double
  public let centerColumn: Double
  public let innerRadius: Double
  public let outerRadius: Double
  public let selectedDetectorPixels: Int
  public let maskFileOffset: UInt64
  public let maskFileBytes: UInt64
  public let maskSHA256: String
  public let valuesFileOffset: UInt64
  public let valuesFileBytes: UInt64
  public let valuesSHA256: String
}

/// Authenticated BF, ABF, and ADF maps embedded in one compact source.
public struct MetalCompactH5PreparedDetectorProducts: Equatable, Sendable {
  public let calibrationSHA256: String
  public let workingUInt8SHA256: String
  public let detectorMaskSHA256: String
  public let products: [MetalCompactH5PreparedDetectorProduct]
}

/// Canonical prepared detector products available to the resident source.
public enum MetalCompactH5PreparedDetectorProductName: String, Sendable {
  case bf
  case abf
  case adf
}

/// One centered DPC component prepared for direct display.
public enum MetalCompactH5DPCComponent: Sendable {
  case row
  case column
}

/// Stable metadata for one QuantEM compact 4D-STEM HDF5 source.
public struct MetalCompactH5Metadata: Equatable, Sendable {
  public let sourceURL: URL
  public let sourceBytes: UInt64
  public let schema: String
  public let payloadCodec: String
  public let sourceDtype: String?
  public let manifestSHA256: String
  public let workingDtype: String
  public let embeddedScientificSemantics: Bool
  public let scanRows: Int
  public let scanColumns: Int
  public let detectorRows: Int
  public let detectorColumns: Int
  public let scansPerShard: Int
  public let scanTile: Int
  public let payloadChunkBytes: Int
  public let sourceIdentitySHA256: String
  public let sourceRawLogicalSHA256: String?
  public let workingLogicalSHA256: String?
  public let detectorMaskSHA256: String?
  public let maskedDetectorPixelsSHA256: String?
  public let maskedDetectorRawValues: [UInt16]?
  public let rawAccessMode: String
  public let detectorCalibration: MetalCompactH5DetectorCalibration?
  public let detectorCalibrationSchema: String?
  public let detectorCalibrationSHA256: String?
  public let preparedDPCMoments: MetalCompactH5PreparedDPCMoments?
  public let preparedDetectorProducts: MetalCompactH5PreparedDetectorProducts?
  public let excludedDetectorPixels: [Int]
  public let shardCount: Int
  public let residentBytes: UInt64

  public var scanCount: Int { scanRows * scanColumns }
  public var detectorPixelCount: Int { detectorRows * detectorColumns }
}

/// Measured phases for a successful compact HDF5 to private-Metal load.
public struct MetalCompactH5LoadMetrics: Equatable, Sendable {
  /// notRequested, miss, stale, or hit. Corrupt matching caches fail closed.
  public let nativeCacheStatus: String
  public let nativeCacheBytes: UInt64
  public let nativeCacheDescriptorSHA256Checks: Int
  /// Conservative overlapping requested buffers plus mapped authentication bytes.
  /// Excludes driver allocations, OS page cache, and caller-owned display products.
  public let plannedAdditionalBytes: UInt64
  /// Requested descriptor-scoped IO policy; this does not certify cold pages.
  public let sourceReadPolicy: String
  /// Maximum bounded shard staging windows used by this load.
  public let maximumInFlightShards: Int
  /// Elapsed pipeline wall time. Individual shard phase metrics are summed
  /// work durations and overlap when maximumInFlightShards is greater than one.
  public let shardPipelineMilliseconds: Double
  public let metadataMilliseconds: Double
  public let sourceReadMilliseconds: Double
  public let descriptorPreparationMilliseconds: Double
  /// Standalone GPU preparation commands: descriptor construction for compact
  /// input, or packing/products for original input. Fused intervals are separate.
  public let gpuPreparationMilliseconds: Double
  /// Decode-only commands. Fused intervals are counted separately; zero does
  /// not imply free decoding when a combined interval is present.
  public let gpuDecodeMilliseconds: Double
  public let decodedIntegrityMilliseconds: Double
  public let privateUploadMilliseconds: Double
  public let preparedDPCReadMilliseconds: Double
  public let preparedDPCAuthenticationMilliseconds: Double
  public let preparedDPCPrimeMilliseconds: Double
  public let preparedDetectorProductReadMilliseconds: Double
  public let preparedDetectorProductAuthenticationMilliseconds: Double
  public let totalMilliseconds: Double
  public let residentBytes: UInt64
  public let maximumTransientBytes: UInt64
  public let deviceAllocatedBytesBefore: UInt64
  public let deviceAllocatedBytesAfter: UInt64
  public let decodedShardSHA256Checks: Int
  /// False for explicitly trusted local loads; structural validation still runs.
  public let checksumsVerified: Bool
  /// Bytes copied after LZ4 decode solely to enter private residency.
  /// Trusted compressed loads decode directly into their final private buffer.
  public let decodedPayloadCopyBytes: UInt64
  public let mappedAuthenticationBytes: UInt64
  public let preparedDPCBytes: UInt64
  public let preparedDetectorProductBytes: UInt64
  public let interactionResidentBytes: UInt64
  public let totalResidentBytes: UInt64
  /// True only when original loading accepted source-bound exact DPC sums.
  public internal(set) var reusedPreparedDPC = false
  /// Combined GPU interval, not separable into decode/header timings. Add this
  /// to decode-only and preparation durations when summing GPU work.
  public internal(set) var gpuDecodeAndHeaderMilliseconds = 0.0
  /// Checked original decode, packing, verification and summary in one interval.
  /// Nil means this path was not used. Add to the other command durations, but
  /// do not attribute a fraction to decoding or packing without isolated timing.
  /// Includes a rejected fused attempt if the successful load retried normally.
  public internal(set) var gpuDecodeAndPackingMilliseconds: Double? = nil
}

/// Authentication and scheduling strategy for exact compact shard payloads.
public enum MetalCompactH5AuthenticationPolicy: Sendable, Equatable {
  /// Authenticate each bounded staging buffer before it is privately uploaded.
  case boundedSequential
  /// Overlap at most three independent shard reads, Metal decodes, SHA checks
  /// and private uploads. No file mapping or native cache is required.
  /// With checksum verification enabled, each shard is authenticated before
  /// publication. The load always waits for all shards and structural checks.
  case boundedConcurrent
  /// Hash independent file-backed shard ranges concurrently before bounded upload.
  /// This maps the complete file and is intended only for machines whose measured
  /// memory budget safely admits the additional file-backed residency.
  case parallelMapped
}

/// Source-descriptor caching policy, independent of the optional native cache.
public enum MetalCompactH5SourceReadPolicy: String, Sendable, Equatable {
  case systemDefault
  /// Require F_NOCACHE on each source descriptor before reading metadata or
  /// payload. This avoids adding read data to the filesystem cache, but it does
  /// not evict existing pages and does not by itself prove a controlled-cold run.
  /// No global cache setting, purge, or read-ahead setting is changed.
  case avoidCaching
}

/// One exact compact detector-mask update over the resident private buffers.
public struct MetalCompactH5DetectorMetrics: Equatable, Sendable {
  public let mode: String
  public let changedDetectorPixels: Int
  public let wallMilliseconds: Double
  public let gpuMilliseconds: Double
  public let fftDispatchCount: Int
}

/// Per-source result from one shared detector update across a resident series.
public struct MetalCompactH5DetectorSeriesItemMetrics: Equatable, Sendable {
  public let sourceIdentitySHA256: String
  public let mode: String
  public let changedDetectorPixels: Int
}

/// Timing and exact update results for one resident-series detector request.
public struct MetalCompactH5DetectorSeriesMetrics: Equatable, Sendable {
  public let sources: [MetalCompactH5DetectorSeriesItemMetrics]
  public let wallMilliseconds: Double
  public let gpuMilliseconds: Double
  public let submissionCount: Int
  public let fftDispatchCount: Int
}

/// Streaming exact hash of all mask-applied working values in scan-major order.
public struct MetalCompactH5LogicalHashMetrics: Equatable, Sendable {
  public let sha256: String
  public let logicalBytes: UInt64
  public let stagingBytes: UInt64
  public let wallMilliseconds: Double
  public let gpuMilliseconds: Double
}

/// Exact compact detector sum and its float32 mean diffraction display map.
public struct MetalCompactH5MeanDiffraction: Equatable, Sendable {
  public let detectorSum: [UInt64]
  public let mean: [Float]
  public let wallMilliseconds: Double
  public let gpuMilliseconds: Double
  public let dispatchCount: Int
  public let readbackBytes: UInt64
}

/// Exact total and detector-coordinate moments for every scan position.
public struct MetalCompactH5ExactDPCMoments: Equatable, Sendable {
  public let total: [UInt64]
  public let detectorRowMoment: [UInt64]
  public let detectorColumnMoment: [UInt64]
  public let sourceIdentitySHA256: String?
  public let detectorMaskSHA256: String?

  public init(
    total: [UInt64], detectorRowMoment: [UInt64], detectorColumnMoment: [UInt64],
    sourceIdentitySHA256: String? = nil, detectorMaskSHA256: String? = nil
  ) {
    self.total = total
    self.detectorRowMoment = detectorRowMoment
    self.detectorColumnMoment = detectorColumnMoment
    self.sourceIdentitySHA256 = sourceIdentitySHA256
    self.detectorMaskSHA256 = detectorMaskSHA256
  }
}

private struct CompactH5ShardRecord {
  let payloadOffset: UInt64
  let payloadBytes: UInt64
  let lengthsOffset: UInt64
  let lengthsBytes: UInt64
  let widthsOffset: UInt64
  let widthsBytes: UInt64
  let decodedBytes: UInt64
  let descriptorCount: UInt32
  let chunkCount: UInt32
  let decodedSHA256: String
  var descriptorsSHA256: String? = nil
}

private enum CompactH5StorageLayout {
  case lz4V1
  case directV3
}

private struct CompactH5ParsedIndex {
  let metadata: MetalCompactH5Metadata
  let manifestWorkingDtype: String
  let storageLayout: CompactH5StorageLayout
  let headerEncoding: UInt32
  let headerWordsPerPixel: Int
  let shards: [CompactH5ShardRecord]
}

private final class ConcurrentStringCollector: @unchecked Sendable {
  private let lock = NSLock()
  private var values: [String] = []

  func append(_ value: String) {
    lock.withLock { values.append(value) }
  }

  func snapshot() -> [String] {
    lock.withLock { values }
  }
}

private struct CompactLZ4Chunk {
  var inputOffset: UInt32
  var inputBytes: UInt32
  var outputWord: UInt32
  var outputBytes: UInt32
}

private struct CompactLZ4Parameters {
  var chunkCount: UInt32
  var compressedBytes: UInt32
}

private struct CompactDescriptorParameters {
  var descriptorCount: UInt32
  var payloadWords: UInt32
  var tileCount: UInt32
  var headerWordsPerPixel: UInt32
  var scanTile: UInt32
  var headerEncoding: UInt32
}

private struct CompactSelectedParameters {
  var scan: UInt32
  var tileCount: UInt32
  var pixelCount: UInt32
  var scanTile: UInt32
  var headerWordsPerPixel: UInt32
  var headerEncoding: UInt32
  var payloadLayout: UInt32 = 0
}

struct CompactDetectorEntry {
  var pixel: UInt32
  var coefficient: Int32
}

private struct CompactDetectorParameters {
  var scanCount: UInt32
  var tileCount: UInt32
  var entryCount: UInt32
  var outputOffset: UInt32
  var mode: UInt32
  var scanTile: UInt32
  var headerWordsPerPixel: UInt32
  var headerEncoding: UInt32
  var payloadLayout: UInt32 = 0
}

private struct CompactFullDecodeParameters {
  var scanCount: UInt32
  var pixelCount: UInt32
  var tileCount: UInt32
  var scanTile: UInt32
  var headerWordsPerPixel: UInt32
  var headerEncoding: UInt32
  var outputWordCount: UInt32
  var payloadLayout: UInt32 = 0
}

private struct CompactDetectorSumParameters {
  var scanCount: UInt32
  var tileCount: UInt32
  var pixelCount: UInt32
  var scanTile: UInt32
  var headerWordsPerPixel: UInt32
  var headerEncoding: UInt32
  var payloadLayout: UInt32 = 0
}

struct CompactResidentShard {
  let payload: MTLBuffer
  let descriptors: MTLBuffer
}

/// Workers never access this buffer on the CPU. Metal command buffers use
/// tracked resources and the validation shader writes only atomic maxima.
/// The caller reads it only after every bounded window has joined.
private struct CompactConcurrentWidthValidation: @unchecked Sendable {
  let buffer: MTLBuffer
}

/// Metal devices, command queues, and immutable pipeline states support
/// concurrent command encoding, but the macOS 15 SDK does not declare their
/// Objective-C protocols as `Sendable`.
private struct CompactConcurrentMetalHandle<Value>: @unchecked Sendable {
  let value: Value
}

private struct CompactShardLoadResult {
  let resident: CompactResidentShard
  let sourceReadMilliseconds: Double
  let descriptorPreparationMilliseconds: Double
  let gpuPreparationMilliseconds: Double
  let gpuDecodeMilliseconds: Double
  let decodedIntegrityMilliseconds: Double
  let privateUploadMilliseconds: Double
  let maximumTransientBytes: UInt64
}

/// Each worker owns its buffers until completion; only completed immutable
/// results cross this lock. Ordered collection preserves scan-shard ordering.
private final class CompactShardLoadCollector: @unchecked Sendable {
  private let lock = NSLock()
  private var values: [Result<CompactShardLoadResult, Error>]

  init(count: Int) {
    values = Array(repeating: .failure(Metal4DSTEMStreamingIOError.cancelled), count: count)
  }

  func set(_ result: Result<CompactShardLoadResult, Error>, at index: Int) {
    lock.withLock { values[index] = result }
  }

  func finish() throws -> [CompactShardLoadResult] {
    try lock.withLock { try values.map { try $0.get() } }
  }
}

private struct CompactPreparedDPCLoadResult {
  let moments: MTLBuffer
  let row: MTLBuffer
  let column: MTLBuffer
  let readMilliseconds: Double
  let authenticationMilliseconds: Double
  let primeMilliseconds: Double
}

private struct CompactPreparedDetectorResident {
  let metadata: MetalCompactH5PreparedDetectorProduct
  let mask: [UInt8]
  let values: MTLBuffer
}

private struct CompactPreparedDetectorLoadResult {
  let products: [String: CompactPreparedDetectorResident]
  let readMilliseconds: Double
  let authenticationMilliseconds: Double
  let bytes: UInt64
}

// A private lifetime owner keeps newer Metal types out of older-platform APIs.
private final class CompactResidencyLease {
  let attach: (MTLCommandBuffer) -> Void
  let allocationCount: Int
  let allocatedBytes: UInt64
  let footprintExcessBytes: UInt64
  private var endAction: (() -> Void)?

  @available(macOS 15.0, iOS 18.0, *)
  init(device: MTLDevice, buffers: [MTLBuffer]) throws {
    var identities = Set<ObjectIdentifier>()
    let unique = buffers.filter { identities.insert(ObjectIdentifier($0 as AnyObject)).inserted }
    let descriptor = MTLResidencySetDescriptor()
    descriptor.label = "Compact source stable allocations"
    descriptor.initialCapacity = unique.count
    let set = try device.makeResidencySet(descriptor: descriptor)
    for buffer in unique { set.addAllocation(buffer) }
    set.commit()
    guard set.allocationCount == unique.count,
      unique.allSatisfy({ set.containsAllocation($0) })
    else {
      set.removeAllAllocations()
      set.commit()
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Metal residency set did not retain every stable source allocation.")
    }
    allocationCount = set.allocationCount
    allocatedBytes = set.allocatedSize
    let resourceBytes = unique.reduce(UInt64(0)) { $0 + UInt64($1.allocatedSize) }
    footprintExcessBytes = set.allocatedSize > resourceBytes ? set.allocatedSize - resourceBytes : 0
    attach = { $0.useResidencySet(set) }
    endAction = {
      set.endResidency()
      set.removeAllAllocations()
      set.commit()
    }
    // Advisory residency, not pinning: competing apps may still defer work.
    set.requestResidency()
  }

  func end() {
    let action = endAction
    endAction = nil
    action?()
  }

  deinit { end() }
}

/// Exact interactions over a validated compact HDF5 source resident in Metal.
///
/// The large payload and descriptor buffers use private Metal storage. Detector
/// outputs are double-buffered and become visible only after a complete command
/// succeeds. The logical order is always
/// `(scan_row, scan_column, detector_row, detector_column)`.
public final class MetalCompactH5ResidentSource {
  public let metadata: MetalCompactH5Metadata
  public let loadMetrics: MetalCompactH5LoadMetrics

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let selectedPipeline: MTLComputePipelineState
  private let detectorPipeline: MTLComputePipelineState
  private let pixelLaneDetectorPipeline: MTLComputePipelineState
  private let planarILPDetectorPipeline: MTLComputePipelineState?
  private let planarScanCooperativePipeline: MTLComputePipelineState?
  private let planarFusedDetectorPipeline: MTLComputePipelineState?
  private let fullDecodePipeline: MTLComputePipelineState
  private let detectorSumPipeline: MTLComputePipelineState
  private let headerEncoding: UInt32
  private let payloadLayout: UInt32
  private let headerWordsPerPixel: UInt32
  private var shards: [CompactResidentShard]
  private var excluded: MTLBuffer?
  private let maximumWidths: [UInt8]
  private let maximumMaskSumBound: UInt64
  private let planarVariant: String
  private var detectorOutputs: [MTLBuffer]
  private var detectorEntryBuffer: MTLBuffer?
  private var diffractionOutput: MTLBuffer?
  private var detectorSumOutput: MTLBuffer?
  private var preparedDPCMomentBuffer: MTLBuffer?
  private var preparedDPCOutputs: [MTLBuffer]
  private var preparedDetectorProducts: [String: CompactPreparedDetectorResident]
  private var detectorRegions: CompactDetectorRegions?
  private var residencyLease: CompactResidencyLease?
  private var activeDetectorOutput = 0
  private var detectorMask: [UInt8]
  private var detectorSumReady = false
  public private(set) var isReleased = false

  fileprivate func markOriginalDetectorSumReady() { detectorSumReady = true }

  private struct PendingDetectorUpdate {
    let normalizedMask: [UInt8]
    var nextOutput: Int
    let mode: String
    let changedDetectorPixels: Int
    let preparedValues: MTLBuffer?
    let clearsOutput: Bool
    let shouldEncode: Bool
    var rawEntryCount: Int = 0
    var aggregateEntryCount: Int = 0
    var widthBucketed = false
    var widthBucketThreshold = 0
  }

  fileprivate init(
    metadata: MetalCompactH5Metadata,
    loadMetrics: MetalCompactH5LoadMetrics,
    device: MTLDevice,
    queue: MTLCommandQueue,
    selectedPipeline: MTLComputePipelineState,
    detectorPipeline: MTLComputePipelineState,
    pixelLaneDetectorPipeline: MTLComputePipelineState,
    planarILPDetectorPipeline: MTLComputePipelineState? = nil,
    planarScanCooperativePipeline: MTLComputePipelineState? = nil,
    planarFusedDetectorPipeline: MTLComputePipelineState? = nil,
    fullDecodePipeline: MTLComputePipelineState,
    detectorSumPipeline: MTLComputePipelineState,
    headerEncoding: UInt32,
    payloadLayout: UInt32,
    headerWordsPerPixel: UInt32,
    shards: [CompactResidentShard],
    excluded: MTLBuffer,
    maximumWidths: [UInt8],
    detectorOutputs: [MTLBuffer],
    detectorEntryBuffer: MTLBuffer,
    diffractionOutput: MTLBuffer,
    detectorSumOutput: MTLBuffer,
    preparedDPCMomentBuffer: MTLBuffer?,
    preparedDPCOutputs: [MTLBuffer],
    preparedDetectorProducts: [String: CompactPreparedDetectorResident],
    residencyLease: CompactResidencyLease? = nil,
    detectorRegions: CompactDetectorRegions? = nil,
    planarVariant: String = "other"
  ) {
    self.metadata = metadata
    self.loadMetrics = loadMetrics
    self.device = device
    self.queue = queue
    self.selectedPipeline = selectedPipeline
    self.detectorPipeline = detectorPipeline
    self.pixelLaneDetectorPipeline = pixelLaneDetectorPipeline
    self.planarILPDetectorPipeline = planarILPDetectorPipeline
    self.planarScanCooperativePipeline = planarScanCooperativePipeline
    self.planarFusedDetectorPipeline = planarFusedDetectorPipeline
    self.fullDecodePipeline = fullDecodePipeline
    self.detectorSumPipeline = detectorSumPipeline
    self.headerEncoding = headerEncoding
    self.payloadLayout = payloadLayout
    self.headerWordsPerPixel = headerWordsPerPixel
    self.shards = shards
    self.excluded = excluded
    self.maximumWidths = maximumWidths
    self.planarVariant = planarVariant
    self.maximumMaskSumBound = maximumWidths.reduce(UInt64(0)) { total, width in
      guard width < 64 else { return UInt64.max }
      let next = total.addingReportingOverflow((UInt64(1) << UInt64(width)) - 1)
      return next.overflow ? UInt64.max : next.partialValue
    }
    self.detectorOutputs = detectorOutputs
    self.detectorEntryBuffer = detectorEntryBuffer
    self.diffractionOutput = diffractionOutput
    self.detectorSumOutput = detectorSumOutput
    self.preparedDPCMomentBuffer = preparedDPCMomentBuffer
    self.preparedDPCOutputs = preparedDPCOutputs
    self.preparedDetectorProducts = preparedDetectorProducts
    self.residencyLease = residencyLease
    self.detectorRegions = detectorRegions
    detectorMask = [UInt8](repeating: 0, count: metadata.detectorPixelCount)
  }

  /// Save a disposable native acceleration cache without changing the source HDF5.
  ///
  /// This explicit one-time operation reads back only compact payload/lookup
  /// buffers, never a dense cube. The destination must not exist. Callers own
  /// cache location and eviction, and must serialize this with release/interaction.
  /// Reopen with `MetalCompactH5Loader.load(..., nativeCacheURL: url)`.
  public func saveNativeCache(
    to destination: URL,
    shouldCancel: () -> Bool = { false }
  ) throws {
    guard !isReleased, headerEncoding == 0, payloadLayout == 0,
      metadata.embeddedScientificSemantics, metadata.workingDtype == "uint16"
    else {
      throw CompactNativeCache.invalid("requires a live exact uint16 QGIX v1 source.")
    }
    guard !FileManager.default.fileExists(atPath: destination.path) else {
      throw CompactNativeCache.invalid("destination exists; choose a new cache path.")
    }
    let index = try MetalCompactH5Loader.parse(sourceURL: metadata.sourceURL)
    guard index.metadata == metadata, index.shards.count == shards.count else {
      throw CompactNativeCache.invalid("source changed after load; reload before preparing.")
    }
    let signature = try CompactNativeCache.signature(sourceURL: metadata.sourceURL)
    let temporary = destination.deletingLastPathComponent()
      .appendingPathComponent(".\(destination.lastPathComponent).\(UUID().uuidString).partial")
    let fd = temporary.path.withCString { Darwin.open($0, O_RDWR | O_CREAT | O_EXCL, 0o600) }
    guard fd >= 0 else { throw CompactNativeCache.invalid("could not create temporary cache.") }
    let file = FileHandle(fileDescriptor: fd, closeOnDealloc: true)
    defer {
      try? file.close()
      try? FileManager.default.removeItem(at: temporary)
    }
    let original = try FileHandle(forReadingFrom: metadata.sourceURL)
    defer { try? original.close() }
    try file.write(contentsOf: Data(count: CompactNativeCache.headerBytes))
    var records: [CompactNativeCache.Shard] = []
    for (ordinal, shard) in shards.enumerated() {
      try autoreleasepool {
        guard !shouldCancel() else { throw Metal4DSTEMStreamingIOError.cancelled }
        let record = index.shards[ordinal]
        guard
          let payload = device.makeBuffer(
            length: shard.payload.length, options: .storageModeShared),
          let descriptors = device.makeBuffer(
            length: shard.descriptors.length, options: .storageModeShared),
          let command = queue.makeCommandBuffer(), let blit = command.makeBlitCommandEncoder()
        else { throw CompactNativeCache.invalid("could not allocate bounded cache readback.") }
        blit.copy(
          from: shard.payload, sourceOffset: 0, to: payload, destinationOffset: 0,
          size: payload.length)
        blit.copy(
          from: shard.descriptors, sourceOffset: 0, to: descriptors, destinationOffset: 0,
          size: descriptors.length)
        blit.endEncoding()
        try Self.complete(command, operation: "native cache readback")
        let payloadData = Data(
          bytesNoCopy: payload.contents(), count: payload.length, deallocator: .none)
        let headerData = Data(
          bytesNoCopy: descriptors.contents(), count: descriptors.length, deallocator: .none)
        let payloadSHA = CompactNativeCache.digest(payloadData)
        guard payloadSHA == record.decodedSHA256 else {
          throw CompactNativeCache.invalid("resident payload differs from source shard \(ordinal).")
        }
        // Bind the lookup table to the authoritative source widths, not merely
        // to a newly computed hash of whatever happened to be resident.
        try original.seek(toOffset: record.widthsOffset)
        let widths = try CompactNativeCache.read(original, count: Int(record.widthsBytes))
        let words = descriptors.contents().bindMemory(to: UInt32.self, capacity: widths.count)
        var wordOffset: UInt32 = 0
        for (i, width) in widths.enumerated() {
          guard width <= 16, words[i] == (wordOffset << 5) | UInt32(width) else {
            throw CompactNativeCache.invalid("lookup differs from source shard \(ordinal).")
          }
          wordOffset += UInt32(width) * 4
        }
        guard UInt64(wordOffset) * 4 == record.decodedBytes else {
          throw CompactNativeCache.invalid("lookup coverage differs from source.")
        }
        let payloadOffset = try file.offset()
        try file.write(contentsOf: payloadData)
        let descriptorsOffset = try file.offset()
        try file.write(contentsOf: headerData)
        records.append(
          CompactNativeCache.Shard(
            payloadOffset: payloadOffset, payloadBytes: UInt64(payload.length),
            payloadSHA256: payloadSHA, descriptorsOffset: descriptorsOffset,
            descriptorsBytes: UInt64(descriptors.length),
            descriptorsSHA256: CompactNativeCache.digest(headerData)
          ))
      }
    }
    guard !shouldCancel() else { throw Metal4DSTEMStreamingIOError.cancelled }
    guard signature == (try CompactNativeCache.signature(sourceURL: metadata.sourceURL)) else {
      throw CompactNativeCache.invalid("source changed during preparation; reload and retry.")
    }
    let cache = CompactNativeCache(
      schema: "quantem.gpu.native-compact-cache/v1", sourceSignature: signature,
      sourceManifestSHA256: metadata.manifestSHA256,
      fileBytes: try file.offset(), shards: records
    )
    try cache.writeHeader(to: file)
    try file.synchronize()
    try file.close()
    // renamex_np excludes existing destinations atomically, including a racer.
    let result = temporary.path.withCString { from in
      destination.path.withCString { to in Darwin.renamex_np(from, to, UInt32(RENAME_EXCL)) }
    }
    guard result == 0 else {
      throw CompactNativeCache.invalid(
        "atomic publication failed; existing destination was preserved.")
    }
  }

  /// Return one complete exact mask-applied diffraction pattern as row-major u32.
  public func extractDiffraction(
    scanRow: Int,
    scanColumn: Int
  ) throws -> [UInt32] {
    guard let diffractionOutput else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source has been released. Load it again before requesting diffraction."
      )
    }
    try extractDiffraction(scanRow: scanRow, scanColumn: scanColumn, into: diffractionOutput)
    return Self.u32Values(diffractionOutput, count: metadata.detectorPixelCount)
  }

  /// Decode one exact mask-applied diffraction pattern into an owned Metal buffer.
  ///
  /// The shared buffer contains detector-row-major uint32 counts and remains
  /// unchanged by later extraction or source release. No CPU array or dense 4D
  /// allocation is made. Serialize this operation with interactions and release;
  /// release snapshots after their display consumers finish.
  public func snapshotDiffraction(scanRow: Int, scanColumn: Int) throws -> MTLBuffer {
    guard !isReleased else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source has been released. Load it again before requesting diffraction."
      )
    }
    let bytes = metadata.detectorPixelCount * MemoryLayout<UInt32>.stride
    guard let output = device.makeBuffer(length: bytes, options: .storageModeShared) else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "selected diffraction snapshot", bytes: UInt64(bytes))
    }
    try extractDiffraction(scanRow: scanRow, scanColumn: scanColumn, into: output)
    return output
  }

  private func extractDiffraction(
    scanRow: Int, scanColumn: Int, into output: MTLBuffer
  ) throws {
    guard !isReleased, let excluded,
      shards.count == metadata.shardCount
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source has been released. Load it again before "
          + "requesting a diffraction pattern."
      )
    }
    guard 0..<metadata.scanRows ~= scanRow,
      0..<metadata.scanColumns ~= scanColumn
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Selected scan (row: \(scanRow), column: \(scanColumn)) is outside "
          + "shape (\(metadata.scanRows), \(metadata.scanColumns))."
      )
    }
    let globalScan = scanRow * metadata.scanColumns + scanColumn
    let shardIndex = globalScan / metadata.scansPerShard
    var parameters = CompactSelectedParameters(
      scan: UInt32(globalScan % metadata.scansPerShard),
      tileCount: UInt32((metadata.scansPerShard + metadata.scanTile - 1) / metadata.scanTile),
      pixelCount: UInt32(metadata.detectorPixelCount),
      scanTile: UInt32(metadata.scanTile),
      headerWordsPerPixel: headerWordsPerPixel,
      headerEncoding: headerEncoding,
      payloadLayout: payloadLayout
    )
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode compact selected-diffraction extraction."
      )
    }
    encoder.setComputePipelineState(selectedPipeline)
    encoder.setBuffer(shards[shardIndex].payload, offset: 0, index: 0)
    encoder.setBuffer(shards[shardIndex].descriptors, offset: 0, index: 1)
    encoder.setBuffer(excluded, offset: 0, index: 2)
    encoder.setBuffer(output, offset: 0, index: 3)
    encoder.setBytes(
      &parameters,
      length: MemoryLayout.stride(ofValue: parameters),
      index: 4
    )
    encoder.dispatchThreads(
      MTLSize(width: metadata.detectorPixelCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.endEncoding()
    residencyLease?.attach(command)
    try Self.complete(command, operation: "selected diffraction")
  }

  /// Return the exact detector sum and float32 mean diffraction pattern.
  ///
  /// A validated sum prepared during original loading is reused immediately.
  /// Otherwise the first call performs one complete resident pass and later
  /// calls reuse that u64 result. No dense 4D tensor is decoded or allocated.
  public func meanDiffractionPattern() throws -> MetalCompactH5MeanDiffraction {
    guard !isReleased, let excluded, let detectorSumOutput else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source has been released. Load it again before reading mean diffraction."
      )
    }
    var wallMilliseconds = 0.0
    var gpuMilliseconds = 0.0
    var dispatchCount = 0
    if !detectorSumReady {
      memset(detectorSumOutput.contents(), 0, detectorSumOutput.length)
      guard let command = queue.makeCommandBuffer(),
        let encoder = command.makeComputeCommandEncoder()
      else {
        throw Metal4DSTEMStreamingIOError.commandFailed(
          "Metal could not encode compact mean diffraction."
        )
      }
      let started = ContinuousClock.now
      for shard in shards {
        var parameters = CompactDetectorSumParameters(
          scanCount: UInt32(metadata.scansPerShard),
          tileCount: UInt32(
            (metadata.scansPerShard + metadata.scanTile - 1) / metadata.scanTile
          ),
          pixelCount: UInt32(metadata.detectorPixelCount),
          scanTile: UInt32(metadata.scanTile),
          headerWordsPerPixel: headerWordsPerPixel,
          headerEncoding: headerEncoding,
          payloadLayout: payloadLayout
        )
        encoder.setComputePipelineState(detectorSumPipeline)
        encoder.setBuffer(shard.payload, offset: 0, index: 0)
        encoder.setBuffer(shard.descriptors, offset: 0, index: 1)
        encoder.setBuffer(excluded, offset: 0, index: 2)
        encoder.setBuffer(detectorSumOutput, offset: 0, index: 3)
        encoder.setBytes(
          &parameters,
          length: MemoryLayout<CompactDetectorSumParameters>.stride,
          index: 4
        )
        encoder.dispatchThreads(
          MTLSize(width: metadata.detectorPixelCount, height: 1, depth: 1),
          threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
        )
        encoder.memoryBarrier(scope: .buffers)
        dispatchCount += 1
      }
      encoder.endEncoding()
      try Self.complete(command, operation: "mean diffraction")
      wallMilliseconds = Self.milliseconds(from: started)
      gpuMilliseconds =
        command.gpuEndTime > command.gpuStartTime
        ? (command.gpuEndTime - command.gpuStartTime) * 1_000 : 0
      detectorSumReady = true
    }
    let detectorSum = Self.u64Values(
      detectorSumOutput,
      count: metadata.detectorPixelCount
    )
    let divisor = Float(metadata.scanCount)
    return MetalCompactH5MeanDiffraction(
      detectorSum: detectorSum,
      mean: detectorSum.map { Float($0) / divisor },
      wallMilliseconds: wallMilliseconds,
      gpuMilliseconds: gpuMilliseconds,
      dispatchCount: dispatchCount,
      readbackBytes: UInt64(detectorSumOutput.length)
    )
  }

  /// Apply a row-major detector mask and atomically publish its exact u32 scan map.
  ///
  /// The first call is a full rebase. Later calls choose signed mask deltas or
  /// a smaller full rebase. A mask that could overflow a u32 accumulator is
  /// rejected before GPU execution.
  @discardableResult
  public func updateVirtualDetector(
    mask: [UInt8],
    forceRebase: Bool = false
  ) throws -> MetalCompactH5DetectorMetrics {
    let batch = try Self.updateVirtualDetectors(
      [self], mask: mask, forceRebase: forceRebase
    )
    let source = batch.sources[0]
    return MetalCompactH5DetectorMetrics(
      mode: source.mode,
      changedDetectorPixels: source.changedDetectorPixels,
      wallMilliseconds: batch.wallMilliseconds,
      gpuMilliseconds: batch.gpuMilliseconds,
      fftDispatchCount: batch.fftDispatchCount
    )
  }

  /// Apply one exact detector mask to every source in a resident tilt series.
  ///
  /// All source updates are encoded into one Metal command buffer and become
  /// visible only after that shared submission completes. Sources must use the
  /// same Metal device and detector geometry. Each source retains its own
  /// persistent output buffer, so no dense 4D tensor or cross-source copy is
  /// introduced.
  ///
  /// ```swift
  /// let metrics = try MetalCompactH5ResidentSource.updateVirtualDetectors(
  ///   residentTilts, mask: brightFieldMask
  /// )
  /// ```
  @discardableResult
  public static func updateVirtualDetectors(
    _ sources: [MetalCompactH5ResidentSource],
    mask: [UInt8],
    forceRebase: Bool = false
  ) throws -> MetalCompactH5DetectorSeriesMetrics {
    try performDetectorUpdates(
      sources, mask: mask, forceRebase: forceRebase, captureSnapshots: false
    ).metrics
  }

  /// Update exact detector maps and return independently owned display images.
  ///
  /// The images contain scan-row-major uint32 counts. Reduction and image
  /// copies share one command submission and completion wait; GPU timing
  /// includes both operations. Later source updates and release cannot change
  /// these snapshots. On failure, the caller's previous snapshots are retained.
  /// Serialize this call with other updates and release on the same sources.
  ///
  /// ```swift
  /// var images: [MTLBuffer] = []
  /// let metrics = try MetalCompactH5ResidentSource.updateVirtualDetectors(
  ///   tilts, mask: detectorMask, snapshots: &images
  /// )
  /// ```
  @discardableResult
  public static func updateVirtualDetectors(
    _ sources: [MetalCompactH5ResidentSource],
    mask: [UInt8],
    forceRebase: Bool = false,
    snapshots: inout [MTLBuffer]
  ) throws -> MetalCompactH5DetectorSeriesMetrics {
    let completed = try performDetectorUpdates(
      sources, mask: mask, forceRebase: forceRebase, captureSnapshots: true
    )
    snapshots = completed.snapshots
    return completed.metrics
  }

  private static func performDetectorUpdates(
    _ sources: [MetalCompactH5ResidentSource],
    mask: [UInt8],
    forceRebase: Bool,
    captureSnapshots: Bool
  ) throws -> (metrics: MetalCompactH5DetectorSeriesMetrics, snapshots: [MTLBuffer]) {
    let hostProfile =
      ProcessInfo.processInfo.environment["COMPACT_UPDATE_HOST_PROFILE"] == "1"
      && ProcessInfo.processInfo.environment["COMPACT_UPDATE_PHASE_PROFILE"] != "1"
    let hostProfileStart = ContinuousClock.now
    var hostStages: [String: Double] = [:]
    var commandHostTimes: [String: Double] = [:]
    func hostStamp(_ name: String) {
      if hostProfile { hostStages[name] = milliseconds(from: hostProfileStart) }
    }
    guard let first = sources.first else {
      return (
        MetalCompactH5DetectorSeriesMetrics(
          sources: [], wallMilliseconds: 0, gpuMilliseconds: 0,
          submissionCount: 0, fftDispatchCount: 0
        ), []
      )
    }
    guard Set(sources.map(ObjectIdentifier.init)).count == sources.count else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "A resident detector series cannot contain the same source twice. "
          + "Remove duplicate sources before updating the series."
      )
    }
    guard
      sources.allSatisfy({
        ($0.device as AnyObject) === (first.device as AnyObject)
          && $0.metadata.detectorRows == first.metadata.detectorRows
          && $0.metadata.detectorColumns == first.metadata.detectorColumns
      })
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "A resident detector series requires one Metal device and matching "
          + "detector (row, column) geometry. Load the tilts together."
      )
    }

    hostStamp("prepare_start_ms")
    // Reuse only a proven-identical mask transition. Count data and output
    // buffers remain source-specific. Heterogeneous histories, exclusions,
    // prepared products or range bounds retain independent validation.
    let sharesPreparation =
      compactKernelOption("SHARED_MASK_PREPARATION", byDefault: true)
      && sources.count > 1
      && sources.allSatisfy {
        !$0.isReleased && $0.detectorOutputs.count == 2 && $0.detectorEntryBuffer != nil
          && $0.shards.count == $0.metadata.shardCount
          && $0.maximumMaskSumBound <= UInt64(UInt32.max)
          && $0.detectorRegions == nil && $0.preparedDetectorProducts.isEmpty
          && $0.metadata.excludedDetectorPixels == first.metadata.excludedDetectorPixels
          && $0.detectorMask == first.detectorMask
          && ($0.preparedDPCMomentBuffer != nil) == (first.preparedDPCMomentBuffer != nil)
      }
    let updates: [PendingDetectorUpdate]
    if sharesPreparation {
      let prepared = try first.prepareDetectorUpdate(mask: mask, forceRebase: forceRebase)
      let bytes = prepared.rawEntryCount * MemoryLayout<CompactDetectorEntry>.stride
      updates = try sources.map { source in
        if source !== first, bytes > 0 {
          guard let from = first.detectorEntryBuffer, let to = source.detectorEntryBuffer,
            bytes <= from.length, bytes <= to.length
          else {
            throw Metal4DSTEMStreamingIOError.invalidRequest(
              "Shared detector entries exceed the admitted source buffer")
          }
          memcpy(to.contents(), from.contents(), bytes)
        }
        var update = prepared
        update.nextOutput = 1 - source.activeDetectorOutput
        return update
      }
    } else {
      updates = try sources.map {
        try $0.prepareDetectorUpdate(mask: mask, forceRebase: forceRebase)
      }
    }
    hostStamp("prepare_end_ms")
    let wallStart = ContinuousClock.now
    let snapshots = captureSnapshots ? try allocateDetectorSnapshots(sources) : []
    hostStamp("snapshot_allocation_end_ms")
    let needsSubmission = captureSnapshots || updates.contains(where: \.shouldEncode)
    var gpuMilliseconds = 0.0
    let profilePhases = ProcessInfo.processInfo.environment["COMPACT_UPDATE_PHASE_PROFILE"] == "1"
    var phaseMilliseconds = [Double](repeating: 0, count: 3)
    var submissionCount = 0
    // Metal limits residency-set hints per command, not scientific series
    // length. Larger series retain ordinary tracked-resource residency.
    let residencyLeases = sources.count <= 32 ? sources.compactMap(\.residencyLease) : []
    if needsSubmission {
      guard var command = first.queue.makeCommandBuffer() else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Metal could not create a command buffer for the resident detector series."
        )
      }
      for lease in residencyLeases { lease.attach(command) }
      for (source, update) in zip(sources, updates)
      where update.shouldEncode && (update.preparedValues != nil || update.clearsOutput) {
        try source.encodeDetectorUpdate(update, command: command)
      }
      let reductions = zip(sources, updates).filter {
        $0.1.shouldEncode && $0.1.preparedValues == nil && !$0.1.clearsOutput
      }
      if !reductions.isEmpty {
        guard let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            "Metal could not create the concurrent resident-series encoder."
          )
        }
        // Residents have independent outputs; shards write disjoint scan ranges.
        // No dispatch consumes another dispatch's output in this encoder.
        for (source, update) in reductions {
          try source.encodeDetectorUpdate(update, command: command, sharedEncoder: encoder)
        }
        encoder.endEncoding()
      }
      if profilePhases {
        try complete(command, operation: "diagnostic raw detector phase")
        submissionCount += 1
        phaseMilliseconds[0] = max(0, command.gpuEndTime - command.gpuStartTime) * 1_000
        guard let next = first.queue.makeCommandBuffer() else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            "Metal could not create diagnostic aggregate command.")
        }
        command = next
        for lease in residencyLeases { lease.attach(command) }
      }
      let aggregateUpdates = zip(sources, updates).filter { $0.1.aggregateEntryCount > 0 }
      if !aggregateUpdates.isEmpty {
        guard let encoder = command.makeComputeCommandEncoder(dispatchType: .concurrent) else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            "Cannot encode exact aggregate detector updates")
        }
        for (source, update) in aggregateUpdates {
          try source.encodeDetectorRegions(update, encoder: encoder)
        }
        encoder.endEncoding()
      }
      if profilePhases {
        try complete(command, operation: "diagnostic aggregate detector phase")
        submissionCount += 1
        phaseMilliseconds[1] = max(0, command.gpuEndTime - command.gpuStartTime) * 1_000
        guard let next = first.queue.makeCommandBuffer() else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            "Metal could not create diagnostic snapshot command.")
        }
        command = next
        for lease in residencyLeases { lease.attach(command) }
      }
      if captureSnapshots {
        guard let encoder = command.makeBlitCommandEncoder() else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            "Metal could not encode the completed detector image snapshots."
          )
        }
        for (index, source) in sources.enumerated() {
          let update = updates[index]
          let outputIndex = update.shouldEncode ? update.nextOutput : source.activeDetectorOutput
          encoder.copy(
            from: source.detectorOutputs[outputIndex], sourceOffset: 0,
            to: snapshots[index], destinationOffset: 0, size: snapshots[index].length
          )
        }
        encoder.endEncoding()
      }
      hostStamp("encoding_end_ms")
      if hostProfile {
        try complete(command, operation: "resident detector series", hostStamp: hostStamp)
      } else {
        try complete(command, operation: "resident detector series")
      }
      if hostProfile {
        // kernel* timestamps describe CPU driver scheduling, not GPU shaders.
        commandHostTimes = [
          "kernel_start": command.kernelStartTime, "kernel_end": command.kernelEndTime,
          "gpu_start": command.gpuStartTime, "gpu_end": command.gpuEndTime,
        ]
      }
      submissionCount += 1
      gpuMilliseconds = max(0, command.gpuEndTime - command.gpuStartTime) * 1_000
      if profilePhases {
        phaseMilliseconds[2] = gpuMilliseconds
        gpuMilliseconds += phaseMilliseconds[0] + phaseMilliseconds[1]
        let aggregateEntries = updates.map(\.aggregateEntryCount)
        let record: [String: Any] = [
          "phase": "detector_update_stages", "diagnostic_split_commands": true,
          "raw_gpu_ms": phaseMilliseconds[0], "aggregate_gpu_ms": phaseMilliseconds[1],
          "snapshot_gpu_ms": phaseMilliseconds[2], "submission_count": submissionCount,
          "raw_entries": updates.map(\.rawEntryCount),
          "aggregate_entries": aggregateEntries, "modes": updates.map(\.mode),
        ]
        var json = try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys])
        json.append(10)
        FileHandle.standardError.write(json)
      }
    }
    for (source, update) in zip(sources, updates) {
      source.publishDetectorUpdate(update)
    }
    let sourceMetrics = zip(sources, updates).map { source, update in
      MetalCompactH5DetectorSeriesItemMetrics(
        sourceIdentitySHA256: source.metadata.sourceIdentitySHA256,
        mode: update.mode,
        changedDetectorPixels: update.changedDetectorPixels
      )
    }
    let measuredWallMilliseconds = milliseconds(from: wallStart)
    hostStamp("publication_end_ms")
    if hostProfile {
      let aggregateEntries = updates.map(\.aggregateEntryCount)
      let record: [String: Any] = [
        "phase": "detector_host_stages", "diagnostic_only": true,
        "shared_mask_preparation": sharesPreparation,
        "stages_ms_from_call_entry": hostStages,
        "command_host_times_seconds": commandHostTimes,
        "wall_after_prepare_ms": measuredWallMilliseconds,
        "gpu_interval_ms": gpuMilliseconds,
        "submission_count": submissionCount, "source_count": sources.count,
        "residency_set_count": residencyLeases.count,
        "planar_pipeline_thread_limits": sources.map {
          $0.planarScanCooperativePipeline?.maxTotalThreadsPerThreadgroup ?? 0
        },
        "planar_variants": sources.map(\.planarVariant),
        "planar_pipeline_required_threads": sources.map { source -> Int in
          if #available(macOS 26.0, iOS 26.0, *) {
            return source.planarScanCooperativePipeline?.requiredThreadsPerThreadgroup.width ?? 0
          }
          return 0
        },
        "mask_sha256": SHA256.hash(data: Data(mask)).map { String(format: "%02x", $0) }.joined(),
        "source_identities": sources.map { $0.metadata.sourceIdentitySHA256 },
        "modes": updates.map(\.mode),
        "raw_entries": updates.map(\.rawEntryCount),
        "width_bucketed": updates.map(\.widthBucketed),
        "width_bucket_threshold": updates.map(\.widthBucketThreshold),
        "aggregate_entries": aggregateEntries,
        "force_rebase": forceRebase, "capture_snapshots": captureSnapshots,
      ]
      // Logging is outside the reported wall interval and must not change
      // scientific success or snapshot publication if diagnostics fail.
      if var json = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) {
        json.append(10)
        FileHandle.standardError.write(json)
      }
    }
    return (
      MetalCompactH5DetectorSeriesMetrics(
        sources: sourceMetrics,
        wallMilliseconds: measuredWallMilliseconds,
        gpuMilliseconds: gpuMilliseconds,
        submissionCount: submissionCount,
        fftDispatchCount: 0
      ), snapshots
    )
  }

  private func prepareDetectorUpdate(
    mask: [UInt8], forceRebase: Bool
  ) throws -> PendingDetectorUpdate {
    guard !isReleased, detectorOutputs.count == 2, let detectorEntryBuffer,
      shards.count == metadata.shardCount
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source has been released. Load it again before "
          + "updating a virtual detector."
      )
    }
    guard mask.count == metadata.detectorPixelCount,
      mask.allSatisfy({ $0 == 0 || $0 == 1 })
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "A compact virtual-detector mask requires one zero-or-one byte for each "
          + "of \(metadata.detectorPixelCount) row-major detector pixels."
      )
    }
    var normalized = mask
    for pixel in metadata.excludedDetectorPixels { normalized[pixel] = 0 }
    var maximumSum: UInt64 = 0
    for pixel in normalized.indices where normalized[pixel] != 0 {
      let width = maximumWidths[pixel]
      maximumSum += width == 0 ? 0 : (UInt64(1) << UInt64(width)) - 1
    }
    guard maximumSum <= UInt64(UInt32.max) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "This detector can sum to \(maximumSum), beyond exact u32 output. "
          + "Use a narrower detector or a future u64 reduction path."
      )
    }
    let nextOutput = 1 - activeDetectorOutput
    if !forceRebase,
      let prepared = preparedDetectorProducts.values.first(where: {
        $0.mask == normalized
      })
    {
      let changedCount = zip(normalized, detectorMask).count { $0 != $1 }
      return PendingDetectorUpdate(
        normalizedMask: normalized, nextOutput: nextOutput,
        mode: changedCount == 0 ? "delta" : "prepared",
        changedDetectorPixels: changedCount,
        preparedValues: changedCount == 0 ? nil : prepared.values,
        clearsOutput: false,
        shouldEncode: changedCount != 0
      )
    }
    let selectedCount = normalized.count { $0 != 0 }
    let changedCount = zip(normalized, detectorMask).count { $0 != $1 }
    if !forceRebase && changedCount == 0 {
      return PendingDetectorUpdate(
        normalizedMask: normalized, nextOutput: nextOutput, mode: "delta",
        changedDetectorPixels: 0, preparedValues: nil, clearsOutput: false,
        shouldEncode: false
      )
    }
    let complementCount =
      metadata.detectorPixelCount
      - metadata.excludedDetectorPixels.count - selectedCount
    // Authenticated exact totals are already resident. Subtracting their
    // complement is cheaper for nearly full detectors and preserves counts.
    // A forced rebase deliberately bypasses all prepared evidence for audits.
    let preferComplement =
      !forceRebase && preparedDPCMomentBuffer != nil
      && complementCount < min(selectedCount, changedCount)
    // A small detector moved far away can be cheaper to sum from zero than
    // to subtract its old support and add the new support. Both are exact.
    let preferRebase =
      forceRebase || detectorMask.allSatisfy({ $0 == 0 })
      || selectedCount < changedCount
    let isComplement = preferComplement
    let isRebase = preferRebase
    var entries: [CompactDetectorEntry] = []
    entries.reserveCapacity(metadata.detectorPixelCount)
    for pixel in normalized.indices {
      if isComplement {
        if normalized[pixel] == 0 && !metadata.excludedDetectorPixels.contains(pixel) {
          entries.append(CompactDetectorEntry(pixel: UInt32(pixel), coefficient: -1))
        }
      } else if isRebase, normalized[pixel] != 0 {
        entries.append(CompactDetectorEntry(pixel: UInt32(pixel), coefficient: 1))
      } else if !isRebase, normalized[pixel] != detectorMask[pixel] {
        entries.append(
          CompactDetectorEntry(
            pixel: UInt32(pixel),
            coefficient: normalized[pixel] == 0 ? -1 : 1
          )
        )
      }
    }
    let entryCount = entries.count
    let aggregateCount =
      forceRebase
      ? 0
      : detectorRegions?.decompose(
        &entries, rows: metadata.detectorRows, columns: metadata.detectorColumns) ?? 0
    // Private experiment: stable linear-time width grouping may reduce SIMD
    // work caused by rare wide pixels. It reorders exact integer additions in
    // the existing entry buffer; source values and resident sizes do not change.
    let widthBucketed =
      compactKernelOption("PLANAR_WIDTH_BUCKETS", byDefault: false)
      && payloadLayout == 1 && aggregateCount == 0 && entries.count >= 32
      && maximumWidths.allSatisfy { $0 <= 16 }
    if widthBucketed {
      let wideOnly = compactKernelOption("PLANAR_WIDE_ONLY", byDefault: false)
      func bucket(_ entry: CompactDetectorEntry) -> Int {
        let width = Int(maximumWidths[Int(entry.pixel)])
        return wideOnly ? (width > 8 ? 1 : 0) : width
      }
      var offsets = [Int](repeating: 0, count: 17)
      for entry in entries { offsets[bucket(entry)] += 1 }
      var prefix = 0
      for width in offsets.indices {
        let count = offsets[width]
        offsets[width] = prefix
        prefix += count
      }
      let output = detectorEntryBuffer.contents().bindMemory(
        to: CompactDetectorEntry.self, capacity: entries.count)
      for entry in entries {
        let width = bucket(entry)
        output[offsets[width]] = entry
        offsets[width] += 1
      }
    } else if !entries.isEmpty {
      _ = entries.withUnsafeBytes { raw in
        memcpy(detectorEntryBuffer.contents(), raw.baseAddress!, raw.count)
      }
    }
    let clearsOutput = !isComplement && isRebase && entryCount == 0
    var update = PendingDetectorUpdate(
      normalizedMask: normalized, nextOutput: nextOutput,
      mode: isComplement ? "complement" : (isRebase ? "rebase" : "delta"),
      changedDetectorPixels: entryCount,
      preparedValues: nil,
      clearsOutput: clearsOutput,
      shouldEncode: isComplement || clearsOutput || entryCount != 0
    )
    update.rawEntryCount = entries.count
    update.aggregateEntryCount = aggregateCount
    update.widthBucketed = widthBucketed
    update.widthBucketThreshold =
      widthBucketed && compactKernelOption("PLANAR_WIDE_ONLY", byDefault: false) ? 8 : 0
    return update
  }

  private func encodeDetectorUpdate(
    _ update: PendingDetectorUpdate,
    command: MTLCommandBuffer,
    sharedEncoder: MTLComputeCommandEncoder? = nil
  ) throws {
    guard let detectorEntryBuffer else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source was released before its detector update. "
          + "Reload the tilt series and try again."
      )
    }
    if let prepared = update.preparedValues {
      guard let blit = command.makeBlitCommandEncoder() else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Metal could not encode a prepared detector product for the resident series."
        )
      }
      blit.copy(
        from: prepared, sourceOffset: 0,
        to: detectorOutputs[update.nextOutput], destinationOffset: 0,
        size: metadata.scanCount * MemoryLayout<UInt32>.stride
      )
      blit.endEncoding()
      return
    }
    if update.clearsOutput {
      guard let blit = command.makeBlitCommandEncoder() else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Metal could not clear a resident detector output for the resident series."
        )
      }
      blit.fill(
        buffer: detectorOutputs[update.nextOutput],
        range: 0..<metadata.scanCount * MemoryLayout<UInt32>.stride,
        value: 0
      )
      blit.endEncoding()
      return
    }
    guard let encoder = sharedEncoder ?? command.makeComputeCommandEncoder() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode an exact detector update for the resident series."
      )
    }
    // Crossover tests cover core, outer and mixed-radius entries. The
    // scan-lane topology wins tiny deltas; pixel lanes win from 32 entries.
    let rawEntryCount = update.rawEntryCount
    let nextBuffer =
      update.aggregateEntryCount > 0
      ? detectorRegions!.intermediate : detectorOutputs[update.nextOutput]
    let usesScanCooperative =
      payloadLayout == 1 && rawEntryCount >= 32
      && planarScanCooperativePipeline != nil
    let widePipeline =
      payloadLayout == 1
      ? (planarScanCooperativePipeline ?? planarILPDetectorPipeline ?? pixelLaneDetectorPipeline)
      : pixelLaneDetectorPipeline
    encoder.setComputePipelineState(rawEntryCount < 32 ? detectorPipeline : widePipeline)
    let scansPerGroup = usesScanCooperative ? 128 : 32
    for shardIndex in shards.indices {
      var parameters = CompactDetectorParameters(
        scanCount: UInt32(metadata.scansPerShard),
        tileCount: UInt32(
          (metadata.scansPerShard + metadata.scanTile - 1) / metadata.scanTile
        ),
        entryCount: UInt32(rawEntryCount),
        outputOffset: UInt32(shardIndex * metadata.scansPerShard),
        mode: update.mode == "complement" ? 2 : (update.mode == "rebase" ? 1 : 0),
        scanTile: UInt32(metadata.scanTile),
        headerWordsPerPixel: headerWordsPerPixel,
        headerEncoding: headerEncoding,
        payloadLayout: payloadLayout
      )
      encoder.setBuffer(shards[shardIndex].payload, offset: 0, index: 0)
      encoder.setBuffer(shards[shardIndex].descriptors, offset: 0, index: 1)
      encoder.setBuffer(detectorEntryBuffer, offset: 0, index: 2)
      encoder.setBuffer(detectorOutputs[activeDetectorOutput], offset: 0, index: 3)
      encoder.setBuffer(nextBuffer, offset: 0, index: 4)
      // The fallback is never read unless mode == 2; binding it keeps the
      // pipeline argument valid for sources without prepared totals.
      encoder.setBuffer(
        preparedDPCMomentBuffer ?? detectorOutputs[activeDetectorOutput], offset: 0, index: 6
      )
      encoder.setBytes(
        &parameters,
        length: MemoryLayout.stride(ofValue: parameters),
        index: 5
      )
      encoder.dispatchThreadgroups(
        MTLSize(
          width: (metadata.scansPerShard + scansPerGroup - 1)
            / scansPerGroup, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
      )
    }
    if sharedEncoder == nil { encoder.endEncoding() }
  }

  private func publishDetectorUpdate(_ update: PendingDetectorUpdate) {
    if update.shouldEncode { activeDetectorOutput = update.nextOutput }
    detectorMask = update.normalizedMask
  }

  private func encodeDetectorRegions(
    _ update: PendingDetectorUpdate, encoder: MTLComputeCommandEncoder
  ) throws {
    guard let detectorRegions, detectorRegions.shards.count == shards.count else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Exact detector aggregate was released; reload the source")
    }
    encoder.setComputePipelineState(
      update.aggregateEntryCount < 32 ? detectorPipeline : pixelLaneDetectorPipeline)
    for index in shards.indices {
      var parameters = CompactDetectorParameters(
        scanCount: UInt32(metadata.scansPerShard), tileCount: UInt32(metadata.scansPerShard / 32),
        entryCount: UInt32(update.aggregateEntryCount),
        outputOffset: UInt32(index * metadata.scansPerShard),
        mode: 0, scanTile: 32, headerWordsPerPixel: 0, headerEncoding: 0, payloadLayout: 0)
      encoder.setBuffer(detectorRegions.shards[index].payload, offset: 0, index: 0)
      encoder.setBuffer(detectorRegions.shards[index].descriptors, offset: 0, index: 1)
      encoder.setBuffer(detectorRegions.entries, offset: 0, index: 2)
      encoder.setBuffer(detectorRegions.intermediate, offset: 0, index: 3)
      encoder.setBuffer(detectorOutputs[update.nextOutput], offset: 0, index: 4)
      encoder.setBytes(&parameters, length: MemoryLayout.stride(ofValue: parameters), index: 5)
      encoder.setBuffer(detectorRegions.intermediate, offset: 0, index: 6)
      encoder.dispatchThreadgroups(
        MTLSize(width: (metadata.scansPerShard + 31) / 32, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1))
    }
  }

  /// Copy one canonical prepared map into the active detector output.
  /// Its checksum verification status is recorded in `loadMetrics`.
  @discardableResult
  public func activatePreparedDetectorProduct(
    _ name: MetalCompactH5PreparedDetectorProductName
  ) throws -> MetalCompactH5DetectorMetrics {
    guard !isReleased, let product = preparedDetectorProducts[name.rawValue] else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Prepared detector product \(name.rawValue) is unavailable for this source."
      )
    }
    return try updateVirtualDetector(mask: product.mask)
  }

  /// Read the last completely published virtual-detector map in scan-row order.
  ///
  /// Use `snapshotVirtualDetectors` for Metal display without an array readback.
  public func virtualDetectorValues() throws -> [UInt32] {
    guard !isReleased, detectorOutputs.count == 2 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source has been released. Load it again before "
          + "reading a virtual detector."
      )
    }
    return Self.u32Values(
      detectorOutputs[activeDetectorOutput],
      count: metadata.scanCount
    )
  }

  /// Copy completed exact detector maps into independently owned Metal buffers.
  ///
  /// The returned buffers contain scan-row-major uint32 counts and remain
  /// unchanged by later source updates or release. All copies share one GPU
  /// submission; no scientific image is read into a CPU array. Callers must
  /// serialize this operation with updates and release on these sources, and
  /// release old snapshots when their display consumers finish.
  ///
  /// ```swift
  /// let images = try MetalCompactH5ResidentSource.snapshotVirtualDetectors(tilts)
  /// ```
  public static func snapshotVirtualDetectors(
    _ sources: [MetalCompactH5ResidentSource]
  ) throws -> [MTLBuffer] {
    guard let first = sources.first else { return [] }
    guard
      sources.allSatisfy({
        !$0.isReleased && $0.detectorOutputs.count == 2
          && ($0.device as AnyObject) === (first.device as AnyObject)
      })
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Detector snapshots require live sources on the same Metal device. "
          + "Load the sources together before requesting display images."
      )
    }
    let outputs = try allocateDetectorSnapshots(sources)
    guard let command = first.queue.makeCommandBuffer(),
      let encoder = command.makeBlitCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode detector image snapshots."
      )
    }
    for (source, output) in zip(sources, outputs) {
      encoder.copy(
        from: source.detectorOutputs[source.activeDetectorOutput], sourceOffset: 0,
        to: output, destinationOffset: 0, size: output.length
      )
    }
    encoder.endEncoding()
    try Self.complete(command, operation: "detector image snapshots")
    return outputs
  }

  private static func allocateDetectorSnapshots(
    _ sources: [MetalCompactH5ResidentSource]
  ) throws -> [MTLBuffer] {
    try sources.map { source in
      guard
        let buffer = source.device.makeBuffer(
          length: source.metadata.scanCount * MemoryLayout<UInt32>.stride,
          options: .storageModeShared
        )
      else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Metal could not allocate a detector image snapshot."
        )
      }
      return buffer
    }
  }

  /// Return the prepared centered row and column DPC maps without rescanning the cube.
  public func preparedDPCValues() throws -> (row: [Float], column: [Float])? {
    guard !isReleased else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source has been released. Load it again before "
          + "reading prepared DPC maps."
      )
    }
    guard preparedDPCOutputs.count == 2 else { return nil }
    return (
      row: Self.floatValues(preparedDPCOutputs[0], count: metadata.scanCount),
      column: Self.floatValues(preparedDPCOutputs[1], count: metadata.scanCount)
    )
  }

  /// Return prepared exact total and row/column detector moments.
  /// Their checksum verification status is recorded in `loadMetrics`.
  public func preparedDPCMomentValues() throws -> MetalCompactH5ExactDPCMoments? {
    guard !isReleased else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source has been released. Load it again before reading DPC moments."
      )
    }
    guard let prepared = metadata.preparedDPCMoments,
      let preparedDPCMomentBuffer
    else { return nil }
    let words = preparedDPCMomentBuffer.contents().bindMemory(
      to: UInt32.self,
      capacity: prepared.scanCount * 8
    )
    var total = [UInt64](repeating: 0, count: prepared.scanCount)
    var row = [UInt64](repeating: 0, count: prepared.scanCount)
    var column = [UInt64](repeating: 0, count: prepared.scanCount)
    for scan in 0..<prepared.scanCount {
      let base = scan * 8
      total[scan] = Self.u64(low: words[base], high: words[base + 1])
      row[scan] = Self.u64(low: words[base + 2], high: words[base + 3])
      column[scan] = Self.u64(low: words[base + 4], high: words[base + 5])
    }
    return MetalCompactH5ExactDPCMoments(
      total: total,
      detectorRowMoment: row,
      detectorColumnMoment: column,
      sourceIdentitySHA256: metadata.sourceIdentitySHA256,
      detectorMaskSHA256: prepared.detectorMaskSHA256
    )
  }

  /// Borrow one persistent prepared DPC buffer for zero-copy Metal display.
  public func preparedDPCDisplayBuffer(
    component: MetalCompactH5DPCComponent
  ) throws -> MTLBuffer? {
    guard !isReleased else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source has been released. Load it again before "
          + "borrowing a prepared DPC display buffer."
      )
    }
    guard preparedDPCOutputs.count == 2 else { return nil }
    switch component {
    case .row:
      return preparedDPCOutputs[0]
    case .column:
      return preparedDPCOutputs[1]
    }
  }

  /// Decode every exact working value in scan-major order and hash it as u8.
  ///
  /// One shard-sized shared buffer is reused, so this validation never creates
  /// the complete dense logical volume. The method is an explicit audit pass,
  /// not an interaction or prepared-reopen timing endpoint.
  public func hashLogicalWorkingU8() throws -> MetalCompactH5LogicalHashMetrics {
    guard !isReleased, let excluded, shards.count == metadata.shardCount else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact resident source has been released. Load it again before "
          + "hashing its logical working values."
      )
    }
    guard maximumWidths.allSatisfy({ $0 <= 8 }) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The compact source contains values wider than u8; request a future u16 audit."
      )
    }
    let outputBytes = metadata.scansPerShard * metadata.detectorPixelCount
    guard outputBytes > 0, outputBytes.isMultiple(of: 4),
      outputBytes <= device.maxBufferLength,
      let outputWords = UInt32(exactly: outputBytes / 4),
      let output = device.makeBuffer(
        length: outputBytes,
        options: .storageModeShared
      )
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "compact logical-hash staging",
        bytes: UInt64(outputBytes)
      )
    }
    var hasher = SHA256()
    var gpuMilliseconds = 0.0
    let wallStart = ContinuousClock.now
    for shardIndex in shards.indices {
      var parameters = CompactFullDecodeParameters(
        scanCount: UInt32(metadata.scansPerShard),
        pixelCount: UInt32(metadata.detectorPixelCount),
        tileCount: UInt32(
          (metadata.scansPerShard + metadata.scanTile - 1) / metadata.scanTile
        ),
        scanTile: UInt32(metadata.scanTile),
        headerWordsPerPixel: headerWordsPerPixel,
        headerEncoding: headerEncoding,
        outputWordCount: outputWords,
        payloadLayout: payloadLayout
      )
      guard let command = queue.makeCommandBuffer(),
        let encoder = command.makeComputeCommandEncoder()
      else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Metal could not encode compact full-volume hash shard \(shardIndex)."
        )
      }
      encoder.setComputePipelineState(fullDecodePipeline)
      encoder.setBuffer(shards[shardIndex].payload, offset: 0, index: 0)
      encoder.setBuffer(shards[shardIndex].descriptors, offset: 0, index: 1)
      encoder.setBuffer(excluded, offset: 0, index: 2)
      encoder.setBuffer(output, offset: 0, index: 3)
      encoder.setBytes(
        &parameters,
        length: MemoryLayout.stride(ofValue: parameters),
        index: 4
      )
      encoder.dispatchThreads(
        MTLSize(width: Int(outputWords), height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
      )
      encoder.endEncoding()
      try Self.complete(command, operation: "full-volume hash shard \(shardIndex)")
      gpuMilliseconds += max(0, command.gpuEndTime - command.gpuStartTime) * 1_000
      hasher.update(
        data: Data(
          bytesNoCopy: output.contents(),
          count: outputBytes,
          deallocator: .none
        )
      )
    }
    return MetalCompactH5LogicalHashMetrics(
      sha256: hasher.finalize().map { String(format: "%02x", $0) }.joined(),
      logicalBytes: UInt64(metadata.scanCount * metadata.detectorPixelCount),
      stagingBytes: UInt64(outputBytes),
      wallMilliseconds: Self.milliseconds(from: wallStart),
      gpuMilliseconds: gpuMilliseconds
    )
  }

  /// Release all payload, descriptor, and interaction buffers owned by this source.
  ///
  /// The caller serializes release with interactions. Metadata and load metrics
  /// remain readable so a file-switch receipt can retain the completed source's
  /// provenance without retaining its Metal allocation.
  public func releaseResidentStorage() {
    guard !isReleased else { return }
    residencyLease?.end()
    residencyLease = nil
    shards.removeAll(keepingCapacity: false)
    detectorOutputs.removeAll(keepingCapacity: false)
    detectorEntryBuffer = nil
    excluded = nil
    diffractionOutput = nil
    detectorSumOutput = nil
    preparedDPCMomentBuffer = nil
    preparedDPCOutputs.removeAll(keepingCapacity: false)
    preparedDetectorProducts.removeAll(keepingCapacity: false)
    detectorRegions = nil
    detectorMask.removeAll(keepingCapacity: false)
    detectorSumReady = false
    isReleased = true
  }

  private static func u32Values(_ buffer: MTLBuffer, count: Int) -> [UInt32] {
    Array(
      UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: UInt32.self, capacity: count),
        count: count
      )
    )
  }

  private static func floatValues(_ buffer: MTLBuffer, count: Int) -> [Float] {
    Array(
      UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: Float.self, capacity: count),
        count: count
      )
    )
  }

  private static func u64Values(_ buffer: MTLBuffer, count: Int) -> [UInt64] {
    Array(
      UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: UInt64.self, capacity: count),
        count: count
      )
    )
  }

  fileprivate static func complete(
    _ command: MTLCommandBuffer,
    operation: String,
    hostStamp: ((String) -> Void)? = nil
  ) throws {
    hostStamp?("before_commit_ms")
    command.commit()
    hostStamp?("after_commit_ms")
    command.waitUntilCompleted()
    hostStamp?("after_wait_ms")
    if let error = command.error {
      throw Metal4DSTEMStreamingIOError.commandFailed(
        "Compact HDF5 \(operation) failed: \(error.localizedDescription)"
      )
    }
    guard command.status == .completed else {
      throw Metal4DSTEMStreamingIOError.commandFailed(
        "Compact HDF5 \(operation) ended with Metal status \(command.status.rawValue)."
      )
    }
  }

  fileprivate static func milliseconds(
    from start: ContinuousClock.Instant
  ) -> Double {
    let duration = start.duration(to: .now)
    return Double(duration.components.seconds) * 1_000
      + Double(duration.components.attoseconds) / 1.0e15
  }

  private static func u64(low: UInt32, high: UInt32) -> UInt64 {
    UInt64(UInt32(littleEndian: low))
      | (UInt64(UInt32(littleEndian: high)) << 32)
  }
}

/// Fail-closed bounded loader for compact 4D-STEM HDF5 user-block payloads.
public enum MetalCompactH5Loader {
  /// Finish the existing interaction object from freshly verified private blocks.
  /// There is no file publication, payload hashing, or payload upload in this step.
  static func residentFromOriginal(
    _ packed: OriginalPackedBuffers, device: MTLDevice, started: ContinuousClock.Instant,
    allocatedBefore: UInt64, maximumAdditionalBytes: UInt64?, shouldCancel: () -> Bool
  ) throws -> MetalCompactH5ResidentSource {
    let dataset = packed.dataset
    let scans = dataset.scanRows * dataset.scanCols
    let pixels = dataset.detectorRows * dataset.detectorCols
    let working = packed.maximum <= 255 ? "uint8" : "uint16"
    let identity = dataset.sourceIdentitySHA256!
    func digest(_ data: Data) -> String {
      SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
    func buffer(_ size: Int) throws -> MTLBuffer {
      guard let result = device.makeBuffer(length: size, options: .storageModeShared) else {
        throw invalid(
          "Cannot allocate original-resident interaction buffers; release another acquisition")
      }
      memset(result.contents(), 0, result.length)
      return result
    }
    let maximumValue: UInt64 = packed.maximum <= 255 ? 255 : 65535
    let totalBound = UInt64(pixels) * maximumValue
    let rowBound = UInt64(pixels * (dataset.detectorRows - 1) / 2) * maximumValue
    let columnBound = UInt64(pixels * (dataset.detectorCols - 1) / 2) * maximumValue
    let maskSHA = digest(Data(repeating: 1, count: pixels))
    let calibrationJSON = try JSONSerialization.data(
      withJSONObject: [
        "source_identity": identity, "center_row": packed.calibration.0,
        "center_column": packed.calibration.1, "radius": packed.calibration.2,
      ], options: [.sortedKeys])
    let manifest = try JSONSerialization.data(
      withJSONObject: [
        "schema": "original-hdf5-resident/v1", "source_identity": identity,
        "shape": [dataset.scanRows, dataset.scanCols, dataset.detectorRows, dataset.detectorCols],
        "source_dtype": dataset.sourceDtype, "working_dtype": working,
        "every_count_roundtrip_verified": true, "logical_hash_computed": false,
        "bad_pixel_policy": "preserve_all_source_counts", "scan_bin": 1, "detector_bin": 1,
      ], options: [.sortedKeys])
    let prepared = MetalCompactH5PreparedDPCMoments(
      fileOffset: 0, fileBytes: UInt64(packed.moments.count), sha256: digest(packed.moments),
      workingLogicalSHA256: nil, workingDtype: working, detectorMaskSHA256: maskSHA,
      scanCount: scans, selectedDetectorPixels: pixels, detectorColumns: dataset.detectorCols,
      totalBound: totalBound, rowMomentBound: rowBound, columnMomentBound: columnBound,
      narrowInteger: totalBound <= UInt32.max,
      narrowProducts: max(rowBound, columnBound) <= UInt32.max)
    let payloadBytes = packed.shards.reduce(UInt64(0)) {
      $0 + UInt64($1.payload.length + $1.headers.length)
    }
    let metadata = MetalCompactH5Metadata(
      sourceURL: URL(fileURLWithPath: dataset.masterPath ?? dataset.dataFiles[0]),
      sourceBytes: UInt64(dataset.sourceBytes), schema: "quantem.gpu.packed-detector-h5/v3",
      payloadCodec: "direct-bitpacked-u32", sourceDtype: dataset.sourceDtype,
      manifestSHA256: digest(manifest), workingDtype: working, embeddedScientificSemantics: true,
      scanRows: dataset.scanRows, scanColumns: dataset.scanCols,
      detectorRows: dataset.detectorRows, detectorColumns: dataset.detectorCols,
      scansPerShard: packed.frames, scanTile: 32, payloadChunkBytes: 0,
      sourceIdentitySHA256: identity, sourceRawLogicalSHA256: nil, workingLogicalSHA256: nil,
      detectorMaskSHA256: maskSHA, maskedDetectorPixelsSHA256: nil, maskedDetectorRawValues: nil,
      rawAccessMode: "exact_no_exclusions",
      detectorCalibration: MetalCompactH5DetectorCalibration(
        detectorCenterRow: packed.calibration.0, detectorCenterColumn: packed.calibration.1,
        brightFieldRadius: packed.calibration.2, dpcRotationDegrees: nil,
        dpcComponentOrderExchanged: nil,
        method: "mean-DP half-p99 threshold; automatic initial detector, not angular calibration"),
      detectorCalibrationSchema: "quantem.gpu.detector-calibration/v1",
      detectorCalibrationSHA256: digest(calibrationJSON), preparedDPCMoments: prepared,
      preparedDetectorProducts: nil, excludedDetectorPixels: [], shardCount: packed.shards.count,
      residentBytes: payloadBytes + UInt64(packed.moments.count))
    let momentBuffer = try packed.moments.withUnsafeBytes { raw -> MTLBuffer in
      guard
        let result = device.makeBuffer(
          bytes: raw.baseAddress!, length: raw.count, options: .storageModeShared)
      else {
        throw invalid("Cannot retain original DPC sums")
      }
      return result
    }
    let dpc = try primeDPC(
      moments: momentBuffer, prepared: prepared, metadata: metadata, device: device)
    let excluded = try buffer(pixels * 4)
    let entries = try buffer(pixels * 8)
    let outputA = try buffer(scans * 4)
    let outputB = try buffer(scans * 4)
    let diffraction = try buffer(pixels * 4)
    let detectorSum = try buffer(pixels * 8)
    packed.detectorSum.withUnsafeBytes {
      detectorSum.contents().copyMemory(from: $0.baseAddress!, byteCount: $0.count)
    }
    var interactionBytes = UInt64(pixels * 24 + scans * 16)
    var totalBytes = metadata.residentBytes + interactionBytes
    var planned = totalBytes + packed.stagingBytes
    if let maximumAdditionalBytes, planned > maximumAdditionalBytes {
      throw invalid(
        "Original resident plus interaction buffers exceeds the memory budget; open fewer acquisitions"
      )
    }
    guard !shouldCancel() else { throw Metal4DSTEMStreamingIOError.cancelled }
    guard let queue = device.makeCommandQueue() else {
      throw invalid("Cannot create resident interaction queue")
    }
    let library = try Metal4DSTEMKernels.makeCompactH5Library(device: device)
    func kernel(_ name: String) throws -> MTLComputePipelineState {
      if name == "compact_h5_detector_update_planar_quad_vector",
        compactKernelOption("RAW_FIXED_THREADS", byDefault: false)
      {
        guard let function = library.makeFunction(name: name) else {
          throw invalid("Missing exact detector kernel; rebuild the Metal resources and retry")
        }
        let descriptor = MTLComputePipelineDescriptor()
        descriptor.computeFunction = function
        descriptor.maxTotalThreadsPerThreadgroup = 128
        descriptor.threadGroupSizeIsMultipleOfThreadExecutionWidth = true
        if compactKernelOption("RAW_REQUIRED_THREADS", byDefault: false) {
          guard #available(macOS 26.0, iOS 26.0, *) else {
            throw invalid("Required-thread profiling needs macOS or iOS 26 or later")
          }
          descriptor.requiredThreadsPerThreadgroup = MTLSize(width: 128, height: 1, depth: 1)
        }
        let result = try device.makeComputePipelineState(
          descriptor: descriptor, options: [], reflection: nil)
        guard result.threadExecutionWidth == 32, result.maxTotalThreadsPerThreadgroup >= 128 else {
          throw invalid(
            "Fixed-thread profiling requires 32-lane SIMD groups and 128-thread dispatches")
        }
        return result
      }
      return try pipeline(library: library, name: name, device: device)
    }
    let selected = try kernel(Metal4DSTEMKernels.compactH5SelectedDiffractionFunction)
    let detector = try kernel(Metal4DSTEMKernels.compactH5DetectorUpdateFunction)
    let pixelLane = try kernel("compact_h5_detector_update_pixel_lanes")
    let originalPlanarILP =
      ProcessInfo.processInfo.environment["COMPACT_RAW_PLANE_ILP"] == "1"
      ? try kernel("compact_h5_detector_update_planar_ilp") : nil
    let quadVector = compactKernelOption("RAW_QUAD_VECTOR", byDefault: true)
    let planarVariant =
      compactKernelOption("RAW_CONSTANT_WIDE", byDefault: false)
      ? "compact_h5_detector_update_planar_quad_constant"
      : "compact_h5_detector_update_planar_quad_vector"
    let originalPlanarScan =
      quadVector
      ? try kernel(planarVariant)
      : (ProcessInfo.processInfo.environment["COMPACT_RAW_SCAN_COOPERATIVE"] == "1"
        ? try kernel("compact_h5_detector_update_planar_scan_cooperative") : nil)
    let fullDecode = try kernel(Metal4DSTEMKernels.compactH5FullDecodeU8Function)
    let sum = try kernel(Metal4DSTEMKernels.compactH5DetectorSumFunction)
    let originalPayloadLayout = packed.payloadLayout
    let residentShards = packed.shards.map {
      CompactResidentShard(payload: $0.payload, descriptors: $0.headers)
    }
    var originalResidencyLease: CompactResidencyLease?
    if compactKernelOption("RESIDENCY_SETS", byDefault: true) {
      if #available(macOS 15.0, iOS 18.0, *) {
        let residencyStarted = ContinuousClock.now
        let stable =
          residentShards.flatMap { [$0.payload, $0.descriptors] }
          + [
            momentBuffer, dpc.row, dpc.column, excluded, entries,
            outputA, outputB, diffraction, detectorSum,
          ]
        let lease = try CompactResidencyLease(device: device, buffers: stable)
        // This hint may not enlarge the resident budget, including
        // driver-reported bookkeeping beyond existing resource allocations.
        let admitted = lease.footprintExcessBytes == 0
        if admitted {
          originalResidencyLease = lease
        } else {
          // Residency is a scheduling hint, not a scientific representation.
          // Decline it when Metal's reported footprint exceeds the existing
          // resources; valid packed data must still load within its budget.
          lease.end()
        }
        let record: [String: Any] = [
          "phase": "original_residency_set_prepared", "advisory_not_pinned": true,
          "admitted": admitted,
          "allocation_count": lease.allocationCount, "set_allocated_bytes": lease.allocatedBytes,
          "footprint_excess_bytes": lease.footprintExcessBytes,
          "host_preparation_ms": milliseconds(from: residencyStarted),
          "source_identity": identity,
        ]
        if var json = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) {
          json.append(10)
          FileHandle.standardError.write(json)
        }
      }
    }
    var detectorRegions: CompactDetectorRegions?
    var auxiliaryScratch: UInt64 = 0
    var auxiliaryGPU = 0.0
    if CompactDetectorRegions.enabled {
      let allocated = UInt64(device.currentAllocatedSize)
      // Original packing has returned and drained its commands. Its scratch
      // peak remains in `planned`, but is not live during this separate phase.
      // Account for every allocation still held, including driver retirement.
      let heldAdditional = max(
        totalBytes, allocated >= allocatedBefore ? allocated - allocatedBefore : 0)
      let available =
        maximumAdditionalBytes.map { $0 > heldAdditional ? $0 - heldAdditional : 0 } ?? 0
      let built = try CompactDetectorRegions.build(
        shards: residentShards, metadata: metadata, headerWords: packed.headerStride,
        headerEncoding: packed.maximum <= 255 ? 1 : 2, payloadLayout: originalPayloadLayout,
        device: device, queue: queue,
        library: library, availableBytes: available, shouldCancel: shouldCancel)
      detectorRegions = built.auxiliary
      auxiliaryScratch = built.scratchBytes
      auxiliaryGPU = built.gpuMilliseconds
      let auxiliaryBytes = detectorRegions?.bytes ?? 0
      // Keep the higher of the two independently bounded phase peaks. Neither
      // phase may consume memory beyond the caller's unchanged admission limit.
      planned = max(planned, heldAdditional + built.peakBytes)
      interactionBytes += auxiliaryBytes
      totalBytes += auxiliaryBytes
    }
    if let maximumAdditionalBytes, planned > maximumAdditionalBytes {
      throw invalid(
        "Original resident preparation exceeds the memory budget; release another acquisition and retry"
      )
    }
    var metrics = MetalCompactH5LoadMetrics(
      nativeCacheStatus: "originalDirect", nativeCacheBytes: 0,
      nativeCacheDescriptorSHA256Checks: 0,
      plannedAdditionalBytes: planned, sourceReadPolicy: "systemDefault", maximumInFlightShards: 1,
      shardPipelineMilliseconds: milliseconds(from: started), metadataMilliseconds: 0,
      sourceReadMilliseconds: packed.readSeconds * 1000, descriptorPreparationMilliseconds: 0,
      gpuPreparationMilliseconds: packed.packingSeconds * 1000 + auxiliaryGPU,
      gpuDecodeMilliseconds: packed.decodeSeconds * 1000,
      decodedIntegrityMilliseconds: 0, privateUploadMilliseconds: 0,
      preparedDPCReadMilliseconds: 0, preparedDPCAuthenticationMilliseconds: 0,
      preparedDPCPrimeMilliseconds: dpc.primeMilliseconds,
      preparedDetectorProductReadMilliseconds: 0,
      preparedDetectorProductAuthenticationMilliseconds: 0,
      totalMilliseconds: milliseconds(from: started), residentBytes: metadata.residentBytes,
      maximumTransientBytes: max(packed.stagingBytes, auxiliaryScratch),
      deviceAllocatedBytesBefore: allocatedBefore,
      deviceAllocatedBytesAfter: UInt64(device.currentAllocatedSize), decodedShardSHA256Checks: 0,
      checksumsVerified: false, decodedPayloadCopyBytes: 0, mappedAuthenticationBytes: 0,
      preparedDPCBytes: UInt64(packed.moments.count), preparedDetectorProductBytes: 0,
      interactionResidentBytes: interactionBytes, totalResidentBytes: totalBytes)
    metrics.reusedPreparedDPC = packed.reusedDPC
    metrics.gpuDecodeAndHeaderMilliseconds = packed.decodeAndHeaderSeconds * 1000
    metrics.gpuDecodeAndPackingMilliseconds = packed.combinedDecodePackingSeconds.map { $0 * 1000 }
    let resident = MetalCompactH5ResidentSource(
      metadata: metadata, loadMetrics: metrics, device: device, queue: queue,
      selectedPipeline: selected, detectorPipeline: detector, pixelLaneDetectorPipeline: pixelLane,
      planarILPDetectorPipeline: originalPlanarILP,
      planarScanCooperativePipeline: originalPlanarScan,
      fullDecodePipeline: fullDecode, detectorSumPipeline: sum,
      headerEncoding: packed.maximum <= 255 ? 1 : 2, payloadLayout: originalPayloadLayout,
      headerWordsPerPixel: UInt32(packed.headerStride),
      shards: residentShards,
      excluded: excluded, maximumWidths: packed.maximumWidths, detectorOutputs: [outputA, outputB],
      detectorEntryBuffer: entries, diffractionOutput: diffraction, detectorSumOutput: detectorSum,
      preparedDPCMomentBuffer: momentBuffer, preparedDPCOutputs: [dpc.row, dpc.column],
      preparedDetectorProducts: [:], residencyLease: originalResidencyLease,
      detectorRegions: detectorRegions, planarVariant: quadVector ? planarVariant : "other")
    // Pipeline creation may take time after the earlier cancellation poll.
    // Do not publish an interaction object if cancellation arrived meanwhile.
    guard !shouldCancel() else {
      resident.releaseResidentStorage()
      throw Metal4DSTEMStreamingIOError.cancelled
    }
    resident.markOriginalDetectorSumReady()
    return resident
  }
  /// Read structurally validated catalog metadata without loading the payload.
  ///
  /// This creates no Metal resources and does not authenticate payload bytes or
  /// establish resident capabilities. `load` revalidates the file on every open.
  public static func inspect(sourceURL: URL) throws -> MetalCompactH5Metadata {
    try parse(sourceURL: sourceURL).metadata
  }

  private static let containerMagic: [UInt8] = [
    0x51, 0x47, 0x50, 0x55, 0x48, 0x35, 0x00, 0x01,
  ]
  private static let indexMagicV1: [UInt8] = [
    0x51, 0x47, 0x49, 0x58, 0x00, 0x00, 0x00, 0x01,
  ]
  private static let indexMagicV3: [UInt8] = [
    0x51, 0x47, 0x49, 0x58, 0x00, 0x00, 0x00, 0x03,
  ]

  /// Decode, validate, and publish one compact source in private Metal buffers.
  ///
  /// `nativeCacheURL` optionally reuses a cache previously saved from this exact
  /// source. Missing/stale caches use the original decode path; corrupt caches
  /// throw and should be removed/rebuilt by the caller. This call never creates
  /// a cache. The default authentication remains bounded and sequential;
  /// `parallelMapped` requires budget for additional file-backed residency.
  /// `maximumAdditionalBytes`, when supplied, rejects an oversized load before
  /// any Metal queue or buffer allocation. It covers requested resident buffers,
  /// worst-shard staging, and mapped authentication bytes. The caller must reserve
  /// additional headroom for driver allocations, OS caches, and app products.
  /// Phase times sum shard work; boundedConcurrent overlaps these durations.
  /// sourceReadPolicy applies to source metadata and payload descriptors. A
  /// nondefault policy cannot be combined with native cache or mapped loading.
  /// `verifyChecksums: false` explicitly trusts local payload bytes. It skips
  /// payload, cached-descriptor, and prepared-product SHA checks, not structural
  /// bounds, decoder-status, dtype, or metadata validation. Such a load is not
  /// checksum-verified. The default remains verified for existing callers.
  public static func load(
    sourceURL: URL,
    device: MTLDevice,
    authenticationPolicy: MetalCompactH5AuthenticationPolicy = .boundedSequential,
    nativeCacheURL: URL? = nil,
    maximumAdditionalBytes: UInt64? = nil,
    sourceReadPolicy: MetalCompactH5SourceReadPolicy = .systemDefault,
    verifyChecksums: Bool = true,
    shouldCancel: () -> Bool = { false }
  ) throws -> MetalCompactH5ResidentSource {
    let totalStart = ContinuousClock.now
    let allocatedBefore = UInt64(device.currentAllocatedSize)
    let metadataStart = ContinuousClock.now
    guard
      sourceReadPolicy == .systemDefault
        || (nativeCacheURL == nil && authenticationPolicy != .parallelMapped)
    else {
      throw invalid(
        "Source avoidCaching reads require nativeCacheURL nil and bounded authentication. "
          + "This policy cannot control native-cache or memory-mapped reads."
      )
    }
    let index = try parse(sourceURL: sourceURL, readPolicy: sourceReadPolicy)
    var cacheStatus = nativeCacheURL == nil ? "notRequested" : "miss"
    var cacheFile: FileHandle?
    var cacheStamp: String?
    var nativeCache: CompactNativeCache?
    var loadingShards = index.shards
    if let nativeCacheURL, FileManager.default.fileExists(atPath: nativeCacheURL.path) {
      let file = try FileHandle(forReadingFrom: nativeCacheURL)
      let initialStamp = try CompactNativeCache.stamp(file)
      let cache = try CompactNativeCache.read(from: file)
      let signature = try CompactNativeCache.signature(sourceURL: sourceURL)
      if cache.sourceSignature != signature
        || cache.sourceManifestSHA256 != index.metadata.manifestSHA256
      {
        cacheStatus = "stale"
        try file.close()
      } else {
        guard index.storageLayout == .lz4V1,
          index.metadata.embeddedScientificSemantics,
          index.metadata.workingDtype == "uint16",
          cache.shards.count == index.shards.count
        else { throw CompactNativeCache.invalid("source contract does not match the cache.") }
        loadingShards = try zip(index.shards, cache.shards).map { original, cached in
          guard cached.payloadBytes == original.decodedBytes,
            cached.payloadSHA256 == original.decodedSHA256,
            cached.descriptorsBytes == UInt64(original.descriptorCount) * 4
          else { throw CompactNativeCache.invalid("shard contract differs from source.") }
          return CompactH5ShardRecord(
            payloadOffset: cached.payloadOffset, payloadBytes: cached.payloadBytes,
            lengthsOffset: 0, lengthsBytes: 0,
            widthsOffset: cached.descriptorsOffset, widthsBytes: cached.descriptorsBytes,
            decodedBytes: cached.payloadBytes, descriptorCount: original.descriptorCount,
            chunkCount: 0, decodedSHA256: original.decodedSHA256,
            descriptorsSHA256: cached.descriptorsSHA256
          )
        }
        nativeCache = cache
        cacheFile = file
        cacheStamp = initialStamp
        cacheStatus = "hit"
      }
    }
    defer { try? cacheFile?.close() }
    var plannedAdditionalBytes = try plannedAdditionalBytes(
      index: index, loadingShards: loadingShards, nativeCache: nativeCache,
      authenticationPolicy: authenticationPolicy
    )
    if let maximumAdditionalBytes, plannedAdditionalBytes > maximumAdditionalBytes {
      throw invalid(
        "Compact loading requires \(plannedAdditionalBytes) additional bytes for "
          + "resident buffers, bounded staging, and authentication mapping; the "
          + "available loader budget is \(maximumAdditionalBytes) bytes. Release "
          + "other residents or use boundedSequential authentication before retrying. "
          + "No Metal storage was allocated. Reserve app and driver headroom separately."
      )
    }
    let metadataMilliseconds = milliseconds(from: metadataStart)
    guard !shouldCancel() else { throw Metal4DSTEMStreamingIOError.cancelled }
    guard let queue = device.makeCommandQueue() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not create the compact HDF5 command queue."
      )
    }
    let library = try Metal4DSTEMKernels.makeCompactH5Library(device: device)
    let metadataKernels =
      index.storageLayout == .lz4V1 && nativeCache == nil
      ? try CompactH5MetadataKernels(device: device, library: library) : nil
    let decode =
      metadataKernels == nil
      ? nil
      : try pipeline(
        library: library, name: "compact_h5_lz4_decode_simd32", device: device)
    if let decode {
      guard decode.threadExecutionWidth == 32, decode.maxTotalThreadsPerThreadgroup >= 256 else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Compact LZ4 decoding requires 32-lane SIMD groups and 256-thread groups.")
      }
    }
    let validateDescriptors = try pipeline(
      library: library,
      name: Metal4DSTEMKernels.compactH5ValidateDescriptorsFunction,
      device: device
    )
    let selected = try pipeline(
      library: library,
      name: Metal4DSTEMKernels.compactH5SelectedDiffractionFunction,
      device: device
    )
    let detector = try pipeline(
      library: library,
      name: Metal4DSTEMKernels.compactH5DetectorUpdateFunction,
      device: device
    )
    let pixelLaneDetector = try pipeline(
      library: library,
      name: "compact_h5_detector_update_pixel_lanes",
      device: device
    )
    let planarILPDetector =
      ProcessInfo.processInfo.environment["COMPACT_RAW_PLANE_ILP"] == "1"
      ? try pipeline(
        library: library, name: "compact_h5_detector_update_planar_ilp", device: device) : nil
    let planarScanCooperative =
      ProcessInfo.processInfo.environment["COMPACT_RAW_SCAN_COOPERATIVE"] == "1"
      ? try pipeline(
        library: library, name: "compact_h5_detector_update_planar_scan_cooperative", device: device
      ) : nil
    let planarFusedDetector =
      ProcessInfo.processInfo.environment["COMPACT_RAW_AUX_FUSED"] == "1"
      ? try pipeline(
        library: library, name: "compact_h5_detector_update_planar_fused", device: device) : nil
    let fullDecode = try pipeline(
      library: library,
      name: Metal4DSTEMKernels.compactH5FullDecodeU8Function,
      device: device
    )
    let detectorSum = try pipeline(
      library: library,
      name: Metal4DSTEMKernels.compactH5DetectorSumFunction,
      device: device
    )

    let descriptor = try openSource(sourceURL, readPolicy: sourceReadPolicy)
    defer { Darwin.close(descriptor) }

    let excludedSet = Set(index.metadata.excludedDetectorPixels)
    var residentShards: [CompactResidentShard] = []
    residentShards.reserveCapacity(index.shards.count)
    guard
      let maximumWidthBuffer = device.makeBuffer(
        length: index.metadata.detectorPixelCount * MemoryLayout<UInt32>.stride,
        options: .storageModeShared
      )
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "compact maximum-width validation",
        bytes: UInt64(index.metadata.detectorPixelCount * 4)
      )
    }
    memset(maximumWidthBuffer.contents(), 0, maximumWidthBuffer.length)
    var maximumWidths = [UInt8](
      repeating: 0,
      count: index.metadata.detectorPixelCount
    )
    var sourceReadMilliseconds = 0.0
    var descriptorPreparationMilliseconds = 0.0
    var gpuPreparationMilliseconds = 0.0
    var gpuDecodeMilliseconds = 0.0
    var decodedIntegrityMilliseconds = 0.0
    var privateUploadMilliseconds = 0.0
    var maximumTransientBytes: UInt64 = 0
    var mappedAuthenticationBytes: UInt64 = 0
    let payloadsPreauthenticated: Bool
    if index.storageLayout == .directV3 || nativeCache != nil,
      authenticationPolicy == .parallelMapped, verifyChecksums
    {
      let authenticationStart = ContinuousClock.now
      try authenticateDirectPayloads(
        fileDescriptor: cacheFile?.fileDescriptor ?? descriptor,
        fileBytes: nativeCache?.fileBytes ?? index.metadata.sourceBytes,
        shards: loadingShards
      )
      decodedIntegrityMilliseconds += milliseconds(from: authenticationStart)
      mappedAuthenticationBytes = nativeCache?.fileBytes ?? index.metadata.sourceBytes
      payloadsPreauthenticated = true
    } else {
      payloadsPreauthenticated = false
    }
    guard !shouldCancel() else { throw Metal4DSTEMStreamingIOError.cancelled }

    let maximumInFlightShards =
      authenticationPolicy == .boundedConcurrent
      ? min(3, loadingShards.count) : 1
    let shardPipelineStart = ContinuousClock.now
    let payloadDescriptor = cacheFile?.fileDescriptor ?? descriptor
    let isDirect = index.storageLayout == .directV3 || nativeCache != nil
    let shardRecords = loadingShards
    let concurrentWidths = CompactConcurrentWidthValidation(buffer: maximumWidthBuffer)
    let concurrentDevice = CompactConcurrentMetalHandle(value: device)
    let concurrentQueue = CompactConcurrentMetalHandle(value: queue)
    let concurrentValidationPipeline = CompactConcurrentMetalHandle(value: validateDescriptors)
    let concurrentDecode = CompactConcurrentMetalHandle(value: decode)
    for first in stride(from: 0, to: loadingShards.count, by: maximumInFlightShards) {
      // Caller cancellation is polled only on the calling thread, between
      // bounded windows. All workers finish before any error/cancellation escapes.
      guard !shouldCancel() else { throw Metal4DSTEMStreamingIOError.cancelled }
      let count = min(maximumInFlightShards, loadingShards.count - first)
      let results = CompactShardLoadCollector(count: count)
      DispatchQueue.concurrentPerform(iterations: count) { slot in
        let result: Result<CompactShardLoadResult, Error> = Result {
          try autoreleasepool {
            let shardIndex = first + slot
            let shard = shardRecords[shardIndex]
            if isDirect {
              return try loadDirectShard(
                fileDescriptor: payloadDescriptor, shardIndex: shardIndex, shard: shard,
                index: index, device: concurrentDevice.value, queue: concurrentQueue.value,
                validationPipeline: concurrentValidationPipeline.value,
                maximumWidthBuffer: concurrentWidths.buffer,
                payloadPreauthenticated: payloadsPreauthenticated,
                verifyChecksums: verifyChecksums
              )
            }
            guard let metadataKernels, let decode = concurrentDecode.value else {
              throw Metal4DSTEMStreamingIOError.metalUnavailable(
                "Compact decode kernels are missing.")
            }
            return try loadCompressedShard(
              descriptor: descriptor, shardIndex: shardIndex, shard: shard, index: index,
              device: concurrentDevice.value, queue: concurrentQueue.value, decode: decode,
              metadataKernels: metadataKernels,
              validateDescriptors: concurrentValidationPipeline.value,
              maximumWidthBuffer: concurrentWidths.buffer,
              verifyChecksums: verifyChecksums
            )
          }
        }
        results.set(result, at: slot)
      }
      guard !shouldCancel() else { throw Metal4DSTEMStreamingIOError.cancelled }
      var windowTransientBytes: UInt64 = 0
      for loaded in try results.finish() {
        sourceReadMilliseconds += loaded.sourceReadMilliseconds
        descriptorPreparationMilliseconds += loaded.descriptorPreparationMilliseconds
        gpuPreparationMilliseconds += loaded.gpuPreparationMilliseconds
        gpuDecodeMilliseconds += loaded.gpuDecodeMilliseconds
        decodedIntegrityMilliseconds += loaded.decodedIntegrityMilliseconds
        privateUploadMilliseconds += loaded.privateUploadMilliseconds
        windowTransientBytes = try add(
          windowTransientBytes, loaded.maximumTransientBytes, label: "in-flight staging"
        )
        residentShards.append(loaded.resident)
      }
      maximumTransientBytes = max(maximumTransientBytes, windowTransientBytes)
    }
    let shardPipelineMilliseconds = milliseconds(from: shardPipelineStart)

    if let cacheFile, let nativeCache {
      guard try CompactNativeCache.stamp(cacheFile) == cacheStamp,
        try CompactNativeCache.signature(sourceURL: sourceURL) == nativeCache.sourceSignature
      else { throw CompactNativeCache.invalid("cache or source changed during load; retry.") }
    }

    let maximumWidthValues = maximumWidthBuffer.contents().bindMemory(
      to: UInt32.self,
      capacity: index.metadata.detectorPixelCount
    )
    for pixel in 0..<index.metadata.detectorPixelCount {
      let maximumWidth = maximumWidthValues[pixel]
      guard maximumWidth <= 16,
        !(index.manifestWorkingDtype == "uint8"
          && maximumWidth > 8 && !excludedSet.contains(pixel)),
        !(index.storageLayout == .directV3
          && excludedSet.contains(pixel) && maximumWidth != 0)
      else {
        throw invalid(
          "Compact detector pixel \(pixel) has maximum width \(maximumWidth), "
            + "which conflicts with its working dtype or authenticated mask."
        )
      }
      maximumWidths[pixel] = UInt8(maximumWidth)
    }

    let preparedDPC = try loadPreparedDPC(
      fileDescriptor: descriptor,
      index: index,
      device: device,
      verifyChecksums: verifyChecksums
    )
    maximumTransientBytes = max(
      maximumTransientBytes,
      index.metadata.preparedDPCMoments?.fileBytes ?? 0
    )
    let preparedDetectorProducts = try loadPreparedDetectorProducts(
      fileDescriptor: descriptor,
      index: index,
      device: device,
      verifyChecksums: verifyChecksums
    )
    maximumTransientBytes = max(
      maximumTransientBytes,
      preparedDetectorProducts.bytes
    )
    guard !shouldCancel() else { throw Metal4DSTEMStreamingIOError.cancelled }

    let excludedValues = (0..<index.metadata.detectorPixelCount).map {
      excludedSet.contains($0) ? UInt32(1) : UInt32(0)
    }
    let scanMapBytes = index.metadata.scanCount * MemoryLayout<UInt32>.stride
    let diffractionBytes =
      index.metadata.detectorPixelCount * MemoryLayout<UInt32>.stride
    let detectorSumBytes =
      index.metadata.detectorPixelCount * MemoryLayout<UInt64>.stride
    guard
      let excludedBuffer = excludedValues.withUnsafeBytes({ raw in
        device.makeBuffer(
          bytes: raw.baseAddress!,
          length: raw.count,
          options: .storageModeShared
        )
      }),
      let detectorA = device.makeBuffer(
        length: scanMapBytes,
        options: .storageModeShared
      ),
      let detectorB = device.makeBuffer(
        length: scanMapBytes,
        options: .storageModeShared
      ),
      let diffraction = device.makeBuffer(
        length: diffractionBytes,
        options: .storageModeShared
      ),
      let detectorSumOutput = device.makeBuffer(
        length: detectorSumBytes,
        options: .storageModeShared
      ),
      let detectorEntries = device.makeBuffer(
        length: index.metadata.detectorPixelCount
          * MemoryLayout<CompactDetectorEntry>.stride,
        options: .storageModeShared
      )
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "compact interaction outputs",
        bytes: UInt64(
          excludedValues.count * 4 + scanMapBytes * 2 + diffractionBytes
            + detectorSumBytes
        )
      )
    }
    memset(detectorA.contents(), 0, detectorA.length)
    memset(detectorB.contents(), 0, detectorB.length)
    memset(diffraction.contents(), 0, diffraction.length)
    memset(detectorSumOutput.contents(), 0, detectorSumOutput.length)
    let preparedDPCDisplayBytes = preparedDPC == nil ? 0 : scanMapBytes * 2
    var interactionResidentBytes = UInt64(
      excludedValues.count * 4 + scanMapBytes * 2 + diffractionBytes
        + detectorSumBytes
        + index.metadata.detectorPixelCount * MemoryLayout<CompactDetectorEntry>.stride
        + preparedDPCDisplayBytes
    )
    let residentPayloadLayout: UInt32 = 0
    var residencyLease: CompactResidencyLease?
    if ProcessInfo.processInfo.environment["COMPACT_RESIDENCY_SETS"] == "1" {
      if #available(macOS 15.0, iOS 18.0, *) {
        let residencyStart = ContinuousClock.now
        var stableBuffers = residentShards.flatMap { [$0.payload, $0.descriptors] }
        stableBuffers += [
          excludedBuffer, detectorA, detectorB, diffraction,
          detectorSumOutput, detectorEntries,
        ]
        if let preparedDPC {
          stableBuffers += [preparedDPC.moments, preparedDPC.row, preparedDPC.column]
        }
        stableBuffers += preparedDetectorProducts.products.values.map(\.values)
        let lease = try CompactResidencyLease(device: device, buffers: stableBuffers)
        residencyLease = lease
        // Conservatively reserve the excess reported set footprint. It may
        // include allocation rounding, not only new driver bookkeeping.
        plannedAdditionalBytes = try add(
          plannedAdditionalBytes, lease.footprintExcessBytes,
          label: "residency set footprint excess")
        interactionResidentBytes = try add(
          interactionResidentBytes, lease.footprintExcessBytes,
          label: "residency set footprint excess")
        let allocatedNow = UInt64(device.currentAllocatedSize)
        let heldAdditional = allocatedNow > allocatedBefore ? allocatedNow - allocatedBefore : 0
        if let maximumAdditionalBytes,
          plannedAdditionalBytes > maximumAdditionalBytes
            || heldAdditional > maximumAdditionalBytes
            || lease.footprintExcessBytes > maximumAdditionalBytes - heldAdditional
        {
          lease.end()
          throw invalid("Residency set bookkeeping exceeds the existing loader memory budget.")
        }
        guard !shouldCancel() else {
          lease.end()
          throw Metal4DSTEMStreamingIOError.cancelled
        }
        let record: [String: Any] = [
          "phase": "residency_set_prepared", "requested": true,
          "advisory_not_pinned": true, "allocation_count": lease.allocationCount,
          "set_allocated_bytes": lease.allocatedBytes,
          "footprint_excess_bytes": lease.footprintExcessBytes,
          "host_preparation_ms": milliseconds(from: residencyStart),
          "source_identity": index.metadata.sourceIdentitySHA256,
        ]
        if var json = try? JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]) {
          json.append(10)
          FileHandle.standardError.write(json)
        }
      }
    }
    let metrics = MetalCompactH5LoadMetrics(
      nativeCacheStatus: cacheStatus,
      nativeCacheBytes: nativeCache?.fileBytes ?? 0,
      nativeCacheDescriptorSHA256Checks: verifyChecksums ? (nativeCache?.shards.count ?? 0) : 0,
      plannedAdditionalBytes: plannedAdditionalBytes,
      sourceReadPolicy: sourceReadPolicy.rawValue,
      maximumInFlightShards: maximumInFlightShards,
      shardPipelineMilliseconds: shardPipelineMilliseconds,
      metadataMilliseconds: metadataMilliseconds,
      sourceReadMilliseconds: sourceReadMilliseconds,
      descriptorPreparationMilliseconds: descriptorPreparationMilliseconds,
      gpuPreparationMilliseconds: gpuPreparationMilliseconds,
      gpuDecodeMilliseconds: gpuDecodeMilliseconds,
      decodedIntegrityMilliseconds: decodedIntegrityMilliseconds,
      privateUploadMilliseconds: privateUploadMilliseconds,
      preparedDPCReadMilliseconds: preparedDPC?.readMilliseconds ?? 0,
      preparedDPCAuthenticationMilliseconds:
        preparedDPC?.authenticationMilliseconds ?? 0,
      preparedDPCPrimeMilliseconds: preparedDPC?.primeMilliseconds ?? 0,
      preparedDetectorProductReadMilliseconds:
        preparedDetectorProducts.readMilliseconds,
      preparedDetectorProductAuthenticationMilliseconds:
        preparedDetectorProducts.authenticationMilliseconds,
      totalMilliseconds: milliseconds(from: totalStart),
      residentBytes: index.metadata.residentBytes,
      maximumTransientBytes: maximumTransientBytes,
      deviceAllocatedBytesBefore: allocatedBefore,
      deviceAllocatedBytesAfter: UInt64(device.currentAllocatedSize),
      decodedShardSHA256Checks: verifyChecksums ? index.shards.count : 0,
      checksumsVerified: verifyChecksums,
      decodedPayloadCopyBytes: verifyChecksums && index.storageLayout == .lz4V1
        && nativeCache == nil
        ? loadingShards.reduce(UInt64(0)) { $0 + $1.decodedBytes } : 0,
      mappedAuthenticationBytes: mappedAuthenticationBytes,
      preparedDPCBytes: index.metadata.preparedDPCMoments?.fileBytes ?? 0,
      preparedDetectorProductBytes: preparedDetectorProducts.bytes,
      interactionResidentBytes: interactionResidentBytes,
      totalResidentBytes: index.metadata.residentBytes + interactionResidentBytes
    )
    let resident = MetalCompactH5ResidentSource(
      metadata: index.metadata,
      loadMetrics: metrics,
      device: device,
      queue: queue,
      selectedPipeline: selected,
      detectorPipeline: detector,
      pixelLaneDetectorPipeline: pixelLaneDetector,
      planarILPDetectorPipeline: planarILPDetector,
      planarScanCooperativePipeline: planarScanCooperative,
      planarFusedDetectorPipeline: planarFusedDetector,
      fullDecodePipeline: fullDecode,
      detectorSumPipeline: detectorSum,
      headerEncoding: index.headerEncoding,
      payloadLayout: residentPayloadLayout,
      headerWordsPerPixel: UInt32(index.headerWordsPerPixel),
      shards: residentShards,
      excluded: excludedBuffer,
      maximumWidths: maximumWidths,
      detectorOutputs: [detectorA, detectorB],
      detectorEntryBuffer: detectorEntries,
      diffractionOutput: diffraction,
      detectorSumOutput: detectorSumOutput,
      preparedDPCMomentBuffer: preparedDPC?.moments,
      preparedDPCOutputs: preparedDPC.map { [$0.row, $0.column] } ?? [],
      preparedDetectorProducts: preparedDetectorProducts.products,
      residencyLease: residencyLease
    )
    return resident
  }

  /// Conservative sum rather than assuming nonoverlap of upload/authentication.
  /// Keep accounting in the loader so a stale caller-side plan cannot bypass it.
  private static func plannedAdditionalBytes(
    index: CompactH5ParsedIndex,
    loadingShards: [CompactH5ShardRecord],
    nativeCache: CompactNativeCache?,
    authenticationPolicy: MetalCompactH5AuthenticationPolicy
  ) throws -> UInt64 {
    let metadata = index.metadata
    let pixelBytes = try multiply(
      UInt64(metadata.detectorPixelCount), UInt64(24), label: "interaction pixel buffers"
    )
    let scanBytes = try multiply(
      UInt64(metadata.scanCount), metadata.preparedDPCMoments == nil ? UInt64(8) : UInt64(16),
      label: "interaction scan buffers"
    )
    var total = try add(metadata.residentBytes, pixelBytes, label: "resident budget")
    total = try add(total, scanBytes, label: "resident interaction budget")
    // Width validation buffer remains alive through publication.
    total = try add(
      total,
      try multiply(UInt64(metadata.detectorPixelCount), UInt64(4), label: "validation bytes"),
      label: "resident validation budget"
    )
    var maximumStaging = metadata.preparedDPCMoments?.fileBytes ?? 0
    for shard in loadingShards {
      var staging = try add(shard.payloadBytes, shard.widthsBytes, label: "shard staging")
      if index.storageLayout == .lz4V1 && nativeCache == nil {
        staging = try add(staging, shard.lengthsBytes, label: "chunk length staging")
        staging = try add(staging, shard.decodedBytes, label: "decoded staging")
        // Include GPU outputs and every hierarchical scan level. These bounds
        // deliberately exceed the geometric-series scratch requirement.
        staging = try add(
          staging,
          try multiply(UInt64(shard.descriptorCount), UInt64(12), label: "descriptor staging"),
          label: "descriptor staging budget"
        )
        staging = try add(
          staging,
          try multiply(UInt64(shard.chunkCount), UInt64(36), label: "chunk metadata and status"),
          label: "chunk staging budget"
        )
        // Compressed Metal payload is rounded to four bytes by the loader.
        staging = try add(staging, (4 - shard.payloadBytes % 4) % 4, label: "payload alignment")
      }
      staging = try add(staging, UInt64(8), label: "descriptor and metadata validation status")
      maximumStaging = max(maximumStaging, staging)
    }
    if let products = metadata.preparedDetectorProducts {
      var productBytes: UInt64 = 0
      for product in products.products {
        productBytes = try add(productBytes, product.maskFileBytes, label: "prepared mask staging")
        productBytes = try add(
          productBytes, product.valuesFileBytes, label: "prepared value staging")
      }
      maximumStaging = max(maximumStaging, productBytes)
    }
    let inFlight = authenticationPolicy == .boundedConcurrent ? min(3, loadingShards.count) : 1
    total = try add(
      total, try multiply(maximumStaging, UInt64(inFlight), label: "in-flight staging budget"),
      label: "resident and staging budget"
    )
    if authenticationPolicy == .parallelMapped,
      index.storageLayout == .directV3 || nativeCache != nil
    {
      total = try add(
        total, nativeCache?.fileBytes ?? metadata.sourceBytes, label: "mapped authentication budget"
      )
    }
    return total
  }

  private static func loadPreparedDPC(
    fileDescriptor: Int32,
    index: CompactH5ParsedIndex,
    device: MTLDevice,
    verifyChecksums: Bool
  ) throws -> CompactPreparedDPCLoadResult? {
    guard let prepared = index.metadata.preparedDPCMoments else { return nil }
    let byteCount = try exactInt(prepared.fileBytes, label: "prepared DPC bytes")
    guard byteCount <= device.maxBufferLength,
      let moments = device.makeBuffer(
        length: byteCount,
        options: .storageModeShared
      )
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "prepared DPC moments",
        bytes: prepared.fileBytes
      )
    }
    let readStart = ContinuousClock.now
    try preadExact(
      fileDescriptor,
      offset: prepared.fileOffset,
      into: moments.contents(),
      byteCount: byteCount,
      label: "prepared DPC moments"
    )
    let readMilliseconds = milliseconds(from: readStart)
    var authenticationMilliseconds = 0.0
    if verifyChecksums {
      let authenticationStart = ContinuousClock.now
      let momentData = Data(
        bytesNoCopy: moments.contents(),
        count: byteCount,
        deallocator: .none
      )
      let observed = SHA256.hash(data: momentData)
        .map { String(format: "%02x", $0) }
        .joined()
      authenticationMilliseconds = milliseconds(from: authenticationStart)
      guard observed == prepared.sha256 else {
        throw invalid(
          "Compact prepared DPC SHA-256 is \(observed), expected \(prepared.sha256)."
        )
      }
    }

    return try primeDPC(
      moments: moments, prepared: prepared, metadata: index.metadata,
      device: device, readMilliseconds: readMilliseconds,
      authenticationMilliseconds: authenticationMilliseconds)
  }

  private static func primeDPC(
    moments: MTLBuffer, prepared: MetalCompactH5PreparedDPCMoments,
    metadata: MetalCompactH5Metadata, device: MTLDevice,
    readMilliseconds: Double = 0, authenticationMilliseconds: Double = 0
  ) throws -> CompactPreparedDPCLoadResult {
    let primeStart = ContinuousClock.now
    let words = moments.contents().bindMemory(
      to: UInt32.self,
      capacity: prepared.scanCount * 8
    )
    var rowValues = [Float](repeating: 0, count: prepared.scanCount)
    var columnValues = [Float](repeating: 0, count: prepared.scanCount)
    for scan in 0..<prepared.scanCount {
      let base = scan * 8
      let total =
        UInt64(UInt32(littleEndian: words[base]))
        | (UInt64(UInt32(littleEndian: words[base + 1])) << 32)
      let rowMoment =
        UInt64(UInt32(littleEndian: words[base + 2]))
        | (UInt64(UInt32(littleEndian: words[base + 3])) << 32)
      let columnMoment =
        UInt64(UInt32(littleEndian: words[base + 4]))
        | (UInt64(UInt32(littleEndian: words[base + 5])) << 32)
      guard words[base + 6] == 0, words[base + 7] == 0 else {
        throw invalid("Compact prepared DPC padding words must be zero.")
      }
      let rowLimit = total.multipliedReportingOverflow(
        by: UInt64(metadata.detectorRows - 1)
      )
      let columnLimit = total.multipliedReportingOverflow(
        by: UInt64(metadata.detectorColumns - 1)
      )
      guard total <= prepared.totalBound,
        rowMoment <= prepared.rowMomentBound,
        columnMoment <= prepared.columnMomentBound,
        !rowLimit.overflow, rowMoment <= rowLimit.partialValue,
        !columnLimit.overflow, columnMoment <= columnLimit.partialValue,
        total != 0 || (rowMoment == 0 && columnMoment == 0)
      else {
        throw invalid("Compact prepared DPC moments violate exact source bounds.")
      }
      if total != 0 {
        rowValues[scan] = Float(Double(rowMoment) / Double(total))
        columnValues[scan] = Float(Double(columnMoment) / Double(total))
      }
    }
    let rowMean = Float(
      rowValues.reduce(Double(0)) { $0 + Double($1) } / Double(prepared.scanCount)
    )
    let columnMean = Float(
      columnValues.reduce(Double(0)) { $0 + Double($1) }
        / Double(prepared.scanCount)
    )
    for scan in 0..<prepared.scanCount {
      rowValues[scan] -= rowMean
      columnValues[scan] -= columnMean
    }
    guard
      let row = rowValues.withUnsafeBytes({ raw in
        device.makeBuffer(
          bytes: raw.baseAddress!,
          length: raw.count,
          options: .storageModeShared
        )
      }),
      let column = columnValues.withUnsafeBytes({ raw in
        device.makeBuffer(
          bytes: raw.baseAddress!,
          length: raw.count,
          options: .storageModeShared
        )
      })
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "prepared DPC display maps",
        bytes: UInt64(prepared.scanCount * MemoryLayout<Float>.stride * 2)
      )
    }
    return CompactPreparedDPCLoadResult(
      moments: moments,
      row: row,
      column: column,
      readMilliseconds: readMilliseconds,
      authenticationMilliseconds: authenticationMilliseconds,
      primeMilliseconds: milliseconds(from: primeStart)
    )
  }

  private static func loadPreparedDetectorProducts(
    fileDescriptor: Int32,
    index: CompactH5ParsedIndex,
    device: MTLDevice,
    verifyChecksums: Bool
  ) throws -> CompactPreparedDetectorLoadResult {
    guard let prepared = index.metadata.preparedDetectorProducts else {
      return CompactPreparedDetectorLoadResult(
        products: [:],
        readMilliseconds: 0,
        authenticationMilliseconds: 0,
        bytes: 0
      )
    }
    let excluded = Set(index.metadata.excludedDetectorPixels)
    var residents: [String: CompactPreparedDetectorResident] = [:]
    var readMilliseconds = 0.0
    var authenticationMilliseconds = 0.0
    var totalBytes: UInt64 = 0
    for product in prepared.products {
      let maskByteCount = try exactInt(
        product.maskFileBytes,
        label: "prepared \(product.name) mask bytes"
      )
      let valuesByteCount = try exactInt(
        product.valuesFileBytes,
        label: "prepared \(product.name) values bytes"
      )
      guard valuesByteCount <= device.maxBufferLength,
        let values = device.makeBuffer(
          length: valuesByteCount,
          options: .storageModeShared
        )
      else {
        throw Metal4DSTEMStreamingIOError.allocationFailed(
          label: "prepared \(product.name) values",
          bytes: product.valuesFileBytes
        )
      }
      let readStart = ContinuousClock.now
      let maskData = try readData(
        fileDescriptor,
        offset: product.maskFileOffset,
        byteCount: maskByteCount,
        label: "prepared \(product.name) mask"
      )
      try preadExact(
        fileDescriptor,
        offset: product.valuesFileOffset,
        into: values.contents(),
        byteCount: valuesByteCount,
        label: "prepared \(product.name) values"
      )
      readMilliseconds += milliseconds(from: readStart)

      if verifyChecksums {
        let authenticationStart = ContinuousClock.now
        let maskObserved = SHA256.hash(data: maskData)
          .map { String(format: "%02x", $0) }
          .joined()
        let valuesData = Data(
          bytesNoCopy: values.contents(),
          count: valuesByteCount,
          deallocator: .none
        )
        let valuesObserved = SHA256.hash(data: valuesData)
          .map { String(format: "%02x", $0) }
          .joined()
        authenticationMilliseconds += milliseconds(from: authenticationStart)
        guard maskObserved == product.maskSHA256 else {
          throw invalid(
            "Compact prepared \(product.name.uppercased()) mask SHA-256 is "
              + "\(maskObserved), expected \(product.maskSHA256)."
          )
        }
        guard valuesObserved == product.valuesSHA256 else {
          throw invalid(
            "Compact prepared \(product.name.uppercased()) values SHA-256 is "
              + "\(valuesObserved), expected \(product.valuesSHA256)."
          )
        }
      }
      let mask = [UInt8](maskData)
      guard mask.allSatisfy({ $0 == 0 || $0 == 1 }) else {
        throw invalid(
          "Compact prepared \(product.name.uppercased()) mask is not binary."
        )
      }
      guard excluded.allSatisfy({ mask[$0] == 0 }) else {
        throw invalid(
          "Compact prepared \(product.name.uppercased()) mask selects an excluded pixel."
        )
      }
      let selected = mask.reduce(0) { $0 + Int($1) }
      guard selected == product.selectedDetectorPixels else {
        throw invalid(
          "Compact prepared \(product.name.uppercased()) mask selects \(selected) "
            + "pixels, expected \(product.selectedDetectorPixels)."
        )
      }
      residents[product.name] = CompactPreparedDetectorResident(
        metadata: product,
        mask: mask,
        values: values
      )
      totalBytes = try add(
        totalBytes,
        try add(
          product.maskFileBytes,
          product.valuesFileBytes,
          label: "prepared detector product bytes"
        ),
        label: "prepared detector products total bytes"
      )
    }
    return CompactPreparedDetectorLoadResult(
      products: residents,
      readMilliseconds: readMilliseconds,
      authenticationMilliseconds: authenticationMilliseconds,
      bytes: totalBytes
    )
  }

  private static func loadCompressedShard(
    descriptor: Int32,
    shardIndex: Int,
    shard: CompactH5ShardRecord,
    index: CompactH5ParsedIndex,
    device: MTLDevice,
    queue: MTLCommandQueue,
    decode: MTLComputePipelineState,
    metadataKernels: CompactH5MetadataKernels,
    validateDescriptors: MTLComputePipelineState,
    maximumWidthBuffer: MTLBuffer,
    verifyChecksums: Bool
  ) throws -> CompactShardLoadResult {
    var sourceReadMilliseconds = 0.0
    var descriptorPreparationMilliseconds = 0.0
    var gpuDecodeMilliseconds = 0.0
    var decodedIntegrityMilliseconds = 0.0
    var privateUploadMilliseconds = 0.0
    var maximumTransientBytes: UInt64 = 0
    let payloadBytes = try exactInt(shard.payloadBytes, label: "payload bytes")
    let lengthsBytes = try exactInt(shard.lengthsBytes, label: "length bytes")
    let widthsBytes = try exactInt(shard.widthsBytes, label: "width bytes")
    let decodedBytes = try exactInt(shard.decodedBytes, label: "decoded bytes")
    let descriptorBytes = try multiply(
      Int(shard.descriptorCount),
      MemoryLayout<UInt32>.stride,
      label: "descriptor bytes"
    )
    let chunkBytes = try multiply(
      Int(shard.chunkCount),
      MemoryLayout<CompactLZ4Chunk>.stride,
      label: "chunk metadata bytes"
    )
    maximumTransientBytes = max(
      maximumTransientBytes,
      UInt64(
        payloadBytes + lengthsBytes + widthsBytes + decodedBytes
          + descriptorBytes * 3 + chunkBytes * 2 + Int(shard.chunkCount) * 4
          + (4 - payloadBytes % 4) % 4 + 8
      )
    )
    guard decodedBytes <= device.maxBufferLength,
      descriptorBytes <= device.maxBufferLength
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "compact private shard \(shardIndex)",
        bytes: max(shard.decodedBytes, UInt64(descriptorBytes))
      )
    }

    let readStart = ContinuousClock.now
    let payloadAllocationBytes = (payloadBytes + 3) & ~3
    guard
      let compressed = device.makeBuffer(
        length: payloadAllocationBytes,
        options: .storageModeShared
      ),
      let decodedStage = device.makeBuffer(
        length: decodedBytes,
        options: verifyChecksums ? .storageModeShared : .storageModePrivate
      )
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "compact decode staging for shard \(shardIndex)",
        bytes: UInt64(payloadAllocationBytes + decodedBytes)
      )
    }
    try preadExact(
      descriptor,
      offset: shard.payloadOffset,
      into: compressed.contents(),
      byteCount: payloadBytes,
      label: "shard \(shardIndex) compressed payload"
    )
    let lengths = try CompactH5MetadataKernels.buffer(device, bytes: lengthsBytes, shared: true)
    try preadExact(
      descriptor, offset: shard.lengthsOffset, into: lengths.contents(),
      byteCount: lengthsBytes, label: "shard \(shardIndex) chunk lengths")
    let widths = try CompactH5MetadataKernels.buffer(device, bytes: widthsBytes, shared: true)
    try preadExact(
      descriptor, offset: shard.widthsOffset, into: widths.contents(),
      byteCount: widthsBytes, label: "shard \(shardIndex) descriptor widths")
    sourceReadMilliseconds += milliseconds(from: readStart)

    let preparationStart = ContinuousClock.now
    let chunkBuffer = try CompactH5MetadataKernels.buffer(device, bytes: chunkBytes)
    let descriptorStage = try CompactH5MetadataKernels.buffer(device, bytes: descriptorBytes)
    let decodeStatus = try CompactH5MetadataKernels.buffer(device, bytes: Int(shard.chunkCount) * 4)
    let metadataStatus = try CompactH5MetadataKernels.buffer(device, bytes: 4, shared: true)
    metadataStatus.contents().storeBytes(of: UInt32(0), as: UInt32.self)
    guard let preparationCommand = queue.makeCommandBuffer() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable("Could not create GPU table command.")
    }
    try metadataKernels.encodeTables(
      widths: widths, lengths: lengths, descriptorCount: Int(shard.descriptorCount),
      chunkCount: Int(shard.chunkCount), decodedWords: UInt32(shard.decodedBytes / 4),
      compressedBytes: UInt32(payloadBytes), chunkBytes: UInt32(index.metadata.payloadChunkBytes),
      descriptorOutput: descriptorStage, chunkOutput: chunkBuffer, status: metadataStatus,
      device: device, command: preparationCommand)
    try MetalCompactH5ResidentSource.complete(
      preparationCommand, operation: "shard \(shardIndex) GPU tables")
    let tableError = metadataStatus.contents().load(as: UInt32.self)
    guard tableError == 0 else {
      throw invalid(
        "Compact shard \(shardIndex) GPU metadata validation failed (status \(tableError)): "
          + "check width bits, offset overflow, and payload/chunk coverage.")
    }
    let gpuPreparationMilliseconds =
      max(
        0, preparationCommand.gpuEndTime - preparationCommand.gpuStartTime) * 1_000
    descriptorPreparationMilliseconds += milliseconds(from: preparationStart)
    let payloadWord = UInt32(shard.decodedBytes / 4)

    var decodeParameters = CompactLZ4Parameters(
      chunkCount: shard.chunkCount,
      compressedBytes: UInt32(shard.payloadBytes)
    )
    guard let decodeCommand = queue.makeCommandBuffer(),
      let decodeEncoder = decodeCommand.makeComputeCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode compact shard \(shardIndex) decode."
      )
    }
    decodeEncoder.setComputePipelineState(decode)
    decodeEncoder.setBuffer(compressed, offset: 0, index: 0)
    decodeEncoder.setBuffer(chunkBuffer, offset: 0, index: 1)
    decodeEncoder.setBuffer(decodedStage, offset: 0, index: 2)
    decodeEncoder.setBuffer(decodeStatus, offset: 0, index: 3)
    decodeEncoder.setBytes(
      &decodeParameters,
      length: MemoryLayout.stride(ofValue: decodeParameters),
      index: 4
    )
    decodeEncoder.dispatchThreadgroups(
      MTLSize(width: (Int(shard.chunkCount) + 7) / 8, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    decodeEncoder.endEncoding()
    try metadataKernels.encodeDecodeStatus(
      input: decodeStatus, count: Int(shard.chunkCount), status: metadataStatus,
      command: decodeCommand)
    try MetalCompactH5ResidentSource.complete(
      decodeCommand,
      operation: "shard \(shardIndex) raw LZ4 decode"
    )
    gpuDecodeMilliseconds +=
      max(
        0,
        decodeCommand.gpuEndTime - decodeCommand.gpuStartTime
      ) * 1_000
    if metadataStatus.contents().load(as: UInt32.self) != 0 {
      throw invalid(
        "Compact shard \(shardIndex) raw LZ4 decoder reported an invalid chunk."
      )
    }

    if verifyChecksums {
      let integrityStart = ContinuousClock.now
      let decodedData = Data(
        bytesNoCopy: decodedStage.contents(),
        count: decodedBytes,
        deallocator: .none
      )
      let decodedSHA256 = SHA256.hash(data: decodedData)
        .map { String(format: "%02x", $0) }
        .joined()
      guard decodedSHA256 == shard.decodedSHA256 else {
        throw invalid(
          "Compact shard \(shardIndex) decoded SHA-256 is \(decodedSHA256), "
            + "expected \(shard.decodedSHA256)."
        )
      }
      decodedIntegrityMilliseconds += milliseconds(from: integrityStart)
    }

    // Authentication needs CPU-visible decoded bytes. Trusted loads instead
    // decode directly into their final private payload, with no whole-payload
    // staging copy. Descriptors are already GPU-built private buffers.
    let privatePayload: MTLBuffer
    if verifyChecksums {
      privatePayload = try CompactH5MetadataKernels.buffer(device, bytes: decodedBytes)
    } else {
      privatePayload = decodedStage
    }
    let privateDescriptors = descriptorStage
    guard
      let descriptorStatus = device.makeBuffer(
        length: MemoryLayout<UInt32>.stride,
        options: .storageModeShared
      )
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "compact private buffers for shard \(shardIndex)",
        bytes: shard.decodedBytes + UInt64(descriptorBytes)
      )
    }
    memset(descriptorStatus.contents(), 0, descriptorStatus.length)
    var descriptorParameters = CompactDescriptorParameters(
      descriptorCount: shard.descriptorCount,
      payloadWords: payloadWord,
      tileCount: UInt32(
        (index.metadata.scansPerShard + index.metadata.scanTile - 1)
          / index.metadata.scanTile
      ),
      headerWordsPerPixel: UInt32(index.headerWordsPerPixel),
      scanTile: UInt32(index.metadata.scanTile),
      headerEncoding: index.headerEncoding
    )
    let uploadStart = ContinuousClock.now
    guard let uploadCommand = queue.makeCommandBuffer() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode compact shard \(shardIndex) private upload."
      )
    }
    if verifyChecksums {
      guard let blit = uploadCommand.makeBlitCommandEncoder() else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Could not encode verified payload copy.")
      }
      blit.copy(
        from: decodedStage, sourceOffset: 0, to: privatePayload,
        destinationOffset: 0, size: decodedBytes)
      blit.endEncoding()
    }
    guard let validationEncoder = uploadCommand.makeComputeCommandEncoder() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode compact shard \(shardIndex) descriptor validation."
      )
    }
    validationEncoder.setComputePipelineState(validateDescriptors)
    validationEncoder.setBuffer(privateDescriptors, offset: 0, index: 0)
    validationEncoder.setBuffer(descriptorStatus, offset: 0, index: 1)
    validationEncoder.setBytes(
      &descriptorParameters,
      length: MemoryLayout.stride(ofValue: descriptorParameters),
      index: 2
    )
    validationEncoder.setBuffer(maximumWidthBuffer, offset: 0, index: 3)
    validationEncoder.dispatchThreads(
      MTLSize(width: Int(shard.descriptorCount), height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    validationEncoder.endEncoding()
    try MetalCompactH5ResidentSource.complete(
      uploadCommand,
      operation: "shard \(shardIndex) private upload"
    )
    guard descriptorStatus.contents().load(as: UInt32.self) == 0 else {
      throw invalid(
        "Compact shard \(shardIndex) failed GPU descriptor coverage validation "
          + "with status \(descriptorStatus.contents().load(as: UInt32.self))."
      )
    }
    privateUploadMilliseconds += milliseconds(from: uploadStart)
    return CompactShardLoadResult(
      resident: CompactResidentShard(payload: privatePayload, descriptors: privateDescriptors),
      sourceReadMilliseconds: sourceReadMilliseconds,
      descriptorPreparationMilliseconds: descriptorPreparationMilliseconds,
      gpuPreparationMilliseconds: gpuPreparationMilliseconds,
      gpuDecodeMilliseconds: gpuDecodeMilliseconds,
      decodedIntegrityMilliseconds: decodedIntegrityMilliseconds,
      privateUploadMilliseconds: privateUploadMilliseconds,
      maximumTransientBytes: maximumTransientBytes
    )

  }

  private static func authenticateDirectPayloads(
    fileDescriptor: Int32,
    fileBytes sourceBytes: UInt64,
    shards: [CompactH5ShardRecord]
  ) throws {
    let fileBytes = try exactInt(
      sourceBytes,
      label: "mapped authentication bytes"
    )
    guard
      let mapping = Darwin.mmap(
        nil,
        fileBytes,
        PROT_READ,
        MAP_PRIVATE,
        fileDescriptor,
        0
      ), mapping != MAP_FAILED
    else {
      throw invalid(
        "Could not map compact direct payloads for parallel authentication: "
          + lastPOSIXError()
      )
    }
    defer { Darwin.munmap(mapping, fileBytes) }
    let mappingAddress = UInt(bitPattern: mapping)
    let failures = ConcurrentStringCollector()
    DispatchQueue.concurrentPerform(iterations: shards.count) { shardIndex in
      autoreleasepool {
        let shard = shards[shardIndex]
        guard let offset = Int(exactly: shard.payloadOffset),
          let count = Int(exactly: shard.payloadBytes)
        else {
          failures.append("shard \(shardIndex) range exceeds the host integer range")
          return
        }
        guard let mappedBytes = UnsafeMutableRawPointer(bitPattern: mappingAddress) else {
          failures.append("shard \(shardIndex) could not resolve the mapped file address")
          return
        }
        let data = Data(
          bytesNoCopy: mappedBytes.advanced(by: offset),
          count: count,
          deallocator: .none
        )
        let observed = SHA256.hash(data: data)
          .map { String(format: "%02x", $0) }
          .joined()
        if observed != shard.decodedSHA256 {
          failures.append(
            "shard \(shardIndex) is \(observed), expected \(shard.decodedSHA256)"
          )
        }
        if let expected = shard.descriptorsSHA256 {
          let headers = Data(
            bytesNoCopy: mappedBytes.advanced(by: Int(shard.widthsOffset)),
            count: Int(shard.widthsBytes), deallocator: .none
          )
          if CompactNativeCache.digest(headers) != expected {
            failures.append("shard \(shardIndex) cached descriptor SHA-256 mismatch")
          }
        }
      }
    }
    let observedFailures = failures.snapshot()
    guard observedFailures.isEmpty else {
      throw invalid(
        "Compact direct parallel authentication failed: "
          + observedFailures.sorted().joined(separator: "; ")
      )
    }
  }

  private static func loadDirectShard(
    fileDescriptor: Int32,
    shardIndex: Int,
    shard: CompactH5ShardRecord,
    index: CompactH5ParsedIndex,
    device: MTLDevice,
    queue: MTLCommandQueue,
    validationPipeline: MTLComputePipelineState,
    maximumWidthBuffer: MTLBuffer,
    payloadPreauthenticated: Bool,
    verifyChecksums: Bool
  ) throws -> CompactShardLoadResult {
    let payloadBytes = try exactInt(shard.payloadBytes, label: "direct payload bytes")
    let headerBytes = try exactInt(shard.widthsBytes, label: "compact header bytes")
    let headerWords = try exactInt(
      UInt64(shard.descriptorCount),
      label: "compact header words"
    )
    guard shard.payloadBytes == shard.decodedBytes,
      payloadBytes.isMultiple(of: MemoryLayout<UInt32>.stride),
      headerBytes == headerWords * MemoryLayout<UInt32>.stride,
      payloadBytes <= device.maxBufferLength,
      headerBytes <= device.maxBufferLength,
      let payloadWords = UInt32(exactly: payloadBytes / 4)
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "direct compact private shard \(shardIndex)",
        bytes: max(shard.payloadBytes, shard.widthsBytes)
      )
    }
    guard
      let payloadStage = device.makeBuffer(
        length: payloadBytes,
        options: .storageModeShared
      ),
      let headerStage = device.makeBuffer(
        length: headerBytes,
        options: .storageModeShared
      )
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "direct compact staging for shard \(shardIndex)",
        bytes: UInt64(payloadBytes + headerBytes)
      )
    }

    let readStart = ContinuousClock.now
    try preadExact(
      fileDescriptor,
      offset: shard.payloadOffset,
      into: payloadStage.contents(),
      byteCount: payloadBytes,
      label: "shard \(shardIndex) direct payload"
    )
    try preadExact(
      fileDescriptor,
      offset: shard.widthsOffset,
      into: headerStage.contents(),
      byteCount: headerBytes,
      label: "shard \(shardIndex) compact headers"
    )
    let sourceReadMilliseconds = milliseconds(from: readStart)

    var decodedIntegrityMilliseconds = 0.0
    if verifyChecksums && !payloadPreauthenticated {
      let integrityStart = ContinuousClock.now
      let payloadData = Data(
        bytesNoCopy: payloadStage.contents(),
        count: payloadBytes,
        deallocator: .none
      )
      let payloadSHA256 = SHA256.hash(data: payloadData)
        .map { String(format: "%02x", $0) }
        .joined()
      guard payloadSHA256 == shard.decodedSHA256 else {
        throw invalid(
          "Compact shard \(shardIndex) direct payload SHA-256 is \(payloadSHA256), "
            + "expected \(shard.decodedSHA256)."
        )
      }
      if let expected = shard.descriptorsSHA256 {
        let headerData = Data(
          bytesNoCopy: headerStage.contents(), count: headerBytes, deallocator: .none
        )
        guard CompactNativeCache.digest(headerData) == expected else {
          throw CompactNativeCache.invalid("shard \(shardIndex) descriptor SHA-256 mismatch.")
        }
      }
      decodedIntegrityMilliseconds = milliseconds(from: integrityStart)
    }

    let preparationStart = ContinuousClock.now
    let tileCount =
      (index.metadata.scansPerShard + index.metadata.scanTile - 1)
      / index.metadata.scanTile
    let checkpointWords = (tileCount + 31) / 32
    let widthWords = (tileCount + 7) / 8
    guard
      (index.headerEncoding == 0 && index.metadata.scanTile == 128
        && index.headerWordsPerPixel == tileCount)
        || ((index.headerEncoding == 1 || index.headerEncoding == 2)
          && index.metadata.scanTile == 32
          && index.headerWordsPerPixel == checkpointWords + widthWords),
      headerWords
        == index.metadata.detectorPixelCount * index.headerWordsPerPixel
    else {
      throw invalid("Compact shard \(shardIndex) has inconsistent direct headers.")
    }
    let descriptorPreparationMilliseconds = milliseconds(from: preparationStart)

    guard
      let privatePayload = device.makeBuffer(
        length: payloadBytes,
        options: .storageModePrivate
      ),
      let privateHeaders = device.makeBuffer(
        length: headerBytes,
        options: .storageModePrivate
      ),
      let descriptorStatus = device.makeBuffer(
        length: MemoryLayout<UInt32>.stride,
        options: .storageModeShared
      )
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "direct compact private buffers for shard \(shardIndex)",
        bytes: UInt64(payloadBytes + headerBytes)
      )
    }
    memset(descriptorStatus.contents(), 0, descriptorStatus.length)
    let validationCount =
      index.headerEncoding == 0
      ? Int(shard.descriptorCount) : index.metadata.detectorPixelCount
    var descriptorParameters = CompactDescriptorParameters(
      descriptorCount: UInt32(validationCount),
      payloadWords: payloadWords,
      tileCount: UInt32(tileCount),
      headerWordsPerPixel: UInt32(index.headerWordsPerPixel),
      scanTile: UInt32(index.metadata.scanTile),
      headerEncoding: index.headerEncoding
    )
    let uploadStart = ContinuousClock.now
    guard let command = queue.makeCommandBuffer(),
      let blit = command.makeBlitCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode direct compact shard \(shardIndex) upload."
      )
    }
    blit.copy(
      from: payloadStage,
      sourceOffset: 0,
      to: privatePayload,
      destinationOffset: 0,
      size: payloadBytes
    )
    blit.copy(
      from: headerStage,
      sourceOffset: 0,
      to: privateHeaders,
      destinationOffset: 0,
      size: headerBytes
    )
    blit.endEncoding()
    guard let validation = command.makeComputeCommandEncoder() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode direct compact shard \(shardIndex) validation."
      )
    }
    validation.setComputePipelineState(validationPipeline)
    validation.setBuffer(privateHeaders, offset: 0, index: 0)
    validation.setBuffer(descriptorStatus, offset: 0, index: 1)
    validation.setBytes(
      &descriptorParameters,
      length: MemoryLayout.stride(ofValue: descriptorParameters),
      index: 2
    )
    validation.setBuffer(maximumWidthBuffer, offset: 0, index: 3)
    validation.dispatchThreads(
      MTLSize(width: validationCount, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    validation.endEncoding()
    try MetalCompactH5ResidentSource.complete(
      command,
      operation: "direct compact shard \(shardIndex) private upload"
    )
    let status = descriptorStatus.contents().load(as: UInt32.self)
    guard status == 0 else {
      throw invalid(
        "Compact shard \(shardIndex) failed GPU direct-header validation "
          + "with status \(status)."
      )
    }
    return CompactShardLoadResult(
      resident: CompactResidentShard(
        payload: privatePayload,
        descriptors: privateHeaders
      ),
      sourceReadMilliseconds: sourceReadMilliseconds,
      descriptorPreparationMilliseconds: descriptorPreparationMilliseconds,
      gpuPreparationMilliseconds: 0,
      gpuDecodeMilliseconds: 0,
      decodedIntegrityMilliseconds: decodedIntegrityMilliseconds,
      privateUploadMilliseconds: milliseconds(from: uploadStart),
      maximumTransientBytes: UInt64(payloadBytes + headerBytes + 4)
    )
  }

  private static func openSource(
    _ sourceURL: URL, readPolicy: MetalCompactH5SourceReadPolicy
  ) throws -> Int32 {
    let descriptor = sourceURL.path.withCString { Darwin.open($0, O_RDONLY) }
    guard descriptor >= 0 else {
      throw invalid("Could not open compact source \(sourceURL.path): \(lastPOSIXError()).")
    }
    if readPolicy == .avoidCaching, Darwin.fcntl(descriptor, F_NOCACHE, 1) != 0 {
      let detail = lastPOSIXError()
      Darwin.close(descriptor)
      throw invalid(
        "Could not apply source descriptor F_NOCACHE: \(detail). "
          + "Use systemDefault reads or correct filesystem support; no cold-read claim is valid."
      )
    }
    return descriptor
  }

  fileprivate static func parse(
    sourceURL: URL, readPolicy: MetalCompactH5SourceReadPolicy = .systemDefault
  ) throws -> CompactH5ParsedIndex {
    let attributes = try FileManager.default.attributesOfItem(atPath: sourceURL.path)
    guard let fileNumber = attributes[.size] as? NSNumber else {
      throw invalid("Could not determine compact source size for \(sourceURL.path).")
    }
    let fileBytes = fileNumber.uint64Value
    let descriptor = try openSource(sourceURL, readPolicy: readPolicy)
    defer { Darwin.close(descriptor) }
    let prelude = try readData(
      descriptor,
      offset: 0,
      byteCount: 24,
      label: "container prelude"
    )
    guard Array(prelude[0..<8]) == containerMagic else {
      throw invalid("\(sourceURL.path) has no compact QGPUH5 v1 user-block index.")
    }
    let headerBytes = Int(readU32(prelude, at: 8))
    let headerCRC32 = readU32(prelude, at: 12)
    let binaryOffset = UInt64(readU32(prelude, at: 16))
    let binaryBytes = Int(readU32(prelude, at: 20))
    guard headerBytes > 0,
      UInt64(24 + headerBytes) <= binaryOffset,
      binaryBytes >= 76,
      binaryOffset <= fileBytes,
      UInt64(binaryBytes) <= fileBytes - binaryOffset
    else {
      throw invalid("Compact JSON or binary index range is outside the file.")
    }
    let header = try readData(
      descriptor,
      offset: 24,
      byteCount: headerBytes,
      label: "JSON header"
    )
    guard crc32(header) == headerCRC32 else {
      throw invalid("Compact JSON header failed its CRC-32 check.")
    }
    guard let manifest = try JSONSerialization.jsonObject(with: header) as? [String: Any]
    else {
      throw invalid("Compact JSON header is not an object.")
    }
    let binary = try readData(
      descriptor,
      offset: binaryOffset,
      byteCount: binaryBytes,
      label: "binary index"
    )
    var cursor = 0
    let magic = Array(binary[0..<8])
    let storageLayout: CompactH5StorageLayout
    let schema: String
    let payloadCodec: String
    var headerEncoding: UInt32
    if magic == indexMagicV1 {
      storageLayout = .lz4V1
      schema = "quantem.gpu.packed-detector-h5/v1"
      payloadCodec = "independent raw LZ4 blocks"
      headerEncoding = 0
    } else if magic == indexMagicV3 {
      storageLayout = .directV3
      schema = "quantem.gpu.packed-detector-h5/v3"
      payloadCodec = "direct-bitpacked-u32"
      headerEncoding = 1
    } else {
      throw invalid("Compact binary index is not supported QGIX v1 or v3.")
    }
    cursor += 8
    let shardCount = Int(readU32(binary, at: cursor))
    cursor += 4
    let secondHeaderWord = readU32(binary, at: cursor)
    cursor += 4
    let payloadChunkBytes =
      storageLayout == .lz4V1
      ? Int(secondHeaderWord)
      : 0
    let scanRows = Int(readU32(binary, at: cursor))
    cursor += 4
    let scanColumns = Int(readU32(binary, at: cursor))
    cursor += 4
    let detectorRows = Int(readU32(binary, at: cursor))
    cursor += 4
    let detectorColumns = Int(readU32(binary, at: cursor))
    cursor += 4
    let scansPerShard = Int(readU32(binary, at: cursor))
    cursor += 4
    let scanTile: Int
    if storageLayout == .directV3 {
      scanTile = Int(readU32(binary, at: cursor))
      cursor += 4
      let encodedHeaderLayout = readU32(binary, at: cursor)
      cursor += 4
      guard secondHeaderWord == 0, scanTile == 32,
        encodedHeaderLayout == 1 || encodedHeaderLayout == 2
      else {
        throw invalid("Compact QGIX v3 header layout is unsupported.")
      }
      headerEncoding = encodedHeaderLayout
    } else {
      scanTile = 128
    }
    let scanCount = try multiply(scanRows, scanColumns, label: "scan count")
    let coveredScans = try multiply(
      shardCount,
      scansPerShard,
      label: "covered scans"
    )
    guard shardCount > 0,
      storageLayout == .directV3 || payloadChunkBytes == 128,
      scanRows > 0, scanColumns > 0, detectorRows > 0, detectorColumns > 0,
      scansPerShard > 0, scanTile > 0,
      scansPerShard.isMultiple(of: scanTile), scanCount == coveredScans
    else {
      throw invalid("Compact binary index has invalid geometry or shard coverage.")
    }
    let detectorPixels = try multiply(
      detectorRows,
      detectorColumns,
      label: "detector pixels"
    )
    guard cursor + 4 <= binary.count else {
      throw invalid("Compact detector exclusion list is truncated.")
    }
    let maskCount = Int(readU32(binary, at: cursor))
    cursor += 4
    let maskBytes = try multiply(maskCount, 4, label: "detector exclusion bytes")
    let requiredMaskBytes = try add(
      maskBytes,
      32,
      label: "detector exclusion and identity bytes"
    )
    guard maskCount <= detectorPixels,
      cursor <= binary.count,
      requiredMaskBytes <= binary.count - cursor
    else {
      throw invalid("Compact detector exclusion list is invalid.")
    }
    var excluded: [Int] = []
    excluded.reserveCapacity(maskCount)
    for _ in 0..<maskCount {
      excluded.append(Int(readU32(binary, at: cursor)))
      cursor += 4
    }
    guard Set(excluded).count == excluded.count,
      excluded.allSatisfy({ 0..<detectorPixels ~= $0 })
    else {
      throw invalid("Compact detector exclusion indices are not unique and in range.")
    }
    let sourceIdentity = binary[cursor..<(cursor + 32)]
      .map { String(format: "%02x", $0) }
      .joined()
    cursor += 32
    let tileCount = (scansPerShard + scanTile - 1) / scanTile
    let headerWordsPerPixel =
      storageLayout == .directV3
      ? (tileCount + 31) / 32 + (tileCount + 7) / 8
      : tileCount
    var shards: [CompactH5ShardRecord] = []
    var ranges: [(UInt64, UInt64, String)] = []
    var residentBytes: UInt64 = 0
    for shardIndex in 0..<shardCount {
      guard cursor + 96 <= binary.count else {
        throw invalid("Compact shard record \(shardIndex) is truncated.")
      }
      let payloadOffset = readU64(binary, at: cursor)
      cursor += 8
      let payloadBytes = readU64(binary, at: cursor)
      cursor += 8
      let lengthsOffset = readU64(binary, at: cursor)
      cursor += 8
      let lengthsBytes = readU64(binary, at: cursor)
      cursor += 8
      let widthsOffset = readU64(binary, at: cursor)
      cursor += 8
      let widthsBytes = readU64(binary, at: cursor)
      cursor += 8
      let decodedBytes = readU64(binary, at: cursor)
      cursor += 8
      let descriptorCount = readU32(binary, at: cursor)
      cursor += 4
      let chunkCount = readU32(binary, at: cursor)
      cursor += 4
      let decodedSHA256 = binary[cursor..<(cursor + 32)]
        .map { String(format: "%02x", $0) }
        .joined()
      cursor += 32
      let expectedHeaderWords = try multiply(
        detectorPixels,
        headerWordsPerPixel,
        label: "descriptor or header count"
      )
      guard let expectedDescriptorCount = UInt32(exactly: expectedHeaderWords),
        descriptorCount == expectedDescriptorCount,
        payloadBytes > 0,
        payloadBytes <= UInt64(UInt32.max),
        decodedBytes > 0,
        decodedBytes.isMultiple(of: 4),
        decodedBytes / 4 <= UInt64(UInt32.max)
      else {
        throw invalid("Compact shard \(shardIndex) has inconsistent counts.")
      }
      if storageLayout == .lz4V1 {
        guard widthsBytes == UInt64(expectedHeaderWords),
          lengthsBytes == UInt64(chunkCount),
          chunkCount == UInt32((decodedBytes - 1) / 128 + 1)
        else {
          throw invalid("Compact shard \(shardIndex) has inconsistent v1 counts.")
        }
      } else {
        guard widthsBytes == UInt64(expectedHeaderWords * 4),
          lengthsOffset == 0, lengthsBytes == 0, chunkCount == 0,
          payloadBytes == decodedBytes
        else {
          throw invalid("Compact shard \(shardIndex) has inconsistent v3 counts.")
        }
      }
      for (offset, count, label) in [
        (payloadOffset, payloadBytes, "payload"),
        (lengthsOffset, lengthsBytes, "lengths"),
        (widthsOffset, widthsBytes, "widths"),
      ] where count != 0 {
        guard offset <= fileBytes, count <= fileBytes - offset else {
          throw invalid("Compact shard \(shardIndex) \(label) is outside the file.")
        }
        ranges.append((offset, offset + count, "shard \(shardIndex) \(label)"))
      }
      residentBytes = try add(
        residentBytes,
        decodedBytes + UInt64(expectedHeaderWords * 4),
        label: "resident bytes"
      )
      shards.append(
        CompactH5ShardRecord(
          payloadOffset: payloadOffset,
          payloadBytes: payloadBytes,
          lengthsOffset: lengthsOffset,
          lengthsBytes: lengthsBytes,
          widthsOffset: widthsOffset,
          widthsBytes: widthsBytes,
          decodedBytes: decodedBytes,
          descriptorCount: descriptorCount,
          chunkCount: chunkCount,
          decodedSHA256: decodedSHA256
        )
      )
    }
    guard cursor == binary.count else {
      throw invalid("Compact binary index has trailing bytes.")
    }
    ranges.sort { $0.0 < $1.0 }
    for (previous, current) in zip(ranges, ranges.dropFirst())
    where current.0 < previous.1 {
      throw invalid(
        "Compact file ranges overlap between \(previous.2) and \(current.2)."
      )
    }

    let shape = [scanRows, scanColumns, detectorRows, detectorColumns]
    guard manifest["schema"] as? String == schema,
      intValue(manifest["shard_count"]) == shardCount,
      intArray(manifest["source_shape"]) == shape,
      manifest["source_identity_sha256"] as? String == sourceIdentity,
      manifest["status"] as? String == "complete"
    else {
      throw invalid("Compact JSON manifest conflicts with the binary contract.")
    }
    let manifestMask: [Int]
    let embeddedScientificSemantics: Bool
    var maskedDetectorPixelsSHA256: String?
    var maskedDetectorRawValues: [UInt16]?
    var rawAccessMode = "mask_applied_only_legacy"
    if storageLayout == .directV3 {
      guard intValue(manifest["scan_tile"]) == scanTile,
        manifest["payload_codec"] as? String == payloadCodec,
        validSHA256(
          manifest[headerEncoding == 2 ? "working_logical_sha256" : "prepared_uint8_sha256"])
      else {
        throw invalid("Compact QGIX v3 JSON contract is incomplete or unsupported.")
      }
      if manifest["source_dtype"] != nil {
        guard ["uint8", "uint16"].contains(manifest["source_dtype"] as? String ?? ""),
          intValue(manifest["scan_bin"]) == 1,
          intValue(manifest["detector_bin"]) == 1,
          manifest["crop"] is NSNull,
          manifest["working_value_definition"] as? String
            == "all admitted source counts exactly; authenticated dead pixels set to zero",
          validSHA256(manifest["detector_mask_sha256"]),
          let flatMask = intArray(manifest["masked_detector_pixels"])
        else {
          throw invalid("Compact QGIX v3 embedded scientific semantics are malformed.")
        }
        embeddedScientificSemantics = true
        manifestMask = flatMask
        let rawValues = intArray(manifest["masked_detector_raw_values"])
        let indexSHA = manifest["masked_detector_pixels_sha256"] as? String
        if let rawValues, let indexSHA {
          guard rawValues.count == flatMask.count,
            rawValues.allSatisfy({ 0...Int(UInt16.max) ~= $0 }),
            validSHA256(indexSHA),
            indexSHA == sha256OrderedDetectorPixels(flatMask)
          else {
            throw invalid("Compact QGIX v3 raw exclusion constants are malformed.")
          }
          maskedDetectorPixelsSHA256 = indexSHA
          maskedDetectorRawValues = rawValues.map(UInt16.init)
          rawAccessMode = "exact_exclusion_constants"
        } else if flatMask.isEmpty {
          rawAccessMode = "exact_no_exclusions"
        }
      } else {
        guard excluded.isEmpty,
          validSHA256(manifest["parent_hdf5_sha256"]),
          manifest["source_raw_logical_sha256"] == nil,
          manifest["detector_mask_sha256"] == nil,
          manifest["masked_detector_pixels"] == nil
        else {
          throw invalid(
            "Compact legacy QGIX v3 metadata is ambiguous; supply a regenerated source."
          )
        }
        embeddedScientificSemantics = false
        manifestMask = []
        rawAccessMode = "external_audit_required"
      }
    } else {
      guard manifest["source_dtype"] as? String == "uint16",
        intValue(manifest["scan_bin"]) == 1,
        intValue(manifest["detector_bin"]) == 1,
        manifest["crop"] is NSNull,
        intValue(manifest["scans_per_shard"]) == scansPerShard,
        intValue(manifest["payload_chunk_bytes"]) == payloadChunkBytes,
        let manifestCoordinates = coordinateArray(
          manifest["masked_detector_pixels"]
        ),
        manifestCoordinates.allSatisfy({ coordinate in
          0..<detectorRows ~= coordinate.0 && 0..<detectorColumns ~= coordinate.1
        })
      else {
        throw invalid("Compact QGIX v1 JSON contract is incomplete or unsupported.")
      }
      manifestMask = manifestCoordinates.map {
        $0.0 * detectorColumns + $0.1
      }
      embeddedScientificSemantics = true
      if excluded.isEmpty {
        rawAccessMode = "exact_no_exclusions"
      } else if manifest["masked_detector_payload_policy"] as? String
        == "retained_exactly_in_payload"
      {
        rawAccessMode = "exact_retained_payload"
      }
      let rawValues = intArray(manifest["masked_detector_raw_values"])
      let indexSHA = manifest["masked_detector_pixels_sha256"] as? String
      if rawValues != nil || indexSHA != nil {
        guard let rawValues, let indexSHA,
          rawValues.count == manifestMask.count,
          rawValues.allSatisfy({ 0...Int(UInt16.max) ~= $0 }),
          validSHA256(indexSHA),
          indexSHA == sha256OrderedDetectorPixels(manifestMask)
        else {
          throw invalid("Compact QGIX v1 raw exclusion constants are malformed.")
        }
        maskedDetectorPixelsSHA256 = indexSHA
        maskedDetectorRawValues = rawValues.map(UInt16.init)
      }
    }
    guard manifestMask == excluded else {
      throw invalid("Compact JSON and binary detector exclusions differ.")
    }
    let workingDtype = manifest["working_dtype"] as? String ?? ""
    guard workingDtype == "uint8" || workingDtype == "uint16" else {
      throw invalid("Compact working dtype or payload codec is unsupported.")
    }
    if storageLayout == .lz4V1 {
      guard manifest["payload_chunk_codec"] as? String == payloadCodec,
        manifest["payload_chunk_length_codec"] as? String
          == "uint8 encoded_bytes_minus_one",
        manifest["descriptor_codec"] as? String == "uint8 five-bit widths"
      else {
        throw invalid("Compact working dtype or payload codec is unsupported.")
      }
    } else if workingDtype != (headerEncoding == 2 ? "uint16" : "uint8") {
      throw invalid("Compact QGIX v3 working dtype disagrees with its header encoding.")
    }
    let sourceRawSHA = manifest["source_raw_logical_sha256"] as? String
    guard sourceRawSHA == nil || validSHA256(sourceRawSHA) else {
      throw invalid("Compact source logical SHA-256 is malformed.")
    }
    let workingLogicalSHA: String? =
      storageLayout == .lz4V1 || workingDtype == "uint16"
      ? manifest["working_logical_sha256"] as? String
      : manifest["prepared_uint8_sha256"] as? String
    guard workingLogicalSHA == nil || validSHA256(workingLogicalSHA) else {
      throw invalid("Compact working logical SHA-256 is malformed.")
    }
    let parsedCalibration = try detectorCalibration(
      manifest["detector_calibration"],
      sourceIdentity: sourceIdentity,
      detectorRows: detectorRows,
      detectorColumns: detectorColumns
    )
    let manifestJSON = try canonicalJSON(manifest)
    guard let manifestData = manifestJSON.data(using: .utf8) else {
      throw invalid("Compact manifest could not be encoded as UTF-8.")
    }
    let manifestSHA256 = SHA256.hash(data: manifestData)
      .map { String(format: "%02x", $0) }
      .joined()
    let detectorCalibrationSchema =
      (manifest["detector_calibration"] as? [String: Any])?["schema"] as? String
    let detectorCalibrationSHA256: String? = try {
      guard let calibration = manifest["detector_calibration"], !(calibration is NSNull)
      else { return nil }
      let encoded = try canonicalJSON(calibration)
      guard let data = encoded.data(using: .utf8) else {
        throw invalid("Compact detector calibration could not be encoded as UTF-8.")
      }
      return SHA256.hash(data: data)
        .map { String(format: "%02x", $0) }
        .joined()
    }()
    let preparedDPC = try preparedDPCMoments(
      manifest["prepared_dpc_moments"],
      parentManifest: manifest,
      fileBytes: fileBytes,
      shape: shape,
      sourceIdentity: sourceIdentity,
      excluded: excluded,
      storageLayout: storageLayout,
      shards: shards
    )
    if let preparedDPC {
      residentBytes = try add(
        residentBytes,
        preparedDPC.fileBytes,
        label: "resident bytes with prepared DPC moments"
      )
    }
    let preparedDetectorProducts = try preparedDetectorProducts(
      manifest["prepared_detector_products"],
      parentManifest: manifest,
      fileBytes: fileBytes,
      shape: shape,
      sourceIdentity: sourceIdentity,
      storageLayout: storageLayout,
      shards: shards,
      preparedDPC: preparedDPC,
      calibration: parsedCalibration
    )
    if let preparedDetectorProducts {
      for product in preparedDetectorProducts.products {
        residentBytes = try add(
          residentBytes,
          try add(
            product.maskFileBytes,
            product.valuesFileBytes,
            label: "prepared detector product bytes"
          ),
          label: "resident bytes with prepared detector products"
        )
      }
    }
    let metadata = MetalCompactH5Metadata(
      sourceURL: sourceURL,
      sourceBytes: fileBytes,
      schema: schema,
      payloadCodec: payloadCodec,
      sourceDtype: manifest["source_dtype"] as? String,
      manifestSHA256: manifestSHA256,
      workingDtype: workingDtype,
      embeddedScientificSemantics: embeddedScientificSemantics,
      scanRows: scanRows,
      scanColumns: scanColumns,
      detectorRows: detectorRows,
      detectorColumns: detectorColumns,
      scansPerShard: scansPerShard,
      scanTile: scanTile,
      payloadChunkBytes: payloadChunkBytes,
      sourceIdentitySHA256: sourceIdentity,
      sourceRawLogicalSHA256: sourceRawSHA,
      workingLogicalSHA256: workingLogicalSHA,
      detectorMaskSHA256: manifest["detector_mask_sha256"] as? String,
      maskedDetectorPixelsSHA256: maskedDetectorPixelsSHA256,
      maskedDetectorRawValues: maskedDetectorRawValues,
      rawAccessMode: rawAccessMode,
      detectorCalibration: parsedCalibration,
      detectorCalibrationSchema: detectorCalibrationSchema,
      detectorCalibrationSHA256: detectorCalibrationSHA256,
      preparedDPCMoments: preparedDPC,
      preparedDetectorProducts: preparedDetectorProducts,
      excludedDetectorPixels: excluded,
      shardCount: shardCount,
      residentBytes: residentBytes
    )
    return CompactH5ParsedIndex(
      metadata: metadata,
      manifestWorkingDtype: workingDtype,
      storageLayout: storageLayout,
      headerEncoding: headerEncoding,
      headerWordsPerPixel: headerWordsPerPixel,
      shards: shards
    )
  }

  private static func pipeline(
    library: MTLLibrary,
    name: String,
    device: MTLDevice
  ) throws -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Compact Metal function \(name) is missing."
      )
    }
    do {
      return try device.makeComputePipelineState(function: function)
    } catch {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Compact Metal pipeline \(name) failed: \(error.localizedDescription)"
      )
    }
  }

  private static func readData(
    _ descriptor: Int32,
    offset: UInt64,
    byteCount: Int,
    label: String
  ) throws -> Data {
    var data = Data(count: byteCount)
    try data.withUnsafeMutableBytes { raw in
      try preadExact(
        descriptor,
        offset: offset,
        into: raw.baseAddress!,
        byteCount: raw.count,
        label: label
      )
    }
    return data
  }

  private static func preadExact(
    _ descriptor: Int32,
    offset: UInt64,
    into destination: UnsafeMutableRawPointer,
    byteCount: Int,
    label: String
  ) throws {
    guard offset <= UInt64(Int64.max) else {
      throw invalid("Compact \(label) offset is outside POSIX range.")
    }
    var completed = 0
    while completed < byteCount {
      let count = Darwin.pread(
        descriptor,
        destination.advanced(by: completed),
        byteCount - completed,
        off_t(offset + UInt64(completed))
      )
      if count < 0 {
        if errno == EINTR { continue }
        throw invalid("Could not read compact \(label): \(lastPOSIXError()).")
      }
      guard count > 0 else {
        throw invalid("Compact \(label) ended before \(byteCount) bytes.")
      }
      completed += count
    }
  }

  private static func readU32(_ data: Data, at offset: Int) -> UInt32 {
    data.withUnsafeBytes {
      UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self))
    }
  }

  private static func readU64(_ data: Data, at offset: Int) -> UInt64 {
    data.withUnsafeBytes {
      UInt64(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt64.self))
    }
  }

  private static func crc32(_ data: Data) -> UInt32 {
    var crc = UInt32.max
    for byte in data {
      crc ^= UInt32(byte)
      for _ in 0..<8 {
        crc = (crc >> 1) ^ (0xedb8_8320 & (0 &- (crc & 1)))
      }
    }
    return ~crc
  }

  private static func intValue(_ value: Any?) -> Int? {
    (value as? NSNumber)?.intValue
  }

  private static func validSHA256(_ value: Any?) -> Bool {
    guard let string = value as? String, string.utf8.count == 64 else {
      return false
    }
    return string.utf8.allSatisfy { byte in
      (48...57).contains(byte) || (97...102).contains(byte)
    }
  }

  private static func sha256OrderedDetectorPixels(_ pixels: [Int]) -> String {
    var bytes = Data()
    bytes.reserveCapacity(pixels.count * MemoryLayout<UInt32>.stride)
    for pixel in pixels {
      var value = UInt32(pixel).littleEndian
      Swift.withUnsafeBytes(of: &value) { bytes.append(contentsOf: $0) }
    }
    return SHA256.hash(data: bytes)
      .map { String(format: "%02x", $0) }
      .joined()
  }

  private static func preparedDPCMoments(
    _ value: Any?,
    parentManifest: [String: Any],
    fileBytes: UInt64,
    shape: [Int],
    sourceIdentity: String,
    excluded: [Int],
    storageLayout: CompactH5StorageLayout,
    shards: [CompactH5ShardRecord]
  ) throws -> MetalCompactH5PreparedDPCMoments? {
    guard let value, !(value is NSNull) else { return nil }
    guard let prepared = value as? [String: Any] else {
      throw invalid("Compact prepared DPC moments are not an object.")
    }
    let schema: String
    let workingDtype: String
    let workingField: String
    let maximumValue: UInt64
    let workingSHA: String
    if storageLayout == .lz4V1,
      parentManifest["working_dtype"] as? String != "uint16"
    {
      throw invalid("Compact prepared DPC moments require uint16 working data.")
    }
    if parentManifest["working_dtype"] as? String == "uint16" {
      schema = "quantem.gpu.prepared-dpc-moments/v2"
      workingDtype = "uint16"
      workingField = "working_logical_sha256"
      maximumValue = UInt64(UInt16.max)
      guard let digest = parentManifest[workingField] as? String,
        validSHA256(digest)
      else {
        throw invalid("Compact prepared DPC parent working identity is invalid.")
      }
      workingSHA = digest
    } else {
      schema = "quantem.gpu.prepared-dpc-moments/v1"
      workingDtype = "uint8"
      workingField = "working_uint8_sha256"
      maximumValue = UInt64(UInt8.max)
      guard let digest = parentManifest["prepared_uint8_sha256"] as? String,
        validSHA256(digest)
      else {
        throw invalid("Compact prepared DPC parent working identity is invalid.")
      }
      workingSHA = digest
    }
    guard let detectorMaskSHA = parentManifest["detector_mask_sha256"] as? String,
      validSHA256(detectorMaskSHA)
    else {
      throw invalid("Compact prepared DPC parent mask identity is invalid.")
    }
    let scanCount = try multiply(shape[0], shape[1], label: "prepared DPC scan count")
    let detectorPixels = try multiply(
      shape[2],
      shape[3],
      label: "prepared DPC detector pixels"
    )
    let selectedDetectorPixels = detectorPixels - excluded.count
    let excludedSet = Set(excluded)
    var rowCoordinates: UInt64 = 0
    var columnCoordinates: UInt64 = 0
    for pixel in 0..<detectorPixels where !excludedSet.contains(pixel) {
      rowCoordinates = try add(
        rowCoordinates,
        UInt64(pixel / shape[3]),
        label: "prepared DPC row-coordinate sum"
      )
      columnCoordinates = try add(
        columnCoordinates,
        UInt64(pixel % shape[3]),
        label: "prepared DPC column-coordinate sum"
      )
    }
    let totalBound = try multiply(
      UInt64(selectedDetectorPixels),
      maximumValue,
      label: "prepared DPC total bound"
    )
    let rowMomentBound = try multiply(
      rowCoordinates,
      maximumValue,
      label: "prepared DPC row-moment bound"
    )
    let columnMomentBound = try multiply(
      columnCoordinates,
      maximumValue,
      label: "prepared DPC column-moment bound"
    )
    let narrowInteger = totalBound <= UInt64(UInt32.max)
    let narrowProducts = max(rowMomentBound, columnMomentBound) <= UInt64(UInt32.max)
    let expectedLayout = [
      "total_lo", "total_hi", "row_lo", "row_hi",
      "column_lo", "column_hi", "padding_0", "padding_1",
    ]
    var expectedStrings = [
      "schema": schema,
      "source_identity_sha256": sourceIdentity,
      workingField: workingSHA,
      "detector_mask_sha256": detectorMaskSHA,
      "detector_selection": "all-nonexcluded-v1",
      "dtype": "little-endian-u32",
      "word_order": "little-endian-u32-pairs",
      "total_bound": String(totalBound),
      "row_moment_bound": String(rowMomentBound),
      "column_moment_bound": String(columnMomentBound),
    ]
    if workingDtype == "uint16" {
      expectedStrings["working_dtype"] = workingDtype
    }
    for (key, expected) in expectedStrings where prepared[key] as? String != expected {
      throw invalid("Compact prepared DPC field \(key) disagrees with the source.")
    }
    guard intValue(prepared["scan_count"]) == scanCount,
      intValue(prepared["selected_detector_pixels"]) == selectedDetectorPixels,
      intValue(prepared["detector_columns"]) == shape[3],
      intValue(prepared["words_per_scan"]) == 8,
      prepared["narrow_integer"] as? Bool == narrowInteger,
      prepared["narrow_products"] as? Bool == narrowProducts
    else {
      throw invalid("Compact prepared DPC fields disagree with the exact source bounds.")
    }
    if workingDtype == "uint16" {
      guard exactUInt64(prepared["maximum_value"]) == maximumValue else {
        throw invalid("Compact prepared DPC maximum value disagrees with uint16.")
      }
    }
    guard prepared["layout"] as? [String] == expectedLayout else {
      throw invalid("Compact prepared DPC word layout is unsupported.")
    }
    let expectedBytes = try multiply(
      UInt64(scanCount),
      UInt64(8 * MemoryLayout<UInt32>.stride),
      label: "prepared DPC byte count"
    )
    guard let fileOffset = exactUInt64(prepared["file_offset"]),
      let preparedBytes = exactUInt64(prepared["file_bytes"]),
      fileOffset.isMultiple(of: UInt64(MemoryLayout<UInt32>.stride)),
      preparedBytes == expectedBytes,
      fileOffset <= fileBytes,
      preparedBytes <= fileBytes - fileOffset
    else {
      throw invalid("Compact prepared DPC byte range is invalid.")
    }
    let preparedEnd = fileOffset + preparedBytes
    for (shardIndex, shard) in shards.enumerated() {
      for (label, offset, count) in [
        ("payload", shard.payloadOffset, shard.payloadBytes),
        ("lengths", shard.lengthsOffset, shard.lengthsBytes),
        ("headers", shard.widthsOffset, shard.widthsBytes),
      ] where count > 0 && fileOffset < offset + count && offset < preparedEnd {
        throw invalid(
          "Compact prepared DPC range overlaps shard \(shardIndex) \(label)."
        )
      }
    }
    guard let digest = prepared["sha256"] as? String, validSHA256(digest) else {
      throw invalid("Compact prepared DPC SHA-256 is invalid.")
    }
    return MetalCompactH5PreparedDPCMoments(
      fileOffset: fileOffset,
      fileBytes: preparedBytes,
      sha256: digest,
      workingLogicalSHA256: workingSHA,
      workingDtype: workingDtype,
      detectorMaskSHA256: detectorMaskSHA,
      scanCount: scanCount,
      selectedDetectorPixels: selectedDetectorPixels,
      detectorColumns: shape[3],
      totalBound: totalBound,
      rowMomentBound: rowMomentBound,
      columnMomentBound: columnMomentBound,
      narrowInteger: narrowInteger,
      narrowProducts: narrowProducts
    )
  }

  private static func preparedDetectorProducts(
    _ value: Any?,
    parentManifest: [String: Any],
    fileBytes: UInt64,
    shape: [Int],
    sourceIdentity: String,
    storageLayout: CompactH5StorageLayout,
    shards: [CompactH5ShardRecord],
    preparedDPC: MetalCompactH5PreparedDPCMoments?,
    calibration: MetalCompactH5DetectorCalibration?
  ) throws -> MetalCompactH5PreparedDetectorProducts? {
    guard let value, !(value is NSNull) else { return nil }
    guard case .directV3 = storageLayout else {
      throw invalid("Prepared detector products require compact QGIX v3.")
    }
    guard let prepared = value as? [String: Any] else {
      throw invalid("Compact prepared detector products are not an object.")
    }
    guard let calibration,
      let calibrationManifest = parentManifest["detector_calibration"],
      let workingSHA = parentManifest["prepared_uint8_sha256"] as? String,
      validSHA256(workingSHA),
      let detectorMaskSHA = parentManifest["detector_mask_sha256"] as? String,
      validSHA256(detectorMaskSHA)
    else {
      throw invalid("Prepared detector products require source-bound calibration.")
    }
    let calibrationJSON = try canonicalJSON(calibrationManifest)
    guard let calibrationData = calibrationJSON.data(using: .utf8) else {
      throw invalid("Compact detector calibration could not be encoded as UTF-8.")
    }
    let calibrationSHA = SHA256.hash(data: calibrationData)
      .map { String(format: "%02x", $0) }
      .joined()
    let expectedStrings = [
      "schema": "quantem.gpu.prepared-detector-products/v1",
      "source_identity_sha256": sourceIdentity,
      "working_uint8_sha256": workingSHA,
      "detector_mask_sha256": detectorMaskSHA,
      "detector_calibration_sha256": calibrationSHA,
      "detector_calibration_digest_encoding":
        "canonical-json-numbers-as-f64be-hex/v1",
      "product_dtype": "little-endian-u32",
      "mask_dtype": "uint8-binary-row-major",
      "mask_rule": "quantem.gpu.detector-mask-inclusive/v1",
    ]
    for (key, expected) in expectedStrings where prepared[key] as? String != expected {
      throw invalid(
        "Compact prepared detector-products field \(key) disagrees with the source."
      )
    }
    guard intArray(prepared["scan_shape"]) == Array(shape[0...1]),
      intArray(prepared["detector_shape"]) == Array(shape[2...3]),
      prepared["product_order"] as? [String] == ["bf", "abf", "adf"]
    else {
      throw invalid("Compact prepared detector-products shape or order is invalid.")
    }
    guard let rawProducts = prepared["products"] as? [[String: Any]],
      rawProducts.count == 3
    else {
      throw invalid("Compact prepared detector product list is incomplete.")
    }
    let names = ["bf", "abf", "adf"]
    let geometries = [
      (0.0, calibration.brightFieldRadius),
      (0.5 * calibration.brightFieldRadius, calibration.brightFieldRadius),
      (calibration.brightFieldRadius, 2.0 * calibration.brightFieldRadius),
    ]
    let detectorPixels = try multiply(
      shape[2], shape[3], label: "prepared detector pixels"
    )
    let scanCount = try multiply(shape[0], shape[1], label: "prepared detector scans")
    var ranges: [(UInt64, UInt64, String)] = []
    for (shardIndex, shard) in shards.enumerated() {
      for (label, offset, count) in [
        ("payload", shard.payloadOffset, shard.payloadBytes),
        ("lengths", shard.lengthsOffset, shard.lengthsBytes),
        ("headers", shard.widthsOffset, shard.widthsBytes),
      ] where count > 0 {
        ranges.append((offset, offset + count, "shard \(shardIndex) \(label)"))
      }
    }
    if let preparedDPC {
      ranges.append(
        (
          preparedDPC.fileOffset,
          preparedDPC.fileOffset + preparedDPC.fileBytes,
          "prepared DPC moments"
        )
      )
    }
    var products: [MetalCompactH5PreparedDetectorProduct] = []
    for ordinal in names.indices {
      let raw = rawProducts[ordinal]
      let name = names[ordinal]
      let geometry = geometries[ordinal]
      guard raw["name"] as? String == name,
        let center = raw["center_px"] as? [NSNumber], center.count == 2,
        center[0].doubleValue == calibration.detectorCenterRow,
        center[1].doubleValue == calibration.detectorCenterColumn,
        (raw["inner_radius_px"] as? NSNumber)?.doubleValue == geometry.0,
        (raw["outer_radius_px"] as? NSNumber)?.doubleValue == geometry.1
      else {
        throw invalid(
          "Compact prepared \(name.uppercased()) geometry disagrees with calibration."
        )
      }
      guard let selected = exactUInt64(raw["selected_detector_pixels"]),
        selected <= UInt64(detectorPixels),
        let maskOffset = exactUInt64(raw["mask_file_offset"]),
        let maskBytes = exactUInt64(raw["mask_file_bytes"]),
        maskBytes == UInt64(detectorPixels),
        maskOffset <= fileBytes,
        maskBytes <= fileBytes - maskOffset,
        let valuesOffset = exactUInt64(raw["values_file_offset"]),
        let valuesBytes = exactUInt64(raw["values_file_bytes"]),
        valuesBytes == UInt64(scanCount * MemoryLayout<UInt32>.stride),
        valuesOffset <= fileBytes,
        valuesBytes <= fileBytes - valuesOffset
      else {
        throw invalid(
          "Compact prepared \(name.uppercased()) byte ranges are invalid."
        )
      }
      guard let maskSHA = raw["mask_sha256"] as? String, validSHA256(maskSHA),
        let valuesSHA = raw["values_sha256"] as? String, validSHA256(valuesSHA),
        let selectedInt = Int(exactly: selected)
      else {
        throw invalid("Compact prepared \(name.uppercased()) SHA-256 is invalid.")
      }
      ranges.append(
        (maskOffset, maskOffset + maskBytes, "prepared \(name) mask")
      )
      ranges.append(
        (valuesOffset, valuesOffset + valuesBytes, "prepared \(name) values")
      )
      products.append(
        MetalCompactH5PreparedDetectorProduct(
          name: name,
          centerRow: calibration.detectorCenterRow,
          centerColumn: calibration.detectorCenterColumn,
          innerRadius: geometry.0,
          outerRadius: geometry.1,
          selectedDetectorPixels: selectedInt,
          maskFileOffset: maskOffset,
          maskFileBytes: maskBytes,
          maskSHA256: maskSHA,
          valuesFileOffset: valuesOffset,
          valuesFileBytes: valuesBytes,
          valuesSHA256: valuesSHA
        )
      )
    }
    ranges.sort { lhs, rhs in lhs.0 < rhs.0 }
    for index in 1..<ranges.count where ranges[index].0 < ranges[index - 1].1 {
      throw invalid(
        "Compact ranges overlap between \(ranges[index - 1].2) and \(ranges[index].2)."
      )
    }
    return MetalCompactH5PreparedDetectorProducts(
      calibrationSHA256: calibrationSHA,
      workingUInt8SHA256: workingSHA,
      detectorMaskSHA256: detectorMaskSHA,
      products: products
    )
  }

  private static func canonicalJSON(_ value: Any) throws -> String {
    if value is NSNull { return "null" }
    if let string = value as? String { return pythonJSONString(string) }
    if let number = value as? NSNumber {
      if CFGetTypeID(number) == CFBooleanGetTypeID() {
        return number.boolValue ? "true" : "false"
      }
      guard number.doubleValue.isFinite else {
        throw invalid("Compact detector calibration contains a non-finite number.")
      }
      var bits = number.doubleValue.bitPattern.bigEndian
      let hex = withUnsafeBytes(of: &bits) {
        $0.map { String(format: "%02x", $0) }.joined()
      }
      return pythonJSONString("f64be:\(hex)")
    }
    if let array = value as? [Any] {
      return "[" + (try array.map(canonicalJSON).joined(separator: ",")) + "]"
    }
    if let object = value as? [String: Any] {
      let entries = try object.keys.sorted().map { key in
        pythonJSONString(key) + ":" + (try canonicalJSON(object[key] as Any))
      }
      return "{" + entries.joined(separator: ",") + "}"
    }
    throw invalid("Compact detector calibration contains an unsupported JSON value.")
  }

  private static func pythonJSONString(_ value: String) -> String {
    var result = "\""
    for scalar in value.unicodeScalars {
      switch scalar.value {
      case 0x08: result += "\\b"
      case 0x09: result += "\\t"
      case 0x0a: result += "\\n"
      case 0x0c: result += "\\f"
      case 0x0d: result += "\\r"
      case 0x22: result += "\\\""
      case 0x5c: result += "\\\\"
      case 0x00...0x1f:
        result += String(format: "\\u%04x", scalar.value)
      case 0x20...0x7e:
        result.unicodeScalars.append(scalar)
      case 0x80...0xffff:
        result += String(format: "\\u%04x", scalar.value)
      default:
        let adjusted = scalar.value - 0x1_0000
        let high = 0xd800 + (adjusted >> 10)
        let low = 0xdc00 + (adjusted & 0x3ff)
        result += String(format: "\\u%04x\\u%04x", high, low)
      }
    }
    return result + "\""
  }

  private static func detectorCalibration(
    _ value: Any?,
    sourceIdentity: String,
    detectorRows: Int,
    detectorColumns: Int
  ) throws -> MetalCompactH5DetectorCalibration? {
    guard let value, !(value is NSNull) else { return nil }
    guard let calibration = value as? [String: Any] else {
      throw invalid("Compact detector calibration is not an object.")
    }
    guard
      calibration["schema"] as? String
        == "quantem.gpu.detector-calibration/v1",
      calibration["source_identity_sha256"] as? String == sourceIdentity
    else {
      throw invalid("Compact detector calibration schema or source identity is invalid.")
    }
    guard let center = calibration["detector_center_px"] as? [NSNumber],
      center.count == 2,
      center.allSatisfy({
        CFGetTypeID($0) != CFBooleanGetTypeID() && $0.doubleValue.isFinite
      }),
      center[0].doubleValue >= 0,
      center[0].doubleValue < Double(detectorRows),
      center[1].doubleValue >= 0,
      center[1].doubleValue < Double(detectorColumns),
      let radius = calibration["bright_field_radius_px"] as? NSNumber,
      CFGetTypeID(radius) != CFBooleanGetTypeID(),
      radius.doubleValue.isFinite,
      radius.doubleValue > 0,
      radius.doubleValue <= hypot(Double(detectorRows), Double(detectorColumns)),
      let method = calibration["method"] as? String,
      !method.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else {
      throw invalid("Compact detector center, bright-field radius, or method is invalid.")
    }
    let rotationValue = calibration["dpc_rotation_degrees"]
    let exchangeValue = calibration["dpc_component_order_exchanged"]
    let hasRotation = rotationValue != nil && !(rotationValue is NSNull)
    let hasExchange = exchangeValue != nil && !(exchangeValue is NSNull)
    guard hasRotation == hasExchange else {
      throw invalid("Compact DPC calibration requires rotation and component order together.")
    }
    var rotation: Double?
    var exchanged: Bool?
    if hasRotation {
      guard let number = rotationValue as? NSNumber,
        CFGetTypeID(number) != CFBooleanGetTypeID(),
        number.doubleValue.isFinite,
        let flag = exchangeValue as? Bool
      else {
        throw invalid("Compact DPC calibration is invalid.")
      }
      rotation = number.doubleValue
      exchanged = flag
    }
    return MetalCompactH5DetectorCalibration(
      detectorCenterRow: center[0].doubleValue,
      detectorCenterColumn: center[1].doubleValue,
      brightFieldRadius: radius.doubleValue,
      dpcRotationDegrees: rotation,
      dpcComponentOrderExchanged: exchanged,
      method: method
    )
  }

  private static func intArray(_ value: Any?) -> [Int]? {
    guard let raw = value as? [Any] else { return nil }
    let result = raw.compactMap { intValue($0) }
    return result.count == raw.count ? result : nil
  }

  private static func exactUInt64(_ value: Any?) -> UInt64? {
    guard let number = value as? NSNumber,
      CFGetTypeID(number) != CFBooleanGetTypeID(),
      number.int64Value >= 0,
      NSNumber(value: number.uint64Value) == number
    else {
      return nil
    }
    return number.uint64Value
  }

  private static func coordinateArray(_ value: Any?) -> [(Int, Int)]? {
    guard let rows = value as? [Any] else { return nil }
    let result = rows.compactMap { row -> (Int, Int)? in
      guard let values = intArray(row), values.count == 2 else { return nil }
      return (values[0], values[1])
    }
    return result.count == rows.count ? result : nil
  }

  private static func exactInt(_ value: UInt64, label: String) throws -> Int {
    guard let result = Int(exactly: value) else {
      throw invalid("Compact \(label) exceeds the host integer range.")
    }
    return result
  }

  private static func multiply(
    _ lhs: Int,
    _ rhs: Int,
    label: String
  ) throws -> Int {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else { throw invalid("Compact \(label) overflowed.") }
    return result.partialValue
  }

  private static func multiply(
    _ lhs: UInt64,
    _ rhs: UInt64,
    label: String
  ) throws -> UInt64 {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else { throw invalid("Compact \(label) overflowed.") }
    return result.partialValue
  }

  private static func add(
    _ lhs: Int,
    _ rhs: Int,
    label: String
  ) throws -> Int {
    let result = lhs.addingReportingOverflow(rhs)
    guard !result.overflow else { throw invalid("Compact \(label) overflowed.") }
    return result.partialValue
  }

  private static func add(
    _ lhs: UInt32,
    _ rhs: UInt32,
    label: String
  ) throws -> UInt32 {
    let result = lhs.addingReportingOverflow(rhs)
    guard !result.overflow else { throw invalid("Compact \(label) overflowed.") }
    return result.partialValue
  }

  private static func add(
    _ lhs: UInt64,
    _ rhs: UInt64,
    label: String
  ) throws -> UInt64 {
    let result = lhs.addingReportingOverflow(rhs)
    guard !result.overflow else { throw invalid("Compact \(label) overflowed.") }
    return result.partialValue
  }

  private static func invalid(_ message: String) -> Metal4DSTEMStreamingIOError {
    .invalidRequest(message)
  }

  private static func lastPOSIXError() -> String {
    String(cString: strerror(errno))
  }

  private static func milliseconds(from start: ContinuousClock.Instant) -> Double {
    MetalCompactH5ResidentSource.milliseconds(from: start)
  }
}
