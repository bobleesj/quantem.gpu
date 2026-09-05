import CoreFoundation
import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMKernels

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
  public let workingLogicalSHA256: String
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
  public var workingUInt8SHA256: String { workingLogicalSHA256 }
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
  public let mappedAuthenticationBytes: UInt64
  public let preparedDPCBytes: UInt64
  public let preparedDetectorProductBytes: UInt64
  public let interactionResidentBytes: UInt64
  public let totalResidentBytes: UInt64
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
}

private struct CompactDetectorEntry {
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
}

private struct CompactFullDecodeParameters {
  var scanCount: UInt32
  var pixelCount: UInt32
  var tileCount: UInt32
  var scanTile: UInt32
  var headerWordsPerPixel: UInt32
  var headerEncoding: UInt32
  var outputWordCount: UInt32
}

private struct CompactDetectorSumParameters {
  var scanCount: UInt32
  var tileCount: UInt32
  var pixelCount: UInt32
  var scanTile: UInt32
  var headerWordsPerPixel: UInt32
  var headerEncoding: UInt32
}

private struct CompactResidentShard {
  let payload: MTLBuffer
  let descriptors: MTLBuffer
}

/// Workers never access this buffer on the CPU. Metal command buffers use
/// tracked resources and the validation shader writes only atomic maxima.
/// The caller reads it only after every bounded window has joined.
private struct CompactConcurrentWidthValidation: @unchecked Sendable {
  let buffer: MTLBuffer
}

private struct CompactShardLoadResult {
  let resident: CompactResidentShard
  let sourceReadMilliseconds: Double
  let descriptorPreparationMilliseconds: Double
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
  private let fullDecodePipeline: MTLComputePipelineState
  private let detectorSumPipeline: MTLComputePipelineState
  private let headerEncoding: UInt32
  private let headerWordsPerPixel: UInt32
  private var shards: [CompactResidentShard]
  private var excluded: MTLBuffer?
  private let maximumWidths: [UInt8]
  private var detectorOutputs: [MTLBuffer]
  private var detectorEntryBuffer: MTLBuffer?
  private var diffractionOutput: MTLBuffer?
  private var detectorSumOutput: MTLBuffer?
  private var preparedDPCMomentBuffer: MTLBuffer?
  private var preparedDPCOutputs: [MTLBuffer]
  private var preparedDetectorProducts: [String: CompactPreparedDetectorResident]
  private var activeDetectorOutput = 0
  private var detectorMask: [UInt8]
  private var detectorSumReady = false
  public private(set) var isReleased = false

  fileprivate init(
    metadata: MetalCompactH5Metadata,
    loadMetrics: MetalCompactH5LoadMetrics,
    device: MTLDevice,
    queue: MTLCommandQueue,
    selectedPipeline: MTLComputePipelineState,
    detectorPipeline: MTLComputePipelineState,
    fullDecodePipeline: MTLComputePipelineState,
    detectorSumPipeline: MTLComputePipelineState,
    headerEncoding: UInt32,
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
    preparedDetectorProducts: [String: CompactPreparedDetectorResident]
  ) {
    self.metadata = metadata
    self.loadMetrics = loadMetrics
    self.device = device
    self.queue = queue
    self.selectedPipeline = selectedPipeline
    self.detectorPipeline = detectorPipeline
    self.fullDecodePipeline = fullDecodePipeline
    self.detectorSumPipeline = detectorSumPipeline
    self.headerEncoding = headerEncoding
    self.headerWordsPerPixel = headerWordsPerPixel
    self.shards = shards
    self.excluded = excluded
    self.maximumWidths = maximumWidths
    self.detectorOutputs = detectorOutputs
    self.detectorEntryBuffer = detectorEntryBuffer
    self.diffractionOutput = diffractionOutput
    self.detectorSumOutput = detectorSumOutput
    self.preparedDPCMomentBuffer = preparedDPCMomentBuffer
    self.preparedDPCOutputs = preparedDPCOutputs
    self.preparedDetectorProducts = preparedDetectorProducts
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
    guard !isReleased, headerEncoding == 0,
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
    guard !isReleased, let excluded, let diffractionOutput,
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
      headerEncoding: headerEncoding
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
    encoder.setBuffer(diffractionOutput, offset: 0, index: 3)
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
    try Self.complete(command, operation: "selected diffraction")
    return Self.u32Values(
      diffractionOutput,
      count: metadata.detectorPixelCount
    )
  }

  /// Return the exact detector sum and float32 mean diffraction pattern.
  ///
  /// The first call performs one complete resident pass. Later calls reuse the
  /// synchronized u64 result. No dense 4D tensor is decoded or allocated.
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
          headerEncoding: headerEncoding
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
  /// The first call is a full rebase. Later calls use signed mask deltas. A mask
  /// that could overflow a u32 accumulator is rejected before GPU execution.
  @discardableResult
  public func updateVirtualDetector(
    mask: [UInt8],
    forceRebase: Bool = false
  ) throws -> MetalCompactH5DetectorMetrics {
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
    let excludedSet = Set(metadata.excludedDetectorPixels)
    var normalized = mask
    for pixel in excludedSet { normalized[pixel] = 0 }
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

    if let prepared = preparedDetectorProducts.values.first(where: {
      $0.mask == normalized
    }) {
      let changedCount = zip(normalized, detectorMask).count { $0 != $1 }
      if changedCount == 0 {
        return MetalCompactH5DetectorMetrics(
          mode: "delta",
          changedDetectorPixels: 0,
          wallMilliseconds: 0,
          gpuMilliseconds: 0,
          fftDispatchCount: 0
        )
      }
      let nextOutput = 1 - activeDetectorOutput
      let wallStart = ContinuousClock.now
      guard let command = queue.makeCommandBuffer(),
        let blit = command.makeBlitCommandEncoder()
      else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Metal could not encode prepared virtual-detector activation."
        )
      }
      blit.copy(
        from: prepared.values,
        sourceOffset: 0,
        to: detectorOutputs[nextOutput],
        destinationOffset: 0,
        size: metadata.scanCount * MemoryLayout<UInt32>.stride
      )
      blit.endEncoding()
      try Self.complete(command, operation: "prepared virtual detector")
      activeDetectorOutput = nextOutput
      detectorMask = normalized
      return MetalCompactH5DetectorMetrics(
        mode: "prepared",
        changedDetectorPixels: changedCount,
        wallMilliseconds: Self.milliseconds(from: wallStart),
        gpuMilliseconds: max(0, command.gpuEndTime - command.gpuStartTime) * 1_000,
        fftDispatchCount: 0
      )
    }

    let isRebase = forceRebase || detectorMask.allSatisfy({ $0 == 0 })
    var entries: [CompactDetectorEntry] = []
    entries.reserveCapacity(metadata.detectorPixelCount)
    for pixel in normalized.indices {
      if isRebase {
        if normalized[pixel] != 0 {
          entries.append(
            CompactDetectorEntry(pixel: UInt32(pixel), coefficient: 1)
          )
        }
      } else if normalized[pixel] != detectorMask[pixel] {
        entries.append(
          CompactDetectorEntry(
            pixel: UInt32(pixel),
            coefficient: normalized[pixel] == 0 ? -1 : 1
          )
        )
      }
    }
    if entries.isEmpty {
      detectorMask = normalized
      return MetalCompactH5DetectorMetrics(
        mode: isRebase ? "rebase" : "delta",
        changedDetectorPixels: 0,
        wallMilliseconds: 0,
        gpuMilliseconds: 0,
        fftDispatchCount: 0
      )
    }
    _ = entries.withUnsafeBytes { raw in
      memcpy(detectorEntryBuffer.contents(), raw.baseAddress!, raw.count)
    }
    let nextOutput = 1 - activeDetectorOutput
    let wallStart = ContinuousClock.now
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode a compact virtual-detector update."
      )
    }
    encoder.setComputePipelineState(detectorPipeline)
    for shardIndex in shards.indices {
      var parameters = CompactDetectorParameters(
        scanCount: UInt32(metadata.scansPerShard),
        tileCount: UInt32(
          (metadata.scansPerShard + metadata.scanTile - 1) / metadata.scanTile
        ),
        entryCount: UInt32(entries.count),
        outputOffset: UInt32(shardIndex * metadata.scansPerShard),
        mode: isRebase ? 1 : 0,
        scanTile: UInt32(metadata.scanTile),
        headerWordsPerPixel: headerWordsPerPixel,
        headerEncoding: headerEncoding
      )
      encoder.setBuffer(shards[shardIndex].payload, offset: 0, index: 0)
      encoder.setBuffer(shards[shardIndex].descriptors, offset: 0, index: 1)
      encoder.setBuffer(detectorEntryBuffer, offset: 0, index: 2)
      encoder.setBuffer(detectorOutputs[activeDetectorOutput], offset: 0, index: 3)
      encoder.setBuffer(detectorOutputs[nextOutput], offset: 0, index: 4)
      encoder.setBytes(
        &parameters,
        length: MemoryLayout.stride(ofValue: parameters),
        index: 5
      )
      encoder.dispatchThreadgroups(
        MTLSize(
          width: (metadata.scansPerShard + 31) / 32,
          height: 1,
          depth: 1
        ),
        threadsPerThreadgroup: MTLSize(width: 128, height: 1, depth: 1)
      )
    }
    encoder.endEncoding()
    try Self.complete(command, operation: "virtual detector")
    activeDetectorOutput = nextOutput
    detectorMask = normalized
    return MetalCompactH5DetectorMetrics(
      mode: isRebase ? "rebase" : "delta",
      changedDetectorPixels: entries.count,
      wallMilliseconds: Self.milliseconds(from: wallStart),
      gpuMilliseconds: max(0, command.gpuEndTime - command.gpuStartTime) * 1_000,
      fftDispatchCount: 0
    )
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
      detectorColumnMoment: column
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
        outputWordCount: outputWords
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
    shards.removeAll(keepingCapacity: false)
    detectorOutputs.removeAll(keepingCapacity: false)
    detectorEntryBuffer = nil
    excluded = nil
    diffractionOutput = nil
    detectorSumOutput = nil
    preparedDPCMomentBuffer = nil
    preparedDPCOutputs.removeAll(keepingCapacity: false)
    preparedDetectorProducts.removeAll(keepingCapacity: false)
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
    operation: String
  ) throws {
    command.commit()
    command.waitUntilCompleted()
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
    let plannedAdditionalBytes = try plannedAdditionalBytes(
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
    let decode = try pipeline(
      library: library,
      name: Metal4DSTEMKernels.compactH5DecodeFunction,
      device: device
    )
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
                index: index, device: device, queue: queue,
                validationPipeline: validateDescriptors,
                maximumWidthBuffer: concurrentWidths.buffer,
                payloadPreauthenticated: payloadsPreauthenticated,
                verifyChecksums: verifyChecksums
              )
            }
            return try loadCompressedShard(
              descriptor: descriptor, shardIndex: shardIndex, shard: shard, index: index,
              excludedSet: excludedSet, device: device, queue: queue, decode: decode,
              validateDescriptors: validateDescriptors, maximumWidthBuffer: concurrentWidths.buffer,
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
    let interactionResidentBytes = UInt64(
      excludedValues.count * 4 + scanMapBytes * 2 + diffractionBytes
        + detectorSumBytes
        + index.metadata.detectorPixelCount * MemoryLayout<CompactDetectorEntry>.stride
        + preparedDPCDisplayBytes
    )
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
      mappedAuthenticationBytes: mappedAuthenticationBytes,
      preparedDPCBytes: index.metadata.preparedDPCMoments?.fileBytes ?? 0,
      preparedDetectorProductBytes: preparedDetectorProducts.bytes,
      interactionResidentBytes: interactionResidentBytes,
      totalResidentBytes: index.metadata.residentBytes + interactionResidentBytes
    )
    return MetalCompactH5ResidentSource(
      metadata: index.metadata,
      loadMetrics: metrics,
      device: device,
      queue: queue,
      selectedPipeline: selected,
      detectorPipeline: detector,
      fullDecodePipeline: fullDecode,
      detectorSumPipeline: detectorSum,
      headerEncoding: index.headerEncoding,
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
      preparedDetectorProducts: preparedDetectorProducts.products
    )
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
        // Arrays and their copied shared Metal buffers coexist until the shard
        // autorelease pool drains: two descriptor and two chunk-table copies.
        staging = try add(
          staging,
          try multiply(UInt64(shard.descriptorCount), UInt64(8), label: "descriptor staging"),
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
      staging = try add(staging, UInt64(4), label: "descriptor validation status")
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
        by: UInt64(index.metadata.detectorRows - 1)
      )
      let columnLimit = total.multipliedReportingOverflow(
        by: UInt64(index.metadata.detectorColumns - 1)
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
    excludedSet: Set<Int>,
    device: MTLDevice,
    queue: MTLCommandQueue,
    decode: MTLComputePipelineState,
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
          + descriptorBytes * 2 + chunkBytes * 2 + Int(shard.chunkCount) * 4
          + (4 - payloadBytes % 4) % 4 + 4
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
        options: .storageModeShared
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
    var lengths = [UInt8](repeating: 0, count: lengthsBytes)
    try lengths.withUnsafeMutableBytes { raw in
      try preadExact(
        descriptor,
        offset: shard.lengthsOffset,
        into: raw.baseAddress!,
        byteCount: raw.count,
        label: "shard \(shardIndex) chunk lengths"
      )
    }
    var widths = [UInt8](repeating: 0, count: widthsBytes)
    try widths.withUnsafeMutableBytes { raw in
      try preadExact(
        descriptor,
        offset: shard.widthsOffset,
        into: raw.baseAddress!,
        byteCount: raw.count,
        label: "shard \(shardIndex) descriptor widths"
      )
    }
    sourceReadMilliseconds += milliseconds(from: readStart)

    let preparationStart = ContinuousClock.now
    let tileCount = (index.metadata.scansPerShard + 127) / 128
    var descriptors = [UInt32]()
    descriptors.reserveCapacity(widths.count)
    var payloadWord: UInt32 = 0
    for (descriptorIndex, widthByte) in widths.enumerated() {
      let width = UInt32(widthByte)
      guard width <= 16 else {
        throw invalid(
          "Compact shard \(shardIndex) descriptor \(descriptorIndex) requires "
            + "\(width) bits, beyond exact uint16."
        )
      }
      let pixel = descriptorIndex / tileCount
      if index.manifestWorkingDtype == "uint8", width > 8,
        !excludedSet.contains(pixel)
      {
        throw invalid(
          "Compact shard \(shardIndex) uses \(width) bits for nonexcluded "
            + "detector pixel \(pixel), contradicting its legacy uint8 manifest."
        )
      }
      guard payloadWord < (1 << 27) else {
        throw invalid(
          "Compact shard \(shardIndex) exceeds the 27-bit descriptor offset."
        )
      }
      descriptors.append((payloadWord << 5) | width)
      payloadWord = try add(
        payloadWord,
        width * 4,
        label: "compact payload word offset"
      )
    }
    guard UInt64(payloadWord) * 4 == shard.decodedBytes else {
      throw invalid(
        "Compact shard \(shardIndex) widths cover \(UInt64(payloadWord) * 4) "
          + "decoded bytes, expected \(shard.decodedBytes)."
      )
    }
    var chunks: [CompactLZ4Chunk] = []
    chunks.reserveCapacity(lengths.count)
    var inputOffset: UInt32 = 0
    for (chunkIndex, encodedLength) in lengths.enumerated() {
      let inputBytes = UInt32(encodedLength) + 1
      let outputOffset = chunkIndex * index.metadata.payloadChunkBytes
      chunks.append(
        CompactLZ4Chunk(
          inputOffset: inputOffset,
          inputBytes: inputBytes,
          outputWord: UInt32(outputOffset / 4),
          outputBytes: UInt32(
            min(index.metadata.payloadChunkBytes, decodedBytes - outputOffset)
          )
        )
      )
      inputOffset = try add(
        inputOffset,
        inputBytes,
        label: "compressed chunk offset"
      )
    }
    guard UInt64(inputOffset) == shard.payloadBytes else {
      throw invalid(
        "Compact shard \(shardIndex) chunk lengths cover \(inputOffset) bytes, "
          + "expected \(shard.payloadBytes)."
      )
    }
    guard
      let chunkBuffer = chunks.withUnsafeBytes({ raw in
        device.makeBuffer(
          bytes: raw.baseAddress!,
          length: raw.count,
          options: .storageModeShared
        )
      }),
      let descriptorStage = descriptors.withUnsafeBytes({ raw in
        device.makeBuffer(
          bytes: raw.baseAddress!,
          length: raw.count,
          options: .storageModeShared
        )
      }),
      let decodeStatus = device.makeBuffer(
        length: Int(shard.chunkCount) * MemoryLayout<UInt32>.stride,
        options: .storageModeShared
      )
    else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "compact metadata staging for shard \(shardIndex)",
        bytes: UInt64(chunkBytes + descriptorBytes + Int(shard.chunkCount) * 4)
      )
    }
    // The Metal decoder initializes its own output words; do not touch the
    // multi-gigabyte decoded staging volume on the CPU before GPU decode.
    memset(decodeStatus.contents(), 0xff, decodeStatus.length)
    descriptorPreparationMilliseconds += milliseconds(from: preparationStart)

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
      MTLSize(width: Int(shard.chunkCount), height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 64, height: 1, depth: 1)
    )
    decodeEncoder.endEncoding()
    try MetalCompactH5ResidentSource.complete(
      decodeCommand,
      operation: "shard \(shardIndex) raw LZ4 decode"
    )
    gpuDecodeMilliseconds +=
      max(
        0,
        decodeCommand.gpuEndTime - decodeCommand.gpuStartTime
      ) * 1_000
    let statuses = decodeStatus.contents().bindMemory(
      to: UInt32.self,
      capacity: Int(shard.chunkCount)
    )
    for chunkIndex in 0..<Int(shard.chunkCount) where statuses[chunkIndex] != 0 {
      throw invalid(
        "Compact shard \(shardIndex) raw LZ4 chunk \(chunkIndex) failed "
          + "with decoder status \(statuses[chunkIndex])."
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

    guard
      let privatePayload = device.makeBuffer(
        length: decodedBytes,
        options: .storageModePrivate
      ),
      let privateDescriptors = device.makeBuffer(
        length: descriptorBytes,
        options: .storageModePrivate
      ),
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
    guard let uploadCommand = queue.makeCommandBuffer(),
      let blit = uploadCommand.makeBlitCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "Metal could not encode compact shard \(shardIndex) private upload."
      )
    }
    blit.copy(
      from: decodedStage,
      sourceOffset: 0,
      to: privatePayload,
      destinationOffset: 0,
      size: decodedBytes
    )
    blit.copy(
      from: descriptorStage,
      sourceOffset: 0,
      to: privateDescriptors,
      destinationOffset: 0,
      size: descriptorBytes
    )
    blit.endEncoding()
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
        || (index.headerEncoding == 1 && index.metadata.scanTile == 32
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
    let headerEncoding: UInt32
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
      guard secondHeaderWord == 0, scanTile == 32, encodedHeaderLayout == 1 else {
        throw invalid("Compact QGIX v3 header layout is unsupported.")
      }
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
        validSHA256(manifest["prepared_uint8_sha256"])
      else {
        throw invalid("Compact QGIX v3 JSON contract is incomplete or unsupported.")
      }
      if manifest["source_dtype"] != nil {
        guard manifest["source_dtype"] as? String == "uint16",
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
    } else if workingDtype != "uint8" {
      throw invalid("Compact QGIX v3 direct payload requires exact uint8 working values.")
    }
    let sourceRawSHA = manifest["source_raw_logical_sha256"] as? String
    guard sourceRawSHA == nil || validSHA256(sourceRawSHA) else {
      throw invalid("Compact source logical SHA-256 is malformed.")
    }
    let workingLogicalSHA: String? =
      storageLayout == .lz4V1
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
    switch storageLayout {
    case .lz4V1:
      guard parentManifest["working_dtype"] as? String == "uint16" else {
        throw invalid("Compact prepared DPC moments require uint16 working data.")
      }
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
    case .directV3:
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
    if case .lz4V1 = storageLayout {
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
    if case .lz4V1 = storageLayout {
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
