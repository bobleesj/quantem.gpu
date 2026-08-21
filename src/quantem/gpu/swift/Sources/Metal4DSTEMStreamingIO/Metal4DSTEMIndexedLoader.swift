import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

/// Fail-closed errors from bounded indexed 4D-STEM loading.
public enum Metal4DSTEMStreamingIOError: LocalizedError, Equatable {
  case invalidRequest(String)
  case allocationFailed(label: String, bytes: UInt64)
  case metalUnavailable(String)
  case commandFailed(String)
  case cancelled

  public var errorDescription: String? {
    switch self {
    case .invalidRequest(let message):
      message
    case .allocationFailed(let label, let bytes):
      "Metal could not allocate \(bytes) bytes for \(label). Choose a smaller "
        + "decoded-window ceiling without changing scientific coverage."
    case .metalUnavailable(let message):
      "Metal 4D-STEM streaming is unavailable: \(message)"
    case .commandFailed(let message):
      "Metal 4D-STEM streaming failed: \(message)"
    case .cancelled:
      "The indexed 4D-STEM load was cancelled between exact decode windows."
    }
  }
}

/// Three caller-defined detector regions encoded as independent bits.
///
/// Bit 0 contributes to `band1`, bit 1 to `band2`, and bit 2 to `band4`.
/// Regions may overlap. QuantEM.GPU validates and sums the supplied regions;
/// it does not derive scientific detector geometry or application policy.
public struct Metal4DSTEMDetectorBands: Codable, Equatable, Sendable {
  public static let band1: UInt8 = 1
  public static let band2: UInt8 = 2
  public static let band4: UInt8 = 4

  public let detectorRows: Int
  public let detectorColumns: Int
  public let membership: [UInt8]

  public init(
    detectorRows: Int,
    detectorColumns: Int,
    membership: [UInt8]
  ) throws {
    guard detectorRows > 0, detectorColumns > 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Detector-band shape must contain positive row and column counts."
      )
    }
    let pixels = detectorRows.multipliedReportingOverflow(by: detectorColumns)
    guard !pixels.overflow, membership.count == pixels.partialValue else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Detector-band membership must contain one byte for every detector pixel."
      )
    }
    guard membership.allSatisfy({ $0 & ~UInt8(7) == 0 }) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Detector-band membership uses unsupported bits; only 1, 2, and 4 are valid."
      )
    }
    self.detectorRows = detectorRows
    self.detectorColumns = detectorColumns
    self.membership = membership
  }
}

/// A caller-inspectable, policy-free memory plan for exact indexed loading.
///
/// The plan bounds transient decoded storage and reports all package-owned
/// buffers. It does not decide whether a device should admit the operation.
public struct Metal4DSTEMIndexedLoadPlan: Equatable, Sendable {
  public static let defaultMaximumInFlightCommandBuffers = 4

  public let datasetID: String
  public let sourceIdentitySHA256: String
  public let sourceScanRows: Int
  public let sourceScanColumns: Int
  public let sourceDetectorRows: Int
  public let sourceDetectorColumns: Int
  public let logicalFrameCount: Int
  public let logicalDecodedBytes: UInt64
  public let compressedSourceBytes: UInt64
  public let sourceDtype: Metal4DSTEMIntegerDType
  public let stagingDtype: Metal4DSTEMIntegerDType
  public let scanBin: Int
  public let detectorBin: Int
  public let maximumDecodedWindowBytes: UInt64
  public let maximumActualWindowBytes: UInt64
  public let persistentOutputBytes: UInt64
  public let auditAndMaskBytes: UInt64
  public let maximumMetadataBytes: UInt64
  /// Allocated buffers, excluding the current no-copy compressed-file mapping.
  public let estimatedAllocatedMetalBytesExcludingMappedSource: UInt64
  public let maximumIndividualMetalBufferBytes: UInt64
  public let maximumMappedCompressedBytes: UInt64
  public let maximumMappedSourceBufferBytes: UInt64
  public let maximumInFlightMappedSourceBytes: UInt64
  public let maximumInFlightCommandBuffers: Int
  public let windows: [Native4DSTEMIndexedWindow]
  public let detectorBands: Metal4DSTEMDetectorBands
  public let detectorBandsSHA256: String
  public let badPixelIndices: [Int]
  public let scanCalibration: Native4DSTEMScanCalibration?
  public let scanSamplingRowNanometer: Double?
  public let scanSamplingColumnNanometer: Double?
  public let detectorSamplingRow: Double?
  public let detectorSamplingColumn: Double?
  public let detectorSamplingUnit: String?

  public init(
    source: Native4DSTEMIndexedSource,
    maximumDecodedWindowBytes: UInt64,
    detectorBands: Metal4DSTEMDetectorBands
  ) throws {
    let dataset = source.dataset
    guard dataset.sourceDtype == Metal4DSTEMIntegerDType.uint16.rawValue,
      source.sourceBytesPerValue == Metal4DSTEMIntegerDType.uint16.bytesPerValue
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Native indexed streaming currently requires a uint16 source and uint16 staging."
      )
    }
    guard detectorBands.detectorRows == dataset.detectorRows,
      detectorBands.detectorColumns == dataset.detectorCols
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Detector-band shape must match the native source detector without crop or binning."
      )
    }
    guard let sourceIdentity = dataset.sourceIdentitySHA256,
      Self.isSHA256(sourceIdentity)
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Exact indexed loading requires a lowercase SHA-256 source identity. "
          + "Prepare the indexed catalog with source hashing enabled."
      )
    }
    guard dataset.sourceBytes >= 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Compressed source bytes cannot be negative. Rebuild the indexed catalog."
      )
    }
    let detectorPixels = try Self.product(
      UInt64(dataset.detectorRows),
      UInt64(dataset.detectorCols),
      label: "detector pixels"
    )
    let frames = UInt64(source.logicalFrameCount)
    guard UInt32(exactly: source.logicalFrameCount) != nil,
      UInt32(exactly: detectorPixels) != nil,
      UInt32(exactly: dataset.detectorCols) != nil
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Native indexed geometry exceeds the reusable Metal kernel's 32-bit index range."
      )
    }
    try Self.validateExactProductBounds(
      detectorRows: dataset.detectorRows,
      detectorColumns: dataset.detectorCols,
      logicalFrameCount: source.logicalFrameCount
    )
    let sortedBadPixels = dataset.badPixelIndices.sorted()
    guard Set(sortedBadPixels).count == sortedBadPixels.count,
      sortedBadPixels.allSatisfy({ $0 >= 0 && UInt64($0) < detectorPixels })
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Bad-pixel indices must be unique native detector offsets."
      )
    }
    let windows = try source.windows(
      maximumDecodedBytes: maximumDecodedWindowBytes,
      alignToScanRows: true
    )
    guard !windows.isEmpty else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The decoded-window plan contains no logical scan frames."
      )
    }
    for window in windows {
      guard UInt32(exactly: window.globalFrameRange.lowerBound) != nil,
        UInt32(exactly: window.globalFrameRange.count) != nil
      else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "A decoded frame window exceeds the reusable Metal kernel's 32-bit range."
        )
      }
      for slice in window.slices {
        guard UInt32(exactly: slice.globalFrameRange.lowerBound) != nil,
          UInt32(exactly: slice.globalFrameRange.count) != nil,
          slice.metadataWordRange.count.isMultiple(of: 2)
        else {
          throw Metal4DSTEMStreamingIOError.invalidRequest(
            "A prepared QH5 slice exceeds the reusable Metal kernel's index range."
          )
        }
      }
    }
    let mapBytes = try Self.product(frames, 8, label: "one exact scan map")
    let allMaps = try Self.product(mapBytes, 6, label: "six exact scan maps")
    let detectorSumBytes = try Self.product(
      detectorPixels, 8, label: "exact detector sum"
    )
    let persistent = try Self.sum(
      [allMaps, detectorSumBytes], label: "exact outputs"
    )
    let auditBytes = try Self.product(frames, 8, label: "count audit")
    let maskBytes = try Self.product(detectorPixels, 2, label: "detector masks")
    let auditAndMasks = try Self.sum(
      [auditBytes, maskBytes], label: "audit and masks"
    )
    let maximumMetadata = try windows.flatMap(\.slices).reduce(UInt64(0)) {
      maximum, slice in
      max(
        maximum,
        try Self.product(
          UInt64(slice.metadataWordRange.count),
          UInt64(MemoryLayout<UInt32>.stride),
          label: "QH5 metadata"
        )
      )
    }
    let actualWindowBytes = windows.map(\.decodedBytes).max() ?? 0
    let working = try Self.sum(
      [actualWindowBytes, persistent, auditAndMasks, maximumMetadata],
      label: "estimated package Metal storage"
    )

    datasetID = dataset.id
    sourceIdentitySHA256 = sourceIdentity
    sourceScanRows = dataset.scanRows
    sourceScanColumns = dataset.scanCols
    sourceDetectorRows = dataset.detectorRows
    sourceDetectorColumns = dataset.detectorCols
    logicalFrameCount = source.logicalFrameCount
    logicalDecodedBytes = source.logicalDecodedBytes
    compressedSourceBytes = UInt64(dataset.sourceBytes)
    sourceDtype = .uint16
    stagingDtype = .uint16
    scanBin = 1
    detectorBin = 1
    self.maximumDecodedWindowBytes = maximumDecodedWindowBytes
    maximumActualWindowBytes = actualWindowBytes
    persistentOutputBytes = persistent
    auditAndMaskBytes = auditAndMasks
    maximumMetadataBytes = maximumMetadata
    estimatedAllocatedMetalBytesExcludingMappedSource = working
    maximumMappedCompressedBytes = source.shards.map(\.index.metadata.sourceBytes).max() ?? 0
    let mappedSourceBytes = try source.shards.map {
      try Self.mappedBufferBytes($0.index.metadata.sourceBytes)
    }
    maximumMappedSourceBufferBytes = mappedSourceBytes.max() ?? 0
    maximumInFlightCommandBuffers = Self.defaultMaximumInFlightCommandBuffers
    maximumInFlightMappedSourceBytes = try Self.sum(
      Array(mappedSourceBytes.sorted(by: >).prefix(maximumInFlightCommandBuffers)),
      label: "in-flight mapped source buffers"
    )
    maximumIndividualMetalBufferBytes =
      [
        actualWindowBytes,
        mapBytes,
        detectorSumBytes,
        auditBytes,
        detectorPixels,
        maximumMetadata,
        maximumMappedSourceBufferBytes,
      ].max() ?? 0
    self.windows = windows
    self.detectorBands = detectorBands
    detectorBandsSHA256 = SHA256.hash(data: Data(detectorBands.membership))
      .map { String(format: "%02x", $0) }
      .joined()
    badPixelIndices = sortedBadPixels
    scanCalibration = dataset.sourceScanCalibration
    scanSamplingRowNanometer = dataset.scanPixelSizeRowNanometer
    scanSamplingColumnNanometer = dataset.scanPixelSizeColNanometer
    detectorSamplingRow = dataset.kPixelSizeRow
    detectorSamplingColumn = dataset.kPixelSizeCol
    detectorSamplingUnit = dataset.kPixelUnit
  }

  fileprivate static func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64
      && value.utf8.allSatisfy {
        (48...57).contains($0) || (97...102).contains($0)
      }
  }

  static func validateExactProductBounds(
    detectorRows: Int,
    detectorColumns: Int,
    logicalFrameCount: Int
  ) throws {
    guard detectorRows > 0, detectorColumns > 0, logicalFrameCount > 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Exact-product geometry must contain positive detector and scan dimensions."
      )
    }
    let rows = UInt64(detectorRows)
    let columns = UInt64(detectorColumns)
    let frames = UInt64(logicalFrameCount)
    let maximumCount = UInt64(UInt16.max)
    let detectorPixels = try product(rows, columns, label: "detector pixels")
    _ = try product(detectorPixels, maximumCount, label: "per-frame total")
    _ = try product(frames, maximumCount, label: "per-pixel detector sum")

    let rowIndexTwiceSum = try product(rows, rows - 1, label: "row-index sum")
    let rowCoordinates = try product(
      rowIndexTwiceSum / 2,
      columns,
      label: "detector-row coordinates"
    )
    _ = try product(
      rowCoordinates,
      maximumCount,
      label: "detector-row moment"
    )

    let columnIndexTwiceSum = try product(
      columns,
      columns - 1,
      label: "column-index sum"
    )
    let columnCoordinates = try product(
      columnIndexTwiceSum / 2,
      rows,
      label: "detector-column coordinates"
    )
    _ = try product(
      columnCoordinates,
      maximumCount,
      label: "detector-column moment"
    )
  }

  private static func mappedBufferBytes(_ sourceBytes: UInt64) throws -> UInt64 {
    let pageBytes = UInt64(getpagesize())
    guard sourceBytes > 0, pageBytes > 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Indexed source mappings require positive file and system-page sizes."
      )
    }
    let rounded = sourceBytes.addingReportingOverflow(pageBytes - 1)
    guard !rounded.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Indexed source mapping size overflows UInt64."
      )
    }
    return (rounded.partialValue / pageBytes) * pageBytes
  }

  private static func product(
    _ lhs: UInt64,
    _ rhs: UInt64,
    label: String
  ) throws -> UInt64 {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Native 4D-STEM \(label) overflows UInt64."
      )
    }
    return result.partialValue
  }

  private static func sum(
    _ values: [UInt64],
    label: String
  ) throws -> UInt64 {
    try values.reduce(UInt64(0)) { total, value in
      let result = total.addingReportingOverflow(value)
      guard !result.overflow else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Native 4D-STEM \(label) overflows UInt64."
        )
      }
      return result.partialValue
    }
  }
}

/// Exact integer screening products from a complete logical scan.
public struct Metal4DSTEMExactProducts: Equatable, Sendable {
  public let detectorSum: [UInt64]
  public let band1: [UInt64]
  public let band2: [UInt64]
  public let band4: [UInt64]
  public let total: [UInt64]
  public let detectorRowMoment: [UInt64]
  public let detectorColumnMoment: [UInt64]
}

/// Stable provenance for exact native indexed products.
public struct Metal4DSTEMIndexedLoadProvenance: Codable, Equatable, Sendable {
  public static let currentSchema = "quantem.gpu.metal-4dstem-indexed-products/v1"

  public let schema: String
  public let datasetID: String
  public let sourceIdentitySHA256: String
  public let valueRangeAuditSHA256: String
  public let sourceScanRows: Int
  public let sourceScanColumns: Int
  public let workingScanRows: Int
  public let workingScanColumns: Int
  public let sourceDetectorRows: Int
  public let sourceDetectorColumns: Int
  public let workingDetectorRows: Int
  public let workingDetectorColumns: Int
  public let sourceDtype: Metal4DSTEMIntegerDType
  public let stagingDtype: Metal4DSTEMIntegerDType
  public let workingDtype: Metal4DSTEMIntegerDType
  public let productDtype: String
  public let scanBin: Int
  public let detectorBin: Int
  public let scanRowStart: Int
  public let scanRowStop: Int
  public let scanColumnStart: Int
  public let scanColumnStop: Int
  public let reduction: Metal4DSTEMReductionSemantics
  public let badPixelIndices: [Int]
  public let maximumSourceCount: UInt32
  public let pixelsAbove255: UInt64
  public let logicalDecodedBytes: UInt64
  public let maximumDecodedWindowBytes: UInt64
  public let maximumActualWindowBytes: UInt64
  public let persistentOutputBytes: UInt64
  public let windowCount: Int
  public let sliceCount: Int
  public let sourceState: String
  public let sourceLayout: String
  public let detectorSumLayout: String
  public let scanMapLayout: String
  public let detectorBandEncoding: String
  public let detectorBandsSHA256: String
  public let meanDetectorPatternDivisor: UInt64
  public let scanCalibration: Native4DSTEMScanCalibration?
  public let scanSamplingRowNanometer: Double?
  public let scanSamplingColumnNanometer: Double?
  public let detectorSamplingRow: Double?
  public let detectorSamplingColumn: Double?
  public let detectorSamplingUnit: String?
}

/// Timings from one synchronized package execution.
///
/// Source-page state is deliberately unspecified. A benchmark harness must
/// label first encounter, page-warm, and prepared-result reopen separately.
public struct Metal4DSTEMIndexedLoadMetrics: Codable, Equatable, Sendable {
  public let wallSeconds: Double
  public let sourceMappingSeconds: Double
  public let gpuSeconds: Double
  public let windowCount: Int
  public let sliceCount: Int
  public let commandBufferCount: Int
  public let peakInFlightCommandBuffers: Int
  public let peakInFlightMappedSourceBytes: UInt64
  public let mappedCompressedSourceBytes: UInt64
  public let maximumMappedCompressedSourceBytes: UInt64
  public let maximumMappedSourceBufferBytes: UInt64
  public let maximumDecodedSliceBytes: UInt64
  public let cacheState: String
}

public struct Metal4DSTEMIndexedLoadResult: Equatable, Sendable {
  public let products: Metal4DSTEMExactProducts
  public let sourceAudit: Metal4DSTEMExactSourceAudit
  public let provenance: Metal4DSTEMIndexedLoadProvenance
  public let metrics: Metal4DSTEMIndexedLoadMetrics
}

public struct Metal4DSTEMDecodedFrame: Equatable, Sendable {
  public let scanRow: Int
  public let scanColumn: Int
  public let detectorRows: Int
  public let detectorColumns: Int
  public let dtype: Metal4DSTEMIntegerDType
  public let values: [UInt16]
  public let maximumSourceCount: UInt32
  public let pixelsAbove255: UInt64
  public let sourceIdentitySHA256: String
  public let badPixelIndices: [Int]
}

/// Synchronous, UI-free exact loader for prepared native QH5 indexes.
///
/// The caller owns device admission, scheduling, caching, cancellation policy,
/// and presentation. This loader owns bounded decode, exact reductions, and
/// source/shape/dtype provenance only.
public final class Metal4DSTEMIndexedLoader {
  // All commands use one ordered queue, so a single bounded scratch buffer is
  // reused without overlap while CPU mapping and command encoding run ahead.
  private static let maximumInFlightCommandBuffers =
    Metal4DSTEMIndexedLoadPlan.defaultMaximumInFlightCommandBuffers

  private let device: MTLDevice
  private let queue: MTLCommandQueue
  private let decodePipeline: MTLComputePipelineState
  private let productPipeline: MTLComputePipelineState
  private let detectorSumPipeline: MTLComputePipelineState
  private let exactBinner: Metal4DSTEMExactBinner

  public init(device: MTLDevice) throws {
    guard let queue = device.makeCommandQueue() else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "the selected device could not create a command queue"
      )
    }
    do {
      let hdf5 = try Metal4DSTEMKernels.makeHDF5Library(device: device)
      let detector = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
      guard let decode = hdf5.makeFunction(name: Metal4DSTEMKernels.decodeU16Function),
        let products = detector.makeFunction(
          name: Metal4DSTEMKernels.detectorProductsU16ExactU64Function
        ),
        let detectorSum = detector.makeFunction(
          name: Metal4DSTEMKernels.detectorAccumulateU16U64Function
        )
      else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "the packaged exact decode or reduction function is missing"
        )
      }
      decodePipeline = try device.makeComputePipelineState(function: decode)
      productPipeline = try device.makeComputePipelineState(function: products)
      detectorSumPipeline = try device.makeComputePipelineState(function: detectorSum)
      exactBinner = try Metal4DSTEMExactBinner(
        device: device,
        detectorLibrary: detector
      )
    } catch let error as Metal4DSTEMStreamingIOError {
      throw error
    } catch {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(error.localizedDescription)
    }
    self.device = device
    self.queue = queue
  }

  /// Decode every native frame through bounded windows and return exact sums.
  public func loadExactProducts(
    source: Native4DSTEMIndexedSource,
    plan: Metal4DSTEMIndexedLoadPlan,
    shouldCancel: () -> Bool = { false }
  ) throws -> Metal4DSTEMIndexedLoadResult {
    try loadExactProductsInternal(
      source: source,
      plan: plan,
      binned: nil,
      shouldCancel: shouldCancel
    ).result
  }

  /// Decode every native frame once, return exact products, and retain exact shards.
  ///
  /// The supplied plan is policy-free and source-identity-bound. A decoded
  /// audit mismatch fails before any working-volume shard is returned.
  public func loadExactBinnedShards(
    source: Native4DSTEMIndexedSource,
    plan: Metal4DSTEMIndexedBinnedLoadPlan,
    shouldCancel: () -> Bool = { false }
  ) throws -> Metal4DSTEMIndexedBinnedLoadResult {
    let totalStarted = CFAbsoluteTimeGetCurrent()
    let expectedPlan = try Metal4DSTEMIndexedBinnedLoadPlan(
      source: source,
      maximumDecodedWindowBytes: plan.productPlan.maximumDecodedWindowBytes,
      detectorBands: plan.productPlan.detectorBands,
      detectorBin: plan.binningProvenance.detectorBin,
      sourceAudit: plan.sourceAudit,
      maximumShardBytes: plan.shardPlan.maximumShardBytes
    )
    guard expectedPlan == plan else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The exact binned load plan does not match its indexed source, audit, "
          + "detector bands, or shard geometry."
      )
    }
    guard plan.maximumIndividualMetalBufferBytes <= UInt64(device.maxBufferLength) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The largest exact binned-load buffer requires "
          + "\(plan.maximumIndividualMetalBufferBytes) bytes, exceeding this Metal "
          + "device's maximum single-buffer length of \(device.maxBufferLength) bytes."
      )
    }
    let allocationStarted = CFAbsoluteTimeGetCurrent()
    let destinations = try plan.shardPlan.shards.map { shard in
      try makeBuffer(
        bytes: shard.payloadBytes,
        options: .storageModeShared,
        label: "exact working-volume shard \(shard.index)"
      )
    }
    let allocationSeconds = CFAbsoluteTimeGetCurrent() - allocationStarted
    let context = BinnedOutputContext(
      resident: plan,
      destinations: destinations
    )
    let loaded = try loadExactProductsInternal(
      source: source,
      plan: plan.productPlan,
      binned: context,
      shouldCancel: shouldCancel
    )
    return Metal4DSTEMIndexedBinnedLoadResult(
      workingVolumeShards: destinations,
      shardPlan: plan.shardPlan,
      products: loaded.result.products,
      sourceAudit: loaded.result.sourceAudit,
      nativeProductProvenance: loaded.result.provenance,
      binningProvenance: plan.binningProvenance,
      samplingPropagation: plan.samplingPropagation,
      metrics: Metal4DSTEMIndexedBinnedLoadMetrics(
        indexedLoad: loaded.result.metrics,
        destinationAllocationSeconds: allocationSeconds,
        totalWallSeconds: CFAbsoluteTimeGetCurrent() - totalStarted,
        binningDispatchCount: loaded.binningDispatchCount,
        workingPayloadBytes: plan.workingPayloadBytes,
        shardCount: plan.shardPlan.shards.count,
        maximumShardBytes: plan.shardPlan.maximumActualShardBytes,
        destinationStorageMode: plan.destinationStorageMode
      )
    )
  }

  /// Decode once and transactionally write an exact file-backed working volume.
  ///
  /// Only one output shard is allocated in Metal at a time. Each shard is
  /// complete before it is appended to a unique temporary payload and released.
  /// The metadata publication marker remains absent until the recomputed source
  /// audit, exact products, full byte coverage, fsync, and SHA-256 seal all
  /// succeed. The caller owns device admission and cache lifecycle.
  public func loadExactBinnedCache(
    source: Native4DSTEMIndexedSource,
    plan: Metal4DSTEMIndexedBinnedLoadPlan,
    payloadURL: URL,
    metadataURL: URL,
    shouldCancel: () -> Bool = { false }
  ) throws -> Metal4DSTEMIndexedBinnedCacheResult {
    let totalStarted = CFAbsoluteTimeGetCurrent()
    let expectedPlan = try Metal4DSTEMIndexedBinnedLoadPlan(
      source: source,
      maximumDecodedWindowBytes: plan.productPlan.maximumDecodedWindowBytes,
      detectorBands: plan.productPlan.detectorBands,
      detectorBin: plan.binningProvenance.detectorBin,
      sourceAudit: plan.sourceAudit,
      maximumShardBytes: plan.shardPlan.maximumShardBytes
    )
    guard expectedPlan == plan else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The exact binned cache plan does not match its indexed source, audit, "
          + "detector bands, or shard geometry."
      )
    }
    guard plan.maximumIndividualMetalBufferBytes <= UInt64(device.maxBufferLength) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The largest exact binned-cache buffer requires "
          + "\(plan.maximumIndividualMetalBufferBytes) bytes, exceeding this Metal "
          + "device's maximum single-buffer length of \(device.maxBufferLength) bytes."
      )
    }
    try BinnedOutputContext.validateStreamable(plan)
    let unsealedMetadata = try residentCacheMetadata(source: source, plan: plan)
    let writer = try Metal4DSTEMResidentCacheStreamWriter(
      payloadURL: payloadURL,
      metadataURL: metadataURL,
      metadata: unsealedMetadata
    )
    defer { writer.cancel() }
    let context = BinnedOutputContext(
      streaming: plan,
      consume: { _, buffer in
        try writer.append(pointer: buffer.contents(), length: buffer.length)
      }
    )
    let loaded = try loadExactProductsInternal(
      source: source,
      plan: plan.productPlan,
      binned: context,
      shouldCancel: shouldCancel
    )
    guard context.completedShardCount == plan.shardPlan.shards.count else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The exact cache build completed \(context.completedShardCount) output "
          + "shards; expected \(plan.shardPlan.shards.count)."
      )
    }
    let finalizeStarted = CFAbsoluteTimeGetCurrent()
    let completeMetadata = try writer.finish()
    let finalizeSeconds = CFAbsoluteTimeGetCurrent() - finalizeStarted
    return Metal4DSTEMIndexedBinnedCacheResult(
      payloadURL: payloadURL,
      metadataURL: metadataURL,
      metadata: completeMetadata,
      products: loaded.result.products,
      sourceAudit: loaded.result.sourceAudit,
      nativeProductProvenance: loaded.result.provenance,
      binningProvenance: plan.binningProvenance,
      samplingPropagation: plan.samplingPropagation,
      metrics: Metal4DSTEMIndexedBinnedCacheMetrics(
        indexedLoad: loaded.result.metrics,
        destinationAllocationSeconds: context.destinationAllocationSeconds,
        payloadWriteSeconds: context.payloadWriteSeconds,
        payloadFinalizeSeconds: finalizeSeconds,
        totalWallSeconds: CFAbsoluteTimeGetCurrent() - totalStarted,
        binningDispatchCount: loaded.binningDispatchCount,
        workingPayloadBytes: plan.workingPayloadBytes,
        shardCount: context.completedShardCount,
        maximumShardBytes: plan.shardPlan.maximumActualShardBytes,
        peakWorkingMetalBytes: context.peakWorkingMetalBytes,
        destinationStorage: "transactional_file_backed_exact_payload"
      )
    )
  }

  private func loadExactProductsInternal(
    source: Native4DSTEMIndexedSource,
    plan: Metal4DSTEMIndexedLoadPlan,
    binned: BinnedOutputContext?,
    shouldCancel: () -> Bool
  ) throws -> (
    result: Metal4DSTEMIndexedLoadResult,
    binningDispatchCount: Int
  ) {
    let expectedPlan = try Metal4DSTEMIndexedLoadPlan(
      source: source,
      maximumDecodedWindowBytes: plan.maximumDecodedWindowBytes,
      detectorBands: plan.detectorBands
    )
    guard expectedPlan == plan else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The indexed load plan does not match the supplied source or detector bands."
      )
    }
    try validateDevice(plan: plan)
    let started = CFAbsoluteTimeGetCurrent()
    let frameCount = plan.logicalFrameCount
    let detectorPixels = plan.sourceDetectorRows * plan.sourceDetectorColumns
    let mapBytes = try byteCount(count: frameCount, stride: 8, label: "one scan map")
    let detectorBytes = try byteCount(
      count: detectorPixels,
      stride: 8,
      label: "detector sum"
    )
    let auditBytes = try byteCount(
      count: frameCount * 2,
      stride: MemoryLayout<UInt32>.stride,
      label: "count audit"
    )
    let scratch = try makeBuffer(
      bytes: plan.maximumActualWindowBytes,
      options: .storageModePrivate,
      label: "decoded native window"
    )
    let band1 = try makeBuffer(bytes: UInt64(mapBytes), label: "band-1 map")
    let band2 = try makeBuffer(bytes: UInt64(mapBytes), label: "band-2 map")
    let band4 = try makeBuffer(bytes: UInt64(mapBytes), label: "band-4 map")
    let total = try makeBuffer(bytes: UInt64(mapBytes), label: "total map")
    let rowMoment = try makeBuffer(bytes: UInt64(mapBytes), label: "detector-row moment")
    let columnMoment = try makeBuffer(
      bytes: UInt64(mapBytes), label: "detector-column moment"
    )
    let detectorSum = try makeBuffer(
      bytes: UInt64(detectorBytes), label: "detector sum"
    )
    let countAudit = try makeBuffer(
      bytes: UInt64(auditBytes), label: "source count audit"
    )
    let badPixelMask = try makeByteMask(
      count: detectorPixels,
      selected: plan.badPixelIndices,
      label: "bad-pixel mask"
    )
    let detectorBands = try makeBuffer(
      values: plan.detectorBands.membership,
      label: "detector-band membership"
    )
    for buffer in [
      band1, band2, band4, total, rowMoment, columnMoment, detectorSum, countAudit,
    ] {
      memset(buffer.contents(), 0, buffer.length)
    }

    var activeShardIndex: Int?
    var activeSource: MappedMetalSource?
    var mapSeconds = 0.0
    var gpuSeconds = 0.0
    var sliceCount = 0
    var binningDispatchCount = 0
    var mappedCompressedSourceBytes: UInt64 = 0
    var mappedShards = Set<Int>()
    var maximumMappedCompressedSourceBytes: UInt64 = 0
    var maximumMappedSourceBufferBytes: UInt64 = 0
    var maximumDecodedSliceBytes: UInt64 = 0
    var pendingCommands: [PendingIndexedCommand] = []
    var peakInFlightCommandBuffers = 0
    var peakInFlightMappedSourceBytes: UInt64 = 0

    for window in plan.windows {
      if shouldCancel() {
        gpuSeconds += try drainAll(&pendingCommands)
        throw Metal4DSTEMStreamingIOError.cancelled
      }
      if let binned,
        try binned.requiresDestinationTransition(for: window.globalFrameRange)
      {
        // A streamed output shard cannot be written or released until every
        // command that targets it has completed. Resident output buffers are
        // all retained and therefore never take this transition path.
        gpuSeconds += try drainAll(&pendingCommands)
        try binned.finishActiveDestination()
        try binned.prepareDestination(
          for: window.globalFrameRange,
          makeBuffer: { shard in
            try self.makeBuffer(
              bytes: shard.payloadBytes,
              options: .storageModeShared,
              label: "transactional exact working-volume shard \(shard.index)"
            )
          }
        )
      }
      for slice in window.slices {
        if shouldCancel() {
          gpuSeconds += try drainAll(&pendingCommands)
          throw Metal4DSTEMStreamingIOError.cancelled
        }
        let shard = source.shards[slice.shardIndex]
        if activeShardIndex != slice.shardIndex {
          let mapStarted = CFAbsoluteTimeGetCurrent()
          let currentIndex = try NativeQH5Index.open(
            sourceURL: shard.sourceURL,
            indexURL: shard.indexURL
          )
          guard currentIndex.metadata == shard.index.metadata,
            currentIndex.metadataWords == shard.index.metadataWords
          else {
            throw Metal4DSTEMStreamingIOError.invalidRequest(
              "Prepared QH5 index changed after the indexed source was opened. Reopen the source."
            )
          }
          activeSource = try MappedMetalSource(
            url: shard.sourceURL,
            expectedBytes: shard.index.metadata.sourceBytes,
            device: device
          )
          activeShardIndex = slice.shardIndex
          mapSeconds += CFAbsoluteTimeGetCurrent() - mapStarted
          if mappedShards.insert(slice.shardIndex).inserted {
            let total = mappedCompressedSourceBytes.addingReportingOverflow(
              shard.index.metadata.sourceBytes
            )
            guard !total.overflow else {
              throw Metal4DSTEMStreamingIOError.invalidRequest(
                "Mapped compressed-source byte accounting overflows UInt64."
              )
            }
            mappedCompressedSourceBytes = total.partialValue
          }
          maximumMappedCompressedSourceBytes = max(
            maximumMappedCompressedSourceBytes,
            shard.index.metadata.sourceBytes
          )
          maximumMappedSourceBufferBytes = max(
            maximumMappedSourceBufferBytes,
            UInt64(activeSource?.buffer.length ?? 0)
          )
        }
        guard let mapped = activeSource else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            "the indexed source mapping was released before decode"
          )
        }
        let decodedSliceBytes =
          UInt64(slice.globalFrameRange.count)
          * source.decodedBytesPerFrame
        maximumDecodedSliceBytes = max(maximumDecodedSliceBytes, decodedSliceBytes)
        let pending = try enqueue(
          source: source,
          slice: slice,
          mapped: mapped,
          scratch: scratch,
          badPixelMask: badPixelMask,
          detectorBands: detectorBands,
          countAudit: countAudit,
          band1: band1,
          band2: band2,
          band4: band4,
          total: total,
          rowMoment: rowMoment,
          columnMoment: columnMoment,
          detectorSum: detectorSum,
          binned: binned
        )
        binningDispatchCount += pending.binningDispatchCount
        pendingCommands.append(pending)
        peakInFlightCommandBuffers = max(
          peakInFlightCommandBuffers,
          pendingCommands.count
        )
        peakInFlightMappedSourceBytes = max(
          peakInFlightMappedSourceBytes,
          try inFlightMappedSourceBytes(pendingCommands)
        )
        if pendingCommands.count >= Self.maximumInFlightCommandBuffers {
          gpuSeconds += try drainOldestOrFinishAll(&pendingCommands)
        }
        if shouldCancel() {
          gpuSeconds += try drainAll(&pendingCommands)
          throw Metal4DSTEMStreamingIOError.cancelled
        }
        sliceCount += 1
      }
    }
    gpuSeconds += try drainAll(&pendingCommands)
    try binned?.finishActiveDestination()
    activeSource = nil

    let auditValues = values(from: countAudit, count: frameCount * 2, as: UInt32.self)
    var maximum: UInt32 = 0
    var above255: UInt64 = 0
    for frame in 0..<frameCount {
      maximum = max(maximum, auditValues[2 * frame])
      let sum = above255.addingReportingOverflow(UInt64(auditValues[2 * frame + 1]))
      guard !sum.overflow else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "The decoded above-255 audit overflows UInt64."
        )
      }
      above255 = sum.partialValue
    }
    let sourceAudit = try Metal4DSTEMExactSourceAudit(
      sourceIdentitySHA256: plan.sourceIdentitySHA256,
      sourceDtype: .uint16,
      badPixelIndices: plan.badPixelIndices,
      maximumSourceCount: maximum,
      pixelsAbove255: above255
    )
    if let binned, sourceAudit != binned.plan.sourceAudit {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The decoded value-range audit does not match the source-identity-bound "
          + "audit that authorized exact \(binned.plan.binningProvenance.outputDtype.rawValue) "
          + "working storage. Discard the shards and rebuild the audit."
      )
    }
    let products = Metal4DSTEMExactProducts(
      detectorSum: values(from: detectorSum, count: detectorPixels, as: UInt64.self),
      band1: values(from: band1, count: frameCount, as: UInt64.self),
      band2: values(from: band2, count: frameCount, as: UInt64.self),
      band4: values(from: band4, count: frameCount, as: UInt64.self),
      total: values(from: total, count: frameCount, as: UInt64.self),
      detectorRowMoment: values(from: rowMoment, count: frameCount, as: UInt64.self),
      detectorColumnMoment: values(
        from: columnMoment, count: frameCount, as: UInt64.self
      )
    )
    let provenance = Metal4DSTEMIndexedLoadProvenance(
      schema: Metal4DSTEMIndexedLoadProvenance.currentSchema,
      datasetID: plan.datasetID,
      sourceIdentitySHA256: plan.sourceIdentitySHA256,
      valueRangeAuditSHA256: sourceAudit.auditSHA256,
      sourceScanRows: plan.sourceScanRows,
      sourceScanColumns: plan.sourceScanColumns,
      workingScanRows: plan.sourceScanRows,
      workingScanColumns: plan.sourceScanColumns,
      sourceDetectorRows: plan.sourceDetectorRows,
      sourceDetectorColumns: plan.sourceDetectorColumns,
      workingDetectorRows: plan.sourceDetectorRows,
      workingDetectorColumns: plan.sourceDetectorColumns,
      sourceDtype: .uint16,
      stagingDtype: .uint16,
      workingDtype: .uint16,
      productDtype: "uint64",
      scanBin: 1,
      detectorBin: 1,
      scanRowStart: 0,
      scanRowStop: plan.sourceScanRows,
      scanColumnStart: 0,
      scanColumnStop: plan.sourceScanColumns,
      reduction: .exactIntegerSum,
      badPixelIndices: plan.badPixelIndices,
      maximumSourceCount: maximum,
      pixelsAbove255: above255,
      logicalDecodedBytes: plan.logicalDecodedBytes,
      maximumDecodedWindowBytes: plan.maximumDecodedWindowBytes,
      maximumActualWindowBytes: plan.maximumActualWindowBytes,
      persistentOutputBytes: plan.persistentOutputBytes,
      windowCount: plan.windows.count,
      sliceCount: sliceCount,
      sourceState: "prepared_qh5_index",
      sourceLayout: "frame_major_scan_row_scan_column_detector_row_detector_column",
      detectorSumLayout: "detector_row_detector_column",
      scanMapLayout: "scan_row_scan_column",
      detectorBandEncoding: "bit0_band1_bit1_band2_bit2_band4",
      detectorBandsSHA256: plan.detectorBandsSHA256,
      meanDetectorPatternDivisor: UInt64(plan.logicalFrameCount),
      scanCalibration: plan.scanCalibration,
      scanSamplingRowNanometer: plan.scanSamplingRowNanometer,
      scanSamplingColumnNanometer: plan.scanSamplingColumnNanometer,
      detectorSamplingRow: plan.detectorSamplingRow,
      detectorSamplingColumn: plan.detectorSamplingColumn,
      detectorSamplingUnit: plan.detectorSamplingUnit
    )
    let metrics = Metal4DSTEMIndexedLoadMetrics(
      wallSeconds: CFAbsoluteTimeGetCurrent() - started,
      sourceMappingSeconds: mapSeconds,
      gpuSeconds: gpuSeconds,
      windowCount: plan.windows.count,
      sliceCount: sliceCount,
      commandBufferCount: sliceCount,
      peakInFlightCommandBuffers: peakInFlightCommandBuffers,
      peakInFlightMappedSourceBytes: peakInFlightMappedSourceBytes,
      mappedCompressedSourceBytes: mappedCompressedSourceBytes,
      maximumMappedCompressedSourceBytes: maximumMappedCompressedSourceBytes,
      maximumMappedSourceBufferBytes: maximumMappedSourceBufferBytes,
      maximumDecodedSliceBytes: maximumDecodedSliceBytes,
      cacheState: binned?.cacheState
        ?? "prepared_qh5_index_source_pages_unspecified"
    )
    return (
      result: Metal4DSTEMIndexedLoadResult(
        products: products,
        sourceAudit: sourceAudit,
        provenance: provenance,
        metrics: metrics
      ),
      binningDispatchCount: binningDispatchCount
    )
  }

  /// Decode one native diffraction pattern at a public `(row, column)` coordinate.
  public func diffractionPattern(
    source: Native4DSTEMIndexedSource,
    scanRow: Int,
    scanColumn: Int
  ) throws -> Metal4DSTEMDecodedFrame {
    guard source.dataset.sourceDtype == Metal4DSTEMIntegerDType.uint16.rawValue,
      let sourceIdentity = source.dataset.sourceIdentitySHA256,
      Metal4DSTEMIndexedLoadPlan.isSHA256(sourceIdentity)
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "On-demand native frames require a source-identity-bound uint16 indexed source."
      )
    }
    let window = try source.frameWindow(scanRow: scanRow, scanColumn: scanColumn)
    guard window.slices.count == 1, let slice = window.slices.first else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "One logical diffraction frame must resolve to exactly one prepared QH5 slice."
      )
    }
    let shard = source.shards[slice.shardIndex]
    let currentIndex = try NativeQH5Index.open(
      sourceURL: shard.sourceURL,
      indexURL: shard.indexURL
    )
    guard currentIndex.metadata == shard.index.metadata,
      currentIndex.metadataWords == shard.index.metadataWords
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Prepared QH5 index changed after the indexed source was opened. Reopen the source."
      )
    }
    let detectorPixels = source.dataset.detectorRows * source.dataset.detectorCols
    let output = try makeBuffer(
      bytes: source.decodedBytesPerFrame,
      label: "one native diffraction pattern"
    )
    let audit = try makeBuffer(
      bytes: UInt64(2 * MemoryLayout<UInt32>.stride),
      label: "one-frame count audit"
    )
    memset(audit.contents(), 0, audit.length)
    let badPixelMask = try makeByteMask(
      count: detectorPixels,
      selected: source.dataset.badPixelIndices,
      label: "bad-pixel mask"
    )
    let mapped = try MappedMetalSource(
      url: shard.sourceURL,
      expectedBytes: shard.index.metadata.sourceBytes,
      device: device
    )
    let metadata = try metadataBuffer(source: source, slice: slice)
    var rangeStart = slice.chunkCompressedByteRange.lowerBound
    var blocksPerFrame = try exactUInt32(
      shard.index.metadata.nBlocksPerFrame,
      label: "blocks per frame"
    )
    var frameElements = try exactUInt32(detectorPixels, label: "detector pixels")
    var metadataFrameOffset: UInt32 = 0
    var auditFrameOffset: UInt32 = 0
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "the device could not start an on-demand frame decode"
      )
    }
    encoder.setComputePipelineState(decodePipeline)
    encoder.setBuffer(mapped.buffer, offset: 0, index: 0)
    encoder.setBuffer(metadata, offset: 0, index: 1)
    encoder.setBytes(&rangeStart, length: 8, index: 2)
    encoder.setBytes(&blocksPerFrame, length: 4, index: 3)
    encoder.setBytes(&frameElements, length: 4, index: 4)
    encoder.setBuffer(output, offset: 0, index: 5)
    encoder.setBytes(&metadataFrameOffset, length: 4, index: 6)
    encoder.setBuffer(badPixelMask, offset: 0, index: 7)
    encoder.setBuffer(audit, offset: 0, index: 8)
    encoder.setBytes(&auditFrameOffset, length: 4, index: 9)
    encoder.dispatchThreadgroups(
      MTLSize(width: 1, height: 1, depth: Int(blocksPerFrame)),
      threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
    )
    encoder.endEncoding()
    command.commit()
    try wait(command)
    let auditValues = values(from: audit, count: 2, as: UInt32.self)
    return Metal4DSTEMDecodedFrame(
      scanRow: scanRow,
      scanColumn: scanColumn,
      detectorRows: source.dataset.detectorRows,
      detectorColumns: source.dataset.detectorCols,
      dtype: .uint16,
      values: values(from: output, count: detectorPixels, as: UInt16.self),
      maximumSourceCount: auditValues[0],
      pixelsAbove255: UInt64(auditValues[1]),
      sourceIdentitySHA256: sourceIdentity,
      badPixelIndices: source.dataset.badPixelIndices.sorted()
    )
  }

  private func enqueue(
    source: Native4DSTEMIndexedSource,
    slice: Native4DSTEMIndexedSlice,
    mapped: MappedMetalSource,
    scratch: MTLBuffer,
    badPixelMask: MTLBuffer,
    detectorBands: MTLBuffer,
    countAudit: MTLBuffer,
    band1: MTLBuffer,
    band2: MTLBuffer,
    band4: MTLBuffer,
    total: MTLBuffer,
    rowMoment: MTLBuffer,
    columnMoment: MTLBuffer,
    detectorSum: MTLBuffer,
    binned: BinnedOutputContext?
  ) throws -> PendingIndexedCommand {
    let shard = source.shards[slice.shardIndex]
    let frameCount = try exactUInt32(slice.globalFrameRange.count, label: "slice frames")
    let detectorPixels = source.dataset.detectorRows * source.dataset.detectorCols
    var detectorPixelCount = try exactUInt32(detectorPixels, label: "detector pixels")
    var detectorColumns = try exactUInt32(
      source.dataset.detectorCols,
      label: "detector columns"
    )
    var globalFrameOffset = try exactUInt32(
      slice.globalFrameRange.lowerBound,
      label: "global frame offset"
    )
    var rangeStart = slice.chunkCompressedByteRange.lowerBound
    var blocksPerFrame = try exactUInt32(
      shard.index.metadata.nBlocksPerFrame,
      label: "blocks per frame"
    )
    var metadataFrameOffset: UInt32 = 0
    let metadata = try metadataBuffer(source: source, slice: slice)
    var parameters = DetectorParameters(
      frameCount: frameCount,
      detectorPixels: detectorPixelCount,
      globalFrameOffset: globalFrameOffset
    )
    guard let command = queue.makeCommandBuffer(),
      let encoder = command.makeComputeCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "the device could not start an indexed decode command"
      )
    }
    encoder.setComputePipelineState(decodePipeline)
    encoder.setBuffer(mapped.buffer, offset: 0, index: 0)
    encoder.setBuffer(metadata, offset: 0, index: 1)
    encoder.setBytes(&rangeStart, length: 8, index: 2)
    encoder.setBytes(&blocksPerFrame, length: 4, index: 3)
    encoder.setBytes(&detectorPixelCount, length: 4, index: 4)
    encoder.setBuffer(scratch, offset: 0, index: 5)
    encoder.setBytes(&metadataFrameOffset, length: 4, index: 6)
    encoder.setBuffer(badPixelMask, offset: 0, index: 7)
    encoder.setBuffer(countAudit, offset: 0, index: 8)
    encoder.setBytes(&globalFrameOffset, length: 4, index: 9)
    encoder.dispatchThreadgroups(
      MTLSize(
        width: Int(frameCount),
        height: 1,
        depth: Int(blocksPerFrame)
      ),
      threadsPerThreadgroup: MTLSize(width: 32, height: 4, depth: 1)
    )
    encoder.memoryBarrier(scope: .buffers)

    encoder.setComputePipelineState(productPipeline)
    encoder.setBuffer(scratch, offset: 0, index: 0)
    encoder.setBuffer(band1, offset: 0, index: 1)
    encoder.setBuffer(band2, offset: 0, index: 2)
    encoder.setBuffer(band4, offset: 0, index: 3)
    withUnsafeBytes(of: &parameters) {
      encoder.setBytes($0.baseAddress!, length: $0.count, index: 4)
    }
    encoder.setBuffer(detectorBands, offset: 0, index: 5)
    encoder.setBuffer(total, offset: 0, index: 6)
    encoder.setBuffer(rowMoment, offset: 0, index: 7)
    encoder.setBuffer(columnMoment, offset: 0, index: 8)
    encoder.setBytes(&detectorColumns, length: 4, index: 9)
    encoder.dispatchThreadgroups(
      MTLSize(width: Int(frameCount), height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
    )
    encoder.memoryBarrier(scope: .buffers)

    encoder.setComputePipelineState(detectorSumPipeline)
    encoder.setBuffer(scratch, offset: 0, index: 0)
    encoder.setBuffer(detectorSum, offset: 0, index: 1)
    encoder.setBytes(&detectorPixelCount, length: 4, index: 2)
    var mutableFrameCount = frameCount
    encoder.setBytes(&mutableFrameCount, length: 4, index: 3)
    let threads = min(256, detectorSumPipeline.maxTotalThreadsPerThreadgroup)
    encoder.dispatchThreads(
      MTLSize(width: detectorPixels, height: 1, depth: 1),
      threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
    )
    encoder.endEncoding()
    let binningDispatchCount =
      try binned.map {
        try encodeBinnedSlice(
          context: $0,
          slice: slice,
          scratch: scratch,
          commandBuffer: command
        )
      } ?? 0
    command.commit()
    return PendingIndexedCommand(
      command: command,
      mappedSource: mapped,
      metadata: metadata,
      binningDispatchCount: binningDispatchCount
    )
  }

  private func encodeBinnedSlice(
    context: BinnedOutputContext,
    slice: Native4DSTEMIndexedSlice,
    scratch: MTLBuffer,
    commandBuffer: MTLCommandBuffer
  ) throws -> Int {
    let plan = context.plan
    let sourceFrameElements = plan.productPlan.sourceDetectorRows
      .multipliedReportingOverflow(by: plan.productPlan.sourceDetectorColumns)
    guard !sourceFrameElements.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "One indexed source-frame element count overflows Int."
      )
    }
    let sourceFrameBytes = sourceFrameElements.partialValue
      .multipliedReportingOverflow(by: MemoryLayout<UInt16>.stride)
    guard !sourceFrameBytes.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "One indexed source-frame byte count overflows Int."
      )
    }
    let frameBytes = sourceFrameBytes.partialValue
    let binningPlan = try plan.binningPlan
    var globalStart = slice.globalFrameRange.lowerBound
    var dispatchCount = 0
    while globalStart < slice.globalFrameRange.upperBound {
      guard
        let shard = plan.shardPlan.shards.first(where: { candidate in
          let stop =
            candidate.outputScanPositionStart
            + candidate.outputScanPositionCount
          return globalStart >= candidate.outputScanPositionStart && globalStart < stop
        })
      else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Exact destination shards do not cover indexed frame \(globalStart)."
        )
      }
      let shardStop = shard.outputScanPositionStart + shard.outputScanPositionCount
      let globalStop = min(slice.globalFrameRange.upperBound, shardStop)
      let localFrameOffset = globalStart - slice.globalFrameRange.lowerBound
      let sourceOffset = localFrameOffset.multipliedReportingOverflow(by: frameBytes)
      guard !sourceOffset.overflow else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Indexed binning source offset overflows Int."
        )
      }
      _ = try exactBinner.encodeContiguousUInt16Frames(
        commandBuffer: commandBuffer,
        stagedSource: scratch,
        stagedSourceOffset: sourceOffset.partialValue,
        destination: try context.destination(for: shard.index),
        destinationView: .scanRowShard(
          plan: plan.shardPlan,
          index: shard.index
        ),
        plan: binningPlan,
        sourceFrameCount: globalStop - globalStart,
        globalScanPositionOffset: globalStart,
        sourceAudit: plan.sourceAudit
      )
      dispatchCount += 1
      globalStart = globalStop
    }
    return dispatchCount
  }

  private func drainFirst(
    _ pendingCommands: inout [PendingIndexedCommand]
  ) throws -> Double {
    guard !pendingCommands.isEmpty else { return 0 }
    let pending = pendingCommands.removeFirst()
    try wait(pending.command)
    return max(0, pending.command.gpuEndTime - pending.command.gpuStartTime)
  }

  private func drainAll(
    _ pendingCommands: inout [PendingIndexedCommand]
  ) throws -> Double {
    var gpuSeconds = 0.0
    var firstError: Error?
    while !pendingCommands.isEmpty {
      do {
        gpuSeconds += try drainFirst(&pendingCommands)
      } catch {
        if firstError == nil { firstError = error }
      }
    }
    if let firstError { throw firstError }
    return gpuSeconds
  }

  private func drainOldestOrFinishAll(
    _ pendingCommands: inout [PendingIndexedCommand]
  ) throws -> Double {
    do {
      return try drainFirst(&pendingCommands)
    } catch {
      // Preserve every no-copy mapping until all already-submitted work has
      // completed, even when the oldest command reports an error.
      _ = try? drainAll(&pendingCommands)
      throw error
    }
  }

  private func inFlightMappedSourceBytes(
    _ pendingCommands: [PendingIndexedCommand]
  ) throws -> UInt64 {
    var seen = Set<ObjectIdentifier>()
    var bytes: UInt64 = 0
    for pending in pendingCommands {
      guard seen.insert(ObjectIdentifier(pending.mappedSource)).inserted else {
        continue
      }
      let updated = bytes.addingReportingOverflow(UInt64(pending.mappedSource.buffer.length))
      guard !updated.overflow else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "In-flight mapped-source byte accounting overflows UInt64."
        )
      }
      bytes = updated.partialValue
    }
    return bytes
  }

  private func metadataBuffer(
    source: Native4DSTEMIndexedSource,
    slice: Native4DSTEMIndexedSlice
  ) throws -> MTLBuffer {
    let words = Array(
      source.shards[slice.shardIndex].index.metadataWords[slice.metadataWordRange]
    )
    return try makeBuffer(values: words, label: "QH5 block metadata")
  }

  private func validateDevice(plan: Metal4DSTEMIndexedLoadPlan) throws {
    guard plan.maximumIndividualMetalBufferBytes <= UInt64(device.maxBufferLength) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The largest indexed-load buffer requires "
          + "\(plan.maximumIndividualMetalBufferBytes) bytes, exceeding this Metal device's "
          + "maximum single-buffer length of \(device.maxBufferLength) bytes."
      )
    }
  }

  private func makeByteMask(
    count: Int,
    selected: [Int],
    label: String
  ) throws -> MTLBuffer {
    var values = [UInt8](repeating: 0, count: count)
    for index in selected.sorted() {
      guard values.indices.contains(index) else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "\(label) contains detector index \(index), outside 0..<\(count)."
        )
      }
      values[index] = 1
    }
    return try makeBuffer(values: values, label: label)
  }

  private func makeBuffer<T>(values: [T], label: String) throws -> MTLBuffer {
    guard !values.isEmpty else {
      throw Metal4DSTEMStreamingIOError.invalidRequest("\(label) cannot be empty.")
    }
    let byteCount = values.count.multipliedReportingOverflow(
      by: MemoryLayout<T>.stride
    )
    guard !byteCount.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest("\(label) byte count overflows Int.")
    }
    return try values.withUnsafeBytes { raw in
      guard let base = raw.baseAddress,
        let buffer = device.makeBuffer(
          bytes: base,
          length: byteCount.partialValue,
          options: .storageModeShared
        )
      else {
        throw Metal4DSTEMStreamingIOError.allocationFailed(
          label: label,
          bytes: UInt64(byteCount.partialValue)
        )
      }
      return buffer
    }
  }

  private func makeBuffer(
    bytes: UInt64,
    options: MTLResourceOptions = .storageModeShared,
    label: String
  ) throws -> MTLBuffer {
    guard bytes > 0, let length = Int(exactly: bytes),
      bytes <= UInt64(device.maxBufferLength)
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "\(label) requires an unsupported Metal buffer length of \(bytes) bytes."
      )
    }
    guard let buffer = device.makeBuffer(length: length, options: options) else {
      throw Metal4DSTEMStreamingIOError.allocationFailed(label: label, bytes: bytes)
    }
    return buffer
  }

  private func byteCount(
    count: Int,
    stride: Int,
    label: String
  ) throws -> Int {
    let result = count.multipliedReportingOverflow(by: stride)
    guard count > 0, stride > 0, !result.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest("\(label) byte count overflows Int.")
    }
    return result.partialValue
  }

  private func exactUInt32(_ value: Int, label: String) throws -> UInt32 {
    guard let result = UInt32(exactly: value) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "\(label) exceeds the reusable Metal kernel's 32-bit range."
      )
    }
    return result
  }

  private func wait(_ command: MTLCommandBuffer) throws {
    command.waitUntilCompleted()
    if let error = command.error {
      throw Metal4DSTEMStreamingIOError.commandFailed(error.localizedDescription)
    }
  }

  private func values<T>(
    from buffer: MTLBuffer,
    count: Int,
    as type: T.Type
  ) -> [T] {
    Array(
      UnsafeBufferPointer(
        start: buffer.contents().bindMemory(to: type, capacity: count),
        count: count
      )
    )
  }

  private func residentCacheMetadata(
    source: Native4DSTEMIndexedSource,
    plan: Metal4DSTEMIndexedBinnedLoadPlan
  ) throws -> Metal4DSTEMResidentCacheMetadata {
    let sourceAudit = plan.sourceAudit
    try sourceAudit.validate()
    let audit = Native4DSTEMValueRangeAudit(
      sourceIdentitySHA256: sourceAudit.sourceIdentitySHA256,
      sourceDtype: sourceAudit.sourceDtype.rawValue,
      badPixelIndices: sourceAudit.badPixelIndices,
      maximum: sourceAudit.maximumSourceCount,
      pixelsAbove255: sourceAudit.pixelsAbove255
    )
    guard try audit.sha256() == sourceAudit.auditSHA256 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The exact source audit digest does not match its canonical cache metadata."
      )
    }
    let provenance = plan.binningProvenance
    var sourceURLs =
      source.dataset.masterPath.map {
        [URL(fileURLWithPath: $0)]
      } ?? []
    sourceURLs.append(contentsOf: source.shards.map(\.sourceURL))
    var seenSourcePaths = Set<String>()
    let sourceIdentities: [Metal4DSTEMSourceIdentity] = try sourceURLs.compactMap {
      url in
      let path = url.standardizedFileURL.path
      guard seenSourcePaths.insert(path).inserted else { return nil }
      return try Metal4DSTEMSourceIdentity(url: url)
    }
    return Metal4DSTEMResidentCacheMetadata(
      datasetID: plan.productPlan.datasetID,
      sourceIdentitySHA256: sourceAudit.sourceIdentitySHA256,
      valueRangeAuditSHA256: sourceAudit.auditSHA256,
      valueRangeAudit: audit,
      sources: sourceIdentities,
      sourceScanRows: provenance.sourceScanRows,
      sourceScanColumns: provenance.sourceScanColumns,
      sourceDetectorRows: provenance.sourceDetectorRows,
      sourceDetectorColumns: provenance.sourceDetectorColumns,
      sourceDtype: provenance.sourceDtype.rawValue,
      outputScanRows: provenance.outputScanRows,
      outputScanColumns: provenance.outputScanColumns,
      outputDetectorRows: provenance.outputDetectorRows,
      outputDetectorColumns: provenance.outputDetectorColumns,
      outputDtype: provenance.outputDtype.rawValue,
      scanRowStart: provenance.scanRegion.rowStart,
      scanRowStop: provenance.scanRegion.rowStop,
      scanColumnStart: provenance.scanRegion.columnStart,
      scanColumnStop: provenance.scanRegion.columnStop,
      scanBin: provenance.scanBin,
      detectorBin: provenance.detectorBin,
      badPixelIndices: provenance.badPixelIndices,
      maxCount: provenance.maximumSourceCount,
      pixelsAbove255: provenance.pixelsAbove255,
      samplingPropagation: residentSamplingPropagation(plan.samplingPropagation),
      payloadBytes: provenance.outputPayloadBytes
    )
  }

  private func residentSamplingPropagation(
    _ value: Metal4DSTEMSamplingPropagation
  ) -> Metal4DSTEMResidentSamplingPropagation {
    func axis(
      _ value: Metal4DSTEMAxisSampling?
    ) -> Metal4DSTEMResidentAxisSampling? {
      value.map {
        Metal4DSTEMResidentAxisSampling(
          row: $0.row,
          column: $0.column,
          unit: $0.unit,
          provenance: $0.provenance,
          evidence: $0.evidence
        )
      }
    }
    func state(
      _ value: Metal4DSTEMSamplingPropagationState
    ) -> Metal4DSTEMResidentSamplingState {
      switch value {
      case .unavailable: .unavailable
      case .unchanged: .unchanged
      case .uniformlyScaled: .uniformlyScaled
      case .nonuniformEdgeBins: .nonuniformEdgeBins
      }
    }
    return Metal4DSTEMResidentSamplingPropagation(
      sourceScan: axis(value.sourceScan),
      workingScan: axis(value.workingScan),
      sourceDetector: axis(value.sourceDetector),
      workingDetector: axis(value.workingDetector),
      scanState: state(value.scanState),
      detectorState: state(value.detectorState),
      scanRegionRowStartInSourcePixels: value.scanRegionRowStartInSourcePixels,
      scanRegionColumnStartInSourcePixels:
        value.scanRegionColumnStartInSourcePixels,
      firstWorkingScanCenterRowInSourcePixels:
        value.firstWorkingScanCenterRowInSourcePixels,
      firstWorkingScanCenterColumnInSourcePixels:
        value.firstWorkingScanCenterColumnInSourcePixels,
      firstWorkingDetectorCenterRowInSourcePixels:
        value.firstWorkingDetectorCenterRowInSourcePixels,
      firstWorkingDetectorCenterColumnInSourcePixels:
        value.firstWorkingDetectorCenterColumnInSourcePixels
    )
  }
}

private struct DetectorParameters {
  var frameCount: UInt32
  var detectorPixels: UInt32
  var globalFrameOffset: UInt32
  var padding: UInt32 = 0
}

/// Strong resource retention for one asynchronously submitted source slice.
private struct PendingIndexedCommand {
  let command: MTLCommandBuffer
  let mappedSource: MappedMetalSource
  let metadata: MTLBuffer
  let binningDispatchCount: Int
}

private final class BinnedOutputContext {
  private enum Storage {
    case resident([MTLBuffer])
    case streaming((Metal4DSTEMExactBinningShard, MTLBuffer) throws -> Void)
  }

  let plan: Metal4DSTEMIndexedBinnedLoadPlan
  let cacheState: String
  private let storage: Storage
  private var activeShardIndex: Int?
  private var activeDestination: MTLBuffer?
  private(set) var destinationAllocationSeconds = 0.0
  private(set) var payloadWriteSeconds = 0.0
  private(set) var completedShardCount = 0
  private(set) var peakWorkingMetalBytes: UInt64 = 0

  init(
    resident plan: Metal4DSTEMIndexedBinnedLoadPlan,
    destinations: [MTLBuffer]
  ) {
    self.plan = plan
    storage = .resident(destinations)
    cacheState =
      "prepared_qh5_index_exact_binned_resident_source_pages_unspecified"
    peakWorkingMetalBytes = destinations.reduce(0) { total, buffer in
      total + UInt64(buffer.length)
    }
  }

  init(
    streaming plan: Metal4DSTEMIndexedBinnedLoadPlan,
    consume: @escaping (Metal4DSTEMExactBinningShard, MTLBuffer) throws -> Void
  ) {
    self.plan = plan
    storage = .streaming(consume)
    cacheState =
      "prepared_qh5_index_exact_binned_transactional_file_source_pages_unspecified"
  }

  func requiresDestinationTransition(for range: Range<Int>) throws -> Bool {
    switch storage {
    case .resident:
      return false
    case .streaming:
      return try shard(containing: range).index != activeShardIndex
    }
  }

  func prepareDestination(
    for range: Range<Int>,
    makeBuffer: (Metal4DSTEMExactBinningShard) throws -> MTLBuffer
  ) throws {
    guard case .streaming = storage else { return }
    let shard = try shard(containing: range)
    guard activeShardIndex == nil, activeDestination == nil else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The previous exact output shard must complete before allocating the next shard."
      )
    }
    guard shard.index == completedShardCount else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Exact output shards must be streamed once in canonical scan-row order; "
          + "expected shard \(completedShardCount), received \(shard.index)."
      )
    }
    let started = CFAbsoluteTimeGetCurrent()
    let destination = try makeBuffer(shard)
    destinationAllocationSeconds += CFAbsoluteTimeGetCurrent() - started
    guard UInt64(destination.length) == shard.payloadBytes else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Exact output shard \(shard.index) allocated \(destination.length) bytes; "
          + "expected \(shard.payloadBytes)."
      )
    }
    activeShardIndex = shard.index
    activeDestination = destination
    peakWorkingMetalBytes = max(peakWorkingMetalBytes, UInt64(destination.length))
  }

  func destination(for shardIndex: Int) throws -> MTLBuffer {
    switch storage {
    case .resident(let destinations):
      guard destinations.indices.contains(shardIndex) else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Exact destination shard index \(shardIndex) is outside the resident plan."
        )
      }
      return destinations[shardIndex]
    case .streaming:
      guard activeShardIndex == shardIndex, let activeDestination else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Exact output shard \(shardIndex) was not prepared before encoding."
        )
      }
      return activeDestination
    }
  }

  func finishActiveDestination() throws {
    guard case .streaming(let consume) = storage,
      let activeShardIndex,
      let activeDestination
    else { return }
    let shard = plan.shardPlan.shards[activeShardIndex]
    let started = CFAbsoluteTimeGetCurrent()
    try consume(shard, activeDestination)
    payloadWriteSeconds += CFAbsoluteTimeGetCurrent() - started
    completedShardCount += 1
    self.activeShardIndex = nil
    self.activeDestination = nil
  }

  private func shard(
    containing range: Range<Int>
  ) throws -> Metal4DSTEMExactBinningShard {
    guard !range.isEmpty,
      let shard = plan.shardPlan.shards.first(where: { candidate in
        let stop =
          candidate.outputScanPositionStart
          + candidate.outputScanPositionCount
        return range.lowerBound >= candidate.outputScanPositionStart
          && range.upperBound <= stop
      })
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Each decoded window must fit completely within one exact output shard. "
          + "Align the private output-shard size to the decoded-window geometry."
      )
    }
    return shard
  }

  static func validateStreamable(
    _ plan: Metal4DSTEMIndexedBinnedLoadPlan
  ) throws {
    let context = BinnedOutputContext(streaming: plan) { _, _ in }
    for window in plan.productPlan.windows {
      _ = try context.shard(containing: window.globalFrameRange)
    }
  }
}

private final class MappedMetalSource {
  let buffer: MTLBuffer

  init(
    url: URL,
    expectedBytes: UInt64,
    device: MTLDevice
  ) throws {
    let descriptor = open(url.path, O_RDONLY)
    guard descriptor >= 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Could not open indexed source \(url.lastPathComponent)."
      )
    }
    var status = stat()
    guard fstat(descriptor, &status) == 0,
      let fileBytes = UInt64(exactly: status.st_size),
      fileBytes == expectedBytes,
      let fileLength = Int(exactly: fileBytes),
      fileLength > 0
    else {
      close(descriptor)
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Indexed source \(url.lastPathComponent) changed size after validation."
      )
    }
    let pageBytes = Int(getpagesize())
    let rounded = fileLength.addingReportingOverflow(pageBytes - 1)
    guard !rounded.overflow else {
      close(descriptor)
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Indexed source \(url.lastPathComponent) mapping size overflows Int."
      )
    }
    let mappedBytes = (rounded.partialValue / pageBytes) * pageBytes
    guard mappedBytes <= device.maxBufferLength else {
      close(descriptor)
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Indexed source shard \(url.lastPathComponent) exceeds the Metal "
          + "single-buffer limit; prepare smaller source shards."
      )
    }
    let pointer = mmap(
      nil,
      mappedBytes,
      PROT_READ | PROT_WRITE,
      MAP_PRIVATE,
      descriptor,
      0
    )
    close(descriptor)
    guard pointer != MAP_FAILED, let pointer else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Could not memory-map indexed source \(url.lastPathComponent)."
      )
    }
    guard
      let buffer = device.makeBuffer(
        bytesNoCopy: pointer,
        length: mappedBytes,
        options: .storageModeShared,
        deallocator: { address, length in munmap(address, length) }
      )
    else {
      munmap(pointer, mappedBytes)
      throw Metal4DSTEMStreamingIOError.allocationFailed(
        label: "mapped source \(url.lastPathComponent)",
        bytes: UInt64(mappedBytes)
      )
    }
    self.buffer = buffer
  }
}
