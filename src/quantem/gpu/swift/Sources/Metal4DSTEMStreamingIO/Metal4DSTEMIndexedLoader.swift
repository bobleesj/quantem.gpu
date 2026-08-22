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

/// How prepared compressed source shards reach Metal-visible memory.
///
/// The caller selects this execution resource contract; QuantEM.GPU does not
/// infer device admission policy. Memory mapping minimizes explicitly allocated
/// staging. Buffered read-ahead uses bounded shared Metal buffers so storage
/// reads can overlap exact decode, reduction, and detector binning.
public enum Metal4DSTEMIndexedSourceTransfer: Codable, Equatable, Sendable {
  /// Map each source shard and retain mappings only while Metal consumes them.
  case memoryMapped

  /// Read source shards into shared Metal buffers ahead of consumption.
  ///
  /// A value of one is double buffering: one shard is consumed while the next
  /// is read. Larger values require more explicitly allocated source staging.
  case bufferedReadAhead(prefetchShardCount: Int)

  public var prefetchShardCount: Int {
    switch self {
    case .memoryMapped: 0
    case .bufferedReadAhead(let count): count
    }
  }

  fileprivate func validate() throws {
    if case .bufferedReadAhead(let count) = self, count <= 0 {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Buffered source read-ahead requires a positive prefetch-shard count. "
          + "Use memoryMapped when no buffered read-ahead is requested."
      )
    }
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
  /// Conservative bound for all source buffers retained by this transfer mode.
  public let maximumRetainedSourceBufferBytes: UInt64
  /// Package Metal allocations plus buffered source transfer, when requested.
  public let estimatedAllocatedMetalBytesIncludingSourceTransfer: UInt64
  public let maximumInFlightCommandBuffers: Int
  public let sourceTransfer: Metal4DSTEMIndexedSourceTransfer
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
    detectorBands: Metal4DSTEMDetectorBands,
    sourceTransfer: Metal4DSTEMIndexedSourceTransfer = .memoryMapped
  ) throws {
    try sourceTransfer.validate()
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
    switch sourceTransfer {
    case .memoryMapped:
      maximumRetainedSourceBufferBytes = maximumInFlightMappedSourceBytes
      estimatedAllocatedMetalBytesIncludingSourceTransfer = working
    case .bufferedReadAhead(let prefetchShardCount):
      let retainedCount = min(
        source.shards.count,
        maximumInFlightCommandBuffers + 1 + prefetchShardCount
      )
      maximumRetainedSourceBufferBytes = try Self.sum(
        Array(mappedSourceBytes.sorted(by: >).prefix(retainedCount)),
        label: "retained buffered source staging"
      )
      estimatedAllocatedMetalBytesIncludingSourceTransfer = try Self.sum(
        [working, maximumRetainedSourceBufferBytes],
        label: "package Metal storage including source transfer"
      )
    }
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
    self.sourceTransfer = sourceTransfer
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

  fileprivate static func mappedBufferBytes(_ sourceBytes: UInt64) throws -> UInt64 {
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
  public let sourceOpenAndStatSeconds: Double
  public let sourceMmapSeconds: Double
  public let sourceReadSeconds: Double
  public let sourceMetalBufferSeconds: Double
  public let sourceTransfer: Metal4DSTEMIndexedSourceTransfer
  public let gpuSeconds: Double
  public let windowCount: Int
  public let sliceCount: Int
  public let commandBufferCount: Int
  public let peakInFlightCommandBuffers: Int
  public let peakInFlightMappedSourceBytes: UInt64
  /// Peak source-buffer bytes retained by commands, active work, and read-ahead.
  public let peakRetainedSourceBufferBytes: UInt64
  public let mappedCompressedSourceBytes: UInt64
  public let maximumMappedCompressedSourceBytes: UInt64
  public let maximumMappedSourceBufferBytes: UInt64
  public let maximumDecodedSliceBytes: UInt64
  public let cacheState: String
  public let synchronizedStageSeconds: Metal4DSTEMIndexedStageSeconds?
}

/// Opt-in synchronized Metal stage attribution for package benchmarks.
public struct Metal4DSTEMIndexedStageSeconds: Codable, Equatable, Sendable {
  public let decodeAndAudit: Double
  public let exactScanProductsAndDetectorBinning: Double
  public let exactDetectorSum: Double
  public let commandQueueRemainder: Double
  public let commandBuffers: Int
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
  private let auditedLow8DecodePipeline: MTLComputePipelineState
  private let auditedLow8Octet192DecodePipeline: MTLComputePipelineState
  private let productPipeline: MTLComputePipelineState
  private let auditedLow8ProductPipeline: MTLComputePipelineState
  private let auditedLow8SIMDProductPipeline: MTLComputePipelineState
  private let auditedLow8Detector192SIMDProductPipeline: MTLComputePipelineState
  private let detectorSumPipeline: MTLComputePipelineState
  private let auditedLow8TiledDetectorSumPipeline: MTLComputePipelineState
  private let fusedTiledLow8Bin1ProductsDetectorPartialsPipeline: MTLComputePipelineState
  private let fusedTiledLow8Bin2ProductsDetectorPartialsPipeline: MTLComputePipelineState
  private let detectorAccumulateU16PartialsPipeline: MTLComputePipelineState
  private let privateResidentPagePreparationPipeline: MTLComputePipelineState
  private let exactBinner: Metal4DSTEMExactBinner
  private let stageProfiler: IndexedStageProfiler?
  private var profiledPreviousGPUEndTime: CFTimeInterval?
  private var profiledGPUIdleSeconds = 0.0
  private var profiledCommandCreationSeconds = 0.0
  private var profiledEncodingSeconds = 0.0
  private var profiledCommitSeconds = 0.0
  private var profiledWaitSeconds = 0.0

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
        let auditedLow8Decode = hdf5.makeFunction(
          name: Metal4DSTEMKernels.decodeU16AuditedLow8DirectFunction
        ),
        let auditedLow8Octet192Decode = hdf5.makeFunction(
          name: Metal4DSTEMKernels.decodeU16AuditedLow8DirectOctet192Function
        ),
        let products = detector.makeFunction(
          name: Metal4DSTEMKernels.detectorProductsU16ExactU64Function
        ),
        let auditedLow8Products = detector.makeFunction(
          name: Metal4DSTEMKernels.detectorProductsU8ExactU64Function
        ),
        let auditedLow8SIMDProducts = detector.makeFunction(
          name: Metal4DSTEMKernels.detectorProductsU8ExactU32SIMDToU64Function
        ),
        let auditedLow8Detector192SIMDProducts = detector.makeFunction(
          name: Metal4DSTEMKernels
            .detectorProductsU8Detector192ExactU32SIMDToU64Function
        ),
        let detectorSum = detector.makeFunction(
          name: Metal4DSTEMKernels.detectorAccumulateU16U64Function
        ),
        let auditedLow8TiledDetectorSum = detector.makeFunction(
          name: Metal4DSTEMKernels.detectorAccumulateU8U64FrameTiledFunction
        ),
        let fusedTiledLow8Bin1ProductsDetectorPartials = detector.makeFunction(
          name: Metal4DSTEMKernels
            .contiguousDetectorBin1U8ProductsDetectorPartialsTiled32x8Function
        ),
        let fusedTiledLow8Bin2ProductsDetectorPartials = detector.makeFunction(
          name: Metal4DSTEMKernels
            .contiguousDetectorBin2U8ProductsDetectorPartialsTiled32x8Function
        ),
        let detectorAccumulateU16Partials = detector.makeFunction(
          name: Metal4DSTEMKernels.detectorAccumulateU16PartialsU64Function
        ),
        let privateResidentPagePreparation = detector.makeFunction(
          name: Metal4DSTEMKernels.preparePrivateResidentPagesFunction
        )
      else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "the packaged exact decode or reduction function is missing"
        )
      }
      decodePipeline = try device.makeComputePipelineState(function: decode)
      auditedLow8DecodePipeline = try device.makeComputePipelineState(
        function: auditedLow8Decode
      )
      auditedLow8Octet192DecodePipeline = try device.makeComputePipelineState(
        function: auditedLow8Octet192Decode
      )
      productPipeline = try device.makeComputePipelineState(function: products)
      auditedLow8ProductPipeline = try device.makeComputePipelineState(
        function: auditedLow8Products
      )
      auditedLow8SIMDProductPipeline = try device.makeComputePipelineState(
        function: auditedLow8SIMDProducts
      )
      auditedLow8Detector192SIMDProductPipeline = try device.makeComputePipelineState(
        function: auditedLow8Detector192SIMDProducts
      )
      detectorSumPipeline = try device.makeComputePipelineState(function: detectorSum)
      auditedLow8TiledDetectorSumPipeline = try device.makeComputePipelineState(
        function: auditedLow8TiledDetectorSum
      )
      fusedTiledLow8Bin1ProductsDetectorPartialsPipeline =
        try device.makeComputePipelineState(
          function: fusedTiledLow8Bin1ProductsDetectorPartials
        )
      fusedTiledLow8Bin2ProductsDetectorPartialsPipeline =
        try device
        .makeComputePipelineState(
          function: fusedTiledLow8Bin2ProductsDetectorPartials
        )
      detectorAccumulateU16PartialsPipeline = try device.makeComputePipelineState(
        function: detectorAccumulateU16Partials
      )
      privateResidentPagePreparationPipeline = try device.makeComputePipelineState(
        function: privateResidentPagePreparation
      )
      exactBinner = try Metal4DSTEMExactBinner(
        device: device,
        detectorLibrary: detector
      )
      stageProfiler = try IndexedStageProfiler.makeIfRequested(device: device)
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
    try validateBinnedLoadPlan(source: source, plan: plan)
    let allocationStarted = CFAbsoluteTimeGetCurrent()
    let allocation = try makeResidentDestinations(plan: plan)
    let allocationSeconds = CFAbsoluteTimeGetCurrent() - allocationStarted
    return try loadExactBinnedShards(
      source: source,
      plan: plan,
      destinations: allocation.buffers,
      destinationPreparations: allocation.preparations,
      destinationAllocationSeconds: allocationSeconds,
      totalStarted: totalStarted,
      shouldCancel: shouldCancel
    )
  }

  /// Decode once and fill exact caller-owned resident shards.
  ///
  /// This overload separates reusable allocation and device-admission policy
  /// from exact loading. `destinationShards` must contain one distinct buffer
  /// for every shard in `plan`, in canonical shard order, with the plan's
  /// declared residency and each buffer's length exactly equal to its
  /// planned payload. On success every payload byte has been overwritten with
  /// the exact result. If loading throws, destination contents are unspecified
  /// and must not be published.
  ///
  /// The caller owns buffer allocation, reuse, synchronization outside this
  /// synchronous call, retention, eviction, and device-memory admission.
  public func loadExactBinnedShards(
    source: Native4DSTEMIndexedSource,
    plan: Metal4DSTEMIndexedBinnedLoadPlan,
    destinationShards: [MTLBuffer],
    shouldCancel: () -> Bool = { false }
  ) throws -> Metal4DSTEMIndexedBinnedLoadResult {
    let totalStarted = CFAbsoluteTimeGetCurrent()
    try validateBinnedLoadPlan(source: source, plan: plan)
    try validateResidentDestinations(destinationShards, plan: plan)
    return try loadExactBinnedShards(
      source: source,
      plan: plan,
      destinations: destinationShards,
      destinationPreparations: [],
      destinationAllocationSeconds: 0,
      totalStarted: totalStarted,
      shouldCancel: shouldCancel
    )
  }

  private func loadExactBinnedShards(
    source: Native4DSTEMIndexedSource,
    plan: Metal4DSTEMIndexedBinnedLoadPlan,
    destinations: [MTLBuffer],
    destinationPreparations: [MTLCommandBuffer],
    destinationAllocationSeconds: Double,
    totalStarted: CFAbsoluteTime,
    shouldCancel: () -> Bool
  ) throws -> Metal4DSTEMIndexedBinnedLoadResult {
    let context = BinnedOutputContext(
      resident: plan,
      destinations: destinations,
      preparations: destinationPreparations
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
        destinationAllocationSeconds: destinationAllocationSeconds,
        totalWallSeconds: CFAbsoluteTimeGetCurrent() - totalStarted,
        binningDispatchCount: loaded.binningDispatchCount,
        workingPayloadBytes: plan.workingPayloadBytes,
        shardCount: plan.shardPlan.shards.count,
        maximumShardBytes: plan.shardPlan.maximumActualShardBytes,
        destinationStorageMode: plan.destinationStorageMode
      )
    )
  }

  private func validateBinnedLoadPlan(
    source: Native4DSTEMIndexedSource,
    plan: Metal4DSTEMIndexedBinnedLoadPlan
  ) throws {
    let expectedPlan = try Metal4DSTEMIndexedBinnedLoadPlan(
      source: source,
      maximumDecodedWindowBytes: plan.productPlan.maximumDecodedWindowBytes,
      detectorBands: plan.productPlan.detectorBands,
      detectorBin: plan.binningProvenance.detectorBin,
      sourceAudit: plan.sourceAudit,
      maximumShardBytes: plan.shardPlan.maximumShardBytes,
      residentStorage: plan.residentStorage,
      sourceTransfer: plan.productPlan.sourceTransfer
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
  }

  private func validateResidentDestinations(
    _ destinations: [MTLBuffer],
    plan: Metal4DSTEMIndexedBinnedLoadPlan
  ) throws {
    guard destinations.count == plan.shardPlan.shards.count else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Exact resident loading requires \(plan.shardPlan.shards.count) destination "
          + "shards in canonical order; received \(destinations.count)."
      )
    }
    guard Set(destinations.map(ObjectIdentifier.init)).count == destinations.count else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Exact resident destination shards must be distinct buffers; aliases are invalid."
      )
    }
    for (shard, destination) in zip(plan.shardPlan.shards, destinations) {
      guard destination.device.registryID == device.registryID else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Exact resident destination shard \(shard.index) belongs to a different "
            + "Metal device. Allocate every shard on the loader's device."
        )
      }
      guard destination.storageMode == plan.residentStorage.metalStorageMode else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Exact resident destination shard \(shard.index) must use "
            + "\(plan.destinationStorageMode) Metal storage; received "
            + "\(destination.storageMode.rawValue)."
        )
      }
      guard UInt64(destination.length) == shard.payloadBytes else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Exact resident destination shard \(shard.index) requires exactly "
            + "\(shard.payloadBytes) bytes; received \(destination.length)."
        )
      }
    }
  }

  /// Decode once and transactionally write an exact file-backed working volume.
  ///
  /// Only one output-shard buffer is allocated in Metal. Each shard is complete
  /// before it is appended to a unique temporary payload and the same bounded
  /// shared staging buffer is reused for the next shard. `residentStorage` is
  /// retained as part of plan validation but does not change the canonical
  /// file-backed payload or its transient shared staging.
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
      maximumShardBytes: plan.shardPlan.maximumShardBytes,
      residentStorage: plan.residentStorage,
      sourceTransfer: plan.productPlan.sourceTransfer
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
      consume: { shard, buffer in
        guard let payloadLength = Int(exactly: shard.payloadBytes) else {
          throw Metal4DSTEMStreamingIOError.invalidRequest(
            "Exact output-shard byte count does not fit this process."
          )
        }
        try writer.append(pointer: buffer.contents(), length: payloadLength)
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
      detectorBands: plan.detectorBands,
      sourceTransfer: plan.sourceTransfer
    )
    guard expectedPlan == plan else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The indexed load plan does not match the supplied source or detector bands."
      )
    }
    try validateDevice(plan: plan)
    stageProfiler?.reset()
    profiledPreviousGPUEndTime = nil
    profiledGPUIdleSeconds = 0
    profiledCommandCreationSeconds = 0
    profiledEncodingSeconds = 0
    profiledCommitSeconds = 0
    profiledWaitSeconds = 0
    let stagingDtype = binned?.plan.binningProvenance.stagingDtype ?? .uint16
    if stagingDtype == .uint8 {
      guard let binned, binned.plan.sourceAudit.provesLosslessUInt8Staging else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Compact indexed staging requires an identity-bound lossless uint8 audit."
        )
      }
    }
    let usesFusedTiledLow8Bin1Products = try supportsFusedAuditedLow8(
      plan: plan,
      binned: binned,
      stagingDtype: stagingDtype,
      detectorBin: 1
    )
    let usesFusedTiledLow8Bin2Products = try supportsFusedAuditedLow8(
      plan: plan,
      binned: binned,
      stagingDtype: stagingDtype,
      detectorBin: 2
    )
    let usesFusedTiledLow8Products =
      usesFusedTiledLow8Bin1Products || usesFusedTiledLow8Bin2Products
    let started = CFAbsoluteTimeGetCurrent()
    let frameCount = plan.logicalFrameCount
    let detectorPixels = plan.sourceDetectorRows * plan.sourceDetectorColumns
    let mapBytes = try byteCount(
      count: frameCount,
      stride:
        usesFusedTiledLow8Products ? 4 : 8,
      label: "one scan-map accumulator"
    )
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
    let maximumNativeSliceBytes = try plan.windows.flatMap(\.slices).reduce(UInt64(0)) {
      maximum, slice in
      let sliceBytes = UInt64(slice.globalFrameRange.count)
        .multipliedReportingOverflow(by: source.decodedBytesPerFrame)
      guard !sliceBytes.overflow else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "One indexed decoded slice exceeds the reusable loader's UInt64 range."
        )
      }
      return max(maximum, sliceBytes.partialValue)
    }
    guard maximumNativeSliceBytes > 0,
      maximumNativeSliceBytes.isMultiple(
        of: UInt64(plan.stagingDtype.bytesPerValue)
      )
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The indexed source has no valid decoded slice storage."
      )
    }
    let scratchBytes =
      maximumNativeSliceBytes
      / UInt64(plan.stagingDtype.bytesPerValue)
      * UInt64(stagingDtype.bytesPerValue)
    let scratch = try makeBuffer(
      bytes: scratchBytes,
      options: .storageModePrivate,
      label: "decoded exact staging window"
    )
    let detectorPartialScratch: MTLBuffer?
    if usesFusedTiledLow8Products {
      guard let binned,
        binned.plan.maximumDetectorPartialScratchBytes > 0
      else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "The exact detector-partial path is missing its planned scratch bound."
        )
      }
      detectorPartialScratch = try makeBuffer(
        bytes: binned.plan.maximumDetectorPartialScratchBytes,
        options: .storageModePrivate,
        label: "exact detector partials"
      )
    } else {
      detectorPartialScratch = nil
    }
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
    let setupFinished = CFAbsoluteTimeGetCurrent()

    var activeShardIndex: Int?
    var activeSource: MappedMetalSource?
    var mapSeconds = 0.0
    var sourceOpenAndStatSeconds = 0.0
    var sourceMmapSeconds = 0.0
    var sourceReadSeconds = 0.0
    var sourceMetalBufferSeconds = 0.0
    var gpuSeconds = 0.0
    var sliceCount = 0
    var binningDispatchCount = 0
    var mappedCompressedSourceBytes: UInt64 = 0
    var mappedShards = Set<Int>()
    var maximumMappedCompressedSourceBytes: UInt64 = 0
    var maximumMappedSourceBufferBytes: UInt64 = 0
    var maximumDecodedSliceBytes: UInt64 = 0
    var pendingCommands: [PendingIndexedCommand] = []
    var commandBufferCount = 0
    var peakInFlightCommandBuffers = 0
    var peakInFlightMappedSourceBytes: UInt64 = 0
    var peakRetainedSourceBufferBytes: UInt64 = 0
    let bufferedPrefetchDepth = plan.sourceTransfer.prefetchShardCount
    var bufferedSourceFutures: [Int: BufferedMetalSourceFuture] = [:]

    func retainedSourceBufferBytes() throws -> UInt64 {
      var seen = Set<ObjectIdentifier>()
      var bytes: UInt64 = 0
      func add(_ value: UInt64) throws {
        let updated = bytes.addingReportingOverflow(value)
        guard !updated.overflow else {
          throw Metal4DSTEMStreamingIOError.invalidRequest(
            "Retained source-buffer byte accounting overflows UInt64."
          )
        }
        bytes = updated.partialValue
      }
      for pending in pendingCommands {
        for mapped in pending.mappedSources
        where seen.insert(ObjectIdentifier(mapped)).inserted {
          try add(UInt64(mapped.buffer.length))
        }
      }
      if let activeSource,
        seen.insert(ObjectIdentifier(activeSource)).inserted
      {
        try add(UInt64(activeSource.buffer.length))
      }
      for future in bufferedSourceFutures.values {
        try add(future.expectedBufferBytes)
      }
      return bytes
    }

    func recordRetainedSourceBufferPeak() throws {
      peakRetainedSourceBufferBytes = max(
        peakRetainedSourceBufferBytes,
        try retainedSourceBufferBytes()
      )
    }

    func scheduleBufferedSource(_ shardIndex: Int) throws {
      guard bufferedPrefetchDepth > 0,
        source.shards.indices.contains(shardIndex),
        bufferedSourceFutures[shardIndex] == nil
      else { return }
      let shard = source.shards[shardIndex]
      bufferedSourceFutures[shardIndex] = try BufferedMetalSourceFuture(
        url: shard.sourceURL,
        expectedBytes: shard.index.metadata.sourceBytes,
        expectedModificationNanoseconds: shard.index.metadata.sourceMtimeNs,
        device: device
      )
      try recordRetainedSourceBufferPeak()
    }

    let streamStarted = CFAbsoluteTimeGetCurrent()
    let maximumPendingCommandCount = Self.maximumInFlightCommandBuffers
    for shardIndex in 0..<min(bufferedPrefetchDepth, source.shards.count) {
      try scheduleBufferedSource(shardIndex)
    }
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
              bytes: binned.plan.shardPlan.maximumActualShardBytes,
              options: .storageModeShared,
              label: "reusable transactional exact working-volume shard \(shard.index)"
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
          if bufferedPrefetchDepth > 0 {
            try scheduleBufferedSource(slice.shardIndex)
            try scheduleBufferedSource(slice.shardIndex + bufferedPrefetchDepth)
            guard
              let future = bufferedSourceFutures.removeValue(
                forKey: slice.shardIndex
              )
            else {
              throw Metal4DSTEMStreamingIOError.invalidRequest(
                "The bounded source prefetch lost shard \(slice.shardIndex)."
              )
            }
            let mapStarted = CFAbsoluteTimeGetCurrent()
            activeSource = try future.value()
            activeShardIndex = slice.shardIndex
            mapSeconds += CFAbsoluteTimeGetCurrent() - mapStarted
            sourceOpenAndStatSeconds += activeSource?.openAndStatSeconds ?? 0
            sourceMmapSeconds += activeSource?.mmapSeconds ?? 0
            sourceReadSeconds += activeSource?.readSeconds ?? 0
            sourceMetalBufferSeconds += activeSource?.metalBufferSeconds ?? 0
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
            try recordRetainedSourceBufferPeak()
          } else {
            let mapStarted = CFAbsoluteTimeGetCurrent()
            activeSource = try autoreleasepool {
              try MappedMetalSource(
                url: shard.sourceURL,
                expectedBytes: shard.index.metadata.sourceBytes,
                expectedModificationNanoseconds: shard.index.metadata.sourceMtimeNs,
                device: device
              )
            }
            activeShardIndex = slice.shardIndex
            mapSeconds += CFAbsoluteTimeGetCurrent() - mapStarted
            sourceOpenAndStatSeconds += activeSource?.openAndStatSeconds ?? 0
            sourceMmapSeconds += activeSource?.mmapSeconds ?? 0
            sourceReadSeconds += activeSource?.readSeconds ?? 0
            sourceMetalBufferSeconds += activeSource?.metalBufferSeconds ?? 0
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
        let commandStarted = CFAbsoluteTimeGetCurrent()
        guard let command = makeStreamingCommandBuffer() else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            "the device could not start an indexed decode command"
          )
        }
        profiledCommandCreationSeconds +=
          CFAbsoluteTimeGetCurrent() - commandStarted
        let counterSamples = try stageProfiler?.makeSampleBuffer()
        let encodingStarted = CFAbsoluteTimeGetCurrent()
        let encoded = try autoreleasepool {
          try encodeSlice(
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
            detectorPartialScratch: detectorPartialScratch,
            stagingDtype: stagingDtype,
            usesFusedTiledLow8Bin1Products: usesFusedTiledLow8Bin1Products,
            usesFusedTiledLow8Bin2Products: usesFusedTiledLow8Bin2Products,
            binned: binned,
            command: command,
            counterSamples: counterSamples
          )
        }
        profiledEncodingSeconds += CFAbsoluteTimeGetCurrent() - encodingStarted
        let commitStarted = CFAbsoluteTimeGetCurrent()
        command.commit()
        profiledCommitSeconds += CFAbsoluteTimeGetCurrent() - commitStarted
        commandBufferCount += 1
        binningDispatchCount += encoded.binningDispatchCount
        pendingCommands.append(
          PendingIndexedCommand(
            command: command,
            mappedSources: [mapped],
            metadataBuffers: [encoded.metadata],
            counterSamples: counterSamples,
            binningDispatchCount: encoded.binningDispatchCount
          )
        )
        peakInFlightCommandBuffers = max(
          peakInFlightCommandBuffers,
          pendingCommands.count
        )
        peakInFlightMappedSourceBytes = max(
          peakInFlightMappedSourceBytes,
          try inFlightMappedSourceBytes(pendingCommands)
        )
        try recordRetainedSourceBufferPeak()
        if pendingCommands.count >= maximumPendingCommandCount {
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
    let streamFinished = CFAbsoluteTimeGetCurrent()
    guard peakRetainedSourceBufferBytes <= plan.maximumRetainedSourceBufferBytes else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Indexed source transfer retained \(peakRetainedSourceBufferBytes) bytes, "
          + "exceeding its planned bound of "
          + "\(plan.maximumRetainedSourceBufferBytes) bytes."
      )
    }

    let sourceAudit: Metal4DSTEMExactSourceAudit
    let auditValues = values(
      from: countAudit, count: frameCount * 2, as: UInt32.self
    )
    var maximum: UInt32 = 0
    var above255: UInt64 = 0
    for frame in 0..<frameCount {
      maximum = max(maximum, auditValues[2 * frame])
      let sum = above255.addingReportingOverflow(
        UInt64(auditValues[2 * frame + 1])
      )
      guard !sum.overflow else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "The decoded above-255 audit overflows UInt64."
        )
      }
      above255 = sum.partialValue
    }
    sourceAudit = try Metal4DSTEMExactSourceAudit(
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
    let auditFinished = CFAbsoluteTimeGetCurrent()
    func exactMapValues(from buffer: MTLBuffer) -> [UInt64] {
      if usesFusedTiledLow8Products {
        return values(from: buffer, count: frameCount, as: UInt32.self).map(UInt64.init)
      }
      return values(from: buffer, count: frameCount, as: UInt64.self)
    }
    let products = Metal4DSTEMExactProducts(
      detectorSum: values(from: detectorSum, count: detectorPixels, as: UInt64.self),
      band1: exactMapValues(from: band1),
      band2: exactMapValues(from: band2),
      band4: exactMapValues(from: band4),
      total: exactMapValues(from: total),
      detectorRowMoment: exactMapValues(from: rowMoment),
      detectorColumnMoment: exactMapValues(from: columnMoment)
    )
    let productsFinished = CFAbsoluteTimeGetCurrent()
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
      stagingDtype: stagingDtype,
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
      maximumSourceCount: sourceAudit.maximumSourceCount,
      pixelsAbove255: sourceAudit.pixelsAbove255,
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
    let resultFinished = CFAbsoluteTimeGetCurrent()
    if ProcessInfo.processInfo.environment["QUANTEM_GPU_METAL_HOST_PROFILE"] == "1" {
      let hostProfile = String(
        format:
          "QGPU_HOST setup=%.9f stream=%.9f gpu_active=%.9f gpu_idle=%.9f map=%.9f open_stat=%.9f mmap=%.9f pread=%.9f metal_buffer=%.9f command_create=%.9f encode=%.9f commit=%.9f wait=%.9f audit=%.9f products=%.9f provenance=%.9f total=%.9f\n",
        setupFinished - started,
        streamFinished - streamStarted,
        gpuSeconds,
        profiledGPUIdleSeconds,
        mapSeconds,
        sourceOpenAndStatSeconds,
        sourceMmapSeconds,
        sourceReadSeconds,
        sourceMetalBufferSeconds,
        profiledCommandCreationSeconds,
        profiledEncodingSeconds,
        profiledCommitSeconds,
        profiledWaitSeconds,
        auditFinished - streamFinished,
        productsFinished - auditFinished,
        resultFinished - productsFinished,
        resultFinished - started
      )
      FileHandle.standardError.write(Data(hostProfile.utf8))
    }
    let metrics = Metal4DSTEMIndexedLoadMetrics(
      wallSeconds: resultFinished - started,
      sourceMappingSeconds: mapSeconds,
      sourceOpenAndStatSeconds: sourceOpenAndStatSeconds,
      sourceMmapSeconds: sourceMmapSeconds,
      sourceReadSeconds: sourceReadSeconds,
      sourceMetalBufferSeconds: sourceMetalBufferSeconds,
      sourceTransfer: plan.sourceTransfer,
      gpuSeconds: gpuSeconds,
      windowCount: plan.windows.count,
      sliceCount: sliceCount,
      commandBufferCount: commandBufferCount,
      peakInFlightCommandBuffers: peakInFlightCommandBuffers,
      peakInFlightMappedSourceBytes: peakInFlightMappedSourceBytes,
      peakRetainedSourceBufferBytes: peakRetainedSourceBufferBytes,
      mappedCompressedSourceBytes: mappedCompressedSourceBytes,
      maximumMappedCompressedSourceBytes: maximumMappedCompressedSourceBytes,
      maximumMappedSourceBufferBytes: maximumMappedSourceBufferBytes,
      maximumDecodedSliceBytes: maximumDecodedSliceBytes,
      cacheState: binned?.cacheState
        ?? "prepared_qh5_index_source_pages_unspecified",
      synchronizedStageSeconds: stageProfiler?.snapshot()
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
      expectedModificationNanoseconds: shard.index.metadata.sourceMtimeNs,
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

  private func supportsFusedAuditedLow8(
    plan: Metal4DSTEMIndexedLoadPlan,
    binned: BinnedOutputContext?,
    stagingDtype: Metal4DSTEMIntegerDType,
    detectorBin: Int
  ) throws -> Bool {
    guard stagingDtype == .uint8, let binned,
      detectorBin == 1 || detectorBin == 2,
      binned.plan.binningProvenance.detectorBin == detectorBin,
      binned.plan.binningProvenance.outputDtype == .uint16,
      binned.plan.sourceAudit.provesLosslessUInt8Staging,
      plan.sourceDetectorRows == 192,
      plan.sourceDetectorColumns == 192,
      binned.plan.binningProvenance.outputDetectorRows == 192 / detectorBin,
      binned.plan.binningProvenance.outputDetectorColumns == 192 / detectorBin,
      binned.plan.maximumDetectorPartialScratchBytes > 0
    else { return false }

    let fullRegion = try Metal4DSTEMScanRegion.full(
      sourceRows: plan.sourceScanRows,
      sourceColumns: plan.sourceScanColumns
    )
    let nativeProductPlan = try Metal4DSTEMLoadPlan(
      sourceScanRows: plan.sourceScanRows,
      sourceScanColumns: plan.sourceScanColumns,
      detectorRows: plan.sourceDetectorRows,
      detectorColumns: plan.sourceDetectorColumns,
      sourceBytesPerValue: plan.sourceDtype.bytesPerValue,
      scanRegion: fullRegion,
      scanBin: 1,
      detectorBin: 1
    )
    let bounds = try nativeProductPlan.exactAccumulatorBounds(
      maxSourceCount: binned.plan.sourceAudit.maximumSourceCount
    )
    let detectorSumBound = UInt64(binned.plan.sourceAudit.maximumSourceCount)
      .multipliedReportingOverflow(by: UInt64(plan.logicalFrameCount))
    guard bounds.fitsUInt32Accumulators,
      !detectorSumBound.overflow,
      detectorSumBound.partialValue <= UInt64(UInt32.max),
      Self.exactDetectorPartialFitsUInt16(
        maximumSourceCount: binned.plan.sourceAudit.maximumSourceCount
      )
    else { return false }

    for window in plan.windows {
      for slice in window.slices {
        let fitsOneShard = binned.plan.shardPlan.shards.contains { shard in
          let stop = shard.outputScanPositionStart + shard.outputScanPositionCount
          return slice.globalFrameRange.lowerBound >= shard.outputScanPositionStart
            && slice.globalFrameRange.upperBound <= stop
        }
        if !fitsOneShard { return false }
      }
    }
    return true
  }

  static func exactDetectorPartialFitsUInt16(
    maximumSourceCount: UInt32
  ) -> Bool {
    let bound = UInt64(maximumSourceCount).multipliedReportingOverflow(by: 32)
    return !bound.overflow && bound.partialValue <= UInt64(UInt16.max)
  }

  private func encodeSlice(
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
    detectorPartialScratch: MTLBuffer?,
    stagingDtype: Metal4DSTEMIntegerDType,
    usesFusedTiledLow8Bin1Products: Bool,
    usesFusedTiledLow8Bin2Products: Bool,
    binned: BinnedOutputContext?,
    command: MTLCommandBuffer,
    counterSamples: MTLCounterSampleBuffer?
  ) throws -> (metadata: MTLBuffer, binningDispatchCount: Int) {
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
    guard
      var encoder = try stageProfiler?.makeComputeEncoder(
        commandBuffer: command,
        sampleBuffer: counterSamples,
        stageIndex: 0
      ) ?? command.makeComputeCommandEncoder()
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "the device could not start an indexed decode encoder"
      )
    }
    let selectedDetectorSumPipeline =
      stagingDtype == .uint8
      ? auditedLow8TiledDetectorSumPipeline : detectorSumPipeline
    let binningDispatchCount: Int
    let selectedDecodePipeline =
      stagingDtype == .uint8
      ? (detectorPixels == 192 * 192
        && source.dataset.detectorRows == 192
        && source.dataset.detectorCols == 192
        ? auditedLow8Octet192DecodePipeline : auditedLow8DecodePipeline)
      : decodePipeline
    let usesDetector192OctetDecode =
      stagingDtype == .uint8
      && detectorPixels == 192 * 192
      && source.dataset.detectorRows == 192
      && source.dataset.detectorCols == 192
    let usesDetector192ProductPipeline =
      stagingDtype == .uint8
      && exactLow8ProductsFitUInt32(source: source)
      && detectorPixels == 192 * 192
      && source.dataset.detectorRows == 192
      && source.dataset.detectorCols == 192
    let selectedProductPipeline =
      stagingDtype == .uint8 && exactLow8ProductsFitUInt32(source: source)
      ? (usesDetector192ProductPipeline
        ? auditedLow8Detector192SIMDProductPipeline : auditedLow8SIMDProductPipeline)
      : (stagingDtype == .uint8 ? auditedLow8ProductPipeline : productPipeline)
    encoder.setComputePipelineState(selectedDecodePipeline)
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
      threadsPerThreadgroup: usesDetector192OctetDecode
        ? MTLSize(width: 32, height: 1, depth: 1)
        : MTLSize(width: 32, height: 4, depth: 1)
    )
    encoder = try nextProfiledEncoder(
      current: encoder,
      commandBuffer: command,
      sampleBuffer: counterSamples,
      stageIndex: 1
    )

    let usesFusedTiledLow8Products =
      usesFusedTiledLow8Bin1Products || usesFusedTiledLow8Bin2Products
    if usesFusedTiledLow8Products {
      guard let binned,
        let outputShard = binned.plan.shardPlan.shards.first(where: { candidate in
          let stop =
            candidate.outputScanPositionStart
            + candidate.outputScanPositionCount
          return slice.globalFrameRange.lowerBound
            >= candidate.outputScanPositionStart
            && slice.globalFrameRange.upperBound <= stop
        }),
        let sourceDetectorRows = UInt32(exactly: source.dataset.detectorRows),
        let destinationScanCount = UInt32(
          exactly: outputShard.outputScanPositionCount
        ),
        let destinationScanOffset = UInt32(
          exactly:
            slice.globalFrameRange.lowerBound
            - outputShard.outputScanPositionStart
        ),
        let outputDetectorRows = UInt32(
          exactly: binned.plan.binningProvenance.outputDetectorRows
        ),
        let outputDetectorColumns = UInt32(
          exactly: binned.plan.binningProvenance.outputDetectorColumns
        )
      else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "The tiled fused exact bin2 slice does not fit one validated destination shard."
        )
      }
      let destination = try binned.destination(for: outputShard.index)
      var fusedParameters = ContiguousBin2ProductsParameters(
        frameCount: frameCount,
        sourceDetectorRows: sourceDetectorRows,
        sourceDetectorColumns: detectorColumns,
        destinationScanCount: destinationScanCount,
        destinationScanOffset: destinationScanOffset,
        globalFrameOffset: globalFrameOffset,
        outputDetectorRows: outputDetectorRows,
        outputDetectorColumns: outputDetectorColumns
      )
      encoder.setComputePipelineState(
        usesFusedTiledLow8Bin1Products
          ? fusedTiledLow8Bin1ProductsDetectorPartialsPipeline
          : fusedTiledLow8Bin2ProductsDetectorPartialsPipeline
      )
      encoder.setBuffer(scratch, offset: 0, index: 0)
      encoder.setBuffer(destination, offset: 0, index: 1)
      withUnsafeBytes(of: &fusedParameters) {
        encoder.setBytes($0.baseAddress!, length: $0.count, index: 2)
      }
      encoder.setBuffer(detectorBands, offset: 0, index: 3)
      encoder.setBuffer(band1, offset: 0, index: 4)
      encoder.setBuffer(band2, offset: 0, index: 5)
      encoder.setBuffer(band4, offset: 0, index: 6)
      encoder.setBuffer(total, offset: 0, index: 7)
      encoder.setBuffer(rowMoment, offset: 0, index: 8)
      encoder.setBuffer(columnMoment, offset: 0, index: 9)
      guard let detectorPartialScratch else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "The fused exact detector-partial path requires bounded scratch storage."
        )
      }
      encoder.setBuffer(detectorPartialScratch, offset: 0, index: 10)
      encoder.dispatchThreadgroups(
        MTLSize(width: (Int(frameCount) + 31) / 32, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 32, height: 8, depth: 1)
      )
      binningDispatchCount = 1
    } else {
      encoder.setComputePipelineState(selectedProductPipeline)
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
      binningDispatchCount = 0
    }
    encoder = try nextProfiledEncoder(
      current: encoder,
      commandBuffer: command,
      sampleBuffer: counterSamples,
      stageIndex: 2
    )

    var mutableFrameCount = frameCount
    if usesFusedTiledLow8Products {
      guard let detectorPartialScratch else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "The fused exact detector-partial reduction lost its scratch storage."
        )
      }
      var partialCount = (frameCount + 31) / 32
      encoder.setComputePipelineState(detectorAccumulateU16PartialsPipeline)
      encoder.setBuffer(detectorPartialScratch, offset: 0, index: 0)
      encoder.setBuffer(detectorSum, offset: 0, index: 1)
      encoder.setBytes(&detectorPixelCount, length: 4, index: 2)
      encoder.setBytes(&partialCount, length: 4, index: 3)
      encoder.dispatchThreads(
        MTLSize(width: detectorPixels, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: 256, height: 1, depth: 1)
      )
    } else if stagingDtype == .uint8 {
      encoder.setComputePipelineState(selectedDetectorSumPipeline)
      encoder.setBuffer(scratch, offset: 0, index: 0)
      encoder.setBuffer(detectorSum, offset: 0, index: 1)
      encoder.setBytes(&detectorPixelCount, length: 4, index: 2)
      encoder.setBytes(&mutableFrameCount, length: 4, index: 3)
      encoder.dispatchThreadgroups(
        MTLSize(width: (detectorPixels + 31) / 32, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(
          width: 32,
          height: 8,
          depth: 1
        )
      )
    } else {
      encoder.setComputePipelineState(selectedDetectorSumPipeline)
      encoder.setBuffer(scratch, offset: 0, index: 0)
      encoder.setBuffer(detectorSum, offset: 0, index: 1)
      encoder.setBytes(&detectorPixelCount, length: 4, index: 2)
      encoder.setBytes(&mutableFrameCount, length: 4, index: 3)
      let threads = min(
        256, selectedDetectorSumPipeline.maxTotalThreadsPerThreadgroup
      )
      encoder.dispatchThreads(
        MTLSize(width: detectorPixels, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threads, height: 1, depth: 1)
      )
    }
    encoder.endEncoding()
    let totalBinningDispatchCount: Int
    if usesFusedTiledLow8Products {
      totalBinningDispatchCount = binningDispatchCount
    } else {
      totalBinningDispatchCount =
        try binned.map {
          try encodeBinnedSlice(
            context: $0,
            slice: slice,
            scratch: scratch,
            stagingDtype: stagingDtype,
            commandBuffer: command
          )
        } ?? 0
    }
    return (metadata, totalBinningDispatchCount)
  }

  private func exactLow8ProductsFitUInt32(
    source: Native4DSTEMIndexedSource
  ) -> Bool {
    guard let rows = UInt64(exactly: source.dataset.detectorRows),
      let columns = UInt64(exactly: source.dataset.detectorCols),
      rows > 0,
      columns > 0
    else { return false }

    func multiply(_ factors: [UInt64]) -> UInt64? {
      var value: UInt64 = 1
      for factor in factors {
        let next = value.multipliedReportingOverflow(by: factor)
        guard !next.overflow else { return nil }
        value = next.partialValue
      }
      return value
    }

    func triangular(_ count: UInt64) -> UInt64? {
      var first = count
      var second = count - 1
      if first.isMultiple(of: 2) {
        first /= 2
      } else {
        second /= 2
      }
      return multiply([first, second])
    }

    guard let rowIndices = triangular(rows),
      let columnIndices = triangular(columns),
      let total = multiply([rows, columns, UInt64(UInt8.max)]),
      let rowMoment = multiply([rowIndices, columns, UInt64(UInt8.max)]),
      let columnMoment = multiply([columnIndices, rows, UInt64(UInt8.max)])
    else { return false }
    let maximum = max(total, rowMoment, columnMoment)
    return maximum <= UInt64(UInt32.max)
  }

  private func encodeBinnedSlice(
    context: BinnedOutputContext,
    slice: Native4DSTEMIndexedSlice,
    scratch: MTLBuffer,
    stagingDtype: Metal4DSTEMIntegerDType,
    commandBuffer: MTLCommandBuffer? = nil,
    encoder: MTLComputeCommandEncoder? = nil
  ) throws -> Int {
    guard (commandBuffer == nil) != (encoder == nil) else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Exact binning requires exactly one command buffer or existing encoder."
      )
    }
    let plan = context.plan
    let sourceFrameElements = plan.productPlan.sourceDetectorRows
      .multipliedReportingOverflow(by: plan.productPlan.sourceDetectorColumns)
    guard !sourceFrameElements.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "One indexed source-frame element count overflows Int."
      )
    }
    let sourceFrameBytes = sourceFrameElements.partialValue
      .multipliedReportingOverflow(by: stagingDtype.bytesPerValue)
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
      let destination = try context.destination(for: shard.index)
      let destinationView = Metal4DSTEMExactBinningDestination.scanRowShard(
        plan: plan.shardPlan,
        index: shard.index
      )
      if stagingDtype == .uint8 {
        if let encoder {
          _ = try exactBinner.encodeContiguousAuditedUInt8Frames(
            encoder: encoder,
            stagedSource: scratch,
            stagedSourceOffset: sourceOffset.partialValue,
            destination: destination,
            destinationView: destinationView,
            plan: binningPlan,
            sourceFrameCount: globalStop - globalStart,
            globalScanPositionOffset: globalStart,
            sourceAudit: plan.sourceAudit
          )
        } else if let commandBuffer {
          _ = try exactBinner.encodeContiguousAuditedUInt8Frames(
            commandBuffer: commandBuffer,
            stagedSource: scratch,
            stagedSourceOffset: sourceOffset.partialValue,
            destination: destination,
            destinationView: destinationView,
            plan: binningPlan,
            sourceFrameCount: globalStop - globalStart,
            globalScanPositionOffset: globalStart,
            sourceAudit: plan.sourceAudit
          )
        }
      } else {
        if let encoder {
          _ = try exactBinner.encodeContiguousUInt16Frames(
            encoder: encoder,
            stagedSource: scratch,
            stagedSourceOffset: sourceOffset.partialValue,
            destination: destination,
            destinationView: destinationView,
            plan: binningPlan,
            sourceFrameCount: globalStop - globalStart,
            globalScanPositionOffset: globalStart,
            sourceAudit: plan.sourceAudit
          )
        } else if let commandBuffer {
          _ = try exactBinner.encodeContiguousUInt16Frames(
            commandBuffer: commandBuffer,
            stagedSource: scratch,
            stagedSourceOffset: sourceOffset.partialValue,
            destination: destination,
            destinationView: destinationView,
            plan: binningPlan,
            sourceFrameCount: globalStop - globalStart,
            globalScanPositionOffset: globalStart,
            sourceAudit: plan.sourceAudit
          )
        }
      }
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
    let waitStarted = CFAbsoluteTimeGetCurrent()
    try wait(pending.command)
    profiledWaitSeconds += CFAbsoluteTimeGetCurrent() - waitStarted
    let gpuSeconds = max(
      0, pending.command.gpuEndTime - pending.command.gpuStartTime
    )
    if ProcessInfo.processInfo.environment["QUANTEM_GPU_METAL_HOST_PROFILE"] == "1" {
      if let previousEnd = profiledPreviousGPUEndTime {
        profiledGPUIdleSeconds += max(
          0, pending.command.gpuStartTime - previousEnd
        )
      }
      profiledPreviousGPUEndTime = pending.command.gpuEndTime
    }
    try stageProfiler?.record(
      sampleBuffer: pending.counterSamples,
      commandGPUSeconds: gpuSeconds
    )
    return gpuSeconds
  }

  private func nextProfiledEncoder(
    current: MTLComputeCommandEncoder,
    commandBuffer: MTLCommandBuffer,
    sampleBuffer: MTLCounterSampleBuffer?,
    stageIndex: Int
  ) throws -> MTLComputeCommandEncoder {
    guard let stageProfiler, let sampleBuffer else {
      current.memoryBarrier(scope: .buffers)
      return current
    }
    current.endEncoding()
    return try stageProfiler.makeComputeEncoder(
      commandBuffer: commandBuffer,
      sampleBuffer: sampleBuffer,
      stageIndex: stageIndex
    )
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
      for mappedSource in pending.mappedSources {
        guard seen.insert(ObjectIdentifier(mappedSource)).inserted else {
          continue
        }
        let updated = bytes.addingReportingOverflow(UInt64(mappedSource.buffer.length))
        guard !updated.overflow else {
          throw Metal4DSTEMStreamingIOError.invalidRequest(
            "In-flight mapped-source byte accounting overflows UInt64."
          )
        }
        bytes = updated.partialValue
      }
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

  /// Create a command whose resources are retained by this loader's pending
  /// submission record instead of duplicated inside every Metal command.
  private func makeStreamingCommandBuffer() -> MTLCommandBuffer? {
    let descriptor = MTLCommandBufferDescriptor()
    descriptor.retainedReferences = false
    return queue.makeCommandBuffer(descriptor: descriptor)
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

  private func makeResidentDestinations(
    plan: Metal4DSTEMIndexedBinnedLoadPlan
  ) throws -> ResidentDestinationAllocation {
    let buffers = try plan.shardPlan.shards.map { shard in
      try makeBuffer(
        bytes: shard.payloadBytes,
        options: plan.residentStorage.resourceOptions,
        label: "exact working-volume shard \(shard.index)"
      )
    }
    guard plan.residentStorage == .privateGPU else {
      return ResidentDestinationAllocation(buffers: buffers, preparations: [])
    }
    let preparationQueues = try (0..<4).map { queueIndex in
      guard let queue = device.makeCommandQueue() else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Metal could not create exact private-residency preparation queue "
            + "\(queueIndex)."
        )
      }
      queue.label = "exact private-residency preparation queue \(queueIndex)"
      return queue
    }
    let preparations = try buffers.enumerated().map { index, buffer in
      let preparationQueue = preparationQueues[index % preparationQueues.count]
      guard let command = preparationQueue.makeCommandBuffer(),
        let encoder = command.makeComputeCommandEncoder()
      else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "Metal could not encode private-residency preparation for shard \(index)."
        )
      }
      command.label = "prepare exact private working-volume shard \(index)"
      encoder.setComputePipelineState(privateResidentPagePreparationPipeline)
      encoder.setBuffer(buffer, offset: 0, index: 0)
      var byteCount = UInt64(buffer.length)
      var pageStride = UInt64(getpagesize())
      encoder.setBytes(
        &byteCount,
        length: MemoryLayout<UInt64>.stride,
        index: 1
      )
      encoder.setBytes(
        &pageStride,
        length: MemoryLayout<UInt64>.stride,
        index: 2
      )
      let pageStrideInt = Int(pageStride)
      let pageCount = (buffer.length + pageStrideInt - 1) / pageStrideInt
      let threadWidth = min(
        privateResidentPagePreparationPipeline.maxTotalThreadsPerThreadgroup,
        max(1, privateResidentPagePreparationPipeline.threadExecutionWidth * 8)
      )
      encoder.dispatchThreads(
        MTLSize(width: pageCount, height: 1, depth: 1),
        threadsPerThreadgroup: MTLSize(width: threadWidth, height: 1, depth: 1)
      )
      encoder.endEncoding()
      command.commit()
      return command
    }
    return ResidentDestinationAllocation(
      buffers: buffers,
      preparations: preparations
    )
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

private struct ContiguousBin2ProductsParameters {
  var frameCount: UInt32
  var sourceDetectorRows: UInt32
  var sourceDetectorColumns: UInt32
  var destinationScanCount: UInt32
  var destinationScanOffset: UInt32
  var globalFrameOffset: UInt32
  var outputDetectorRows: UInt32
  var outputDetectorColumns: UInt32
}

/// Strong resource retention for one asynchronously submitted source slice.
private struct PendingIndexedCommand {
  let command: MTLCommandBuffer
  let mappedSources: [MappedMetalSource]
  let metadataBuffers: [MTLBuffer]
  let counterSamples: MTLCounterSampleBuffer?
  let binningDispatchCount: Int
}

private final class IndexedStageProfiler {
  private let device: MTLDevice
  private let timestampCounterSet: MTLCounterSet
  private var decodeAndAudit = 0.0
  private var exactScanProductsAndDetectorBinning = 0.0
  private var exactDetectorSum = 0.0
  private var commandQueueRemainder = 0.0
  private var commandBuffers = 0

  private init(device: MTLDevice, timestampCounterSet: MTLCounterSet) {
    self.device = device
    self.timestampCounterSet = timestampCounterSet
  }

  static func makeIfRequested(device: MTLDevice) throws -> IndexedStageProfiler? {
    guard
      ProcessInfo.processInfo.environment["QUANTEM_GPU_METAL_STAGE_PROFILE"] == "1"
    else { return nil }
    guard device.supportsCounterSampling(.atStageBoundary) else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "the selected device does not expose Metal stage-boundary counters"
      )
    }
    guard let counterSet = device.counterSets?.first(where: { $0.name == "timestamp" })
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "the selected device does not expose Metal timestamp counters"
      )
    }
    return IndexedStageProfiler(device: device, timestampCounterSet: counterSet)
  }

  func reset() {
    decodeAndAudit = 0
    exactScanProductsAndDetectorBinning = 0
    exactDetectorSum = 0
    commandQueueRemainder = 0
    commandBuffers = 0
  }

  func makeSampleBuffer() throws -> MTLCounterSampleBuffer {
    let descriptor = MTLCounterSampleBufferDescriptor()
    descriptor.counterSet = timestampCounterSet
    descriptor.storageMode = .shared
    descriptor.sampleCount = 6
    return try device.makeCounterSampleBuffer(descriptor: descriptor)
  }

  func makeComputeEncoder(
    commandBuffer: MTLCommandBuffer,
    sampleBuffer: MTLCounterSampleBuffer?,
    stageIndex: Int
  ) throws -> MTLComputeCommandEncoder {
    guard let sampleBuffer else {
      guard let encoder = commandBuffer.makeComputeCommandEncoder() else {
        throw Metal4DSTEMStreamingIOError.metalUnavailable(
          "the device could not create a compute encoder"
        )
      }
      return encoder
    }
    let descriptor = MTLComputePassDescriptor()
    guard let attachment = descriptor.sampleBufferAttachments[0] else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "the device could not create a counter attachment"
      )
    }
    attachment.sampleBuffer = sampleBuffer
    attachment.startOfEncoderSampleIndex = stageIndex * 2
    attachment.endOfEncoderSampleIndex = stageIndex * 2 + 1
    guard
      let encoder = commandBuffer.makeComputeCommandEncoder(
        descriptor: descriptor
      )
    else {
      throw Metal4DSTEMStreamingIOError.metalUnavailable(
        "the device could not create a profiled compute encoder"
      )
    }
    return encoder
  }

  func record(
    sampleBuffer: MTLCounterSampleBuffer?,
    commandGPUSeconds: Double
  ) throws {
    guard let sampleBuffer,
      let data = try sampleBuffer.resolveCounterRange(0..<6)
    else { return }
    let timestamps = data.withUnsafeBytes {
      Array($0.bindMemory(to: MTLCounterResultTimestamp.self)).map(\.timestamp)
    }
    guard timestamps.count == 6,
      timestamps[1] >= timestamps[0],
      timestamps[3] >= timestamps[2],
      timestamps[5] >= timestamps[4]
    else {
      throw Metal4DSTEMStreamingIOError.commandFailed(
        "Metal returned invalid synchronized stage timestamps."
      )
    }
    let secondsPerNanosecond = 1.0e-9
    let decode = Double(timestamps[1] - timestamps[0]) * secondsPerNanosecond
    let products = Double(timestamps[3] - timestamps[2]) * secondsPerNanosecond
    let detectorSum = Double(timestamps[5] - timestamps[4]) * secondsPerNanosecond
    decodeAndAudit += decode
    exactScanProductsAndDetectorBinning += products
    exactDetectorSum += detectorSum
    commandQueueRemainder += max(
      0, commandGPUSeconds - decode - products - detectorSum
    )
    commandBuffers += 1
  }

  func snapshot() -> Metal4DSTEMIndexedStageSeconds {
    Metal4DSTEMIndexedStageSeconds(
      decodeAndAudit: decodeAndAudit,
      exactScanProductsAndDetectorBinning:
        exactScanProductsAndDetectorBinning,
      exactDetectorSum: exactDetectorSum,
      commandQueueRemainder: commandQueueRemainder,
      commandBuffers: commandBuffers
    )
  }
}

private struct ResidentDestinationAllocation {
  let buffers: [MTLBuffer]
  let preparations: [MTLCommandBuffer]
}

private final class BinnedOutputContext {
  private enum Storage {
    case resident([MTLBuffer])
    case streaming((Metal4DSTEMExactBinningShard, MTLBuffer) throws -> Void)
  }

  let plan: Metal4DSTEMIndexedBinnedLoadPlan
  let cacheState: String
  private let storage: Storage
  private var residentPreparations: [MTLCommandBuffer?]
  private var activeShardIndex: Int?
  private var streamingDestination: MTLBuffer?
  private(set) var destinationAllocationSeconds = 0.0
  private(set) var payloadWriteSeconds = 0.0
  private(set) var completedShardCount = 0
  private(set) var peakWorkingMetalBytes: UInt64 = 0

  var retainsAllDestinations: Bool {
    if case .resident = storage { return true }
    return false
  }

  init(
    resident plan: Metal4DSTEMIndexedBinnedLoadPlan,
    destinations: [MTLBuffer],
    preparations: [MTLCommandBuffer]
  ) {
    self.plan = plan
    storage = .resident(destinations)
    residentPreparations = preparations.map(Optional.some)
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
    residentPreparations = []
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
    guard activeShardIndex == nil else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The previous exact output shard must complete before reusing its Metal buffer."
      )
    }
    guard shard.index == completedShardCount else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Exact output shards must be streamed once in canonical scan-row order; "
          + "expected shard \(completedShardCount), received \(shard.index)."
      )
    }
    let destination: MTLBuffer
    if let reusable = streamingDestination {
      destination = reusable
    } else {
      let started = CFAbsoluteTimeGetCurrent()
      destination = try autoreleasepool {
        try makeBuffer(shard)
      }
      destinationAllocationSeconds += CFAbsoluteTimeGetCurrent() - started
      streamingDestination = destination
    }
    guard UInt64(destination.length) >= shard.payloadBytes else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Exact output shard \(shard.index) allocated \(destination.length) bytes; "
          + "at least \(shard.payloadBytes) are required."
      )
    }
    activeShardIndex = shard.index
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
      if residentPreparations.indices.contains(shardIndex),
        let preparation = residentPreparations[shardIndex]
      {
        preparation.waitUntilCompleted()
        guard preparation.status == .completed else {
          throw Metal4DSTEMStreamingIOError.metalUnavailable(
            preparation.error?.localizedDescription
              ?? "Private-residency preparation failed for shard \(shardIndex)."
          )
        }
        residentPreparations[shardIndex] = nil
      }
      return destinations[shardIndex]
    case .streaming:
      guard activeShardIndex == shardIndex, let streamingDestination else {
        throw Metal4DSTEMStreamingIOError.invalidRequest(
          "Exact output shard \(shardIndex) was not prepared before encoding."
        )
      }
      return streamingDestination
    }
  }

  func finishActiveDestination() throws {
    guard case .streaming(let consume) = storage,
      let activeShardIndex,
      let streamingDestination
    else { return }
    defer {
      self.activeShardIndex = nil
    }
    let shard = plan.shardPlan.shards[activeShardIndex]
    let started = CFAbsoluteTimeGetCurrent()
    try consume(shard, streamingDestination)
    payloadWriteSeconds += CFAbsoluteTimeGetCurrent() - started
    completedShardCount += 1
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

private final class BufferedMetalSourceFuture: @unchecked Sendable {
  private let group = DispatchGroup()
  private let lock = NSLock()
  private var result: Result<MappedMetalSource, Error>?
  let expectedBufferBytes: UInt64

  init(
    url: URL,
    expectedBytes: UInt64,
    expectedModificationNanoseconds: UInt64,
    device: MTLDevice
  ) throws {
    expectedBufferBytes = try Metal4DSTEMIndexedLoadPlan.mappedBufferBytes(
      expectedBytes
    )
    group.enter()
    DispatchQueue.global(qos: .userInitiated).async { [self] in
      let completed = Result {
        try MappedMetalSource(
          url: url,
          expectedBytes: expectedBytes,
          expectedModificationNanoseconds: expectedModificationNanoseconds,
          device: device,
          bufferedRead: true
        )
      }
      lock.lock()
      result = completed
      lock.unlock()
      group.leave()
    }
  }

  func value() throws -> MappedMetalSource {
    group.wait()
    lock.lock()
    defer { lock.unlock() }
    guard let result else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The bounded source prefetch completed without a result."
      )
    }
    return try result.get()
  }
}

private final class MappedMetalSource {
  let buffer: MTLBuffer
  let openAndStatSeconds: Double
  let mmapSeconds: Double
  let readSeconds: Double
  let metalBufferSeconds: Double

  init(
    url: URL,
    expectedBytes: UInt64,
    expectedModificationNanoseconds: UInt64,
    device: MTLDevice,
    bufferedRead: Bool = false
  ) throws {
    let openAndStatStarted = CFAbsoluteTimeGetCurrent()
    let descriptor = open(url.path, O_RDONLY)
    guard descriptor >= 0 else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Could not open indexed source \(url.lastPathComponent)."
      )
    }
    var status = stat()
    let statResult = fstat(descriptor, &status)
    let seconds = UInt64(exactly: status.st_mtimespec.tv_sec)
    let nanoseconds = UInt64(exactly: status.st_mtimespec.tv_nsec)
    let modification = seconds?.multipliedReportingOverflow(by: 1_000_000_000)
    let modificationNanoseconds = modification?.partialValue.addingReportingOverflow(
      nanoseconds ?? 0
    )
    guard statResult == 0,
      let fileBytes = UInt64(exactly: status.st_size),
      fileBytes == expectedBytes,
      let nanoseconds,
      nanoseconds < 1_000_000_000,
      let modification,
      !modification.overflow,
      let modificationNanoseconds,
      !modificationNanoseconds.overflow,
      modificationNanoseconds.partialValue == expectedModificationNanoseconds,
      let fileLength = Int(exactly: fileBytes),
      fileLength > 0
    else {
      close(descriptor)
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Indexed source \(url.lastPathComponent) changed size or modification time after validation."
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
    let openAndStatFinished = CFAbsoluteTimeGetCurrent()
    if bufferedRead {
      let metalBufferStarted = CFAbsoluteTimeGetCurrent()
      guard
        let buffer = device.makeBuffer(
          length: mappedBytes,
          options: .storageModeShared
        )
      else {
        close(descriptor)
        throw Metal4DSTEMStreamingIOError.allocationFailed(
          label: "buffered source \(url.lastPathComponent)",
          bytes: UInt64(mappedBytes)
        )
      }
      let allocationSeconds = CFAbsoluteTimeGetCurrent() - metalBufferStarted
      let readStarted = CFAbsoluteTimeGetCurrent()
      var offset = 0
      while offset < fileLength {
        let count = pread(
          descriptor,
          buffer.contents().advanced(by: offset),
          fileLength - offset,
          off_t(offset)
        )
        if count < 0 && errno == EINTR { continue }
        guard count > 0 else {
          close(descriptor)
          throw Metal4DSTEMStreamingIOError.invalidRequest(
            "Could not read indexed source \(url.lastPathComponent) into bounded Metal staging."
          )
        }
        offset += count
      }
      close(descriptor)
      self.buffer = buffer
      openAndStatSeconds = openAndStatFinished - openAndStatStarted
      mmapSeconds = 0
      readSeconds = CFAbsoluteTimeGetCurrent() - readStarted
      metalBufferSeconds = allocationSeconds
      return
    }
    let mmapStarted = CFAbsoluteTimeGetCurrent()
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
    let mmapFinished = CFAbsoluteTimeGetCurrent()
    let metalBufferStarted = CFAbsoluteTimeGetCurrent()
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
    openAndStatSeconds = openAndStatFinished - openAndStatStarted
    mmapSeconds = mmapFinished - mmapStarted
    readSeconds = 0
    metalBufferSeconds = CFAbsoluteTimeGetCurrent() - metalBufferStarted
  }
}
