import Foundation
import Metal
import Metal4DSTEMKernels
import Native4DSTEMIO

/// Policy-free plan for exact indexed products plus a sharded working volume.
///
/// QuantEM.GPU reports package-owned allocation sizes and validates scientific
/// geometry, dtype, source identity, and exact-sum bounds. It does not decide
/// whether a device should admit the operation.
public struct Metal4DSTEMIndexedBinnedLoadPlan: Equatable, Sendable {
  public let productPlan: Metal4DSTEMIndexedLoadPlan
  public let sourceAudit: Metal4DSTEMExactSourceAudit
  public let binningProvenance: Metal4DSTEMExactBinningProvenance
  public let shardPlan: Metal4DSTEMExactBinningShardPlan
  public let samplingPropagation: Metal4DSTEMSamplingPropagation
  public let workingPayloadBytes: UInt64
  public let packagePersistentOutputBytes: UInt64
  /// Maximum private scratch used by the exact fused detector-sum reduction.
  public let maximumDetectorPartialScratchBytes: UInt64
  public let estimatedAllocatedMetalBytesExcludingMappedSource: UInt64
  public let maximumRetainedSourceBufferBytes: UInt64
  public let estimatedAllocatedMetalBytesIncludingSourceTransfer: UInt64
  public let maximumIndividualMetalBufferBytes: UInt64
  public let destinationStorageMode: String

  internal var binningPlan: Metal4DSTEMLoadPlan {
    get throws {
      try Metal4DSTEMLoadPlan(
        sourceScanRows: binningProvenance.sourceScanRows,
        sourceScanColumns: binningProvenance.sourceScanColumns,
        detectorRows: binningProvenance.sourceDetectorRows,
        detectorColumns: binningProvenance.sourceDetectorColumns,
        sourceBytesPerValue: binningProvenance.sourceDtype.bytesPerValue,
        scanRegion: binningProvenance.scanRegion,
        scanBin: binningProvenance.scanBin,
        detectorBin: binningProvenance.detectorBin
      )
    }
  }

  public init(
    source: Native4DSTEMIndexedSource,
    maximumDecodedWindowBytes: UInt64,
    detectorBands: Metal4DSTEMDetectorBands,
    detectorBin: Int,
    sourceAudit: Metal4DSTEMExactSourceAudit,
    maximumShardBytes: UInt64,
    sourceTransfer: Metal4DSTEMIndexedSourceTransfer = .memoryMapped
  ) throws {
    try sourceAudit.validate()
    let productPlan = try Metal4DSTEMIndexedLoadPlan(
      source: source,
      maximumDecodedWindowBytes: maximumDecodedWindowBytes,
      detectorBands: detectorBands,
      sourceTransfer: sourceTransfer
    )
    guard sourceAudit.sourceIdentitySHA256 == productPlan.sourceIdentitySHA256,
      sourceAudit.sourceDtype == .uint16,
      sourceAudit.badPixelIndices == productPlan.badPixelIndices
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The exact source audit does not match the indexed source identity, uint16 "
          + "dtype, or bad-pixel policy. Rebuild the audit from this source."
      )
    }
    let fullRegion = try Metal4DSTEMScanRegion.full(
      sourceRows: productPlan.sourceScanRows,
      sourceColumns: productPlan.sourceScanColumns
    )
    let binningPlan = try Metal4DSTEMLoadPlan(
      sourceScanRows: productPlan.sourceScanRows,
      sourceScanColumns: productPlan.sourceScanColumns,
      detectorRows: productPlan.sourceDetectorRows,
      detectorColumns: productPlan.sourceDetectorColumns,
      sourceBytesPerValue: Metal4DSTEMIntegerDType.uint16.bytesPerValue,
      scanRegion: fullRegion,
      scanBin: 1,
      detectorBin: detectorBin
    )
    let stagingDtype: Metal4DSTEMIntegerDType =
      sourceAudit.provesLosslessUInt8Staging ? .uint8 : .uint16
    let provenance = try Metal4DSTEMExactBinner.provenance(
      plan: binningPlan,
      sourceAudit: sourceAudit,
      stagingDtype: stagingDtype,
      outputDtype: .uint16
    )
    let shards = try Metal4DSTEMExactBinningShardPlan(
      provenance: provenance,
      maximumShardBytes: maximumShardBytes
    )
    let sourceScanSampling = try Self.sampling(
      row: productPlan.scanSamplingRowNanometer,
      column: productPlan.scanSamplingColumnNanometer,
      unit: "nm",
      provenance: "native_dataset_scan_sampling",
      evidence: productPlan.sourceIdentitySHA256,
      label: "scan"
    )
    let sourceDetectorSampling = try Self.sampling(
      row: productPlan.detectorSamplingRow,
      column: productPlan.detectorSamplingColumn,
      unit: productPlan.detectorSamplingUnit,
      provenance: "native_dataset_detector_sampling",
      evidence: productPlan.sourceIdentitySHA256,
      label: "detector"
    )
    let propagation = try provenance.propagatingSampling(
      sourceScan: sourceScanSampling,
      sourceDetector: sourceDetectorSampling
    )
    let persistent = try Self.sum(
      productPlan.persistentOutputBytes,
      shards.totalPayloadBytes,
      label: "persistent exact outputs"
    )
    guard
      productPlan.estimatedAllocatedMetalBytesExcludingMappedSource
        >= productPlan.maximumActualWindowBytes
    else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The indexed product plan reports inconsistent transient storage."
      )
    }
    let stagingWindowBytes =
      productPlan.maximumActualWindowBytes
      / UInt64(productPlan.stagingDtype.bytesPerValue)
      * UInt64(stagingDtype.bytesPerValue)
    let productStorageWithSelectedStaging =
      productPlan.estimatedAllocatedMetalBytesExcludingMappedSource
      - productPlan.maximumActualWindowBytes
      + stagingWindowBytes
    let maximumSliceFrames =
      productPlan.windows.flatMap(\.slices).map {
        UInt64($0.globalFrameRange.count)
      }.max() ?? 0
    let detectorPartialScratchBytes: UInt64
    if stagingDtype == .uint8,
      detectorBin == 2,
      productPlan.sourceDetectorRows == 192,
      productPlan.sourceDetectorColumns == 192
    {
      detectorPartialScratchBytes = try Self.exactDetectorPartialScratchBytes(
        maximumSliceFrames: maximumSliceFrames,
        detectorRows: UInt64(productPlan.sourceDetectorRows),
        detectorColumns: UInt64(productPlan.sourceDetectorColumns)
      )
    } else {
      detectorPartialScratchBytes = 0
    }
    let allocated = try Self.sum(
      try Self.sum(
        productStorageWithSelectedStaging,
        shards.totalPayloadBytes,
        label: "package Metal allocation"
      ),
      detectorPartialScratchBytes,
      label: "package Metal allocation"
    )

    self.productPlan = productPlan
    self.sourceAudit = sourceAudit
    binningProvenance = provenance
    shardPlan = shards
    samplingPropagation = propagation
    workingPayloadBytes = shards.totalPayloadBytes
    packagePersistentOutputBytes = persistent
    maximumDetectorPartialScratchBytes = detectorPartialScratchBytes
    estimatedAllocatedMetalBytesExcludingMappedSource = allocated
    maximumRetainedSourceBufferBytes = productPlan.maximumRetainedSourceBufferBytes
    switch sourceTransfer {
    case .memoryMapped:
      estimatedAllocatedMetalBytesIncludingSourceTransfer = allocated
    case .bufferedReadAhead:
      estimatedAllocatedMetalBytesIncludingSourceTransfer = try Self.sum(
        allocated,
        productPlan.maximumRetainedSourceBufferBytes,
        label: "package Metal allocation including source transfer"
      )
    }
    maximumIndividualMetalBufferBytes = max(
      productPlan.maximumIndividualMetalBufferBytes,
      shards.maximumActualShardBytes,
      detectorPartialScratchBytes
    )
    destinationStorageMode = "shared"
  }

  private static func sampling(
    row: Double?,
    column: Double?,
    unit: String?,
    provenance: String,
    evidence: String,
    label: String
  ) throws -> Metal4DSTEMAxisSampling? {
    if row == nil, column == nil { return nil }
    guard let row, let column, let unit else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "The indexed \(label) sampling is incomplete. Provide row, column, and unit "
          + "together or leave all three unavailable."
      )
    }
    return try Metal4DSTEMAxisSampling(
      row: row,
      column: column,
      unit: unit,
      provenance: provenance,
      evidence: evidence
    )
  }

  private static func sum(
    _ lhs: UInt64,
    _ rhs: UInt64,
    label: String
  ) throws -> UInt64 {
    let result = lhs.addingReportingOverflow(rhs)
    guard !result.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Indexed \(label) overflows UInt64."
      )
    }
    return result.partialValue
  }

  private static func product(
    _ lhs: UInt64,
    _ rhs: UInt64,
    label: String
  ) throws -> UInt64 {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Indexed \(label) overflows UInt64."
      )
    }
    return result.partialValue
  }

  static func exactDetectorPartialScratchBytes(
    maximumSliceFrames: UInt64,
    detectorRows: UInt64,
    detectorColumns: UInt64
  ) throws -> UInt64 {
    guard maximumSliceFrames > 0, detectorRows > 0, detectorColumns > 0 else {
      return 0
    }
    let groupNumerator = maximumSliceFrames.addingReportingOverflow(31)
    guard !groupNumerator.overflow else {
      throw Metal4DSTEMStreamingIOError.invalidRequest(
        "Indexed detector partial group count overflows UInt64."
      )
    }
    let groups = groupNumerator.partialValue / 32
    return try product(
      try product(
        try product(groups, detectorRows, label: "detector partial rows"),
        detectorColumns,
        label: "detector partial elements"
      ),
      UInt64(MemoryLayout<UInt16>.stride),
      label: "detector partial scratch"
    )
  }
}

/// Metrics for one synchronized exact binned indexed load.
public struct Metal4DSTEMIndexedBinnedLoadMetrics: Codable, Equatable, Sendable {
  public let indexedLoad: Metal4DSTEMIndexedLoadMetrics
  public let destinationAllocationSeconds: Double
  public let totalWallSeconds: Double
  public let binningDispatchCount: Int
  public let workingPayloadBytes: UInt64
  public let shardCount: Int
  public let maximumShardBytes: UInt64
  public let destinationStorageMode: String
}

/// Exact products and physical shards for one complete logical working volume.
///
/// `workingVolumeShards` use the scan-row intervals and packed detector-word-
/// major uint16 layout declared by `shardPlan`. Their aggregate logical shape
/// and dtype come from `binningProvenance`; physical sharding does not change
/// scan coverage or detector resolution.
public struct Metal4DSTEMIndexedBinnedLoadResult {
  public let workingVolumeShards: [MTLBuffer]
  public let shardPlan: Metal4DSTEMExactBinningShardPlan
  public let products: Metal4DSTEMExactProducts
  public let sourceAudit: Metal4DSTEMExactSourceAudit
  public let nativeProductProvenance: Metal4DSTEMIndexedLoadProvenance
  public let binningProvenance: Metal4DSTEMExactBinningProvenance
  public let samplingPropagation: Metal4DSTEMSamplingPropagation
  public let metrics: Metal4DSTEMIndexedBinnedLoadMetrics
}

/// Metrics for one transactional exact-binned cache build.
public struct Metal4DSTEMIndexedBinnedCacheMetrics: Codable, Equatable, Sendable {
  public let indexedLoad: Metal4DSTEMIndexedLoadMetrics
  public let destinationAllocationSeconds: Double
  public let payloadWriteSeconds: Double
  public let payloadFinalizeSeconds: Double
  public let totalWallSeconds: Double
  public let binningDispatchCount: Int
  public let workingPayloadBytes: UInt64
  public let shardCount: Int
  public let maximumShardBytes: UInt64
  public let peakWorkingMetalBytes: UInt64
  public let destinationStorage: String
}

/// Exact products and a sealed file-backed logical working volume.
///
/// The cache payload follows `metadata`'s detector-word-major packed-uint16
/// contract. The caller chooses the destination and owns admission, naming,
/// eviction, and lifecycle. QuantEM.GPU writes only a unique temporary payload
/// until source audit, coverage, byte count, storage synchronization, and
/// SHA-256 sealing all succeed.
public struct Metal4DSTEMIndexedBinnedCacheResult {
  public let payloadURL: URL
  public let metadataURL: URL
  public let metadata: Metal4DSTEMResidentCacheMetadata
  public let products: Metal4DSTEMExactProducts
  public let sourceAudit: Metal4DSTEMExactSourceAudit
  public let nativeProductProvenance: Metal4DSTEMIndexedLoadProvenance
  public let binningProvenance: Metal4DSTEMExactBinningProvenance
  public let samplingPropagation: Metal4DSTEMSamplingPropagation
  public let metrics: Metal4DSTEMIndexedBinnedCacheMetrics
}
