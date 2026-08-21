import CryptoKit
import Foundation
import Metal

/// Exact integer storage used at a native 4D-STEM load boundary.
public enum Metal4DSTEMIntegerDType: String, Codable, CaseIterable, Sendable {
  case uint8
  case uint16
  case uint32

  public var bytesPerValue: Int {
    switch self {
    case .uint8: MemoryLayout<UInt8>.stride
    case .uint16: MemoryLayout<UInt16>.stride
    case .uint32: MemoryLayout<UInt32>.stride
    }
  }
}

/// Row/column sampling with an opaque scientific unit.
public struct Metal4DSTEMAxisSampling: Codable, Equatable, Sendable {
  public let row: Double
  public let column: Double
  public let unit: String
  public let provenance: String
  public let evidence: String

  public init(
    row: Double,
    column: Double,
    unit: String,
    provenance: String,
    evidence: String
  ) throws {
    guard row.isFinite, row > 0, column.isFinite, column > 0,
      !unit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      !provenance.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty,
      !evidence.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    else { throw Metal4DSTEMExactBinnerError.invalidCalibration }
    self.row = row
    self.column = column
    self.unit = unit
    self.provenance = provenance
    self.evidence = evidence
  }

  fileprivate func scaled(by factor: Int) throws -> Self {
    try Self(
      row: row * Double(factor),
      column: column * Double(factor),
      unit: unit,
      provenance: provenance,
      evidence: evidence
    )
  }
}

public enum Metal4DSTEMSamplingPropagationState: String, Codable, Equatable, Sendable {
  case unavailable
  case unchanged
  case uniformlyScaled = "uniformly_scaled"
  case nonuniformEdgeBins = "nonuniform_edge_bins"
}

/// Source and working row/column sampling attached to an exact load selection.
///
/// This contract intentionally does not claim to transform detector centers,
/// affine calibrations, masks, or radii. Those require their own typed
/// coordinate transforms at the consumer boundary.
public struct Metal4DSTEMSamplingPropagation: Codable, Equatable, Sendable {
  public let sourceScan: Metal4DSTEMAxisSampling?
  public let workingScan: Metal4DSTEMAxisSampling?
  public let sourceDetector: Metal4DSTEMAxisSampling?
  public let workingDetector: Metal4DSTEMAxisSampling?
  public let scanState: Metal4DSTEMSamplingPropagationState
  public let detectorState: Metal4DSTEMSamplingPropagationState
  public let scanRegionRowStartInSourcePixels: Int
  public let scanRegionColumnStartInSourcePixels: Int
  public let firstWorkingScanCenterRowInSourcePixels: Double
  public let firstWorkingScanCenterColumnInSourcePixels: Double
  public let firstWorkingDetectorCenterRowInSourcePixels: Double
  public let firstWorkingDetectorCenterColumnInSourcePixels: Double
}

/// Source-identity-bound evidence used for exact dtype decisions.
public struct Metal4DSTEMExactSourceAudit: Codable, Equatable, Sendable {
  public static let currentSchema = "quantem.gpu.value-range-audit/v1"

  public let schema: String
  public let sourceIdentitySHA256: String
  public let auditSHA256: String
  public let sourceDtype: Metal4DSTEMIntegerDType
  public let badPixelIndices: [Int]
  public let maximumSourceCount: UInt32
  public let pixelsAbove255: UInt64

  public init(
    sourceIdentitySHA256: String,
    auditSHA256: String,
    sourceDtype: Metal4DSTEMIntegerDType,
    badPixelIndices: [Int],
    maximumSourceCount: UInt32,
    pixelsAbove255: UInt64
  ) throws {
    guard Self.isSHA256(sourceIdentitySHA256), Self.isSHA256(auditSHA256) else {
      throw Metal4DSTEMExactBinnerError.invalidAuditIdentity
    }
    guard sourceDtype == .uint8 || sourceDtype == .uint16 else {
      throw Metal4DSTEMExactBinnerError.unsupportedSourceDtype(sourceDtype)
    }
    let sortedBadPixels = badPixelIndices.sorted()
    guard sortedBadPixels.allSatisfy({ $0 >= 0 }),
      Set(sortedBadPixels).count == sortedBadPixels.count,
      maximumSourceCount <= UInt32(UInt16.max),
      (maximumSourceCount > UInt32(UInt8.max)) == (pixelsAbove255 > 0),
      sourceDtype != .uint8
        || (maximumSourceCount <= UInt32(UInt8.max) && pixelsAbove255 == 0)
    else { throw Metal4DSTEMExactBinnerError.invalidAuditValues }
    let canonicalDigest = try Self.canonicalSHA256(
      schema: Self.currentSchema,
      sourceIdentitySHA256: sourceIdentitySHA256,
      sourceDtype: sourceDtype,
      badPixelIndices: sortedBadPixels,
      maximumSourceCount: maximumSourceCount,
      pixelsAbove255: pixelsAbove255
    )
    guard canonicalDigest == auditSHA256 else {
      throw Metal4DSTEMExactBinnerError.invalidAuditIdentity
    }
    schema = Self.currentSchema
    self.sourceIdentitySHA256 = sourceIdentitySHA256
    self.auditSHA256 = auditSHA256
    self.sourceDtype = sourceDtype
    self.badPixelIndices = sortedBadPixels
    self.maximumSourceCount = maximumSourceCount
    self.pixelsAbove255 = pixelsAbove255
  }

  public init(
    sourceIdentitySHA256: String,
    sourceDtype: Metal4DSTEMIntegerDType,
    badPixelIndices: [Int],
    maximumSourceCount: UInt32,
    pixelsAbove255: UInt64
  ) throws {
    let sortedBadPixels = badPixelIndices.sorted()
    let digest = try Self.canonicalSHA256(
      schema: Self.currentSchema,
      sourceIdentitySHA256: sourceIdentitySHA256,
      sourceDtype: sourceDtype,
      badPixelIndices: sortedBadPixels,
      maximumSourceCount: maximumSourceCount,
      pixelsAbove255: pixelsAbove255
    )
    try self.init(
      sourceIdentitySHA256: sourceIdentitySHA256,
      auditSHA256: digest,
      sourceDtype: sourceDtype,
      badPixelIndices: sortedBadPixels,
      maximumSourceCount: maximumSourceCount,
      pixelsAbove255: pixelsAbove255
    )
  }

  public var provesLosslessUInt8Staging: Bool {
    pixelsAbove255 == 0 && maximumSourceCount <= UInt32(UInt8.max)
  }

  /// Revalidate a decoded audit before it authorizes a dtype decision.
  public func validate() throws {
    guard Self.isSHA256(sourceIdentitySHA256), Self.isSHA256(auditSHA256) else {
      throw Metal4DSTEMExactBinnerError.invalidAuditIdentity
    }
    guard schema == Self.currentSchema,
      sourceDtype == .uint8 || sourceDtype == .uint16,
      badPixelIndices == badPixelIndices.sorted(),
      badPixelIndices.allSatisfy({ $0 >= 0 }),
      Set(badPixelIndices).count == badPixelIndices.count,
      maximumSourceCount <= UInt32(UInt16.max),
      (maximumSourceCount > UInt32(UInt8.max)) == (pixelsAbove255 > 0),
      sourceDtype != .uint8
        || (maximumSourceCount <= UInt32(UInt8.max) && pixelsAbove255 == 0)
    else { throw Metal4DSTEMExactBinnerError.invalidAuditValues }
    let canonicalDigest = try Self.canonicalSHA256(
      schema: schema,
      sourceIdentitySHA256: sourceIdentitySHA256,
      sourceDtype: sourceDtype,
      badPixelIndices: badPixelIndices,
      maximumSourceCount: maximumSourceCount,
      pixelsAbove255: pixelsAbove255
    )
    guard canonicalDigest == auditSHA256 else {
      throw Metal4DSTEMExactBinnerError.invalidAuditIdentity
    }
  }

  private struct CanonicalPayload: Codable {
    let schema: String
    let sourceIdentitySHA256: String
    let sourceDtype: Metal4DSTEMIntegerDType
    let badPixelIndices: [Int]
    let maximum: UInt32
    let pixelsAbove255: UInt64
  }

  private static func canonicalSHA256(
    schema: String,
    sourceIdentitySHA256: String,
    sourceDtype: Metal4DSTEMIntegerDType,
    badPixelIndices: [Int],
    maximumSourceCount: UInt32,
    pixelsAbove255: UInt64
  ) throws -> String {
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
    let data = try encoder.encode(
      CanonicalPayload(
        schema: schema,
        sourceIdentitySHA256: sourceIdentitySHA256,
        sourceDtype: sourceDtype,
        badPixelIndices: badPixelIndices,
        maximum: maximumSourceCount,
        pixelsAbove255: pixelsAbove255
      )
    )
    return SHA256.hash(data: data)
      .map { String(format: "%02x", $0) }
      .joined()
  }

  private static func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64
      && value.utf8.allSatisfy {
        (48...57).contains($0) || (97...102).contains($0)
      }
  }
}

public enum Metal4DSTEMReductionSemantics: String, Codable, Equatable, Sendable {
  case exactIntegerSum = "exact_integer_sum"
}

public enum Metal4DSTEMStagingLayout: String, Codable, Equatable, Sendable {
  /// Frame-major selected scan columns; audited bad pixels are already zeroed.
  case frameMajorSelectedColumns = "frame_major_selected_columns"
}

public enum Metal4DSTEMOutputLayout: String, Codable, Equatable, Sendable {
  /// Detector-word-major uint32 words containing low/high uint16 detector lanes.
  case detectorWordMajorPackedUInt16 = "detector_word_major_packed_uint16"

  /// Detector-word-major uint32 values, one value per detector pixel.
  case detectorWordMajorUInt32 = "detector_word_major_uint32"
}

public enum Metal4DSTEMBadPixelApplication: String, Codable, Equatable, Sendable {
  case alreadyZeroedUsingAudit = "already_zeroed_using_audit"
}

/// Stable, UI-free provenance for one exact load/bin selection.
public struct Metal4DSTEMExactBinningProvenance: Codable, Equatable, Sendable {
  public static let currentSchema = "quantem.gpu.metal-4dstem-exact-binning/v1"

  public let schema: String
  public let sourceScanRows: Int
  public let sourceScanColumns: Int
  public let scanRegion: Metal4DSTEMScanRegion
  public let sourceDetectorRows: Int
  public let sourceDetectorColumns: Int
  public let outputScanRows: Int
  public let outputScanColumns: Int
  public let outputDetectorRows: Int
  public let outputDetectorColumns: Int
  public let scanBin: Int
  public let detectorBin: Int
  public let sourceIdentitySHA256: String
  public let valueRangeAuditSHA256: String
  public let sourceDtype: Metal4DSTEMIntegerDType
  public let stagingDtype: Metal4DSTEMIntegerDType
  public let outputDtype: Metal4DSTEMIntegerDType
  public let stagingLayout: Metal4DSTEMStagingLayout
  public let outputLayout: Metal4DSTEMOutputLayout
  public let badPixelApplication: Metal4DSTEMBadPixelApplication
  public let badPixelIndices: [Int]
  public let reduction: Metal4DSTEMReductionSemantics
  public let maximumSourceCount: UInt32
  public let pixelsAbove255: UInt64
  public let maximumOutputCount: UInt64
  public let outputPayloadBytes: UInt64

  /// Revalidate decoded provenance against its identity-bound source audit.
  public func validate(sourceAudit: Metal4DSTEMExactSourceAudit) throws {
    let plan = try Metal4DSTEMLoadPlan(
      sourceScanRows: sourceScanRows,
      sourceScanColumns: sourceScanColumns,
      detectorRows: sourceDetectorRows,
      detectorColumns: sourceDetectorColumns,
      sourceBytesPerValue: sourceDtype.bytesPerValue,
      scanRegion: scanRegion,
      scanBin: scanBin,
      detectorBin: detectorBin
    )
    let expected = try Metal4DSTEMExactBinner.provenance(
      plan: plan,
      sourceAudit: sourceAudit,
      stagingDtype: stagingDtype,
      outputDtype: outputDtype
    )
    guard self == expected else {
      throw Metal4DSTEMExactBinnerError.invalidProvenance
    }
  }

  /// Propagate row/column sampling without changing scientific coverage.
  ///
  /// Scan sampling scales by `scanBin`; detector sampling scales by
  /// `detectorBin`. The half-open scan origin remains explicit in source-pixel
  /// coordinates so a crop cannot be mistaken for a native full scan.
  public func propagatingSampling(
    sourceScan: Metal4DSTEMAxisSampling?,
    sourceDetector: Metal4DSTEMAxisSampling?
  ) throws -> Metal4DSTEMSamplingPropagation {
    let scanHasUniformBins =
      scanRegion.rows % scanBin == 0
      && scanRegion.columns % scanBin == 0
    let detectorHasUniformBins =
      sourceDetectorRows % detectorBin == 0
      && sourceDetectorColumns % detectorBin == 0
    let scanState: Metal4DSTEMSamplingPropagationState =
      sourceScan == nil
      ? .unavailable
      : !scanHasUniformBins
        ? .nonuniformEdgeBins
        : scanBin == 1 ? .unchanged : .uniformlyScaled
    let detectorState: Metal4DSTEMSamplingPropagationState =
      sourceDetector == nil
      ? .unavailable
      : !detectorHasUniformBins
        ? .nonuniformEdgeBins
        : detectorBin == 1 ? .unchanged : .uniformlyScaled
    return Metal4DSTEMSamplingPropagation(
      sourceScan: sourceScan,
      workingScan: scanHasUniformBins
        ? try sourceScan?.scaled(by: scanBin) : nil,
      sourceDetector: sourceDetector,
      workingDetector: detectorHasUniformBins
        ? try sourceDetector?.scaled(by: detectorBin) : nil,
      scanState: scanState,
      detectorState: detectorState,
      scanRegionRowStartInSourcePixels: scanRegion.rowStart,
      scanRegionColumnStartInSourcePixels: scanRegion.columnStart,
      firstWorkingScanCenterRowInSourcePixels:
        Double(scanRegion.rowStart) + Double(min(scanBin, scanRegion.rows) - 1) / 2,
      firstWorkingScanCenterColumnInSourcePixels:
        Double(scanRegion.columnStart) + Double(min(scanBin, scanRegion.columns) - 1) / 2,
      firstWorkingDetectorCenterRowInSourcePixels:
        Double(min(detectorBin, sourceDetectorRows) - 1) / 2,
      firstWorkingDetectorCenterColumnInSourcePixels:
        Double(min(detectorBin, sourceDetectorColumns) - 1) / 2
    )
  }
}

/// One scan-row-contiguous shard of an exact word-major working volume.
///
/// Each shard retains every detector pixel and every selected scan column for
/// its half-open output scan-row interval. Detector words remain the major
/// dimension inside the shard; `outputScanPositionCount` is their local stride.
public struct Metal4DSTEMExactBinningShard: Codable, Equatable, Sendable {
  public let index: Int
  public let outputScanRowStart: Int
  public let outputScanRowStop: Int
  public let outputScanPositionStart: Int
  public let outputScanPositionCount: Int
  public let payloadBytes: UInt64
  public let outputLayout: Metal4DSTEMOutputLayout
}

/// Deterministic bounded-buffer storage for one exact logical working volume.
///
/// Sharding changes only physical storage. The source/working geometry, dtype,
/// exact-sum semantics, calibration propagation, and complete scan coverage are
/// inherited unchanged from `provenance`.
public struct Metal4DSTEMExactBinningShardPlan: Codable, Equatable, Sendable {
  public static let currentSchema = "quantem.gpu.metal-4dstem-exact-binning-shards/v1"

  public let schema: String
  public let provenance: Metal4DSTEMExactBinningProvenance
  public let maximumShardBytes: UInt64
  public let bytesPerOutputScanRow: UInt64
  public let shards: [Metal4DSTEMExactBinningShard]
  public let totalPayloadBytes: UInt64
  public let maximumActualShardBytes: UInt64

  public init(
    provenance: Metal4DSTEMExactBinningProvenance,
    maximumShardBytes: UInt64
  ) throws {
    guard provenance.schema == Metal4DSTEMExactBinningProvenance.currentSchema,
      provenance.outputScanRows > 0,
      provenance.outputScanColumns > 0,
      provenance.outputDetectorRows > 0,
      provenance.outputDetectorColumns > 0
    else { throw Metal4DSTEMExactBinnerError.invalidShardPlan }

    guard
      let detectorPixels = Self.checkedProduct(
        UInt64(provenance.outputDetectorRows),
        UInt64(provenance.outputDetectorColumns)
      )
    else { throw Metal4DSTEMExactBinnerError.arithmeticOverflow }
    let wordsPerScan =
      provenance.outputDtype == .uint16
      ? (detectorPixels + 1) / 2
      : detectorPixels
    let expectedLayout: Metal4DSTEMOutputLayout =
      provenance.outputDtype == .uint16
      ? .detectorWordMajorPackedUInt16 : .detectorWordMajorUInt32
    guard provenance.outputDtype == .uint16 || provenance.outputDtype == .uint32,
      provenance.outputLayout == expectedLayout,
      let wordsPerRow = Self.checkedProduct(
        wordsPerScan, UInt64(provenance.outputScanColumns)
      ),
      let rowBytes = Self.checkedProduct(
        wordsPerRow, UInt64(MemoryLayout<UInt32>.stride)
      ),
      let expectedTotal = Self.checkedProduct(
        rowBytes, UInt64(provenance.outputScanRows)
      ),
      expectedTotal == provenance.outputPayloadBytes
    else { throw Metal4DSTEMExactBinnerError.invalidShardPlan }
    guard maximumShardBytes >= rowBytes else {
      throw Metal4DSTEMExactBinnerError.maximumShardBytesTooSmall(
        required: rowBytes, actual: maximumShardBytes
      )
    }

    let rowsPerShard = Int(
      min(
        UInt64(provenance.outputScanRows),
        maximumShardBytes / rowBytes
      )
    )
    guard rowsPerShard > 0 else {
      throw Metal4DSTEMExactBinnerError.maximumShardBytesTooSmall(
        required: rowBytes, actual: maximumShardBytes
      )
    }
    var generated: [Metal4DSTEMExactBinningShard] = []
    var rowStart = 0
    var total: UInt64 = 0
    var largest: UInt64 = 0
    while rowStart < provenance.outputScanRows {
      let rowCount = min(
        provenance.outputScanRows - rowStart,
        rowsPerShard
      )
      let rowStopResult = rowStart.addingReportingOverflow(rowCount)
      guard !rowStopResult.overflow else {
        throw Metal4DSTEMExactBinnerError.arithmeticOverflow
      }
      let rowStop = rowStopResult.partialValue
      let positionStartResult = rowStart.multipliedReportingOverflow(
        by: provenance.outputScanColumns
      )
      let positionCountResult = rowCount.multipliedReportingOverflow(
        by: provenance.outputScanColumns
      )
      guard !positionStartResult.overflow, !positionCountResult.overflow,
        let payload = Self.checkedProduct(UInt64(rowCount), rowBytes),
        payload <= maximumShardBytes
      else { throw Metal4DSTEMExactBinnerError.arithmeticOverflow }
      let nextTotal = total.addingReportingOverflow(payload)
      guard !nextTotal.overflow else {
        throw Metal4DSTEMExactBinnerError.arithmeticOverflow
      }
      generated.append(
        Metal4DSTEMExactBinningShard(
          index: generated.count,
          outputScanRowStart: rowStart,
          outputScanRowStop: rowStop,
          outputScanPositionStart: positionStartResult.partialValue,
          outputScanPositionCount: positionCountResult.partialValue,
          payloadBytes: payload,
          outputLayout: provenance.outputLayout
        )
      )
      total = nextTotal.partialValue
      largest = max(largest, payload)
      rowStart = rowStop
    }
    guard !generated.isEmpty, total == provenance.outputPayloadBytes else {
      throw Metal4DSTEMExactBinnerError.invalidShardPlan
    }

    schema = Self.currentSchema
    self.provenance = provenance
    self.maximumShardBytes = maximumShardBytes
    bytesPerOutputScanRow = rowBytes
    shards = generated
    totalPayloadBytes = total
    maximumActualShardBytes = largest
  }

  /// Rebuild and compare a decoded plan before allocating or encoding shards.
  public func validate(
    provenance expectedProvenance: Metal4DSTEMExactBinningProvenance
  ) throws {
    let expected = try Self(
      provenance: expectedProvenance,
      maximumShardBytes: maximumShardBytes
    )
    guard self == expected else {
      throw Metal4DSTEMExactBinnerError.invalidShardPlan
    }
  }

  /// Revalidate both provenance and physical sharding against a source audit.
  public func validate(sourceAudit: Metal4DSTEMExactSourceAudit) throws {
    try provenance.validate(sourceAudit: sourceAudit)
    try validate(provenance: provenance)
  }

  private static func checkedProduct(_ lhs: UInt64, _ rhs: UInt64) -> UInt64? {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    return result.overflow ? nil : result.partialValue
  }
}

/// The physical destination selected for one exact binning dispatch.
public enum Metal4DSTEMExactBinningDestination: Equatable, Sendable {
  /// One complete detector-word-major destination.
  case complete

  /// One validated scan-row shard of the complete logical destination.
  case scanRowShard(plan: Metal4DSTEMExactBinningShardPlan, index: Int)
}

/// Validation failures for exact Metal load/bin encoding.
public enum Metal4DSTEMExactBinnerError: LocalizedError, Equatable {
  case invalidCalibration
  case invalidAuditIdentity
  case invalidAuditValues
  case invalidProvenance
  case unsupportedSourceDtype(Metal4DSTEMIntegerDType)
  case sourceDtypeMismatch(expectedBytes: Int, actual: Metal4DSTEMIntegerDType)
  case unsupportedStagingDtype(Metal4DSTEMIntegerDType)
  case unsupportedOutputDtype(Metal4DSTEMIntegerDType)
  case stagingRangeOverflow(maximum: UInt32, dtype: Metal4DSTEMIntegerDType)
  case outputRangeOverflow(maximum: UInt64, dtype: Metal4DSTEMIntegerDType)
  case invalidBatchRows(Int)
  case invalidBatchCoverage(rows: Int, remaining: Int)
  case misalignedNonfinalBatch(rows: Int, scanBin: Int)
  case invalidDestinationScanRowOffset(Int)
  case maximumShardBytesTooSmall(required: UInt64, actual: UInt64)
  case invalidShardPlan
  case invalidDestinationShard(Int)
  case batchCrossesDestinationShard(
    batchStart: Int, batchStop: Int, shardStart: Int, shardStop: Int
  )
  case contiguousFramesRequireFullScanBinOne
  case invalidGlobalScanPositionRange(offset: Int, count: Int)
  case frameRangeCrossesDestinationShard(
    frameStart: Int, frameStop: Int, shardStart: Int, shardStop: Int
  )
  case invalidSourceOffset(Int)
  case arithmeticOverflow
  case sourceBufferTooSmall(expected: UInt64, actual: UInt64)
  case destinationBufferTooSmall(expected: UInt64, actual: UInt64)
  case commandEncoderUnavailable

  public var errorDescription: String? {
    switch self {
    case .invalidCalibration:
      "Sampling must be finite and positive, and unit, provenance, and evidence "
        + "must be nonempty."
    case .invalidAuditIdentity:
      "Exact dtype decisions require lowercase SHA-256 identities for both source and audit."
    case .invalidAuditValues:
      "The exact value-range audit contains inconsistent maximum, above-255, or bad-pixel values."
    case .invalidProvenance:
      "Exact binning provenance does not match its source audit, shape, dtype, or layout. "
        + "Rebuild it from the validated load plan."
    case .unsupportedSourceDtype(let dtype):
      "Exact native 4D-STEM loading does not support \(dtype.rawValue) sources. "
        + "Use uint8 or uint16 source counts."
    case .sourceDtypeMismatch(let expectedBytes, let actual):
      "The load plan records \(expectedBytes) source bytes per value, but the "
        + "identity-bound audit records \(actual.rawValue). Rebuild the plan or audit "
        + "from the same source metadata."
    case .unsupportedStagingDtype(let dtype):
      "Exact Metal binning cannot read \(dtype.rawValue) staging values. Use uint8 or uint16."
    case .unsupportedOutputDtype(let dtype):
      "Exact Metal binning cannot write \(dtype.rawValue) output. Use uint16 or uint32."
    case .stagingRangeOverflow(let maximum, let dtype):
      "The audited source maximum \(maximum) does not fit \(dtype.rawValue) staging. "
        + "Use a wider exact staging dtype."
    case .outputRangeOverflow(let maximum, let dtype):
      "The exact output bound \(maximum) does not fit \(dtype.rawValue). "
        + "Use a wider exact output dtype without changing scan or detector coverage."
    case .invalidBatchRows(let rows):
      "The exact Metal binning batch must contain a positive number of scan rows, not \(rows)."
    case .invalidBatchCoverage(let rows, let remaining):
      "The exact Metal binning batch contains \(rows) source rows, but \(remaining) "
        + "rows remain at this destination offset. Encode each selected source row exactly once."
    case .misalignedNonfinalBatch(let rows, let scanBin):
      "A nonfinal batch with \(rows) scan rows is not aligned to scan bin \(scanBin)."
    case .invalidDestinationScanRowOffset(let offset):
      "Destination scan-row offset \(offset) is outside the working scan."
    case .maximumShardBytesTooSmall(let required, let actual):
      "An exact output scan row requires \(required) bytes, but the shard limit "
        + "is \(actual). Increase the physical shard limit without changing the "
        + "scientific load plan."
    case .invalidShardPlan:
      "The exact output shard plan does not match its complete provenance or "
        + "deterministic scan-row partition. Rebuild it from validated provenance."
    case .invalidDestinationShard(let index):
      "Exact output shard index \(index) is outside the declared shard plan."
    case .batchCrossesDestinationShard(
      let batchStart, let batchStop, let shardStart, let shardStop
    ):
      "Exact output rows [\(batchStart), \(batchStop)) cross the selected shard "
        + "rows [\(shardStart), \(shardStop)). Split the source batch at the "
        + "declared shard boundary."
    case .contiguousFramesRequireFullScanBinOne:
      "Contiguous indexed-frame binning requires the complete source scan, scan bin 1, "
        + "and no real-space crop. Use the row-batch encoder for a different selection."
    case .invalidGlobalScanPositionRange(let offset, let count):
      "Contiguous indexed-frame range [\(offset), \(offset + count)) is outside the "
        + "complete working scan."
    case .frameRangeCrossesDestinationShard(
      let frameStart, let frameStop, let shardStart, let shardStop
    ):
      "Contiguous indexed frames [\(frameStart), \(frameStop)) cross the selected "
        + "destination shard positions [\(shardStart), \(shardStop)). Split the "
        + "dispatch at the declared shard boundary."
    case .invalidSourceOffset(let offset):
      "Source byte offset \(offset) is negative or misaligned for the staging dtype."
    case .arithmeticOverflow:
      "The exact Metal binning geometry exceeds a safe host or 32-bit Metal index range."
    case .sourceBufferTooSmall(let expected, let actual):
      "The exact Metal binning source buffer contains \(actual) bytes; expected at least \(expected)."
    case .destinationBufferTooSmall(let expected, let actual):
      "The exact Metal binning destination buffer contains \(actual) bytes; expected at least \(expected)."
    case .commandEncoderUnavailable:
      "Metal could not create the exact 4D-STEM binning command encoder."
    }
  }
}

/// Typed encoder for exact scan/detector summation into word-major storage.
///
/// This type does not choose a device policy, allocate a resident volume,
/// commit a command buffer, or synchronize the GPU. A consumer selects the
/// plan and memory policy, then uses the returned provenance at its UI/cache
/// boundary.
public final class Metal4DSTEMExactBinner {
  let u8ToU16: MTLComputePipelineState
  let u16ToU16: MTLComputePipelineState
  let u8ToU32: MTLComputePipelineState
  let u16ToU32: MTLComputePipelineState
  let contiguousU16ToU16: MTLComputePipelineState

  public convenience init(device: MTLDevice) throws {
    let library = try Metal4DSTEMKernels.makeDetectorLibrary(device: device)
    try self.init(device: device, detectorLibrary: library)
  }

  /// Build pipelines from an already compiled packaged detector library.
  ///
  /// Loader implementations use this initializer to avoid compiling the same
  /// Metal source twice. The supplied library must come from
  /// `Metal4DSTEMKernels.makeDetectorLibrary(device:)` for the same device.
  public init(device: MTLDevice, detectorLibrary library: MTLLibrary) throws {
    u8ToU16 = try Self.pipeline(
      device: device, library: library,
      name: Metal4DSTEMKernels.scanDetectorBinU8ToU16Function
    )
    u16ToU16 = try Self.pipeline(
      device: device, library: library,
      name: Metal4DSTEMKernels.scanDetectorBinU16ToU16Function
    )
    u8ToU32 = try Self.pipeline(
      device: device, library: library,
      name: Metal4DSTEMKernels.scanDetectorBinU8Function
    )
    u16ToU32 = try Self.pipeline(
      device: device, library: library,
      name: Metal4DSTEMKernels.scanDetectorBinU16Function
    )
    contiguousU16ToU16 = try Self.pipeline(
      device: device, library: library,
      name: Metal4DSTEMKernels.contiguousDetectorBinU16ToU16Function
    )
  }

  /// Build validated provenance before allocating or encoding a load.
  public static func provenance(
    plan: Metal4DSTEMLoadPlan,
    sourceAudit: Metal4DSTEMExactSourceAudit,
    stagingDtype: Metal4DSTEMIntegerDType,
    outputDtype: Metal4DSTEMIntegerDType
  ) throws -> Metal4DSTEMExactBinningProvenance {
    try sourceAudit.validate()
    guard stagingDtype == .uint8 || stagingDtype == .uint16 else {
      throw Metal4DSTEMExactBinnerError.unsupportedStagingDtype(stagingDtype)
    }
    guard outputDtype == .uint16 || outputDtype == .uint32 else {
      throw Metal4DSTEMExactBinnerError.unsupportedOutputDtype(outputDtype)
    }
    let expectedSourceBytes = sourceAudit.sourceDtype.bytesPerValue
    guard expectedSourceBytes == plan.sourceBytesPerValue else {
      throw Metal4DSTEMExactBinnerError.sourceDtypeMismatch(
        expectedBytes: plan.sourceBytesPerValue,
        actual: sourceAudit.sourceDtype
      )
    }
    guard sourceAudit.badPixelIndices.allSatisfy({ $0 < plan.detectorPixels }) else {
      throw Metal4DSTEMExactBinnerError.invalidAuditValues
    }
    let stagingMaximum: UInt32 =
      stagingDtype == .uint8
      ? UInt32(UInt8.max) : UInt32(UInt16.max)
    guard sourceAudit.maximumSourceCount <= stagingMaximum,
      stagingDtype != .uint8 || sourceAudit.provesLosslessUInt8Staging
    else {
      throw Metal4DSTEMExactBinnerError.stagingRangeOverflow(
        maximum: sourceAudit.maximumSourceCount, dtype: stagingDtype
      )
    }
    let bounds = try plan.exactOutputSampleBounds(
      maxSourceCount: sourceAudit.maximumSourceCount
    )
    let fitsOutput = outputDtype == .uint16 ? bounds.fitsUInt16 : bounds.fitsUInt32
    guard fitsOutput else {
      throw Metal4DSTEMExactBinnerError.outputRangeOverflow(
        maximum: bounds.maximumOutputCount, dtype: outputDtype
      )
    }
    guard let outputDetectorPixels = UInt64(exactly: plan.outputDetectorPixels),
      let outputScanPositions = UInt64(exactly: plan.outputScanPositions)
    else { throw Metal4DSTEMExactBinnerError.arithmeticOverflow }
    let outputWordsPerScan =
      outputDtype == .uint16
      ? (outputDetectorPixels + 1) / 2
      : outputDetectorPixels
    guard
      let outputWords = Self.checkedProduct(
        outputScanPositions, outputWordsPerScan
      ),
      let outputBytes = Self.checkedProduct(
        outputWords, UInt64(MemoryLayout<UInt32>.stride)
      )
    else { throw Metal4DSTEMExactBinnerError.arithmeticOverflow }
    return Metal4DSTEMExactBinningProvenance(
      schema: Metal4DSTEMExactBinningProvenance.currentSchema,
      sourceScanRows: plan.sourceScanRows,
      sourceScanColumns: plan.sourceScanColumns,
      scanRegion: plan.scanRegion,
      sourceDetectorRows: plan.detectorRows,
      sourceDetectorColumns: plan.detectorColumns,
      outputScanRows: plan.outputScanRows,
      outputScanColumns: plan.outputScanColumns,
      outputDetectorRows: plan.outputDetectorRows,
      outputDetectorColumns: plan.outputDetectorColumns,
      scanBin: plan.scanBin,
      detectorBin: plan.detectorBin,
      sourceIdentitySHA256: sourceAudit.sourceIdentitySHA256,
      valueRangeAuditSHA256: sourceAudit.auditSHA256,
      sourceDtype: sourceAudit.sourceDtype,
      stagingDtype: stagingDtype,
      outputDtype: outputDtype,
      stagingLayout: .frameMajorSelectedColumns,
      outputLayout: outputDtype == .uint16
        ? .detectorWordMajorPackedUInt16 : .detectorWordMajorUInt32,
      badPixelApplication: .alreadyZeroedUsingAudit,
      badPixelIndices: sourceAudit.badPixelIndices,
      reduction: .exactIntegerSum,
      maximumSourceCount: sourceAudit.maximumSourceCount,
      pixelsAbove255: sourceAudit.pixelsAbove255,
      maximumOutputCount: bounds.maximumOutputCount,
      outputPayloadBytes: outputBytes
    )
  }

  func pipeline(
    stagingDtype: Metal4DSTEMIntegerDType,
    outputDtype: Metal4DSTEMIntegerDType
  ) -> MTLComputePipelineState {
    switch (stagingDtype, outputDtype) {
    case (.uint8, .uint16): u8ToU16
    case (.uint16, .uint16): u16ToU16
    case (.uint8, .uint32): u8ToU32
    case (.uint16, .uint32): u16ToU32
    default:
      preconditionFailure("Exact binner dtype validation must run before pipeline selection")
    }
  }

  private static func pipeline(
    device: MTLDevice,
    library: MTLLibrary,
    name: String
  ) throws -> MTLComputePipelineState {
    guard let function = library.makeFunction(name: name) else {
      throw Metal4DSTEMKernelsError.missingResource("Metal function \(name)")
    }
    return try device.makeComputePipelineState(function: function)
  }

  static func checkedProduct(_ lhs: UInt64, _ rhs: UInt64) -> UInt64? {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    return result.overflow ? nil : result.partialValue
  }

  static func checkedSum(_ lhs: UInt64, _ rhs: UInt64) -> UInt64? {
    let result = lhs.addingReportingOverflow(rhs)
    return result.overflow ? nil : result.partialValue
  }

  static func fitsMetalUIntProduct(_ lhs: Int, _ rhs: Int) -> Bool {
    guard let left = UInt64(exactly: lhs), let right = UInt64(exactly: rhs),
      let product = checkedProduct(left, right)
    else { return false }
    return product <= UInt64(UInt32.max)
  }
}
