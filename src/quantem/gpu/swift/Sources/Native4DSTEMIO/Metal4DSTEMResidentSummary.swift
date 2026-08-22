import CryptoKit
import Foundation

/// Exact integer quantities retained beside a resident 4D-STEM cache.
public enum Metal4DSTEMResidentSummaryRole: String, Codable, CaseIterable, Hashable, Sendable {
  case brightField = "bright_field"
  case annularBrightField = "annular_bright_field"
  case annularDarkField = "annular_dark_field"
  case totalIntensity = "total_intensity"
  case detectorRowMoment = "detector_row_moment"
  case detectorColumnMoment = "detector_column_moment"
  case selectedDiffraction = "selected_diffraction"

  public var fileName: String {
    switch self {
    case .brightField: "bf.u32"
    case .annularBrightField: "abf.u32"
    case .annularDarkField: "adf.u32"
    case .totalIntensity: "total.u64"
    case .detectorRowMoment: "detector_row_moment.u64"
    case .detectorColumnMoment: "detector_column_moment.u64"
    case .selectedDiffraction: "selected.u32"
    }
  }

  public var dtype: String {
    switch self {
    case .totalIntensity, .detectorRowMoment, .detectorColumnMoment: "uint64"
    default: "uint32"
    }
  }

  var bytesPerValue: UInt64 { dtype == "uint64" ? 8 : 4 }
}

/// Shape, dtype, and digest of one exact resident-summary artifact.
public struct Metal4DSTEMResidentSummaryArtifact: Codable, Equatable, Sendable {
  public let role: Metal4DSTEMResidentSummaryRole
  public let fileName: String
  public let dtype: String
  public let rows: Int
  public let columns: Int
  public let byteCount: UInt64
  public let sha256: String
}

/// Provenance for exact products and sufficient statistics cached with a resident volume.
public struct Metal4DSTEMResidentSummaryMetadata: Codable, Equatable, Sendable {
  public static let currentFormatVersion = 1
  public static let currentSchema = "quantem.gpu.resident-summary/v1"
  public static let manifestFileName = "summary.json"

  public let formatVersion: Int
  public let schema: String
  public let datasetID: String
  public let sourceIdentitySHA256: String?
  public let residentPayloadSHA256: String
  public let outputScanRows: Int
  public let outputScanColumns: Int
  public let outputDetectorRows: Int
  public let outputDetectorColumns: Int
  public let outputDtype: String
  public let scanRowStart: Int
  public let scanRowStop: Int
  public let scanColumnStart: Int
  public let scanColumnStop: Int
  public let scanBin: Int
  public let detectorBin: Int
  public let maxCount: UInt32
  public let pixelsAbove255: UInt64
  public let detectorBandsSHA256: String
  public let selectedScanRow: Int
  public let selectedScanColumn: Int
  public let artifacts: [Metal4DSTEMResidentSummaryArtifact]
}

/// A validated summary whose arrays are exact little-endian `uint32` or `uint64` values.
public struct Metal4DSTEMResidentSummary: Sendable {
  public let metadata: Metal4DSTEMResidentSummaryMetadata
  public let artifacts: [Metal4DSTEMResidentSummaryRole: Data]
}

public enum Metal4DSTEMResidentSummaryError: LocalizedError {
  case destinationExists(String)
  case invalidMetadata(String)
  case missingArtifact(Metal4DSTEMResidentSummaryRole)
  case artifactSizeMismatch(
    Metal4DSTEMResidentSummaryRole,
    expected: UInt64,
    actual: UInt64
  )
  case artifactDigestMismatch(
    Metal4DSTEMResidentSummaryRole,
    expected: String,
    actual: String
  )

  public var errorDescription: String? {
    switch self {
    case .destinationExists(let path):
      "Resident summary destination already exists at \(path). Choose a new directory."
    case .invalidMetadata(let reason):
      "Resident summary metadata is invalid: \(reason)"
    case .missingArtifact(let role):
      "Resident summary is missing the exact \(role.rawValue) artifact."
    case .artifactSizeMismatch(let role, let expected, let actual):
      "Resident summary \(role.rawValue) contains \(actual) bytes; expected \(expected)."
    case .artifactDigestMismatch(let role, let expected, let actual):
      "Resident summary \(role.rawValue) SHA-256 is \(actual); expected \(expected)."
    }
  }
}

/// Atomically write and strictly validate exact resident-cache summaries.
public enum Metal4DSTEMResidentSummaryIO {
  public static func write(
    to directoryURL: URL,
    residentMetadata: Metal4DSTEMResidentCacheMetadata,
    detectorBands: Data,
    selectedScanRow: Int,
    selectedScanColumn: Int,
    artifacts: [Metal4DSTEMResidentSummaryRole: Data]
  ) throws -> Metal4DSTEMResidentSummaryMetadata {
    guard !FileManager.default.fileExists(atPath: directoryURL.path) else {
      throw Metal4DSTEMResidentSummaryError.destinationExists(directoryURL.path)
    }
    try Metal4DSTEMResidentCacheIO.validateMetadata(
      residentMetadata,
      requireSealedPayload: true
    )
    try validateDetectorBands(detectorBands, residentMetadata: residentMetadata)
    try validateVirtualDetectorBounds(residentMetadata)
    try validateSelectedScan(
      row: selectedScanRow,
      column: selectedScanColumn,
      residentMetadata: residentMetadata
    )

    let descriptors = try Metal4DSTEMResidentSummaryRole.allCases.map { role in
      guard let data = artifacts[role] else {
        throw Metal4DSTEMResidentSummaryError.missingArtifact(role)
      }
      let shape = artifactShape(role, residentMetadata: residentMetadata)
      let expectedBytes = try artifactByteCount(
        rows: shape.rows,
        columns: shape.columns,
        bytesPerValue: role.bytesPerValue
      )
      guard UInt64(data.count) == expectedBytes else {
        throw Metal4DSTEMResidentSummaryError.artifactSizeMismatch(
          role,
          expected: expectedBytes,
          actual: UInt64(data.count)
        )
      }
      return Metal4DSTEMResidentSummaryArtifact(
        role: role,
        fileName: role.fileName,
        dtype: role.dtype,
        rows: shape.rows,
        columns: shape.columns,
        byteCount: expectedBytes,
        sha256: sha256(data)
      )
    }
    guard artifacts.count == descriptors.count else {
      throw Metal4DSTEMResidentSummaryError.invalidMetadata(
        "artifacts contains an unsupported role"
      )
    }

    let metadata = Metal4DSTEMResidentSummaryMetadata(
      formatVersion: Metal4DSTEMResidentSummaryMetadata.currentFormatVersion,
      schema: Metal4DSTEMResidentSummaryMetadata.currentSchema,
      datasetID: residentMetadata.datasetID,
      sourceIdentitySHA256: residentMetadata.sourceIdentitySHA256,
      residentPayloadSHA256: residentMetadata.payloadSHA256,
      outputScanRows: residentMetadata.outputScanRows,
      outputScanColumns: residentMetadata.outputScanColumns,
      outputDetectorRows: residentMetadata.outputDetectorRows,
      outputDetectorColumns: residentMetadata.outputDetectorColumns,
      outputDtype: residentMetadata.outputDtype,
      scanRowStart: residentMetadata.scanRowStart,
      scanRowStop: residentMetadata.scanRowStop,
      scanColumnStart: residentMetadata.scanColumnStart,
      scanColumnStop: residentMetadata.scanColumnStop,
      scanBin: residentMetadata.scanBin,
      detectorBin: residentMetadata.detectorBin,
      maxCount: residentMetadata.maxCount,
      pixelsAbove255: residentMetadata.pixelsAbove255,
      detectorBandsSHA256: sha256(detectorBands),
      selectedScanRow: selectedScanRow,
      selectedScanColumn: selectedScanColumn,
      artifacts: descriptors
    )
    try validateMetadata(
      metadata,
      residentMetadata: residentMetadata,
      detectorBands: detectorBands
    )

    let parent = directoryURL.deletingLastPathComponent()
    try FileManager.default.createDirectory(at: parent, withIntermediateDirectories: true)
    let temporary = parent.appendingPathComponent(
      ".\(directoryURL.lastPathComponent).\(UUID().uuidString).tmp",
      isDirectory: true
    )
    try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: false)
    defer { try? FileManager.default.removeItem(at: temporary) }
    for descriptor in descriptors {
      try artifacts[descriptor.role]!.write(
        to: temporary.appendingPathComponent(descriptor.fileName),
        options: .atomic
      )
    }
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
    try encoder.encode(metadata).write(
      to: temporary.appendingPathComponent(
        Metal4DSTEMResidentSummaryMetadata.manifestFileName
      ),
      options: .atomic
    )
    try FileManager.default.moveItem(at: temporary, to: directoryURL)
    return metadata
  }

  public static func read(
    from directoryURL: URL,
    residentMetadata: Metal4DSTEMResidentCacheMetadata,
    detectorBands: Data
  ) throws -> Metal4DSTEMResidentSummary {
    let manifestURL = directoryURL.appendingPathComponent(
      Metal4DSTEMResidentSummaryMetadata.manifestFileName
    )
    let metadata = try JSONDecoder().decode(
      Metal4DSTEMResidentSummaryMetadata.self,
      from: Data(contentsOf: manifestURL)
    )
    try validateMetadata(
      metadata,
      residentMetadata: residentMetadata,
      detectorBands: detectorBands
    )

    var payloads: [Metal4DSTEMResidentSummaryRole: Data] = [:]
    for descriptor in metadata.artifacts {
      let data = try Data(
        contentsOf: directoryURL.appendingPathComponent(descriptor.fileName)
      )
      guard UInt64(data.count) == descriptor.byteCount else {
        throw Metal4DSTEMResidentSummaryError.artifactSizeMismatch(
          descriptor.role,
          expected: descriptor.byteCount,
          actual: UInt64(data.count)
        )
      }
      let actual = sha256(data)
      guard actual == descriptor.sha256 else {
        throw Metal4DSTEMResidentSummaryError.artifactDigestMismatch(
          descriptor.role,
          expected: descriptor.sha256,
          actual: actual
        )
      }
      payloads[descriptor.role] = data
    }
    return Metal4DSTEMResidentSummary(metadata: metadata, artifacts: payloads)
  }

  private static func validateMetadata(
    _ metadata: Metal4DSTEMResidentSummaryMetadata,
    residentMetadata: Metal4DSTEMResidentCacheMetadata,
    detectorBands: Data
  ) throws {
    func invalid(_ reason: String) throws -> Never {
      throw Metal4DSTEMResidentSummaryError.invalidMetadata(reason)
    }
    try Metal4DSTEMResidentCacheIO.validateMetadata(
      residentMetadata,
      requireSealedPayload: true
    )
    guard metadata.formatVersion == Metal4DSTEMResidentSummaryMetadata.currentFormatVersion,
      metadata.schema == Metal4DSTEMResidentSummaryMetadata.currentSchema
    else { try invalid("format version or schema is not supported") }
    guard metadata.datasetID == residentMetadata.datasetID,
      metadata.sourceIdentitySHA256 == residentMetadata.sourceIdentitySHA256,
      metadata.residentPayloadSHA256 == residentMetadata.payloadSHA256
    else { try invalid("source or resident payload provenance does not match") }
    guard metadata.outputScanRows == residentMetadata.outputScanRows,
      metadata.outputScanColumns == residentMetadata.outputScanColumns,
      metadata.outputDetectorRows == residentMetadata.outputDetectorRows,
      metadata.outputDetectorColumns == residentMetadata.outputDetectorColumns,
      metadata.outputDtype == residentMetadata.outputDtype,
      metadata.scanRowStart == residentMetadata.scanRowStart,
      metadata.scanRowStop == residentMetadata.scanRowStop,
      metadata.scanColumnStart == residentMetadata.scanColumnStart,
      metadata.scanColumnStop == residentMetadata.scanColumnStop,
      metadata.scanBin == residentMetadata.scanBin,
      metadata.detectorBin == residentMetadata.detectorBin,
      metadata.maxCount == residentMetadata.maxCount,
      metadata.pixelsAbove255 == residentMetadata.pixelsAbove255
    else { try invalid("shape, dtype, scan region, binning, or count audit does not match") }
    try validateDetectorBands(detectorBands, residentMetadata: residentMetadata)
    guard metadata.detectorBandsSHA256 == sha256(detectorBands) else {
      try invalid("detector band definition does not match")
    }
    try validateVirtualDetectorBounds(residentMetadata)
    try validateSelectedScan(
      row: metadata.selectedScanRow,
      column: metadata.selectedScanColumn,
      residentMetadata: residentMetadata
    )

    guard metadata.artifacts.count == Metal4DSTEMResidentSummaryRole.allCases.count else {
      try invalid("artifact set is incomplete")
    }
    var roles = Set<Metal4DSTEMResidentSummaryRole>()
    for descriptor in metadata.artifacts {
      guard roles.insert(descriptor.role).inserted else {
        try invalid("artifact role \(descriptor.role.rawValue) is duplicated")
      }
      let shape = artifactShape(descriptor.role, residentMetadata: residentMetadata)
      let expectedBytes = try artifactByteCount(
        rows: shape.rows,
        columns: shape.columns,
        bytesPerValue: descriptor.role.bytesPerValue
      )
      guard descriptor.fileName == descriptor.role.fileName,
        descriptor.dtype == descriptor.role.dtype,
        descriptor.rows == shape.rows,
        descriptor.columns == shape.columns,
        descriptor.byteCount == expectedBytes,
        isSHA256(descriptor.sha256)
      else { try invalid("artifact descriptor for \(descriptor.role.rawValue) is invalid") }
    }
    guard roles == Set(Metal4DSTEMResidentSummaryRole.allCases) else {
      try invalid("artifact set is incomplete")
    }
  }

  private static func validateDetectorBands(
    _ detectorBands: Data,
    residentMetadata: Metal4DSTEMResidentCacheMetadata
  ) throws {
    let expected = try checkedProduct(
      UInt64(residentMetadata.outputDetectorRows),
      UInt64(residentMetadata.outputDetectorColumns),
      reason: "detector shape overflows UInt64"
    )
    guard UInt64(detectorBands.count) == expected else {
      throw Metal4DSTEMResidentSummaryError.invalidMetadata(
        "detector bands contains \(detectorBands.count) bytes; expected \(expected)"
      )
    }
  }

  private static func validateSelectedScan(
    row: Int,
    column: Int,
    residentMetadata: Metal4DSTEMResidentCacheMetadata
  ) throws {
    guard row >= 0, row < residentMetadata.outputScanRows,
      column >= 0, column < residentMetadata.outputScanColumns
    else {
      throw Metal4DSTEMResidentSummaryError.invalidMetadata(
        "selected scan (\(row), \(column)) is outside the output scan"
      )
    }
  }

  private static func validateVirtualDetectorBounds(
    _ metadata: Metal4DSTEMResidentCacheMetadata
  ) throws {
    let detectorPixels = try checkedProduct(
      UInt64(metadata.sourceDetectorRows),
      UInt64(metadata.sourceDetectorColumns),
      reason: "source detector shape overflows UInt64"
    )
    let scanContributions = try checkedProduct(
      UInt64(metadata.scanBin),
      UInt64(metadata.scanBin),
      reason: "scan contribution count overflows UInt64"
    )
    let maximumBandSum = try checkedProduct(
      UInt64(metadata.maxCount),
      detectorPixels,
      reason: "virtual-detector bound overflows UInt64"
    )
    let scanBinnedMaximumBandSum = try checkedProduct(
      maximumBandSum,
      scanContributions,
      reason: "scan-binned virtual-detector bound overflows UInt64"
    )
    guard scanBinnedMaximumBandSum <= UInt64(UInt32.max) else {
      throw Metal4DSTEMResidentSummaryError.invalidMetadata(
        "exact virtual-detector sums do not fit uint32"
      )
    }
  }

  private static func artifactShape(
    _ role: Metal4DSTEMResidentSummaryRole,
    residentMetadata: Metal4DSTEMResidentCacheMetadata
  ) -> (rows: Int, columns: Int) {
    if role == .selectedDiffraction {
      return (
        residentMetadata.outputDetectorRows,
        residentMetadata.outputDetectorColumns
      )
    }
    return (
      residentMetadata.outputScanRows,
      residentMetadata.outputScanColumns
    )
  }

  private static func artifactByteCount(
    rows: Int,
    columns: Int,
    bytesPerValue: UInt64
  ) throws -> UInt64 {
    let values = try checkedProduct(
      UInt64(rows),
      UInt64(columns),
      reason: "artifact shape overflows UInt64"
    )
    return try checkedProduct(
      values,
      bytesPerValue,
      reason: "artifact byte count overflows UInt64"
    )
  }

  private static func checkedProduct(
    _ lhs: UInt64,
    _ rhs: UInt64,
    reason: String
  ) throws -> UInt64 {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    guard !result.overflow else {
      throw Metal4DSTEMResidentSummaryError.invalidMetadata(reason)
    }
    return result.partialValue
  }

  private static func sha256(_ data: Data) -> String {
    SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private static func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64
      && value.utf8.allSatisfy {
        (48...57).contains($0) || (97...102).contains($0)
      }
  }
}
