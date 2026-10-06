import CryptoKit
import Darwin
import Foundation

/// Stable file identity used to reject a stale resident 4D-STEM cache.
public struct Metal4DSTEMSourceIdentity: Codable, Equatable, Sendable {
  public let fileName: String
  public let byteCount: UInt64
  public let modificationTimeNanoseconds: Int64

  public init(fileName: String, byteCount: UInt64, modificationTimeNanoseconds: Int64) {
    self.fileName = fileName
    self.byteCount = byteCount
    self.modificationTimeNanoseconds = modificationTimeNanoseconds
  }

  public init(url: URL) throws {
    var status = stat()
    guard lstat(url.path, &status) == 0, status.st_size >= 0,
      status.st_mtimespec.tv_sec >= 0, status.st_mtimespec.tv_nsec >= 0
    else {
      throw Metal4DSTEMResidentCacheError.cannotInspectSource(url.path)
    }
    fileName = url.lastPathComponent
    byteCount = UInt64(status.st_size)
    modificationTimeNanoseconds =
      Int64(status.st_mtimespec.tv_sec) * 1_000_000_000
      + Int64(status.st_mtimespec.tv_nsec)
  }
}

/// Row/column sampling stored with an exact resident-cache payload.
public struct Metal4DSTEMResidentAxisSampling: Codable, Equatable, Sendable {
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
  ) {
    self.row = row
    self.column = column
    self.unit = unit
    self.provenance = provenance
    self.evidence = evidence
  }
}

/// Sampling state stored independently of Metal or application policy.
public enum Metal4DSTEMResidentSamplingState: String, Codable, Equatable, Sendable {
  case unavailable
  case unchanged
  case uniformlyScaled = "uniformly_scaled"
  case nonuniformEdgeBins = "nonuniform_edge_bins"
}

/// Source-to-working sampling provenance for one exact cached selection.
///
/// This contract intentionally covers row/column sampling and bin centroids,
/// not detector centers, affine transforms, masks, or radii.
public struct Metal4DSTEMResidentSamplingPropagation: Codable, Equatable, Sendable {
  public let sourceScan: Metal4DSTEMResidentAxisSampling?
  public let workingScan: Metal4DSTEMResidentAxisSampling?
  public let sourceDetector: Metal4DSTEMResidentAxisSampling?
  public let workingDetector: Metal4DSTEMResidentAxisSampling?
  public let scanState: Metal4DSTEMResidentSamplingState
  public let detectorState: Metal4DSTEMResidentSamplingState
  public let scanRegionRowStartInSourcePixels: Int
  public let scanRegionColumnStartInSourcePixels: Int
  public let firstWorkingScanCenterRowInSourcePixels: Double
  public let firstWorkingScanCenterColumnInSourcePixels: Double
  public let firstWorkingDetectorCenterRowInSourcePixels: Double
  public let firstWorkingDetectorCenterColumnInSourcePixels: Double

  public init(
    sourceScan: Metal4DSTEMResidentAxisSampling?,
    workingScan: Metal4DSTEMResidentAxisSampling?,
    sourceDetector: Metal4DSTEMResidentAxisSampling?,
    workingDetector: Metal4DSTEMResidentAxisSampling?,
    scanState: Metal4DSTEMResidentSamplingState,
    detectorState: Metal4DSTEMResidentSamplingState,
    scanRegionRowStartInSourcePixels: Int,
    scanRegionColumnStartInSourcePixels: Int,
    firstWorkingScanCenterRowInSourcePixels: Double,
    firstWorkingScanCenterColumnInSourcePixels: Double,
    firstWorkingDetectorCenterRowInSourcePixels: Double,
    firstWorkingDetectorCenterColumnInSourcePixels: Double
  ) {
    self.sourceScan = sourceScan
    self.workingScan = workingScan
    self.sourceDetector = sourceDetector
    self.workingDetector = workingDetector
    self.scanState = scanState
    self.detectorState = detectorState
    self.scanRegionRowStartInSourcePixels = scanRegionRowStartInSourcePixels
    self.scanRegionColumnStartInSourcePixels = scanRegionColumnStartInSourcePixels
    self.firstWorkingScanCenterRowInSourcePixels =
      firstWorkingScanCenterRowInSourcePixels
    self.firstWorkingScanCenterColumnInSourcePixels =
      firstWorkingScanCenterColumnInSourcePixels
    self.firstWorkingDetectorCenterRowInSourcePixels =
      firstWorkingDetectorCenterRowInSourcePixels
    self.firstWorkingDetectorCenterColumnInSourcePixels =
      firstWorkingDetectorCenterColumnInSourcePixels
  }
}

/// Scientific and storage provenance for an exact detector-word-major cache.
public struct Metal4DSTEMResidentCacheMetadata: Codable, Equatable, Sendable {
  public static let currentFormatVersion = 2

  public let formatVersion: Int
  public let datasetID: String
  public let sourceIdentitySHA256: String?
  public let valueRangeAuditSHA256: String?
  public let valueRangeAudit: Native4DSTEMValueRangeAudit?
  public let sources: [Metal4DSTEMSourceIdentity]
  public let payloadIdentity: Metal4DSTEMSourceIdentity?
  public let sourceScanRows: Int
  public let sourceScanColumns: Int
  public let sourceDetectorRows: Int
  public let sourceDetectorColumns: Int
  public let sourceDtype: String
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
  public let badPixelIndices: [Int]
  public let maxCount: UInt32
  public let pixelsAbove255: UInt64
  public let samplingPropagation: Metal4DSTEMResidentSamplingPropagation?
  public let payloadBytes: UInt64
  public let payloadSHA256: String

  public init(
    datasetID: String,
    sourceIdentitySHA256: String? = nil,
    valueRangeAuditSHA256: String? = nil,
    valueRangeAudit: Native4DSTEMValueRangeAudit? = nil,
    sources: [Metal4DSTEMSourceIdentity],
    payloadIdentity: Metal4DSTEMSourceIdentity? = nil,
    sourceScanRows: Int,
    sourceScanColumns: Int,
    sourceDetectorRows: Int,
    sourceDetectorColumns: Int,
    sourceDtype: String,
    outputScanRows: Int,
    outputScanColumns: Int,
    outputDetectorRows: Int,
    outputDetectorColumns: Int,
    outputDtype: String,
    scanRowStart: Int,
    scanRowStop: Int,
    scanColumnStart: Int,
    scanColumnStop: Int,
    scanBin: Int,
    detectorBin: Int,
    badPixelIndices: [Int],
    maxCount: UInt32,
    pixelsAbove255: UInt64,
    samplingPropagation: Metal4DSTEMResidentSamplingPropagation? = nil,
    payloadBytes: UInt64,
    payloadSHA256: String = ""
  ) {
    formatVersion = Self.currentFormatVersion
    self.datasetID = datasetID
    self.sourceIdentitySHA256 = sourceIdentitySHA256
    self.valueRangeAuditSHA256 = valueRangeAuditSHA256
    self.valueRangeAudit = valueRangeAudit
    self.sources = sources
    self.payloadIdentity = payloadIdentity
    self.sourceScanRows = sourceScanRows
    self.sourceScanColumns = sourceScanColumns
    self.sourceDetectorRows = sourceDetectorRows
    self.sourceDetectorColumns = sourceDetectorColumns
    self.sourceDtype = sourceDtype
    self.outputScanRows = outputScanRows
    self.outputScanColumns = outputScanColumns
    self.outputDetectorRows = outputDetectorRows
    self.outputDetectorColumns = outputDetectorColumns
    self.outputDtype = outputDtype
    self.scanRowStart = scanRowStart
    self.scanRowStop = scanRowStop
    self.scanColumnStart = scanColumnStart
    self.scanColumnStop = scanColumnStop
    self.scanBin = scanBin
    self.detectorBin = detectorBin
    self.badPixelIndices = badPixelIndices
    self.maxCount = maxCount
    self.pixelsAbove255 = pixelsAbove255
    self.samplingPropagation = samplingPropagation
    self.payloadBytes = payloadBytes
    self.payloadSHA256 = payloadSHA256
  }

  func withPayloadSHA256(
    _ digest: String,
    identity: Metal4DSTEMSourceIdentity
  ) -> Self {
    Self(
      datasetID: datasetID,
      sourceIdentitySHA256: sourceIdentitySHA256,
      valueRangeAuditSHA256: valueRangeAuditSHA256,
      valueRangeAudit: valueRangeAudit,
      sources: sources,
      payloadIdentity: identity,
      sourceScanRows: sourceScanRows,
      sourceScanColumns: sourceScanColumns,
      sourceDetectorRows: sourceDetectorRows,
      sourceDetectorColumns: sourceDetectorColumns,
      sourceDtype: sourceDtype,
      outputScanRows: outputScanRows,
      outputScanColumns: outputScanColumns,
      outputDetectorRows: outputDetectorRows,
      outputDetectorColumns: outputDetectorColumns,
      outputDtype: outputDtype,
      scanRowStart: scanRowStart,
      scanRowStop: scanRowStop,
      scanColumnStart: scanColumnStart,
      scanColumnStop: scanColumnStop,
      scanBin: scanBin,
      detectorBin: detectorBin,
      badPixelIndices: badPixelIndices,
      maxCount: maxCount,
      pixelsAbove255: pixelsAbove255,
      samplingPropagation: samplingPropagation,
      payloadBytes: payloadBytes,
      payloadSHA256: digest
    )
  }
}

public enum Metal4DSTEMResidentCacheError: LocalizedError {
  case cannotInspectSource(String)
  case destinationExists(String)
  case cannotCreatePayload(String)
  case cannotWritePayload(String)
  case invalidMetadata(String)
  case payloadSizeMismatch(expected: UInt64, actual: UInt64)
  case payloadDigestMismatch(expected: String, actual: String)

  public var errorDescription: String? {
    switch self {
    case .cannotInspectSource(let path):
      "Could not inspect 4D-STEM source file at \(path)."
    case .destinationExists(let path):
      "Resident cache destination already exists at \(path). Choose a new cache path."
    case .cannotCreatePayload(let path):
      "Could not create resident cache payload at \(path)."
    case .cannotWritePayload(let path):
      "Could not write the complete resident cache payload at \(path)."
    case .invalidMetadata(let reason):
      "Resident cache metadata is invalid: \(reason)"
    case .payloadSizeMismatch(let expected, let actual):
      "Resident cache payload contains \(actual) bytes; expected \(expected)."
    case .payloadDigestMismatch(let expected, let actual):
      "Resident cache SHA-256 is \(actual); expected \(expected)."
    }
  }
}

/// Transactional writer for an exact resident-cache payload.
///
/// Callers append nonoverlapping payload regions in canonical byte order. The
/// metadata publication marker remains absent until every declared byte has
/// been written, synchronized to storage, hashed, and sealed. If a write fails
/// or the writer is abandoned, its unique temporary payload is removed. Cache
/// admission, naming, eviction, and lifecycle remain caller policy.
public final class Metal4DSTEMResidentCacheStreamWriter {
  private let payloadURL: URL
  private let metadataURL: URL
  private let temporaryPayloadURL: URL
  private let metadata: Metal4DSTEMResidentCacheMetadata
  private var descriptor: Int32
  private var hasher = SHA256()
  private var writtenBytes: UInt64 = 0
  private var didFinish = false

  public init(
    payloadURL: URL,
    metadataURL: URL,
    metadata: Metal4DSTEMResidentCacheMetadata
  ) throws {
    guard !FileManager.default.fileExists(atPath: payloadURL.path),
      !FileManager.default.fileExists(atPath: metadataURL.path)
    else {
      throw Metal4DSTEMResidentCacheError.destinationExists(payloadURL.path)
    }
    try Metal4DSTEMResidentCacheIO.validateMetadata(
      metadata,
      requireSealedPayload: false
    )
    let temporary = payloadURL.deletingLastPathComponent().appendingPathComponent(
      ".\(payloadURL.lastPathComponent).\(UUID().uuidString).tmp"
    )
    let descriptor = open(
      temporary.path,
      O_WRONLY | O_CREAT | O_EXCL,
      S_IRUSR | S_IWUSR
    )
    guard descriptor >= 0 else {
      throw Metal4DSTEMResidentCacheError.cannotCreatePayload(payloadURL.path)
    }
    self.payloadURL = payloadURL
    self.metadataURL = metadataURL
    temporaryPayloadURL = temporary
    self.metadata = metadata
    self.descriptor = descriptor
  }

  deinit {
    cancel()
  }

  /// Append one exact canonical payload region without materializing a copy.
  public func append(pointer: UnsafeRawPointer, length: Int) throws {
    guard !didFinish, descriptor >= 0, length > 0 else {
      throw Metal4DSTEMResidentCacheError.cannotWritePayload(payloadURL.path)
    }
    guard let length64 = UInt64(exactly: length) else {
      throw Metal4DSTEMResidentCacheError.cannotWritePayload(payloadURL.path)
    }
    let newTotal = writtenBytes.addingReportingOverflow(length64)
    guard !newTotal.overflow, newTotal.partialValue <= metadata.payloadBytes else {
      throw Metal4DSTEMResidentCacheError.payloadSizeMismatch(
        expected: metadata.payloadBytes,
        actual: newTotal.partialValue
      )
    }
    var chunkStart = 0
    while chunkStart < length {
      let chunkLength = min(16 * 1024 * 1024, length - chunkStart)
      var chunkWritten = 0
      while chunkWritten < chunkLength {
        let written = Darwin.write(
          descriptor,
          pointer.advanced(by: chunkStart + chunkWritten),
          chunkLength - chunkWritten
        )
        if written < 0, errno == EINTR { continue }
        guard written > 0 else {
          throw Metal4DSTEMResidentCacheError.cannotWritePayload(payloadURL.path)
        }
        chunkWritten += written
      }
      autoreleasepool {
        let bytes = Data(
          bytesNoCopy: UnsafeMutableRawPointer(
            mutating: pointer.advanced(by: chunkStart)
          ),
          count: chunkLength,
          deallocator: .none
        )
        hasher.update(data: bytes)
      }
      chunkStart += chunkLength
    }
    writtenBytes = newTotal.partialValue
  }

  /// Publish the complete payload and its sealed metadata atomically by file.
  public func finish() throws -> Metal4DSTEMResidentCacheMetadata {
    guard !didFinish, descriptor >= 0 else {
      throw Metal4DSTEMResidentCacheError.cannotWritePayload(payloadURL.path)
    }
    guard writtenBytes == metadata.payloadBytes else {
      throw Metal4DSTEMResidentCacheError.payloadSizeMismatch(
        expected: metadata.payloadBytes,
        actual: writtenBytes
      )
    }
    guard fsync(descriptor) == 0 else {
      throw Metal4DSTEMResidentCacheError.cannotWritePayload(payloadURL.path)
    }
    guard close(descriptor) == 0 else {
      descriptor = -1
      throw Metal4DSTEMResidentCacheError.cannotWritePayload(payloadURL.path)
    }
    descriptor = -1
    let digest = hasher.finalize().map { String(format: "%02x", $0) }.joined()
    var movedPayload = false
    do {
      try FileManager.default.moveItem(at: temporaryPayloadURL, to: payloadURL)
      movedPayload = true
      let identity = try Metal4DSTEMSourceIdentity(url: payloadURL)
      let complete = metadata.withPayloadSHA256(digest, identity: identity)
      try Metal4DSTEMResidentCacheIO.validateMetadata(
        complete,
        requireSealedPayload: true
      )
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let temporaryMetadataURL = metadataURL.deletingLastPathComponent()
        .appendingPathComponent(
          ".\(metadataURL.lastPathComponent).\(UUID().uuidString).tmp"
        )
      defer { try? FileManager.default.removeItem(at: temporaryMetadataURL) }
      try encoder.encode(complete).write(
        to: temporaryMetadataURL,
        options: .atomic
      )
      try FileManager.default.moveItem(
        at: temporaryMetadataURL,
        to: metadataURL
      )
      didFinish = true
      return complete
    } catch {
      if movedPayload {
        try? FileManager.default.removeItem(at: payloadURL)
      }
      throw error
    }
  }

  /// Discard an incomplete transaction. Published files are never removed.
  public func cancel() {
    guard !didFinish else { return }
    if descriptor >= 0 {
      close(descriptor)
      descriptor = -1
    }
    try? FileManager.default.removeItem(at: temporaryPayloadURL)
  }
}

/// Read and write exact resident-cache payloads without materializing a copy.
public enum Metal4DSTEMResidentCacheIO {
  public static func write(
    pointer: UnsafeRawPointer,
    length: Int,
    payloadURL: URL,
    metadataURL: URL,
    metadata: Metal4DSTEMResidentCacheMetadata
  ) throws -> Metal4DSTEMResidentCacheMetadata {
    guard length >= 0, metadata.payloadBytes == UInt64(length) else {
      throw Metal4DSTEMResidentCacheError.invalidMetadata(
        "payloadBytes \(metadata.payloadBytes) does not match buffer length \(length)"
      )
    }
    let writer = try Metal4DSTEMResidentCacheStreamWriter(
      payloadURL: payloadURL,
      metadataURL: metadataURL,
      metadata: metadata
    )
    defer { writer.cancel() }
    try writer.append(pointer: pointer, length: length)
    return try writer.finish()
  }

  public static func readMetadata(from url: URL) throws -> Metal4DSTEMResidentCacheMetadata {
    let metadata = try JSONDecoder().decode(
      Metal4DSTEMResidentCacheMetadata.self,
      from: Data(contentsOf: url)
    )
    try validateMetadata(metadata, requireSealedPayload: true)
    return metadata
  }

  public static func validatePayload(
    at url: URL,
    metadata: Metal4DSTEMResidentCacheMetadata,
    verifySHA256: Bool = true
  ) throws {
    try validateMetadata(metadata, requireSealedPayload: true)
    let identity = try Metal4DSTEMSourceIdentity(url: url)
    guard identity.byteCount == metadata.payloadBytes else {
      throw Metal4DSTEMResidentCacheError.payloadSizeMismatch(
        expected: metadata.payloadBytes,
        actual: identity.byteCount
      )
    }
    guard verifySHA256 else {
      guard let expectedIdentity = metadata.payloadIdentity,
        expectedIdentity == identity,
        !metadata.payloadSHA256.isEmpty
      else {
        throw Metal4DSTEMResidentCacheError.invalidMetadata(
          "payload identity changed or was not sealed by the cache writer"
        )
      }
      return
    }
    let data = try Data(contentsOf: url, options: .mappedIfSafe)
    let actual = SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    guard actual == metadata.payloadSHA256 else {
      throw Metal4DSTEMResidentCacheError.payloadDigestMismatch(
        expected: metadata.payloadSHA256,
        actual: actual
      )
    }
  }

  static func validateMetadata(
    _ metadata: Metal4DSTEMResidentCacheMetadata,
    requireSealedPayload: Bool
  ) throws {
    func invalid(_ reason: String) throws -> Never {
      throw Metal4DSTEMResidentCacheError.invalidMetadata(reason)
    }
    guard metadata.formatVersion == Metal4DSTEMResidentCacheMetadata.currentFormatVersion else {
      try invalid("format version \(metadata.formatVersion) is not supported")
    }
    guard !metadata.datasetID.isEmpty else { try invalid("datasetID is empty") }
    guard !metadata.sources.isEmpty else { try invalid("sources is empty") }
    guard metadata.sourceScanRows > 0, metadata.sourceScanColumns > 0,
      metadata.sourceDetectorRows > 0, metadata.sourceDetectorColumns > 0
    else { try invalid("source shapes must be positive") }
    guard metadata.scanRowStart >= 0,
      metadata.scanRowStop > metadata.scanRowStart,
      metadata.scanRowStop <= metadata.sourceScanRows,
      metadata.scanColumnStart >= 0,
      metadata.scanColumnStop > metadata.scanColumnStart,
      metadata.scanColumnStop <= metadata.sourceScanColumns
    else {
      try invalid(
        "scan region [\(metadata.scanRowStart), \(metadata.scanRowStop)) × "
          + "[\(metadata.scanColumnStart), \(metadata.scanColumnStop)) is outside "
          + "\(metadata.sourceScanRows)×\(metadata.sourceScanColumns)"
      )
    }
    guard metadata.scanBin > 0, metadata.detectorBin > 0 else {
      try invalid(
        "scanBin and detectorBin must be positive; received "
          + "\(metadata.scanBin) and \(metadata.detectorBin)"
      )
    }
    let selectedRows = metadata.scanRowStop - metadata.scanRowStart
    let selectedColumns = metadata.scanColumnStop - metadata.scanColumnStart
    let expectedScanRows = (selectedRows - 1) / metadata.scanBin + 1
    let expectedScanColumns = (selectedColumns - 1) / metadata.scanBin + 1
    let expectedDetectorRows = (metadata.sourceDetectorRows - 1) / metadata.detectorBin + 1
    let expectedDetectorColumns =
      (metadata.sourceDetectorColumns - 1) / metadata.detectorBin + 1
    guard metadata.outputScanRows == expectedScanRows,
      metadata.outputScanColumns == expectedScanColumns,
      metadata.outputDetectorRows == expectedDetectorRows,
      metadata.outputDetectorColumns == expectedDetectorColumns
    else {
      try invalid(
        "output shape \(metadata.outputScanRows)×\(metadata.outputScanColumns)×"
          + "\(metadata.outputDetectorRows)×\(metadata.outputDetectorColumns) disagrees "
          + "with the recorded scan region and bin factors"
      )
    }
    guard
      let sourceDetectorPixels = checkedProduct(
        UInt64(metadata.sourceDetectorRows), UInt64(metadata.sourceDetectorColumns)
      )
    else { try invalid("source detector shape overflows UInt64") }
    let badPixels = Set(metadata.badPixelIndices)
    guard metadata.badPixelIndices == metadata.badPixelIndices.sorted(),
      badPixels.count == metadata.badPixelIndices.count,
      badPixels.allSatisfy({ $0 >= 0 && UInt64($0) < sourceDetectorPixels })
    else { try invalid("badPixelIndices contains duplicates or out-of-range values") }
    try validateValueRangeContract(
      metadata,
      selectedRows: selectedRows,
      selectedColumns: selectedColumns
    )
    try validateSamplingPropagation(
      metadata,
      selectedRows: selectedRows,
      selectedColumns: selectedColumns
    )
    guard
      let outputScanPositions = checkedProduct(
        UInt64(metadata.outputScanRows), UInt64(metadata.outputScanColumns)
      ),
      let outputDetectorPixels = checkedProduct(
        UInt64(metadata.outputDetectorRows), UInt64(metadata.outputDetectorColumns)
      )
    else { try invalid("output shape overflows UInt64") }
    let wordsPerFrame: UInt64
    switch metadata.outputDtype {
    case "uint8": wordsPerFrame = (outputDetectorPixels + 3) / 4
    case "uint16": wordsPerFrame = (outputDetectorPixels + 1) / 2
    default: wordsPerFrame = outputDetectorPixels
    }
    guard let payloadWords = checkedProduct(outputScanPositions, wordsPerFrame),
      let expectedPayloadBytes = checkedProduct(
        payloadWords, UInt64(MemoryLayout<UInt32>.stride)
      )
    else { try invalid("payload size overflows UInt64") }
    guard metadata.payloadBytes == expectedPayloadBytes else {
      try invalid(
        "payloadBytes \(metadata.payloadBytes) does not match the recorded shape and dtype "
          + "(expected \(expectedPayloadBytes))"
      )
    }
    if let sourceDigest = metadata.sourceIdentitySHA256,
      !isSHA256(sourceDigest)
    {
      try invalid("sourceIdentitySHA256 is not a lowercase SHA-256 digest")
    }
    guard
      !requireSealedPayload
        || (metadata.payloadIdentity?.byteCount == metadata.payloadBytes
          && isSHA256(metadata.payloadSHA256))
    else { try invalid("payload identity or SHA-256 seal is missing") }
  }

  private static func validateValueRangeContract(
    _ metadata: Metal4DSTEMResidentCacheMetadata,
    selectedRows: Int,
    selectedColumns: Int
  ) throws {
    func invalid(_ reason: String) throws -> Never {
      throw Metal4DSTEMResidentCacheError.invalidMetadata(reason)
    }
    guard metadata.sourceDtype == "uint8" || metadata.sourceDtype == "uint16" else {
      try invalid("source dtype \(metadata.sourceDtype) is not supported")
    }
    guard metadata.maxCount <= UInt32(UInt16.max),
      (metadata.maxCount > UInt32(UInt8.max)) == (metadata.pixelsAbove255 > 0),
      metadata.sourceDtype != "uint8"
        || (metadata.maxCount <= UInt32(UInt8.max) && metadata.pixelsAbove255 == 0)
    else {
      try invalid(
        "maxCount and pixelsAbove255 cannot describe \(metadata.sourceDtype) source values"
      )
    }
    guard ["uint8", "uint16", "uint32"].contains(metadata.outputDtype) else {
      try invalid("output dtype \(metadata.outputDtype) is not supported")
    }
    guard
      let maximumScanContributions = checkedProduct(
        UInt64(min(metadata.scanBin, selectedRows)),
        UInt64(min(metadata.scanBin, selectedColumns))
      ),
      let maximumDetectorContributions = checkedProduct(
        UInt64(min(metadata.detectorBin, metadata.sourceDetectorRows)),
        UInt64(min(metadata.detectorBin, metadata.sourceDetectorColumns))
      ),
      let maximumContributions = checkedProduct(
        maximumScanContributions, maximumDetectorContributions
      ),
      let maximumOutputCount = checkedProduct(
        UInt64(metadata.maxCount), maximumContributions
      )
    else { try invalid("exact output count bound overflows UInt64") }
    let outputMaximum: UInt64
    switch metadata.outputDtype {
    case "uint8": outputMaximum = UInt64(UInt8.max)
    case "uint16": outputMaximum = UInt64(UInt16.max)
    default: outputMaximum = UInt64(UInt32.max)
    }
    guard maximumOutputCount <= outputMaximum else {
      try invalid(
        "exact output bound \(maximumOutputCount) does not fit \(metadata.outputDtype)"
      )
    }
    let sourceTypeMaximum =
      metadata.sourceDtype == "uint8" ? UInt64(UInt8.max) : UInt64(UInt16.max)
    guard
      let typeMaximumOutputCount = checkedProduct(
        sourceTypeMaximum, maximumContributions
      )
    else { try invalid("source dtype output bound overflows UInt64") }
    let requiresAudit = typeMaximumOutputCount > outputMaximum

    if let audit = metadata.valueRangeAudit {
      do {
        try audit.validate()
        let digest = try audit.sha256()
        guard let sourceDigest = metadata.sourceIdentitySHA256,
          audit.sourceIdentitySHA256 == sourceDigest,
          audit.sourceDtype == metadata.sourceDtype,
          audit.badPixelIndices == metadata.badPixelIndices,
          audit.maximum == metadata.maxCount,
          audit.pixelsAbove255 == metadata.pixelsAbove255,
          digest == metadata.valueRangeAuditSHA256
        else {
          try invalid(
            "value-range audit does not match source identity, dtype, bad pixels, or counts"
          )
        }
      } catch let error as Metal4DSTEMResidentCacheError {
        throw error
      } catch {
        try invalid("value-range audit is invalid: \(error.localizedDescription)")
      }
    } else if metadata.valueRangeAuditSHA256 != nil {
      try invalid("valueRangeAuditSHA256 is present without its sealed audit fields")
    }
    if requiresAudit {
      guard let sourceDigest = metadata.sourceIdentitySHA256,
        isSHA256(sourceDigest),
        let auditDigest = metadata.valueRangeAuditSHA256,
        isSHA256(auditDigest),
        metadata.valueRangeAudit != nil
      else {
        try invalid(
          "narrow exact integer sums require source identity and a sealed value-range audit"
        )
      }
    }
  }

  private static func validateSamplingPropagation(
    _ metadata: Metal4DSTEMResidentCacheMetadata,
    selectedRows: Int,
    selectedColumns: Int
  ) throws {
    guard let sampling = metadata.samplingPropagation else { return }
    func invalid(_ reason: String) throws -> Never {
      throw Metal4DSTEMResidentCacheError.invalidMetadata(reason)
    }
    func valid(_ axis: Metal4DSTEMResidentAxisSampling) -> Bool {
      axis.row.isFinite && axis.row > 0
        && axis.column.isFinite && axis.column > 0
        && !axis.unit.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        && !axis.provenance.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        && !axis.evidence.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
    }
    func validateAxis(
      source: Metal4DSTEMResidentAxisSampling?,
      working: Metal4DSTEMResidentAxisSampling?,
      state: Metal4DSTEMResidentSamplingState,
      bin: Int,
      label: String
    ) throws {
      guard source.map(valid) ?? true, working.map(valid) ?? true else {
        try invalid("\(label) sampling contains a nonpositive or incomplete axis")
      }
      switch state {
      case .unavailable:
        guard source == nil, working == nil else {
          try invalid("unavailable \(label) sampling must not contain axis values")
        }
      case .unchanged:
        guard bin == 1, let source, source == working else {
          try invalid("unchanged \(label) sampling must match source values at bin 1")
        }
      case .uniformlyScaled:
        guard bin > 1, let source, let working,
          working.row == source.row * Double(bin),
          working.column == source.column * Double(bin),
          working.unit == source.unit,
          working.provenance == source.provenance,
          working.evidence == source.evidence
        else {
          try invalid("uniform \(label) sampling does not match bin \(bin)")
        }
      case .nonuniformEdgeBins:
        guard source != nil, working == nil else {
          try invalid("nonuniform-edge \(label) sampling must retain only source values")
        }
      }
    }
    try validateAxis(
      source: sampling.sourceScan,
      working: sampling.workingScan,
      state: sampling.scanState,
      bin: metadata.scanBin,
      label: "scan"
    )
    try validateAxis(
      source: sampling.sourceDetector,
      working: sampling.workingDetector,
      state: sampling.detectorState,
      bin: metadata.detectorBin,
      label: "detector"
    )
    let expectedScanRowCenter =
      Double(metadata.scanRowStart)
      + Double(min(metadata.scanBin, selectedRows) - 1) / 2
    let expectedScanColumnCenter =
      Double(metadata.scanColumnStart)
      + Double(min(metadata.scanBin, selectedColumns) - 1) / 2
    let expectedDetectorRowCenter =
      Double(min(metadata.detectorBin, metadata.sourceDetectorRows) - 1) / 2
    let expectedDetectorColumnCenter =
      Double(min(metadata.detectorBin, metadata.sourceDetectorColumns) - 1) / 2
    guard sampling.scanRegionRowStartInSourcePixels == metadata.scanRowStart,
      sampling.scanRegionColumnStartInSourcePixels == metadata.scanColumnStart,
      sampling.firstWorkingScanCenterRowInSourcePixels == expectedScanRowCenter,
      sampling.firstWorkingScanCenterColumnInSourcePixels == expectedScanColumnCenter,
      sampling.firstWorkingDetectorCenterRowInSourcePixels == expectedDetectorRowCenter,
      sampling.firstWorkingDetectorCenterColumnInSourcePixels
        == expectedDetectorColumnCenter
    else {
      try invalid("sampling origins or first-bin centroids disagree with geometry")
    }
  }

  private static func checkedProduct(_ lhs: UInt64, _ rhs: UInt64) -> UInt64? {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    return result.overflow ? nil : result.partialValue
  }

  private static func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64
      && value.utf8.allSatisfy {
        (48...57).contains($0) || (97...102).contains($0)
      }
  }
}
