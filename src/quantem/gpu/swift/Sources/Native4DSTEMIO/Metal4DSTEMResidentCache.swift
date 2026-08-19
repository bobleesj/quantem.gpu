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

/// Scientific and storage provenance for an exact detector-word-major cache.
public struct Metal4DSTEMResidentCacheMetadata: Codable, Equatable, Sendable {
  public static let currentFormatVersion = 1

  public let formatVersion: Int
  public let datasetID: String
  public let sourceIdentitySHA256: String?
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
  public let payloadBytes: UInt64
  public let payloadSHA256: String

  public init(
    datasetID: String,
    sourceIdentitySHA256: String? = nil,
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
    payloadBytes: UInt64,
    payloadSHA256: String = ""
  ) {
    formatVersion = Self.currentFormatVersion
    self.datasetID = datasetID
    self.sourceIdentitySHA256 = sourceIdentitySHA256
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

/// Read and write exact resident-cache payloads without materializing a copy.
public enum Metal4DSTEMResidentCacheIO {
  public static func write(
    pointer: UnsafeRawPointer,
    length: Int,
    payloadURL: URL,
    metadataURL: URL,
    metadata: Metal4DSTEMResidentCacheMetadata
  ) throws -> Metal4DSTEMResidentCacheMetadata {
    guard !FileManager.default.fileExists(atPath: payloadURL.path),
      !FileManager.default.fileExists(atPath: metadataURL.path)
    else {
      throw Metal4DSTEMResidentCacheError.destinationExists(payloadURL.path)
    }
    try validateMetadata(metadata, requireSealedPayload: false)
    guard length >= 0, metadata.payloadBytes == UInt64(length) else {
      throw Metal4DSTEMResidentCacheError.invalidMetadata(
        "payloadBytes \(metadata.payloadBytes) does not match buffer length \(length)"
      )
    }

    let temporaryPayload = payloadURL.deletingLastPathComponent().appendingPathComponent(
      ".\(payloadURL.lastPathComponent).\(UUID().uuidString).tmp"
    )
    var descriptor = open(
      temporaryPayload.path,
      O_WRONLY | O_CREAT | O_EXCL,
      S_IRUSR | S_IWUSR
    )
    guard descriptor >= 0 else {
      throw Metal4DSTEMResidentCacheError.cannotCreatePayload(payloadURL.path)
    }
    defer {
      if descriptor >= 0 { close(descriptor) }
      try? FileManager.default.removeItem(at: temporaryPayload)
    }
    var offset = 0
    while offset < length {
      let count = min(16 * 1024 * 1024, length - offset)
      let written = Darwin.write(descriptor, pointer.advanced(by: offset), count)
      guard written > 0 else {
        throw Metal4DSTEMResidentCacheError.cannotWritePayload(payloadURL.path)
      }
      offset += written
    }
    guard fsync(descriptor) == 0 else {
      throw Metal4DSTEMResidentCacheError.cannotWritePayload(payloadURL.path)
    }
    guard close(descriptor) == 0 else {
      descriptor = -1
      throw Metal4DSTEMResidentCacheError.cannotWritePayload(payloadURL.path)
    }
    descriptor = -1
    try FileManager.default.moveItem(at: temporaryPayload, to: payloadURL)
    do {
      let digest = sha256(pointer: pointer, length: length)
      let identity = try Metal4DSTEMSourceIdentity(url: payloadURL)
      let complete = metadata.withPayloadSHA256(digest, identity: identity)
      try validateMetadata(complete, requireSealedPayload: true)
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      try encoder.encode(complete).write(to: metadataURL, options: .atomic)
      return complete
    } catch {
      try? FileManager.default.removeItem(at: payloadURL)
      throw error
    }
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

  private static func sha256(pointer: UnsafeRawPointer, length: Int) -> String {
    let data = Data(
      bytesNoCopy: UnsafeMutableRawPointer(mutating: pointer),
      count: length,
      deallocator: .none
    )
    return SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
  }

  private static func validateMetadata(
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
    guard metadata.outputDtype == "uint16" || metadata.outputDtype == "uint32" else {
      try invalid("output dtype \(metadata.outputDtype) is not supported")
    }
    guard let sourceDetectorPixels = checkedProduct(
      UInt64(metadata.sourceDetectorRows), UInt64(metadata.sourceDetectorColumns)
    ) else { try invalid("source detector shape overflows UInt64") }
    let badPixels = Set(metadata.badPixelIndices)
    guard badPixels.count == metadata.badPixelIndices.count,
      badPixels.allSatisfy({ $0 >= 0 && UInt64($0) < sourceDetectorPixels })
    else { try invalid("badPixelIndices contains duplicates or out-of-range values") }
    guard let outputScanPositions = checkedProduct(
      UInt64(metadata.outputScanRows), UInt64(metadata.outputScanColumns)
    ), let outputDetectorPixels = checkedProduct(
      UInt64(metadata.outputDetectorRows), UInt64(metadata.outputDetectorColumns)
    ) else { try invalid("output shape overflows UInt64") }
    let wordsPerFrame = metadata.outputDtype == "uint16"
      ? (outputDetectorPixels + 1) / 2
      : outputDetectorPixels
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
    guard !requireSealedPayload
      || (metadata.payloadIdentity?.byteCount == metadata.payloadBytes
        && isSHA256(metadata.payloadSHA256))
    else { try invalid("payload identity or SHA-256 seal is missing") }
  }

  private static func checkedProduct(_ lhs: UInt64, _ rhs: UInt64) -> UInt64? {
    let result = lhs.multipliedReportingOverflow(by: rhs)
    return result.overflow ? nil : result.partialValue
  }

  private static func isSHA256(_ value: String) -> Bool {
    value.utf8.count == 64 && value.utf8.allSatisfy {
      (48...57).contains($0) || (97...102).contains($0)
    }
  }
}
