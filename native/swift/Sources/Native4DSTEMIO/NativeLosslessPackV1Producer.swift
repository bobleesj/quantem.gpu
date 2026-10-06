import CNativeHDF5
import CryptoKit
import Darwin
import Foundation

/// One immutable member of the original HDF5 source family.
public struct NativeLosslessPackV1SourceMember: Codable, Equatable, Sendable {
  public let role: String
  public let path: String
  public let bytes: UInt64
  public let sha256: String
}

/// Source calibration fields retained in the lossless-pack receipt.
public struct NativeLosslessPackV1CalibrationIdentity: Codable, Equatable, Sendable {
  public let scanRowSamplingNanometer: Double?
  public let scanColumnSamplingNanometer: Double?
  public let detectorRowSampling: Double?
  public let detectorColumnSampling: Double?
  public let detectorSamplingUnit: String?
  public let sha256: String
}

/// Small, allocation-free scientific identity used to plan one exact cache.
public struct NativeLosslessPackV1SourceInspection: Sendable {
  public let dataset: Native4DSTEMDataset
  public let sourceMembers: [NativeLosslessPackV1SourceMember]
  public let sourceIdentitySHA256: String
  public let sourceShape: [Int]
  public let sourceDtype: String
  public let sourceLogicalBytes: UInt64
  public let preparedIndexBytes: UInt64
  public let maximumCompressedBlockBytes: UInt64
  public let badPixelIndices: [Int]
  public let badPixelIdentitySHA256: String
  public let detectorMaskIdentitySHA256: String
  public let detectorMaskIdentityOrigin: String
  public let calibration: NativeLosslessPackV1CalibrationIdentity
}

/// Requested packing implementation. Unsupported backends fail rather than fall back.
public enum NativeLosslessPackV1ExecutionBackend: String, Codable, Equatable, Sendable {
  case cpuReference = "cpu-reference"
  case metal
  case cuda
  case vulkan
  case webGPU = "webgpu"

  public var isGPUAccelerated: Bool { self != .cpuReference }
}

/// Exact storage profile selected within Lossless Pack Format v1.
public enum NativeLosslessPackV1EncodingProfile: String, Codable, Equatable, Sendable {
  case exactUInt16LZ4 = "exact-uint16-lz4"
  case exactUInt8Bitpacked = "exact-uint8-bitpacked"
}

/// A fail-closed resource plan created before payload allocation.
public struct NativeLosslessPackV1ProductionPlan: Sendable {
  public let inspection: NativeLosslessPackV1SourceInspection
  public let destination: URL
  public let receiptDestination: URL
  public let encodingProfile: NativeLosslessPackV1EncodingProfile
  public let executionBackend: NativeLosslessPackV1ExecutionBackend
  public let scanTile: Int
  public let scansPerShard: Int
  public let shardCount: Int
  public let maximumTransientBytes: UInt64
  public let availableOutputDiskBytes: UInt64
  public let predictedPackedResidentMaximumBytes: UInt64
  public let predictedPeakTransientBytes: UInt64
  public let predictedOutputFileMaximumBytes: UInt64
  public let predictedReceiptFileMaximumBytes: UInt64
  public let predictedOutputDiskMaximumBytes: UInt64
}

/// Exact observed provenance for one published Lossless Pack Format v1 shard.
public struct NativeLosslessPackV1ShardReceipt: Codable, Equatable, Sendable {
  public let ordinal: Int
  public let firstScan: Int
  public let scanCount: Int
  public let payloadBytes: UInt64
  public let payloadSHA256: String
  public let headerBytes: UInt64
  public let headerSHA256: String
  public let maximumBitWidth: Int
}

public struct NativeLosslessPackV1ProductionTiming: Codable, Equatable, Sendable {
  public let sourceAuthenticationMilliseconds: Double
  public let decodeAndPackingMilliseconds: Double
  public let containerWriteMilliseconds: Double
  public let stabilityAndOutputHashMilliseconds: Double
  public let totalBeforeAtomicPublicationMilliseconds: Double
}

/// Versioned receipt for one atomically published exact Lossless Pack Format v1 cache.
public struct NativeLosslessPackV1ProductionReceipt: Codable, Equatable, Sendable {
  public static let currentSchema = "quantem.gpu.lossless-pack-production/v1"
  public static let formatSchema = "quantem.gpu.lossless-pack-format/v1"

  public let schema: String
  public let formatSchema: String
  public let encodingProfile: NativeLosslessPackV1EncodingProfile
  public let completedUTC: String
  public let destination: String
  public let outputBytes: UInt64
  public let outputSHA256: String
  public let sourceMembers: [NativeLosslessPackV1SourceMember]
  public let sourceIdentitySHA256: String
  public let sourceRawLogicalSHA256: String
  public let workingLogicalSHA256: String
  public let sourceShape: [Int]
  public let sourceDtype: String
  public let workingDtype: String
  public let preparedIndexBytes: UInt64
  public let maximumCompressedBlockBytes: UInt64
  public let executionBackend: NativeLosslessPackV1ExecutionBackend
  public let gpuAccelerated: Bool
  public let scanTile: Int
  public let scansPerShard: Int
  public let shardCount: Int
  public let scanBin: Int
  public let detectorBin: Int
  public let crop: String?
  public let badPixelIndices: [Int]
  public let badPixelIdentitySHA256: String
  public let detectorMaskIdentitySHA256: String
  public let detectorMaskIdentityOrigin: String
  public let badPixelRawValues: [UInt16]
  public let calibration: NativeLosslessPackV1CalibrationIdentity
  public let predictedPackedResidentMaximumBytes: UInt64
  public let observedPackedResidentBytes: UInt64
  public let predictedPeakTransientBytes: UInt64
  public let accountedPeakTransientBytes: UInt64
  public let predictedOutputFileMaximumBytes: UInt64
  public let predictedReceiptFileMaximumBytes: UInt64
  public let predictedOutputDiskMaximumBytes: UInt64
  public let timing: NativeLosslessPackV1ProductionTiming
  public let shards: [NativeLosslessPackV1ShardReceipt]
}

public enum NativeLosslessPackV1ProducerError: LocalizedError, Equatable {
  case cancelled
  case invalidRequest(String)
  case insufficientMemory(required: UInt64, available: UInt64)
  case insufficientDisk(required: UInt64, available: UInt64)
  case sourceChanged(String)
  case unsupportedValue(scan: Int, detectorPixel: Int, value: UInt16)
  case nonconstantBadPixel(detectorPixel: Int)
  case destinationExists(String)
  case nativeWriter(String)
  case unavailableExecutionBackend(NativeLosslessPackV1ExecutionBackend)

  public var errorDescription: String? {
    switch self {
    case .cancelled:
      return "Lossless Pack Format v1 production was cancelled; no cache was published."
    case .invalidRequest(let message), .sourceChanged(let message),
      .nativeWriter(let message):
      return message
    case .insufficientMemory(let required, let available):
      return
        "Lossless Pack Format v1 production needs a \(required)-byte transient budget; only \(available) bytes were admitted."
    case .insufficientDisk(let required, let available):
      return
        "Lossless Pack Format v1 atomic production needs \(required) free output bytes; only \(available) bytes are available."
    case .unsupportedValue(let scan, let detectorPixel, let value):
      return
        "The exact-uint8/bitpacked profile cannot represent source value \(value) at scan \(scan), detector pixel \(detectorPixel) without precision loss. Use the Lossless Pack Format v1 exact-uint16/LZ4 profile."
    case .nonconstantBadPixel(let detectorPixel):
      return
        "Excluded detector pixel \(detectorPixel) is not constant across the source, so Lossless Pack Format v1 cannot restore its exact raw counts."
    case .destinationExists(let path):
      return
        "Lossless Pack Format v1 output already exists at \(path); choose a new destination or validate the existing receipt."
    case .unavailableExecutionBackend(let backend):
      return
        "The requested Lossless Pack Format v1 \(backend.rawValue) producer is not implemented in this revision; no CPU fallback was used."
    }
  }
}

/// Produces a native, exact Lossless Pack Format v1 cache from an original indexed HDF5 family.
///
/// The lifecycle is deliberately explicit. `inspect(input:)` authenticates the
/// source and creates only small QH5 indexes. `plan(...)` proves aggregate
/// transient-memory and output-disk admission. `produce(...)` then performs
/// bounded CPU decode and lossless packing without allocating the logical 4D
/// tensor. The destination appears only after the complete file and receipt are
/// synchronized and linked into place.
public struct NativeLosslessPackV1Producer: Sendable {
  public static let scanTile = 32
  public static let targetScansPerShard = 4096
  public static let userBlockBytes = 64 * 1024

  public let cacheDirectory: URL

  public init(cacheDirectory: URL) {
    self.cacheDirectory = cacheDirectory
  }

  /// Authenticate one ordinary HDF5 family without allocating its logical tensor.
  public func inspect(input: URL) throws -> NativeLosslessPackV1SourceInspection {
    let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: cacheDirectory)
      .prepare(input: input, mode: .indexed)
    guard catalog.datasets.count == 1, let dataset = catalog.datasets.first else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 production requires one unambiguous HDF5 dataset; inspect each catalog entry separately."
      )
    }
    return try inspect(dataset: dataset)
  }

  /// Authenticate one dataset already selected from a native catalog.
  public func inspect(
    dataset: Native4DSTEMDataset
  ) throws -> NativeLosslessPackV1SourceInspection {
    guard dataset.sourceDtype == "uint16" else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 native production currently requires an original uint16 HDF5 source."
      )
    }
    guard dataset.scanRows > 0, dataset.scanCols > 0,
      dataset.detectorRows > 0, dataset.detectorCols > 0
    else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 native production requires positive scan and detector shapes."
      )
    }
    guard let sourceIdentity = dataset.sourceIdentitySHA256,
      let memberHashes = dataset.orderedMemberSHA256,
      memberHashes.count == dataset.dataFiles.count
    else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "The HDF5 catalog is missing exact source hashes; rerun indexed inspection."
      )
    }
    guard (dataset.masterPath == nil) == (dataset.masterSHA256 == nil) else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "The HDF5 catalog has an incomplete master-file identity; rerun indexed inspection."
      )
    }
    let scanCount = try checkedProduct(
      UInt64(dataset.scanRows), UInt64(dataset.scanCols), label: "scan count"
    )
    guard scanCount.isMultiple(of: UInt64(Self.scanTile)) else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 requires complete \(Self.scanTile)-scan tiles; source scan shape is \(dataset.scanRows)×\(dataset.scanCols)."
      )
    }
    let detectorPixels = try checkedProduct(
      UInt64(dataset.detectorRows), UInt64(dataset.detectorCols), label: "detector pixels"
    )
    guard let detectorPixelCount = Int(exactly: detectorPixels) else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "The detector shape exceeds this process's addressable range."
      )
    }
    let logicalValues = try checkedProduct(scanCount, detectorPixels, label: "logical values")
    let logicalBytes = try checkedProduct(logicalValues, 2, label: "logical source bytes")
    let badPixels = dataset.badPixelIndices.sorted()
    guard Set(badPixels).count == badPixels.count,
      badPixels.allSatisfy({ $0 >= 0 && UInt64($0) < detectorPixels }),
      badPixels.count < detectorPixelCount
    else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "The source bad-pixel list is duplicated, out of range, or excludes the complete detector."
      )
    }
    let indexed = try Native4DSTEMIndexedSource.open(dataset: dataset)
    var preparedIndexBytes: UInt64 = 0
    var maximumCompressedBlockBytes: UInt64 = 0
    for (indexPath, shard) in zip(dataset.indexFiles, indexed.shards) {
      preparedIndexBytes = try checkedSum(
        preparedIndexBytes,
        try nativeFileIdentity(for: URL(fileURLWithPath: indexPath)).bytes,
        label: "prepared index bytes"
      )
      for word in stride(from: 1, to: shard.index.metadataWords.count, by: 2) {
        maximumCompressedBlockBytes = max(
          maximumCompressedBlockBytes,
          UInt64(shard.index.metadataWords[word])
        )
      }
    }
    guard maximumCompressedBlockBytes > 0 else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "The prepared QH5 indexes contain no compressed detector blocks."
      )
    }

    var members: [NativeLosslessPackV1SourceMember] = []
    if let masterPath = dataset.masterPath, let masterHash = dataset.masterSHA256 {
      let url = URL(fileURLWithPath: masterPath)
      members.append(
        NativeLosslessPackV1SourceMember(
          role: "master",
          path: nativeCanonicalURL(url).path,
          bytes: try nativeFileIdentity(for: url).bytes,
          sha256: masterHash
        )
      )
    }
    for (path, hash) in zip(dataset.dataFiles, memberHashes) {
      let url = URL(fileURLWithPath: path)
      members.append(
        NativeLosslessPackV1SourceMember(
          role: "data",
          path: nativeCanonicalURL(url).path,
          bytes: try nativeFileIdentity(for: url).bytes,
          sha256: hash
        )
      )
    }
    let calibration = try calibrationIdentity(for: dataset)
    let detectorMaskIdentitySHA256: String
    let detectorMaskIdentityOrigin: String
    if let sourceMaskSHA256 = dataset.detectorMaskSHA256 {
      _ = try dataFromSHA256(sourceMaskSHA256)
      detectorMaskIdentitySHA256 = sourceMaskSHA256
      detectorMaskIdentityOrigin = "source-little-endian-u32-mask"
    } else {
      guard badPixels.isEmpty else {
        throw NativeLosslessPackV1ProducerError.invalidRequest(
          "The source has excluded detector pixels but no full pixel-mask identity; rebuild its indexed catalog."
        )
      }
      detectorMaskIdentitySHA256 = zeroDetectorMaskSHA256(pixelCount: detectorPixels)
      detectorMaskIdentityOrigin = "implicit-all-admitted-u32-mask"
    }
    return NativeLosslessPackV1SourceInspection(
      dataset: dataset,
      sourceMembers: members,
      sourceIdentitySHA256: sourceIdentity,
      sourceShape: [
        dataset.scanRows, dataset.scanCols, dataset.detectorRows, dataset.detectorCols,
      ],
      sourceDtype: dataset.sourceDtype,
      sourceLogicalBytes: logicalBytes,
      preparedIndexBytes: preparedIndexBytes,
      maximumCompressedBlockBytes: maximumCompressedBlockBytes,
      badPixelIndices: badPixels,
      badPixelIdentitySHA256: orderedPixelSHA256(badPixels),
      detectorMaskIdentitySHA256: detectorMaskIdentitySHA256,
      detectorMaskIdentityOrigin: detectorMaskIdentityOrigin,
      calibration: calibration
    )
  }

  /// Admit an exact cache before allocating decoded or packed payloads.
  public func plan(
    inspection: NativeLosslessPackV1SourceInspection,
    destination: URL,
    maximumTransientBytes: UInt64,
    availableOutputDiskBytes: UInt64? = nil,
    executionBackend: NativeLosslessPackV1ExecutionBackend = .cpuReference
  ) throws -> NativeLosslessPackV1ProductionPlan {
    guard executionBackend == .cpuReference else {
      throw NativeLosslessPackV1ProducerError.unavailableExecutionBackend(executionBackend)
    }
    let shape = inspection.sourceShape
    let datasetShape = [
      inspection.dataset.scanRows,
      inspection.dataset.scanCols,
      inspection.dataset.detectorRows,
      inspection.dataset.detectorCols,
    ]
    let expectedCalibration = try calibrationIdentity(for: inspection.dataset)
    var expectedMembers: [(role: String, path: String, sha256: String)] = []
    if let masterPath = inspection.dataset.masterPath,
      let masterSHA256 = inspection.dataset.masterSHA256
    {
      expectedMembers.append(
        ("master", nativeCanonicalURL(URL(fileURLWithPath: masterPath)).path, masterSHA256))
    }
    if let hashes = inspection.dataset.orderedMemberSHA256 {
      expectedMembers.append(
        contentsOf: zip(inspection.dataset.dataFiles, hashes).map {
          (
            role: "data",
            path: nativeCanonicalURL(URL(fileURLWithPath: $0.0)).path,
            sha256: $0.1
          )
        }
      )
    }
    let membersMatch = zip(inspection.sourceMembers, expectedMembers).allSatisfy {
      $0.role == $1.role && $0.path == $1.path && $0.sha256 == $1.sha256
    }
    guard shape.count == 4, shape.allSatisfy({ $0 > 0 }),
      inspection.sourceDtype == "uint16",
      inspection.sourceDtype == inspection.dataset.sourceDtype,
      shape == datasetShape,
      inspection.badPixelIndices == inspection.dataset.badPixelIndices.sorted(),
      inspection.sourceIdentitySHA256 == inspection.dataset.sourceIdentitySHA256,
      inspection.badPixelIdentitySHA256 == orderedPixelSHA256(inspection.badPixelIndices),
      inspection.calibration == expectedCalibration,
      inspection.sourceMembers.count == expectedMembers.count,
      membersMatch
    else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 planning requires a self-consistent inspected uint16 source in (scan row, scan column, detector row, detector column) order."
      )
    }
    let scanCount = try checkedProduct(UInt64(shape[0]), UInt64(shape[1]), label: "scan count")
    let detectorPixels = try checkedProduct(
      UInt64(shape[2]), UInt64(shape[3]), label: "detector pixels"
    )
    let expectedDetectorMaskIdentity: (sha256: String, origin: String)
    if let sourceMaskSHA256 = inspection.dataset.detectorMaskSHA256 {
      expectedDetectorMaskIdentity = (sourceMaskSHA256, "source-little-endian-u32-mask")
    } else {
      expectedDetectorMaskIdentity = (
        zeroDetectorMaskSHA256(pixelCount: detectorPixels),
        "implicit-all-admitted-u32-mask"
      )
    }
    guard inspection.detectorMaskIdentitySHA256 == expectedDetectorMaskIdentity.sha256,
      inspection.detectorMaskIdentityOrigin == expectedDetectorMaskIdentity.origin
    else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 detector-mask identity does not match the inspected source."
      )
    }
    guard Int(exactly: scanCount) != nil, Int(exactly: detectorPixels) != nil else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 source geometry exceeds this process's addressable range."
      )
    }
    let logicalBytes = try checkedProduct(
      try checkedProduct(scanCount, detectorPixels, label: "logical values"),
      2,
      label: "logical source bytes"
    )
    guard inspection.sourceLogicalBytes == logicalBytes else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 inspected logical bytes do not match the source shape and dtype."
      )
    }
    _ = try dataFromSHA256(inspection.sourceIdentitySHA256)
    let scansPerShard = try shardScanCount(scanCount: scanCount)
    guard let shardCount = Int(exactly: scanCount / UInt64(scansPerShard)) else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 shard count exceeds this process's addressable range."
      )
    }
    let tilesPerShard = scansPerShard / Self.scanTile
    let checkpointWords = (tilesPerShard + 31) / 32
    let widthWords = (tilesPerShard + 7) / 8
    let headerWordsPerPixel = checkpointWords + widthWords
    let headerBytesPerShard = try checkedProduct(
      try checkedProduct(detectorPixels, UInt64(headerWordsPerPixel), label: "header words"),
      4,
      label: "header bytes"
    )
    let admittedPixels = detectorPixels - UInt64(inspection.badPixelIndices.count)
    let payloadBytesPerShard = try checkedProduct(
      admittedPixels, UInt64(scansPerShard), label: "maximum payload bytes"
    )
    let residentMaximum = try checkedProduct(
      try checkedSum(payloadBytesPerShard, headerBytesPerShard, label: "shard resident bytes"),
      UInt64(shardCount),
      label: "packed resident bytes"
    )
    let decodedTileBytes = try checkedProduct(
      try checkedProduct(detectorPixels, UInt64(Self.scanTile), label: "decoded tile values"),
      2,
      label: "decoded tile bytes"
    )
    let workingTileBytes = try checkedProduct(
      detectorPixels, UInt64(Self.scanTile), label: "working tile bytes"
    )
    let widthBytes = try checkedProduct(
      detectorPixels, UInt64(tilesPerShard), label: "width bytes"
    )
    let offsetBytes = try checkedProduct(widthBytes, 4, label: "payload offset bytes")
    let decoderScratchBytes = try checkedSum(
      inspection.maximumCompressedBlockBytes,
      8192,
      label: "compressed and decoded block scratch"
    )
    let predictedTransient = try checkedSum(
      try checkedSum(
        try checkedSum(widthBytes, offsetBytes, label: "packing metadata allocation"),
        try checkedSum(
          try checkedSum(
            payloadBytesPerShard, headerBytesPerShard, label: "packed shard allocation"),
          try checkedSum(decodedTileBytes, workingTileBytes, label: "decoded tile allocation"),
          label: "tile and packed shard allocation"
        ),
        label: "packing buffers"
      ),
      try checkedSum(
        inspection.preparedIndexBytes,
        decoderScratchBytes,
        label: "index and decoder scratch"
      ),
      label: "peak transient bytes"
    )
    guard maximumTransientBytes >= predictedTransient else {
      throw NativeLosslessPackV1ProducerError.insufficientMemory(
        required: predictedTransient,
        available: maximumTransientBytes
      )
    }
    let hdf5Reserve = try checkedSum(
      4 * 1024 * 1024,
      try checkedProduct(UInt64(shardCount), 64 * 1024, label: "HDF5 shard reserve"),
      label: "HDF5 metadata reserve"
    )
    let outputMaximum = try checkedSum(
      UInt64(Self.userBlockBytes),
      try checkedSum(residentMaximum, hdf5Reserve, label: "container body"),
      label: "output file maximum"
    )
    let receiptMaximum = try checkedSum(
      try checkedSum(
        1024 * 1024,
        try checkedProduct(UInt64(shardCount), 1024, label: "receipt shard reserve"),
        label: "receipt structure reserve"
      ),
      try checkedSum(
        try checkedProduct(
          UInt64(inspection.sourceMembers.count),
          1024,
          label: "receipt source-member reserve"
        ),
        UInt64(inspection.sourceMembers.reduce(0) { $0 + $1.path.utf8.count }),
        label: "receipt source-path reserve"
      ),
      label: "receipt file reserve"
    )
    let outputDiskMaximum = try checkedSum(
      outputMaximum,
      receiptMaximum,
      label: "atomic output and receipt disk bytes"
    )
    let available = try availableOutputDiskBytes ?? outputCapacity(at: destination)
    guard available >= outputDiskMaximum else {
      throw NativeLosslessPackV1ProducerError.insufficientDisk(
        required: outputDiskMaximum,
        available: available
      )
    }
    let finalDestination = nativeCanonicalURL(destination)
    let receipt = finalDestination.appendingPathExtension("receipt.json")
    let plan = NativeLosslessPackV1ProductionPlan(
      inspection: inspection,
      destination: finalDestination,
      receiptDestination: receipt,
      encodingProfile: .exactUInt8Bitpacked,
      executionBackend: executionBackend,
      scanTile: Self.scanTile,
      scansPerShard: scansPerShard,
      shardCount: shardCount,
      maximumTransientBytes: maximumTransientBytes,
      availableOutputDiskBytes: available,
      predictedPackedResidentMaximumBytes: residentMaximum,
      predictedPeakTransientBytes: predictedTransient,
      predictedOutputFileMaximumBytes: outputMaximum,
      predictedReceiptFileMaximumBytes: receiptMaximum,
      predictedOutputDiskMaximumBytes: outputDiskMaximum
    )
    let maximumManifest = try makeManifest(
      plan: plan,
      sourceRawSHA: String(repeating: "f", count: 64),
      workingSHA: String(repeating: "f", count: 64),
      badPixelRawValues: [UInt16](
        repeating: UInt16.max,
        count: inspection.badPixelIndices.count
      )
    )
    let binaryBytes = try checkedSum(
      try checkedSum(
        80,
        try checkedProduct(
          UInt64(inspection.badPixelIndices.count),
          4,
          label: "binary bad-pixel index bytes"
        ),
        label: "binary index prefix"
      ),
      try checkedProduct(UInt64(shardCount), 96, label: "binary shard records"),
      label: "binary index bytes"
    )
    let binaryOffset = UInt64((24 + maximumManifest.count + 7) & ~7)
    let userBlockUsage = try checkedSum(binaryOffset, binaryBytes, label: "user-block bytes")
    guard userBlockUsage <= UInt64(Self.userBlockBytes) else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 metadata needs at least \(userBlockUsage) user-block bytes; the portable limit is \(Self.userBlockBytes)."
      )
    }
    return plan
  }

  /// Build and atomically publish the exact cache and its versioned receipt.
  ///
  /// The returned destination and receipt are complete caller-owned files. This
  /// API does not return borrowed buffers or retain hidden runtime resources.
  public func produce(
    _ plan: NativeLosslessPackV1ProductionPlan,
    shouldCancel: @Sendable () -> Bool = { false }
  ) throws -> NativeLosslessPackV1ProductionReceipt {
    let productionStarted = ProcessInfo.processInfo.systemUptime
    try cancelled(shouldCancel)
    let fileManager = FileManager.default
    guard !fileManager.fileExists(atPath: plan.destination.path) else {
      throw NativeLosslessPackV1ProducerError.destinationExists(plan.destination.path)
    }
    guard !fileManager.fileExists(atPath: plan.receiptDestination.path) else {
      throw NativeLosslessPackV1ProducerError.destinationExists(plan.receiptDestination.path)
    }
    try fileManager.createDirectory(
      at: plan.destination.deletingLastPathComponent(),
      withIntermediateDirectories: true
    )
    let authenticationStarted = ProcessInfo.processInfo.systemUptime
    let beforeAuthentication = try plan.inspection.sourceMembers.map {
      try nativeFileIdentity(for: URL(fileURLWithPath: $0.path))
    }
    try verifySource(inspection: plan.inspection)
    let sourceSnapshots = try plan.inspection.sourceMembers.map {
      try nativeFileIdentity(for: URL(fileURLWithPath: $0.path))
    }
    guard zip(beforeAuthentication, sourceSnapshots).allSatisfy({ sameIdentity($0, $1) }) else {
      throw NativeLosslessPackV1ProducerError.sourceChanged(
        "An original HDF5 member changed while its SHA-256 identity was authenticated; inspect it again."
      )
    }
    let authenticationMilliseconds =
      (ProcessInfo.processInfo.systemUptime - authenticationStarted) * 1_000
    let indexed = try Native4DSTEMIndexedSource.open(dataset: plan.inspection.dataset)
    let decoder = try NativeQH5CPUDecoder(source: indexed)
    defer { decoder.close() }

    let token = UUID().uuidString
    let temporaryOutput = plan.destination.deletingLastPathComponent()
      .appendingPathComponent(".\(plan.destination.lastPathComponent).\(token).partial")
    let temporaryReceipt = plan.receiptDestination.deletingLastPathComponent()
      .appendingPathComponent(".\(plan.receiptDestination.lastPathComponent).\(token).partial")
    defer {
      try? fileManager.removeItem(at: temporaryOutput)
      try? fileManager.removeItem(at: temporaryReceipt)
    }

    var containerWriteMilliseconds = 0.0
    var decodeAndPackingMilliseconds = 0.0
    var writer: OpaquePointer?
    let writerOpenStarted = ProcessInfo.processInfo.systemUptime
    try openWriter(path: temporaryOutput, writer: &writer)
    containerWriteMilliseconds +=
      (ProcessInfo.processInfo.systemUptime - writerOpenStarted) * 1_000
    var writerNeedsAbort = true
    defer {
      if writerNeedsAbort, let writer { qh5_lossless_pack_v1_writer_abort(writer) }
    }

    let detectorPixels =
      plan.inspection.dataset.detectorRows
      * plan.inspection.dataset.detectorCols
    let excluded = Set(plan.inspection.badPixelIndices)
    var excludedRawValues = [UInt16?](repeating: nil, count: plan.inspection.badPixelIndices.count)
    let excludedOrdinals = Dictionary(
      uniqueKeysWithValues: plan.inspection.badPixelIndices.enumerated().map {
        ($0.element, $0.offset)
      }
    )
    var sourceDigest = SHA256()
    var workingDigest = SHA256()
    var layouts: [NativeLosslessPackV1NativeShardLayout] = []
    var shardReceipts: [NativeLosslessPackV1ShardReceipt] = []
    var observedResidentBytes: UInt64 = 0
    var accountedPeakTransientBytes: UInt64 = 0

    for ordinal in 0..<plan.shardCount {
      try cancelled(shouldCancel)
      let firstScan = ordinal * plan.scansPerShard
      let packingStarted = ProcessInfo.processInfo.systemUptime
      let packed = try packShard(
        decoder: decoder,
        firstScan: firstScan,
        scanCount: plan.scansPerShard,
        detectorPixels: detectorPixels,
        excluded: excluded,
        excludedOrdinals: excludedOrdinals,
        excludedRawValues: &excludedRawValues,
        sourceDigest: &sourceDigest,
        workingDigest: &workingDigest,
        accountedPersistentBytes: try checkedSum(
          plan.inspection.preparedIndexBytes,
          try checkedSum(
            plan.inspection.maximumCompressedBlockBytes,
            8192,
            label: "observed decoder scratch"
          ),
          label: "observed index and decoder scratch"
        ),
        shouldCancel: shouldCancel
      )
      decodeAndPackingMilliseconds +=
        (ProcessInfo.processInfo.systemUptime - packingStarted) * 1_000
      accountedPeakTransientBytes = max(accountedPeakTransientBytes, packed.transientBytes)
      let appendStarted = ProcessInfo.processInfo.systemUptime
      let layout = try appendShard(
        writer: writer,
        ordinal: ordinal,
        payload: packed.payload,
        headers: packed.headers
      )
      containerWriteMilliseconds +=
        (ProcessInfo.processInfo.systemUptime - appendStarted) * 1_000
      layouts.append(layout)
      let payloadBytes = UInt64(packed.payload.count * 4)
      let headerBytes = UInt64(packed.headers.count * 4)
      observedResidentBytes = try checkedSum(
        observedResidentBytes,
        try checkedSum(payloadBytes, headerBytes, label: "observed shard bytes"),
        label: "observed resident bytes"
      )
      shardReceipts.append(
        NativeLosslessPackV1ShardReceipt(
          ordinal: ordinal,
          firstScan: firstScan,
          scanCount: plan.scansPerShard,
          payloadBytes: payloadBytes,
          payloadSHA256: sha256(words: packed.payload),
          headerBytes: headerBytes,
          headerSHA256: sha256(words: packed.headers),
          maximumBitWidth: packed.maximumBitWidth
        )
      )
    }
    let closeStarted = ProcessInfo.processInfo.systemUptime
    try closeWriter(&writer)
    containerWriteMilliseconds +=
      (ProcessInfo.processInfo.systemUptime - closeStarted) * 1_000
    writerNeedsAbort = false

    let rawValues = try excludedRawValues.enumerated().map { ordinal, value in
      guard let value else {
        throw NativeLosslessPackV1ProducerError.nonconstantBadPixel(
          detectorPixel: plan.inspection.badPixelIndices[ordinal]
        )
      }
      return value
    }
    let sourceRawSHA = hex(sourceDigest.finalize())
    let workingSHA = hex(workingDigest.finalize())
    let manifest = try makeManifest(
      plan: plan,
      sourceRawSHA: sourceRawSHA,
      workingSHA: workingSHA,
      badPixelRawValues: rawValues
    )
    let binary = try makeBinaryIndex(plan: plan, layouts: layouts, receipts: shardReceipts)
    try writeUserBlock(manifest: manifest, binary: binary, to: temporaryOutput)
    try synchronize(temporaryOutput)

    let stabilityStarted = ProcessInfo.processInfo.systemUptime
    let afterSnapshots = try plan.inspection.sourceMembers.map {
      try nativeFileIdentity(for: URL(fileURLWithPath: $0.path))
    }
    guard zip(sourceSnapshots, afterSnapshots).allSatisfy({ sameIdentity($0, $1) }) else {
      throw NativeLosslessPackV1ProducerError.sourceChanged(
        "An original HDF5 member changed during Lossless Pack Format v1 production; no cache was published."
      )
    }
    let outputBytes = try nativeFileIdentity(for: temporaryOutput).bytes
    guard outputBytes <= plan.predictedOutputFileMaximumBytes else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Observed lossless-pack output \(outputBytes) bytes exceeded the admitted \(plan.predictedOutputFileMaximumBytes)-byte bound."
      )
    }
    let outputSHA = try sha256(file: temporaryOutput)
    let stabilityAndOutputHashMilliseconds =
      (ProcessInfo.processInfo.systemUptime - stabilityStarted) * 1_000
    let timing = NativeLosslessPackV1ProductionTiming(
      sourceAuthenticationMilliseconds: authenticationMilliseconds,
      decodeAndPackingMilliseconds: decodeAndPackingMilliseconds,
      containerWriteMilliseconds: containerWriteMilliseconds,
      stabilityAndOutputHashMilliseconds: stabilityAndOutputHashMilliseconds,
      totalBeforeAtomicPublicationMilliseconds: (ProcessInfo.processInfo.systemUptime
        - productionStarted) * 1_000
    )
    let receipt = NativeLosslessPackV1ProductionReceipt(
      schema: NativeLosslessPackV1ProductionReceipt.currentSchema,
      formatSchema: NativeLosslessPackV1ProductionReceipt.formatSchema,
      encodingProfile: plan.encodingProfile,
      completedUTC: ISO8601DateFormatter().string(from: Date()),
      destination: plan.destination.path,
      outputBytes: outputBytes,
      outputSHA256: outputSHA,
      sourceMembers: plan.inspection.sourceMembers,
      sourceIdentitySHA256: plan.inspection.sourceIdentitySHA256,
      sourceRawLogicalSHA256: sourceRawSHA,
      workingLogicalSHA256: workingSHA,
      sourceShape: plan.inspection.sourceShape,
      sourceDtype: "uint16",
      workingDtype: "uint8",
      preparedIndexBytes: plan.inspection.preparedIndexBytes,
      maximumCompressedBlockBytes: plan.inspection.maximumCompressedBlockBytes,
      executionBackend: plan.executionBackend,
      gpuAccelerated: plan.executionBackend.isGPUAccelerated,
      scanTile: plan.scanTile,
      scansPerShard: plan.scansPerShard,
      shardCount: plan.shardCount,
      scanBin: 1,
      detectorBin: 1,
      crop: nil,
      badPixelIndices: plan.inspection.badPixelIndices,
      badPixelIdentitySHA256: plan.inspection.badPixelIdentitySHA256,
      detectorMaskIdentitySHA256: plan.inspection.detectorMaskIdentitySHA256,
      detectorMaskIdentityOrigin: plan.inspection.detectorMaskIdentityOrigin,
      badPixelRawValues: rawValues,
      calibration: plan.inspection.calibration,
      predictedPackedResidentMaximumBytes: plan.predictedPackedResidentMaximumBytes,
      observedPackedResidentBytes: observedResidentBytes,
      predictedPeakTransientBytes: plan.predictedPeakTransientBytes,
      accountedPeakTransientBytes: accountedPeakTransientBytes,
      predictedOutputFileMaximumBytes: plan.predictedOutputFileMaximumBytes,
      predictedReceiptFileMaximumBytes: plan.predictedReceiptFileMaximumBytes,
      predictedOutputDiskMaximumBytes: plan.predictedOutputDiskMaximumBytes,
      timing: timing,
      shards: shardReceipts
    )
    let encoder = JSONEncoder()
    encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    var receiptData = try encoder.encode(receipt)
    receiptData.append(0x0a)
    guard UInt64(receiptData.count) <= plan.predictedReceiptFileMaximumBytes else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Observed lossless-pack receipt \(receiptData.count) bytes exceeded the admitted \(plan.predictedReceiptFileMaximumBytes)-byte bound."
      )
    }
    let observedOutputDiskBytes = try checkedSum(
      outputBytes,
      UInt64(receiptData.count),
      label: "observed output and receipt disk bytes"
    )
    guard observedOutputDiskBytes <= plan.predictedOutputDiskMaximumBytes else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Observed lossless-pack output and receipt \(observedOutputDiskBytes) bytes exceeded the admitted \(plan.predictedOutputDiskMaximumBytes)-byte bound."
      )
    }
    try receiptData.write(to: temporaryReceipt, options: .withoutOverwriting)
    try synchronize(temporaryReceipt)
    try publishAtomically(
      temporaryOutput: temporaryOutput,
      destination: plan.destination,
      temporaryReceipt: temporaryReceipt,
      receiptDestination: plan.receiptDestination
    )
    return receipt
  }
}

private struct NativeLosslessPackV1PackedShard {
  let payload: [UInt32]
  let headers: [UInt32]
  let maximumBitWidth: Int
  let transientBytes: UInt64
}

private struct NativeLosslessPackV1NativeShardLayout {
  let payloadOffset: UInt64
  let payloadBytes: UInt64
  let headersOffset: UInt64
  let headersBytes: UInt64
}

private final class NativeQH5CPUDecoder {
  let source: Native4DSTEMIndexedSource
  private var descriptors: [Int32]

  init(source: Native4DSTEMIndexedSource) throws {
    self.source = source
    var opened: [Int32] = []
    do {
      for shard in source.shards {
        let descriptor = Darwin.open(shard.sourceURL.path, O_RDONLY)
        guard descriptor >= 0 else {
          throw Native4DSTEMIOError.invalidData(
            "Could not open original HDF5 shard \(shard.sourceURL.path)"
          )
        }
        opened.append(descriptor)
      }
    } catch {
      for descriptor in opened { Darwin.close(descriptor) }
      throw error
    }
    descriptors = opened
  }

  func close() {
    for descriptor in descriptors where descriptor >= 0 { Darwin.close(descriptor) }
    descriptors = []
  }

  func readFrames(firstScan: Int, count: Int) throws -> [UInt16] {
    let detectorPixels = source.dataset.detectorRows * source.dataset.detectorCols
    var values = [UInt16](repeating: 0, count: count * detectorPixels)
    var shuffled = [UInt8](repeating: 0, count: 8192)
    for frameOffset in 0..<count {
      let globalFrame = firstScan + frameOffset
      guard
        let shardIndex = source.shards.firstIndex(where: {
          $0.globalFrameRange.contains(globalFrame)
        })
      else {
        throw Native4DSTEMIOError.invalidData("QH5 indexes do not cover scan \(globalFrame)")
      }
      let shard = source.shards[shardIndex]
      let localFrame = globalFrame - shard.globalFrameRange.lowerBound
      guard
        let chunk = shard.index.metadata.chunks.first(where: {
          $0.startFrame <= localFrame && localFrame < $0.startFrame + $0.nFrames
        })
      else {
        throw Native4DSTEMIOError.invalidData("QH5 index does not cover local scan \(localFrame)")
      }
      let blocks = shard.index.metadata.nBlocksPerFrame
      let firstWord = chunk.metaOffsetWords + (localFrame - chunk.startFrame) * blocks * 2
      for block in 0..<blocks {
        let word = firstWord + block * 2
        let relative = UInt64(shard.index.metadataWords[word])
        let compressedBytes = Int(shard.index.metadataWords[word + 1])
        let offset = chunk.rangeStart + relative
        var compressed = [UInt8](repeating: 0, count: compressedBytes)
        try compressed.withUnsafeMutableBytes {
          try readExact(
            descriptor: descriptors[shardIndex],
            offset: offset,
            buffer: $0,
            label: shard.sourceURL.lastPathComponent
          )
        }
        try lz4Decode(compressed, output: &shuffled)
        let destination = (frameOffset * detectorPixels) + block * 4096
        bitUnshuffleUInt16(shuffled, output: &values, destination: destination)
      }
    }
    return values
  }
}

extension NativeLosslessPackV1Producer {
  private func packShard(
    decoder: NativeQH5CPUDecoder,
    firstScan: Int,
    scanCount: Int,
    detectorPixels: Int,
    excluded: Set<Int>,
    excludedOrdinals: [Int: Int],
    excludedRawValues: inout [UInt16?],
    sourceDigest: inout SHA256,
    workingDigest: inout SHA256,
    accountedPersistentBytes: UInt64,
    shouldCancel: @Sendable () -> Bool
  ) throws -> NativeLosslessPackV1PackedShard {
    let tileCount = scanCount / Self.scanTile
    var widths = [UInt8](repeating: 0, count: detectorPixels * tileCount)
    var maximumBitWidth = 0
    for tile in 0..<tileCount {
      try cancelled(shouldCancel)
      let tileFirstScan = firstScan + tile * Self.scanTile
      let decoded = try decoder.readFrames(firstScan: tileFirstScan, count: Self.scanTile)
      decoded.withUnsafeBytes { sourceDigest.update(bufferPointer: $0) }
      var working = [UInt8](repeating: 0, count: decoded.count)
      for scan in 0..<Self.scanTile {
        let frameBase = scan * detectorPixels
        for pixel in 0..<detectorPixels {
          let value = decoded[frameBase + pixel]
          if let excludedOrdinal = excludedOrdinals[pixel] {
            if let expected = excludedRawValues[excludedOrdinal], expected != value {
              continue
            }
            excludedRawValues[excludedOrdinal] = value
          } else if value <= UInt16(UInt8.max) {
            working[frameBase + pixel] = UInt8(value)
          }
        }
      }
      working.withUnsafeBytes { workingDigest.update(bufferPointer: $0) }
      for pixel in 0..<detectorPixels {
        if excluded.contains(pixel) {
          guard let ordinal = excludedOrdinals[pixel], let expected = excludedRawValues[ordinal],
            (0..<Self.scanTile).allSatisfy({ decoded[$0 * detectorPixels + pixel] == expected })
          else {
            throw NativeLosslessPackV1ProducerError.nonconstantBadPixel(detectorPixel: pixel)
          }
          continue
        }
        var maximum: UInt16 = 0
        for scan in 0..<Self.scanTile {
          let value = decoded[scan * detectorPixels + pixel]
          guard value <= UInt16(UInt8.max) else {
            throw NativeLosslessPackV1ProducerError.unsupportedValue(
              scan: tileFirstScan + scan,
              detectorPixel: pixel,
              value: value
            )
          }
          maximum = max(maximum, value)
        }
        let width = maximum == 0 ? 0 : UInt8(UInt16.bitWidth - maximum.leadingZeroBitCount)
        widths[pixel * tileCount + tile] = width
        maximumBitWidth = max(maximumBitWidth, Int(width))
      }
    }

    let checkpointWords = (tileCount + 31) / 32
    let widthWords = (tileCount + 7) / 8
    let headerWordsPerPixel = checkpointWords + widthWords
    var headers = [UInt32](repeating: 0, count: detectorPixels * headerWordsPerPixel)
    var offsets = [UInt32](repeating: 0, count: widths.count)
    var payloadWords: UInt64 = 0
    for pixel in 0..<detectorPixels {
      guard let base = UInt32(exactly: payloadWords) else {
        throw NativeLosslessPackV1ProducerError.invalidRequest(
          "Lossless Pack Format v1 shard payload exceeds its 32-bit direct offset contract."
        )
      }
      let headerBase = pixel * headerWordsPerPixel
      headers[headerBase] = base
      var relative: UInt32 = 0
      for tile in 0..<tileCount {
        if tile > 0, tile.isMultiple(of: 32) {
          headers[headerBase + tile / 32] = relative
        }
        guard let globalOffset = UInt32(exactly: payloadWords + UInt64(relative)) else {
          throw NativeLosslessPackV1ProducerError.invalidRequest(
            "Lossless Pack Format v1 shard payload exceeds its 32-bit direct offset contract."
          )
        }
        offsets[pixel * tileCount + tile] = globalOffset
        let width = UInt32(widths[pixel * tileCount + tile])
        headers[headerBase + checkpointWords + tile / 8] |= width << UInt32((tile % 8) * 4)
        relative += width
      }
      payloadWords += UInt64(relative)
    }
    if payloadWords == 0 { payloadWords = 1 }
    guard payloadWords <= UInt64(UInt32.max), let payloadCount = Int(exactly: payloadWords) else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 shard payload exceeds the portable uint32 word range."
      )
    }
    var payload = [UInt32](repeating: 0, count: payloadCount)
    for tile in 0..<tileCount {
      try cancelled(shouldCancel)
      let decoded = try decoder.readFrames(
        firstScan: firstScan + tile * Self.scanTile,
        count: Self.scanTile
      )
      for pixel in 0..<detectorPixels {
        let width = Int(widths[pixel * tileCount + tile])
        guard width > 0 else { continue }
        let firstWord = Int(offsets[pixel * tileCount + tile])
        for scan in 0..<Self.scanTile {
          let value = UInt32(decoded[scan * detectorPixels + pixel])
          let bit = scan * width
          let word = firstWord + bit / 32
          let shift = bit % 32
          payload[word] |= value << UInt32(shift)
          if shift + width > 32 {
            payload[word + 1] |= value >> UInt32(32 - shift)
          }
        }
      }
    }
    let transientBytes = try checkedSum(
      UInt64(
        payload.count * 4 + headers.count * 4 + widths.count
          + offsets.count * 4 + detectorPixels * Self.scanTile * 3
      ),
      accountedPersistentBytes,
      label: "observed producer buffers"
    )
    return NativeLosslessPackV1PackedShard(
      payload: payload,
      headers: headers,
      maximumBitWidth: maximumBitWidth,
      transientBytes: transientBytes
    )
  }

  private func openWriter(path: URL, writer: inout OpaquePointer?) throws {
    var message: UnsafeMutablePointer<CChar>?
    let status = path.path.withCString {
      qh5_lossless_pack_v1_writer_open($0, UInt64(Self.userBlockBytes), &writer, &message)
    }
    defer { qh5_free_error(message) }
    guard status == 0, writer != nil else {
      throw NativeLosslessPackV1ProducerError.nativeWriter(
        message.map { String(cString: $0) }
          ?? "Could not create the native Lossless Pack Format v1 container."
      )
    }
  }

  private func appendShard(
    writer: OpaquePointer?,
    ordinal: Int,
    payload: [UInt32],
    headers: [UInt32]
  ) throws -> NativeLosslessPackV1NativeShardLayout {
    guard let writer, let ordinal = UInt32(exactly: ordinal) else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 writer state is invalid.")
    }
    var layout = qh5_lossless_pack_v1_shard_layout()
    var message: UnsafeMutablePointer<CChar>?
    let status = payload.withUnsafeBufferPointer { payloadBuffer in
      headers.withUnsafeBufferPointer { headerBuffer in
        qh5_lossless_pack_v1_writer_append_shard(
          writer,
          ordinal,
          payloadBuffer.baseAddress,
          UInt64(payloadBuffer.count),
          headerBuffer.baseAddress,
          UInt64(headerBuffer.count),
          &layout,
          &message
        )
      }
    }
    defer { qh5_free_error(message) }
    guard status == 0 else {
      throw NativeLosslessPackV1ProducerError.nativeWriter(
        message.map { String(cString: $0) } ?? "Could not append a Lossless Pack Format v1 shard."
      )
    }
    return NativeLosslessPackV1NativeShardLayout(
      payloadOffset: layout.payload_offset,
      payloadBytes: layout.payload_bytes,
      headersOffset: layout.headers_offset,
      headersBytes: layout.headers_bytes
    )
  }

  private func closeWriter(_ writer: inout OpaquePointer?) throws {
    guard let openWriter = writer else { return }
    var message: UnsafeMutablePointer<CChar>?
    let status = qh5_lossless_pack_v1_writer_close(openWriter, &message)
    writer = nil
    defer { qh5_free_error(message) }
    guard status == 0 else {
      throw NativeLosslessPackV1ProducerError.nativeWriter(
        message.map { String(cString: $0) }
          ?? "Could not close the Lossless Pack Format v1 container."
      )
    }
  }

  private func makeManifest(
    plan: NativeLosslessPackV1ProductionPlan,
    sourceRawSHA: String,
    workingSHA: String,
    badPixelRawValues: [UInt16]
  ) throws -> Data {
    var calibration: [String: Any] = [
      "sha256": plan.inspection.calibration.sha256
    ]
    if let value = plan.inspection.calibration.scanRowSamplingNanometer {
      calibration["scan_row_sampling_nm"] = value
    }
    if let value = plan.inspection.calibration.scanColumnSamplingNanometer {
      calibration["scan_column_sampling_nm"] = value
    }
    if let value = plan.inspection.calibration.detectorRowSampling {
      calibration["detector_row_sampling"] = value
    }
    if let value = plan.inspection.calibration.detectorColumnSampling {
      calibration["detector_column_sampling"] = value
    }
    if let value = plan.inspection.calibration.detectorSamplingUnit {
      calibration["detector_sampling_unit"] = value
    }
    let manifest: [String: Any] = [
      "schema": "quantem.gpu.packed-detector-h5/v3",
      "format_schema": NativeLosslessPackV1ProductionReceipt.formatSchema,
      "encoding_profile": plan.encodingProfile.rawValue,
      "status": "complete",
      "payload_codec": "direct-bitpacked-u32",
      "source_identity_sha256": plan.inspection.sourceIdentitySHA256,
      "source_raw_logical_sha256": sourceRawSHA,
      "source_shape": plan.inspection.sourceShape,
      "source_dtype": "uint16",
      "working_dtype": "uint8",
      "working_value_definition":
        "all admitted source counts exactly; authenticated dead pixels set to zero",
      "prepared_uint8_sha256": workingSHA,
      "detector_mask_sha256": plan.inspection.detectorMaskIdentitySHA256,
      "detector_mask_identity_origin": plan.inspection.detectorMaskIdentityOrigin,
      "masked_detector_pixels": plan.inspection.badPixelIndices,
      "masked_detector_pixels_sha256": plan.inspection.badPixelIdentitySHA256,
      "masked_detector_raw_values": badPixelRawValues,
      "scan_bin": 1,
      "detector_bin": 1,
      "crop": NSNull(),
      "scan_tile": plan.scanTile,
      "shard_count": plan.shardCount,
      "source_calibration": calibration,
      "producer_receipt_schema": NativeLosslessPackV1ProductionReceipt.currentSchema,
      "producer_execution_backend": plan.executionBackend.rawValue,
      "producer_gpu_accelerated": plan.executionBackend.isGPUAccelerated,
    ]
    return try JSONSerialization.data(
      withJSONObject: manifest,
      options: [.sortedKeys, .withoutEscapingSlashes]
    )
  }

  private func makeBinaryIndex(
    plan: NativeLosslessPackV1ProductionPlan,
    layouts: [NativeLosslessPackV1NativeShardLayout],
    receipts: [NativeLosslessPackV1ShardReceipt]
  ) throws -> Data {
    guard layouts.count == plan.shardCount, receipts.count == plan.shardCount else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 shard records are incomplete.")
    }
    var data = Data([0x51, 0x47, 0x49, 0x58, 0, 0, 0, 3])
    appendLE(UInt32(plan.shardCount), to: &data)
    appendLE(UInt32(0), to: &data)
    for value in plan.inspection.sourceShape {
      guard let word = UInt32(exactly: value) else {
        throw NativeLosslessPackV1ProducerError.invalidRequest(
          "Lossless Pack Format v1 shape exceeds UInt32.")
      }
      appendLE(word, to: &data)
    }
    appendLE(UInt32(plan.scansPerShard), to: &data)
    appendLE(UInt32(plan.scanTile), to: &data)
    appendLE(UInt32(1), to: &data)
    appendLE(UInt32(plan.inspection.badPixelIndices.count), to: &data)
    for pixel in plan.inspection.badPixelIndices { appendLE(UInt32(pixel), to: &data) }
    data.append(try dataFromSHA256(plan.inspection.sourceIdentitySHA256))
    for (layout, receipt) in zip(layouts, receipts) {
      appendLE(layout.payloadOffset, to: &data)
      appendLE(layout.payloadBytes, to: &data)
      appendLE(UInt64(0), to: &data)
      appendLE(UInt64(0), to: &data)
      appendLE(layout.headersOffset, to: &data)
      appendLE(layout.headersBytes, to: &data)
      appendLE(receipt.payloadBytes, to: &data)
      appendLE(UInt32(receipt.headerBytes / 4), to: &data)
      appendLE(UInt32(0), to: &data)
      data.append(try dataFromSHA256(receipt.payloadSHA256))
    }
    return data
  }

  private func writeUserBlock(manifest: Data, binary: Data, to url: URL) throws {
    let binaryOffset = (24 + manifest.count + 7) & ~7
    guard binaryOffset + binary.count <= Self.userBlockBytes else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Lossless Pack Format v1 metadata needs \(binaryOffset + binary.count) user-block bytes; the portable limit is \(Self.userBlockBytes)."
      )
    }
    var userBlock = Data(repeating: 0, count: Self.userBlockBytes)
    var prelude = Data([0x51, 0x47, 0x50, 0x55, 0x48, 0x35, 0, 1])
    appendLE(UInt32(manifest.count), to: &prelude)
    appendLE(crc32(manifest), to: &prelude)
    appendLE(UInt32(binaryOffset), to: &prelude)
    appendLE(UInt32(binary.count), to: &prelude)
    userBlock.replaceSubrange(0..<prelude.count, with: prelude)
    userBlock.replaceSubrange(24..<(24 + manifest.count), with: manifest)
    userBlock.replaceSubrange(binaryOffset..<(binaryOffset + binary.count), with: binary)
    let handle = try FileHandle(forWritingTo: url)
    defer { try? handle.close() }
    try handle.seek(toOffset: 0)
    try handle.write(contentsOf: userBlock)
  }

  private func verifySource(inspection: NativeLosslessPackV1SourceInspection) throws {
    let master = inspection.dataset.masterPath.map(URL.init(fileURLWithPath:))
    let dataFiles = inspection.dataset.dataFiles.map(URL.init(fileURLWithPath:))
    let hashes = try nativeSourceHashes(master: master, dataFiles: dataFiles)
    guard hashes.aggregate == inspection.sourceIdentitySHA256,
      hashes.master == inspection.dataset.masterSHA256,
      hashes.members == inspection.dataset.orderedMemberSHA256
    else {
      throw NativeLosslessPackV1ProducerError.sourceChanged(
        "The original HDF5 family no longer matches the inspected SHA-256 identity; inspect it again."
      )
    }
  }

  private func calibrationIdentity(
    for dataset: Native4DSTEMDataset
  ) throws -> NativeLosslessPackV1CalibrationIdentity {
    let numericValues = [
      dataset.scanPixelSizeRowNanometer,
      dataset.scanPixelSizeColNanometer,
      dataset.kPixelSizeRow,
      dataset.kPixelSizeCol,
    ].compactMap { $0 }
    guard numericValues.allSatisfy(\.isFinite) else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "The source calibration contains a non-finite value; correct the HDF5 metadata before packing."
      )
    }
    var fields: [String: Any] = [:]
    if let value = dataset.scanPixelSizeRowNanometer {
      fields["scan_row_sampling_nm"] = value
    }
    if let value = dataset.scanPixelSizeColNanometer {
      fields["scan_column_sampling_nm"] = value
    }
    if let value = dataset.kPixelSizeRow {
      fields["detector_row_sampling"] = value
    }
    if let value = dataset.kPixelSizeCol {
      fields["detector_column_sampling"] = value
    }
    if let value = dataset.kPixelUnit {
      fields["detector_sampling_unit"] = value
    }
    let encoded = try JSONSerialization.data(
      withJSONObject: fields,
      options: [.sortedKeys, .withoutEscapingSlashes]
    )
    return NativeLosslessPackV1CalibrationIdentity(
      scanRowSamplingNanometer: dataset.scanPixelSizeRowNanometer,
      scanColumnSamplingNanometer: dataset.scanPixelSizeColNanometer,
      detectorRowSampling: dataset.kPixelSizeRow,
      detectorColumnSampling: dataset.kPixelSizeCol,
      detectorSamplingUnit: dataset.kPixelUnit,
      sha256: hex(SHA256.hash(data: encoded))
    )
  }

  private func shardScanCount(scanCount: UInt64) throws -> Int {
    let limit = min(scanCount, UInt64(Self.targetScansPerShard))
    var candidate = limit - (limit % UInt64(Self.scanTile))
    while candidate >= UInt64(Self.scanTile) {
      if scanCount.isMultiple(of: candidate), let exact = Int(exactly: candidate) {
        return exact
      }
      candidate -= UInt64(Self.scanTile)
    }
    throw NativeLosslessPackV1ProducerError.invalidRequest(
      "Lossless Pack Format v1 cannot partition this scan count into complete 32-scan portable shards."
    )
  }

  private func outputCapacity(at destination: URL) throws -> UInt64 {
    let fileManager = FileManager.default
    var directory = destination.deletingLastPathComponent()
    while !fileManager.fileExists(atPath: directory.path) {
      let parent = directory.deletingLastPathComponent()
      guard parent.path != directory.path else {
        throw NativeLosslessPackV1ProducerError.invalidRequest(
          "Could not find an existing output ancestor for \(destination.path)."
        )
      }
      directory = parent
    }
    let values = try directory.resourceValues(forKeys: [
      .volumeAvailableCapacityForImportantUsageKey
    ])
    guard let available = values.volumeAvailableCapacityForImportantUsage, available >= 0 else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Could not determine available output capacity for \(directory.path)."
      )
    }
    return UInt64(available)
  }

  private func cancelled(_ predicate: @Sendable () -> Bool) throws {
    guard !predicate() else { throw NativeLosslessPackV1ProducerError.cancelled }
  }
}

private func readExact(
  descriptor: Int32,
  offset: UInt64,
  buffer: UnsafeMutableRawBufferPointer,
  label: String
) throws {
  var consumed = 0
  while consumed < buffer.count {
    let result = Darwin.pread(
      descriptor,
      buffer.baseAddress!.advanced(by: consumed),
      buffer.count - consumed,
      off_t(offset + UInt64(consumed))
    )
    if result < 0, errno == EINTR { continue }
    guard result > 0 else {
      throw Native4DSTEMIOError.invalidData(
        "Could not read exact compressed QH5 bytes from \(label)"
      )
    }
    consumed += result
  }
}

private func lz4Decode(_ compressed: [UInt8], output: inout [UInt8]) throws {
  var input = 0
  var written = 0
  func length(_ nibble: Int) throws -> Int {
    guard nibble == 15 else { return nibble }
    var result = nibble
    var byte = 255
    while byte == 255 {
      guard input < compressed.count else {
        throw Native4DSTEMIOError.invalidData("QH5 LZ4 length is truncated")
      }
      byte = Int(compressed[input])
      input += 1
      result += byte
    }
    return result
  }
  while input < compressed.count, written < output.count {
    let token = Int(compressed[input])
    input += 1
    let literalBytes = try length(token >> 4)
    guard literalBytes <= compressed.count - input,
      literalBytes <= output.count - written
    else { throw Native4DSTEMIOError.invalidData("QH5 LZ4 literal range is invalid") }
    output.replaceSubrange(
      written..<(written + literalBytes), with: compressed[input..<(input + literalBytes)])
    input += literalBytes
    written += literalBytes
    if input == compressed.count { break }
    guard compressed.count - input >= 2 else {
      throw Native4DSTEMIOError.invalidData("QH5 LZ4 match offset is truncated")
    }
    let matchOffset = Int(compressed[input]) | Int(compressed[input + 1]) << 8
    input += 2
    guard matchOffset > 0, matchOffset <= written else {
      throw Native4DSTEMIOError.invalidData("QH5 LZ4 match offset is invalid")
    }
    let matchBytes = try length(token & 15) + 4
    guard matchBytes <= output.count - written else {
      throw Native4DSTEMIOError.invalidData("QH5 LZ4 match exceeds the decoded block")
    }
    for index in 0..<matchBytes {
      output[written + index] = output[written + index - matchOffset]
    }
    written += matchBytes
  }
  guard input == compressed.count, written == output.count else {
    throw Native4DSTEMIOError.invalidData(
      "QH5 LZ4 block consumed \(input) of \(compressed.count) bytes and decoded \(written) of \(output.count) bytes"
    )
  }
}

private func bitUnshuffleUInt16(
  _ shuffled: [UInt8],
  output: inout [UInt16],
  destination: Int
) {
  for group in 0..<128 {
    for lane in 0..<32 {
      var value: UInt16 = 0
      for bit in 0..<16 {
        let offset = (bit * 128 + group) * 4
        let word =
          UInt32(shuffled[offset])
          | UInt32(shuffled[offset + 1]) << 8
          | UInt32(shuffled[offset + 2]) << 16
          | UInt32(shuffled[offset + 3]) << 24
        if word & (UInt32(1) << UInt32(lane)) != 0 {
          value |= UInt16(1) << UInt16(bit)
        }
      }
      output[destination + group * 32 + lane] = value
    }
  }
}

private func orderedPixelSHA256(_ pixels: [Int]) -> String {
  var digest = SHA256()
  for pixel in pixels {
    var value = UInt32(pixel).littleEndian
    withUnsafeBytes(of: &value) { digest.update(bufferPointer: $0) }
  }
  return hex(digest.finalize())
}

private func zeroDetectorMaskSHA256(pixelCount: UInt64) -> String {
  var digest = SHA256()
  let zeroBytes = [UInt8](repeating: 0, count: 16 * 1024)
  var remaining = pixelCount * UInt64(MemoryLayout<UInt32>.stride)
  zeroBytes.withUnsafeBytes { buffer in
    while remaining > 0 {
      let count = min(remaining, UInt64(buffer.count))
      digest.update(
        bufferPointer: UnsafeRawBufferPointer(
          start: buffer.baseAddress,
          count: Int(count)
        )
      )
      remaining -= count
    }
  }
  return hex(digest.finalize())
}

private func sha256(words: [UInt32]) -> String {
  words.withUnsafeBytes { hex(SHA256.hash(data: Data($0))) }
}

private func sha256(file: URL) throws -> String {
  let descriptor = Darwin.open(file.path, O_RDONLY)
  guard descriptor >= 0 else {
    throw Native4DSTEMIOError.invalidData("Could not hash \(file.path)")
  }
  defer { Darwin.close(descriptor) }
  let allocation = UnsafeMutableRawPointer.allocate(
    byteCount: 8 * 1024 * 1024,
    alignment: Int(getpagesize())
  )
  defer { allocation.deallocate() }
  var digest = SHA256()
  while true {
    let count = Darwin.read(descriptor, allocation, 8 * 1024 * 1024)
    if count < 0, errno == EINTR { continue }
    guard count >= 0 else {
      throw Native4DSTEMIOError.invalidData("Could not hash \(file.path)")
    }
    if count == 0 { break }
    digest.update(bufferPointer: UnsafeRawBufferPointer(start: allocation, count: count))
  }
  return hex(digest.finalize())
}

private func dataFromSHA256(_ text: String) throws -> Data {
  guard text.count == 64 else {
    throw NativeLosslessPackV1ProducerError.invalidRequest("Expected one lowercase SHA-256 digest.")
  }
  var result = Data()
  result.reserveCapacity(32)
  var index = text.startIndex
  for _ in 0..<32 {
    let next = text.index(index, offsetBy: 2)
    guard let byte = UInt8(text[index..<next], radix: 16) else {
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Expected one lowercase SHA-256 digest.")
    }
    result.append(byte)
    index = next
  }
  return result
}

private func appendLE(_ value: UInt32, to data: inout Data) {
  var little = value.littleEndian
  withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
}

private func appendLE(_ value: UInt64, to data: inout Data) {
  var little = value.littleEndian
  withUnsafeBytes(of: &little) { data.append(contentsOf: $0) }
}

private func crc32(_ data: Data) -> UInt32 {
  var crc = UInt32.max
  for byte in data {
    crc ^= UInt32(byte)
    for _ in 0..<8 {
      crc = (crc >> 1) ^ (0xedb8_8320 & (UInt32(0) &- (crc & 1)))
    }
  }
  return ~crc
}

private func synchronize(_ url: URL) throws {
  let handle = try FileHandle(forUpdating: url)
  defer { try? handle.close() }
  try handle.synchronize()
}

private func publishAtomically(
  temporaryOutput: URL,
  destination: URL,
  temporaryReceipt: URL,
  receiptDestination: URL
) throws {
  guard Darwin.link(temporaryReceipt.path, receiptDestination.path) == 0 else {
    if errno == EEXIST {
      throw NativeLosslessPackV1ProducerError.destinationExists(receiptDestination.path)
    }
    throw NativeLosslessPackV1ProducerError.invalidRequest(
      "Could not atomically stage the Lossless Pack Format v1 receipt: \(String(cString: strerror(errno)))."
    )
  }
  do {
    guard Darwin.link(temporaryOutput.path, destination.path) == 0 else {
      if errno == EEXIST {
        throw NativeLosslessPackV1ProducerError.destinationExists(destination.path)
      }
      throw NativeLosslessPackV1ProducerError.invalidRequest(
        "Could not atomically publish Lossless Pack Format v1 output: \(String(cString: strerror(errno)))."
      )
    }
  } catch {
    Darwin.unlink(receiptDestination.path)
    throw error
  }
  Darwin.unlink(temporaryOutput.path)
  Darwin.unlink(temporaryReceipt.path)
}

private func sameIdentity(_ lhs: NativeFileIdentity, _ rhs: NativeFileIdentity) -> Bool {
  lhs.path == rhs.path && lhs.device == rhs.device && lhs.inode == rhs.inode
    && lhs.bytes == rhs.bytes
    && lhs.modificationNanoseconds == rhs.modificationNanoseconds
}

private func checkedProduct(_ lhs: UInt64, _ rhs: UInt64, label: String) throws -> UInt64 {
  let result = lhs.multipliedReportingOverflow(by: rhs)
  guard !result.overflow else {
    throw NativeLosslessPackV1ProducerError.invalidRequest(
      "Lossless Pack Format v1 \(label) overflows UInt64.")
  }
  return result.partialValue
}

private func checkedSum(_ lhs: UInt64, _ rhs: UInt64, label: String) throws -> UInt64 {
  let result = lhs.addingReportingOverflow(rhs)
  guard !result.overflow else {
    throw NativeLosslessPackV1ProducerError.invalidRequest(
      "Lossless Pack Format v1 \(label) overflows UInt64.")
  }
  return result.partialValue
}

private func hex<S: Sequence>(_ bytes: S) -> String where S.Element == UInt8 {
  bytes.map { String(format: "%02x", $0) }.joined()
}
