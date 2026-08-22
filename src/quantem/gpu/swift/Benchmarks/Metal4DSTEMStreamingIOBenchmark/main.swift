import CryptoKit
import Darwin
import Foundation
import Metal
import Metal4DSTEMKernels
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

private struct Options {
  static let usage = """
    Usage: metal-4dstem-indexed-load-benchmark \\
      --input PATH \\
      --cache-dir PATH \\
      --output-dir EMPTY_PATH \\
      --revision 40_CHARACTER_COMMIT_SHA \\
      (--bands-file PATH | --all-bands) \\
      [--window-scan-rows N] \\
      [--iterations N] \\
      [--buffered-read-ahead-shards N] \\
      [--exact-working-audit PATH --detector-bin N --maximum-shard-bytes N] \\
      [--destination-storage shared|private] \\
      [--reuse-working-destinations] \\
      [--boundary-working-volume-hashes] \\
      [--exact-working-cache-payload PATH \\
       --exact-working-cache-metadata PATH \\
       --cache-reopen-iterations N]

    Source-page state is not purged or inferred. The first trial is labeled as
    a first process encounter with prepared indexes and unspecified source pages;
    later trials are same-process repeats. Supplying all three exact-working
    options retains the complete full-scan working volume as bounded packed
    uint16 shards. Supplying both cache paths instead writes those same exact
    shards transactionally with only one output shard resident at a time;
    cache mode requires --iterations 1 and records separately labeled reopen
    validation. --reuse-working-destinations retains the first resident load's
    caller-owned Metal shards and exactly overwrites them on later trials; its
    fresh-allocation trial and reused-buffer distribution remain separate.
    Application UI and product presentation are not measured.
    """

  let input: URL
  let cacheDirectory: URL
  let outputDirectory: URL
  let revision: String
  let windowScanRows: Int
  let iterations: Int
  let bufferedReadAheadShards: Int
  let bandsFile: URL?
  let allBands: Bool
  let exactWorkingAudit: URL?
  let detectorBin: Int?
  let maximumShardBytes: UInt64?
  let residentStorage: Metal4DSTEMResidentStorage
  let reuseWorkingDestinations: Bool
  let boundaryWorkingVolumeHashes: Bool
  let exactWorkingCachePayload: URL?
  let exactWorkingCacheMetadata: URL?
  let cacheReopenIterations: Int

  static func parse(_ arguments: [String]) throws -> Self {
    var values: [String: String] = [:]
    var flags = Set<String>()
    var index = 0
    while index < arguments.count {
      let argument = arguments[index]
      if argument == "--all-bands" || argument == "--reuse-working-destinations"
        || argument == "--boundary-working-volume-hashes"
      {
        flags.insert(argument)
        index += 1
        continue
      }
      guard argument.hasPrefix("--"), index + 1 < arguments.count else {
        throw BenchmarkError.usage("Unexpected argument: \(argument)")
      }
      values[argument] = arguments[index + 1]
      index += 2
    }
    guard let input = values["--input"],
      let cache = values["--cache-dir"],
      let output = values["--output-dir"],
      let revision = values["--revision"],
      revision.utf8.count == 40,
      revision.utf8.allSatisfy({
        (48...57).contains($0) || (97...102).contains($0)
      })
    else {
      throw BenchmarkError.usage(Self.usage)
    }
    let bandsFile = values["--bands-file"].map { URL(fileURLWithPath: $0) }
    let allBands = flags.contains("--all-bands")
    guard (bandsFile != nil) != allBands else {
      throw BenchmarkError.usage(
        "Choose exactly one explicit detector-band source: --bands-file or --all-bands."
      )
    }
    let windowRows = Int(values["--window-scan-rows"] ?? "8") ?? 0
    let iterations = Int(values["--iterations"] ?? "1") ?? 0
    let bufferedReadAheadShards =
      Int(values["--buffered-read-ahead-shards"] ?? "0") ?? -1
    guard windowRows > 0, iterations > 0, bufferedReadAheadShards >= 0 else {
      throw BenchmarkError.usage(
        "Window rows and iterations must be positive; buffered read-ahead "
          + "shards must be a nonnegative integer."
      )
    }
    let audit = values["--exact-working-audit"].map {
      URL(fileURLWithPath: $0)
    }
    let detectorBin = values["--detector-bin"].flatMap(Int.init)
    let maximumShardBytes = values["--maximum-shard-bytes"].flatMap(UInt64.init)
    let exactOptions = [audit != nil, detectorBin != nil, maximumShardBytes != nil]
    guard exactOptions.allSatisfy({ $0 }) || exactOptions.allSatisfy({ !$0 }) else {
      throw BenchmarkError.usage(
        "Provide --exact-working-audit, --detector-bin, and --maximum-shard-bytes together."
      )
    }
    if let detectorBin {
      guard Metal4DSTEMLoadPlan.supportedDetectorBins.contains(detectorBin),
        let maximumShardBytes, maximumShardBytes > 0
      else {
        throw BenchmarkError.usage(
          "Exact working-volume detector bin must be 1, 2, or 4 and shard bytes must be positive."
        )
      }
    }
    let residentStorageRaw = values["--destination-storage"] ?? "shared"
    guard
      let residentStorage = Metal4DSTEMResidentStorage(
        rawValue: residentStorageRaw
      )
    else {
      throw BenchmarkError.usage(
        "--destination-storage must be shared or private."
      )
    }
    if values["--destination-storage"] != nil, audit == nil {
      throw BenchmarkError.usage(
        "--destination-storage requires resident exact-working options."
      )
    }
    let cachePayload = values["--exact-working-cache-payload"].map {
      URL(fileURLWithPath: $0)
    }
    let cacheMetadata = values["--exact-working-cache-metadata"].map {
      URL(fileURLWithPath: $0)
    }
    guard (cachePayload != nil) == (cacheMetadata != nil) else {
      throw BenchmarkError.usage(
        "Provide --exact-working-cache-payload and "
          + "--exact-working-cache-metadata together."
      )
    }
    let cacheReopenIterations =
      Int(values["--cache-reopen-iterations"] ?? (cachePayload == nil ? "0" : "7"))
      ?? -1
    guard cacheReopenIterations >= 0 else {
      throw BenchmarkError.usage("--cache-reopen-iterations must be nonnegative.")
    }
    if cachePayload != nil {
      guard audit != nil, iterations == 1, cacheReopenIterations > 0,
        residentStorage == .shared
      else {
        throw BenchmarkError.usage(
          "Exact cache mode requires exact-working options, --iterations 1, "
            + "shared destination storage, and at least one cache-reopen iteration."
        )
      }
    } else if values["--cache-reopen-iterations"] != nil {
      throw BenchmarkError.usage(
        "--cache-reopen-iterations is valid only with exact cache paths."
      )
    }
    let reuseWorkingDestinations = flags.contains("--reuse-working-destinations")
    let boundaryWorkingVolumeHashes = flags.contains(
      "--boundary-working-volume-hashes"
    )
    if reuseWorkingDestinations {
      guard audit != nil, cachePayload == nil, iterations >= 2 else {
        throw BenchmarkError.usage(
          "--reuse-working-destinations requires resident exact-working options, "
            + "no cache payload, and at least two iterations."
        )
      }
    }
    return Self(
      input: URL(fileURLWithPath: input),
      cacheDirectory: URL(fileURLWithPath: cache, isDirectory: true),
      outputDirectory: URL(fileURLWithPath: output, isDirectory: true),
      revision: revision,
      windowScanRows: windowRows,
      iterations: iterations,
      bufferedReadAheadShards: bufferedReadAheadShards,
      bandsFile: bandsFile,
      allBands: allBands,
      exactWorkingAudit: audit,
      detectorBin: detectorBin,
      maximumShardBytes: maximumShardBytes,
      residentStorage: residentStorage,
      reuseWorkingDestinations: reuseWorkingDestinations,
      boundaryWorkingVolumeHashes: boundaryWorkingVolumeHashes,
      exactWorkingCachePayload: cachePayload,
      exactWorkingCacheMetadata: cacheMetadata,
      cacheReopenIterations: cacheReopenIterations
    )
  }
}

private enum BenchmarkError: LocalizedError {
  case usage(String)
  case invalid(String)

  var errorDescription: String? {
    switch self {
    case .usage(let message), .invalid(let message): message
    }
  }
}

private struct RunRecord: Codable {
  let trial: Int
  let state: String
  let destinationReused: Bool
  let wallSeconds: Double
  let destinationAllocationSeconds: Double?
  let workingVolumeHashSeconds: Double?
  let payloadWriteSeconds: Double?
  let payloadFinalizeSeconds: Double?
  let peakWorkingMetalBytes: UInt64?
  let destinationStorage: String?
  let gpuSeconds: Double
  let synchronizedStageSeconds: Metal4DSTEMIndexedStageSeconds?
  let sourceMappingSeconds: Double
  let sourceOpenAndStatSeconds: Double
  let sourceMmapSeconds: Double
  let sourceReadSeconds: Double
  let sourceMetalBufferSeconds: Double
  let sourceTransfer: Metal4DSTEMIndexedSourceTransfer
  let commandBufferCount: Int
  let peakInFlightCommandBuffers: Int
  let peakInFlightMappedSourceBytes: UInt64
  let peakRetainedSourceBufferBytes: UInt64
  let mappedCompressedSourceBytes: UInt64
  let maximumMappedCompressedSourceBytes: UInt64
  let maximumMappedSourceBufferBytes: UInt64
  let maximumDecodedSliceBytes: UInt64
  let maximumSourceCount: UInt32
  let pixelsAbove255: UInt64
  let binningDispatchCount: Int?
  let workingVolumeSHA256: String?
  let metalAllocatedBytesBeforeLoad: UInt64
  let metalAllocatedBytesAfterLoad: UInt64
  let metalAllocatedBytesAfterRelease: UInt64
  let hashes: [String: String]
}

private struct CacheReopenRecord: Codable {
  let trial: Int
  let state: String
  let wallSeconds: Double
  let metadataReadSeconds: Double
  let identityValidationSeconds: Double
  let payloadBytes: UInt64
  let payloadSHA256: String
}

private struct Distribution: Codable {
  let samples: Int
  let p50: Double
  let p95: Double
  let maximum: Double
}

private struct BenchmarkSummary: Codable {
  let schema: String
  let revision: String
  let timestamp: String
  let host: String
  let os: String
  let device: String
  let input: String
  let sourceIdentitySHA256: String
  let sourceShape: [Int]
  let workingShape: [Int]
  let sourceDtype: String
  let workingDtype: String
  let scanBin: Int
  let detectorBin: Int
  let crop: String
  let bandSource: String
  let compressedSourceBytes: UInt64
  let logicalDecodedBytes: UInt64
  let maximumDecodedWindowBytes: UInt64
  let requestedWindowScanRows: Int
  let maximumActualWindowBytes: UInt64
  let maximumDetectorPartialScratchBytes: UInt64?
  let estimatedAllocatedMetalBytesExcludingMappedSource: UInt64
  let estimatedAllocatedMetalBytesIncludingSourceTransfer: UInt64
  let sourceTransfer: Metal4DSTEMIndexedSourceTransfer
  let maximumRetainedSourceBufferBytes: UInt64
  let maximumMappedCompressedBytes: UInt64
  let maximumMappedSourceBufferBytes: UInt64
  let maximumInFlightMappedSourceBytes: UInt64
  let maximumInFlightCommandBuffers: Int
  let maximumIndividualMetalBufferBytes: UInt64
  let workingPayloadBytes: UInt64?
  let shardCount: Int?
  let maximumActualShardBytes: UInt64?
  let valueRangeAuditSHA256: String?
  let workingVolumeSHA256: String?
  let destinationStorage: String?
  let cachePayload: String?
  let cacheMetadata: String?
  let cacheReopenValidation: Distribution?
  let cacheReopenRuns: [CacheReopenRecord]
  let processPeakResidentBytes: UInt64?
  let catalogSeconds: Double
  let pipelineCompilationSeconds: Double
  let planSeconds: Double
  let firstPackageEndToEndSeconds: Double
  let reusesResidentDestinationShards: Bool
  let firstFreshDestinationWallSeconds: Double?
  let reusedDestinationLoadWall: Distribution?
  let repeatedLoadWall: Distribution
  let runs: [RunRecord]
  let provenance: Metal4DSTEMIndexedLoadProvenance
  let binningProvenance: Metal4DSTEMExactBinningProvenance?
  let samplingPropagation: Metal4DSTEMSamplingPropagation?
  let shardPlan: Metal4DSTEMExactBinningShardPlan?
}

private func checkedProduct(_ values: [UInt64], label: String) throws -> UInt64 {
  try values.reduce(UInt64(1)) { result, value in
    let product = result.multipliedReportingOverflow(by: value)
    guard !product.overflow else {
      throw BenchmarkError.invalid("\(label) overflows UInt64")
    }
    return product.partialValue
  }
}

private func percentile(_ values: [Double], fraction: Double) -> Double {
  let sorted = values.sorted()
  guard !sorted.isEmpty else { return .nan }
  let rank = Int(ceil(fraction * Double(sorted.count))) - 1
  return sorted[max(0, min(sorted.count - 1, rank))]
}

private func littleEndianData(_ values: [UInt64]) -> Data {
  let copy = values.map(\.littleEndian)
  return copy.withUnsafeBytes { Data($0) }
}

private func sha256(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func sha256(_ buffers: [MTLBuffer], device: MTLDevice) throws -> String {
  var hasher = SHA256()
  let maximumLength = buffers.map(\.length).max() ?? 0
  let privateReadback: MTLBuffer?
  let privateQueue: MTLCommandQueue?
  if buffers.contains(where: { $0.storageMode == .private }) {
    privateReadback = device.makeBuffer(
      length: maximumLength,
      options: .storageModeShared
    )
    privateQueue = device.makeCommandQueue()
    guard privateReadback != nil, privateQueue != nil else {
      throw BenchmarkError.invalid(
        "Metal could not allocate the explicit private-volume parity readback."
      )
    }
  } else {
    privateReadback = nil
    privateQueue = nil
  }
  for buffer in buffers {
    let readable: MTLBuffer
    switch buffer.storageMode {
    case .shared:
      readable = buffer
    case .private:
      guard let privateReadback, let privateQueue,
        let command = privateQueue.makeCommandBuffer(),
        let encoder = command.makeBlitCommandEncoder()
      else {
        throw BenchmarkError.invalid(
          "Metal could not create the explicit private-volume parity copy."
        )
      }
      encoder.copy(
        from: buffer,
        sourceOffset: 0,
        to: privateReadback,
        destinationOffset: 0,
        size: buffer.length
      )
      encoder.endEncoding()
      command.commit()
      command.waitUntilCompleted()
      guard command.status == .completed else {
        throw BenchmarkError.invalid(
          "The explicit private-volume parity copy failed: "
            + (command.error?.localizedDescription ?? "unknown Metal error")
        )
      }
      readable = privateReadback
    default:
      throw BenchmarkError.invalid(
        "Working-volume hashing supports shared or private Metal shards."
      )
    }
    hasher.update(
      bufferPointer: UnsafeRawBufferPointer(
        start: readable.contents(),
        count: buffer.length
      )
    )
  }
  return hasher.finalize().map { String(format: "%02x", $0) }.joined()
}

private func productData(_ products: Metal4DSTEMExactProducts) -> [String: Data] {
  [
    "detector_sum.u64": littleEndianData(products.detectorSum),
    "band1.u64": littleEndianData(products.band1),
    "band2.u64": littleEndianData(products.band2),
    "band4.u64": littleEndianData(products.band4),
    "total.u64": littleEndianData(products.total),
    "detector_row_moment.u64": littleEndianData(products.detectorRowMoment),
    "detector_column_moment.u64": littleEndianData(products.detectorColumnMoment),
  ]
}

private struct LoadedTrial {
  let products: Metal4DSTEMExactProducts
  let sourceAudit: Metal4DSTEMExactSourceAudit
  let provenance: Metal4DSTEMIndexedLoadProvenance
  let metrics: Metal4DSTEMIndexedLoadMetrics
  let wallSeconds: Double
  let destinationAllocationSeconds: Double?
  let payloadWriteSeconds: Double?
  let payloadFinalizeSeconds: Double?
  let peakWorkingMetalBytes: UInt64?
  let destinationStorage: String?
  let binningDispatchCount: Int?
  let workingVolumeSHA256: String?
  let workingVolumeHashSeconds: Double?
  let metalAllocatedBytesAfterLoad: UInt64
  let destinationReused: Bool
  let retainedWorkingVolumeShards: [MTLBuffer]?
}

private func run() throws {
  let arguments = Array(CommandLine.arguments.dropFirst())
  if arguments.contains("--help") || arguments.contains("-h") {
    print(Options.usage)
    return
  }
  let options = try Options.parse(arguments)
  if FileManager.default.fileExists(atPath: options.outputDirectory.path) {
    let existing = try FileManager.default.contentsOfDirectory(
      at: options.outputDirectory,
      includingPropertiesForKeys: nil
    )
    guard existing.isEmpty else {
      throw BenchmarkError.invalid(
        "Benchmark output directory is not empty; choose a new immutable trial directory."
      )
    }
  }
  let catalogStarted = CFAbsoluteTimeGetCurrent()
  let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: options.cacheDirectory)
    .prepare(input: options.input, mode: .indexed)
  let catalogSeconds = CFAbsoluteTimeGetCurrent() - catalogStarted
  guard catalog.datasets.count == 1, let dataset = catalog.datasets.first else {
    throw BenchmarkError.invalid(
      "Benchmark input must resolve to exactly one indexed 4D-STEM dataset."
    )
  }
  guard options.windowScanRows <= dataset.scanRows else {
    throw BenchmarkError.invalid(
      "--window-scan-rows cannot exceed the source scan-row count of "
        + "\(dataset.scanRows)."
    )
  }
  let source = try Native4DSTEMIndexedSource.open(dataset: dataset)
  let detectorPixels = try checkedProduct(
    [UInt64(dataset.detectorRows), UInt64(dataset.detectorCols)],
    label: "detector pixels"
  )
  guard let detectorPixelCount = Int(exactly: detectorPixels) else {
    throw BenchmarkError.invalid("Detector geometry exceeds this process's address space.")
  }
  let membership: [UInt8]
  let bandSource: String
  if let bandsFile = options.bandsFile {
    let data = try Data(contentsOf: bandsFile)
    guard data.count == detectorPixelCount else {
      throw BenchmarkError.invalid(
        "Detector-band file contains \(data.count) bytes; expected \(detectorPixelCount)."
      )
    }
    membership = Array(data)
    bandSource = "file:\(bandsFile.path):sha256:\(sha256(data))"
  } else {
    membership = [UInt8](repeating: 7, count: detectorPixelCount)
    bandSource = "explicit_all_pixels_in_all_three_bands"
  }
  let bands = try Metal4DSTEMDetectorBands(
    detectorRows: dataset.detectorRows,
    detectorColumns: dataset.detectorCols,
    membership: membership
  )
  let windowBytes = try checkedProduct(
    [
      UInt64(options.windowScanRows),
      UInt64(dataset.scanCols),
      UInt64(dataset.detectorRows),
      UInt64(dataset.detectorCols),
      UInt64(MemoryLayout<UInt16>.stride),
    ],
    label: "decoded window"
  )
  let sourceTransfer: Metal4DSTEMIndexedSourceTransfer =
    options.bufferedReadAheadShards == 0
    ? .memoryMapped
    : .bufferedReadAhead(
      prefetchShardCount: options.bufferedReadAheadShards
    )
  let planStarted = CFAbsoluteTimeGetCurrent()
  let productPlan = try Metal4DSTEMIndexedLoadPlan(
    source: source,
    maximumDecodedWindowBytes: windowBytes,
    detectorBands: bands,
    sourceTransfer: sourceTransfer
  )
  let binnedPlan: Metal4DSTEMIndexedBinnedLoadPlan?
  if let auditURL = options.exactWorkingAudit,
    let detectorBin = options.detectorBin,
    let maximumShardBytes = options.maximumShardBytes
  {
    let audit = try JSONDecoder().decode(
      Metal4DSTEMExactSourceAudit.self,
      from: Data(contentsOf: auditURL)
    )
    binnedPlan = try Metal4DSTEMIndexedBinnedLoadPlan(
      source: source,
      maximumDecodedWindowBytes: windowBytes,
      detectorBands: bands,
      detectorBin: detectorBin,
      sourceAudit: audit,
      maximumShardBytes: maximumShardBytes,
      residentStorage: options.residentStorage,
      sourceTransfer: sourceTransfer
    )
  } else {
    binnedPlan = nil
  }
  let planSeconds = CFAbsoluteTimeGetCurrent() - planStarted
  guard let device = MTLCreateSystemDefaultDevice() else {
    throw BenchmarkError.invalid("No Metal device is visible to this benchmark process.")
  }
  let compileStarted = CFAbsoluteTimeGetCurrent()
  let loader = try Metal4DSTEMIndexedLoader(device: device)
  let compileSeconds = CFAbsoluteTimeGetCurrent() - compileStarted

  var records: [RunRecord] = []
  var acceptedHashes: [String: String]?
  var acceptedProvenance: Metal4DSTEMIndexedLoadProvenance?
  var acceptedWorkingVolumeSHA256: String?
  var acceptedData: [String: Data] = [:]
  var reusableWorkingVolumeShards: [MTLBuffer]?
  for trial in 1...options.iterations {
    let metalAllocatedBytesBeforeLoad = UInt64(device.currentAllocatedSize)
    let loaded: LoadedTrial = try autoreleasepool {
      if let binnedPlan {
        if let payloadURL = options.exactWorkingCachePayload,
          let metadataURL = options.exactWorkingCacheMetadata
        {
          let result = try loader.loadExactBinnedCache(
            source: source,
            plan: binnedPlan,
            payloadURL: payloadURL,
            metadataURL: metadataURL
          )
          return LoadedTrial(
            products: result.products,
            sourceAudit: result.sourceAudit,
            provenance: result.nativeProductProvenance,
            metrics: result.metrics.indexedLoad,
            wallSeconds: result.metrics.totalWallSeconds,
            destinationAllocationSeconds:
              result.metrics.destinationAllocationSeconds,
            payloadWriteSeconds: result.metrics.payloadWriteSeconds,
            payloadFinalizeSeconds: result.metrics.payloadFinalizeSeconds,
            peakWorkingMetalBytes: result.metrics.peakWorkingMetalBytes,
            destinationStorage: result.metrics.destinationStorage,
            binningDispatchCount: result.metrics.binningDispatchCount,
            workingVolumeSHA256: result.metadata.payloadSHA256,
            workingVolumeHashSeconds: nil,
            metalAllocatedBytesAfterLoad: UInt64(device.currentAllocatedSize),
            destinationReused: false,
            retainedWorkingVolumeShards: nil
          )
        }
        let result: Metal4DSTEMIndexedBinnedLoadResult
        let destinationReused: Bool
        if let reusableWorkingVolumeShards {
          result = try loader.loadExactBinnedShards(
            source: source,
            plan: binnedPlan,
            destinationShards: reusableWorkingVolumeShards
          )
          destinationReused = true
        } else {
          result = try loader.loadExactBinnedShards(
            source: source,
            plan: binnedPlan
          )
          destinationReused = false
        }
        let hashesWorkingVolume =
          !options.boundaryWorkingVolumeHashes
          || trial == 1 || trial == options.iterations
        let hashStarted = CFAbsoluteTimeGetCurrent()
        let workingVolumeSHA256 =
          hashesWorkingVolume
          ? try sha256(result.workingVolumeShards, device: device) : nil
        let hashSeconds =
          hashesWorkingVolume
          ? CFAbsoluteTimeGetCurrent() - hashStarted : nil
        return LoadedTrial(
          products: result.products,
          sourceAudit: result.sourceAudit,
          provenance: result.nativeProductProvenance,
          metrics: result.metrics.indexedLoad,
          wallSeconds: result.metrics.totalWallSeconds,
          destinationAllocationSeconds: result.metrics.destinationAllocationSeconds,
          payloadWriteSeconds: nil,
          payloadFinalizeSeconds: nil,
          peakWorkingMetalBytes: result.metrics.workingPayloadBytes,
          destinationStorage: result.metrics.destinationStorageMode,
          binningDispatchCount: result.metrics.binningDispatchCount,
          workingVolumeSHA256: workingVolumeSHA256,
          workingVolumeHashSeconds: hashSeconds,
          metalAllocatedBytesAfterLoad: UInt64(device.currentAllocatedSize),
          destinationReused: destinationReused,
          retainedWorkingVolumeShards:
            options.reuseWorkingDestinations ? result.workingVolumeShards : nil
        )
      }
      let result = try loader.loadExactProducts(source: source, plan: productPlan)
      return LoadedTrial(
        products: result.products,
        sourceAudit: result.sourceAudit,
        provenance: result.provenance,
        metrics: result.metrics,
        wallSeconds: result.metrics.wallSeconds,
        destinationAllocationSeconds: nil,
        payloadWriteSeconds: nil,
        payloadFinalizeSeconds: nil,
        peakWorkingMetalBytes: nil,
        destinationStorage: nil,
        binningDispatchCount: nil,
        workingVolumeSHA256: nil,
        workingVolumeHashSeconds: nil,
        metalAllocatedBytesAfterLoad: UInt64(device.currentAllocatedSize),
        destinationReused: false,
        retainedWorkingVolumeShards: nil
      )
    }
    if let retained = loaded.retainedWorkingVolumeShards {
      reusableWorkingVolumeShards = retained
    }
    let metalAllocatedBytesAfterRelease = UInt64(device.currentAllocatedSize)
    let data = productData(loaded.products)
    let hashes = data.mapValues(sha256)
    if let acceptedHashes, hashes != acceptedHashes {
      throw BenchmarkError.invalid(
        "Exact product hashes changed between repeated load trials."
      )
    }
    if let acceptedProvenance {
      guard acceptedProvenance == loaded.provenance else {
        throw BenchmarkError.invalid(
          "Exact product provenance changed between repeated load trials."
        )
      }
    }
    if let workingVolumeSHA256 = loaded.workingVolumeSHA256 {
      if let acceptedWorkingVolumeSHA256,
        acceptedWorkingVolumeSHA256 != workingVolumeSHA256
      {
        throw BenchmarkError.invalid(
          "Exact working-volume SHA-256 changed between repeated load trials."
        )
      }
      acceptedWorkingVolumeSHA256 = workingVolumeSHA256
    }
    acceptedHashes = hashes
    acceptedProvenance = loaded.provenance
    acceptedData = data
    records.append(
      RunRecord(
        trial: trial,
        state: loaded.destinationReused
          ? "same_process_\(loaded.metrics.cacheState)_caller_destination_reused"
          : (trial == 1
            ? "first_process_\(loaded.metrics.cacheState)_fresh_destination_allocation"
            : "same_process_repeat_\(loaded.metrics.cacheState)"),
        destinationReused: loaded.destinationReused,
        wallSeconds: loaded.wallSeconds,
        destinationAllocationSeconds: loaded.destinationAllocationSeconds,
        workingVolumeHashSeconds: loaded.workingVolumeHashSeconds,
        payloadWriteSeconds: loaded.payloadWriteSeconds,
        payloadFinalizeSeconds: loaded.payloadFinalizeSeconds,
        peakWorkingMetalBytes: loaded.peakWorkingMetalBytes,
        destinationStorage: loaded.destinationStorage,
        gpuSeconds: loaded.metrics.gpuSeconds,
        synchronizedStageSeconds: loaded.metrics.synchronizedStageSeconds,
        sourceMappingSeconds: loaded.metrics.sourceMappingSeconds,
        sourceOpenAndStatSeconds: loaded.metrics.sourceOpenAndStatSeconds,
        sourceMmapSeconds: loaded.metrics.sourceMmapSeconds,
        sourceReadSeconds: loaded.metrics.sourceReadSeconds,
        sourceMetalBufferSeconds: loaded.metrics.sourceMetalBufferSeconds,
        sourceTransfer: loaded.metrics.sourceTransfer,
        commandBufferCount: loaded.metrics.commandBufferCount,
        peakInFlightCommandBuffers: loaded.metrics.peakInFlightCommandBuffers,
        peakInFlightMappedSourceBytes: loaded.metrics.peakInFlightMappedSourceBytes,
        peakRetainedSourceBufferBytes:
          loaded.metrics.peakRetainedSourceBufferBytes,
        mappedCompressedSourceBytes: loaded.metrics.mappedCompressedSourceBytes,
        maximumMappedCompressedSourceBytes:
          loaded.metrics.maximumMappedCompressedSourceBytes,
        maximumMappedSourceBufferBytes:
          loaded.metrics.maximumMappedSourceBufferBytes,
        maximumDecodedSliceBytes: loaded.metrics.maximumDecodedSliceBytes,
        maximumSourceCount: loaded.sourceAudit.maximumSourceCount,
        pixelsAbove255: loaded.sourceAudit.pixelsAbove255,
        binningDispatchCount: loaded.binningDispatchCount,
        workingVolumeSHA256: loaded.workingVolumeSHA256,
        metalAllocatedBytesBeforeLoad: metalAllocatedBytesBeforeLoad,
        metalAllocatedBytesAfterLoad: loaded.metalAllocatedBytesAfterLoad,
        metalAllocatedBytesAfterRelease: metalAllocatedBytesAfterRelease,
        hashes: hashes
      )
    )
  }
  var cacheReopenRuns: [CacheReopenRecord] = []
  if let payloadURL = options.exactWorkingCachePayload,
    let metadataURL = options.exactWorkingCacheMetadata
  {
    for trial in 1...options.cacheReopenIterations {
      let started = CFAbsoluteTimeGetCurrent()
      let metadataStarted = CFAbsoluteTimeGetCurrent()
      let metadata = try Metal4DSTEMResidentCacheIO.readMetadata(from: metadataURL)
      let metadataSeconds = CFAbsoluteTimeGetCurrent() - metadataStarted
      let validationStarted = CFAbsoluteTimeGetCurrent()
      try Metal4DSTEMResidentCacheIO.validatePayload(
        at: payloadURL,
        metadata: metadata,
        verifySHA256: false
      )
      let validationSeconds = CFAbsoluteTimeGetCurrent() - validationStarted
      guard metadata.payloadSHA256 == acceptedWorkingVolumeSHA256 else {
        throw BenchmarkError.invalid(
          "Prepared-cache reopen changed the exact working-volume SHA-256."
        )
      }
      cacheReopenRuns.append(
        CacheReopenRecord(
          trial: trial,
          state:
            "prepared_cache_metadata_and_identity_reopen_no_payload_hash_or_product_presentation",
          wallSeconds: CFAbsoluteTimeGetCurrent() - started,
          metadataReadSeconds: metadataSeconds,
          identityValidationSeconds: validationSeconds,
          payloadBytes: metadata.payloadBytes,
          payloadSHA256: metadata.payloadSHA256
        )
      )
    }
  }
  guard let provenance = acceptedProvenance, let first = records.first else {
    throw BenchmarkError.invalid("The benchmark produced no accepted load trial.")
  }
  let wall = records.map(\.wallSeconds)
  let reusedDestinationWall = records.filter(\.destinationReused).map(\.wallSeconds)
  let repeatedWall = reusedDestinationWall.isEmpty ? wall : reusedDestinationWall
  let distribution = Distribution(
    samples: repeatedWall.count,
    p50: percentile(repeatedWall, fraction: 0.50),
    p95: percentile(repeatedWall, fraction: 0.95),
    maximum: repeatedWall.max() ?? .nan
  )
  let reusedDestinationDistribution: Distribution? =
    reusedDestinationWall.isEmpty
    ? nil
    : Distribution(
      samples: reusedDestinationWall.count,
      p50: percentile(reusedDestinationWall, fraction: 0.50),
      p95: percentile(reusedDestinationWall, fraction: 0.95),
      maximum: reusedDestinationWall.max() ?? .nan
    )
  let cacheReopenDistribution: Distribution? =
    cacheReopenRuns.isEmpty
    ? nil
    : Distribution(
      samples: cacheReopenRuns.count,
      p50: percentile(cacheReopenRuns.map(\.wallSeconds), fraction: 0.50),
      p95: percentile(cacheReopenRuns.map(\.wallSeconds), fraction: 0.95),
      maximum: cacheReopenRuns.map(\.wallSeconds).max() ?? .nan
    )
  let workingShape =
    binnedPlan.map {
      [
        $0.binningProvenance.outputScanRows,
        $0.binningProvenance.outputScanColumns,
        $0.binningProvenance.outputDetectorRows,
        $0.binningProvenance.outputDetectorColumns,
      ]
    } ?? [
      productPlan.sourceScanRows,
      productPlan.sourceScanColumns,
      productPlan.sourceDetectorRows,
      productPlan.sourceDetectorColumns,
    ]
  let estimatedAllocatedMetalBytes: UInt64
  if options.exactWorkingCachePayload != nil, let binnedPlan {
    let outputSum = productPlan.estimatedAllocatedMetalBytesExcludingMappedSource
      .addingReportingOverflow(binnedPlan.shardPlan.maximumActualShardBytes)
    guard !outputSum.overflow else {
      throw BenchmarkError.invalid(
        "Bounded cache-build Metal allocation estimate overflows UInt64."
      )
    }
    let scratchSum = outputSum.partialValue.addingReportingOverflow(
      binnedPlan.maximumDetectorPartialScratchBytes
    )
    guard !scratchSum.overflow else {
      throw BenchmarkError.invalid(
        "Bounded cache-build scratch estimate overflows UInt64."
      )
    }
    estimatedAllocatedMetalBytes = scratchSum.partialValue
  } else {
    estimatedAllocatedMetalBytes =
      binnedPlan?.estimatedAllocatedMetalBytesExcludingMappedSource
      ?? productPlan.estimatedAllocatedMetalBytesExcludingMappedSource
  }
  let maximumRetainedSourceBufferBytes =
    binnedPlan?.maximumRetainedSourceBufferBytes
    ?? productPlan.maximumRetainedSourceBufferBytes
  let estimatedAllocatedMetalBytesIncludingSourceTransfer: UInt64
  switch sourceTransfer {
  case .memoryMapped:
    estimatedAllocatedMetalBytesIncludingSourceTransfer =
      estimatedAllocatedMetalBytes
  case .bufferedReadAhead:
    let total = estimatedAllocatedMetalBytes.addingReportingOverflow(
      maximumRetainedSourceBufferBytes
    )
    guard !total.overflow else {
      throw BenchmarkError.invalid(
        "Metal allocation including source transfer overflows UInt64."
      )
    }
    estimatedAllocatedMetalBytesIncludingSourceTransfer = total.partialValue
  }
  let summary = BenchmarkSummary(
    schema: "quantem.gpu.metal-4dstem-indexed-load-benchmark/v6",
    revision: options.revision,
    timestamp: ISO8601DateFormatter().string(from: Date()),
    host: ProcessInfo.processInfo.hostName,
    os: ProcessInfo.processInfo.operatingSystemVersionString,
    device: device.name,
    input: options.input.path,
    sourceIdentitySHA256: productPlan.sourceIdentitySHA256,
    sourceShape: [
      productPlan.sourceScanRows,
      productPlan.sourceScanColumns,
      productPlan.sourceDetectorRows,
      productPlan.sourceDetectorColumns,
    ],
    workingShape: workingShape,
    sourceDtype: productPlan.sourceDtype.rawValue,
    workingDtype: binnedPlan?.binningProvenance.outputDtype.rawValue
      ?? productPlan.stagingDtype.rawValue,
    scanBin: binnedPlan?.binningProvenance.scanBin ?? productPlan.scanBin,
    detectorBin: binnedPlan?.binningProvenance.detectorBin
      ?? productPlan.detectorBin,
    crop: "none",
    bandSource: bandSource,
    compressedSourceBytes: productPlan.compressedSourceBytes,
    logicalDecodedBytes: productPlan.logicalDecodedBytes,
    maximumDecodedWindowBytes: productPlan.maximumDecodedWindowBytes,
    requestedWindowScanRows: options.windowScanRows,
    maximumActualWindowBytes: productPlan.maximumActualWindowBytes,
    maximumDetectorPartialScratchBytes:
      binnedPlan?.maximumDetectorPartialScratchBytes,
    estimatedAllocatedMetalBytesExcludingMappedSource:
      estimatedAllocatedMetalBytes,
    estimatedAllocatedMetalBytesIncludingSourceTransfer:
      estimatedAllocatedMetalBytesIncludingSourceTransfer,
    sourceTransfer: sourceTransfer,
    maximumRetainedSourceBufferBytes: maximumRetainedSourceBufferBytes,
    maximumMappedCompressedBytes: productPlan.maximumMappedCompressedBytes,
    maximumMappedSourceBufferBytes: productPlan.maximumMappedSourceBufferBytes,
    maximumInFlightMappedSourceBytes: productPlan.maximumInFlightMappedSourceBytes,
    maximumInFlightCommandBuffers: productPlan.maximumInFlightCommandBuffers,
    maximumIndividualMetalBufferBytes:
      binnedPlan?.maximumIndividualMetalBufferBytes
      ?? productPlan.maximumIndividualMetalBufferBytes,
    workingPayloadBytes: binnedPlan?.workingPayloadBytes,
    shardCount: binnedPlan?.shardPlan.shards.count,
    maximumActualShardBytes: binnedPlan?.shardPlan.maximumActualShardBytes,
    valueRangeAuditSHA256: binnedPlan?.sourceAudit.auditSHA256,
    workingVolumeSHA256: acceptedWorkingVolumeSHA256,
    destinationStorage: first.destinationStorage,
    cachePayload: options.exactWorkingCachePayload?.path,
    cacheMetadata: options.exactWorkingCacheMetadata?.path,
    cacheReopenValidation: cacheReopenDistribution,
    cacheReopenRuns: cacheReopenRuns,
    processPeakResidentBytes: processPeakResidentBytes(),
    catalogSeconds: catalogSeconds,
    pipelineCompilationSeconds: compileSeconds,
    planSeconds: planSeconds,
    firstPackageEndToEndSeconds:
      catalogSeconds + compileSeconds + planSeconds + first.wallSeconds,
    reusesResidentDestinationShards: options.reuseWorkingDestinations,
    firstFreshDestinationWallSeconds:
      options.reuseWorkingDestinations ? first.wallSeconds : nil,
    reusedDestinationLoadWall: reusedDestinationDistribution,
    repeatedLoadWall: distribution,
    runs: records,
    provenance: provenance,
    binningProvenance: binnedPlan?.binningProvenance,
    samplingPropagation: binnedPlan?.samplingPropagation,
    shardPlan: binnedPlan?.shardPlan
  )
  try FileManager.default.createDirectory(
    at: options.outputDirectory,
    withIntermediateDirectories: true
  )
  for (name, data) in acceptedData {
    try data.write(
      to: options.outputDirectory.appendingPathComponent(name),
      options: .atomic
    )
  }
  let encoder = JSONEncoder()
  encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
  let summaryData = try encoder.encode(summary)
  try summaryData.write(
    to: options.outputDirectory.appendingPathComponent("summary.json"),
    options: .atomic
  )
  FileHandle.standardOutput.write(summaryData)
  FileHandle.standardOutput.write(Data("\n".utf8))
}

private func processPeakResidentBytes() -> UInt64? {
  var usage = rusage()
  guard getrusage(RUSAGE_SELF, &usage) == 0,
    usage.ru_maxrss >= 0
  else { return nil }
  return UInt64(usage.ru_maxrss)
}

do {
  try run()
} catch {
  FileHandle.standardError.write(Data("error: \(error.localizedDescription)\n".utf8))
  exit(2)
}
