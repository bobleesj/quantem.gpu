import CryptoKit
import Darwin
import Foundation
import Metal
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
      [--iterations N]

    Source-page state is not purged or inferred. The first trial is labeled as
    a first process encounter with prepared indexes and unspecified source pages;
    later trials are same-process repeats. Application UI time is not measured.
    """

  let input: URL
  let cacheDirectory: URL
  let outputDirectory: URL
  let revision: String
  let windowScanRows: Int
  let iterations: Int
  let bandsFile: URL?
  let allBands: Bool

  static func parse(_ arguments: [String]) throws -> Self {
    var values: [String: String] = [:]
    var flags = Set<String>()
    var index = 0
    while index < arguments.count {
      let argument = arguments[index]
      if argument == "--all-bands" {
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
    guard windowRows > 0, iterations > 0 else {
      throw BenchmarkError.usage(
        "--window-scan-rows and --iterations must be positive integers."
      )
    }
    return Self(
      input: URL(fileURLWithPath: input),
      cacheDirectory: URL(fileURLWithPath: cache, isDirectory: true),
      outputDirectory: URL(fileURLWithPath: output, isDirectory: true),
      revision: revision,
      windowScanRows: windowRows,
      iterations: iterations,
      bandsFile: bandsFile,
      allBands: allBands
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
  let wallSeconds: Double
  let gpuSeconds: Double
  let sourceMappingSeconds: Double
  let mappedCompressedSourceBytes: UInt64
  let maximumMappedCompressedSourceBytes: UInt64
  let maximumMappedSourceBufferBytes: UInt64
  let maximumDecodedSliceBytes: UInt64
  let maximumSourceCount: UInt32
  let pixelsAbove255: UInt64
  let hashes: [String: String]
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
  let estimatedAllocatedMetalBytesExcludingMappedSource: UInt64
  let maximumMappedCompressedBytes: UInt64
  let maximumMappedSourceBufferBytes: UInt64
  let maximumIndividualMetalBufferBytes: UInt64
  let processPeakResidentBytes: UInt64?
  let catalogSeconds: Double
  let pipelineCompilationSeconds: Double
  let planSeconds: Double
  let firstPackageEndToEndSeconds: Double
  let repeatedLoadWall: Distribution
  let runs: [RunRecord]
  let provenance: Metal4DSTEMIndexedLoadProvenance
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
  let planStarted = CFAbsoluteTimeGetCurrent()
  let plan = try Metal4DSTEMIndexedLoadPlan(
    source: source,
    maximumDecodedWindowBytes: windowBytes,
    detectorBands: bands
  )
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
  var acceptedData: [String: Data] = [:]
  for trial in 1...options.iterations {
    let result = try loader.loadExactProducts(source: source, plan: plan)
    let data = productData(result.products)
    let hashes = data.mapValues(sha256)
    if let acceptedHashes, hashes != acceptedHashes {
      throw BenchmarkError.invalid(
        "Exact product hashes changed between repeated load trials."
      )
    }
    if let acceptedProvenance {
      guard acceptedProvenance == result.provenance else {
        throw BenchmarkError.invalid(
          "Exact product provenance changed between repeated load trials."
        )
      }
    }
    acceptedHashes = hashes
    acceptedProvenance = result.provenance
    acceptedData = data
    records.append(
      RunRecord(
        trial: trial,
        state: trial == 1
          ? "first_process_prepared_index_source_pages_unspecified"
          : "same_process_repeat_source_pages_unspecified",
        wallSeconds: result.metrics.wallSeconds,
        gpuSeconds: result.metrics.gpuSeconds,
        sourceMappingSeconds: result.metrics.sourceMappingSeconds,
        mappedCompressedSourceBytes: result.metrics.mappedCompressedSourceBytes,
        maximumMappedCompressedSourceBytes:
          result.metrics.maximumMappedCompressedSourceBytes,
        maximumMappedSourceBufferBytes:
          result.metrics.maximumMappedSourceBufferBytes,
        maximumDecodedSliceBytes: result.metrics.maximumDecodedSliceBytes,
        maximumSourceCount: result.sourceAudit.maximumSourceCount,
        pixelsAbove255: result.sourceAudit.pixelsAbove255,
        hashes: hashes
      )
    )
  }
  guard let provenance = acceptedProvenance, let first = records.first else {
    throw BenchmarkError.invalid("The benchmark produced no accepted load trial.")
  }
  let wall = records.map(\.wallSeconds)
  let distribution = Distribution(
    samples: wall.count,
    p50: percentile(wall, fraction: 0.50),
    p95: percentile(wall, fraction: 0.95),
    maximum: wall.max() ?? .nan
  )
  let summary = BenchmarkSummary(
    schema: "quantem.gpu.metal-4dstem-indexed-load-benchmark/v1",
    revision: options.revision,
    timestamp: ISO8601DateFormatter().string(from: Date()),
    host: ProcessInfo.processInfo.hostName,
    os: ProcessInfo.processInfo.operatingSystemVersionString,
    device: device.name,
    input: options.input.path,
    sourceIdentitySHA256: plan.sourceIdentitySHA256,
    sourceShape: [
      plan.sourceScanRows,
      plan.sourceScanColumns,
      plan.sourceDetectorRows,
      plan.sourceDetectorColumns,
    ],
    workingShape: [
      plan.sourceScanRows,
      plan.sourceScanColumns,
      plan.sourceDetectorRows,
      plan.sourceDetectorColumns,
    ],
    sourceDtype: plan.sourceDtype.rawValue,
    workingDtype: plan.stagingDtype.rawValue,
    scanBin: plan.scanBin,
    detectorBin: plan.detectorBin,
    crop: "none",
    bandSource: bandSource,
    compressedSourceBytes: plan.compressedSourceBytes,
    logicalDecodedBytes: plan.logicalDecodedBytes,
    maximumDecodedWindowBytes: plan.maximumDecodedWindowBytes,
    requestedWindowScanRows: options.windowScanRows,
    maximumActualWindowBytes: plan.maximumActualWindowBytes,
    estimatedAllocatedMetalBytesExcludingMappedSource:
      plan.estimatedAllocatedMetalBytesExcludingMappedSource,
    maximumMappedCompressedBytes: plan.maximumMappedCompressedBytes,
    maximumMappedSourceBufferBytes: plan.maximumMappedSourceBufferBytes,
    maximumIndividualMetalBufferBytes: plan.maximumIndividualMetalBufferBytes,
    processPeakResidentBytes: processPeakResidentBytes(),
    catalogSeconds: catalogSeconds,
    pipelineCompilationSeconds: compileSeconds,
    planSeconds: planSeconds,
    firstPackageEndToEndSeconds:
      catalogSeconds + compileSeconds + planSeconds + first.wallSeconds,
    repeatedLoadWall: distribution,
    runs: records,
    provenance: provenance
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
