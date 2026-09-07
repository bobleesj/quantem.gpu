import CryptoKit
import Foundation
import Metal
import Metal4DSTEMStreamingIO
import Native4DSTEMIO

/// Original-file reload timing, independent of any frontend or display cache.
@main
enum MetalOriginalHDF5Benchmark {
  struct Options {
    let input: URL
    let indexDirectory: URL
    let planDirectory: URL?
    let repeats: Int
    let budget: UInt64
    let reuseProducts: Bool
    let oracle: [String: String]

    init() throws {
      var args = Array(CommandLine.arguments.dropFirst())
      guard args.count >= 2 else {
        throw failure(
          "Usage: metal-original-hdf5-benchmark INPUT INDEX_DIRECTORY "
            + "[--repeats N] [--budget-bytes N] [--plan-directory DIR] "
            + "[--reuse-products] [--oracle JSON]"
        )
      }
      input = URL(fileURLWithPath: args.removeFirst())
      indexDirectory = URL(fileURLWithPath: args.removeFirst())
      var repeats = 5
      var budget: UInt64 = 4 << 30
      var plan: URL?
      var oracle: [String: String] = [:]
      var reuse = false
      while !args.isEmpty {
        let flag = args.removeFirst()
        if flag == "--reuse-products" {
          reuse = true
          continue
        }
        guard !args.isEmpty else { throw failure("Missing value for \(flag)") }
        let value = args.removeFirst()
        switch flag {
        case "--repeats":
          guard let count = Int(value), (1...500).contains(count) else {
            throw failure("Repeats must be in 1...500")
          }
          repeats = count
        case "--budget-bytes":
          guard let bytes = UInt64(value), bytes > 0 else {
            throw failure("Budget must be a positive byte count")
          }
          budget = bytes
        case "--plan-directory": plan = URL(fileURLWithPath: value)
        case "--oracle":
          oracle = try JSONDecoder().decode(
            [String: String].self, from: Data(contentsOf: URL(fileURLWithPath: value)))
          guard !oracle.isEmpty,
            oracle.allSatisfy({ isSHA256($0.key) && isSHA256($0.value) })
          else { throw failure("Oracle must map source identities to full-count SHA-256 values") }
        default: throw failure("Unknown option \(flag)")
        }
      }
      self.repeats = repeats
      self.budget = budget
      planDirectory = plan
      reuseProducts = reuse
      self.oracle = oracle
    }
  }

  static func main() {
    do {
      try run()
    } catch {
      try? emit(["phase": "failed", "complete": false])
      // Error details may include a local input path; keep stderr private.
      fputs("\(error.localizedDescription)\n", stderr)
      exit(1)
    }
  }

  static func run() throws {
    let options = try Options()
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw failure("A physical Metal device is required")
    }
    if let directory = options.planDirectory {
      try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
    }
    let started = CFAbsoluteTimeGetCurrent()
    let catalog = try Native4DSTEMCatalogBuilder(cacheDirectory: options.indexDirectory)
      .prepare(input: options.input)
    let catalogSeconds = CFAbsoluteTimeGetCurrent() - started
    guard !catalog.datasets.isEmpty else { throw failure("No supported acquisitions found") }
    let identities = catalog.datasets.compactMap(\.sourceIdentitySHA256)
    guard Set(identities).count == catalog.datasets.count else {
      throw failure("Each catalog acquisition must have a distinct source identity")
    }
    try emit([
      "phase": "setup", "schema": "quantem-gpu-original-packed-benchmark/v1",
      "device": device.name, "physical_memory_bytes": ProcessInfo.processInfo.physicalMemory,
      "os_version": ProcessInfo.processInfo.operatingSystemVersionString,
      "catalog_seconds": catalogSeconds, "acquisitions": identities.count,
      "repeats": options.repeats, "budget_bytes": options.budget,
      "reuse_products": options.reuseProducts, "packing_plan_enabled": options.planDirectory != nil,
      "cold_io_claim": false, "ui_present_measured": false,
      "timing_boundary": "indexed source open through complete packed resident load return",
      "cache_state":
        "index/layout metadata may be reused; OS pages uncontrolled; no 4D count cache",
      "full_count_hash_encoding":
        "scan-row/scan-column/detector-row/detector-column uint32 little-endian",
    ])
    var products: [String: MetalCompactH5ExactDPCMoments] = [:]
    var fingerprints: [String: [String]] = [:]
    var audited = Set<String>()
    var timings: [Double] = []
    var releasedBaseline: UInt64?
    for cycle in 0..<options.repeats {
      for dataset in catalog.datasets {
        let identity = dataset.sourceIdentitySHA256!
        try autoreleasepool {
          let loadStarted = CFAbsoluteTimeGetCurrent()
          let indexed = try Native4DSTEMIndexedSource.open(dataset: dataset)
          let resident = try MetalCompactH5Loader.load(
            source: indexed, device: device, maximumAdditionalBytes: options.budget,
            preparedDPC: products[identity],
            packingPlanURL: options.planDirectory?.appendingPathComponent(identity + ".qgplan")
          )
          defer { resident.releaseResidentStorage() }
          let seconds = CFAbsoluteTimeGetCurrent() - loadStarted
          try emit([
            "phase": "resident_ready", "cycle": cycle, "source_identity": identity,
            "seconds": seconds, "independent_validation_pending": true,
          ])
          let capabilities = try Metal4DSTEMResidentCapabilities.compact(resident)
          let receipt = capabilities.residentReceipt
          guard receipt.losslessExact, receipt.scanBin == 1, receipt.detectorBin == 1,
            receipt.crop == nil, receipt.sourceShape == receipt.workingShape,
            receipt.detectorMaskCount == 0
          else { throw failure("Resident did not preserve the full unmodified acquisition") }
          let metadata = resident.metadata
          let frames = [0, metadata.scanCount / 2, metadata.scanCount - 1]
          let observed = try frames.map { frame in
            try digest(
              resident.extractDiffraction(
                scanRow: frame / metadata.scanColumns, scanColumn: frame % metadata.scanColumns))
          }
          if let prior = fingerprints[identity], prior != observed {
            throw failure("Return visit changed diffraction counts")
          }
          fingerprints[identity] = observed
          let dpc = try resident.preparedDPCMomentValues()
          if let prior = products[identity], prior != dpc {
            throw failure("Return visit changed DPC sums")
          }
          if options.reuseProducts { products[identity] = dpc }
          if !options.oracle.isEmpty && !audited.contains(identity) {
            guard let expected = options.oracle[identity] else {
              throw failure("Independent full-count oracle missing for an acquisition")
            }
            let auditStarted = CFAbsoluteTimeGetCurrent()
            var hash = SHA256()
            for frame in 0..<metadata.scanCount {
              try autoreleasepool {
                let values = try resident.extractDiffraction(
                  scanRow: frame / metadata.scanColumns, scanColumn: frame % metadata.scanColumns)
                values.withUnsafeBytes { hash.update(bufferPointer: $0) }
              }
            }
            let observed = hex(hash.finalize())
            guard observed == expected else {
              throw failure("Independent full-count parity failed")
            }
            audited.insert(identity)
            try emit([
              "phase": "full_count_parity", "source_identity": identity,
              "sha256_u32_le": observed, "pass": true,
              "seconds_excluded_from_load": CFAbsoluteTimeGetCurrent() - auditStarted,
            ])
          }
          let metrics = resident.loadMetrics
          timings.append(seconds)
          try emit([
            "phase": "resident", "cycle": cycle, "source_identity": identity,
            "shape": receipt.sourceShape, "source_dtype": receipt.sourceDtype,
            "working_dtype": receipt.workingDtype, "seconds": seconds,
            "resident_bytes": metrics.totalResidentBytes,
            "planned_additional_bytes": metrics.plannedAdditionalBytes,
            "device_allocated_after_bytes": device.currentAllocatedSize,
            "source_read_seconds": metrics.sourceReadMilliseconds / 1000,
            "gpu_decode_seconds": metrics.gpuDecodeMilliseconds / 1000,
            "gpu_decode_header_seconds": metrics.gpuDecodeAndHeaderMilliseconds / 1000,
            "gpu_decode_packing_seconds": metrics.gpuDecodeAndPackingMilliseconds.map {
              $0 / 1000 as Any
            } ?? NSNull(),
            "gpu_preparation_seconds": metrics.gpuPreparationMilliseconds / 1000,
            "reused_dpc": metrics.reusedPreparedDPC, "sample_hashes": observed,
            "independent_full_count_parity": audited.contains(identity), "resident_count": 1,
          ])
        }
        let released = UInt64(device.currentAllocatedSize)
        if let baseline = releasedBaseline, released > baseline + (64 << 20) {
          throw failure("Released device allocations grew by more than 64 MiB")
        }
        releasedBaseline = releasedBaseline ?? released
        try emit(["phase": "released", "resident_count": 0, "device_allocated_bytes": released])
      }
    }
    let sorted = timings.sorted()
    let middle = sorted.count / 2
    let median =
      sorted.count.isMultiple(of: 2)
      ? (sorted[middle - 1] + sorted[middle]) / 2 : sorted[middle]
    try emit([
      "phase": "complete", "loads": sorted.count, "p50_seconds": median,
      "p95_seconds": sorted[Int(ceil(Double(sorted.count) * 0.95)) - 1],
      "max_seconds": sorted.last!, "first_load_included": true,
      "independently_audited_acquisitions": audited.count,
    ])
  }

  static func failure(_ message: String) -> NSError { NSError(domain: message, code: 1) }
  static func isSHA256(_ value: String) -> Bool {
    value.count == 64 && value.allSatisfy { "0123456789abcdef".contains($0) }
  }
  static func hex(_ value: SHA256.Digest) -> String {
    value.map { String(format: "%02x", $0) }.joined()
  }
  static func digest(_ values: [UInt32]) -> String {
    values.withUnsafeBytes { hex(SHA256.hash(data: Data($0))) }
  }
  static func emit(_ record: [String: Any]) throws {
    print(
      String(
        data: try JSONSerialization.data(withJSONObject: record, options: [.sortedKeys]),
        encoding: .utf8)!)
    fflush(stdout)
  }
}
