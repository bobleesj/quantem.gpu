import CryptoKit
import Darwin
import Foundation
import Metal
@_spi(PairedRuntimeTANSPrototype) import Metal4DSTEMStreamingIO
@_spi(FourWayCheckpointPrototype) import Metal4DSTEMStreamingIO
import Native4DSTEMIO

@main
@available(macOS 15.0, *)
enum MetalPairedRuntimeTANSSeriesBenchmark {
  static func main() async throws {
    let arguments = Array(CommandLine.arguments.dropFirst())
    guard arguments.count == 2 else {
      throw failure("Usage: metal-paired-runtime-tans-series-benchmark FOLDER INDEX_DIRECTORY")
    }
    guard let device = MTLCreateSystemDefaultDevice() else {
      throw failure("A physical Metal device is required")
    }
    let catalog = try Native4DSTEMCatalogBuilder(
      cacheDirectory: URL(fileURLWithPath: arguments[1])
    ).prepare(input: URL(fileURLWithPath: arguments[0]))
    let indexed = try catalog.datasets.map(Native4DSTEMIndexedSource.open(dataset:))
    guard indexed.count == 7 else {
      throw failure("The paired-runtime acceptance benchmark requires exactly seven acquisitions")
    }
    let runOptimizationExperiment = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_EXPERIMENT"] == "1"
    let runResidentLoop = ProcessInfo.processInfo.environment[
      "QGPU_ANS_RESIDENT_LOOP"] == "1"
    let polarIndexAtStartup = ProcessInfo.processInfo.environment[
      "QGPU_PAIRED_RUNTIME_POLAR_INDEX"] == "1"
    let reader32Only = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_READER32_ONLY"] == "1"
    let polarOnly = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_POLAR_ONLY"] == "1"
    let polarQueryScan512Only = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_POLAR_QUERY_SCAN512"] == "1"
    let macro2Only = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_MACRO2"] == "1"
    if runResidentLoop {
      // Prepare every optional pipeline once. The loop changes selection flags
      // between queries, while the seven compact uint16 residents stay live.
      setenv("QGPU_PAIRED_RUNTIME_PREPARE_PARTIAL_STORES", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_PREPARE_PACKET_SPLITS", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_PREPARE_REUSE_WORD", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_PREPARE_REGISTER_SUMS", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_PREPARE_PLAIN_SUMS", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_PREPARE_DECODE_CHECKSUM", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_PREPARE_LAZY_REFILL", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_SPARSE_SPLIT", "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_DENSE_COMPACTION", "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_MACRO", macro2Only ? "1" : "0", 1)
      if macro2Only {
      setenv("QGPU_PAIRED_RUNTIME_MACRO_LOOKAHEAD_BITS", "2", 1)
      }
      setenv("QGPU_PAIRED_RUNTIME_COOPERATIVE", "0", 1)
      setenv(
        "QGPU_PAIRED_RUNTIME_READER32",
        ProcessInfo.processInfo.environment["QGPU_PAIRED_RUNTIME_PREPARE_READER32"] == "1"
          ? "1" : "0",
        1)
      setenv("QGPU_PAIRED_RUNTIME_POLAR_INDEX",
        (polarIndexAtStartup || polarQueryScan512Only) ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_PREPARE_POLAR_QUERY_SCAN512",
        polarQueryScan512Only ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT", "packet-groups", 1)
    } else if runOptimizationExperiment {
      // Optional specializations are created at resident initialization, while
      // selection between experiment arms is reevaluated per query.
      setenv("QGPU_PAIRED_RUNTIME_SPARSE_SPLIT", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_DENSE_COMPACTION", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_MACRO", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_COOPERATIVE", "1", 1)
      setenv("QGPU_PAIRED_RUNTIME_READER32", reader32Only ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_POLAR_INDEX",
        (polarOnly || polarQueryScan512Only) ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_PREPARE_POLAR_QUERY_SCAN512",
        polarQueryScan512Only ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT", "packet-groups", 1)
    }
    let started = CFAbsoluteTimeGetCurrent()
    let concurrentLoads = Int(
      ProcessInfo.processInfo.environment["QGPU_PAIRED_RUNTIME_CONCURRENT_LOADS"] ?? "3") ?? 3
    let residents = try await MetalPairedRuntimeTANSSeriesLoader.load(
      sources: indexed, device: device, maximumConcurrentLoads: concurrentLoads,
      maximumAdditionalBytesPerLoad: ProcessInfo.processInfo.physicalMemory)
    let loadSeconds = CFAbsoluteTimeGetCurrent() - started
    if runResidentLoop {
      try await residentLoop(
        residents: residents, indexed: indexed, device: device, loadSeconds: loadSeconds,
        concurrentLoads: concurrentLoads,
        polarIndexPrepared: polarIndexAtStartup || polarQueryScan512Only)
      return
    }
    if runOptimizationExperiment {
      setenv("QGPU_PAIRED_RUNTIME_SPARSE_SPLIT", "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_DENSE_COMPACTION", "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_MACRO", "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_COOPERATIVE", "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_READER32", "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_POLAR_INDEX", "0", 1)
      try await optimizationExperiment(
        residents: residents, indexed: indexed, device: device,
        loadSeconds: loadSeconds, concurrentLoads: concurrentLoads)
      return
    }

    for resident in residents {
      for record in 0..<16 {
        let scan = record * 16_384
        _ = try resident.extractRawDiffraction(scanRow: scan / 512, scanColumn: scan % 512)
      }
    }
    var priorityDP: [Double] = []
    var allDP: [Double] = []
    for trial in 0..<60 {
      let row = (trial * 37) % 512
      let column = (trial * 61) % 512
      var tick = CFAbsoluteTimeGetCurrent()
      _ = try residents[trial % residents.count].extractRawDiffraction(
        scanRow: row, scanColumn: column)
      priorityDP.append((CFAbsoluteTimeGetCurrent() - tick) * 1_000)
      tick = CFAbsoluteTimeGetCurrent()
      for resident in residents {
        _ = try resident.extractRawDiffraction(scanRow: row, scanColumn: column)
      }
      allDP.append((CFAbsoluteTimeGetCurrent() - tick) * 1_000)
    }

    // Exercise the same broad masks used by the native interaction harness.
    // Alternating translated masks forces a realistic large detector delta;
    // changing only one radius would measure a much smaller boundary update.
    let bfMasks = (0..<4).map { offset in
      circularMask(centerColumnOffset: offset * 8, centerRowOffset: offset * 5,
        innerRadius: 0, outerRadius: 46)
    }
    let abfMasks = (0..<4).map { offset in
      circularMask(centerColumnOffset: offset * 8, centerRowOffset: offset * 5,
        innerRadius: 24, outerRadius: 64)
    }
    let adfMasks = (0..<4).map { offset in
      circularMask(centerColumnOffset: offset * 8, centerRowOffset: offset * 5,
        innerRadius: 48, outerRadius: 94)
    }
    let bf = try await detectorTrials(residents: residents, masks: bfMasks)
    let abf = try await detectorTrials(residents: residents, masks: abfMasks)
    let adf = try await detectorTrials(residents: residents, masks: adfMasks)
    let output: [String: Any] = [
      "schema": "quantem-gpu-paired-runtime-tans-series-benchmark/v1",
      "acquisition_count": residents.count,
      "series_load_seconds": loadSeconds,
      "concurrent_loads": concurrentLoads,
      "acquisition_load_seconds": residents.map(\.loadMetrics.totalSeconds),
      "series_resident_bytes": residents.reduce(0) { $0 + $1.residentBytes },
      "metal_current_allocated_bytes": device.currentAllocatedSize,
      "priority_dp_p95_milliseconds": percentile(priorityDP, 0.95),
      "all_dp_p95_milliseconds": percentile(allDP, 0.95),
      "bf_wall_p95_milliseconds": percentile(bf, 0.95),
      "abf_wall_p95_milliseconds": percentile(abf, 0.95),
      "adf_wall_p95_milliseconds": percentile(adf, 0.95),
      "crop": NSNull(),
      "scan_bin": 1,
      "detector_bin": 1,
      "saved_ans_file": false,
    ]
    print(
      String(
        data: try JSONSerialization.data(withJSONObject: output, options: [.sortedKeys]),
        encoding: .utf8)!)
  }

  private struct ResidentLoopConfiguration {
    var mode = "raw"
    var kernel = "packet-owner2"
    var polarQueryVariant = "packet-groups"
    var partialGroups = 8
    var chooseBase = false
    var partialStores = false
    var streamsPerLane = "2"
    var packetSplits = "1"
    var reader32 = false
    var batch = true
    var boundedConcurrency = 7
    var reuseWord = false
    var registerSums = false
    var history = false
    var historyBase = false
    var plainSums = false
    var trustedTable = false
    var profile = false
    var lazyRefill = false
    var jointPlan = false
    var simdEntropyFastPath = false
    var vectorPairReduction = false
    var macro = false

    mutating func apply(_ command: [String: Any]) throws {
      if let value = try ResidentLoopBenchmark.bool(command, "joint_plan") {
        jointPlan = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "simd_entropy_fast_path") {
        simdEntropyFastPath = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "vector_pair_reduction") {
        vectorPairReduction = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "macro") {
        macro = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "reader32") {
        reader32 = value
      }
      if let mode = ResidentLoopBenchmark.value(command, "mode") as? String {
        guard ["raw", "indexed"].contains(mode) else {
          throw failure("resident loop mode must be raw or indexed")
        }
        self.mode = mode
      }
      if let kernel = (ResidentLoopBenchmark.value(command, "kernel")
        ?? ResidentLoopBenchmark.value(command, "detector_kernel")) as? String {
        guard ["packet-owner2", "partials", "adaptive-partials"].contains(kernel) else {
          throw failure("resident loop kernel must be packet-owner2, partials, or adaptive-partials")
        }
        self.kernel = kernel
      }
      if let rawVariant = ResidentLoopBenchmark.value(command, "polar_query_variant") {
        guard let variant = rawVariant as? String,
          ["packet-groups", "scan512", "scan512-stripe2", "scan512-stripe4",
            "scan512-stripe8", "scan512-field4", "scan512-contiguous-quad"]
            .contains(variant)
        else {
          throw failure(
            "resident loop polar_query_variant must be packet-groups, scan512, "
              + "scan512-stripe2, scan512-stripe4, scan512-stripe8, scan512-field4, "
              + "or scan512-contiguous-quad")
        }
        polarQueryVariant = variant
      }
      if let number = ResidentLoopBenchmark.number(command, "partial_groups") {
        guard [8, 16, 32].contains(number) else {
          throw failure("resident loop partial_groups must be 8, 16, or 32")
        }
        partialGroups = number
      }
      if let value = try ResidentLoopBenchmark.bool(command, "choose_base") {
        chooseBase = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "partial_stores") {
        partialStores = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "batch") {
        batch = value
      }
      if let number = ResidentLoopBenchmark.number(command, "bounded_concurrency") {
        guard [1, 2, 4, 7].contains(number) else {
          throw failure("resident loop bounded_concurrency must be 1, 2, 4, or 7")
        }
        boundedConcurrency = number
      }
      if let value = try ResidentLoopBenchmark.bool(command, "reuse_word") {
        reuseWord = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "register_sums") {
        registerSums = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "history") {
        history = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "history_base") {
        historyBase = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "plain_sums") {
        plainSums = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "trusted_table") {
        trustedTable = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "profile") {
        profile = value
      }
      if let value = try ResidentLoopBenchmark.bool(command, "lazy_refill") {
        lazyRefill = value
      }
      if let value = ResidentLoopBenchmark.value(command, "streams_per_lane") {
        let text = value is NSNumber ? String(describing: (value as! NSNumber).intValue) : "\(value)"
        guard ["1", "2", "4"].contains(text) else {
          throw failure("resident loop streams_per_lane must be 1, 2, or 4")
        }
        streamsPerLane = text
      }
      if let value = ResidentLoopBenchmark.value(command, "packet_splits") {
        let text = value is NSNumber ? String(describing: (value as! NSNumber).intValue) : "\(value)"
        guard ["1", "2", "4", "8"].contains(text) else {
          throw failure("resident loop packet_splits must be 1, 2, 4, or 8")
        }
        packetSplits = text
      }
      guard !simdEntropyFastPath || (mode == "indexed" && kernel == "packet-owner2"
        && streamsPerLane == "2" && packetSplits == "1" && !jointPlan
        && !reuseWord && !registerSums && !plainSums && !trustedTable && !lazyRefill)
      else {
        throw failure(
          "SIMD entropy fast path requires indexed packet-owner2 with two streams/lane, "
            + "one packet split, and other optimizations disabled")
      }
      guard !reader32 || (mode == "indexed" && kernel == "packet-owner2"
        && streamsPerLane == "2" && packetSplits == "1" && !macro
        && !reuseWord && !registerSums && !plainSums && !trustedTable && !lazyRefill
        && !simdEntropyFastPath)
      else {
        throw failure(
          "reader32 requires indexed packet-owner2 with two streams/lane, one packet split, "
            + "macro/trusted-table and other reader specializations disabled")
      }
      guard !vectorPairReduction || (mode == "indexed" && kernel == "packet-owner2"
        && streamsPerLane == "2" && packetSplits == "1" && !reader32 && !macro
        && !reuseWord && !registerSums && !plainSums && !lazyRefill && !jointPlan
        && !simdEntropyFastPath && !history && !historyBase
        && ProcessInfo.processInfo.environment[
          "QGPU_PAIRED_RUNTIME_PREPARE_VECTOR_PAIR_REDUCTION"] == "1")
      else {
        throw failure(
          "vector pair reduction requires its prepared indexed two-stream packet-owner2 "
            + "pipeline with other detector specializations disabled")
      }
    }

    func control() -> ResidentLoopConfiguration {
      var result = self
      result.kernel = "packet-owner2"
      result.chooseBase = false
      result.partialStores = false
      result.packetSplits = "1"
      result.reader32 = false
      result.reuseWord = false
      result.registerSums = false
      result.history = false
      result.historyBase = false
      result.plainSums = false
      result.trustedTable = false
      result.profile = false
      result.lazyRefill = false
      result.jointPlan = false
      result.simdEntropyFastPath = false
      result.vectorPairReduction = false
      result.macro = false
      return result
    }

    var json: [String: Any] {
      var result: [String: Any] = [
        "mode": mode, "kernel": kernel, "polar_query_variant": polarQueryVariant,
        "partial_groups": partialGroups,
        "choose_base": chooseBase, "partial_stores": partialStores,
        "streams_per_lane": Int(streamsPerLane)!, "packet_splits": Int(packetSplits)!,
        "batch": batch, "bounded_concurrency": boundedConcurrency, "reuse_word": reuseWord,
        "register_sums": registerSums,
        "history": history,
        "history_base": historyBase,
        "plain_sums": plainSums,
        "trusted_table": trustedTable,
        "profile": profile,
        "lazy_refill": lazyRefill,
        "joint_plan": jointPlan,
        "simd_entropy_fast_path": simdEntropyFastPath,
        "macro": macro,
      ]
      // Preserve the established response schema for every default/control arm.
      // The opt-in key appears only for the experiment that actually enables it.
      if reader32 { result["reader32"] = true }
      if vectorPairReduction { result["vector_pair_reduction"] = true }
      return result
    }
  }

  private struct ResidentLoopRun {
    let maps: [String: [[UInt32]]]
    let samples: [[String: Any]]
  }

  private enum ResidentLoopBenchmark {
    static func value(_ command: [String: Any], _ key: String) -> Any? {
      command[key] ?? command[key.replacingOccurrences(of: "_", with: "-")]
    }

    static func number(_ command: [String: Any], _ key: String) -> Int? {
      if let number = value(command, key) as? NSNumber { return number.intValue }
      if let text = value(command, key) as? String { return Int(text) }
      return nil
    }

    static func bool(_ command: [String: Any], _ key: String) throws -> Bool? {
      guard let value = value(command, key) else { return nil }
      if let result = value as? Bool { return result }
      if let number = value as? NSNumber, number.intValue == 0 || number.intValue == 1 {
        return number.intValue == 1
      }
      if let text = value as? String, text == "0" || text == "1" {
        return text == "1"
      }
      throw failure("resident loop \(key) must be a boolean or 0/1")
    }

    static func setEnvironment(_ configuration: ResidentLoopConfiguration) {
      setenv("QGPU_PAIRED_RUNTIME_JOINT_PLAN", configuration.jointPlan ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_POLAR_INDEX", configuration.mode == "indexed" ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT", configuration.polarQueryVariant, 1)
      setenv("QGPU_PAIRED_RUNTIME_DETECTOR_KERNEL", configuration.kernel, 1)
      setenv("QGPU_PAIRED_RUNTIME_PARTIAL_MAX_GROUPS", "\(configuration.partialGroups)", 1)
      setenv("QGPU_PAIRED_RUNTIME_CHOOSE_BASE", configuration.chooseBase ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_SPARSE_SPLIT", "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_DENSE_COMPACTION", "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_MACRO", configuration.macro ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_COOPERATIVE", "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_READER32", configuration.reader32 ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_PARTIAL_STORES", configuration.partialStores ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_STREAMS_PER_LANE", configuration.streamsPerLane, 1)
      setenv("QGPU_PAIRED_RUNTIME_PACKET_SPLITS", configuration.packetSplits, 1)
      setenv("QGPU_PAIRED_RUNTIME_REUSE_WORD", configuration.reuseWord ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_REGISTER_SUMS", configuration.registerSums ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_HISTORY", configuration.history ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_HISTORY_BASE", configuration.historyBase ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_PLAIN_SUMS", configuration.plainSums ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_TRUSTED_TABLE", configuration.trustedTable ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_PROFILE", configuration.profile ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_LAZY_REFILL", configuration.lazyRefill ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_PLAIN_SCRATCH", "0", 1)
      setenv(
        "QGPU_PAIRED_RUNTIME_SIMD_ENTROPY_FAST_PATH",
        configuration.simdEntropyFastPath ? "1" : "0", 1)
      setenv(
        "QGPU_PAIRED_RUNTIME_VECTOR_PAIR_REDUCTION",
        configuration.vectorPairReduction ? "1" : "0", 1)
    }

    static func masks(_ command: [String: Any]) throws -> [ExperimentMask] {
      guard let requested = value(command, "masks") ?? value(command, "mask_names") else {
        return experimentMasks()
      }
      guard let names = requested as? [String], !names.isEmpty else {
        throw failure("resident loop masks must be a non-empty array of experiment mask names")
      }
      var available = experimentMasks()
      if names.contains(where: { $0.hasPrefix("adf-drag-column-") }) {
        available.append(contentsOf: (1...20).map { column in
          ExperimentMask(
            name: "adf-drag-column-\(column)",
            values: circularMask(
              centerColumnOffset: column, centerRowOffset: 0,
              innerRadius: 48, outerRadius: 94))
        })
      }
      let selected = names.map { name in available.first { $0.name == name } }
      guard selected.allSatisfy({ $0 != nil }) else {
        throw failure("resident loop masks must use names from experimentMasks()")
      }
      return selected.compactMap { $0 }
    }

    static func run(
      residents: [MetalPairedRuntimeTANSResidentSource], masks: [ExperimentMask], cycles: Int,
      configuration: ResidentLoopConfiguration
    ) async throws -> ResidentLoopRun {
      var maps: [String: [[UInt32]]] = [:]
      var samples: [[String: Any]] = []
      let zero = [UInt8](repeating: 0, count: 192 * 192)
      for cycle in 0..<cycles {
        try await updateAll(residents: residents, mask: zero)
        var currentMaps: [String: [[UInt32]]] = [:]
        for mask in masks {
          let effectiveConfiguration = configuration
          setEnvironment(effectiveConfiguration)
          let planProfileBefore = effectiveConfiguration.profile
            ? MetalPairedRuntimeTANSResidentSource.polarPlanCacheProfileSnapshot() : [:]
          let started = CFAbsoluteTimeGetCurrent()
          let results = try await experimentUpdateAll(
            residents: residents, mask: mask.values, batched: configuration.batch,
            boundedConcurrency: configuration.boundedConcurrency)
          let allWall = (CFAbsoluteTimeGetCurrent() - started) * 1_000
          let updateProfiles = effectiveConfiguration.profile
            ? residents.map(\.lastUpdateProfile) : []
          let planProfileAfter = effectiveConfiguration.profile
            ? MetalPairedRuntimeTANSResidentSource.polarPlanCacheProfileSnapshot() : [:]
          var planProfileDelta: [String: Double] = [:]
          for (key, value) in planProfileAfter where key != "shared_cache_enabled" {
            planProfileDelta[key] = value - (planProfileBefore[key] ?? 0)
          }
          if let reference = maps[mask.name], results.map(\.values) != reference {
            throw failure("Full-map parity changed on repeated mask \(mask.name) in cycle \(cycle)")
          }
          currentMaps[mask.name] = results.map(\.values)
          if cycle == 0 { maps[mask.name] = currentMaps[mask.name]! }
          for result in results {
            var sample: [String: Any] = [
              "cycle": cycle, "mask": mask.name, "source": result.source,
              "wall_ms": result.wallMilliseconds, "gpu_ms": result.gpuMilliseconds,
              "all_seven_wall_ms": allWall, "changed_pixels": result.changed,
              "excluded_pixels": result.excluded, "polar_field_count": result.polarFields,
              "polar_residual_count": result.polarResiduals,
              "sha256_u32_le": hash(result.values),
              "effective_kernel": effectiveConfiguration.kernel,
              "history_hit": residents[result.source].lastHistoryHit,
              "history_base": residents[result.source].lastHistoryBase,
            ]
            if effectiveConfiguration.profile {
              sample["update_profile"] = updateProfiles[result.source]
              sample["polar_plan_cache_profile"] = planProfileDelta
              sample["shared_polar_plan_cache_enabled"] =
                planProfileAfter["shared_cache_enabled"] ?? 0
            }
            samples.append(sample)
          }
          setEnvironment(configuration)
        }
        guard parity(actual: currentMaps, expected: maps).passed else {
          throw failure("Full-map parity changed within cycle \(cycle)")
        }
      }
      return ResidentLoopRun(maps: maps, samples: samples)
    }

    static func hashes(_ maps: [String: [[UInt32]]]) -> [String: [String]] {
      maps.mapValues { $0.map(hash) }
    }

    static func parity(
      actual: [String: [[UInt32]]], expected: [String: [[UInt32]]]
    ) -> (passed: Bool, mismatches: [[String: Any]]) {
      var mismatches: [[String: Any]] = []
      for name in expected.keys.sorted() {
        guard let actualMaps = actual[name], let expectedMaps = expected[name] else {
          mismatches.append(["mask": name, "reason": "missing full map"])
          continue
        }
        for source in expectedMaps.indices {
          guard source < actualMaps.count, actualMaps[source] == expectedMaps[source] else {
            let expectedValues = expectedMaps[source]
            let actualValues = source < actualMaps.count ? actualMaps[source] : []
            let first = zip(actualValues, expectedValues).enumerated().first {
              $0.element.0 != $0.element.1
            }?.offset
              ?? min(actualValues.count, expectedValues.count)
            mismatches.append([
              "mask": name, "source": source, "first_pixel": first,
              "expected": first < expectedValues.count ? expectedValues[first] : NSNull(),
              "actual": first < actualValues.count ? actualValues[first] : NSNull(),
            ])
            continue
          }
        }
      }
      return (mismatches.isEmpty, mismatches)
    }
  }

  private static func residentLoop(
    residents: [MetalPairedRuntimeTANSResidentSource],
    indexed: [Native4DSTEMIndexedSource], device: MTLDevice,
    loadSeconds: Double, concurrentLoads: Int, polarIndexPrepared: Bool
  ) async throws {
    var configuration = ResidentLoopConfiguration()
    configuration.mode = polarIndexPrepared ? "indexed" : "raw"
    ResidentLoopBenchmark.setEnvironment(configuration)
    var baseline: [String: [[UInt32]]] = [:]
    var sequence = 0
    emitJSON([
      "event": "ans_resident_loop_ready",
      "schema": "quantem-gpu-paired-runtime-tans-resident-loop/v1",
      "resident_count": residents.count, "shape": [512, 512, 192, 192],
      "logical_dtype": "uint16", "series_load_seconds": loadSeconds,
      "concurrent_loads": concurrentLoads,
      "series_resident_bytes": residents.reduce(0) { $0 + $1.residentBytes },
      "resident_bytes_by_source": residents.map(\.residentBytes),
      "metal_current_allocated_bytes": device.currentAllocatedSize,
      "recommended_working_set_bytes": device.recommendedMaxWorkingSetSize,
      "source_identity_sha256": residents.map(\.sourceIdentitySHA256),
      "acquisition_load_seconds": residents.map(\.loadMetrics.totalSeconds),
      "indexed_resident_index_bytes": residents.map(\.polarIndexBytes),
      "indexed_mode_available": polarIndexPrepared,
      "polar_query_variant": ProcessInfo.processInfo.environment[
        "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT"] ?? "packet-groups",
      "polar_query_scan512_ab_a1_b_a2": ProcessInfo.processInfo.environment[
        "QGPU_ANS_OPT_POLAR_QUERY_SCAN512"] == "1",
      "polar_query_scan512_pipeline_prepared": residents.map(
        \.polarScan512QueryPipelinePrepared),
      "simd_entropy_fast_path_pipeline_prepared": ProcessInfo.processInfo.environment[
        "QGPU_PAIRED_RUNTIME_PREPARE_SIMD_ENTROPY_FAST_PATH"] == "1",
      "compact_offsets_enabled": residents.map(\.compactOffsetsEnabled),
      "compact_offset_bytes": residents.map(\.compactOffsetBytes),
      "macro_lookahead_bits_prepared": residents.map(\.macroLookaheadBits),
      "macro_table_bytes_by_source": residents.map(\.macroTableBytes),
      "compact_offsets_requested": ProcessInfo.processInfo.environment[
        "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS"] == "1",
      "protocol": "JSON lines: set, run, quit; run responses include exact A1 hashes and full-map parity",
    ])
    func cycleParity(
      _ run: ResidentLoopRun, expected: [String: [[UInt32]]]
    ) -> (passed: Bool, mismatches: [[String: Any]]) {
      // Every cycle was compared to the retained first cycle while running.
      // Compare that cycle to A1 without retaining another volume per repeat.
      guard run.maps.keys.allSatisfy({ expected[$0] != nil }) else {
        return (false, [["reason": "requested mask is absent from the frozen A1 maps"]])
      }
      return ResidentLoopBenchmark.parity(
        actual: run.maps, expected: expected.filter { run.maps[$0.key] != nil })
    }
    while let line = readLine() {
      sequence += 1
      guard let data = line.data(using: .utf8),
        let command = try? JSONSerialization.jsonObject(with: data) as? [String: Any]
      else {
        emitJSON(["event": "ans_resident_loop_error", "sequence": sequence,
                  "error": "command must be a JSON object"])
        continue
      }
      let operation = ((command["command"] as? String) ?? (command["op"] as? String) ?? "run")
        .lowercased()
      if operation == "quit" || operation == "exit" {
        let allocatedBeforeRelease = device.currentAllocatedSize
        for resident in residents { resident.releaseResidentStorage() }
        emitJSON(["event": "ans_resident_loop_end", "sequence": sequence,
                  "resident_count": residents.count,
                  "metal_before_release_bytes": allocatedBeforeRelease,
                  "metal_after_release_bytes": device.currentAllocatedSize,
                  "all_released": residents.allSatisfy(\.isReleased)])
        return
      }
      do {
      if operation == "entropy_census" {
        let masks = experimentMasks()
        let adfStart = masks.first { $0.name == "adf-center-8" }!.values
        let adfEnd = masks.first { $0.name == "adf-center-20" }!.values
        let adfDelta = zip(adfStart, adfEnd).map { $0 == $1 ? UInt8(0) : UInt8(1) }
        let allRequested = [UInt8](repeating: 1, count: 192 * 192)
        for (name, requested) in [
          ("adf-center-8-to-20-delta", adfDelta),
          ("all-valid-detector-pixels", allRequested),
        ] {
          let requestedPixels = requested.reduce(0) { $0 + Int($1) }
          let requestedMaskSHA256 = SHA256.hash(data: Data(requested))
            .map { String(format: "%02x", $0) }.joined()
          for (source, resident) in residents.enumerated() {
            let effective = effectiveMask(requested, resident: resident)
            let census = try resident.detectorEntropyModeCount(mask: effective.values)
            let selectedPixels = effective.values.reduce(0) { $0 + Int($1) }
            let packets = 512
            let effectiveMaskSHA256 = SHA256.hash(data: Data(effective.values))
              .map { String(format: "%02x", $0) }.joined()
            emitJSON([
              "event": "ans_opt_entropy_mode_counts", "sequence": sequence,
              "mask": name, "source": source,
              "requested_pixels": requestedPixels,
              "selected_pixels": selectedPixels,
              "excluded_pixels": effective.excluded,
              "requested_mask_sha256": requestedMaskSHA256,
              "effective_mask_sha256": effectiveMaskSHA256,
              "total_stream_count": selectedPixels * packets,
              "entropy_stream_count": census.count,
              "diagnostic_metal_allocated_bytes": census.metalAllocatedBytes,
              "validity_policy": "requested_mask_and_source_detectorValidityMask",
            ])
          }
        }
        continue
      }
      if operation == "fourway_checkpoint" {
        guard residents.count == 7, indexed.count == residents.count else {
          throw failure("Four-way checkpoint parity requires seven aligned indexed sources")
        }
        let sourceIndex = (command["source"] as? Int) ?? 0
        guard residents.indices.contains(sourceIndex) else {
          throw failure("Four-way checkpoint source must be in 0..<7")
        }
        let shaderPath = ProcessInfo.processInfo.environment[
          "QGPU_FOURWAY_CHECKPOINT_SHADER"] ?? ""
        guard !shaderPath.isEmpty else {
          throw failure("Set QGPU_FOURWAY_CHECKPOINT_SHADER to the registered prototype MSL file")
        }
        let masks = experimentMasks()
        guard let previous = masks.first(where: { $0.name == "adf-center-8" })?.values,
          let target = masks.first(where: { $0.name == "adf-center-20" })?.values
        else { throw failure("Four-way checkpoint parity requires the registered ADF masks") }
        let requestedDelta = zip(previous, target).map { $0 == $1 ? UInt8(0) : UInt8(1) }
        let effectiveDelta = effectiveMask(requestedDelta, resident: residents[sourceIndex])
        let residualPixels = effectiveDelta.values.indices.compactMap {
          effectiveDelta.values[$0] == 0 ? nil : UInt32($0)
        }
        guard !residualPixels.isEmpty else {
          throw failure("Four-way checkpoint ADF residual mask is empty")
        }
        let candidateLimit = min(
          max(1, (command["candidate_limit"] as? Int) ?? 8), residualPixels.count)
        let loader = try Metal4DSTEMIndexedLoader(device: device)
        var rejectedCandidates: [[String: Any]] = []
        var decodedCandidate = false
        for pixel in residualPixels.prefix(candidateLimit) {
          let callStarted = CFAbsoluteTimeGetCurrent()
          let diagnostic = try residents[sourceIndex].diagnoseFourWayCheckpoint(
            selectedStreamIndices: [pixel], residualDetectorPixels: residualPixels,
            shaderSourceURL: URL(fileURLWithPath: shaderPath))
          let diagnosticWallMilliseconds = (CFAbsoluteTimeGetCurrent() - callStarted) * 1_000
          guard diagnostic.sourceIdentitySHA256
              == residents[sourceIndex].sourceIdentitySHA256
          else { throw failure("Four-way checkpoint source identity changed during the call") }
          if diagnostic.outcome == "rejected-non-entropy-modes" {
            rejectedCandidates.append([
              "pixel": pixel,
              "outcome": diagnostic.outcome,
              "mode_histogram": diagnostic.modeHistogram,
              "selected_fallback_count": diagnostic.selectedFallbackCount,
              "selected_unsupported_count": diagnostic.selectedUnsupportedCount,
            ])
            continue
          }
          guard diagnostic.outcome == "decoded-entropy-subset-parity-unchecked",
            let decoded = diagnostic.decodedValues, decoded.count == 512,
            let checkpointCaptureWallMilliseconds = diagnostic.captureWallMilliseconds,
            let fourWayDecodeWallMilliseconds = diagnostic.segmentWallMilliseconds,
            diagnostic.selectedEntropyCount == 1,
            diagnostic.selectedFallbackCount == 0,
            diagnostic.selectedUnsupportedCount == 0,
            diagnostic.captureStatusCounts == [0, 1, 0, 0, 0, 0, 0, 0],
            diagnostic.segmentStatusCounts == [0, 4, 0, 0, 0, 0, 0]
          else {
            throw failure(
              "Four-way runtime decoder failed validation for entropy candidate \(pixel): "
                + diagnostic.outcome
                + "; capture=\(diagnostic.captureStatusCounts ?? [])"
                + "; segments=\(diagnostic.segmentStatusCounts ?? [])")
          }

          var reference = [UInt16]()
          reference.reserveCapacity(512)
          for scanColumn in 0..<512 {
            let frame = try loader.diffractionPattern(
              source: indexed[sourceIndex], scanRow: 0, scanColumn: scanColumn)
            guard frame.sourceIdentitySHA256 == diagnostic.sourceIdentitySHA256,
              frame.values.count == 192 * 192
            else { throw failure("Original-HDF5 frame identity or detector shape mismatch") }
            reference.append(frame.values[Int(pixel)])
          }
          let exact = decoded == reference
          let decodedBytes = decoded.withUnsafeBytes { Data($0) }
          let referenceBytes = reference.withUnsafeBytes { Data($0) }
          let decodedHash = SHA256.hash(data: decodedBytes)
            .map { String(format: "%02x", $0) }.joined()
          let referenceHash = SHA256.hash(data: referenceBytes)
            .map { String(format: "%02x", $0) }.joined()
          let mismatchIndices = decoded.indices.compactMap {
            decoded[$0] == reference[$0] ? nil : $0
          }
          emitJSON([
            "event": "fourway_checkpoint_parity",
            "sequence": sequence,
            "source": sourceIndex,
            "source_identity_sha256": diagnostic.sourceIdentitySHA256,
            "shape": [512, 512, 192, 192],
            "dtype": "uint16",
            "transition": "adf-center-8-to-adf-center-20",
            "selected_stream_index": pixel,
            "scan_packet": 0,
            "selected_stream_in_residual": residualPixels.contains(pixel),
            "residual_detector_pixel_count": diagnostic.residualDetectorPixelCount,
            "residual_stream_count": diagnostic.residualStreamCount,
            "coverage_fraction": diagnostic.coverageFraction,
            "mode_histogram": diagnostic.modeHistogram,
            "exact_512_count_parity": exact,
            "mismatch_count": mismatchIndices.count,
            "mismatch_scan_indices": mismatchIndices,
            "decoded_sha256": decodedHash,
            "original_hdf5_sha256": referenceHash,
            "decoded_values": decoded,
            "original_hdf5_values": reference,
            "timing_ms": [
              "whole_diagnostic_with_compile": diagnosticWallMilliseconds,
              "runtime_compile": diagnostic.runtimeCompileMilliseconds,
              "mode_inspection_wall": diagnostic.modeInspectionWallMilliseconds,
              "checkpoint_capture_wall": checkpointCaptureWallMilliseconds,
              "fourway_segment_decode_wall": fourWayDecodeWallMilliseconds,
              "original_hdf5_reference_excluded": true,
            ],
            "memory_bytes": [
              "resident": diagnostic.residentBytes,
              "explicit_diagnostic_buffers": diagnostic.diagnosticAllocationBytes,
              "allocated_before": diagnostic.allocatedSizeBefore,
              "allocated_after_segments": diagnostic.allocatedSizeAfter,
              "compact_offsets": diagnostic.compactOffsetBytes,
            ],
            "candidate_rejections_before_success": rejectedCandidates,
          ])
          decodedCandidate = true
          break
        }
        if !decodedCandidate {
          emitJSON([
            "event": "fourway_checkpoint_no_entropy_candidate",
            "sequence": sequence,
            "source": sourceIndex,
            "candidate_limit": candidateLimit,
            "residual_detector_pixel_count": residualPixels.count,
            "candidate_rejections": rejectedCandidates,
            "exact_parity": false,
          ])
        }
        continue
      }
        if operation == "set" || operation == "config" {
          try configuration.apply(command)
          guard !configuration.macro || residents.allSatisfy({ $0.macroLookaheadBits == 2 }) else {
            throw failure(
              "The 2-bit macro pipeline must be prepared before loading all residents; "
                + "launch with QGPU_ANS_OPT_MACRO2=1")
          }
          ResidentLoopBenchmark.setEnvironment(configuration)
          emitJSON(["event": "ans_resident_loop_config", "sequence": sequence,
                    "configuration": configuration.json])
          continue
        }
        if operation == "stage_isolation" {
          let jointPlan = (command["joint_plan"] as? Bool) ?? false
          let branchlessPop = (command["branchless_pop"] as? Bool) ?? false
          let reuseScratch = (command["reuse_scratch"] as? Bool) ?? false
          let refillThreshold = (command["refill_threshold"] as? Int) ?? 32
          let phasedReaders = (command["phased_readers"] as? Bool) ?? false
          let pairUnroll = (command["pair_unroll"] as? Int) ?? 1
          let masks = experimentMasks()
          let previous = masks.first { $0.name == "adf-center-8" }!.values
          let target = masks.first { $0.name == "adf-center-20" }!.values
          var control = configuration.control()
          control.mode = "indexed"
          control.batch = false
          ResidentLoopBenchmark.setEnvironment(control)
          let before = try await experimentUpdateAll(residents: residents, mask: previous,
            batched: false, boundedConcurrency: 7)
          let after = try await experimentUpdateAll(residents: residents, mask: target,
            batched: false, boundedConcurrency: 7)
          let expected = zip(before, after).map { pair in
            zip(pair.0.values, pair.1.values).map { $1 &- $0 }
          }
          setenv("QGPU_PAIRED_RUNTIME_JOINT_PLAN", jointPlan ? "1" : "0", 1)
          var stageMaps: [String: [[UInt32]]] = [:]
          var samples: [[String: Any]] = []
          let stages = sequence % 2 == 0 ? ["combined", "residual", "index"] : ["index", "residual", "combined"]
          for stage in stages {
            let started = CFAbsoluteTimeGetCurrent()
            let results = try await withThrowingTaskGroup(of: (Int, [UInt32], Double, Int, Int, Int, Int, Bool).self) { group in
              for (source, resident) in residents.enumerated() {
                group.addTask {
                  let result = try resident.isolateDetectorStage(previous: previous, target: target,
                    stage: stage, branchlessPop: branchlessPop, reuseScratch: reuseScratch,
                    refillThreshold: refillThreshold, phasedReaders: phasedReaders, pairUnroll: pairUnroll)
                  return (source, result.values, result.gpuMilliseconds, result.fields, result.residuals,
                    result.allocatedScratchBytes, result.preparedScratchBytes, result.reusedScratch)
                }
              }
              var values: [(Int, [UInt32], Double, Int, Int, Int, Int, Bool)] = []
              for try await value in group { values.append(value) }
              return values.sorted { $0.0 < $1.0 }
            }
            let wall = (CFAbsoluteTimeGetCurrent() - started) * 1000
            stageMaps[stage] = results.map { $0.1 }
            for result in results {
              samples.append(["stage": stage, "source": result.0, "gpu_ms": result.2,
                "all_seven_wall_ms": wall, "fields": result.3, "residuals": result.4,
                "allocated_scratch_bytes": result.5, "prepared_scratch_bytes": result.6,
                "reused_scratch": result.7])
            }
          }
          for source in residents.indices {
            let sum = zip(stageMaps["index"]![source], stageMaps["residual"]![source]).map { $0 &+ $1 }
            guard sum == expected[source], stageMaps["combined"]![source] == expected[source] else {
              throw failure("Stage contribution parity failed for source \(source)")
            }
          }
          emitJSON(["event": "stage_isolation", "sequence": sequence, "exact": true,
            "branchless_pop": branchlessPop,
            "reuse_scratch": reuseScratch, "refill_threshold": refillThreshold,
            "phased_readers": phasedReaders, "metal_current_allocated_bytes": device.currentAllocatedSize,
            "pair_unroll": pairUnroll,
            "joint_plan": jointPlan,
            "samples": samples, "resident_bytes": residents.reduce(0) { $0 + $1.residentBytes }])
          continue
        }
        if operation == "entropy_chunk_census" {
          guard residents.count == 7, polarIndexPrepared,
            configuration.mode == "indexed", configuration.kernel == "packet-owner2",
            configuration.streamsPerLane == "2", configuration.packetSplits == "1",
            !configuration.jointPlan
          else {
            throw failure(
              "Entropy-chunk census requires seven indexed residents and the baseline "
                + "packet-owner2 schedule (2 streams/lane, 1 packet split, joint plan off)")
          }
          let masks = experimentMasks()
          guard let previous = masks.first(where: { $0.name == "adf-center-8" })?.values,
            let target = masks.first(where: { $0.name == "adf-center-20" })?.values
          else { throw failure("Entropy-chunk census ADF masks are missing") }
          let identitiesBefore = residents.map(\.sourceIdentitySHA256)
          guard Set(identitiesBefore).count == residents.count else {
            throw failure("Entropy-chunk census requires seven distinct source identities")
          }
          let residentBytesBefore = residents.map(\.residentBytes)
          var samples: [[String: Any]] = []
          var allEntropyFullChunksTotal: UInt64 = 0
          var mixedFullChunksTotal: UInt64 = 0
          var allEntropyTailChunksTotal: UInt64 = 0
          var mixedTailChunksTotal: UInt64 = 0
          var allEntropySIMDFullChunksTotal: UInt64 = 0
          var mixedSIMDFullChunksTotal: UInt64 = 0
          var allEntropySIMDTailChunksTotal: UInt64 = 0
          var mixedSIMDTailChunksTotal: UInt64 = 0
          var metalAllocatedBytesDuringCensus: [Int] = []
          for (source, resident) in residents.enumerated() {
            let effectivePrevious = effectiveMask(previous, resident: resident)
            let effectiveTarget = effectiveMask(target, resident: resident)
            let census = try resident.detectorEntropyChunkCensus(
              previous: effectivePrevious.values, target: effectiveTarget.values)
            guard census.residualStreams == 1_067, census.packets == 512,
              census.fullChunksPerPacket == 16, census.tailStreamsPerPacket == 43,
              census.sourceIdentitySHA256 == identitiesBefore[source],
              census.residentBytesBefore == residentBytesBefore[source],
              census.residentBytesAfter == residentBytesBefore[source]
            else {
              throw failure("Entropy-chunk census invariants failed for source \(source)")
            }
            allEntropyFullChunksTotal += UInt64(census.allEntropyFullChunks)
            mixedFullChunksTotal += UInt64(census.mixedFullChunks)
            allEntropyTailChunksTotal += UInt64(census.allEntropyTailChunks)
            mixedTailChunksTotal += UInt64(census.mixedTailChunks)
            allEntropySIMDFullChunksTotal += UInt64(census.allEntropySIMDFullChunks)
            mixedSIMDFullChunksTotal += UInt64(census.mixedSIMDFullChunks)
            allEntropySIMDTailChunksTotal += UInt64(census.allEntropySIMDTailChunks)
            mixedSIMDTailChunksTotal += UInt64(census.mixedSIMDTailChunks)
            metalAllocatedBytesDuringCensus.append(census.metalAllocatedBytesDuringCensus)
            samples.append([
              "source": source,
              "source_identity_sha256": census.sourceIdentitySHA256,
              "rows": census.residualStreams,
              "residual_streams": census.residualStreams,
              "packets": census.packets,
              "chunk_width": 64,
              "full_chunks_per_packet": census.fullChunksPerPacket,
              "eligible_full_chunk_denominator": census.fullChunksPerPacket * census.packets,
              "all_entropy_full_chunks": census.allEntropyFullChunks,
              "mixed_full_chunks": census.mixedFullChunks,
              "all_entropy_full_chunk_fraction":
                Double(census.allEntropyFullChunks)
                  / Double(census.fullChunksPerPacket * census.packets),
              "tail_streams_per_packet": census.tailStreamsPerPacket,
              "tail_partial_chunk_count": census.tailStreamsPerPacket == 0 ? 0 : census.packets,
              "all_entropy_tail_chunks_not_eligible": census.allEntropyTailChunks,
              "mixed_tail_chunks_not_eligible": census.mixedTailChunks,
              "simd_width": 32,
              "full_simd_chunks_per_packet": census.simdFullChunksPerPacket,
              "eligible_full_simd_chunk_denominator":
                census.simdFullChunksPerPacket * census.packets,
              "all_entropy_full_simd_chunks": census.allEntropySIMDFullChunks,
              "mixed_full_simd_chunks": census.mixedSIMDFullChunks,
              "all_entropy_full_simd_chunk_fraction":
                Double(census.allEntropySIMDFullChunks)
                  / Double(census.simdFullChunksPerPacket * census.packets),
              "simd_tail_streams_per_packet": census.simdTailStreamsPerPacket,
              "simd_tail_partial_chunk_count":
                census.simdTailStreamsPerPacket == 0 ? 0 : census.packets,
              "all_entropy_simd_tail_chunks_not_eligible":
                census.allEntropySIMDTailChunks,
              "mixed_simd_tail_chunks_not_eligible": census.mixedSIMDTailChunks,
              "resident_bytes_before": census.residentBytesBefore,
              "resident_bytes_after": census.residentBytesAfter,
              "metal_allocated_bytes_during_census": census.metalAllocatedBytesDuringCensus,
            ])
          }
          emitJSON([
            "event": "ans_resident_loop_entropy_chunk_census",
            "sequence": sequence,
            "exact": true,
            "decoder_schedule": [
              "kernel": configuration.kernel,
              "streams_per_lane": Int(configuration.streamsPerLane)!,
              "packet_splits": Int(configuration.packetSplits)!,
              "joint_plan": configuration.jointPlan,
            ],
            "mask_delta": "adf-center-8-to-adf-center-20",
            "validity_policy": "same-validPixels-masked-delta-as-isolateDetectorStage",
            "rows": 1_067,
            "residual_streams": 1_067,
            "packets": 512,
            "chunk_width": 64,
            "full_chunks_per_packet": 16,
            "eligible_full_chunk_denominator_per_source": 8_192,
            "eligible_full_chunk_denominator_all_sources": residents.count * 8_192,
            "all_entropy_full_chunks_all_sources": allEntropyFullChunksTotal,
            "mixed_full_chunks_all_sources": mixedFullChunksTotal,
            "trailing_streams_per_packet_not_eligible": 43,
            "tail_partial_chunk_count_per_source": 512,
            "all_entropy_tail_chunks_all_sources_not_eligible": allEntropyTailChunksTotal,
            "mixed_tail_chunks_all_sources_not_eligible": mixedTailChunksTotal,
            "simd_width": 32,
            "full_simd_chunks_per_packet": 33,
            "eligible_full_simd_chunk_denominator_per_source": 16_896,
            "eligible_full_simd_chunk_denominator_all_sources": residents.count * 16_896,
            "all_entropy_full_simd_chunks_all_sources": allEntropySIMDFullChunksTotal,
            "mixed_full_simd_chunks_all_sources": mixedSIMDFullChunksTotal,
            "simd_tail_streams_per_packet_not_eligible": 11,
            "simd_tail_partial_chunk_count_per_source": 512,
            "all_entropy_simd_tail_chunks_all_sources_not_eligible":
              allEntropySIMDTailChunksTotal,
            "mixed_simd_tail_chunks_all_sources_not_eligible": mixedSIMDTailChunksTotal,
            "source_count": residents.count,
            "source_identities_unique_and_unchanged":
              identitiesBefore == residents.map(\.sourceIdentitySHA256),
            "resident_bytes_unchanged":
              residentBytesBefore == residents.map(\.residentBytes),
            "resident_bytes_before_by_source": residentBytesBefore,
            "resident_bytes_after_by_source": residents.map(\.residentBytes),
            "metal_allocated_bytes_during_census_max_sampled":
              metalAllocatedBytesDuringCensus.max() ?? device.currentAllocatedSize,
            "diagnostic_transient_requested_bytes_per_source":
              1_067 * MemoryLayout<UInt32>.stride + 9 * MemoryLayout<UInt32>.stride,
            "samples": samples,
          ])
          continue
        }
        if operation == "dp_audit" {
          let scans = [0, 1, 30, 31, 32, 33, 510, 511, 512, 513,
                       16_383, 16_384, 16_385, 262_142, 262_143]
            + (0..<20).map { ($0 * 7919 + 123) % 262_144 }
          var samples: [[String: Any]] = []
          for scan in scans {
            for (source, resident) in residents.enumerated() {
              let started = CFAbsoluteTimeGetCurrent()
              let values = try resident.extractRawDiffraction(
                scanRow: scan / 512, scanColumn: scan % 512)
              let wall = (CFAbsoluteTimeGetCurrent() - started) * 1000
              samples.append(["source": source, "scan": scan,
                              "sha256_u32_le": hash(values), "wall_ms": wall])
            }
          }
          emitJSON(["event": "dp_audit", "sequence": sequence, "samples": samples,
                    "metal_current_allocated_bytes": device.currentAllocatedSize])
          continue
        }
        if operation == "decode_checksum" {
          try configuration.apply(command)
          guard configuration.mode != "indexed" || polarIndexPrepared else {
            throw failure(
              "indexed mode was requested but no polar index was prepared; relaunch with "
                + "QGPU_PAIRED_RUNTIME_POLAR_INDEX=1")
          }
          let names: [ExperimentMask]
          if ResidentLoopBenchmark.value(command, "masks") == nil
            && ResidentLoopBenchmark.value(command, "mask_names") == nil
          {
            names = modeDiagnosticMasks(experimentMasks()).map {
              ExperimentMask(name: $0.0, values: $0.1)
            }
          } else {
            names = try ResidentLoopBenchmark.masks(command)
          }
          guard names.count <= 20 else {
            throw failure("resident loop supports at most 20 masks per command")
          }
          var samples: [[String: Any]] = []
          var allPassed = true
          var allChecksumParity = true
          var allImageParity = true
          for mask in names {
            var control = configuration.control()
            control.mode = "raw"
            ResidentLoopBenchmark.setEnvironment(control)
            let zero = [UInt8](repeating: 0, count: 192 * 192)
            _ = try await experimentUpdateAll(
              residents: residents, mask: zero, batched: control.batch,
              boundedConcurrency: control.boundedConcurrency)
            let referenceStarted = CFAbsoluteTimeGetCurrent()
            let references = try await experimentUpdateAll(
              residents: residents, mask: mask.values, batched: control.batch,
              boundedConcurrency: control.boundedConcurrency)
            let referenceAllWall = (CFAbsoluteTimeGetCurrent() - referenceStarted) * 1_000
            let expected = references.map { result in
              stride(from: 0, to: result.values.count, by: 512).map { first in
                result.values[first..<min(first + 512, result.values.count)].reduce(UInt32(0)) {
                  $0 &+ $1
                }
              }
            }
            ResidentLoopBenchmark.setEnvironment(configuration)
            let diagnosticStarted = CFAbsoluteTimeGetCurrent()
            let diagnostics = try await withThrowingTaskGroup(
              of: (Int, [UInt32], Double, Double).self
            ) { group in
              for (source, resident) in residents.enumerated() {
                group.addTask {
                  let effective = effectiveMask(mask.values, resident: resident)
                  let result = try resident.detectorDecodeChecksum(mask: effective.values)
                  return (source, result.values, result.wallMilliseconds, result.gpuMilliseconds)
                }
              }
              var ordered = [(Int, [UInt32], Double, Double)]()
              for try await result in group { ordered.append(result) }
              return ordered.sorted { $0.0 < $1.0 }
            }
            let allWall = (CFAbsoluteTimeGetCurrent() - diagnosticStarted) * 1_000

            ResidentLoopBenchmark.setEnvironment(control)
            _ = try await experimentUpdateAll(
              residents: residents, mask: zero, batched: control.batch,
              boundedConcurrency: control.boundedConcurrency)
            let repeatedStarted = CFAbsoluteTimeGetCurrent()
            let repeated = try await experimentUpdateAll(
              residents: residents, mask: mask.values, batched: control.batch,
              boundedConcurrency: control.boundedConcurrency)
            let repeatedAllWall = (CFAbsoluteTimeGetCurrent() - repeatedStarted) * 1_000
            for diagnostic in diagnostics {
              let source = diagnostic.0
              let imageParity = repeated[source].values == references[source].values
              let checksumParity = diagnostic.1 == expected[source]
              let passed = checksumParity && imageParity
              allPassed = allPassed && passed
              allChecksumParity = allChecksumParity && checksumParity
              allImageParity = allImageParity && imageParity
              samples.append([
                "mask": mask.name, "source": source,
                "wall_ms": diagnostic.2, "gpu_ms": diagnostic.3,
                "all_seven_wall_ms": allWall,
                "packet_count": diagnostic.1.count,
                "checksum_parity": checksumParity,
                "diagnostic_not_image": true,
                "sha256_u32_le": hash(diagnostic.1),
                "reference_wall_ms": references[source].wallMilliseconds,
                "reference_gpu_ms": references[source].gpuMilliseconds,
                "reference_all_seven_wall_ms": referenceAllWall,
                "reference_sha256_u32_le": hash(references[source].values),
                "repeated_reference_wall_ms": repeated[source].wallMilliseconds,
                "repeated_reference_gpu_ms": repeated[source].gpuMilliseconds,
                "repeated_reference_all_seven_wall_ms": repeatedAllWall,
                "repeated_reference_sha256_u32_le": hash(repeated[source].values),
                "reference_image_parity": imageParity,
              ])
            }
            ResidentLoopBenchmark.setEnvironment(configuration)
          }
          emitJSON([
            "event": "ans_resident_loop_decode_checksum", "sequence": sequence,
            "configuration": configuration.json, "masks": names.map(\.name),
            "checksum_parity": allChecksumParity, "all_parity": allPassed,
            "diagnostic_not_image": true,
            "reference_image_parity": allImageParity,
            "workload": "absolute all-positive selected-region mask",
            "samples": samples,
          ])
          continue
        }
        guard operation == "run" else {
          throw failure(
            "resident loop command must be set, run, stage_isolation, entropy_chunk_census, "
              + "decode_checksum, or quit")
        }
        try configuration.apply(command)
        let requested = configuration
        guard !requested.macro || residents.allSatisfy({ $0.macroLookaheadBits == 2 }) else {
          throw failure(
            "The 2-bit macro pipeline must be prepared before loading all residents; "
              + "launch with QGPU_ANS_OPT_MACRO2=1")
        }
        guard requested.mode != "indexed" || polarIndexPrepared else {
          throw failure(
            "indexed mode was requested but no polar index was prepared; relaunch with "
              + "QGPU_PAIRED_RUNTIME_POLAR_INDEX=1")
        }
        let names = try ResidentLoopBenchmark.masks(command)
        let parityMapBytes = names.count * residents.count * 512 * 512
          * MemoryLayout<UInt32>.stride
        guard names.count <= 21, parityMapBytes <= 150 * 1024 * 1024 else {
          throw failure(
            "resident loop supports up to 21 masks within its 150 MiB parity-map bound")
        }
        let cycles = ResidentLoopBenchmark.number(command, "cycles") ?? 1
        guard cycles > 0 && cycles <= 100 else {
          throw failure("resident loop cycles must be between 1 and 100")
        }
        let requestedArm = ((command["arm"] as? String) ?? "candidate").lowercased()
        let arm = (requestedArm == "a1" || requestedArm == "control")
          ? "A1" : requestedArm == "a2" ? "A2" : requestedArm
        guard ["A1", "candidate", "A2"].contains(arm) else {
          throw failure("resident loop arm must be A1, A2, or candidate")
        }
        var baselineCreated = false
        if baseline.isEmpty && arm != "A1" {
          let control = requested.control()
          ResidentLoopBenchmark.setEnvironment(control)
          let controlRun = try await ResidentLoopBenchmark.run(
            residents: residents, masks: names, cycles: cycles, configuration: control)
          baseline = controlRun.maps
          baselineCreated = true
          let controlParity = cycleParity(controlRun, expected: baseline)
          guard controlParity.passed else {
            throw failure("A1 control changed between resident-loop cycles")
          }
        }
        var effective = arm == "A1" ? requested.control() : requested
        if arm == "A1"
          && ProcessInfo.processInfo.environment[
            "QGPU_PAIRED_RUNTIME_PRESERVE_TRUSTED_TABLE_CONTROL"] == "1"
        {
          effective.trustedTable = requested.trustedTable
        }
        ResidentLoopBenchmark.setEnvironment(effective)
        let run = try await ResidentLoopBenchmark.run(
          residents: residents, masks: names, cycles: cycles, configuration: effective)
        if baseline.isEmpty {
          baseline = run.maps
          baselineCreated = true
        }
        let check = cycleParity(run, expected: baseline)
        configuration = requested
        ResidentLoopBenchmark.setEnvironment(configuration)
        emitJSON([
          "event": "ans_resident_loop_result", "sequence": sequence,
          "arm": arm, "configuration": effective.json, "requested_configuration": requested.json,
          "cycles": cycles, "masks": names.map(\.name), "baseline_created": baselineCreated,
          "a1_sha256_u32_le": ResidentLoopBenchmark.hashes(baseline),
          "sha256_u32_le": ResidentLoopBenchmark.hashes(run.maps),
          "exact_a1_hashes": check.passed, "fullmap_parity": check.passed,
          "series_resident_bytes": residents.reduce(0) { $0 + $1.residentBytes },
          "metal_current_allocated_bytes": device.currentAllocatedSize,
          "fullmap_mismatches": check.mismatches, "samples": run.samples,
        ])
      } catch {
        emitJSON(["event": "ans_resident_loop_error", "sequence": sequence,
                  "error": (error as NSError).localizedDescription])
      }
    }
    emitJSON(["event": "ans_resident_loop_end", "reason": "stdin_eof_without_quit",
              "resident_count": residents.count])
  }

  static func detectorTrials(
    residents: [MetalPairedRuntimeTANSResidentSource], masks: [[UInt8]]
  ) async throws -> [Double] {
    try await updateAll(residents: residents, mask: masks[0])
    var milliseconds: [Double] = []
    for trial in 0..<24 {
      let started = CFAbsoluteTimeGetCurrent()
      try await updateAll(residents: residents, mask: masks[(trial + 1) % masks.count])
      milliseconds.append((CFAbsoluteTimeGetCurrent() - started) * 1_000)
    }
    return milliseconds
  }

  static func updateAll(
    residents: [MetalPairedRuntimeTANSResidentSource], mask: [UInt8]
  ) async throws {
    try await withThrowingTaskGroup(of: Void.self) { group in
      for resident in residents {
        group.addTask { _ = try resident.updateVirtualDetector(mask: mask) }
      }
      try await group.waitForAll()
    }
  }

  private struct ExperimentMask {
    let name: String
    let values: [UInt8]
  }

  private struct ExperimentResult: Sendable {
    let source: Int
    let values: [UInt32]
    let wallMilliseconds: Double
    let gpuMilliseconds: Double
    let changed: Int
    let excluded: Int
    let polarFields: Int
    let polarResiduals: Int
  }

  private static func optimizationExperiment(
    residents: [MetalPairedRuntimeTANSResidentSource],
    indexed: [Native4DSTEMIndexedSource], device: MTLDevice,
    loadSeconds: Double, concurrentLoads: Int
  ) async throws {
    let masks = experimentMasks()
    let cycles = Int(ProcessInfo.processInfo.environment["QGPU_ANS_OPT_CYCLES"] ?? "3") ?? 0
    guard cycles > 0 else {
      throw failure("QGPU_ANS_OPT_CYCLES must be a positive integer")
    }
    // Full seven-source UInt32 maps are retained only for the A1 reference;
    // later arms are checked against that map set and discarded.
    let retainedBytes = masks.count * residents.count * 512 * 512 * 4
    guard retainedBytes <= 150 * 1024 * 1024 else {
      throw failure("The optimization experiment exceeded its 150 MiB parity-map bound")
    }
    emitJSON([
      "event": "ans_opt_begin", "masks": masks.count,
      "retained_map_bytes": retainedBytes, "sources": residents.count,
      "cycles_per_arm": cycles, "series_load_seconds": loadSeconds,
      "concurrent_loads": concurrentLoads,
      "acquisition_load_seconds": residents.map(\.loadMetrics.totalSeconds),
      "series_resident_bytes": residents.reduce(0) { $0 + $1.residentBytes },
      "metal_current_allocated_bytes": device.currentAllocatedSize,
      "polar_index_bytes": residents.map(\.polarIndexBytes),
      "polar_index_build_milliseconds": residents.map(\.polarIndexBuildMilliseconds),
      "polar_query_variant": ProcessInfo.processInfo.environment[
        "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT"] ?? "packet-groups",
      "polar_query_scan512_ab_a1_b_a2": ProcessInfo.processInfo.environment[
        "QGPU_ANS_OPT_POLAR_QUERY_SCAN512"] == "1",
      "polar_query_scan512_pipeline_prepared": residents.map(
        \.polarScan512QueryPipelinePrepared),
    ])

    // These profiling dispatches are deliberately outside every timed arm.
    for (name, profileMask) in modeDiagnosticMasks(masks) {
      for (source, resident) in residents.enumerated() {
        let effective = effectiveMask(profileMask, resident: resident)
        let counts = try resident.detectorStreamModeCounts(mask: effective.values)
        emitJSON([
          "event": "ans_opt_mode_counts", "mask": name, "source": source,
          "selected_pixels": effective.values.reduce(0) { $0 + Int($1) },
          "excluded_pixels": effective.excluded,
          "validity_policy": "requested_mask_and_source_detectorValidityMask",
          "counts_by_mode": counts,
        ])
      }
    }

    var baseline: [String: [[UInt32]]] = [:]
    let sparseOnly = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_SPARSE_ONLY"] == "1"
    let polarOnly = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_POLAR_ONLY"] == "1"
    let reader32Only = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_READER32_ONLY"] == "1"
    let cooperativeOnly = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_COOPERATIVE_ONLY"] == "1"
    let polarQueryScan512Only = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_POLAR_QUERY_SCAN512"] == "1"
    let arms = polarQueryScan512Only
      ? [
        ("A1", false, false, false, false, false, true, "packet-groups"),
        ("B-scan512", false, false, false, false, false, true, "scan512"),
        ("A2", false, false, false, false, false, true, "packet-groups"),
      ]
      : polarOnly
      ? [
        ("A1", false, false, false, false, false, false, "packet-groups"),
        ("G", false, false, false, false, false, true, "packet-groups"),
        ("A2", false, false, false, false, false, false, "packet-groups"),
      ]
      : sparseOnly
        ? [
          ("A1", false, false, false, false, false, false, "packet-groups"),
          ("B", true, false, false, false, false, false, "packet-groups"),
          ("A2", false, false, false, false, false, false, "packet-groups"),
        ]
        : reader32Only
          ? [
            ("A1", false, false, false, false, false, false, "packet-groups"),
            ("F", false, false, false, false, true, false, "packet-groups"),
            ("A2", false, false, false, false, false, false, "packet-groups"),
          ]
          : cooperativeOnly
            ? [
              ("A1", false, false, false, false, false, false, "packet-groups"),
              ("E", true, false, false, true, false, false, "packet-groups"),
              ("A2", false, false, false, false, false, false, "packet-groups"),
            ]
            : [
              ("A1", false, false, false, false, false, false, "packet-groups"),
              ("B", true, false, false, false, false, false, "packet-groups"),
              ("C", true, true, false, false, false, false, "packet-groups"),
              ("D", false, false, true, false, false, false, "packet-groups"),
              ("A2", false, false, false, false, false, false, "packet-groups"),
            ]
    for (arm, sparse, dense, macro, cooperative, reader32, polar, polarQueryVariant) in arms {
      setenv("QGPU_PAIRED_RUNTIME_SPARSE_SPLIT", sparse ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_DENSE_COMPACTION", dense ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_MACRO", macro ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_COOPERATIVE", cooperative ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_READER32", reader32 ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_POLAR_INDEX", polar ? "1" : "0", 1)
      setenv("QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT", polarQueryVariant, 1)
      let adaptive = polar && ProcessInfo.processInfo.environment[
        "QGPU_ANS_OPT_POLAR_PARTIALS"] == "1"
      setenv("QGPU_PAIRED_RUNTIME_DETECTOR_KERNEL", adaptive ? "adaptive-partials" : "packet-owner2", 1)
      for cycle in 0..<cycles {
        _ = try await experimentUpdateAll(
          residents: residents, mask: [UInt8](repeating: 0, count: 192 * 192))
        for mask in masks {
          let started = CFAbsoluteTimeGetCurrent()
          let results = try await experimentUpdateAll(
            residents: residents, mask: mask.values)
          let totalWall = (CFAbsoluteTimeGetCurrent() - started) * 1_000
          if arm == "A1" && cycle == 0 {
            baseline[mask.name] = results.map(\.values)
          }
          guard let expected = baseline[mask.name] else {
            throw failure("Missing frozen A1 map for \(mask.name)")
          }
          for result in results {
            guard result.values == expected[result.source] else {
              throw failure(
                "\(arm) cycle \(cycle) source \(result.source) changed \(mask.name)")
            }
            emitJSON([
              "event": "ans_opt_sample", "arm": arm, "cycle": cycle,
              "mask": mask.name, "source": result.source,
              "wall_ms": result.wallMilliseconds, "gpu_ms": result.gpuMilliseconds,
              "all_seven_wall_ms": totalWall, "changed_pixels": result.changed,
              "excluded_pixels": result.excluded,
              "polar_field_count": result.polarFields,
              "polar_residual_count": result.polarResiduals,
              "polar_query_variant": polarQueryVariant,
              "polar_query_scan512_pipeline_prepared": residents.map(
                \.polarScan512QueryPipelinePrepared),
              "validity_policy": "requested_mask_and_source_detectorValidityMask",
              "sha256_u32_le": hash(result.values), "exact_vs_a1": true,
            ])
          }
        }
      }
    }

    let skipIndependent = ProcessInfo.processInfo.environment[
      "QGPU_ANS_OPT_SKIP_INDEPENDENT_PARITY"] == "1"
    if skipIndependent {
      emitJSON([
        "event": "ans_opt_independent_parity", "status": "pending",
        "reason": "QGPU_ANS_OPT_SKIP_INDEPENDENT_PARITY=1",
      ])
    } else {
      for sourceIndex in indexed.indices {
        try autoreleasepool {
          let allocated = UInt64(device.currentAllocatedSize)
          let reserve = UInt64(1024 * 1024 * 1024)
          let available = ProcessInfo.processInfo.physicalMemory > allocated + reserve
            ? ProcessInfo.processInfo.physicalMemory - allocated - reserve : 0
          guard available > 0 else {
            throw failure("Insufficient bounded memory for independent runtime-rANS parity")
          }
          let reference = try MetalRuntimeANSResidentSource.load(
            source: indexed[sourceIndex], device: device, maximumAdditionalBytes: available)
          let series = try MetalRuntimeANSSeries(sources: [reference])
          defer {
            series.release()
            reference.releaseResidentStorage()
          }
          for mask in masks {
            let result = try series.updateVirtualDetectorBuffers(
              mask: mask.values, forceRebase: true)
            guard let buffer = result.buffers.first,
              let expected = baseline[mask.name]?[sourceIndex]
            else { throw failure("Missing independent parity buffers") }
            let actual = Array(
              UnsafeBufferPointer(
                start: buffer.contents().assumingMemoryBound(to: UInt32.self),
                count: 512 * 512))
            guard actual == expected else {
              let first = zip(actual.indices, zip(actual, expected)).first {
                $0.1.0 != $0.1.1
              }!.0
              let mismatchCount = zip(actual, expected).reduce(0) {
                $0 + ($1.0 == $1.1 ? 0 : 1)
              }
              let row = first / 512
              let column = first % 512
              let pairedDP = try residents[sourceIndex].extractRawDiffraction(
                scanRow: row, scanColumn: column)
              let referenceDP = try reference.extractRawDiffraction(
                scanRow: row, scanColumn: column)
              let effective = effectiveMask(mask.values, resident: residents[sourceIndex])
              let pairedSum = pairedDP.indices.reduce(UInt64(0)) {
                $0 + (effective.values[$1] == 1 ? UInt64(pairedDP[$1]) : 0)
              }
              let referenceSum = referenceDP.indices.reduce(UInt64(0)) {
                $0 + (effective.values[$1] == 1 ? UInt64(referenceDP[$1]) : 0)
              }
              emitJSON([
                "event": "ans_opt_independent_mismatch", "source": sourceIndex,
                "mask": mask.name, "first_scan": first, "scan_row": row,
                "scan_column": column, "mismatch_count": mismatchCount,
                "paired_value": expected[first], "reference_value": actual[first],
                "paired_raw_dp_sum": pairedSum, "reference_raw_dp_sum": referenceSum,
                "paired_raw_dp_sha256": hash(pairedDP),
                "reference_raw_dp_sha256": hash(referenceDP),
              ])
              throw failure(
                "Independent runtime-rANS mismatch for source \(sourceIndex) \(mask.name)")
            }
            emitJSON([
              "event": "ans_opt_independent_parity", "status": "passed",
              "source": sourceIndex, "mask": mask.name,
              "sha256_u32_le": hash(actual), "values": actual.count,
            ])
          }
        }
      }
    }
    emitJSON(["event": "ans_opt_complete", "independent_parity_skipped": skipIndependent])
  }

  private static func experimentUpdateAll(
    residents: [MetalPairedRuntimeTANSResidentSource], mask: [UInt8], batched: Bool = false,
    boundedConcurrency: Int = 7
  ) async throws -> [ExperimentResult] {
    guard boundedConcurrency > 0 else {
      throw failure("resident loop bounded_concurrency must be positive")
    }
    if batched {
      let effective = residents.map { effectiveMask(mask, resident: $0) }
      var ordered = [ExperimentResult?](repeating: nil, count: residents.count)
      for start in stride(from: 0, to: residents.count, by: boundedConcurrency) {
        let end = min(start + boundedConcurrency, residents.count)
        let indices = Array(start..<end)
        let updates = try MetalPairedRuntimeTANSResidentSource.updateVirtualDetectors(
          sources: indices.map { residents[$0] },
          masks: indices.map { effective[$0].values })
        for (offset, source) in indices.enumerated() {
          let result = updates[offset]
          ordered[source] = ExperimentResult(
            source: source, values: result.values,
            wallMilliseconds: result.wallMilliseconds,
            gpuMilliseconds: result.gpuMilliseconds,
            changed: result.changedDetectorPixels, excluded: effective[source].excluded,
            polarFields: residents[source].lastPolarFieldCount,
            polarResiduals: residents[source].lastPolarResidualCount)
        }
      }
      return ordered.compactMap { $0 }
    }
    return try await withThrowingTaskGroup(of: ExperimentResult.self) { group in
      var nextSource = 0
      var ordered = [ExperimentResult?](repeating: nil, count: residents.count)

      // Keep the ordinary per-resident update path, but make its maximum
      // number of in-flight source updates measurable. Previously this
      // setting only affected the batched API, leaving the fastest unbatched
      // path fixed at all sources submitted simultaneously.
      while nextSource < min(boundedConcurrency, residents.count) {
        let source = nextSource
        let resident = residents[source]
        nextSource += 1
        group.addTask {
          let effective = effectiveMask(mask, resident: resident)
          let result = try resident.updateVirtualDetector(mask: effective.values)
          return ExperimentResult(
            source: source, values: result.values,
            wallMilliseconds: result.wallMilliseconds,
            gpuMilliseconds: result.gpuMilliseconds,
            changed: result.changedDetectorPixels, excluded: effective.excluded,
            polarFields: resident.lastPolarFieldCount,
            polarResiduals: resident.lastPolarResidualCount)
        }
      }

      while let result = try await group.next() {
        ordered[result.source] = result
        guard nextSource < residents.count else { continue }
        let source = nextSource
        let resident = residents[source]
        nextSource += 1
        group.addTask {
          let effective = effectiveMask(mask, resident: resident)
          let result = try resident.updateVirtualDetector(mask: effective.values)
          return ExperimentResult(
            source: source, values: result.values,
            wallMilliseconds: result.wallMilliseconds,
            gpuMilliseconds: result.gpuMilliseconds,
            changed: result.changedDetectorPixels, excluded: effective.excluded,
            polarFields: resident.lastPolarFieldCount,
            polarResiduals: resident.lastPolarResidualCount)
        }
      }
      return ordered.compactMap { $0 }
    }
  }

  private static func effectiveMask(
    _ requested: [UInt8], resident: MetalPairedRuntimeTANSResidentSource
  ) -> (values: [UInt8], excluded: Int) {
    let valid = resident.detectorValidityMask
    var values = requested
    var excluded = 0
    for pixel in values.indices where values[pixel] != 0 && valid[pixel] == 0 {
      values[pixel] = 0
      excluded += 1
    }
    return (values, excluded)
  }

  private static func modeDiagnosticMasks(
    _ masks: [ExperimentMask]
  ) -> [(String, [UInt8])] {
    func values(_ name: String) -> [UInt8] {
      masks.first { $0.name == name }!.values
    }
    func delta(_ first: [UInt8], _ second: [UInt8]) -> [UInt8] {
      zip(first, second).map { $0 == $1 ? 0 : 1 }
    }
    let bf = values("bf-base")
    let adf = values("adf-base")
    return [
      ("bf-base", bf), ("bf-center-1-delta", delta(bf, values("bf-center-1"))),
      ("adf-base", adf), ("adf-center-1-delta", delta(adf, values("adf-center-1"))),
    ]
  }

  private static func experimentMasks() -> [ExperimentMask] {
    func mask(
      _ name: String, _ row: Int, _ column: Int, _ inner: Int, _ outer: Int
    ) -> ExperimentMask {
      ExperimentMask(
        name: name,
        values: circularMask(
          centerColumnOffset: column, centerRowOffset: row,
          innerRadius: inner, outerRadius: outer))
    }
    return [
      mask("bf-base", 0, 0, 0, 46), mask("bf-center-1", 0, 1, 0, 46),
      mask("bf-center-8", 5, 8, 0, 46), mask("bf-center-20", 12, 20, 0, 46),
      mask("bf-radius-1", 0, 0, 0, 47), mask("bf-radius-8", 0, 0, 0, 54),
      mask("bf-radius-20", 0, 0, 0, 66),
      mask("abf-base", 0, 0, 24, 64), mask("abf-center-1", 0, 1, 24, 64),
      mask("abf-center-8", 5, 8, 24, 64), mask("abf-radius-1", 0, 0, 24, 65),
      mask("abf-radius-8", 0, 0, 24, 72), mask("abf-radius-20", 0, 0, 24, 84),
      mask("adf-base", 0, 0, 48, 94), mask("adf-center-1", 0, 1, 48, 94),
      mask("adf-center-8", 5, 8, 48, 94), mask("adf-center-20", 12, 20, 48, 94),
      mask("adf-radius-1", 0, 0, 48, 95), mask("adf-radius-8", 0, 0, 48, 102),
      mask("adf-radius-20", 0, 0, 48, 114),
    ]
  }

  private static func hash(_ values: [UInt32]) -> String {
    values.withUnsafeBytes {
      SHA256.hash(data: Data($0)).map { String(format: "%02x", $0) }.joined()
    }
  }

  private static func emitJSON(_ value: [String: Any]) {
    guard let data = try? JSONSerialization.data(withJSONObject: value, options: [.sortedKeys]),
      let line = String(data: data, encoding: .utf8)
    else { return }
    print(line)
    fflush(stdout)
  }

  static func circularMask(
    centerColumnOffset: Int = 0, centerRowOffset: Int = 0,
    innerRadius: Int, outerRadius: Int
  ) -> [UInt8] {
    let center = 96
    return (0..<(192 * 192)).map { pixel in
      let row = pixel / 192 - center - centerRowOffset
      let column = pixel % 192 - center - centerColumnOffset
      let squared = row * row + column * column
      return squared >= innerRadius * innerRadius && squared <= outerRadius * outerRadius ? 1 : 0
    }
  }

  static func percentile(_ values: [Double], _ fraction: Double) -> Double {
    let sorted = values.sorted()
    return sorted[min(sorted.count - 1, Int(ceil(Double(sorted.count) * fraction)) - 1)]
  }

  static func failure(_ message: String) -> NSError {
    NSError(domain: message, code: 1)
  }
}
