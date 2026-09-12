import CryptoKit
import Foundation
import Metal
import XCTest

@testable import Metal4DSTEMStreamingIO

/// Package diagnostic for the "all acquisitions move together" interaction:
/// every pointer input publishes one exact image per retained acquisition.
/// It measures the exact cost curve of that query as a function of how far the
/// detector geometry moved since the previous publication, under a sweep of
/// decoder launch configurations, and verifies bit-exactness of every measured
/// configuration against an unseeded recompute.
///
/// It is a throughput diagnostic, not an application frame rate and not an FPS
/// claim: the application's published frame rate follows from this curve plus
/// its own per-frame overhead and the pointer speed.
final class TANSLockstepThroughputExperimentTests: XCTestCase {
  private struct Sample: Encodable {
    let configuration: String
    let scenario: String
    let stepPixels: Double
    let step: Int
    let acquisitions: Int
    let wallMs: Double
    let gpuMs: Double
    let encodeMs: Double
    let waitMs: Double
    let columns: Int
    let tiles: Int
    let modelGroups: Int
    let mixedModelGroups: Int
    let paddedModelLanes: Int
    let usedPrevious: Bool
    let atlasField: Int
  }

  private struct ConfigurationResult: Encodable {
    let name: String
    let supported: Bool
    let error: String?
    let exact: Bool
    let samples: [Sample]
  }

  /// One named decoder launch configuration. `apply` runs after every knob has
  /// been reset to the package defaults, so each entry states only its
  /// difference from that baseline.
  private struct Configuration {
    let name: String
    let apply: (MetalTANSResidentSeries) -> Void
  }

  /// The kernel `configureInteractiveGrouping` selects for the application's
  /// interactive queries: the packet-owner kernel unless
  /// QUANTEM_TANS_PACKET_OWNER=0 (the same rule, so the `app-current*` arms
  /// measure what the application runs).
  private static let applicationPacketOwnerKernel =
    ProcessInfo.processInfo.environment["QUANTEM_TANS_PACKET_OWNER"] != "0"

  private func configurations() -> [Configuration] {
    [
      Configuration(name: "app-baseline", apply: { _ in }),
      Configuration(name: "plan-choose-base", apply: { $0.experimentalPlanAfterIndex = true }),
      Configuration(name: "threads64", apply: { $0.experimentalSharedModelThreadgroupWidth = 64 }),
      Configuration(
        name: "threads256", apply: { $0.experimentalSharedModelThreadgroupWidth = 256 }),
      Configuration(
        name: "threads512", apply: { $0.experimentalSharedModelThreadgroupWidth = 512 }),
      Configuration(name: "unroll2", apply: { $0.experimentalPairLoopUnroll = 2 }),
      Configuration(name: "unroll4", apply: { $0.experimentalPairLoopUnroll = 4 }),
      Configuration(name: "deferred2", apply: { $0.experimentalDeferredReductionPairs = 2 }),
      Configuration(name: "deferred4", apply: { $0.experimentalDeferredReductionPairs = 4 }),
      Configuration(name: "bit-extract", apply: { $0.experimentalDecoderBitExtract = true }),
      Configuration(name: "prefetch-entry", apply: { $0.experimentalPrefetchDecoderEntry = true }),
      Configuration(
        name: "staged-reduction", apply: { $0.experimentalStagedDetectorReduction = true }),
      Configuration(
        name: "zero-runs",
        apply: {
          $0.experimentalZeroRunDecoding = true
          $0.experimentalPreparedZeroRuns = true
        }),
      Configuration(name: "signed-pair", apply: { $0.experimentalSignedPairReduction = true }),
      // The application enables mixed-model tails but with a savings gate of
      // 1/8, which live evidence shows is never met on all-acquisition frames
      // (mixed groups stay near 0.3% of all groups). Divisor 0 regroups every
      // partial model group; it is a launch-geometry change only.
      Configuration(name: "divisor0", apply: { $0.experimentalMixedTailSavingsDivisor = 0 }),
      Configuration(
        name: "divisor0-prefetch",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalPrefetchDecoderEntry = true
        }),
      Configuration(
        name: "divisor0-concurrent",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalConcurrentDetectorDispatches = true
        }),
      Configuration(
        name: "divisor0-records64",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalDetectorRecordsPerCommand = 64
        }),
      Configuration(
        name: "divisor0-packet-major",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalPacketMajorGrid = true
        }),
      Configuration(
        name: "divisor0-base-prefetch",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalPrefetchDecoderEntry = true
          $0.experimentalPlanAfterIndex = true
        }),
      // What `configureInteractiveGrouping(mixedModelTails: true,
      // chooseCheaperBase: true)` now gives the application, including its
      // kernel selection.
      Configuration(
        name: "app-current",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalPrefetchDecoderEntry = true
          $0.experimentalPlanAfterIndex = true
          $0.experimentalPacketOwnerKernel = Self.applicationPacketOwnerKernel
        }),
      Configuration(
        name: "app-current-tile13",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalPrefetchDecoderEntry = true
          $0.experimentalPlanAfterIndex = true
          $0.experimentalTileCost = 1.3
          $0.experimentalPacketOwnerKernel = Self.applicationPacketOwnerKernel
        }),
      Configuration(
        name: "app-current-tile20",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalPrefetchDecoderEntry = true
          $0.experimentalPlanAfterIndex = true
          $0.experimentalTileCost = 2.0
          $0.experimentalPacketOwnerKernel = Self.applicationPacketOwnerKernel
        }),
      Configuration(
        name: "app-current-tile20-atlas",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalPrefetchDecoderEntry = true
          $0.experimentalPlanAfterIndex = true
          $0.experimentalTileCost = 2.0
          $0.experimentalUseAtlas = true
          $0.experimentalPacketOwnerKernel = Self.applicationPacketOwnerKernel
        }),
      // In-run controls: the same two configurations decoded by the independent
      // shared-model kernel, the timing counterpart of the cross-check.
      Configuration(
        name: "app-current-tile20-shared",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalPrefetchDecoderEntry = true
          $0.experimentalPlanAfterIndex = true
          $0.experimentalTileCost = 2.0
          $0.experimentalPacketOwnerKernel = false
        }),
      Configuration(
        name: "app-current-tile20-atlas-shared",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalPrefetchDecoderEntry = true
          $0.experimentalPlanAfterIndex = true
          $0.experimentalTileCost = 2.0
          $0.experimentalUseAtlas = true
          $0.experimentalPacketOwnerKernel = false
        }),
      Configuration(
        name: "app-current-tile30",
        apply: {
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalPrefetchDecoderEntry = true
          $0.experimentalPlanAfterIndex = true
          $0.experimentalTileCost = 3.0
          $0.experimentalPacketOwnerKernel = Self.applicationPacketOwnerKernel
        }),
      Configuration(
        name: "records-per-command-64", apply: { $0.experimentalDetectorRecordsPerCommand = 64 }),
      Configuration(
        name: "no-mixed-tails",
        apply: {
          $0.experimentalMixedModelTails = false
          $0.experimentalMixedTailSavingsDivisor = 0
          $0.experimentalSeparateMixedDispatches = false
          $0.experimentalMixedOnlySpecialization = false
        }),
    ]
  }

  /// Reset every decoder knob to the package defaults (the shared-model kernel
  /// unless QUANTEM_TANS_PACKET_OWNER=1, as a series is constructed), so each
  /// arm states only its own difference. The application's own state is
  /// `app-current-tile20`, which selects the kernel as
  /// `configureInteractiveGrouping` does.
  private func resetToPackageDefaults(_ source: MetalTANSResidentSeries) {
    source.experimentalDetectorStreamsPerLane = 32
    source.experimentalUseTileIndex = true
    source.experimentalPlanAfterIndex = false
    source.experimentalTileCost = 0.5
    source.experimentalUseAtlas = false
    source.experimentalSharedModelThreadgroupWidth = nil
    source.experimentalSharedPacketsPerLane = 1
    source.experimentalPairLoopUnroll = 1
    source.experimentalDeferredReductionPairs = 1
    source.experimentalDecoderBitExtract = false
    source.experimentalPrefetchDecoderEntry = false
    source.experimentalStagedDetectorReduction = false
    source.experimentalSharedRefill16 = false
    source.experimentalDirectDecodingTable = false
    source.experimentalZeroRunDecoding = false
    source.experimentalPreparedZeroRuns = false
    source.experimentalZeroBitArithmetic = false
    source.experimentalSignedPairReduction = false
    source.experimentalPacketMajorGrid = false
    source.experimentalPairLookupBits = 0
    source.experimentalCompilerThreadgroupLimit = 0
    source.experimentalDetectorRecordsPerCommand = 0
    source.experimentalDetectorQueueCount = 1
    source.experimentalConcurrentDetectorDispatches = false
    source.experimentalNarrowOutputDeclarations = false
    source.experimentalMixedModelTails = true
    source.experimentalMixedTailSavingsDivisor = 8
    source.experimentalSeparateMixedDispatches = true
    source.experimentalMixedOnlySpecialization = true
    source.experimentalPacketOwnerKernel = MetalTANSResidentSeries.defaultPacketOwner
  }

  func testAllAcquisitionLockstepThroughputWhenConfigured() throws {
    let environment = ProcessInfo.processInfo.environment
    guard let path = environment["QUANTEM_TANS_LOCKSTEP_FIXTURE"] else {
      throw XCTSkip("Requires the complete sealed 66-acquisition archive and a 128 GB-class device")
    }
    let device = try XCTUnwrap(MTLCreateSystemDefaultDevice())
    let source = try MetalTANSResidentSeries(
      directory: URL(fileURLWithPath: path), acquisitions: Array(0..<66), device: device,
      maximumAdditionalBytes: ProcessInfo.processInfo.physicalMemory * 4 / 5
        - UInt64(device.currentAllocatedSize))
    defer { source.releaseResidentStorage() }
    let acquisitionCount = source.acquisitionIndices.count

    let prepareStart = ProcessInfo.processInfo.systemUptime
    // Layout and budget are selectable so layouts can be compared on the same
    // archive. The on-disk cache only ever holds the application's layout.
    let layout =
      TANSExactTileIndex.Layout(rawValue: environment["QUANTEM_TANS_LOCKSTEP_LAYOUT"] ?? "")
      ?? .centerFine
    let indexBudget = UInt64(Int(environment["QUANTEM_TANS_LOCKSTEP_INDEX_GIB"] ?? "") ?? 2) << 30
    if layout == .centerFine, let cache = environment["QUANTEM_TANS_LOCKSTEP_CACHE"] {
      do {
        try source.importExperimentalTileIndex(
          from: URL(fileURLWithPath: cache), maximumIndexBytes: indexBudget)
      } catch {
        try source.prepareExperimentalTileIndex(maximumIndexBytes: indexBudget)
      }
    } else {
      try source.prepareExperimentalTileIndex(maximumIndexBytes: indexBudget, layout: layout)
    }
    let indexSeconds = ProcessInfo.processInfo.systemUptime - prepareStart
    print(
      "LOCKSTEP_LOAD layout=\(layout.rawValue) acquisitions=\(acquisitionCount) source_bytes=\(source.residentBytes) index_bytes=\(source.experimentalTileIndexBytes) load_s=\(source.loadSeconds) index_s=\(indexSeconds)"
    )
    fflush(stdout)

    let valid = source.validDetectorMask
    // Same geometry rule as the application's ADF product: an annulus around a
    // fractional centre, over the archive's valid detector pixels only.
    func mask(row: Double, col: Double, inner: Double, outer: Double) -> [UInt8] {
      (0..<36864).map { q in
        let r = Double(q / 192) - row
        let c = Double(q % 192) - col
        let radius2 = r * r + c * c
        return valid[q] != 0 && radius2 >= inner * inner && radius2 <= outer * outer ? 1 : 0
      }
    }
    func hashes(_ buffers: [MTLBuffer]) -> [String] {
      buffers.map { buffer in
        SHA256.hash(
          data: Data(bytesNoCopy: buffer.contents(), count: buffer.length, deallocator: .none)
        ).map { String(format: "%02x", $0) }.joined()
      }
    }

    let inner = 40.0
    let outer = 80.0
    let steps =
      (environment["QUANTEM_TANS_LOCKSTEP_STEPS"] ?? "1,2,4,8,16,24")
      .split(separator: ",").compactMap { Double($0) }
    let walkLength = Int(environment["QUANTEM_TANS_LOCKSTEP_WALK"] ?? "") ?? 6
    let requested = (environment["QUANTEM_TANS_LOCKSTEP_CONFIGS"] ?? "")
      .split(separator: ",").map(String.init)
    let selected = configurations().filter { requested.isEmpty || requested.contains($0.name) }
    XCTAssertFalse(selected.isEmpty, "No known configuration matched the request")

    var results: [ConfigurationResult] = []
    let budget: UInt64 = 8 << 30

    // Exact annulus atlas: build only the lattice centres this run touches,
    // the four around every walk frame and every phase probe. A frame's cost
    // depends only on its nearest centres, so this matches a full lattice.
    let atlasSpacing = Double(environment["QUANTEM_TANS_LOCKSTEP_ATLAS_SPACING"] ?? "")
    var phaseCentres: [(row: Double, col: Double)] = []
    if let spacing = atlasSpacing {
      var lattice = Set<[Int]>()
      func touch(_ row: Double, _ col: Double) {
        let i = Int(((row - 95.5) / spacing).rounded(.down))
        let j = Int(((col - 95.5) / spacing).rounded(.down))
        for di in 0...1 { for dj in 0...1 { lattice.insert([i + di, j + dj]) } }
      }
      touch(95.5, 95.5)
      for stepPixels in steps {
        var row = 95.5
        var col = 95.5
        for step in 0..<walkLength {
          let angle = Double(step) / Double(max(1, walkLength)) * .pi / 3
          row += stepPixels * sin(angle)
          col += stepPixels * cos(angle)
          touch(row, col)
        }
      }
      // Phase probes: random sub-pixel centres 0.3-1.4 px from a lattice
      // centre, half near the pattern centre and half 16-22 px out.
      // Probe distance band from the nearest lattice centre (default 0.3-2 px).
      // A narrow band near 0 measures the small residuals an anchored step cap
      // would publish.
      let phaseMin = Double(environment["QUANTEM_TANS_LOCKSTEP_PHASE_MIN_PX"] ?? "") ?? 0.3
      let phaseMax = Double(environment["QUANTEM_TANS_LOCKSTEP_PHASE_MAX_PX"] ?? "") ?? 2.0
      var generator = SplitMix64(state: 20_260_910)
      let cells: [(Int, Int, Int)] = [(0, 0, 8), (-1, 0, 7), (9, 0, 5), (0, -10, 5), (6, -7, 5)]
      for (i, j, count) in cells {
        var made = 0
        while made < count {
          let u = Double.random(in: 0..<spacing, using: &generator)
          let v = Double.random(in: 0..<spacing, using: &generator)
          let nearest = [(0.0, 0.0), (spacing, 0.0), (0.0, spacing), (spacing, spacing)]
            .map { ((u - $0.0) * (u - $0.0) + (v - $0.1) * (v - $0.1)).squareRoot() }.min()!
          guard nearest >= phaseMin, nearest <= phaseMax else { continue }
          let row = 95.5 + Double(i) * spacing + u
          let col = 95.5 + Double(j) * spacing + v
          phaseCentres.append((row, col))
          touch(row, col)
          made += 1
        }
      }
      let centres = lattice.sorted { $0[0] != $1[0] ? $0[0] < $1[0] : $0[1] < $1[1] }
      let masks = centres.map {
        mask(
          row: 95.5 + Double($0[0]) * spacing, col: 95.5 + Double($0[1]) * spacing,
          inner: inner, outer: outer)
      }
      let atlasBudget = UInt64(Int(environment["QUANTEM_TANS_LOCKSTEP_ATLAS_GIB"] ?? "") ?? 6) << 30
      let buildStart = ProcessInfo.processInfo.systemUptime
      try source.prepareExperimentalAtlas(masks: masks, maximumBytes: atlasBudget)
      let buildSeconds = ProcessInfo.processInfo.systemUptime - buildStart
      print(
        "LOCKSTEP_ATLAS spacing_px=\(spacing) fields=\(masks.count) bytes=\(source.experimentalAtlasBytes) bytes_per_field=\(source.experimentalAtlasBytes / masks.count) build_s=\(buildSeconds) recommended_working_set=\(device.recommendedMaxWorkingSetSize) allocated=\(device.currentAllocatedSize)"
      )
      fflush(stdout)
    }

    for configuration in selected {
      var samples: [Sample] = []
      var exact = true
      var failure: String?
      do {
        resetToPackageDefaults(source)
        configuration.apply(source)

        for stepPixels in steps {
          // Start every walk from the same geometry with an unseeded recompute,
          // so each measured chain is independent and reproducible.
          var row = 95.5
          var col = 95.5
          var current = mask(row: row, col: col, inner: inner, outer: outer)
          let rebaseStart = ProcessInfo.processInfo.systemUptime
          _ = try source.detectorImages(
            mask: current, maximumAdditionalBytes: budget, rebase: true)
          let rebaseWall = ProcessInfo.processInfo.systemUptime - rebaseStart
          samples.append(
            sample(
              source, configuration: configuration.name, scenario: "rebase-all",
              stepPixels: stepPixels, step: 0, acquisitions: acquisitionCount, wall: rebaseWall))

          var chained: [String] = []
          for step in 0..<walkLength {
            // A straight walk, one geometry change per published frame: this is
            // what the interactive tier sees when the pointer moves `stepPixels`
            // detector pixels between two publications.
            let angle = Double(step) / Double(max(1, walkLength)) * .pi / 3
            row += stepPixels * sin(angle)
            col += stepPixels * cos(angle)
            current = mask(row: row, col: col, inner: inner, outer: outer)
            let started = ProcessInfo.processInfo.systemUptime
            let images = try source.detectorImages(
              mask: current, maximumAdditionalBytes: budget, rebase: false)
            let wall = ProcessInfo.processInfo.systemUptime - started
            XCTAssertEqual(images.count, acquisitionCount)
            samples.append(
              sample(
                source, configuration: configuration.name, scenario: "lockstep-walk",
                stepPixels: stepPixels, step: step, acquisitions: images.count, wall: wall))
            if step == walkLength - 1 { chained = hashes(images) }
            // With the atlas on, every frame is checked on its own against an
            // unseeded recompute with the atlas off, so the reference cannot
            // start from the atlas itself. The ring has three slots, so hash
            // before the reference query rotates it.
            if source.experimentalUseAtlas {
              let produced = hashes(images)
              source.experimentalUseAtlas = false
              let reference = hashes(
                try source.detectorImages(
                  mask: current, maximumAdditionalBytes: budget, rebase: true))
              source.experimentalUseAtlas = true
              if reference != produced {
                exact = false
                failure = "step=\(stepPixels) frame=\(step) atlas frame differs from recompute"
              }
              XCTAssertEqual(
                reference, produced,
                "\(configuration.name): atlas frame \(step) at \(stepPixels) px must equal an unseeded recompute"
              )
            }
          }

          // Exactness of the whole seeded chain for every acquisition: the final
          // images must equal an unseeded recompute of the same final geometry.
          let reference = try source.detectorImages(
            mask: current, maximumAdditionalBytes: budget, rebase: true)
          let referenceHashes = hashes(reference)
          if referenceHashes != chained {
            exact = false
            let mismatched = zip(referenceHashes, chained).enumerated()
              .filter { $0.element.0 != $0.element.1 }.map { $0.offset }
            failure =
              "step=\(stepPixels) mismatched acquisitions \(mismatched.prefix(8).map(String.init).joined(separator: ","))"
          }
          XCTAssertEqual(
            referenceHashes, chained,
            "\(configuration.name): seeded lockstep chain at \(stepPixels) px must equal an unseeded recompute"
          )
          // Independent-kernel cross-check: the same final geometry recomputed
          // by the shared-model kernel.
          if source.experimentalPacketOwnerKernel
            && environment["QUANTEM_TANS_LOCKSTEP_CROSSCHECK"] == "1"
          {
            source.experimentalPacketOwnerKernel = false
            let independent = hashes(
              try source.detectorImages(
                mask: current, maximumAdditionalBytes: budget, rebase: true))
            source.experimentalPacketOwnerKernel = true
            print(
              "LOCKSTEP_CROSSCHECK configuration=\(configuration.name) step_px=\(stepPixels) exact=\(independent == chained)"
            )
            fflush(stdout)
            if independent != chained {
              exact = false
              failure = "step=\(stepPixels) packet-owner chain differs from the shared-model kernel"
            }
            XCTAssertEqual(
              independent, chained,
              "\(configuration.name): packet-owner chain must equal the shared-model kernel")
          }
        }
        results.append(
          ConfigurationResult(
            name: configuration.name, supported: true, error: failure, exact: exact,
            samples: samples))
      } catch {
        // An unsupported knob combination throws its guard message; record it
        // and continue so one sweep can cover every candidate.
        print("LOCKSTEP_UNSUPPORTED configuration=\(configuration.name) error=\(error)")
        fflush(stdout)
        results.append(
          ConfigurationResult(
            name: configuration.name, supported: false, error: "\(error)", exact: false,
            samples: samples))
      }
    }

    resetToPackageDefaults(source)

    // Machine-readable summary: median wall/GPU per (configuration, step) over
    // the walk, which is the cost curve the application's frame rate follows.
    for result in results where result.supported {
      for stepPixels in steps {
        let walk = result.samples.filter {
          $0.scenario == "lockstep-walk" && $0.stepPixels == stepPixels
        }
        guard !walk.isEmpty else { continue }
        func median(_ values: [Double]) -> Double {
          let sorted = values.sorted()
          return sorted[sorted.count / 2]
        }
        print(
          "LOCKSTEP_SAMPLE configuration=\(result.name) step_px=\(stepPixels) n=\(walk.count) wall_ms=\(median(walk.map(\.wallMs))) gpu_ms=\(median(walk.map(\.gpuMs))) encode_ms=\(median(walk.map(\.encodeMs))) columns=\(median(walk.map { Double($0.columns) })) tiles=\(median(walk.map { Double($0.tiles) })) model_groups=\(median(walk.map { Double($0.modelGroups) })) mixed_groups=\(median(walk.map { Double($0.mixedModelGroups) })) padded_lanes=\(median(walk.map { Double($0.paddedModelLanes) })) gpu_max_ms=\(walk.map(\.gpuMs).max()!) atlas_frames=\(walk.filter { $0.atlasField >= 0 }.count) exact=\(result.exact)"
        )
      }
      fflush(stdout)
    }

    // Full recompute from the tile index: the cost a fast drag falls back to
    // when continuing from the previous image would cost more.
    for result in results where result.supported {
      let full = result.samples.filter { $0.scenario == "rebase-all" }
      guard !full.isEmpty else { continue }
      let wall = full.map(\.wallMs).sorted()[full.count / 2]
      let gpu = full.map(\.gpuMs).sorted()[full.count / 2]
      let columns = full.map(\.columns).sorted()[full.count / 2]
      let tiles = full.map(\.tiles).sorted()[full.count / 2]
      print(
        "LOCKSTEP_FULL configuration=\(result.name) layout=\(layout.rawValue) n=\(full.count) wall_ms=\(wall) gpu_ms=\(gpu) columns=\(columns) tiles=\(tiles)"
      )
    }
    fflush(stdout)

    // Atlas phase probes: each is an unseeded query with the atlas on, checked
    // against the same query with the atlas off. Cost should depend only on the
    // distance to the nearest lattice centre, not on where the pattern is.
    if let spacing = atlasSpacing {
      resetToPackageDefaults(source)
      configurations().first { $0.name == "app-current-tile20-atlas" }!.apply(source)
      for (probe, centre) in phaseCentres.enumerated() {
        let probeMask = mask(row: centre.row, col: centre.col, inner: inner, outer: outer)
        source.experimentalUseAtlas = true
        let started = ProcessInfo.processInfo.systemUptime
        let images = try source.detectorImages(
          mask: probeMask, maximumAdditionalBytes: budget, rebase: true)
        let wall = ProcessInfo.processInfo.systemUptime - started
        let produced = hashes(images)
        let gpu = source.lastDetectorGPUSeconds * 1000
        let columns = source.lastDetectorDecodedColumns
        let groups = source.lastDetectorModelGroups
        let field = source.lastDetectorAtlasField ?? -1
        source.experimentalUseAtlas = false
        let reference = hashes(
          try source.detectorImages(mask: probeMask, maximumAdditionalBytes: budget, rebase: true))
        XCTAssertEqual(reference, produced, "atlas phase probe \(probe) must equal a recompute")
        if source.experimentalPacketOwnerKernel
          && environment["QUANTEM_TANS_LOCKSTEP_CROSSCHECK"] == "1"
        {
          source.experimentalPacketOwnerKernel = false
          let independent = hashes(
            try source.detectorImages(
              mask: probeMask, maximumAdditionalBytes: budget, rebase: true))
          source.experimentalPacketOwnerKernel = true
          print("LOCKSTEP_PROBE_CROSSCHECK probe=\(probe) exact=\(independent == produced)")
          XCTAssertEqual(
            independent, produced, "atlas phase probe \(probe) must equal the shared-model kernel")
        }
        let dr = centre.row - 95.5
        let dc = centre.col - 95.5
        let lr = dr - (dr / spacing).rounded() * spacing
        let lc = dc - (dc / spacing).rounded() * spacing
        print(
          "LOCKSTEP_ATLAS_PHASE probe=\(probe) offset_px=\((dr * dr + dc * dc).squareRoot()) lattice_px=\((lr * lr + lc * lc).squareRoot()) wall_ms=\(wall * 1000) gpu_ms=\(gpu) columns=\(columns) model_groups=\(groups) atlas_field=\(field) exact=\(reference == produced)"
        )
        fflush(stdout)
      }
      resetToPackageDefaults(source)
    }

    // Tiny-residual frames: unseeded atlas queries 0.05-0.15 px from a stored
    // lattice centre (about one mixed group per record), each checked bit for
    // bit against an atlas-off recompute. Two passes; the first warms.
    if atlasSpacing != nil, environment["QUANTEM_TANS_LOCKSTEP_TINY"] != "0" {
      let tinyCentres: [(Double, Double)] = [
        (95.6, 95.5), (95.5, 95.62), (97.4, 95.5), (97.5, 97.35),
        (95.45, 97.6), (97.58, 97.42), (95.52, 95.47), (97.45, 95.55),
      ]
      resetToPackageDefaults(source)
      configurations().first { $0.name == "app-current-tile20-atlas" }!.apply(source)
      var gpus: [Double] = []
      var groups: [Double] = []
      var mixed: [Double] = []
      var columns: [Double] = []
      var exactCount = 0
      for pass in 0..<2 {
        for centre in tinyCentres {
          let tinyMask = mask(row: centre.0, col: centre.1, inner: inner, outer: outer)
          source.experimentalUseAtlas = true
          let images = try source.detectorImages(
            mask: tinyMask, maximumAdditionalBytes: budget, rebase: true)
          let produced = hashes(images)
          if pass == 1 {
            gpus.append(source.lastDetectorGPUSeconds * 1000)
            groups.append(Double(source.lastDetectorModelGroups))
            mixed.append(Double(source.lastDetectorMixedModelGroups))
            columns.append(Double(source.lastDetectorDecodedColumns))
          }
          source.experimentalUseAtlas = false
          let reference = hashes(
            try source.detectorImages(
              mask: tinyMask, maximumAdditionalBytes: budget, rebase: true))
          XCTAssertEqual(reference, produced, "tiny atlas frame must equal a recompute")
          if pass == 1 && reference == produced { exactCount += 1 }
        }
      }
      func median(_ values: [Double]) -> Double { values.sorted()[values.count / 2] }
      print(
        "LOCKSTEP_TINY gpu_med=\(median(gpus)) gpu_min=\(gpus.min()!) groups=\(median(groups)) mixed=\(median(mixed)) cols=\(median(columns)) exact=\(exactCount)/\(tinyCentres.count)"
      )
      fflush(stdout)
      resetToPackageDefaults(source)
    }

    // Seed fragmentation after a catch-up pass: every acquisition refreshed in
    // its own call with the same geometry, as the application's catch-up tier
    // does, then one all-acquisition query. Reports how many seed groups that
    // query dispatched and verifies it against an unseeded recompute.
    if environment["QUANTEM_TANS_LOCKSTEP_FRAGMENT"] == "1" {
      resetToPackageDefaults(source)
      configurations().first { $0.name == "app-current" }!.apply(source)
      let anchorMask = mask(row: 95.5, col: 95.5, inner: inner, outer: outer)
      _ = try source.detectorImages(
        mask: anchorMask, maximumAdditionalBytes: budget, rebase: true)
      let caughtUp = mask(row: 95.5, col: 98.5, inner: inner, outer: outer)
      for acquisition in source.acquisitionIndices {
        _ = try source.detectorImages(
          mask: caughtUp, maximumAdditionalBytes: budget, rebase: false,
          selectedAcquisitions: [acquisition])
      }
      let next = mask(row: 95.5, col: 99.5, inner: inner, outer: outer)
      let started = ProcessInfo.processInfo.systemUptime
      let images = try source.detectorImages(
        mask: next, maximumAdditionalBytes: budget, rebase: false)
      let wall = ProcessInfo.processInfo.systemUptime - started
      let seedGroups = Int(source.lastDetectorCommandTiming["seed_groups"] ?? -1)
      let gpu = source.lastDetectorGPUSeconds * 1000
      let columns = source.lastDetectorDecodedColumns
      let chained = hashes(images)
      let reference = hashes(
        try source.detectorImages(mask: next, maximumAdditionalBytes: budget, rebase: true))
      XCTAssertEqual(chained, reference, "post-catch-up all-acquisition query must be exact")
      print(
        "LOCKSTEP_FRAGMENT seed_groups=\(seedGroups) wall_ms=\(wall * 1000) gpu_ms=\(gpu) columns=\(columns) exact=\(chained == reference)"
      )
      fflush(stdout)
    }

    if let output = environment["QUANTEM_TANS_LOCKSTEP_OUTPUT"] {
      let encoder = JSONEncoder()
      encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
      let report: [String: Any] = [
        "fixture": path, "acquisitions": acquisitionCount,
        "source_bytes": source.residentBytes,
        "index_bytes": source.experimentalTileIndexBytes,
        "load_seconds": source.loadSeconds, "index_seconds": indexSeconds,
        "configurations": try JSONSerialization.jsonObject(with: encoder.encode(results)),
      ]
      try JSONSerialization.data(withJSONObject: report, options: [.prettyPrinted, .sortedKeys])
        .write(to: URL(fileURLWithPath: output))
    }
  }

  private func sample(
    _ source: MetalTANSResidentSeries, configuration: String, scenario: String,
    stepPixels: Double, step: Int, acquisitions: Int, wall: Double
  ) -> Sample {
    let timing = source.lastDetectorCommandTiming
    return Sample(
      configuration: configuration, scenario: scenario, stepPixels: stepPixels, step: step,
      acquisitions: acquisitions, wallMs: wall * 1000,
      gpuMs: source.lastDetectorGPUSeconds * 1000,
      encodeMs: timing["encode_ms"] ?? -1, waitMs: timing["wait_ms"] ?? -1,
      columns: source.lastDetectorDecodedColumns, tiles: source.lastDetectorTileFields,
      modelGroups: source.lastDetectorModelGroups,
      mixedModelGroups: source.lastDetectorMixedModelGroups,
      paddedModelLanes: source.lastDetectorPaddedModelLanes,
      usedPrevious: source.lastDetectorUsedPrevious,
      atlasField: source.lastDetectorAtlasField ?? -1)
  }
}

/// Deterministic generator so the phase probes are identical on every run.
private struct SplitMix64: RandomNumberGenerator {
  var state: UInt64
  mutating func next() -> UInt64 {
    state &+= 0x9E37_79B9_7F4A_7C15
    var z = state
    z = (z ^ (z >> 30)) &* 0xBF58_476D_1CE4_E5B9
    z = (z ^ (z >> 27)) &* 0x94D0_49BB_1331_11EB
    return z ^ (z >> 31)
  }
}
