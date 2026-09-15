import Foundation
import os

/// Choose the memory/interaction tradeoff without changing scientific counts.
public enum MetalResidentInteractionMode: String, Sendable, CaseIterable {
  case normal
  case fast
}

/// A resident keeps its configuration for its entire lifetime. UI changes must
/// never mutate process environment while another resident is executing.
struct PairedRuntimeConfiguration: Sendable {
  let mode: MetalResidentInteractionMode?
  private let values: [String: String]

  init(mode: MetalResidentInteractionMode?) {
    self.mode = mode
    values =
      mode == .fast
      ? pairedRuntimeInteractiveDefaults.merging(pairedRuntimeSpeedDefaults) { _, fast in fast }
      : [:]
  }

  func value(_ name: String) -> String? {
    mode == nil ? pairedRuntimeEnvironment(name) : values[name]
  }

  func isExplicit(_ name: String) -> Bool {
    mode == nil ? pairedRuntimeEnvironmentIsExplicit(name) : values[name] != nil
  }
}

/// Read one environment variable without copying the whole process environment.
/// `ProcessInfo.processInfo.environment` builds a dictionary on every access
/// (~50 µs with ~100 variables), which dominated per-update host preparation.
private struct PairedRuntimeEnvironmentSnapshot {
  var generation: UInt64 = 1
  var cachedGeneration: UInt64 = 0
  var values: [String: String] = [:]
  var legacyDictionaryLookup = false
  var directGetenv = false
  var interactiveDefaults = false
  var speedDefaults = false
}

/// Exact interactive path used when a key is unset and
/// `QGPU_PAIRED_RUNTIME_PROFILE_DEFAULTS` explicitly selects `interactive`:
/// radial1/leaf16 regional
/// sums with the scan512 query, validated trusted decode table, window bit reader,
/// trusted stream setup, radial1 stream order, rank-ordered residuals and compact
/// block-32 stream offsets. Every setting is lossless and adds no resident memory
/// beyond the regional-sum index.
private let pairedRuntimeInteractiveDefaults: [String: String] = [
  "QGPU_PAIRED_RUNTIME_POLAR_INDEX": "1",
  "QGPU_PAIRED_RUNTIME_POLAR_LEAF_PIXELS": "16",
  // radial1 with 4-pixel leaves inside radius 24 (radius 32: +0.75 GB index for seven tilts,
  // ABF center drags 73 -> 98 fps in the native viewer; 20260914-apple-m5-ans-core4-index).
  "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT": "radial1core4",
  "QGPU_PAIRED_RUNTIME_PREPARE_POLAR_QUERY_SCAN512": "1",
  "QGPU_PAIRED_RUNTIME_POLAR_QUERY_VARIANT": "scan512",
  "QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_TABLE": "1",
  "QGPU_PAIRED_RUNTIME_TRUSTED_TABLE": "1",
  "QGPU_PAIRED_RUNTIME_PREPARE_WINDOW_READER": "1",
  "QGPU_PAIRED_RUNTIME_WINDOW_READER": "1",
  "QGPU_PAIRED_RUNTIME_PREPARE_TRUSTED_SETUP": "1",
  "QGPU_PAIRED_RUNTIME_TRUSTED_SETUP": "1",
  "QGPU_PAIRED_RUNTIME_RESIDUAL_ORDER": "rank",
  "QGPU_PAIRED_RUNTIME_STREAM_ORDER": "radial1",
  "QGPU_PAIRED_RUNTIME_COMPACT_OFFSETS": "1",
]

/// Added by `QGPU_PAIRED_RUNTIME_PROFILE_DEFAULTS=interactive-speed`: exact 1-byte
/// gap/value events for streams with at most 192 nonzero counts (mode 251), decoded
/// up to four per trip, loaded without the per-record scratch, with 4-pixel regional
/// sums in the ADF ring as well as the core. Costs resident memory (about +2.6 to
/// +3.4 GB over `interactive` for seven 512 x 512 x 192 x 192 tilts).
private let pairedRuntimeSpeedDefaults: [String: String] = [
  "QGPU_PAIRED_RUNTIME_COMPACT_EVENT_MAX_NONZERO": "192",
  "QGPU_PAIRED_RUNTIME_SCRATCHLESS_ENCODE": "1",
  "QGPU_PAIRED_RUNTIME_PREPARE_COMPACT_PAIRS": "1",
  "QGPU_PAIRED_RUNTIME_COMPACT_PAIRS": "1",
  "QGPU_PAIRED_RUNTIME_COMPACT_QUADS": "1",
  // 4-pixel leaves also in the ADF ring 40-60 px. Additional memory depends on
  // the acquisition; native timing and memory evidence lives in the registry.
  "QGPU_PAIRED_RUNTIME_POLAR_LAYOUT": "radial1fine4",
]

private let pairedRuntimeEnvironmentState = OSAllocatedUnfairLock(
  initialState: PairedRuntimeEnvironmentSnapshot())

/// Read one QGPU configuration variable. Values come from a snapshot of the
/// process environment taken on first use and after every
/// `pairedRuntimeEnvironmentDidChange()`: `getenv` serializes on the C library's
/// environment lock, and seven concurrent source updates made about 60 reads each.
@inline(__always)
func pairedRuntimeEnvironment(_ name: String) -> String? {
  let (value, legacy, direct) = pairedRuntimeEnvironmentState.withLock {
    state -> (String?, Bool, Bool) in
    if state.cachedGeneration != state.generation {
      state.values = ProcessInfo.processInfo.environment.filter { $0.key.hasPrefix("QGPU_") }
      // Benchmark-only A/B switches restoring earlier lookup costs.
      state.legacyDictionaryLookup =
        state.values["QGPU_PAIRED_RUNTIME_LEGACY_ENVIRONMENT_LOOKUP"] == "1"
      state.directGetenv = state.values["QGPU_PAIRED_RUNTIME_DIRECT_GETENV"] == "1"
      state.interactiveDefaults = ["interactive", "interactive-speed"].contains(
        state.values["QGPU_PAIRED_RUNTIME_PROFILE_DEFAULTS"] ?? "legacy")
      state.speedDefaults =
        state.values["QGPU_PAIRED_RUNTIME_PROFILE_DEFAULTS"] == "interactive-speed"
      state.cachedGeneration = state.generation
    }
    let explicit = state.values[name]
    let resolved =
      explicit
      ?? (state.speedDefaults ? pairedRuntimeSpeedDefaults[name] : nil)
      ?? (state.interactiveDefaults ? pairedRuntimeInteractiveDefaults[name] : nil)
    return (resolved, state.legacyDictionaryLookup, state.directGetenv)
  }
  if legacy { return ProcessInfo.processInfo.environment[name] }
  if direct {
    guard let raw = getenv(name) else { return nil }
    return String(cString: raw)
  }
  return value
}

/// Whether a QGPU setting was set explicitly rather than supplied by the interactive profile.
func pairedRuntimeEnvironmentIsExplicit(_ name: String) -> Bool {
  pairedRuntimeEnvironmentState.withLock { state in
    if state.cachedGeneration != state.generation {
      state.values = ProcessInfo.processInfo.environment.filter { $0.key.hasPrefix("QGPU_") }
      state.legacyDictionaryLookup =
        state.values["QGPU_PAIRED_RUNTIME_LEGACY_ENVIRONMENT_LOOKUP"] == "1"
      state.directGetenv = state.values["QGPU_PAIRED_RUNTIME_DIRECT_GETENV"] == "1"
      state.interactiveDefaults = ["interactive", "interactive-speed"].contains(
        state.values["QGPU_PAIRED_RUNTIME_PROFILE_DEFAULTS"] ?? "legacy")
      state.speedDefaults =
        state.values["QGPU_PAIRED_RUNTIME_PROFILE_DEFAULTS"] == "interactive-speed"
      state.cachedGeneration = state.generation
    }
    return state.values[name] != nil || !state.interactiveDefaults
  }
}

/// Invalidate the configuration snapshot after changing QGPU variables at runtime.
@_spi(PairedRuntimeTANSPrototype)
public func pairedRuntimeEnvironmentDidChange() {
  pairedRuntimeEnvironmentState.withLock { $0.generation &+= 1 }
}

/// Exact signed-mask decomposition for the paired-runtime polar interaction index.
struct PairedRuntimeTANSPolarPlan: Sendable {
  static let leafPixels = 64
  static let rootLeaves = 16

  let detectorRows: Int
  let detectorColumns: Int
  let selectedFields: [UInt32]
  let fieldCoefficients: [Int32]
  let residualPixels: [UInt32]
  let residualCoefficients: [Int32]
  let usedIndex: Bool
  let leafPixelCount: Int
  let layoutKind: String
  private let validPixels: [UInt8]

  private struct CacheKey: Equatable {
    let delta: [Int32]
    let validPixels: [UInt8]
    let detectorRows: Int
    let detectorColumns: Int
    let leafPixels: Int
    let layoutKind: String
    let jointPlan: Bool
  }

  private struct CacheEntry {
    let key: CacheKey
    let plan: PairedRuntimeTANSPolarPlan
  }

  private static let cacheLock = NSLock()
  nonisolated(unsafe) private static var cachedPlan: CacheEntry?
  nonisolated(unsafe) private static var cacheHits: UInt64 = 0
  nonisolated(unsafe) private static var cacheMisses: UInt64 = 0
  nonisolated(unsafe) private static var cacheLockWaitSeconds: Double = 0
  nonisolated(unsafe) private static var cacheBuildSeconds: Double = 0
  nonisolated(unsafe) private static var uncachedCalls: UInt64 = 0
  nonisolated(unsafe) private static var uncachedBuildSeconds: Double = 0

  struct IndexLayout: Sendable {
    let permutation: [Int32]
    let leaves: Int
    let roots: Int
  }

  /// Cached field membership for the paired-runtime 192 by 192 detector ABI.
  private static let indexLayout192Leaf16 = makeLayout(
    detectorRows: 192, detectorColumns: 192, leafPixels: 16)
  private static let indexLayout192Leaf32 = makeLayout(
    detectorRows: 192, detectorColumns: 192, leafPixels: 32)
  static let indexLayout192 = makeLayout(
    detectorRows: 192, detectorColumns: 192, leafPixels: leafPixels)
  private static let radial1Layouts = [16, 32, 64].reduce(into: [Int: IndexLayout]()) {
    $0[$1] = makeLayout(
      detectorRows: 192, detectorColumns: 192, leafPixels: $1, layoutKind: "radial1")
  }
  /// radial1 order with 4-pixel leaves (4 pixels in 16 slots) inside radius 24, where
  /// off-center crescents are a few pixels wide, and 16-pixel leaves elsewhere.
  static let radial1Core4Radius = 24.0
  private static let radial1Core4Layouts: [Int: IndexLayout] = [
    16: makeLayout(
      detectorRows: 192, detectorColumns: 192, leafPixels: 16, layoutKind: "radial1core4")
  ]
  /// radial1core4 plus 4-pixel leaves in the ring 40 <= radius < 60 (inner ADF edges).
  private static let radial1Fine4Layouts: [Int: IndexLayout] = [
    16: makeLayout(
      detectorRows: 192, detectorColumns: 192, leafPixels: 16, layoutKind: "radial1fine4")
  ]
  /// Layouts whose leaves may hold fewer real pixels than slots.
  static func isPaddedLayout(_ layoutKind: String) -> Bool {
    layoutKind == "radial1core4" || layoutKind == "radial1fine4"
  }
  static func usesFineLeaf(radius: Double, layoutKind: String) -> Bool {
    radius < radial1Core4Radius
      || (layoutKind == "radial1fine4" && radius >= 40 && radius < 60)
  }
  private static let radialHalfLayouts = [16, 32, 64].reduce(into: [Int: IndexLayout]()) {
    $0[$1] = makeLayout(
      detectorRows: 192, detectorColumns: 192, leafPixels: $1, layoutKind: "radialhalf")
  }

  /// Return cached field membership for one supported exact leaf width.
  static func indexLayout(leafPixels: Int, layoutKind: String = "polar") -> IndexLayout? {
    if layoutKind == "radial1" { return radial1Layouts[leafPixels] }
    if layoutKind == "radial1core4" { return radial1Core4Layouts[leafPixels] }
    if layoutKind == "radial1fine4" { return radial1Fine4Layouts[leafPixels] }
    if layoutKind == "radialhalf" { return radialHalfLayouts[leafPixels] }
    guard layoutKind == "polar" else { return nil }
    return switch leafPixels {
    case 16: indexLayout192Leaf16
    case 32: indexLayout192Leaf32
    case 64: indexLayout192
    default: nil
    }
  }

  var estimatedCost: Int {
    selectedFields.count + 4 * residualPixels.count
  }

  /// Build the same unweighted polar plan as `_compact.paired.polar_planner`.
  ///
  /// The greedy plan for cached 192 by 192 layouts uses `makeGreedyFast`;
  /// `QGPU_PAIRED_RUNTIME_LEGACY_PLANNER=1` selects the original planner for A/B runs.
  static func make(
    delta: [Int32], validPixels: [UInt8], detectorRows: Int, detectorColumns: Int,
    leafPixels requestedLeafPixels: Int = 64, layoutKind: String = "polar"
  ) -> PairedRuntimeTANSPolarPlan {
    let profile = pairedRuntimeEnvironment("QGPU_PAIRED_RUNTIME_PROFILE") == "1"
    let jointPlan = pairedRuntimeEnvironment("QGPU_PAIRED_RUNTIME_JOINT_PLAN") == "1"
    let legacyPlanner = pairedRuntimeEnvironment("QGPU_PAIRED_RUNTIME_LEGACY_PLANNER") == "1"
    guard pairedRuntimeEnvironment("QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN") == "1" else {
      let buildStarted = profile ? CFAbsoluteTimeGetCurrent() : 0
      let plan = makeUncached(
        delta: delta, validPixels: validPixels,
        detectorRows: detectorRows, detectorColumns: detectorColumns,
        leafPixels: requestedLeafPixels, layoutKind: layoutKind,
        jointPlan: jointPlan, legacyPlanner: legacyPlanner)
      if profile {
        let elapsed = CFAbsoluteTimeGetCurrent() - buildStarted
        cacheLock.lock()
        uncachedCalls += 1
        uncachedBuildSeconds += elapsed
        cacheLock.unlock()
      }
      return plan
    }
    let key = CacheKey(
      delta: delta, validPixels: validPixels,
      detectorRows: detectorRows, detectorColumns: detectorColumns,
      leafPixels: requestedLeafPixels, layoutKind: layoutKind, jointPlan: jointPlan)
    let lockStarted = profile ? CFAbsoluteTimeGetCurrent() : 0
    cacheLock.lock()
    defer { cacheLock.unlock() }
    if profile { cacheLockWaitSeconds += CFAbsoluteTimeGetCurrent() - lockStarted }
    if let cachedPlan, cachedPlan.key == key {
      if profile { cacheHits += 1 }
      return cachedPlan.plan
    }
    let buildStarted = profile ? CFAbsoluteTimeGetCurrent() : 0
    let plan = makeUncached(
      delta: delta, validPixels: validPixels,
      detectorRows: detectorRows, detectorColumns: detectorColumns,
      leafPixels: requestedLeafPixels, layoutKind: layoutKind,
      jointPlan: jointPlan, legacyPlanner: legacyPlanner)
    if profile {
      cacheMisses += 1
      cacheBuildSeconds += CFAbsoluteTimeGetCurrent() - buildStarted
    }
    cachedPlan = CacheEntry(key: key, plan: plan)
    return plan
  }

  /// Return cumulative opt-in cache diagnostics under the cache lock.
  static func cacheProfileSnapshot() -> [String: Double] {
    cacheLock.lock()
    defer { cacheLock.unlock() }
    return [
      "hits": Double(cacheHits),
      "misses": Double(cacheMisses),
      "lock_wait_seconds": cacheLockWaitSeconds,
      "build_seconds": cacheBuildSeconds,
      "uncached_calls": Double(uncachedCalls),
      "uncached_build_seconds": uncachedBuildSeconds,
      "shared_cache_enabled": pairedRuntimeEnvironment("QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN")
        == "1" ? 1 : 0,
    ]
  }

  private static func makeUncached(
    delta: [Int32], validPixels: [UInt8], detectorRows: Int, detectorColumns: Int,
    leafPixels requestedLeafPixels: Int, layoutKind: String, jointPlan: Bool,
    legacyPlanner: Bool
  ) -> PairedRuntimeTANSPolarPlan {
    if !jointPlan, !legacyPlanner,
      let plan = makeGreedyFast(
        delta: delta, validPixels: validPixels,
        detectorRows: detectorRows, detectorColumns: detectorColumns,
        leafPixels: requestedLeafPixels, layoutKind: layoutKind)
    {
      return plan
    }
    let product = detectorRows.multipliedReportingOverflow(by: detectorColumns)
    guard detectorRows > 0, detectorColumns > 0, !product.overflow,
      delta.count == product.partialValue,
      validPixels.count == product.partialValue
    else {
      return direct(
        delta: delta, validPixels: validPixels,
        detectorRows: detectorRows, detectorColumns: detectorColumns,
        leafPixels: requestedLeafPixels, layoutKind: layoutKind)
    }
    let pixelCount = product.partialValue
    var effective = delta
    for pixel in effective.indices where validPixels[pixel] == 0 {
      effective[pixel] = 0
    }
    guard detectorRows == 192, detectorColumns == 192 else {
      return direct(
        delta: effective, validPixels: [UInt8](repeating: 1, count: pixelCount),
        detectorRows: detectorRows, detectorColumns: detectorColumns,
        leafPixels: requestedLeafPixels, layoutKind: layoutKind)
    }
    guard
      let layout = indexLayout(
        leafPixels: requestedLeafPixels, layoutKind: layoutKind)
    else {
      return direct(
        delta: effective, validPixels: [UInt8](repeating: 1, count: pixelCount),
        detectorRows: detectorRows, detectorColumns: detectorColumns,
        leafPixels: requestedLeafPixels, layoutKind: layoutKind)
    }
    let options: [Int32] = [0, 1, -1]
    var leafCoefficients = [Int32](repeating: 0, count: layout.leaves)
    var values = [Int32](repeating: 0, count: layout.permutation.count)
    for ordinal in layout.permutation.indices {
      let pixel = Int(layout.permutation[ordinal])
      if pixel >= 0 { values[ordinal] = effective[pixel] }
    }
    var rootCoefficients = [Int32](repeating: 0, count: layout.roots)
    var residualValues = [Int32](repeating: 0, count: pixelCount)
    if jointPlan {
      // Optimize the existing two-level basis without changing its stored sums.
      // `localTarget` is the detector coefficient represented by a root plus
      // its leaf. The emitted leaf coefficient is localTarget - root, which
      // can be +/-2 and is supported by the Int32 query coefficients.
      var negativeCounts = [UInt8](repeating: 0, count: layout.leaves)
      var zeroCounts = [UInt8](repeating: 0, count: layout.leaves)
      var positiveCounts = [UInt8](repeating: 0, count: layout.leaves)
      var validCounts = [UInt8](repeating: 0, count: layout.leaves)
      for ordinal in layout.permutation.indices {
        let pixel = Int(layout.permutation[ordinal])
        guard pixel >= 0, validPixels[pixel] != 0 else { continue }
        let leaf = ordinal / requestedLeafPixels
        validCounts[leaf] += 1
        switch values[ordinal] {
        case -1: negativeCounts[leaf] += 1
        case 0: zeroCounts[leaf] += 1
        case 1: positiveCounts[leaf] += 1
        default: break
        }
      }

      for root in 0..<layout.roots {
        var bestRoot = options[0]
        var bestRootCost = Int.max
        var bestTargets = [Int32](repeating: 0, count: rootLeaves)
        for rootOption in options {
          var rootCost = rootOption == 0 ? 0 : 1
          var targets = [Int32](repeating: 0, count: rootLeaves)
          for offset in 0..<rootLeaves {
            let leaf = root * rootLeaves + offset
            guard leaf < layout.leaves else { continue }
            let validCount = Int(validCounts[leaf])
            var bestTarget = options[0]
            var bestLeafCost = Int.max
            for target in options {
              let matchingCount: Int
              switch target {
              case -1: matchingCount = Int(negativeCounts[leaf])
              case 0: matchingCount = Int(zeroCounts[leaf])
              default: matchingCount = Int(positiveCounts[leaf])
              }
              let leafFieldCost = target == rootOption ? 0 : 1
              let residualCost = 4 * (validCount - matchingCount)
              let leafCost = leafFieldCost + residualCost
              if leafCost < bestLeafCost
                || (leafCost == bestLeafCost && target == rootOption
                  && bestTarget != rootOption)
              {
                bestTarget = target
                bestLeafCost = leafCost
              }
            }
            targets[offset] = bestTarget
            rootCost += bestLeafCost
          }
          if rootCost < bestRootCost {
            bestRoot = rootOption
            bestRootCost = rootCost
            bestTargets = targets
          }
        }
        rootCoefficients[root] = bestRoot
        for offset in 0..<rootLeaves {
          let leaf = root * rootLeaves + offset
          if leaf < layout.leaves {
            leafCoefficients[leaf] = bestTargets[offset] - bestRoot
          }
        }
      }
      for ordinal in layout.permutation.indices {
        let pixel = Int(layout.permutation[ordinal])
        if pixel >= 0 && validPixels[pixel] != 0 {
          let leaf = ordinal / requestedLeafPixels
          let root = leaf / rootLeaves
          let localTarget = leafCoefficients[leaf] + rootCoefficients[root]
          residualValues[pixel] = values[ordinal] - localTarget
        }
      }
    } else {
      // Preserve the established greedy planner exactly when the diagnostic
      // environment gate is absent.
      for leaf in 0..<layout.leaves {
        let first = leaf * requestedLeafPixels
        var best = options[0]
        var bestCount = -1
        for option in options {
          var count = 0
          for ordinal in first..<(first + requestedLeafPixels) {
            let pixel = Int(layout.permutation[ordinal])
            // Padding counts toward option 0 only in fully packed layouts; padded
            // core leaves vote with their real pixels.
            if pixel >= 0 ? validPixels[pixel] != 0 : !isPaddedLayout(layoutKind) {
              count += values[ordinal] == option ? 1 : 0
            }
          }
          if count > bestCount {
            best = option
            bestCount = count
          }
        }
        leafCoefficients[leaf] = best
      }

      for ordinal in layout.permutation.indices {
        let pixel = Int(layout.permutation[ordinal])
        if pixel >= 0 && validPixels[pixel] != 0 {
          residualValues[pixel] = values[ordinal] - leafCoefficients[ordinal / requestedLeafPixels]
        }
      }
      for root in 0..<layout.roots {
        var best = options[0]
        var bestCount = -1
        for option in options {
          var count = 0
          for offset in 0..<rootLeaves {
            let leaf = root * rootLeaves + offset
            let value = leaf < layout.leaves ? leafCoefficients[leaf] : 0
            if value == option { count += 1 }
          }
          if count > bestCount {
            best = option
            bestCount = count
          }
        }
        rootCoefficients[root] = best
      }
      for leaf in leafCoefficients.indices {
        leafCoefficients[leaf] -= rootCoefficients[leaf / rootLeaves]
      }
    }

    let fields = leafCoefficients + rootCoefficients
    let selectedFields = fields.indices.compactMap {
      fields[$0] == 0 ? nil : UInt32($0)
    }
    let fieldCoefficients = selectedFields.map { fields[Int($0)] }
    let residualPixels = residualValues.indices.compactMap {
      residualValues[$0] == 0 ? nil : UInt32($0)
    }
    let residualCoefficients = residualPixels.map { residualValues[Int($0)] }
    let indexedCost = selectedFields.count + 4 * residualPixels.count
    let directCost = 4 * effective.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
    guard indexedCost < directCost else {
      return direct(
        delta: effective, validPixels: [UInt8](repeating: 1, count: pixelCount),
        detectorRows: detectorRows, detectorColumns: detectorColumns,
        leafPixels: requestedLeafPixels, layoutKind: layoutKind)
    }
    return PairedRuntimeTANSPolarPlan(
      detectorRows: detectorRows, detectorColumns: detectorColumns,
      selectedFields: selectedFields, fieldCoefficients: fieldCoefficients,
      residualPixels: residualPixels, residualCoefficients: residualCoefficients,
      usedIndex: true, leafPixelCount: requestedLeafPixels,
      layoutKind: layoutKind, validPixels: validPixels)
  }

  /// Reconstruct the complete signed detector delta exactly on the CPU.
  func reconstructedDelta() -> [Int32] {
    guard usedIndex, detectorRows == 192, detectorColumns == 192 else {
      var result = [Int32](repeating: 0, count: max(0, detectorRows * detectorColumns))
      for (pixel, coefficient) in zip(residualPixels, residualCoefficients)
      where Int(pixel) < result.count {
        result[Int(pixel)] += coefficient
      }
      return result
    }
    guard
      let layout = Self.indexLayout(
        leafPixels: leafPixelCount, layoutKind: layoutKind)
    else { return [] }
    var fields = [Int32](repeating: 0, count: layout.leaves + layout.roots)
    for (field, coefficient) in zip(selectedFields, fieldCoefficients) {
      guard Int(field) < fields.count else { return [] }
      fields[Int(field)] = coefficient
    }
    var result = [Int32](repeating: 0, count: detectorRows * detectorColumns)
    for ordinal in layout.permutation.indices {
      let pixel = Int(layout.permutation[ordinal])
      if pixel >= 0 && validPixels[pixel] != 0 {
        let leaf = ordinal / leafPixelCount
        let root = leaf / Self.rootLeaves
        result[pixel] = fields[leaf] + fields[layout.leaves + root]
      }
    }
    for (pixel, coefficient) in zip(residualPixels, residualCoefficients) {
      guard Int(pixel) < result.count else { return [] }
      result[Int(pixel)] += coefficient
    }
    return result
  }

  private struct OrderedPixel {
    let pixel: Int
    let radialBand: Int
    let angle: Double
    let radius: Double
  }

  private static func makeLayout(
    detectorRows: Int, detectorColumns: Int, leafPixels: Int,
    layoutKind: String = "polar"
  ) -> IndexLayout {
    let centerRow = Double(detectorRows - 1) / 2
    let centerColumn = Double(detectorColumns - 1) / 2
    var ordered: [OrderedPixel] = []
    ordered.reserveCapacity(detectorRows * detectorColumns)
    for detectorRow in 0..<detectorRows {
      for detectorColumn in 0..<detectorColumns {
        let row = Double(detectorRow) - centerRow
        let column = Double(detectorColumn) - centerColumn
        let radius = hypot(row, column)
        ordered.append(
          OrderedPixel(
            pixel: detectorRow * detectorColumns + detectorColumn,
            radialBand: layoutKind == "radial1" || isPaddedLayout(layoutKind)
              ? Int(floor(radius))
              : layoutKind == "radialhalf"
                ? Int(floor(radius * 2))
                : Int(floor(pow(radius, 1.5) / 45)),
            angle: atan2(row, column), radius: radius))
      }
    }
    ordered.sort {
      if $0.radialBand != $1.radialBand { return $0.radialBand < $1.radialBand }
      if $0.angle != $1.angle { return $0.angle < $1.angle }
      if $0.radius != $1.radius { return $0.radius < $1.radius }
      return $0.pixel < $1.pixel
    }
    let pixelCount = detectorRows * detectorColumns
    if isPaddedLayout(layoutKind) {
      // Fine leaves hold four pixels and twelve empty slots; the leaf count is padded
      // to whole roots so every root still sums sixteen leaves.
      var slots: [Int32] = []
      var index = 0
      while index < ordered.count {
        let core = usesFineLeaf(radius: ordered[index].radius, layoutKind: layoutKind)
        let take = core ? 4 : leafPixels
        var leaf = [Int32](repeating: -1, count: leafPixels)
        var filled = 0
        while filled < take, index < ordered.count,
          usesFineLeaf(radius: ordered[index].radius, layoutKind: layoutKind) == core
        {
          leaf[filled] = Int32(ordered[index].pixel)
          filled += 1
          index += 1
        }
        slots += leaf
      }
      var leaves = slots.count / leafPixels
      let paddedLeaves = (leaves + rootLeaves - 1) / rootLeaves * rootLeaves
      slots += [Int32](repeating: -1, count: (paddedLeaves - leaves) * leafPixels)
      leaves = paddedLeaves
      return IndexLayout(permutation: slots, leaves: leaves, roots: leaves / rootLeaves)
    }
    let leaves = (pixelCount + leafPixels - 1) / leafPixels
    let roots = (leaves + rootLeaves - 1) / rootLeaves
    var permutation = [Int32](repeating: -1, count: leaves * leafPixels)
    for (ordinal, item) in ordered.enumerated() { permutation[ordinal] = Int32(item.pixel) }
    return IndexLayout(permutation: permutation, leaves: leaves, roots: roots)
  }

  private static func direct(
    delta: [Int32], validPixels: [UInt8], detectorRows: Int, detectorColumns: Int,
    leafPixels: Int, layoutKind: String
  ) -> PairedRuntimeTANSPolarPlan {
    let count = min(delta.count, validPixels.count)
    var pixels: [UInt32] = []
    var coefficients: [Int32] = []
    pixels.reserveCapacity(count)
    coefficients.reserveCapacity(count)
    for pixel in 0..<count where validPixels[pixel] != 0 && delta[pixel] != 0 {
      pixels.append(UInt32(pixel))
      coefficients.append(delta[pixel])
    }
    return PairedRuntimeTANSPolarPlan(
      detectorRows: detectorRows, detectorColumns: detectorColumns,
      selectedFields: [], fieldCoefficients: [],
      residualPixels: pixels, residualCoefficients: coefficients,
      usedIndex: false, leafPixelCount: leafPixels,
      layoutKind: layoutKind, validPixels: validPixels)
  }
}

// MARK: - Greedy fast path

extension PairedRuntimeTANSPolarPlan {
  /// Detector-order view of one cached index layout for `makeGreedyFast`.
  ///
  /// `leafOfPixel[pixel]` names the leaf that holds a detector pixel. The same
  /// membership is stored as row-major 64-bit bitmap words: leaf `l` covers
  /// entries `spanStarts[l]..<spanStarts[l + 1]` of `spanWords` and `spanBits`.
  private final class GreedyLayout: Sendable {
    let pixelCount: Int
    let words: Int
    let leafPixels: Int
    let leaves: Int
    let roots: Int
    let leafOfPixel: [UInt16]
    let leafPixelCounts: [UInt8]
    let spanStarts: [Int32]
    let spanWords: [UInt16]
    let spanBits: [UInt64]

    /// Return nil unless the permutation places every detector pixel in exactly one leaf.
    init?(layout: IndexLayout, pixelCount: Int, leafPixels: Int) {
      let leaves = layout.leaves
      let rootLeaves = PairedRuntimeTANSPolarPlan.rootLeaves
      let words = (pixelCount + 63) / 64
      // Packed leaf counts keep four 8-bit fields; leaf and word ids fit in UInt16.
      guard pixelCount > 0, (1...255).contains(leafPixels),
        (1...Int(UInt16.max) + 1).contains(leaves), words <= Int(UInt16.max) + 1,
        layout.permutation.count == leaves * leafPixels,
        layout.roots == (leaves + rootLeaves - 1) / rootLeaves
      else { return nil }
      var leafOfPixel = [UInt16](repeating: 0, count: pixelCount)
      var leafPixelCounts = [UInt8](repeating: 0, count: leaves)
      var covered = [Bool](repeating: false, count: pixelCount)
      var spanStarts: [Int32] = [0]
      var spanWords: [UInt16] = []
      var spanBits: [UInt64] = []
      spanStarts.reserveCapacity(leaves + 1)
      for leaf in 0..<leaves {
        let firstSpan = spanWords.count
        for ordinal in leaf * leafPixels..<(leaf + 1) * leafPixels {
          let pixel = Int(layout.permutation[ordinal])
          guard pixel >= 0 else { continue }
          guard pixel < pixelCount, !covered[pixel] else { return nil }
          covered[pixel] = true
          leafOfPixel[pixel] = UInt16(leaf)
          leafPixelCounts[leaf] += 1
          let word = UInt16(pixel >> 6)
          let bit = UInt64(1) &<< UInt64(pixel & 63)
          if let span = spanWords[firstSpan...].firstIndex(of: word) {
            spanBits[span] |= bit
          } else {
            spanWords.append(word)
            spanBits.append(bit)
          }
        }
        spanStarts.append(Int32(spanWords.count))
      }
      guard !covered.contains(false) else { return nil }
      self.pixelCount = pixelCount
      self.words = words
      self.leafPixels = leafPixels
      self.leaves = leaves
      self.roots = layout.roots
      self.leafOfPixel = leafOfPixel
      self.leafPixelCounts = leafPixelCounts
      self.spanStarts = spanStarts
      self.spanWords = spanWords
      self.spanBits = spanBits
    }
  }

  /// All-valid mask stored by 192 by 192 direct fallback plans, shared instead of reallocated.
  private static let allValidPixels192 = [UInt8](repeating: 1, count: 192 * 192)
  private static let polarGreedyLayouts = greedyLayouts(layoutKind: "polar")
  private static let radial1GreedyLayouts = greedyLayouts(layoutKind: "radial1")
  private static let radialHalfGreedyLayouts = greedyLayouts(layoutKind: "radialhalf")
  private static let radial1Core4GreedyLayouts = greedyLayouts(layoutKind: "radial1core4")
  private static let radial1Fine4GreedyLayouts = greedyLayouts(layoutKind: "radial1fine4")

  private static func greedyLayouts(layoutKind: String) -> [Int: GreedyLayout] {
    [16, 32, 64].reduce(into: [Int: GreedyLayout]()) { layouts, leafPixels in
      if let layout = indexLayout(leafPixels: leafPixels, layoutKind: layoutKind),
        let greedy = GreedyLayout(layout: layout, pixelCount: 192 * 192, leafPixels: leafPixels)
      {
        layouts[leafPixels] = greedy
      }
    }
  }

  private static func greedyLayout(leafPixels: Int, layoutKind: String) -> GreedyLayout? {
    switch layoutKind {
    case "radial1": radial1GreedyLayouts[leafPixels]
    case "radialhalf": radialHalfGreedyLayouts[leafPixels]
    case "radial1core4": radial1Core4GreedyLayouts[leafPixels]
    case "radial1fine4": radial1Fine4GreedyLayouts[leafPixels]
    case "polar": polarGreedyLayouts[leafPixels]
    default: nil
    }
  }

  /// Build the non-joint greedy plan without per-pixel permutation passes.
  ///
  /// This reproduces the legacy greedy branch of `makeUncached` bit for bit:
  /// invalid pixels read as zero, each leaf takes the first most frequent
  /// option in [0, 1, -1] over padding and valid pixels, each root takes the
  /// same majority over its 16 leaf options, and residuals are listed in
  /// ascending pixel order. Row-major detector bitmaps and packed per-leaf
  /// counts replace the permuted value copy, and residuals follow from bitmap
  /// words. Returns nil when the layout or detector shape is not cached, so
  /// the legacy planner handles every other case and its early returns.
  private static func makeGreedyFast(
    delta: [Int32], validPixels: [UInt8], detectorRows: Int, detectorColumns: Int,
    leafPixels: Int, layoutKind: String
  ) -> PairedRuntimeTANSPolarPlan? {
    guard detectorRows == 192, detectorColumns == 192,
      let layout = greedyLayout(leafPixels: leafPixels, layoutKind: layoutKind),
      delta.count == layout.pixelCount, validPixels.count == layout.pixelCount
    else { return nil }
    return withUnsafeTemporaryAllocation(of: UInt64.self, capacity: 7 * layout.words) {
      bitmaps in
      withUnsafeTemporaryAllocation(of: UInt32.self, capacity: layout.leaves) { counts in
        withUnsafeTemporaryAllocation(
          of: Int8.self, capacity: layout.leaves + 2 * layout.roots
        ) { options in
          delta.withUnsafeBufferPointer { delta in
            validPixels.withUnsafeBufferPointer { valid in
              layout.leafOfPixel.withUnsafeBufferPointer { leafOfPixel in
                greedyPlan(
                  delta: delta.baseAddress!, valid: valid.baseAddress!,
                  leafOfPixel: leafOfPixel.baseAddress!, layout: layout,
                  bitmaps: bitmaps.baseAddress!, counts: counts.baseAddress!,
                  options: options.baseAddress!, validPixels: validPixels,
                  layoutKind: layoutKind)
              }
            }
          }
        }
      }
    }
  }

  private static func greedyPlan(
    delta: UnsafePointer<Int32>, valid: UnsafePointer<UInt8>,
    leafOfPixel: UnsafePointer<UInt16>, layout: GreedyLayout,
    bitmaps: UnsafeMutablePointer<UInt64>, counts: UnsafeMutablePointer<UInt32>,
    options: UnsafeMutablePointer<Int8>, validPixels: [UInt8], layoutKind: String
  ) -> PairedRuntimeTANSPolarPlan? {
    let pixelCount = layout.pixelCount
    let words = layout.words
    let leafPixels = layout.leafPixels
    let leaves = layout.leaves
    let roots = layout.roots
    // Row-major detector bitmaps, one 64-bit word per 64 pixels.
    let nonzero = bitmaps  // valid pixels with delta != 0
    let positive = bitmaps + words  // valid pixels with delta == 1
    let negative = bitmaps + 2 * words  // valid pixels with delta == -1
    let validBits = bitmaps + 3 * words
    let positiveLeaves = bitmaps + 4 * words  // pixels of leaves whose option is 1
    let negativeLeaves = bitmaps + 5 * words  // pixels of leaves whose option is -1
    let residual = bitmaps + 6 * words  // valid pixels whose delta differs from their leaf option
    positive.initialize(repeating: 0, count: 2 * words)
    positiveLeaves.initialize(repeating: 0, count: 2 * words)
    // Packed per-leaf counts: byte 0 delta == 1, byte 1 delta == -1,
    // byte 2 valid nonzero delta, byte 3 invalid pixels.
    counts.initialize(repeating: 0, count: leaves)

    // Valid pixels. memchr jumps to the next word that holds an invalid pixel.
    validBits.initialize(repeating: ~0, count: words)
    let tailWidth = pixelCount - (words - 1) * 64
    if tailWidth < 64 { validBits[words - 1] = (1 &<< UInt64(tailWidth)) &- 1 }
    let byteWeights = SIMD16<UInt8>(1, 2, 4, 8, 16, 32, 64, 128, 1, 2, 4, 8, 16, 32, 64, 128)
    var searchStart = 0
    while searchStart < pixelCount,
      let hit = memchr(valid + searchStart, 0, pixelCount - searchStart)
    {
      let word = UnsafeRawPointer(valid).distance(to: UnsafeRawPointer(hit)) / 64
      let first = word * 64
      var bits: UInt64 = 0
      if first + 64 <= pixelCount {
        let wordBytes = UnsafeRawPointer(valid + first)
        for block in 0..<4 {
          let bytes = wordBytes.loadUnaligned(fromByteOffset: 16 * block, as: SIMD16<UInt8>.self)
          let weighted = SIMD16<UInt8>()
            .replacing(with: byteWeights, where: bytes .!= SIMD16<UInt8>())
          let blockBits =
            UInt64(weighted.lowHalf.wrappedSum()) | UInt64(weighted.highHalf.wrappedSum()) &<< 8
          bits |= blockBits &<< UInt64(16 * block)
        }
      } else {
        for offset in 0..<tailWidth {
          bits |= (valid[first + offset] != 0 ? 1 : 0) &<< UInt64(offset)
        }
      }
      var invalid = validBits[word] & ~bits
      validBits[word] = bits
      while invalid != 0 {
        counts[Int(leafOfPixel[first + invalid.trailingZeroBitCount])] &+= 1 &<< 24
        invalid &= invalid &- 1
      }
      searchStart = first + 64
    }

    // Nonzero valid deltas, from the nonzero mask of four 16-pixel blocks per word.
    let rawDelta = UnsafeRawPointer(delta)
    let blockWeights = SIMD16<Int32>(
      1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768)
    var nonzeroCount = 0
    for word in 0..<words {
      let first = word * 64
      var bits: UInt64 = 0
      if first + 64 <= pixelCount {
        for block in 0..<4 {
          let values = rawDelta.loadUnaligned(
            fromByteOffset: (first + 16 * block) * 4, as: SIMD16<Int32>.self)
          let blockBits = SIMD16<Int32>()
            .replacing(with: blockWeights, where: values .!= SIMD16<Int32>())
            .wrappedSum()
          bits |= UInt64(UInt32(bitPattern: blockBits)) &<< UInt64(16 * block)
        }
      } else {
        for offset in 0..<tailWidth {
          bits |= (delta[first + offset] != 0 ? 1 : 0) &<< UInt64(offset)
        }
      }
      bits &= validBits[word]
      nonzero[word] = bits
      nonzeroCount += bits.nonzeroBitCount
    }

    // Classify nonzero deltas and count them per leaf.
    for word in 0..<words where nonzero[word] != 0 {
      let first = word * 64
      var positiveBits: UInt64 = 0
      var negativeBits: UInt64 = 0
      var remaining = nonzero[word]
      repeat {
        let offset = remaining.trailingZeroBitCount
        let value = delta[first + offset]
        let isPositive: UInt64 = value == 1 ? 1 : 0
        let isNegative: UInt64 = value == -1 ? 1 : 0
        positiveBits |= isPositive &<< UInt64(offset)
        negativeBits |= isNegative &<< UInt64(offset)
        counts[Int(leafOfPixel[first + offset])] &+=
          0x1_0000 | UInt32(truncatingIfNeeded: isPositive)
          | UInt32(truncatingIfNeeded: isNegative) &<< 8
        remaining &= remaining &- 1
      } while remaining != 0
      positive[word] = positiveBits
      negative[word] = negativeBits
    }

    // Leaf and root options, one root at a time; roots without counts keep option 0.
    let leafOptions = options
    let rootOptions = options + leaves
    let rootHasLeafOptions = options + leaves + roots
    options.initialize(repeating: 0, count: leaves + 2 * roots)
    var selectedCount = 0
    layout.spanStarts.withUnsafeBufferPointer { spanStarts in
      layout.spanWords.withUnsafeBufferPointer { spanWords in
        layout.spanBits.withUnsafeBufferPointer { spanBits in
          for root in 0..<roots {
            let first = root * rootLeaves
            let end = min(first + rootLeaves, leaves)
            var touched: UInt32 = 0
            for leaf in first..<end { touched |= counts[leaf] }
            guard touched != 0 else { continue }
            var positiveLeafCount = 0
            var negativeLeafCount = 0
            for leaf in first..<end {
              let packed = counts[leaf]
              let positivePixels = Int(packed & 0xff)
              let negativePixels = Int((packed &>> 8) & 0xff)
              // Padding ordinals and valid zero deltas both count toward 0, except in
              // padded core leaves, which vote with their real pixels only.
              let slots =
                isPaddedLayout(layoutKind) ? Int(layout.leafPixelCounts[leaf]) : leafPixels
              let zeroPixels = slots - Int(packed &>> 24) - Int((packed &>> 16) & 0xff)
              // First maximum over [0, 1, -1], as in the legacy planner.
              var best: Int8 = 0
              var bestCount = zeroPixels
              if positivePixels > bestCount {
                best = 1
                bestCount = positivePixels
              }
              if negativePixels > bestCount { best = -1 }
              guard best != 0 else { continue }
              leafOptions[leaf] = best
              let leafBits: UnsafeMutablePointer<UInt64>
              if best == 1 {
                positiveLeafCount += 1
                leafBits = positiveLeaves
              } else {
                negativeLeafCount += 1
                leafBits = negativeLeaves
              }
              for span in Int(spanStarts[leaf])..<Int(spanStarts[leaf + 1]) {
                leafBits[Int(spanWords[span])] |= spanBits[span]
              }
            }
            guard positiveLeafCount | negativeLeafCount != 0 else { continue }
            rootHasLeafOptions[root] = 1
            // Leaves missing from a partial root count as option 0.
            var best: Int8 = 0
            var bestCount = rootLeaves - positiveLeafCount - negativeLeafCount
            if positiveLeafCount > bestCount {
              best = 1
              bestCount = positiveLeafCount
            }
            if negativeLeafCount > bestCount { best = -1 }
            rootOptions[root] = best
            let matchingLeaves =
              switch best {
              case 1: positiveLeafCount
              case -1: negativeLeafCount
              default: end - first - positiveLeafCount - negativeLeafCount
              }
            selectedCount += end - first - matchingLeaves + (best == 0 ? 0 : 1)
          }
        }
      }
    }

    var residualCount = 0
    for word in 0..<words {
      let positiveLeaf = positiveLeaves[word]
      let negativeLeaf = negativeLeaves[word]
      let zeroLeafNonzero = nonzero[word] & ~(positiveLeaf | negativeLeaf)
      let positiveLeafMismatch = positiveLeaf & ~positive[word]
      let negativeLeafMismatch = negativeLeaf & ~negative[word]
      let bits =
        (zeroLeafNonzero | positiveLeafMismatch | negativeLeafMismatch) & validBits[word]
      residual[word] = bits
      residualCount += bits.nonzeroBitCount
    }

    guard selectedCount + 4 * residualCount < 4 * nonzeroCount else {
      // Same fallback as `direct(delta: effective, validPixels: all ones)`.
      var emitted = 0
      var residualCoefficients: [Int32] = []
      let residualPixels = [UInt32](unsafeUninitializedCapacity: nonzeroCount) {
        pixelBuffer, pixelsInitialized in
        residualCoefficients = [Int32](unsafeUninitializedCapacity: nonzeroCount) {
          coefficientBuffer, coefficientsInitialized in
          for word in 0..<words {
            var remaining = nonzero[word]
            while remaining != 0 {
              let pixel = word * 64 + remaining.trailingZeroBitCount
              if emitted < nonzeroCount {
                pixelBuffer[emitted] = UInt32(truncatingIfNeeded: pixel)
                coefficientBuffer[emitted] = delta[pixel]
              }
              emitted += 1
              remaining &= remaining &- 1
            }
          }
          coefficientsInitialized = min(emitted, nonzeroCount)
        }
        pixelsInitialized = min(emitted, nonzeroCount)
      }
      guard emitted == nonzeroCount else { return nil }
      return PairedRuntimeTANSPolarPlan(
        detectorRows: 192, detectorColumns: 192,
        selectedFields: [], fieldCoefficients: [],
        residualPixels: residualPixels, residualCoefficients: residualCoefficients,
        usedIndex: false, leafPixelCount: leafPixels,
        layoutKind: layoutKind, validPixels: allValidPixels192)
    }

    // Leaf fields (leaf option minus root option), then nonzero root fields.
    var fieldsEmitted = 0
    var fieldCoefficients: [Int32] = []
    let selectedFields = [UInt32](unsafeUninitializedCapacity: selectedCount) {
      fieldBuffer, fieldsInitialized in
      fieldCoefficients = [Int32](unsafeUninitializedCapacity: selectedCount) {
        coefficientBuffer, coefficientsInitialized in
        for root in 0..<roots where rootHasLeafOptions[root] != 0 {
          let rootOption = Int32(rootOptions[root])
          for leaf in root * rootLeaves..<min((root + 1) * rootLeaves, leaves) {
            let coefficient = Int32(leafOptions[leaf]) - rootOption
            guard coefficient != 0 else { continue }
            if fieldsEmitted < selectedCount {
              fieldBuffer[fieldsEmitted] = UInt32(truncatingIfNeeded: leaf)
              coefficientBuffer[fieldsEmitted] = coefficient
            }
            fieldsEmitted += 1
          }
        }
        for root in 0..<roots where rootOptions[root] != 0 {
          if fieldsEmitted < selectedCount {
            fieldBuffer[fieldsEmitted] = UInt32(truncatingIfNeeded: leaves + root)
            coefficientBuffer[fieldsEmitted] = Int32(rootOptions[root])
          }
          fieldsEmitted += 1
        }
        coefficientsInitialized = min(fieldsEmitted, selectedCount)
      }
      fieldsInitialized = min(fieldsEmitted, selectedCount)
    }

    // Residuals in ascending pixel order. The subtraction stays checked so an
    // overflowing Int32 delta traps exactly where the legacy planner traps.
    var residualsEmitted = 0
    var residualCoefficients: [Int32] = []
    let residualPixels = [UInt32](unsafeUninitializedCapacity: residualCount) {
      pixelBuffer, pixelsInitialized in
      residualCoefficients = [Int32](unsafeUninitializedCapacity: residualCount) {
        coefficientBuffer, coefficientsInitialized in
        for word in 0..<words where residual[word] != 0 {
          let positiveLeaf = positiveLeaves[word]
          let negativeLeaf = negativeLeaves[word]
          let first = word * 64
          var remaining = residual[word]
          repeat {
            let offset = remaining.trailingZeroBitCount
            let option =
              Int32(truncatingIfNeeded: (positiveLeaf &>> UInt64(offset)) & 1)
              &- Int32(truncatingIfNeeded: (negativeLeaf &>> UInt64(offset)) & 1)
            if residualsEmitted < residualCount {
              pixelBuffer[residualsEmitted] = UInt32(truncatingIfNeeded: first + offset)
              coefficientBuffer[residualsEmitted] = delta[first + offset] - option
            }
            residualsEmitted += 1
            remaining &= remaining &- 1
          } while remaining != 0
        }
        coefficientsInitialized = min(residualsEmitted, residualCount)
      }
      pixelsInitialized = min(residualsEmitted, residualCount)
    }
    guard fieldsEmitted == selectedCount, residualsEmitted == residualCount else { return nil }
    return PairedRuntimeTANSPolarPlan(
      detectorRows: 192, detectorColumns: 192,
      selectedFields: selectedFields, fieldCoefficients: fieldCoefficients,
      residualPixels: residualPixels, residualCoefficients: residualCoefficients,
      usedIndex: true, leafPixelCount: leafPixels,
      layoutKind: layoutKind, validPixels: validPixels)
  }
}
