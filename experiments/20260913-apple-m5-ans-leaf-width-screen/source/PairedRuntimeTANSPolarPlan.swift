import Foundation

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
  private static let radialHalfLayouts = [16, 32, 64].reduce(into: [Int: IndexLayout]()) {
    $0[$1] = makeLayout(
      detectorRows: 192, detectorColumns: 192, leafPixels: $1, layoutKind: "radialhalf")
  }

  /// Return cached field membership for one supported exact leaf width.
  static func indexLayout(leafPixels: Int, layoutKind: String = "polar") -> IndexLayout? {
    if layoutKind == "radial1" { return radial1Layouts[leafPixels] }
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
  static func make(
    delta: [Int32], validPixels: [UInt8], detectorRows: Int, detectorColumns: Int,
    leafPixels requestedLeafPixels: Int = 64, layoutKind: String = "polar"
  ) -> PairedRuntimeTANSPolarPlan {
    let environment = ProcessInfo.processInfo.environment
    let profile = environment["QGPU_PAIRED_RUNTIME_PROFILE"] == "1"
    let jointPlan = environment["QGPU_PAIRED_RUNTIME_JOINT_PLAN"] == "1"
    guard environment["QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN"] == "1" else {
      let buildStarted = profile ? CFAbsoluteTimeGetCurrent() : 0
      let plan = makeUncached(
        delta: delta, validPixels: validPixels,
        detectorRows: detectorRows, detectorColumns: detectorColumns,
        leafPixels: requestedLeafPixels, layoutKind: layoutKind,
        jointPlan: jointPlan)
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
      jointPlan: jointPlan)
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
      "shared_cache_enabled": ProcessInfo.processInfo.environment[
        "QGPU_PAIRED_RUNTIME_SHARED_POLAR_PLAN"] == "1" ? 1 : 0,
    ]
  }

  private static func makeUncached(
    delta: [Int32], validPixels: [UInt8], detectorRows: Int, detectorColumns: Int,
    leafPixels requestedLeafPixels: Int, layoutKind: String, jointPlan: Bool
  ) -> PairedRuntimeTANSPolarPlan {
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
    guard let layout = indexLayout(
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
            if pixel < 0 || validPixels[pixel] != 0 {
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
        where Int(pixel) < result.count
      {
        result[Int(pixel)] += coefficient
      }
      return result
    }
    guard let layout = Self.indexLayout(
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
            radialBand: layoutKind == "radial1" ? Int(floor(radius))
              : layoutKind == "radialhalf" ? Int(floor(radius * 2))
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
