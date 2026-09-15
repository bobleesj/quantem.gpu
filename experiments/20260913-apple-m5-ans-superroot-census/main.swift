import Foundation

let rows = 192
let columns = 192
let pixelCount = rows * columns
let leafPixels = 16
let leavesPerRoot = 16
let validPixels = [UInt8](repeating: 1, count: pixelCount)

func mask(_ rowOffset: Int, _ columnOffset: Int, _ inner: Int, _ outer: Int) -> [UInt8] {
  (0..<pixelCount).map { pixel in
    let row = pixel / columns - 96 - rowOffset
    let column = pixel % columns - 96 - columnOffset
    let radiusSquared = row * row + column * column
    return radiusSquared >= inner * inner && radiusSquared <= outer * outer ? 1 : 0
  }
}

struct Transition {
  let name: String
  let previous: [UInt8]
  let target: [UInt8]
}

func transitions() -> [Transition] {
  let bf = mask(0, 0, 0, 46)
  let abf = mask(0, 0, 24, 64)
  let adf = mask(0, 0, 48, 94)
  return [
    Transition(name: "bf-center-1", previous: bf, target: mask(0, 1, 0, 46)),
    Transition(name: "bf-center-8", previous: bf, target: mask(5, 8, 0, 46)),
    Transition(name: "bf-center-8-to-20", previous: mask(5, 8, 0, 46), target: mask(12, 20, 0, 46)),
    Transition(name: "bf-radius-1", previous: bf, target: mask(0, 0, 0, 47)),
    Transition(name: "bf-radius-8", previous: bf, target: mask(0, 0, 0, 54)),
    Transition(name: "bf-radius-20", previous: bf, target: mask(0, 0, 0, 66)),
    Transition(name: "abf-center-1", previous: abf, target: mask(0, 1, 24, 64)),
    Transition(name: "abf-center-8", previous: abf, target: mask(5, 8, 24, 64)),
    Transition(name: "abf-center-8-to-20", previous: mask(5, 8, 24, 64), target: mask(12, 20, 24, 64)),
    Transition(name: "abf-radius-1", previous: abf, target: mask(0, 0, 24, 65)),
    Transition(name: "abf-radius-8", previous: abf, target: mask(0, 0, 24, 72)),
    Transition(name: "abf-radius-20", previous: abf, target: mask(0, 0, 24, 84)),
    Transition(name: "adf-center-1", previous: adf, target: mask(0, 1, 48, 94)),
    Transition(name: "adf-center-8", previous: adf, target: mask(5, 8, 48, 94)),
    Transition(name: "adf-center-8-to-20", previous: mask(5, 8, 48, 94), target: mask(12, 20, 48, 94)),
    Transition(name: "adf-radius-1", previous: adf, target: mask(0, 0, 48, 95)),
    Transition(name: "adf-radius-8", previous: adf, target: mask(0, 0, 48, 102)),
    Transition(name: "adf-radius-20", previous: adf, target: mask(0, 0, 48, 114)),
  ]
}

struct Metrics {
  let fields: Int
  let residuals: Int
  let changedPixels: Int
  let proxyCost: Int
  let exact: Bool
}

func currentTwoLevel(delta: [Int32]) -> Metrics {
  let plan = PairedRuntimeTANSPolarPlan.make(
    delta: delta, validPixels: validPixels, detectorRows: rows, detectorColumns: columns,
    leafPixels: leafPixels, layoutKind: "radial1")
  return Metrics(
    fields: plan.selectedFields.count,
    residuals: plan.residualPixels.count,
    changedPixels: delta.filter { $0 != 0 }.count,
    proxyCost: plan.estimatedCost,
    exact: plan.reconstructedDelta() == delta)
}

/// Build an exact greedy hierarchy. Each level chooses the modal signed coefficient
/// within its own cells; residuals retain any detector-pixel disagreement.
func threeLevel(delta: [Int32], groupRoots: Int) -> Metrics {
  guard let layout = PairedRuntimeTANSPolarPlan.indexLayout(
    leafPixels: leafPixels, layoutKind: "radial1")
  else { fatalError("radial1 leaf16 layout unavailable") }

  let leafCount = layout.leaves
  let rootCount = layout.roots
  let superRootCount = (rootCount + groupRoots - 1) / groupRoots
  let options: [Int32] = [0, 1, -1]
  var orderedValues = [Int32](repeating: 0, count: layout.permutation.count)
  for ordinal in layout.permutation.indices {
    let pixel = Int(layout.permutation[ordinal])
    if pixel >= 0 { orderedValues[ordinal] = delta[pixel] }
  }

  func mode(_ values: [Int32]) -> Int32 {
    var best = options[0]
    var bestCount = -1
    for option in options {
      let count = values.reduce(0) { $0 + ($1 == option ? 1 : 0) }
      if count > bestCount {
        best = option
        bestCount = count
      }
    }
    return best
  }

  var leafBase = [Int32](repeating: 0, count: leafCount)
  for leaf in 0..<leafCount {
    let start = leaf * leafPixels
    leafBase[leaf] = mode(Array(orderedValues[start..<(start + leafPixels)]))
  }

  var rootBase = [Int32](repeating: 0, count: rootCount)
  for root in 0..<rootCount {
    let start = root * leavesPerRoot
    rootBase[root] = mode(Array(leafBase[start..<(start + leavesPerRoot)]))
  }

  var superRootBase = [Int32](repeating: 0, count: superRootCount)
  for superRoot in 0..<superRootCount {
    let start = superRoot * groupRoots
    let end = min(rootCount, start + groupRoots)
    superRootBase[superRoot] = mode(Array(rootBase[start..<end]))
  }

  let superRootFields = superRootBase
  let rootFields = rootBase.indices.map { root in
    rootBase[root] - superRootBase[root / groupRoots]
  }
  let leafFields = leafBase.indices.map { leaf in
    leafBase[leaf] - rootBase[leaf / leavesPerRoot]
  }

  var residual = [Int32](repeating: 0, count: pixelCount)
  for ordinal in layout.permutation.indices {
    let pixel = Int(layout.permutation[ordinal])
    if pixel >= 0 {
      let leaf = ordinal / leafPixels
      let root = leaf / leavesPerRoot
      let superRoot = root / groupRoots
      let reconstructedByFields = superRootFields[superRoot] + rootFields[root] + leafFields[leaf]
      residual[pixel] = delta[pixel] - reconstructedByFields
    }
  }

  var reconstructed = [Int32](repeating: 0, count: pixelCount)
  for ordinal in layout.permutation.indices {
    let pixel = Int(layout.permutation[ordinal])
    if pixel >= 0 {
      let leaf = ordinal / leafPixels
      let root = leaf / leavesPerRoot
      let superRoot = root / groupRoots
      reconstructed[pixel] = superRootFields[superRoot] + rootFields[root] + leafFields[leaf]
    }
  }
  for pixel in residual.indices { reconstructed[pixel] += residual[pixel] }

  let fieldCount = superRootFields.filter { $0 != 0 }.count
    + rootFields.filter { $0 != 0 }.count
    + leafFields.filter { $0 != 0 }.count
  let residualCount = residual.filter { $0 != 0 }.count
  return Metrics(
    fields: fieldCount,
    residuals: residualCount,
    changedPixels: delta.filter { $0 != 0 }.count,
    proxyCost: fieldCount + 4 * residualCount,
    exact: reconstructed == delta)
}

func dictionary(_ metrics: Metrics) -> [String: Any] {
  [
    "changed_pixels": metrics.changedPixels,
    "fields": metrics.fields,
    "residuals": metrics.residuals,
    "proxy_cost": metrics.proxyCost,
    "exact_reconstruction": metrics.exact,
  ]
}

guard CommandLine.arguments.count == 2 else {
  fputs("usage: ans-superroot-census output.json\n", stderr)
  exit(2)
}
let layout = PairedRuntimeTANSPolarPlan.indexLayout(leafPixels: leafPixels, layoutKind: "radial1")!
let records: [[String: Any]] = transitions().map { transition in
  let delta = zip(transition.previous, transition.target).map { Int32($1) - Int32($0) }
  let baseline = currentTwoLevel(delta: delta)
  let group4 = threeLevel(delta: delta, groupRoots: 4)
  let group16 = threeLevel(delta: delta, groupRoots: 16)
  guard baseline.exact, group4.exact, group16.exact else {
    fputs("FAIL: exact reconstruction failed for \(transition.name)\n", stderr)
    exit(1)
  }
  return [
    "transition": transition.name,
    "two_level_leaf16_radial1": dictionary(baseline),
    "three_level_superroot4_roots": dictionary(group4),
    "three_level_superroot16_roots": dictionary(group16),
  ]
}

func aggregate(_ key: String) -> [String: Any] {
  let rows = records.compactMap { $0[key] as? [String: Any] }
  let totalFields = rows.reduce(0) { $0 + ($1["fields"] as? Int ?? 0) }
  let totalResiduals = rows.reduce(0) { $0 + ($1["residuals"] as? Int ?? 0) }
  let totalCost = rows.reduce(0) { $0 + ($1["proxy_cost"] as? Int ?? 0) }
  let worstCost = rows.map { $0["proxy_cost"] as? Int ?? 0 }.max() ?? 0
  let worstResiduals = rows.map { $0["residuals"] as? Int ?? 0 }.max() ?? 0
  return [
    "total_fields": totalFields,
    "total_residuals": totalResiduals,
    "total_proxy_cost": totalCost,
    "worst_proxy_cost": worstCost,
    "worst_residuals": worstResiduals,
  ]
}

let targetIndex = records.firstIndex { ($0["transition"] as? String) == "adf-center-8-to-20" }!
let summary: [String: Any] = [
  "two_level_leaf16_radial1": aggregate("two_level_leaf16_radial1"),
  "three_level_superroot4_roots": aggregate("three_level_superroot4_roots"),
  "three_level_superroot16_roots": aggregate("three_level_superroot16_roots"),
  "target_adf_center_8_to_20": records[targetIndex],
  "all_54_plans_exact": true,
  "scope": "all-valid geometry only; not actual source validity, image counts, GPU timing, or measured GPU memory",
  "added_index_bytes_upper_bound": [
    "persistent_mapping": "0 bytes when superroot = root / groupRoots is computed arithmetically",
    "optional_root_to_superroot_map_replicated_for_seven_sources": 4032,
    "extra_selected_superroot_id_and_coefficient_payload_for_seven_sources_group4": 2016,
    "extra_selected_superroot_id_and_coefficient_payload_for_seven_sources_group16": 504,
    "current_radial1_leaf16_permutation_bytes": layout.permutation.count * MemoryLayout<Int32>.stride,
  ],
]
let result: [String: Any] = [
  "schema": "quantem-gpu-ans-superroot-census/v1",
  "detector_shape": [rows, columns],
  "leaf_pixels": leafPixels,
  "leaves_per_root": leavesPerRoot,
  "superroot_group_sizes_in_roots": [4, 16],
  "validity": "all-valid geometry-only; actual per-source validity masks are not represented",
  "transitions": records,
  "summary": summary,
]
let output = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(
  at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
try data.write(to: output, options: .atomic)
print("wrote \(output.path); \(records.count * 3) exact plans")
