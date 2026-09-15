import Foundation

let detectorRows = 192
let detectorColumns = 192
let detectorPixels = detectorRows * detectorColumns
let badPixelIndices = [5319, 15050, 21710, 29965]
var validPixels = [UInt8](repeating: 1, count: detectorPixels)
for pixel in badPixelIndices {
  validPixels[pixel] = 0
}

func mask(_ row: Int, _ column: Int, _ inner: Int, _ outer: Int) -> [UInt8] {
  (0..<detectorPixels).map { pixel in
    let y = pixel / detectorColumns - 96 - row
    let x = pixel % detectorColumns - 96 - column
    let radiusSquared = y * y + x * x
    return radiusSquared >= inner * inner && radiusSquared <= outer * outer ? 1 : 0
  }
}

struct Transition {
  let name: String
  let previousName: String
  let targetName: String
  let previous: [UInt8]
  let target: [UInt8]
}

func transition(
  _ name: String, _ previousName: String, _ previous: [UInt8],
  _ targetName: String, _ target: [UInt8]
) -> Transition {
  Transition(
    name: name, previousName: previousName, targetName: targetName,
    previous: previous, target: target)
}

let adfBase = mask(0, 0, 48, 94)
let adfCenter1 = mask(0, 1, 48, 94)
let adfCenter8 = mask(5, 8, 48, 94)
let adfCenter20 = mask(12, 20, 48, 94)
let transitions = [
  transition("adf-center-1", "adf-base", adfBase, "adf-center-1", adfCenter1),
  transition("adf-center-8", "adf-base", adfBase, "adf-center-8", adfCenter8),
  transition(
    "adf-center-8-to-20", "adf-center-8", adfCenter8,
    "adf-center-20", adfCenter20),
  transition("adf-radius-1", "adf-base", adfBase, "adf-radius-1", mask(0, 0, 48, 95)),
  transition("adf-radius-8", "adf-base", adfBase, "adf-radius-8", mask(0, 0, 48, 102)),
  transition("adf-radius-20", "adf-base", adfBase, "adf-radius-20", mask(0, 0, 48, 114)),
] + (1...20).map { column in
  transition(
    "adf-drag-column-\(column)", "adf-drag-column-\(column - 1)",
    mask(0, column - 1, 48, 94), "adf-drag-column-\(column)",
    mask(0, column, 48, 94))
}

struct WidthRecord {
  let leafPixels: Int
  let fields: Int
  let residuals: Int
  let estimatedCost: Int
}

var records: [[String: Any]] = []
var totals: [Int: (fields: Int, residuals: Int, cost: Int, cases: Int)] = [:]
var maximumCosts: [Int: (name: String, cost: Int)] = [:]

for item in transitions {
  let previous = zip(item.previous, validPixels).map { $0 == 0 || $1 == 0 ? UInt8(0) : $0 }
  let target = zip(item.target, validPixels).map { $0 == 0 || $1 == 0 ? UInt8(0) : $0 }
  let delta = zip(previous, target).map { Int32($1) - Int32($0) }
  let changedPixels = delta.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
  var widths: [[String: Any]] = []

  for leafPixels in [16, 32] {
    let plan = PairedRuntimeTANSPolarPlan.make(
      delta: delta, validPixels: validPixels,
      detectorRows: detectorRows, detectorColumns: detectorColumns,
      leafPixels: leafPixels, layoutKind: "radial1")
    guard plan.reconstructedDelta() == delta else {
      fputs("FAIL: radial1/leaf\(leafPixels)/\(item.name) did not reconstruct exactly\n", stderr)
      exit(1)
    }
    let cost = plan.estimatedCost
    let prior = totals[leafPixels] ?? (0, 0, 0, 0)
    totals[leafPixels] = (
      prior.fields + plan.selectedFields.count,
      prior.residuals + plan.residualPixels.count,
      prior.cost + cost,
      prior.cases + 1)
    if cost > (maximumCosts[leafPixels]?.cost ?? -1) {
      maximumCosts[leafPixels] = (item.name, cost)
    }
    widths.append([
      "leaf_pixels": leafPixels,
      "fields": plan.selectedFields.count,
      "residuals": plan.residualPixels.count,
      "estimated_cost": cost,
      "exact_signed_reconstruction": true,
    ])
  }

  records.append([
    "transition": item.name,
    "previous_mask": item.previousName,
    "target_mask": item.targetName,
    "changed_pixels_after_validity_mask": changedPixels,
    "exact_signed_reconstruction": true,
    "widths": widths,
  ])
}

let summaries: [[String: Any]] = [16, 32].map { leafPixels in
  let total = totals[leafPixels]!
  let worst = maximumCosts[leafPixels]!
  let target = records.first { $0["transition"] as? String == "adf-center-8-to-20" }!
  let targetWidths = target["widths"] as! [[String: Any]]
  let targetWidth = targetWidths.first { $0["leaf_pixels"] as? Int == leafPixels }!
  return [
    "leaf_pixels": leafPixels,
    "transition_count": total.cases,
    "total_fields": total.fields,
    "total_residuals": total.residuals,
    "total_estimated_cost": total.cost,
    "worst_transition": worst.name,
    "worst_estimated_cost": worst.cost,
    "adf_center_8_to_20": targetWidth,
  ]
}

let result: [String: Any] = [
  "schema": "quantem-gpu-ans-radial1-leafwidth-cpu-screen/v1",
  "detector_shape": [detectorRows, detectorColumns],
  "layout_kind": "radial1",
  "leaf_widths": [16, 32],
  "validity": [
    "source": "seven-source maped-seven-tilts master HDF5 metadata",
    "source_count": 7,
    "bad_pixel_indices_common_to_all_sources": badPixelIndices,
    "valid_pixel_count": detectorPixels - badPixelIndices.count,
    "detector_mask_sha256": "33f5b1988e3f4360e9578a8855884e5bdcb2cbc2b6b25c0c830d65bec6c69b47",
  ],
  "transition_count": records.count,
  "exact_reconstruction_count": records.count * 2,
  "cost_definition": "selected field count + 4 * residual pixel count; operation proxy, not a timing model",
  "gpu_timing_performed": false,
  "summaries": summaries,
  "cases": records,
]

guard CommandLine.arguments.count == 2 else {
  fputs("usage: radial1-leafwidth-cpu-screen output.json\n", stderr)
  exit(2)
}
let output = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(
  at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
try data.write(to: output, options: .atomic)
print("wrote \(output.path); \(records.count * 2) exact plans")
