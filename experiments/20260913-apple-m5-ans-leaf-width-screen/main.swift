import Foundation

let rows = 192
let columns = 192
let pixels = rows * columns
let valid = [UInt8](repeating: 1, count: pixels)

func mask(_ rowOffset: Int, _ columnOffset: Int, _ inner: Int, _ outer: Int) -> [UInt8] {
  (0..<pixels).map { pixel in
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

struct CaseResult {
  let transition: String
  let leafPixels: Int
  let changedPixels: Int
  let fields: Int
  let residuals: Int
  let estimatedCost: Int
  let usedIndex: Bool

  var json: [String: Any] {
    [
      "transition": transition,
      "leaf_pixels": leafPixels,
      "changed_pixels": changedPixels,
      "fields": fields,
      "residuals": residuals,
      "estimated_cost": estimatedCost,
      "used_index": usedIndex,
      "exact_reconstruction": true,
    ]
  }
}

let matrix = transitions()
var cases: [CaseResult] = []
for leafPixels in [16, 32, 64] {
  for transition in matrix {
    let delta = zip(transition.previous, transition.target).map { Int32($1) - Int32($0) }
    let plan = PairedRuntimeTANSPolarPlan.make(
      delta: delta, validPixels: valid, detectorRows: rows, detectorColumns: columns,
      leafPixels: leafPixels, layoutKind: "polar")
    guard plan.reconstructedDelta() == delta else {
      fputs("FAIL: leafPixels=\(leafPixels)/\(transition.name) did not reconstruct exactly\n", stderr)
      exit(1)
    }
    let changed = delta.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
    cases.append(CaseResult(
      transition: transition.name, leafPixels: leafPixels, changedPixels: changed,
      fields: plan.selectedFields.count, residuals: plan.residualPixels.count,
      estimatedCost: plan.estimatedCost, usedIndex: plan.usedIndex))
  }
}

var summaries: [[String: Any]] = []
for leafPixels in [16, 32, 64] {
  let widthCases = cases.filter { $0.leafPixels == leafPixels }
  guard let worst = widthCases.max(by: { $0.estimatedCost < $1.estimatedCost }),
    let adfTarget = widthCases.first(where: { $0.transition == "adf-center-8-to-20" })
  else { fatalError("Expected 18 cases including the large ADF target") }
  summaries.append([
    "leaf_pixels": leafPixels,
    "total_fields": widthCases.reduce(0) { $0 + $1.fields },
    "total_residuals": widthCases.reduce(0) { $0 + $1.residuals },
    "total_estimated_cost": widthCases.reduce(0) { $0 + $1.estimatedCost },
    "worst_case": [
      "transition": worst.transition, "fields": worst.fields,
      "residuals": worst.residuals, "estimated_cost": worst.estimatedCost,
    ],
    "adf_center_8_to_20": [
      "changed_pixels": adfTarget.changedPixels, "fields": adfTarget.fields,
      "residuals": adfTarget.residuals, "estimated_cost": adfTarget.estimatedCost,
    ],
  ])
}

let result: [String: Any] = [
  "schema": "quantem-gpu-ans-leaf-width-screen/v1",
  "detector_shape": [rows, columns],
  "layout_kind": "polar",
  "leaf_widths": [16, 32, 64],
  "validity": "all-valid geometry-only; real-source masks are not represented",
  "transition_count": matrix.count,
  "exact_reconstruction_count": cases.count,
  "cases": cases.map(\.json),
  "summaries": summaries,
  "cost_definition": "PairedRuntimeTANSPolarPlan.estimatedCost = selected field count + 4 * residual pixel count; this is an operation proxy, not measured time",
  "gpu_timing_performed": false,
]
guard CommandLine.arguments.count == 2 else {
  fputs("usage: ans-leaf-width-screen output.json\n", stderr)
  exit(2)
}
let output = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(
  at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
try data.write(to: output, options: .atomic)
print("wrote \(output.path); \(cases.count) exact plans")
