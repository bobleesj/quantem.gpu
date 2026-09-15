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

func census(layoutKind: String, _ transitions: [Transition]) -> [[String: Any]] {
  transitions.map { transition in
    let delta = zip(transition.previous, transition.target).map { Int32($1) - Int32($0) }
    let plan = PairedRuntimeTANSPolarPlan.make(
      delta: delta, validPixels: valid, detectorRows: rows, detectorColumns: columns,
      leafPixels: 16, layoutKind: layoutKind)
    let expected = delta
    let reconstructed = plan.reconstructedDelta()
    guard reconstructed == expected else {
      fputs("FAIL: \(layoutKind)/\(transition.name) did not reconstruct exactly\n", stderr)
      exit(1)
    }
    let changed = delta.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
    return [
      "transition": transition.name,
      "layout": layoutKind,
      "changed_pixels": changed,
      "fields": plan.selectedFields.count,
      "residuals": plan.residualPixels.count,
      "estimated_cost": plan.estimatedCost,
      "exact_reconstruction": true,
    ]
  }
}

let matrix = transitions()
let records = census(layoutKind: "radial1", matrix) + census(layoutKind: "radialhalf", matrix)
let result: [String: Any] = [
  "schema": "quantem-gpu-ans-layout-census/v1",
  "detector_shape": [rows, columns],
  "leaf_pixels": 16,
  "validity": "all-valid geometry-only; real-source masks are not represented",
  "transitions": records,
]
guard CommandLine.arguments.count == 2 else {
  fputs("usage: ans-layout-census output.json\n", stderr)
  exit(2)
}
let output = URL(fileURLWithPath: CommandLine.arguments[1])
try FileManager.default.createDirectory(
  at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
try data.write(to: output, options: .atomic)
print("wrote \(output.path); \(records.count) exact plans")
