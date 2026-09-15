import Foundation

private enum ProbeError: Error { case failed(String) }
private func check(_ value: @autoclosure () -> Bool, _ message: String) throws {
  if !value() { throw ProbeError.failed(message) }
}

private struct Choice {
  let plan: PairedRuntimeTANSPolarPlan
  let startsFromZero: Bool
}

private func choose(
  current: [UInt8], target: [UInt8], validity: [UInt8], leaf: Int, layout: String
) -> Choice? {
  var delta = [Int32](repeating: 0, count: target.count)
  var changed = false
  for pixel in target.indices where current[pixel] != target[pixel] {
    delta[pixel] = target[pixel] == 1 ? 1 : -1
    changed = true
  }
  guard changed else { return nil }
  let previous = PairedRuntimeTANSPolarPlan.make(
    delta: delta, validPixels: validity, detectorRows: 192, detectorColumns: 192,
    leafPixels: leaf, layoutKind: layout)
  let absolute = target.map(Int32.init)
  let zero = PairedRuntimeTANSPolarPlan.make(
    delta: absolute, validPixels: validity, detectorRows: 192, detectorColumns: 192,
    leafPixels: leaf, layoutKind: layout)
  return zero.estimatedCost + 1 < previous.estimatedCost
    ? Choice(plan: zero, startsFromZero: true)
    : Choice(plan: previous, startsFromZero: false)
}

@main enum PolarChooseBaseProbe {
  static func main() throws {
    let count = 192 * 192
    var validity = [UInt8](repeating: 1, count: count)
    for pixel in stride(from: 17, to: count, by: 313) { validity[pixel] = 0 }
    let zero = [UInt8](repeating: 0, count: count)
    var one = [UInt8](repeating: 1, count: count)
    for pixel in validity.indices where validity[pixel] == 0 { one[pixel] = 0 }
    var scattered = zero
    for pixel in scattered.indices where validity[pixel] != 0 && pixel % 3 != 0 {
      scattered[pixel] = 1
    }
    var compact = zero
    for row in 70..<122 { for column in 61..<131 { compact[row * 192 + column] = 1 } }
    for pixel in validity.indices where validity[pixel] == 0 { compact[pixel] = 0 }

    let cases = [(one, zero), (scattered, one), (one, compact), (compact, scattered),
      (zero, compact)]
    var sawZeroBase = false
    var sawPreviousBase = false
    var sawEmptySelected = false
    for layout in ["polar", "radial1", "radialhalf"] {
      for leaf in [16, 32, 64] {
        for (current, target) in cases {
          guard let choice = choose(
            current: current, target: target, validity: validity, leaf: leaf, layout: layout)
          else { throw ProbeError.failed("differing masks produced no choice") }
          let reconstructed = choice.plan.reconstructedDelta()
          let expected = choice.startsFromZero
            ? target.map(Int32.init)
            : zip(current, target).map { Int32($1) - Int32($0) }
          try check(reconstructed == expected,
            "chosen plan changed the exact delta for \(layout), leaf \(leaf)")
          sawZeroBase = sawZeroBase || choice.startsFromZero
          sawPreviousBase = sawPreviousBase || !choice.startsFromZero
          sawEmptySelected = sawEmptySelected
            || (choice.plan.selectedFields.isEmpty && choice.plan.residualPixels.isEmpty)
        }
      }
    }
    try check(choose(current: zero, target: zero, validity: validity,
      leaf: 16, layout: "polar") == nil, "unchanged zero mask should select no work")
    try check(sawZeroBase, "fixtures did not choose the zero base")
    try check(sawPreviousBase, "fixtures did not retain the previous base")
    try check(sawEmptySelected, "fixtures did not cover an empty selected plan")
    print("polar choose-base CPU parity passed: 45 differing cases plus unchanged zero")
  }
}
