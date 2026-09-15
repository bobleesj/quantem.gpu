import Foundation

let detectorRows = 192
let detectorColumns = 192
let pixelCount = detectorRows * detectorColumns
let jointEnabled = ProcessInfo.processInfo.environment["QGPU_PAIRED_RUNTIME_JOINT_PLAN"] == "1"

struct SeededGenerator {
  private var state: UInt64

  init(seed: UInt64) {
    state = seed == 0 ? 0x9e37_79b9_7f4a_7c15 : seed
  }

  mutating func next() -> UInt64 {
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return state
  }
}

func require(_ condition: @autoclosure () -> Bool, _ message: String) {
  guard condition() else {
    fputs("FAIL: \(message)\n", stderr)
    exit(1)
  }
}

func signature(_ values: [Int32]) -> String {
  var hash: UInt64 = 14_695_981_039_346_656_037
  for value in values {
    var bits = UInt32(bitPattern: value).littleEndian
    withUnsafeBytes(of: &bits) { bytes in
      for byte in bytes {
        hash = (hash ^ UInt64(byte)) &* 1_099_511_628_211
      }
    }
  }
  return String(hash, radix: 16)
}

func verify(
  _ label: String, delta: [Int32], valid: [UInt8], rows: Int = detectorRows,
  columns: Int = detectorColumns, leafPixels: Int, layoutKind: String = "polar",
  expectedIndexed: Bool? = nil
) -> PairedRuntimeTANSPolarPlan {
  let plan = PairedRuntimeTANSPolarPlan.make(
    delta: delta, validPixels: valid, detectorRows: rows, detectorColumns: columns,
    leafPixels: leafPixels, layoutKind: layoutKind)
  let actual = plan.reconstructedDelta()
  let expected = delta.indices.map { valid[$0] == 0 ? Int32(0) : delta[$0] }
  require(actual == expected, "\(label) failed exact delta reconstruction")
  if let expectedIndexed {
    require(plan.usedIndex == expectedIndexed, "\(label) selected the wrong direct/index path")
  }
  print("PARITY \(label) \(signature(actual))")
  print(
    "PLAN \(label) indexed=\(plan.usedIndex) fields=\(plan.selectedFields.count) "
      + "residuals=\(plan.residualPixels.count) cost=\(plan.estimatedCost)")
  return plan
}

func scalarProduct(
  _ plan: PairedRuntimeTANSPolarPlan, counts: [UInt16], valid: [UInt8],
  layout: PairedRuntimeTANSPolarPlan.IndexLayout
) -> Int64 {
  var total: Int64 = 0
  if plan.usedIndex {
    var fields = [Int32](repeating: 0, count: layout.leaves + layout.roots)
    for (field, coefficient) in zip(plan.selectedFields, plan.fieldCoefficients) {
      fields[Int(field)] = coefficient
    }
    for field in plan.selectedFields {
      let index = Int(field)
      let coefficient = fields[index]
      let firstOrdinal: Int
      let endOrdinal: Int
      if index < layout.leaves {
        firstOrdinal = index * plan.leafPixelCount
        endOrdinal = min(firstOrdinal + plan.leafPixelCount, layout.permutation.count)
      } else {
        firstOrdinal = (index - layout.leaves) * 16 * plan.leafPixelCount
        endOrdinal = min(firstOrdinal + 16 * plan.leafPixelCount, layout.permutation.count)
      }
      for ordinal in firstOrdinal..<endOrdinal {
        let pixel = Int(layout.permutation[ordinal])
        if pixel >= 0 && valid[pixel] != 0 {
          total += Int64(coefficient) * Int64(counts[pixel])
        }
      }
    }
  }
  for (pixel, coefficient) in zip(plan.residualPixels, plan.residualCoefficients) {
    total += Int64(coefficient) * Int64(counts[Int(pixel)])
  }
  return total
}

func verifyScalarProducts(
  _ label: String, plan: PairedRuntimeTANSPolarPlan, delta: [Int32], valid: [UInt8],
  layout: PairedRuntimeTANSPolarPlan.IndexLayout, seed: UInt64
) {
  var generator = SeededGenerator(seed: seed)
  for trial in 0..<4 {
    let counts = (0..<pixelCount).map { _ in UInt16(truncatingIfNeeded: generator.next()) }
    let actual = scalarProduct(plan, counts: counts, valid: valid, layout: layout)
    let expected = delta.indices.reduce(Int64(0)) { sum, pixel in
      guard valid[pixel] != 0 else { return sum }
      return sum + Int64(delta[pixel]) * Int64(counts[pixel])
    }
    require(actual == expected, "\(label) full-u16 scalar product trial \(trial) differs")
  }
}

for leafPixels in [16, 32, 64] {
  let layoutKind = leafPixels == 16 ? "radial1" : "polar"
  guard let layout = PairedRuntimeTANSPolarPlan.indexLayout(
    leafPixels: leafPixels, layoutKind: layoutKind)
  else {
    fputs("FAIL: missing \(layoutKind) layout for leaf width \(leafPixels)\n", stderr)
    exit(1)
  }

  // Fifteen all-positive leaves make the parent positive. The sixteenth leaf
  // ties zero and +1, so the joint plan can drop its leaf correction without
  // changing residual coverage.
  var tieDelta = [Int32](repeating: 0, count: pixelCount)
  for root in 0..<layout.roots {
    for offset in 0..<16 {
      let leaf = root * 16 + offset
      let first = leaf * leafPixels
      for withinLeaf in 0..<leafPixels {
        let ordinal = first + withinLeaf
        let pixel = Int(layout.permutation[ordinal])
        tieDelta[pixel] = offset == 15 && withinLeaf < leafPixels / 2 ? 0 : 1
      }
    }
  }
  let allValid = [UInt8](repeating: 1, count: pixelCount)
  let tiePlan = verify(
    "tie-\(leafPixels)", delta: tieDelta, valid: allValid,
    leafPixels: leafPixels, layoutKind: layoutKind, expectedIndexed: true)
  let expectedTieResiduals = layout.roots * (leafPixels / 2)
  require(
    tiePlan.residualPixels.count == expectedTieResiduals,
    "tie-\(leafPixels) changed residual coverage")
  let expectedTieFields = layout.roots * (jointEnabled ? 1 : 2)
  require(
    tiePlan.selectedFields.count == expectedTieFields,
    "tie-\(leafPixels) did not match the expected tie-aware field count")

  // Exercise arbitrary signed deltas, full uint16 magnitudes, and validity
  // exclusions. The plan must preserve every valid value and suppress only
  // pixels already excluded by the stored source-validity map.
  let values: [Int32] = [-65_535, -3, -1, 0, 1, 2, 65_535]
  var mixedDelta = [Int32](repeating: 0, count: pixelCount)
  var mixedValid = [UInt8](repeating: 1, count: pixelCount)
  for pixel in 0..<pixelCount {
    mixedDelta[pixel] = values[(pixel * 17 + leafPixels) % values.count]
    if (pixel * 13 + leafPixels) % 29 == 0 { mixedValid[pixel] = 0 }
  }
  _ = verify(
    "mixed-\(leafPixels)", delta: mixedDelta, valid: mixedValid,
    leafPixels: leafPixels, layoutKind: layoutKind)

  // A single residual pixel ties indexed and direct proxy costs; direct must
  // continue to win because the existing fallback uses a strict `<` test.
  var onePixel = [Int32](repeating: 0, count: pixelCount)
  onePixel[0] = -1
  _ = verify(
    "direct-tie-\(leafPixels)", delta: onePixel, valid: allValid,
    leafPixels: leafPixels, layoutKind: layoutKind, expectedIndexed: false)

  var invalidOnly = [Int32](repeating: 0, count: pixelCount)
  var invalidOnlyMask = allValid
  invalidOnly[0] = 65_535
  invalidOnlyMask[0] = 0
  _ = verify(
    "invalid-only-\(leafPixels)", delta: invalidOnly, valid: invalidOnlyMask,
    leafPixels: leafPixels, layoutKind: layoutKind, expectedIndexed: false)

  // The root can be +1 while a leaf's local target is -1, requiring a -2
  // emitted leaf coefficient. This path must stay exact.
  var signedTwoDelta = [Int32](repeating: 0, count: pixelCount)
  for root in 0..<layout.roots {
    for offset in 0..<16 {
      let leaf = root * 16 + offset
      let first = leaf * leafPixels
      for withinLeaf in 0..<leafPixels {
        let pixel = Int(layout.permutation[first + withinLeaf])
        signedTwoDelta[pixel] = offset == 15 ? -1 : 1
      }
    }
  }
  let signedTwoPlan = verify(
    "signed-two-\(leafPixels)", delta: signedTwoDelta, valid: allValid,
    leafPixels: leafPixels, layoutKind: layoutKind, expectedIndexed: true)
  if jointEnabled {
    require(
      signedTwoPlan.fieldCoefficients.contains(-2),
      "signed-two-\(leafPixels) did not retain a -2 leaf coefficient")
  }
  verifyScalarProducts(
    "signed-two-\(leafPixels)", plan: signedTwoPlan, delta: signedTwoDelta,
    valid: allValid, layout: layout, seed: UInt64(0x51_6e_65_64 + leafPixels))

  // Seeded prior/target binary count masks exercise realistic exact deltas,
  // including invalid-pixel exclusions and small perturbations around coherent
  // leaf modes. Identical seeds in both planner processes enable cost comparison.
  var randomCostTotal = 0
  var randomIndexedCases = 0
  var generator = SeededGenerator(seed: UInt64(0x504c_414e_0000 + leafPixels))
  for caseIndex in 0..<100 {
    var prior = [Int32](repeating: 0, count: pixelCount)
    var target = [Int32](repeating: 0, count: pixelCount)
    var valid = [UInt8](repeating: 1, count: pixelCount)
    for leaf in 0..<layout.leaves {
      let mode = Int(generator.next() % 3) - 1
      for withinLeaf in 0..<leafPixels {
        let pixel = Int(layout.permutation[leaf * leafPixels + withinLeaf])
        switch mode {
        case -1:
          prior[pixel] = 1
          target[pixel] = 0
        case 1:
          prior[pixel] = 0
          target[pixel] = 1
        default:
          let bit = Int32(generator.next() & 1)
          prior[pixel] = bit
          target[pixel] = bit
        }
        if generator.next() % 23 == 0 { target[pixel] = 1 - target[pixel] }
        if generator.next() % 31 == 0 { valid[pixel] = 0 }
      }
    }
    let delta = target.indices.map { target[$0] - prior[$0] }
    let label = "random-l\(leafPixels)-s\(caseIndex)"
    let plan = verify(
      label, delta: delta, valid: valid, leafPixels: leafPixels,
      layoutKind: layoutKind)
    randomCostTotal += plan.estimatedCost
    if plan.usedIndex { randomIndexedCases += 1 }
    print("COST \(label) \(plan.estimatedCost)")
  }
  print(
    "RANDOM_TOTAL leaf=\(leafPixels) cost=\(randomCostTotal) "
      + "indexed_cases=\(randomIndexedCases)")
}

let tinyDelta: [Int32] = [1, -1, 65_535, -65_535, 0, 2]
let tinyValid = [UInt8](repeating: 1, count: tinyDelta.count)
_ = verify(
  "unsupported-geometry", delta: tinyDelta, valid: tinyValid,
  rows: 2, columns: 3, leafPixels: 64, expectedIndexed: false)

print("PASS planner parity joint=\(jointEnabled)")
