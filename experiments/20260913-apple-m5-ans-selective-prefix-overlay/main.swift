import CryptoKit
import Foundation

private let detectorRows = 192
private let detectorColumns = 192
private let detectorPixels = detectorRows * detectorColumns
private let sourceCount = 7
private let leafPixels = 16
private let rootLeaves = 16
private let prefixWidth = 4
private let prefixesPerLeaf = 3
private let trainingLeafBudget = 25
private let badPixelIndices = [5319, 15050, 21710, 29965]
private let validPixels: [UInt8] = {
  var values = [UInt8](repeating: 1, count: detectorPixels)
  for pixel in badPixelIndices { values[pixel] = 0 }
  return values
}()

private struct MaskDefinition {
  let name: String
  let row: Int
  let column: Int
  let inner: Int
  let outer: Int
}

private struct Transition {
  let name: String
  let previous: String
  let target: String
  let isHeldOut: Bool
}

private struct BasePlan {
  let fields: [Int32]
  let residual: [Int32]
  let fieldCount: Int
  let residualCount: Int
  let estimatedCost: Int
}

private struct PrefixChoice {
  let segmentCorrections: [Int32]
  let prefixCoefficients: [Int32]
  let residualCount: Int
  let queryFieldCount: Int
  let estimatedCost: Int
}

private struct TransitionEvaluation {
  let definition: Transition
  let delta: [Int32]
  let base: BasePlan
  let leafChoices: [PrefixChoice]
}

private func require(_ condition: @autoclosure () -> Bool, _ message: String) {
  guard condition() else {
    fputs("FAIL: \(message)\n", stderr)
    exit(1)
  }
}

private func mask(_ definition: MaskDefinition) -> [UInt8] {
  (0..<detectorPixels).map { pixel in
    let row = pixel / detectorColumns - 96 - definition.row
    let column = pixel % detectorColumns - 96 - definition.column
    let radiusSquared = row * row + column * column
    return radiusSquared >= definition.inner * definition.inner
      && radiusSquared <= definition.outer * definition.outer ? 1 : 0
  }
}

private func maskDefinitions() -> [MaskDefinition] {
  [
    MaskDefinition(name: "bf-base", row: 0, column: 0, inner: 0, outer: 46),
    MaskDefinition(name: "bf-center-1", row: 0, column: 1, inner: 0, outer: 46),
    MaskDefinition(name: "bf-center-8", row: 5, column: 8, inner: 0, outer: 46),
    MaskDefinition(name: "bf-center-20", row: 12, column: 20, inner: 0, outer: 46),
    MaskDefinition(name: "bf-radius-1", row: 0, column: 0, inner: 0, outer: 47),
    MaskDefinition(name: "bf-radius-8", row: 0, column: 0, inner: 0, outer: 54),
    MaskDefinition(name: "bf-radius-20", row: 0, column: 0, inner: 0, outer: 66),
    MaskDefinition(name: "abf-base", row: 0, column: 0, inner: 24, outer: 64),
    MaskDefinition(name: "abf-center-1", row: 0, column: 1, inner: 24, outer: 64),
    MaskDefinition(name: "abf-center-8", row: 5, column: 8, inner: 24, outer: 64),
    MaskDefinition(name: "abf-radius-1", row: 0, column: 0, inner: 24, outer: 65),
    MaskDefinition(name: "abf-radius-8", row: 0, column: 0, inner: 24, outer: 72),
    MaskDefinition(name: "abf-radius-20", row: 0, column: 0, inner: 24, outer: 84),
    MaskDefinition(name: "adf-base", row: 0, column: 0, inner: 48, outer: 94),
    MaskDefinition(name: "adf-center-1", row: 0, column: 1, inner: 48, outer: 94),
    MaskDefinition(name: "adf-center-8", row: 5, column: 8, inner: 48, outer: 94),
    MaskDefinition(name: "adf-center-20", row: 12, column: 20, inner: 48, outer: 94),
    MaskDefinition(name: "adf-radius-1", row: 0, column: 0, inner: 48, outer: 95),
    MaskDefinition(name: "adf-radius-8", row: 0, column: 0, inner: 48, outer: 102),
    MaskDefinition(name: "adf-radius-20", row: 0, column: 0, inner: 48, outer: 114),
  ]
}

private func transitions() -> [Transition] {
  [
    Transition(name: "bf-center-1", previous: "bf-base", target: "bf-center-1", isHeldOut: false),
    Transition(name: "bf-center-8", previous: "bf-base", target: "bf-center-8", isHeldOut: false),
    Transition(name: "bf-center-8-to-20", previous: "bf-center-8", target: "bf-center-20", isHeldOut: false),
    Transition(name: "bf-radius-1", previous: "bf-base", target: "bf-radius-1", isHeldOut: false),
    Transition(name: "bf-radius-8", previous: "bf-base", target: "bf-radius-8", isHeldOut: false),
    Transition(name: "bf-radius-20", previous: "bf-base", target: "bf-radius-20", isHeldOut: false),
    Transition(name: "abf-center-1", previous: "abf-base", target: "abf-center-1", isHeldOut: false),
    Transition(name: "abf-center-8", previous: "abf-base", target: "abf-center-8", isHeldOut: false),
    Transition(name: "abf-radius-1", previous: "abf-base", target: "abf-radius-1", isHeldOut: false),
    Transition(name: "abf-radius-8", previous: "abf-base", target: "abf-radius-8", isHeldOut: false),
    Transition(name: "abf-radius-20", previous: "abf-base", target: "abf-radius-20", isHeldOut: false),
    Transition(name: "adf-center-1", previous: "adf-base", target: "adf-center-1", isHeldOut: false),
    Transition(name: "adf-center-8", previous: "adf-base", target: "adf-center-8", isHeldOut: false),
    Transition(
      name: "adf-center-8-to-20", previous: "adf-center-8", target: "adf-center-20",
      isHeldOut: true),
    Transition(name: "adf-radius-1", previous: "adf-base", target: "adf-radius-1", isHeldOut: false),
    Transition(name: "adf-radius-8", previous: "adf-base", target: "adf-radius-8", isHeldOut: false),
    Transition(name: "adf-radius-20", previous: "adf-base", target: "adf-radius-20", isHeldOut: false),
  ]
}

private func readFrozenMaskNames(_ url: URL) throws -> (Set<String>, Int) {
  let text = try String(contentsOf: url, encoding: .utf8)
  var names = Set<String>()
  var keys = Set<String>()
  var parityRows = 0
  for line in text.split(separator: "\n") {
    guard let data = String(line).data(using: .utf8),
      let raw = try JSONSerialization.jsonObject(with: data) as? [String: Any]
    else { continue }
    guard raw["event"] as? String == "ans_opt_independent_parity" else { continue }
    guard let name = raw["mask"] as? String,
      let source = raw["source"] as? Int,
      let valueCount = raw["values"] as? Int,
      raw["status"] as? String == "passed"
    else { throw NSError(domain: "malformed frozen parity row", code: 1) }
    require(valueCount == 512 * 512, "frozen map \(name) is not full scan size")
    require((0..<sourceCount).contains(source), "frozen source id is out of range")
    let key = "\(name)/\(source)"
    require(keys.insert(key).inserted, "duplicate frozen mask/source key \(key)")
    names.insert(name)
    parityRows += 1
  }
  require(parityRows == 20 * sourceCount, "expected 20 masks × 7 frozen maps")
  return (names, parityRows)
}

private func basePlan(for delta: [Int32], layout: PairedRuntimeTANSPolarPlan.IndexLayout)
  -> BasePlan
{
  let plan = PairedRuntimeTANSPolarPlan.make(
    delta: delta, validPixels: validPixels,
    detectorRows: detectorRows, detectorColumns: detectorColumns,
    leafPixels: leafPixels, layoutKind: "radial1")
  let expected = plan.reconstructedDelta()
  require(expected == delta, "existing planner failed full-map baseline parity")

  var fieldCoefficients = [Int32](repeating: 0, count: layout.leaves + layout.roots)
  for (field, coefficient) in zip(plan.selectedFields, plan.fieldCoefficients) {
    fieldCoefficients[Int(field)] = coefficient
  }
  var fields = [Int32](repeating: 0, count: detectorPixels)
  for ordinal in layout.permutation.indices {
    let pixel = Int(layout.permutation[ordinal])
    if pixel >= 0 && validPixels[pixel] != 0 {
      let leaf = ordinal / leafPixels
      let root = leaf / rootLeaves
      fields[pixel] = fieldCoefficients[leaf] + fieldCoefficients[layout.leaves + root]
    }
  }
  let residual = delta.indices.map { delta[$0] - fields[$0] }
  let residualCount = residual.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
  require(residualCount == plan.residualPixels.count,
          "baseline residual map disagrees with existing planner's residual list")
  for (pixel, coefficient) in zip(plan.residualPixels, plan.residualCoefficients) {
    require(residual[Int(pixel)] == coefficient,
            "baseline residual coefficient differs at detector pixel \(pixel)")
  }
  require(fields.indices.allSatisfy({ fields[$0] + residual[$0] == delta[$0] }),
          "baseline full-map decomposition is not exact")
  return BasePlan(
    fields: fields, residual: residual,
    fieldCount: plan.selectedFields.count, residualCount: residualCount,
    estimatedCost: plan.estimatedCost)
}

private func bestPrefixChoice(
  residual: [Int32], leaf: Int, layout: PairedRuntimeTANSPolarPlan.IndexLayout
) -> PrefixChoice {
  let start = leaf * leafPixels
  let ranges = (0..<prefixesPerLeaf).map { segment in
    (start + segment * prefixWidth)..<(start + (segment + 1) * prefixWidth)
  }
  var lastSegmentResiduals = 0
  for ordinal in (start + prefixesPerLeaf * prefixWidth)..<(start + leafPixels) {
    let pixel = Int(layout.permutation[ordinal])
    if validPixels[pixel] != 0 { lastSegmentResiduals += residual[pixel] == 0 ? 0 : 1 }
  }
  let candidates = ranges.map { range -> [Int32] in
    var values = Set<Int32>([0])
    for ordinal in range {
      let pixel = Int(layout.permutation[ordinal])
      if validPixels[pixel] != 0 { values.insert(residual[pixel]) }
    }
    return values.sorted()
  }

  var bestCorrections = [Int32](repeating: 0, count: prefixesPerLeaf)
  var bestCoefficients = [Int32](repeating: 0, count: prefixesPerLeaf)
  var bestResidualCount = Int.max
  var bestQueries = Int.max
  var bestCost = Int.max
  for first in candidates[0] {
    for second in candidates[1] {
      for third in candidates[2] {
        let corrections = [first, second, third]
        // A field is the cumulative prefix (first 4, first 8, first 12).
        // These differences make its per-segment contribution exactly q0/q1/q2.
        let coefficients = [first - second, second - third, third]
        let queryFields = coefficients.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
        var remaining = 0
        for (segment, range) in ranges.enumerated() {
          for ordinal in range {
            let pixel = Int(layout.permutation[ordinal])
            if validPixels[pixel] != 0 && residual[pixel] - corrections[segment] != 0 {
              remaining += 1
            }
          }
        }
        remaining += lastSegmentResiduals
        // Preserve the established 1-field : 4-residual proxy only for this
        // CPU screen. It is not a GPU cycle model.
        let cost = queryFields + 4 * remaining
        if cost < bestCost
          || (cost == bestCost && remaining < bestResidualCount)
          || (cost == bestCost && remaining == bestResidualCount && queryFields < bestQueries)
        {
          bestCorrections = corrections
          bestCoefficients = coefficients
          bestResidualCount = remaining
          bestQueries = queryFields
          bestCost = cost
        }
      }
    }
  }
  return PrefixChoice(
    segmentCorrections: bestCorrections, prefixCoefficients: bestCoefficients,
    residualCount: bestResidualCount, queryFieldCount: bestQueries,
    estimatedCost: bestCost)
}

private func evaluate(
  _ transition: Transition, masks: [String: [UInt8]],
  layout: PairedRuntimeTANSPolarPlan.IndexLayout
) -> TransitionEvaluation {
  guard let previous = masks[transition.previous], let target = masks[transition.target] else {
    fputs("FAIL: missing mask for \(transition.name)\n", stderr)
    exit(1)
  }
  let delta = previous.indices.map {
    validPixels[$0] == 0 ? Int32(0) : Int32(target[$0]) - Int32(previous[$0])
  }
  let base = basePlan(for: delta, layout: layout)
  let choices = (0..<layout.leaves).map {
    bestPrefixChoice(residual: base.residual, leaf: $0, layout: layout)
  }
  return TransitionEvaluation(definition: transition, delta: delta, base: base, leafChoices: choices)
}

private func leafScores(_ evaluations: [TransitionEvaluation]) -> [Int] {
  let layout = PairedRuntimeTANSPolarPlan.indexLayout(
    leafPixels: leafPixels, layoutKind: "radial1")!
  var score = [Int](repeating: 0, count: evaluations[0].leafChoices.count)
  for evaluation in evaluations {
    for leaf in score.indices {
      let first = leaf * leafPixels
      var oldResiduals = 0
      for ordinal in first..<(first + leafPixels) {
        let pixel = Int(layout.permutation[ordinal])
        oldResiduals += evaluation.base.residual[pixel] == 0 ? 0 : 1
      }
      let choice = evaluation.leafChoices[leaf]
      let oldLeafCost = 4 * oldResiduals
      score[leaf] += oldLeafCost - choice.estimatedCost
    }
  }
  return score
}

private func rankedLeaves(_ score: [Int]) -> [Int] {
  score.indices
    .filter { score[$0] > 0 }
    .sorted {
      if score[$0] != score[$1] { return score[$0] > score[$1] }
      return $0 < $1
    }
    .map { $0 }
}

private func applyOverlay(
  _ evaluation: TransitionEvaluation, selectedLeaves: [Int],
  layout: PairedRuntimeTANSPolarPlan.IndexLayout
) -> (residual: [Int32], prefix: [Int32], storedFields: Int, queryFields: Int) {
  var residual = evaluation.base.residual
  var prefix = [Int32](repeating: 0, count: detectorPixels)
  var queryFields = 0
  for leaf in selectedLeaves {
    let choice = evaluation.leafChoices[leaf]
    queryFields += choice.queryFieldCount
    let start = leaf * leafPixels
    for segment in 0..<prefixesPerLeaf {
      for ordinal in (start + segment * prefixWidth)..<(start + (segment + 1) * prefixWidth) {
        let pixel = Int(layout.permutation[ordinal])
        guard validPixels[pixel] != 0 else { continue }
        let contribution = choice.segmentCorrections[segment]
        prefix[pixel] = contribution
        residual[pixel] -= contribution
      }
    }
  }
  let storedFields = selectedLeaves.count * prefixesPerLeaf
  for pixel in 0..<detectorPixels {
    require(
      evaluation.base.fields[pixel] + prefix[pixel] + residual[pixel] == evaluation.delta[pixel],
      "overlay reconstruction failed for \(evaluation.definition.name) at pixel \(pixel)")
  }
  return (residual, prefix, storedFields, queryFields)
}

private struct RNG {
  var state: UInt64
  mutating func next() -> UInt64 {
    state ^= state << 13
    state ^= state >> 7
    state ^= state << 17
    return state
  }
}

private func verifySyntheticU16DotProducts(
  _ evaluation: TransitionEvaluation,
  selectedLeaves: [Int],
  layout: PairedRuntimeTANSPolarPlan.IndexLayout,
  source: Int
) -> Bool {
  let overlay = applyOverlay(evaluation, selectedLeaves: selectedLeaves, layout: layout)
  var rng = RNG(state: UInt64(0x5eed_0000 + source * 997 + evaluation.definition.name.utf8.count))
  var direct: Int64 = 0
  var reconstructed: Int64 = 0
  for pixel in 0..<detectorPixels {
    let count = Int64(UInt16(truncatingIfNeeded: rng.next()))
    direct += Int64(evaluation.delta[pixel]) * count
    reconstructed += Int64(evaluation.base.fields[pixel] + overlay.prefix[pixel] + overlay.residual[pixel])
      * count
  }
  return direct == reconstructed
}

private func sha256(_ data: Data) -> String {
  SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
}

private func outputRecord(
  _ evaluation: TransitionEvaluation,
  selectedLeaves: [Int],
  layout: PairedRuntimeTANSPolarPlan.IndexLayout,
  trainingLeafScore: [Int]
) -> [String: Any] {
  let overlay = applyOverlay(evaluation, selectedLeaves: selectedLeaves, layout: layout)
  let newResidualCount = overlay.residual.reduce(0) { $0 + ($1 == 0 ? 0 : 1) }
  let directCost = evaluation.base.fieldCount + 4 * evaluation.base.residualCount
  let overlayCost = evaluation.base.fieldCount + overlay.queryFields + 4 * newResidualCount
  let scalarParity = (0..<sourceCount).allSatisfy {
    verifySyntheticU16DotProducts(evaluation, selectedLeaves: selectedLeaves, layout: layout, source: $0)
  }
  require(scalarParity, "synthetic full-u16 scalar-product parity failed")
  return [
    "transition": evaluation.definition.name,
    "role": evaluation.definition.isHeldOut ? "held-out" : "training",
    "previous_mask": evaluation.definition.previous,
    "target_mask": evaluation.definition.target,
    "changed_pixels": evaluation.delta.reduce(0) { $0 + ($1 == 0 ? 0 : 1) },
    "baseline_fields_per_source": evaluation.base.fieldCount,
    "baseline_residual_pixels_per_source": evaluation.base.residualCount,
    "baseline_residual_pixels_all_seven_geometry_only": evaluation.base.residualCount * sourceCount,
    "baseline_proxy_cost_per_source": directCost,
    "overlay_residual_pixels_per_source": newResidualCount,
    "overlay_residual_pixels_all_seven_geometry_only": newResidualCount * sourceCount,
    "residual_reduction_per_source": evaluation.base.residualCount - newResidualCount,
    "residual_reduction_fraction": evaluation.base.residualCount == 0
      ? 0.0 : Double(evaluation.base.residualCount - newResidualCount) / Double(evaluation.base.residualCount),
    "extra_prefix_fields_stored_per_source": overlay.storedFields,
    "extra_prefix_fields_stored_all_seven": overlay.storedFields * sourceCount,
    "extra_prefix_queries_used_per_source": overlay.queryFields,
    "extra_prefix_queries_used_all_seven": overlay.queryFields * sourceCount,
    "overlay_proxy_cost_per_source": overlayCost,
    "proxy_cost_reduction_per_source": directCost - overlayCost,
    "full_map_reconstruction_exact": true,
    "synthetic_uint16_scalar_parity_all_seven": scalarParity,
    "training_score_min_leaf": selectedLeaves.map { trainingLeafScore[$0] }.min() ?? 0,
  ]
}

private func main() throws {
  guard CommandLine.arguments.count == 3 else {
    fputs("usage: ans-prefix-overlay <frozen-20-mask-jsonl> <output-json>\n", stderr)
    exit(2)
  }
  let frozenURL = URL(fileURLWithPath: CommandLine.arguments[1])
  let outputURL = URL(fileURLWithPath: CommandLine.arguments[2])
  let (frozenNames, frozenRows) = try readFrozenMaskNames(frozenURL)
  let definitions = maskDefinitions()
  let definedNames = Set(definitions.map(\.name))
  require(definitions.count == 20 && definedNames.count == 20, "mask definition count is not 20")
  require(definedNames == frozenNames, "mask geometry names differ from frozen 20-mask reference")

  let definitionsByName = Dictionary(uniqueKeysWithValues: definitions.map { ($0.name, $0) })
  let masks = Dictionary(uniqueKeysWithValues: definitions.map { ($0.name, mask($0)) })
  let transitionList = transitions()
  require(transitionList.count == 17, "expected 17 transitions from the frozen 20-mask geometry")
  let targetCount = transitionList.filter(\.isHeldOut).count
  require(targetCount == 1, "target transition must appear exactly once")
  let target = transitionList.first(where: \.isHeldOut)!
  require(
    target.previous == "adf-center-8" && target.target == "adf-center-20",
    "held-out target must be the large ADF center-(5,8)→(12,20) move")
  for transition in transitionList {
    require(definitionsByName[transition.previous] != nil, "unknown previous mask")
    require(definitionsByName[transition.target] != nil, "unknown target mask")
  }

  guard let layout = PairedRuntimeTANSPolarPlan.indexLayout(
    leafPixels: leafPixels, layoutKind: "radial1") else {
    fputs("FAIL: missing current radial1/leaf16 baseline layout\n", stderr)
    exit(1)
  }
  let evaluations = transitionList.map { evaluate($0, masks: masks, layout: layout) }
  let training = evaluations.filter { !$0.definition.isHeldOut }
  let heldOut = evaluations.filter(\.definition.isHeldOut)
  require(training.count == 16 && heldOut.count == 1, "incorrect split between train and holdout")

  let broadTrainingScores = leafScores(training)
  let broadTrainingRanking = rankedLeaves(broadTrainingScores)
  let adfTraining = training.filter { $0.definition.name.hasPrefix("adf-") }
  require(adfTraining.count == 5, "expected five non-held-out ADF training transitions")
  let adfTrainingScores = leafScores(adfTraining)
  let adfTrainingRanking = rankedLeaves(adfTrainingScores)
  let targetOracleScores = leafScores(heldOut)
  let targetOracleRanking = rankedLeaves(targetOracleScores)
  let budgets = [0, 5, 10, 15, 20, trainingLeafBudget]
  func budgetFamily(_ policy: String, _ ranking: [Int], _ scores: [Int]) -> [[String: Any]] {
    budgets.map { budget in
      let selectedLeaves = Array(ranking.prefix(budget))
      let selected = evaluations.map {
        outputRecord($0, selectedLeaves: selectedLeaves, layout: layout, trainingLeafScore: scores)
      }
      let targetRecord = selected.first { $0["role"] as? String == "held-out" }!
      return [
        "selection_policy": policy,
        "selected_leaf_budget": budget,
        "selected_leaf_count": selectedLeaves.count,
        "selected_leaves_radial1_leaf16": selectedLeaves,
        "training_score_by_leaf": selectedLeaves.map { scores[$0] },
        "max_prefix_fields_stored_per_source": selectedLeaves.count * prefixesPerLeaf,
        "max_prefix_fields_stored_all_seven_sources": selectedLeaves.count * prefixesPerLeaf * sourceCount,
        "held_out_target": targetRecord,
        "all_transitions": selected,
        "exact_all_transition_maps": selected.allSatisfy { $0["full_map_reconstruction_exact"] as? Bool == true },
        "exact_all_seven_source_scalar_products": selected.allSatisfy {
          $0["synthetic_uint16_scalar_parity_all_seven"] as? Bool == true
        },
      ]
    }
  }
  let budgetResults = budgetFamily(
    "aggregate BF+ABF+ADF training transitions; ADF center-8-to-20 held out",
    broadTrainingRanking, broadTrainingScores)
    + budgetFamily(
      "ADF-only training transitions; ADF center-8-to-20 held out",
      adfTrainingRanking, adfTrainingScores)
  let oracleLeaves = Array(targetOracleRanking.prefix(trainingLeafBudget))
  let targetOracle = outputRecord(
    heldOut[0], selectedLeaves: oracleLeaves, layout: layout,
    trainingLeafScore: targetOracleScores)

  let frozenData = try Data(contentsOf: frozenURL)
  let result: [String: Any] = [
    "schema": "quantem-gpu-ans-selective-prefix-overlay/v1",
    "scope": "CPU-only exact planner prototype; geometry replay and seven-source multiplicity; no GPU kernel, resident allocation, or timing",
    "dataset_context": [
      "dataset_id": "tilt-series-seven-native-v1",
      "logical_dtype": "uint16",
      "full_shape": [512, 512, 192, 192],
      "source_count": sourceCount,
      "source_reference_masks": frozenNames.count,
      "frozen_reference_rows": frozenRows,
      "frozen_reference_sha256": sha256(frozenData),
      "validity_note": "Applied the same four HDF5 bad-pixel indices found in all seven source masters by the independent catalog-only validity census.",
      "common_bad_pixel_indices": badPixelIndices,
      "bad_pixel_mask_source": "experiments/20260913-apple-m5-ans-radial1-leafwidth-cpu-screen/results/screen.json",
      "bad_pixel_mask_sha256": "33f5b1988e3f4360e9578a8855884e5bdcb2cbc2b6b25c0c830d65bec6c69b47",
    ],
    "method": [
      "baseline": "PairedRuntimeTANSPolarPlan.make, leafPixels=16, layout=radial1, jointPlan=off",
      "prefix_basis": "Per selected 16-pixel leaf, cumulative exact sums over first 4, first 8, and first 12 radial1-ordered detector pixels.",
      "planner": "Choose three independent 4-pixel correction values q0/q1/q2; emit prefix coefficients [q0-q1, q1-q2, q2], then keep the remaining nonzero signed coefficients as direct residuals.",
      "leaf_selection": "Compare broad 16-transition training against five ADF-only training transitions. The held-out ADF center-8→center-20 move is excluded from both rankings.",
      "proxy_warning": "The existing proxy is not a timing model. It does not establish Metal throughput or total index-build/retained bytes.",
    ],
    "geometry": [
      "BF, ABF, and ADF mask parameters copied from MetalPairedRuntimeTANSSeriesBenchmark/experimentMasks and circularMask.",
      "20 exact named masks were required to match the frozen seven-source reference names.",
      "17 mask transitions replayed; the 16 training transitions cover BF/ABF/ADF centers and radii, while the ADF center-(5,8)→(12,20) move is held out.",
      "The same four HDF5 catalog bad-pixel indices were applied to all seven sources before planner decomposition.",
    ],
    "leaf_width": leafPixels,
    "prefix_width": prefixWidth,
    "prefixes_per_selected_leaf": prefixesPerLeaf,
    "budgets": budgetResults,
    "held_out_target_oracle_diagnostic": [
      "This selection deliberately leaks the held-out target and is only an upper-bound diagnostic; it is not a training result.",
      "If target-only selection produces little benefit, the fixed four-pixel prefix basis itself has limited potential, not just a poor training ranking.",
    ],
    "target_oracle_budget": [
      "selection_policy": "target-only oracle diagnostic; leaks holdout",
      "selected_leaf_budget": trainingLeafBudget,
      "selected_leaf_count": oracleLeaves.count,
      "selected_leaves_radial1_leaf16": oracleLeaves,
      "max_prefix_fields_stored_per_source": oracleLeaves.count * prefixesPerLeaf,
      "max_prefix_fields_stored_all_seven_sources": oracleLeaves.count * prefixesPerLeaf * sourceCount,
      "held_out_target": targetOracle,
    ],
  ]
  try FileManager.default.createDirectory(at: outputURL.deletingLastPathComponent(), withIntermediateDirectories: true)
  let data = try JSONSerialization.data(withJSONObject: result, options: [.prettyPrinted, .sortedKeys])
  try data.write(to: outputURL, options: .atomic)
  print("PASS: frozen masks=\(frozenNames.count), sources=\(sourceCount), transitions=\(evaluations.count), training=\(training.count), holdout=\(target.name)")
  for budget in budgetResults where budget["selected_leaf_budget"] as? Int == trainingLeafBudget {
    let target = budget["held_out_target"] as! [String: Any]
    print(
      "POLICY \(budget["selection_policy"]!) BUDGET leaves=\(budget["selected_leaf_count"]!) stored_prefix_fields/source=\(budget["max_prefix_fields_stored_per_source"]!) "
        + "target_residual/source=\(target["overlay_residual_pixels_per_source"]!) "
        + "baseline=\(target["baseline_residual_pixels_per_source"]!) "
        + "queries/source=\(target["extra_prefix_queries_used_per_source"]!) "
        + "proxy=\(target["baseline_proxy_cost_per_source"]!)->\(target["overlay_proxy_cost_per_source"]!)")
  }
  print(
    "ORACLE target_residual/source=\(targetOracle["overlay_residual_pixels_per_source"]!) "
      + "baseline=\(targetOracle["baseline_residual_pixels_per_source"]!) "
      + "queries/source=\(targetOracle["extra_prefix_queries_used_per_source"]!) "
      + "proxy=\(targetOracle["baseline_proxy_cost_per_source"]!)->\(targetOracle["overlay_proxy_cost_per_source"]!)")
  print("wrote \(outputURL.path)")
}

do {
  try main()
} catch {
  fputs("FAIL: \(error)\n", stderr)
  exit(1)
}
